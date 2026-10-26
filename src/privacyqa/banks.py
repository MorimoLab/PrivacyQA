"""Request banks built from the catalogs alone, the common test, and the reconstruction of released views.

Each bank holds up to `max_cores` cores of one to three evidence atoms, with an equal quota per core length; a length
with too few cores passes its unused quota on. The main bank ranks cores by catalog match count. Its variants are the
core alone and the core plus one catalog attribute, nearest to `target_answers` complete answers first; the first
`max_options` are kept when their answers number `answer_lower` to `answer_upper`, and Recipe narrows larger answer sets
with the RecipeADQA ingredient search. The random bank selects, per core length, as many cores as the main bank has
units, in a fixed pseudorandom order, and samples additional attributes (Movie, Beauty) or ingredients (Recipe) from a
random answer of the core. The variants of a core form one unit, whose support is measured on the core. No user data
enter this module.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
import random

from privacyqa.catalog import domain_schema, extend_recipe, ingredient_atom, load_catalog, load_recipe_search
from privacyqa.common import DOMAINS, RATING_DOMAINS, ARMS, atoms, progress, public_rank, rank, read_json, require, write_json

BANK = 'unified-template-bank-v1'  # Namespace of the variant identities.


def variant_id(domain, core, additions):
    return f'unified:{domain}:' + public_rank(BANK, [domain, list(core), list(additions)])


@dataclass(frozen=True)
class Variant:
    id: str
    evidence_constraints: tuple
    catalog_constraints: tuple
    answers: tuple

    @property
    def constraints(self):
        return tuple(sorted(self.evidence_constraints + self.catalog_constraints))


# ---------------------------------------------------------------- cores

def core_counts(catalog, evidence, lengths):
    """Catalog items matching each co-occurring combination of evidence atoms."""
    rows = Counter(tuple(sorted(attributes & evidence)) for attributes, _ in catalog.items.values())
    counts = Counter()
    for tags, multiplicity in rows.items():
        for length in lengths:
            for core in itertools.combinations(tags, length):
                counts[core] += multiplicity
    return counts


def length_quotas(available, limit):
    """Equal quota per core length; a length with too few cores passes its unused quota on."""
    pending, quotas, remaining = sorted(available), {}, limit
    while pending:
        share = remaining // len(pending)
        small = [n for n in pending if available[n] <= share]
        if not small:
            for index, n in enumerate(pending):
                quotas[n] = remaining // len(pending) + (index < remaining % len(pending))
            break
        for n in small:
            quotas[n] = available[n]
            remaining -= available[n]
            pending.remove(n)
    return quotas


def select_main_cores(counts, lower, rules):
    options = {n: [c for c, k in counts.items() if len(c) == n and k >= lower] for n in rules['core_lengths']}
    quotas = length_quotas({n: len(v) for n, v in options.items()}, rules['max_cores'])
    seed = rules['public_seed']
    chosen = [c for n in rules['core_lengths']
              for c in sorted(options[n], key=lambda c: (-counts[c], public_rank(seed, list(c)), c))[:quotas[n]]]
    return chosen, {str(n): len(v) for n, v in options.items()}, {str(n): q for n, q in quotas.items()}


def select_random_cores(counts, lower, rules, quotas):
    seed = rules['random']['public_seed']
    options = {n: sorted((c for c, k in counts.items() if len(c) == n and k >= lower), key=lambda c: (public_rank(seed, list(c)), c))
               for n in rules['core_lengths']}
    chosen = [c for n in rules['core_lengths'] for c in options[n][:quotas.get(str(n), 0)]]
    return chosen, {str(n): len(v) for n, v in options.items()}, dict(quotas)


# ---------------------------------------------------------------- main-bank variants

def rating_options(catalog, core, answers, rules):
    """The core alone and the core plus one co-occurring catalog attribute, nearest to the target first."""
    upper, sizes, members = rules['answer_upper'], Counter(), {}
    for item in answers:
        for tag in catalog.items[item][0]:
            if tag in core:
                continue
            sizes[tag] += 1
            if sizes[tag] <= upper:
                members.setdefault(tag, []).append(item)
    options = [((), len(answers))] + [((tag,), n) for tag, n in sizes.items()]
    options.sort(key=lambda x: (abs(x[1] - rules['target_answers']), len(x[0]), x[0]))
    return options, members


def rating_variants(catalog, domain, core, rules):
    answers = frozenset(catalog.execute(core))
    ranked, members = rating_options(catalog, core, answers, rules)
    lower, upper = rules['answer_lower'][domain], rules['answer_upper']
    return [(extra, tuple(sorted(answers if not extra else members[extra[0]])))
            for extra, n in ranked[:rules['max_options']] if lower <= n <= upper]


def recipe_options(catalog, core, rules):
    options = [()] + [(tag,) for tag in sorted(catalog.schema - set(core))]
    sizes = {extra: len(catalog.execute(core, extra)) for extra in options}
    return sorted((extra for extra in options if sizes[extra] > 0),
                  key=lambda extra: (abs(sizes[extra] - rules['target_answers']), len(extra), extra)), sizes


def recipe_variants(catalog, core, rules, search):
    ranked, _ = recipe_options(catalog, core, rules)
    found = []
    for extra in ranked[:rules['max_options']]:
        _, additions, answers = extend_recipe(catalog, core, extra, search, rules['max_conditions'])
        if rules['answer_lower']['recipe_text'] <= len(answers) <= rules['answer_upper']:
            found.append((additions, tuple(answers)))
    return found


# ---------------------------------------------------------------- random-bank variants

def random_rating_variants(catalog, domain, core, rules):
    """A random subset of the other attributes of one random answer of the core; answers are never truncated."""
    settings, lower, upper = rules['random'], rules['answer_lower'][domain], rules['answer_upper']
    anchors = catalog.execute(core)
    rng = random.Random(json.dumps([settings['public_seed'], domain, list(core)]))
    variants = {}
    if lower <= len(anchors) <= upper:
        variants[()] = anchors
    for _ in range(settings['attempts_per_core']):
        if not anchors or len(variants) == settings['max_extensions']:
            break
        pool = sorted(catalog.items[rng.choice(anchors)][0] - set(core))
        if not pool:
            continue
        extra = tuple(sorted(rng.sample(pool, rng.randint(1, min(rules['max_conditions'] - len(core), len(pool))))))
        answers = catalog.execute(core, extra)
        if lower <= len(answers) <= upper:
            variants[extra] = answers
    return sorted(variants.items())


def recipe_ingredient_draw(ingredients, vocabulary, rng):
    """Liked and disliked ingredients, adapted from RecipeADQA kgqa_style.sample_one (Apache-2.0) without nutrients.

    Dislikes come from the ingredients of the core's answers that the drawn recipe lacks.
    """
    include, exclude = sorted(ingredients), sorted(vocabulary - ingredients)
    if not include or not exclude:
        return None
    likes = sorted(rng.sample(include, rng.randint(1, min(5, len(include)))))
    dislikes = sorted(rng.sample(exclude, rng.randint(1, min(3, len(exclude)))))
    return likes, dislikes


def random_recipe_variants(catalog, core, rules):
    settings, lower, upper = rules['random'], rules['answer_lower']['recipe_text'], rules['answer_upper']
    anchors = catalog.execute(core)
    rng = random.Random(json.dumps([settings['public_seed'], catalog.domain, list(core)]))
    vocabulary = frozenset(name for item in anchors for name in catalog.items[item][1])
    variants = {}
    for _ in range(settings['attempts_per_core']):
        if not anchors or len(variants) == settings['max_extensions']:
            break
        _, ingredients = catalog.items[rng.choice(anchors)]
        drawn = recipe_ingredient_draw(ingredients, vocabulary, rng)
        if drawn is None:
            continue
        likes, dislikes = drawn
        if len(likes) + len(dislikes) > rules['max_conditions'] - len(core):
            continue
        extra = tuple(sorted([ingredient_atom('include', v) for v in likes] + [ingredient_atom('exclude', v) for v in dislikes]))
        answers = catalog.execute(core, extra)
        if lower <= len(answers) <= upper:
            variants[extra] = answers
    return sorted(variants.items())


# ---------------------------------------------------------------- banks

def build_bank(domain, arm, catalog, schema, rules, search, quotas=None):
    """One bank; `quotas` (executable main-bank cores per length) fixes the size of the random bank."""
    evidence = frozenset(schema['evidence_atoms'])
    lower = rules['rating_min_core_matches'] if domain in RATING_DOMAINS else 1
    counts = core_counts(catalog, evidence, rules['core_lengths'])
    if arm == 'main':
        cores, available, planned = select_main_cores(counts, lower, rules)
    else:
        cores, available, planned = select_random_cores(counts, lower, rules, quotas)
    candidates, records = [], []
    for core in cores:
        if domain in RATING_DOMAINS:
            variants = (rating_variants if arm == 'main' else random_rating_variants)(catalog, domain, core, rules)
        else:
            variants = recipe_variants(catalog, core, rules, search) if arm == 'main' else random_recipe_variants(catalog, core, rules)
        entries = [dict(id=variant_id(domain, core, extra), evidence_constraints=list(core),
                        catalog_constraints=list(extra), answers=list(answers)) for extra, answers in variants]
        require(len({e['id'] for e in entries}) == len(entries), 'duplicate_variant', 'Variants of a core are distinct')
        candidates.extend(entries)
        if entries:
            records.append(dict(core=list(core), core_answer_count=counts[core], candidate_ids=[e['id'] for e in entries]))
    coverage = dict(eligible_cores_by_length=available, planned_cores_by_length=planned, selected_cores=len(cores),
                    selected_lengths=dict(Counter(str(len(c)) for c in cores)), executable_units=len(records),
                    executable_lengths=dict(Counter(str(len(r['core'])) for r in records)), variants=len(candidates),
                    variant_conditions=dict(sorted(Counter(str(len(c['evidence_constraints']) + len(c['catalog_constraints']))
                                                           for c in candidates).items())),
                    failed_cores=len(cores) - len(records))
    return dict(bank_version=BANK, domain=domain, arm=arm, schema_version=schema['schema_version'],
                kb_version=schema['kb_version'], rules=rules, cores=records, candidates=candidates, coverage=coverage)


class Bank:
    """A frozen bank with its catalog: units, core answers and the rows of released views."""

    def __init__(self, data, catalog, schema, rules):
        require(data['bank_version'] == BANK and data['domain'] == catalog.domain and data['rules'] == rules
                and (data['schema_version'], data['kb_version']) == (schema['schema_version'], schema['kb_version']),
                'invalid_bank', 'Use a bank built from this catalog, schema and rules')
        self.data, self.catalog, self.rules = data, catalog, rules
        self.domain, self.arm = data['domain'], data['arm']
        self.schema_version, self.kb_version = data['schema_version'], data['kb_version']
        self.candidates = tuple(Variant(c['id'], atoms(c['evidence_constraints']), atoms(c['catalog_constraints']),
                                        tuple(c['answers'])) for c in data['candidates'])
        require(all([list(v.evidence_constraints), list(v.catalog_constraints)] == [c['evidence_constraints'], c['catalog_constraints']]
                    and not set(v.evidence_constraints) & set(v.catalog_constraints)
                    and v.id == variant_id(self.domain, v.evidence_constraints, v.catalog_constraints)
                    and list(v.answers) == sorted(set(v.answers)) for v, c in zip(self.candidates, data['candidates'])),
                'invalid_variant', 'Variants have canonical constraints, their identity and sorted answers')
        self.by_id = {c.id: c for c in self.candidates}
        self.units = tuple(tuple(r['candidate_ids']) for r in data['cores'])
        self.cores = tuple(tuple(r['core']) for r in data['cores'])
        flat = [k for unit in self.units for k in unit]
        require(len(flat) == len(set(flat)) == len(self.by_id) == len(self.candidates) and set(flat) == set(self.by_id)
                and len(set(self.cores)) == len(self.cores) <= rules['max_cores'],
                'invalid_bank_partition', 'Every variant belongs to exactly one core, and every core occurs once')
        evidence, lower, upper = frozenset(schema['evidence_atoms']), rules['answer_lower'][self.domain], rules['answer_upper']
        self.core_answers = []
        for record, core, unit in zip(data['cores'], self.cores, self.units):
            answers = catalog.execute(core)
            require(set(core) <= evidence and 1 <= len(core) <= 3 and len(answers) == record['core_answer_count']
                    and all(self.by_id[k].evidence_constraints == core for k in unit),
                    'invalid_bank_core', 'A unit shares one evidence core, and counts are complete executions')
            self.core_answers.append(frozenset(answers))
        for c in self.candidates:
            require(c.answers == catalog.execute(c.evidence_constraints, c.catalog_constraints, max_conditions=rules['max_conditions'])
                    and lower <= len(c.answers) <= upper, 'incomplete_answers',
                    'Answers are complete executions of every constraint, within the declared range')
        self.rows = {c.id: dict(id=c.id, domain=self.domain, constraints=list(c.constraints),
                                evidence_constraints=list(c.evidence_constraints), catalog_constraints=list(c.catalog_constraints),
                                answers=list(c.answers), answer_count=len(c.answers), schema_version=self.schema_version,
                                kb_version=self.kb_version) for c in self.candidates}

    def reconstruct(self, mass, release_id, threshold=0):
        """Views of one released vector: units with positive mass above the threshold, weights uniform within a unit."""
        require(len(mass) == len(self.units) and all(type(v) is int for v in mass) and type(threshold) is int and threshold >= 0,
                'invalid_release_vector', 'Reconstruct from the complete integer vector of this bank')
        masses = [Fraction(max(value - threshold, 0)) for value in mass]
        total = sum(masses, Fraction(0))
        plain, weighted = [], []
        for unit, value in zip(self.units, masses):
            if value <= 0:
                continue
            weight = float(value / total / len(unit))
            for cid in unit:
                row = dict(self.rows[cid], release_id=release_id)
                plain.append(row)
                weighted.append(dict(row, weight=weight))
        return {'C-only': plain, 'C+weight': weighted}


def common_test(bank, seed, requests):
    """Main-bank requests drawn round-robin over condition counts; alternate answer classes form the two tracks."""
    strata = defaultdict(list)
    for c in bank.candidates:
        strata[len(c.constraints)].append(c)
    for values in strata.values():
        values.sort(key=lambda c: rank(seed, bank.domain, 'utility', c.id))
    selected, seen = [], set()
    while any(strata.values()) and len(selected) < requests:
        for length in sorted(strata):
            if not strata[length] or len(selected) == requests:
                continue
            candidate = strata[length].pop(0)
            signature = candidate.evidence_constraints, candidate.catalog_constraints
            if signature not in seen:
                selected.append(candidate)
                seen.add(signature)
    classes = sorted({c.answers for c in selected}, key=lambda a: rank(seed, bank.domain, 'answer-class', a))
    answer_holdout = set(classes[::2])
    public = {r['id']: r for r in bank.reconstruct([1] * len(bank.units), 'common-test')['C-only']}
    return [dict(public[c.id], evaluation_track='answer_class_holdout' if c.answers in answer_holdout
                 else 'constraint_signature_holdout') for c in selected]


def build_banks(config, output):
    rules = config['banks']
    search = load_recipe_search(config.file(config['recipe_search_module']))
    tests, summary = {}, {}
    for domain in DOMAINS:
        schema, catalog = domain_schema(config, domain), load_catalog(config, domain)
        progress(output, 'bank_start', domain=domain, arm='main')
        main = build_bank(domain, 'main', catalog, schema, rules, search)
        write_json(output / f'{domain}-main.json', main)
        progress(output, 'bank_start', domain=domain, arm='random')
        randomized = build_bank(domain, 'random', catalog, schema, rules, search, main['coverage']['executable_lengths'])
        write_json(output / f'{domain}-random.json', randomized)
        bank = Bank(main, catalog, schema, rules)
        Bank(randomized, catalog, schema, rules)
        tests[domain] = common_test(bank, config['test']['seed'], config['test']['requests'])
        summary[domain] = dict(main=main['coverage'], random=randomized['coverage'], test_requests=len(tests[domain]),
                               test_answer_classes=len({tuple(r['answers']) for r in tests[domain]}),
                               test_tracks=dict(Counter(r['evaluation_track'] for r in tests[domain])))
        progress(output, 'banks_complete', domain=domain, units=len(bank.units))
    write_json(output / 'common_test.json', tests)
    manifest = dict(status='banks_built', rules=rules, test=config['test'], domains=summary, user_data_read=False)
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_banks(config, arms=ARMS):
    """The frozen banks of both arms, sharing one catalog per domain."""
    manifest = config.finished('banks', 'banks_built')
    require(manifest['rules'] == config['banks'], 'bank_rules_mismatch', 'The banks were built with other rules')
    result = {arm: {} for arm in arms}
    for domain in DOMAINS:
        schema, catalog = domain_schema(config, domain), load_catalog(config, domain)
        for arm in arms:
            result[arm][domain] = Bank(read_json(config.stage('banks') / f'{domain}-{arm}.json'), catalog, schema, config['banks'])
    return result
