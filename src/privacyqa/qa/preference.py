"""The preference-weighted test: questions drawn in proportion to the reference users' support, outside every bank.

Each domain draws main-bank units with probability proportional to their share of reference supporters. A draw
receives a request of its unit that lies in neither bank, the common test nor the development set: the construction
rule of the bank is replayed for the unit's core and continued past the bank's own options. Questions use the
candidates and template of the common test and are weighted by their number of draws. No history is read here.
"""
from collections import Counter
import math
import random

import numpy as np

from privacyqa.banks import load_banks, rating_options, recipe_options, variant_id
from privacyqa.catalog import extend_recipe, load_recipe_search
from privacyqa.common import DOMAINS, iter_rows, progress, read_json, require, write_json, write_rows
from privacyqa.qa.analysis import scored_outputs
from privacyqa.qa.records import CandidateBuilder
from privacyqa.release import METHODS

COMPARISONS = (('Non-private', 'Public-only'), ('Bounded-non-private', 'Public-only'), ('DP', 'Public-only'), ('DP', 'Non-private'))
TRACKS = ('answer_seen', 'answer_novel')
METRICS = ('f1', 'set_exact', 'format_valid')


def unit_preferences(weights, units):
    """Each unit's share of the reference supporters."""
    index = {cid: j for j, unit in enumerate(units) for cid in unit}
    require(set(weights) <= set(index), 'preference_unit_mismatch', 'Reference weights belong to bank units')
    mass = [0.0] * len(units)
    for cid, weight in weights.items():
        mass[index[cid]] += weight
    total = sum(mass)
    require(total > 0 and all(math.isfinite(p) and p >= 0 for p in mass), 'invalid_preference', 'Use a positive reference distribution')
    return [p / total for p in mass]


def draw_units(preference, count, seed):
    return random.Random(seed).choices(range(len(preference)), weights=preference, k=count)


def continued_variants(domain, catalog, core, excluded, needed, search, rules):
    """Replay the bank's first options of a core, then continue its ranking with requests outside `excluded`."""
    lower, upper, skip = rules['answer_lower'][domain], rules['answer_upper'], rules['max_options']
    if domain == 'recipe_text':
        ranked, _ = recipe_options(catalog, core, rules)

        def realize(extra):
            _, additions, answers = extend_recipe(catalog, core, extra, search, rules['max_conditions'])
            return additions, tuple(answers)
    else:
        complete = frozenset(catalog.execute(core))
        options, members = rating_options(catalog, core, complete, rules)
        ranked, sizes = [extra for extra, _ in options], dict(options)

        def realize(extra):
            if sizes[extra] > upper:
                return extra, None
            return extra, tuple(sorted(complete if not extra else members[extra[0]]))
    bank = []
    for extra in ranked[:skip]:
        additions, answers = realize(extra)
        if answers is not None and lower <= len(answers) <= upper:
            bank.append((core, additions, answers))
    result, seen, tried = [], set(), 0
    for extra in ranked[skip:]:
        if len(result) == needed:
            break
        tried += 1
        additions, answers = realize(extra)
        if answers is None or not lower <= len(answers) <= upper or tuple(sorted(core + additions)) in excluded or answers in seen:
            continue
        seen.add(answers)
        result.append((core, additions, answers))
    return result, bank, dict(options=len(ranked), searched=tried)


def unit_variants(bank, unit, needed, excluded, search):
    """New requests of a unit, after checking that the replayed rule yields exactly the bank's variants."""
    members = [bank.by_id[cid] for cid in bank.units[unit]]
    variants, replayed, facts = continued_variants(bank.domain, bank.catalog, members[0].evidence_constraints, excluded, needed, search, bank.rules)
    require({(c.evidence_constraints, c.catalog_constraints, c.answers) for c in members} == set(replayed), 'bank_rule_mismatch',
            'The replayed construction rule reproduces the bank unit')
    return variants, dict(facts, bank_variants=len(members))


def request_row(bank, evidence, additions, answers, track):
    evidence, additions = list(evidence), list(additions)
    return dict(id=variant_id(bank.domain, tuple(evidence), tuple(additions)), domain=bank.domain, evidence_constraints=evidence,
                catalog_constraints=additions, constraints=sorted(evidence + additions), answers=list(answers), answer_count=len(answers),
                schema_version=bank.schema_version, kb_version=bank.kb_version, release_id='preference-test', evaluation_track=track)


def domain_context(config, domain, bank, controls):
    weights = read_json(config.stage('qa-fidelity') / f'{domain}-distributions.json')['main'][0]
    common = config.stage('qa-data') / domain / 'common'
    original = list(iter_rows(common / 'development.gold.jsonl', 'jsonl')) + list(iter_rows(common / 'test.gold.jsonl', 'jsonl'))
    excluded = {tuple(sorted(r['evidence_constraints'] + r['catalog_constraints'])) for r in original} | {tuple(r['constraints']) for r in controls}
    unit_of = {cid: j for j, unit in enumerate(bank.units) for cid in unit}
    answers, trained = set(), {}
    for method in METHODS:
        rows = list(iter_rows(config.stage('qa-data') / domain / method / 'constraints.jsonl', 'jsonl'))
        answers |= {tuple(r['answers']) for r in rows}
        trained[method] = {unit_of[r['id']] for r in rows}
    return weights, dict(excluded=excluded, training_answers=answers, training_units=trained, original_keys={r['qa_key'] for r in original})


def build_domain(bank, weights, context, plan, rules, search, output):
    preference = unit_preferences(weights, bank.units)
    draws = draw_units(preference, rules['draws'], rules['draw_seed'] + DOMAINS.index(bank.domain))
    counts = Counter(draws)
    variants, facts = {}, {}
    for unit in sorted(counts):
        variants[unit], facts[unit] = unit_variants(bank, unit, min(counts[unit], rules['variant_limit']), context['excluded'], search)
    builder = CandidateBuilder(bank.catalog, plan['seed'], plan['candidates'], plan['negative_reservoir'], plan['posting_sample'])
    occurrence, records, packets = Counter(), [], {}
    for position, unit in enumerate(draws):
        k, options = occurrence[unit], variants[unit]
        occurrence[unit] += 1
        record = dict(draw=position, unit=unit, occurrence=k, preference=preference[unit],
                      trained={m: unit in context['training_units'][m] for m in METHODS})
        if not options:
            records.append(dict(record, variant=None, qa_key=None))
            continue
        evidence, additions, answers = options[k % len(options)]
        track = 'answer_seen' if answers in context['training_answers'] else 'answer_novel'
        row = request_row(bank, evidence, additions, answers, track)
        require(tuple(row['constraints']) not in context['excluded'] and answers == bank.catalog.execute(row['evidence_constraints'], row['catalog_constraints']),
                'preference_request_overlap', 'New requests lie outside both banks, the test and the development set')
        model, gold = builder.build(row, 'test')
        require(model['qa_key'] not in context['original_keys'], 'preference_key_overlap', 'No test or development question is reused')
        if model['qa_key'] not in packets:
            packets[model['qa_key']] = (model, dict(gold, unit=unit, preference_draws=0))
        packets[model['qa_key']][1]['preference_draws'] += 1
        records.append(dict(record, variant=k % len(options), qa_key=model['qa_key'], track=track, condition_count=len(row['constraints']),
                            answer_count=len(answers)))
    folder = output / bank.domain
    folder.mkdir(mode=0o700)
    write_rows(folder / 'test.inputs.jsonl', (p[0] for p in packets.values()))
    write_rows(folder / 'test.gold.jsonl', (p[1] for p in packets.values()))
    write_json(folder / 'draws.json', dict(draws=records, unit_facts={str(u): f for u, f in facts.items()}))
    done = [r for r in records if r['qa_key'] is not None]
    failed = sorted({r['unit'] for r in records if r['qa_key'] is None})
    return dict(draws=len(records), successful_draws=len(done), failed_draws=len(records) - len(done), failed_units=len(failed),
                failed_unit_preference_mass=sum(preference[u] for u in failed), positive_units=sum(p > 0 for p in preference),
                drawn_units=len(counts), max_draws_per_unit=max(counts.values()), unique_questions=len(packets),
                cycled_units=sum(0 < len(variants[u]) < counts[u] for u in counts), draw_tracks=dict(Counter(r['track'] for r in done)),
                question_tracks=dict(Counter(p[1]['evaluation_track'] for p in packets.values())),
                draw_condition_counts=dict(sorted(Counter(r['condition_count'] for r in done).items())),
                draw_answer_counts=dict(sorted(Counter(min(r['answer_count'], 26) for r in done).items())),
                unit_in_training_draw_share={m: sum(r['trained'][m] for r in done) / len(done) if done else None for m in METHODS},
                candidate_count=builder.count)


def prepare_preference(config, output):
    rules = config['preference']
    banks = load_banks(config)
    config.finished('qa-data', 'qa_data_prepared')
    config.finished('qa-fidelity', 'qa_fidelity_completed')
    controls = read_json(config.stage('banks') / 'common_test.json')
    search = load_recipe_search(config.file(config['recipe_search_module']))
    domains = {}
    for domain in DOMAINS:
        bank = banks['main'][domain]
        weights, context = domain_context(config, domain, bank, controls[domain])
        context['excluded'] |= {c.constraints for arm in banks for c in banks[arm][domain].candidates}
        domains[domain] = build_domain(bank, weights, context, config['qa'], rules, search, output)
        progress(output, 'preference_domain_prepared', domain=domain, questions=domains[domain]['unique_questions'])
    manifest = dict(status='preference_data_prepared', rules=rules, methods=list(METHODS), domains=domains,
                    preference='share_of_reference_supporters_per_unit')
    write_json(output / 'manifest.json', manifest)
    return manifest


def bootstrap(values, repetitions, seed):
    """Paired resampling of whole draws: every method keeps the same resampled draws."""
    rng = np.random.default_rng(seed)
    n = len(values)
    return np.stack([values[rng.integers(0, n, size=n)].mean(axis=0) for _ in range(repetitions)])


def summarize(values, draws, tracks, trained, rules):
    """Draw-weighted means with paired intervals, by track and by whether the drawn unit is in the training source."""
    point = values.mean(axis=0)
    boot = bootstrap(values, rules['bootstrap_replicates'], rules['bootstrap_seed'])
    methods = {}
    for i, method in enumerate(METHODS):
        by_track = {}
        for track in TRACKS:
            mask = tracks == track
            by_track[track] = dict(draws=int(mask.sum()), **{m: float(values[mask, i, j].mean()) if mask.any() else None for j, m in enumerate(METRICS)})
        coverage = {}
        for flag in (True, False):
            mask = trained[:, i] == flag
            coverage['unit_trained' if flag else 'unit_not_trained'] = dict(draws=int(mask.sum()), f1=float(values[mask, i, 0].mean()) if mask.any() else None)
        methods[method] = dict(**{m: float(point[i, j]) for j, m in enumerate(METRICS)},
                               intervals={m: np.quantile(boot[:, i, j], [.025, .975]).tolist() for j, m in enumerate(METRICS)},
                               tracks=by_track, training_coverage=coverage)
    paired = []
    for left, right in COMPARISONS:
        i, k = METHODS.index(left), METHODS.index(right)
        paired.append(dict(left=left, right=right, metrics={m: dict(difference=float(point[i, j] - point[k, j]),
                           interval=np.quantile(boot[:, i, j] - boot[:, k, j], [.025, .975]).tolist()) for j, m in enumerate(METRICS)}))
    return dict(draws=len(draws), methods=methods, paired=paired)


def analyze_preference(config, output):
    manifest = config.finished('preference-data', 'preference_data_prepared')
    rules, domains, predictions = manifest['rules'], {}, 0
    for domain in DOMAINS:
        folder = config.stage('preference-data') / domain
        gold = list(iter_rows(folder / 'test.gold.jsonl', 'jsonl'))
        draws = [d for d in read_json(folder / 'draws.json')['draws'] if d['qa_key'] is not None]
        index = {g['qa_key']: i for i, g in enumerate(gold)}
        require(len(draws) == manifest['domains'][domain]['successful_draws']
                and Counter(d['qa_key'] for d in draws) == Counter({g['qa_key']: g['preference_draws'] for g in gold}),
                'preference_draw_mismatch', 'Each question is weighted by its draws')
        scored, runs = {}, {}
        for method in METHODS:
            run = config.stage('preference-models') / f'{domain}-{method}'
            scored[method], report = scored_outputs(run, gold)
            require(report['status'] == 'inference_completed' and (report['mode'], report['domain'], report['method']) == ('preference', domain, method)
                    and report['complete_test'], 'preference_run_mismatch', 'Every main-bank model answers every preference question')
            runs[method] = dict(seconds=report['seconds'], output_token_limit_reached=report['output_token_limit_reached'])
            predictions += len(gold)
        questions = np.asarray([[[scored[m][i][x] for x in METRICS] for m in METHODS] for i in range(len(gold))], dtype=float)
        result = summarize(questions[[index[d['qa_key']] for d in draws]], draws, np.asarray([d['track'] for d in draws]),
                           np.asarray([[d['trained'][m] for m in METHODS] for d in draws]), rules)
        result['unweighted_unique_questions'] = {m: dict(questions=len(gold), **{x: float(questions[:, i, j].mean()) for j, x in enumerate(METRICS)})
                                                 for i, m in enumerate(METHODS)}
        replay = read_json(config.stage('preference-models') / f'replay-{domain}-DP' / 'manifest.json')
        require(replay['status'] == 'replay_completed' and replay['domain'] == domain, 'missing_replay', 'Replay the DP model before comparing')
        result['replay'] = {k: replay[k] for k in ('method', 'compared', 'identical_outputs', 'identical_scores', 'original_f1', 'replayed_f1')}
        result['runs'] = runs
        result['data'] = manifest['domains'][domain]
        domains[domain] = result
    report = dict(status='preference_analysis_completed', domains=domains, predictions=predictions, bootstrap_replicates=rules['bootstrap_replicates'],
                  bootstrap_unit='preference_draw')
    write_json(output / 'manifest.json', report)
    return report
