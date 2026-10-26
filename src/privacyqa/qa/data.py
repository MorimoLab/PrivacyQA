"""QA data from the target releases, and the fidelity of every training source to held-out reference users.

The eight training sources are the C+weight views of the target release of each bank under each procedure. The
reference users' histories are read only here, after the banks, sources and test are fixed, and never for training.
"""
import numpy as np

from privacyqa.banks import load_banks
from privacyqa.cohorts import load_cohorts
from privacyqa.common import ARMS, DOMAINS, progress, read_json, require, save_arrays, write_json
from privacyqa.evidence import SupportRule, histories, positive_evidence
from privacyqa.qa.records import build_domain
from privacyqa.release import METHODS, load_releases, read_view
from privacyqa.supports import Aggregate

SOURCES = (*METHODS, *(f'Random-{m}' for m in METHODS))


def source_name(arm, method):
    return method if arm == 'main' else 'Random-' + method


def training_sources(config, releases):
    """{domain: {source: C+weight rows}} of the target releases."""
    result = {d: {} for d in DOMAINS}
    for arm in ARMS:
        for domain in DOMAINS:
            targets = [r for r in releases['records'][arm][domain] if r['split'] == 'test']
            require(len(targets) == 1, 'invalid_release_records', 'One target release per bank and domain')
            for method in METHODS:
                result[domain][source_name(arm, method)] = read_view(config, targets[0]['views'][method], 'C+weight')
    return result


def prepare_qa(config, output):
    plan = config['qa']
    banks = load_banks(config, arms=('main',))
    releases = load_releases(config)
    controls = read_json(config.stage('banks') / 'common_test.json')
    sources = training_sources(config, releases)
    summaries = {}
    for domain in DOMAINS:
        write_json(output / f'{domain}-sources.json', sources[domain])
        summaries[domain] = build_domain(banks['main'][domain].catalog, controls[domain], sources[domain], plan, output / domain)
        progress(output, 'qa_domain_prepared', domain=domain,
                 training_records={m: v['training_records'] for m, v in summaries[domain]['sources'].items()})
    manifest = dict(status='qa_data_prepared', plan={k: plan[k] for k in ('seed', 'development_requests', 'candidates', 'negative_reservoir', 'posting_sample')},
                    construction_users={d: releases['populations'][d]['construction_users'] for d in DOMAINS}, domains=summaries)
    write_json(output / 'manifest.json', manifest)
    return manifest


def tv(left, right):
    if not left or not right:
        return None
    return sum(abs(left.get(k, 0) - right.get(k, 0)) for k in left.keys() | right.keys()) / 2


def reference_fidelity(config, output):
    """User coverage, mass coverage and total variation distance of every source to the reference users' support."""
    banks = load_banks(config)
    groups, _ = load_cohorts(config)
    releases = load_releases(config)
    sources = training_sources(config, releases)
    results = {}
    for domain in DOMAINS:
        users = sorted(groups[domain]['reference_users'])
        rules = {arm: SupportRule(banks[arm][domain]) for arm in ARMS}
        aggregates = {arm: Aggregate(len(banks[arm][domain].units)) for arm in ARMS}
        matrices = {arm: [] for arm in ARMS}
        catalog = banks['main'][domain].catalog
        for _, records in histories(config, domain, users):
            positive = positive_evidence(domain, records, catalog)
            for arm, rule in rules.items():
                support = rule.encode(positive)
                aggregates[arm].add(support)
                matrices[arm].append(support.astype(np.uint8))
        distributions, per_source = {}, {}
        for arm, aggregate in aggregates.items():
            bank = banks[arm][domain]
            distributions[arm] = [{r['id']: r['weight'] for r in bank.reconstruct(v.tolist(), 'reference')['C+weight']}
                                  for v in (aggregate.raw, aggregate.bounded)]
            matrix = np.stack(matrices[arm])
            save_arrays(output / f'{domain}-{arm}-support.npz', support=matrix)
            unit_of = {cid: i for i, unit in enumerate(bank.units) for cid in unit}
            raw, bounded = distributions[arm]
            for method in METHODS:
                name = source_name(arm, method)
                p = {r['id']: r['weight'] for r in sources[domain][name]}
                covered = int(matrix[:, sorted({unit_of[k] for k in p})].any(axis=1).sum())
                per_source[name] = dict(reference_users_covered=covered, reference_user_coverage=covered / len(users),
                                        reference_probability_mass_covered=sum(raw.get(k, 0) for k in p) if raw else None,
                                        weight_TV_raw_reference=tv(p, raw), weight_TV_bounded_reference=tv(p, bounded))
        write_json(output / f'{domain}-distributions.json', distributions)
        results[domain] = dict(users=len(users), sources=per_source)
        progress(output, 'fidelity_completed', domain=domain, users=len(users))
    manifest = dict(status='qa_fidelity_completed', domains=results)
    write_json(output / 'manifest.json', manifest)
    return manifest
