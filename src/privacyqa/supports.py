"""Supports and bounded contributions of all construction users on both banks, before any noise.

Every construction user enters, including users who support no unit. The member targets frozen with the cohorts must
have the same contributions here as in the cohort matrices, because the releases replace them by shadow members.
"""
from collections import Counter
from fractions import Fraction

import numpy as np

from privacyqa.banks import load_banks
from privacyqa.cohorts import cohort_matrix, load_cohorts, ordered_users
from privacyqa.common import ARMS, DOMAINS, progress, read_json, require, write_json
from privacyqa.evidence import SupportRule, histories, positive_evidence
from privacyqa.mechanism import B, I64_MAX, Q
from privacyqa.roles import load_roles


class Aggregate:
    def __init__(self, dimensions):
        self.raw = np.zeros(dimensions, dtype=np.int64)
        self.bounded = np.zeros(dimensions, dtype=np.int64)
        self.histogram = Counter()

    def add(self, support):
        require(np.all((support == 0) | (support == 1)), 'nonbinary_support', 'Supports are binary')
        k = int(support.sum())
        bounded = support * (B * Q // max(B, k))
        self.raw += support
        self.bounded += bounded
        self.histogram[k] += 1
        return bounded

    def metrics(self):
        users = sum(self.histogram.values())
        clipped = sum(min(k, B) * n for k, n in self.histogram.items())
        bounded = int(self.bounded.sum())
        return dict(users=users, users_with_support=users - self.histogram[0], zero_support_users=self.histogram[0],
                    support_coverage=(users - self.histogram[0]) / users if users else None,
                    units_with_support=int(np.count_nonzero(self.raw)), dimensions=len(self.raw),
                    raw_mass=int(self.raw.sum()), bounded_mass=bounded / Q, clipping_loss=int(self.raw.sum()) - clipped,
                    quantization_error=str(Fraction(clipped) - Fraction(bounded, Q)),
                    users_clipped=sum(n for k, n in self.histogram.items() if k > B),
                    supported_units_histogram={str(k): n for k, n in sorted(self.histogram.items())})


def build_supports(config, output):
    banks = load_banks(config)
    cohorts, _ = load_cohorts(config)
    summary = {}
    for domain in DOMAINS:
        users = sorted(u for u, role in load_roles(config, domain).items() if role == 'protected')
        require(users and len(users) * B * Q <= I64_MAX, 'invalid_population', 'The bounded masses fit signed 64-bit integers')
        rules = {arm: SupportRule(banks[arm][domain]) for arm in ARMS}
        aggregates = {arm: Aggregate(len(banks[arm][domain].units)) for arm in ARMS}
        positions = {u: i for i, u in enumerate(ordered_users(cohorts[domain]))}
        members = {u: positions[u] for u in cohorts[domain]['target_members']}
        cohort = {arm: (cohort_matrix(config, arm, domain, 'raw'), cohort_matrix(config, arm, domain, 'bounded')) for arm in ARMS}
        checked = Counter()
        catalog = banks['main'][domain].catalog
        for done, (user, records) in enumerate(histories(config, domain, users), 1):
            positive = positive_evidence(domain, records, catalog)
            for arm, rule in rules.items():
                support = rule.encode(positive)
                bounded = aggregates[arm].add(support)
                if user in members:
                    raw_rows, bounded_rows = cohort[arm]
                    require(np.array_equal(support, raw_rows.getrow(members[user]).toarray().ravel())
                            and np.array_equal(bounded, bounded_rows.getrow(members[user]).toarray().ravel()),
                            'member_support_mismatch', 'Member targets have the same contributions as in the cohort matrices')
                    checked[arm] += 1
            if done % 50000 == 0:
                progress(output, 'construction_users_encoded', domain=domain, users=done, total=len(users))
        require(all(checked[arm] == len(members) for arm in ARMS), 'missing_member_targets',
                'Every member target is a construction user')
        summary[domain] = dict(users=len(users), arms={})
        for arm, aggregate in aggregates.items():
            metrics = aggregate.metrics()
            write_json(output / f'{domain}-{arm}.json', dict(metrics=metrics, raw_mass=aggregate.raw.tolist(),
                       bounded_mass=aggregate.bounded.tolist(), units=[list(u) for u in banks[arm][domain].units]))
            summary[domain]['arms'][arm] = metrics
        progress(output, 'construction_supports_built', domain=domain, users=len(users))
    manifest = dict(status='supports_built', domains=summary)
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_statistics(config, arm, domain, bank):
    statistics = read_json(config.stage('supports') / f'{domain}-{arm}.json')
    require(statistics['units'] == [list(u) for u in bank.units], 'support_unit_mismatch', 'Supports and bank share their units')
    return statistics
