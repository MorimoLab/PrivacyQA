"""Cohorts of the membership experiments, the schedule of shadow releases, and the cohorts' supports and partial histories.

Per domain, member targets come from the construction users, test and calibration nonmembers from the nonmember role,
two pools of attack-reference users (each twice the number of members) feed the shadow-training and shadow-calibration
releases, and reference users are held out for evaluation. The groups come from the private cohort file when the
configuration names one, and otherwise from a fixed pseudorandom order of each role. Reference histories are not read.
"""
from collections import Counter, defaultdict
import random

import numpy as np
from scipy import sparse

from privacyqa.banks import load_banks
from privacyqa.common import ARMS, DOMAINS, load_arrays, load_sparse, progress, rank, read_json, require, save_arrays, save_sparse, write_json
from privacyqa.evidence import SupportRule, histories, partial_history, positive_evidence
from privacyqa.mechanism import B, Q
from privacyqa.roles import load_roles

GROUPS = ('train_pool', 'calibration_pool', 'target_members', 'target_calibration', 'target_nonmembers')
SCHEDULE = ('training_releases', 'calibration_releases', 'targets_per_class', 'schedule_seed')
ROLE_OF = dict(train_pool='attack_reference', calibration_pool='attack_reference', target_members='protected',
               target_calibration='attack_nonmember', target_nonmembers='attack_nonmember', reference_users='utility_reference')


def group_sizes(plan):
    return dict(train_pool=2 * plan['members'], calibration_pool=2 * plan['members'], target_members=plan['members'],
                target_calibration=plan['calibration_nonmembers'], target_nonmembers=plan['test_nonmembers'],
                reference_users=plan['reference_users'])


def select_groups(roles, domain, plan):
    """Consecutive blocks of each role in a fixed pseudorandom order."""
    by_role = defaultdict(list)
    for user, role in roles.items():
        by_role[role].append(user)
    for users in by_role.values():
        users.sort(key=lambda u: (rank(plan['selection_seed'], domain, u), u))
    taken, groups = Counter(), {}
    for group, size in group_sizes(plan).items():
        role = ROLE_OF[group]
        groups[group] = by_role[role][taken[role]:taken[role] + size]
        taken[role] += size
        require(len(groups[group]) == size, 'insufficient_users', f'{domain}: the {role} role has too few users for {group}')
    return groups


def check_groups(groups, roles, domain, plan):
    sizes = group_sizes(plan)
    require(set(groups) == set(sizes) and all(len(groups[g]) == n for g, n in sizes.items()), 'invalid_cohort',
            f'{domain}: the cohort file gives every group with the planned size')
    flat = [u for users in groups.values() for u in users]
    require(len(flat) == len(set(flat)) and all(roles.get(u) == ROLE_OF[g] for g, users in groups.items() for u in users),
            'cohort_role_mismatch', f'{domain}: cohort groups are disjoint and drawn from their roles')
    return groups


def ordered_users(groups):
    """Row order of every cohort matrix."""
    return [u for g in GROUPS for u in groups[g]]


def shadow_schedule(groups, plan, domain):
    """Shadow releases in complementary pairs, then the target release.

    Each pair splits a pool into halves, and each half joins one release of the pair; the pair's targets are
    `targets_per_class` users from each half, so every target is a member of exactly one release of the pair. The
    target release has the member targets as members and labels members 1, test nonmembers 0, calibration nonmembers -1.
    """
    ordered = ordered_users(groups)
    positions = {u: i for i, u in enumerate(ordered)}
    n, m, result = len(groups['target_members']), plan['targets_per_class'], []
    for split, count in (('train', plan['training_releases']), ('calibration', plan['calibration_releases'])):
        pool = [positions[u] for u in groups[split + '_pool']]
        require(len(pool) == 2 * n and count % 2 == 0, 'invalid_schedule', 'Pools hold two member populations; releases come in pairs')
        for pair in range(count // 2):
            rng = random.Random(rank(plan['schedule_seed'], domain, split, pair))
            shuffled = rng.sample(pool, len(pool))
            halves = shuffled[:n], shuffled[n:]
            targets = rng.sample(halves[0], m) + rng.sample(halves[1], m)
            for side in (0, 1):
                members = sorted(halves[side])
                member_set = set(members)
                result.append(dict(release_id=f'{domain}-{split}-{2 * pair + side:03d}', split=split, pair=pair,
                                   members=members, target_positions=targets, labels=[int(t in member_set) for t in targets]))
    targets = [positions[u] for g in ('target_members', 'target_nonmembers', 'target_calibration') for u in groups[g]]
    result.append(dict(release_id=f'{domain}-test', split='test', pair=None,
                       members=[positions[u] for u in groups['target_members']], target_positions=targets,
                       labels=[1] * n + [0] * len(groups['target_nonmembers']) + [-1] * len(groups['target_calibration'])))
    return result


class Knowledge:
    """Units matched by each cohort user's partial history (sparse), and the size of that partial history."""

    def __init__(self, rows, units, observations):
        self.rows, self.units, self.observations = rows, units, observations

    @classmethod
    def from_matches(cls, matches, observations):
        return cls(np.repeat(np.arange(len(matches)), [len(v) for v in matches]),
                   np.concatenate(matches).astype(np.int64), np.asarray(observations))

    def save(self, path):
        save_arrays(path, rows=self.rows, units=self.units, observations=self.observations)

    @classmethod
    def load(cls, path):
        values = load_arrays(path)
        return cls(values['rows'], values['units'], values['observations'])

    def take(self, positions):
        """The partial histories of the listed users, renumbered in the listed order."""
        require(len(positions) == len(set(positions)), 'duplicate_target', 'One row per target')
        mapping = np.full(len(self.observations), -1, dtype=np.int64)
        mapping[positions] = np.arange(len(positions))
        selected = mapping[self.rows] >= 0
        return Knowledge(mapping[self.rows[selected]], self.units[selected], self.observations[positions])


def freeze_cohorts(config, output):
    plan, seed, file = config['cohorts'], config['private']['knowledge_seed'], config['private']['cohort_file']
    banks = load_banks(config)
    given = read_json(config.file(file))['domains'] if file else None
    cohorts, schedules, populations = {}, {}, {arm: {} for arm in ARMS}
    for arm in ARMS:
        (output / arm).mkdir(mode=0o700)
    for domain in DOMAINS:
        roles = load_roles(config, domain)
        groups = check_groups(given[domain], roles, domain, plan) if given else select_groups(roles, domain, plan)
        cohorts[domain] = groups
        schedules[domain] = shadow_schedule(groups, config['release'], domain)
        ordered = ordered_users({g: groups[g] for g in GROUPS})
        positions = {u: i for i, u in enumerate(ordered)}
        rules = {arm: SupportRule(banks[arm][domain]) for arm in ARMS}
        entries = {arm: ([], [], []) for arm in ARMS}
        matches = {arm: [None] * len(ordered) for arm in ARMS}
        observations = np.zeros(len(ordered), dtype=np.int16)
        budget = plan['known_reviews'] if domain == 'recipe_text' else plan['known_ratings']
        catalog = banks['main'][domain].catalog
        for done, (user, records) in enumerate(histories(config, domain, sorted(ordered)), 1):
            position = positions[user]
            positive = positive_evidence(domain, records, catalog)
            known, observations[position] = partial_history(domain, user, records, seed, budget)
            for arm, rule in rules.items():
                units = np.flatnonzero(rule.encode(positive))
                rows, cols, masses = entries[arm]
                rows.extend([position] * len(units))
                cols.extend(units.tolist())
                masses.extend([B * Q // max(B, len(units))] * len(units))
                matches[arm][position] = np.flatnonzero(rule.encode(known))
            if done % 5000 == 0:
                progress(output, 'cohort_users_encoded', domain=domain, users=done, total=len(ordered))
        for arm in ARMS:
            rows, cols, masses = entries[arm]
            shape = len(ordered), len(banks[arm][domain].units)
            raw = sparse.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=shape)
            bounded = sparse.csr_matrix((np.asarray(masses, dtype=np.int64), (rows, cols)), shape=shape)
            save_sparse(output / arm / f'{domain}-raw.npz', raw)
            save_sparse(output / arm / f'{domain}-bounded.npz', bounded)
            Knowledge.from_matches(matches[arm], observations).save(output / arm / f'{domain}-known.npz')
            active = np.diff(raw.indptr) > 0
            populations[arm][domain] = {g: dict(users=len(groups[g]), users_with_support=int(sum(active[positions[u]] for u in groups[g])),
                                                users_with_known_matches=int(sum(len(matches[arm][positions[u]]) > 0 for u in groups[g])))
                                        for g in GROUPS}
        progress(output, 'cohort_domain_frozen', domain=domain)
    write_json(output / 'cohorts.json', cohorts)
    write_json(output / 'schedules.json', schedules)
    manifest = dict(status='cohorts_frozen', plan=plan, schedule={k: config['release'][k] for k in SCHEDULE}, cohort_file_used=bool(file),
                    groups={d: {g: len(v) for g, v in groups.items()} for d, groups in cohorts.items()}, populations=populations)
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_cohorts(config):
    manifest = config.finished('cohorts', 'cohorts_frozen')
    require(manifest['plan'] == config['cohorts'] and manifest['schedule'] == {k: config['release'][k] for k in SCHEDULE}, 'cohort_plan_mismatch',
            'The cohorts were frozen under another plan')
    return read_json(config.stage('cohorts') / 'cohorts.json'), read_json(config.stage('cohorts') / 'schedules.json')


def cohort_matrix(config, arm, domain, kind):
    return load_sparse(config.stage('cohorts') / arm / f'{domain}-{kind}.npz')
