"""The releases: per bank and domain, the shadow-training, shadow-calibration and target releases of the schedule.

All releases share a background of every construction user except the member targets. A shadow release adds the
members of its schedule entry, attack-reference users; the target release adds the member targets, so it contains every
construction user, and the released datasets come from it. Every release is computed by the full procedure (DP) and by
three ablations without noise: Public-only (every unit), Non-private (raw support) and Bounded-non-private (bounded
support). Each DP release is one measurement, reserved in the ledger before the first one is taken.
"""
import numpy as np
from scipy import sparse

from privacyqa.banks import load_banks
from privacyqa.cohorts import cohort_matrix, load_cohorts
from privacyqa.common import ARMS, DOMAINS, VIEWS, progress, read_json, require, write_json
from privacyqa.mechanism import B, EPSILON, I64_MAX, Q, Ledger, add_noise, threshold
from privacyqa.supports import load_statistics

METHODS = ('Public-only', 'Non-private', 'Bounded-non-private', 'DP')


def release_masses(statistics, raw, bounded, schedule, population):
    """Raw and bounded masses of every scheduled release: the background plus the release's members.

    The background is the complete construction population minus the member targets, whose contributions are those of
    the cohort matrices; contributions are bounded per user once and never again.
    """
    require(sparse.isspmatrix_csr(raw) and sparse.isspmatrix_csr(bounded) and raw.shape == bounded.shape
            and raw.has_canonical_format and bounded.has_canonical_format and np.all(raw.data == 1)
            and np.array_equal(raw.indptr, bounded.indptr) and np.array_equal(raw.indices, bounded.indices),
            'invalid_cohort_contributions', 'Use the binary cohort supports and their bounded contributions')
    counts = np.diff(raw.indptr)
    require(np.array_equal(bounded.data, np.repeat(B * Q // np.maximum(B, counts), counts)), 'invalid_cohort_contributions',
            'Bounded contributions follow the bound of each user')
    target = schedule[-1]
    n = len(target['members'])
    require(target['split'] == 'test' and 0 < n <= population and population * B * Q <= I64_MAX, 'invalid_population',
            'The member targets are a nonempty part of the construction population')
    backgrounds = {}
    for name, matrix, scale in (('raw_mass', raw, 1), ('bounded_mass', bounded, Q)):
        total = np.asarray(statistics[name], dtype=np.int64)
        background = total - np.asarray(matrix[target['members']].sum(axis=0)).ravel()
        require(len(total) == raw.shape[1] and np.all(background >= 0) and np.all(background <= (population - n) * scale)
                and (name == 'raw_mass' or int(background.sum()) <= (population - n) * B * Q),
                'invalid_background', 'The member targets are part of the complete construction population')
        backgrounds[name] = background
    result = {}
    for item in schedule:
        require(len(item['members']) == n == len(set(item['members'])), 'invalid_schedule', 'Every release adds as many members')
        result[item['release_id']] = {name: (background + np.asarray((raw if name == 'raw_mass' else bounded)[item['members']].sum(axis=0)).ravel()).tolist()
                                      for name, background in backgrounds.items()}
    require(result[target['release_id']] == {k: statistics[k] for k in backgrounds}, 'target_release_mismatch',
            'The target release contains exactly the complete construction population')
    return result


def plan_releases(config):
    plan = config['release']
    banks = load_banks(config)
    groups, schedules = load_cohorts(config)
    supports = config.finished('supports', 'supports_built')
    planned = len(ARMS) * len(DOMAINS) * (plan['training_releases'] + plan['calibration_releases'] + 1)
    require(plan['measurements'] == planned, 'measurement_count_mismatch', f'Declare the {planned} planned measurements')
    thresholds = {arm: threshold(sum(len(banks[arm][d].units) for d in DOMAINS), plan['familywise_alpha']) for arm in ARMS}
    populations = {d: dict(construction_users=supports['domains'][d]['users'], member_targets=len(groups[d]['target_members']),
                           background_users=supports['domains'][d]['users'] - len(groups[d]['target_members']),
                           test_nonmembers=len(groups[d]['target_nonmembers']), calibration_nonmembers=len(groups[d]['target_calibration']))
                   for d in DOMAINS}
    masses = {arm: {d: release_masses(load_statistics(config, arm, d, banks[arm][d]), cohort_matrix(config, arm, d, 'raw'),
                                      cohort_matrix(config, arm, d, 'bounded'), schedules[d], populations[d]['construction_users'])
                    for d in DOMAINS} for arm in ARMS}
    frozen = dict(experiment_id=plan['experiment_id'], measurements=planned, epsilon_per_measurement=EPSILON, delta=0,
                  sequential_epsilon=planned * EPSILON, thresholds=thresholds, populations=populations,
                  releases={d: [dict(release_id=r['release_id'], split=r['split']) for r in schedules[d]] for d in DOMAINS})
    return banks, schedules, masses, frozen


def run_releases(config, output, dry_run=False):
    banks, schedules, masses, frozen = plan_releases(config)
    write_json(output / 'plan.json', frozen)
    if dry_run:
        manifest = dict(status='releases_planned', **frozen, measurements_taken=0)
        write_json(output / 'manifest.json', manifest)
        return manifest
    prefix = frozen['experiment_id']
    ledger = Ledger(output / 'ledger.sqlite', [(f'{prefix}-{arm}-{r["release_id"]}', d) for arm in ARMS for d in DOMAINS for r in schedules[d]])
    records = {arm: {d: [] for d in DOMAINS} for arm in ARMS}
    try:
        for arm in ARMS:
            for domain in DOMAINS:
                bank = banks[arm][domain]
                for item in schedules[domain]:
                    views = {}
                    for method in METHODS:
                        name = 'public' if method == 'Public-only' else item['release_id']
                        folder = output / arm / domain / method / name
                        release_id = (f'{prefix}-{arm}-{domain}-public' if method == 'Public-only' else
                                      f'{prefix}-{arm}-{item["release_id"]}' + ('' if method == 'DP' else '-' + method))
                        views[method] = dict(directory=str(folder.relative_to(output)), release_id=release_id)
                        if folder.exists():
                            continue
                        folder.mkdir(parents=True, mode=0o700)
                        mass, cutoff = masses[arm][domain][item['release_id']], 0
                        if method == 'DP':
                            ledger.mark(release_id, 'measuring')
                            try:
                                vector, mechanism = add_noise(mass['bounded_mass'])
                            except Exception:
                                ledger.mark(release_id, 'failed')
                                raise
                            ledger.mark(release_id, 'complete')
                            write_json(folder / 'dp_mass.json', dict(release_id=release_id, integer_mass=vector, mechanism=mechanism))
                            cutoff = frozen['thresholds'][arm]['threshold_integer']
                        else:
                            vector = [1] * len(bank.units) if method == 'Public-only' else mass['raw_mass' if method == 'Non-private' else 'bounded_mass']
                        released = bank.reconstruct(vector, release_id, cutoff)
                        for view in VIEWS:
                            write_json(folder / f'{view}.json', released[view])
                    records[arm][domain].append(dict(release_id=item['release_id'], split=item['split'], views=views))
                    progress(output, 'release_completed', arm=arm, domain=domain, release=item['release_id'])
        accounting = ledger.totals()
    finally:
        ledger.close()
    manifest = dict(status='releases_completed', **frozen, records=records, ledger=accounting, noise_saved=False)
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_releases(config):
    manifest = config.finished('releases', 'releases_completed')
    require(manifest['experiment_id'] == config['release']['experiment_id'] and manifest['ledger']['statuses'] == {'complete': manifest['measurements']},
            'incomplete_releases', 'Use the completed releases of this run, with every measurement complete')
    return manifest


def read_view(config, entry, view):
    return read_json(config.stage('releases') / entry['directory'] / f'{view}.json')
