"""Negative controls and likelihood-ratio attacks on the target releases; no measurement is taken.

The realistic adversary predicts each unit's support count among the construction users from the support of the
attack-reference users (negative binomial, Gamma prior) and scores a target by the log-likelihood ratio of one
additional supporter on the units its partial history matches. The informed adversary knows the contributions of every
other user. The controls score the frozen classifiers on placebo releases (shadow-calibration releases, which contain
no target) and fit history-only classifiers by cross-validation. Every analyzed view is first checked against the
post-processing of its stored vector.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp
from scipy.stats import mannwhitneyu, norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from privacyqa.attacks.features import ATTACKS, KnownIndex, PublishedView, classifier_scores, score_metrics, unit_features, validate_view
from privacyqa.attacks.metrics import binomial_interval, calibrate_points, saved_scores
from privacyqa.banks import load_banks
from privacyqa.cohorts import Knowledge, cohort_matrix, load_cohorts, ordered_users
from privacyqa.common import ARMS, DOMAINS, VIEWS, read_json, require, save_arrays, write_json
from privacyqa.mechanism import EPSILON, Q, SCALE
from privacyqa.release import load_releases, read_view
from privacyqa.supports import load_statistics

USER_SCALE = SCALE / Q  # The noise scale in user-equivalents (5).
SENTINEL = -1e12  # Score of a target whose membership is impossible; finite for tie-aware metrics.
POINTS = ('0_1pct', '1pct')
NOISELESS = ('Non-private', 'Bounded-non-private')
REFERENCE_GROUPS = ('train_pool', 'calibration_pool')
K_GROUPS = (('0', 0, 0), ('1', 1, 1), ('2-4', 2, 4), ('5-9', 5, 9), ('10-19', 10, 19), ('20+', 20, None))


# ---------------------------------------------------------------- shared statistics

def operating_points(scores, labels, mask=None):
    """Thresholds come from calibration nonmembers; the test mask never sets them."""
    calibration = calibrate_points(scores[labels == -1])
    test = (labels >= 0) if mask is None else (labels >= 0) & mask
    truth, values = labels[test], scores[test]
    npos, nneg = int(np.sum(truth == 1)), int(np.sum(truth == 0))
    result = dict(members=npos, nonmembers=nneg, auc=score_metrics(values[None], truth, [None])['auc'] if npos and nneg else None)
    for name in POINTS:
        cutoff = calibration[name]['threshold']
        predicted = values > cutoff
        tp, fp = int(np.sum(predicted & (truth == 1))), int(np.sum(predicted & (truth == 0)))
        result[name] = dict(threshold=float(cutoff), calibration_fpr=calibration[name]['calibration_fpr'],
                            low_resolution=calibration[name]['low_resolution'], tp=tp, fp=fp, tpr=tp / npos if npos else None,
                            fpr=fp / nneg if nneg else None, precision=tp / (tp + fp) if tp + fp else None,
                            tpr_cp95=binomial_interval(tp, npos) if npos else None, fpr_cp95=binomial_interval(fp, nneg) if nneg else None)
    return result


def target_sums(rows, units, unit_scores, targets):
    """Sum unit evidence over each target's matched units; one impossible unit decides the target."""
    values = unit_scores[units]
    impossible = np.bincount(rows, weights=np.isneginf(values), minlength=targets) > 0
    total = np.bincount(rows, weights=np.where(np.isneginf(values), 0., values), minlength=targets)
    return np.where(impossible, SENTINEL, total)


# ---------------------------------------------------------------- likelihood components

def log_nb(counts, shape, probability):
    """Negative-binomial log pmf with the Gamma extension to real counts; -inf below zero."""
    counts, shape = np.broadcast_arrays(np.asarray(counts, dtype=np.float64), np.asarray(shape, dtype=np.float64))
    result = np.full(counts.shape, -np.inf)
    valid = counts >= 0
    c, a = counts[valid], shape[valid]
    result[valid] = gammaln(c + a) - gammaln(a) - gammaln(c + 1) + a * math.log(probability) + c * math.log1p(-probability)
    return result


def count_llr(counts, shape, probability, contribution=1.):
    """log P(observed | the target adds `contribution`) - log P(observed | the target is absent)."""
    counts = np.asarray(counts, dtype=np.float64)
    present = log_nb(counts - contribution, shape, probability)
    return np.where(counts - contribution >= 0, present - log_nb(counts, shape, probability), -np.inf)


def release_llr(released, shape, probability):
    """C-only without noise: a member forces the release of its units; otherwise a unit is released with 1 - p^a."""
    log_zero = np.asarray(shape, dtype=np.float64) * math.log(probability)
    return np.where(released, -np.log1p(-np.exp(log_zero)), -np.inf)


def laplace_logpdf(z, scale):
    return -np.abs(z) / scale - math.log(2 * scale)


def laplace_logsf(z, scale):
    """log P(L > z) for a continuous Laplace variable."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, math.log(.5) - z / scale, np.log1p(-.5 * np.exp(np.minimum(z, 0) / scale)))


def laplace_logcdf(z, scale):
    return laplace_logsf(-np.asarray(z, dtype=np.float64), scale)


def discrete_laplace_logsf(m, scale=SCALE):
    """log P(Z > m) for the integer discrete Laplace with pmf proportional to exp(-|z|/scale)."""
    m = np.asarray(m, dtype=np.float64)
    log_q = -1. / scale
    log_norm = math.log1p(math.exp(log_q))
    upper = (np.maximum(m, 0) + 1) * log_q - log_norm
    lower = np.log1p(-np.exp(np.maximum(-m, 1) * log_q - log_norm))
    return np.where(m >= 0, upper, lower)


def discrete_laplace_logcdf(m, scale=SCALE):
    """log P(Z <= m), computed directly in both tails."""
    m = np.asarray(m, dtype=np.float64)
    log_q = -1. / scale
    log_norm = math.log1p(math.exp(log_q))
    upper = np.log1p(-np.exp((np.maximum(m, 0) + 1) * log_q - log_norm))
    lower = np.maximum(-m, 1) * log_q - log_norm
    return np.where(m >= 0, upper, lower)


def noisy_unit_llr(noisy, released, cutoff, shape, probability, rules, scale=USER_SCALE):
    """Unit evidence when a negative-binomial count plus Laplace noise is observed, in user-equivalents.

    `noisy` is the unthresholded vector; `released` is the C-only event at the threshold `cutoff`.
    """
    mean = shape * (1 - probability) / probability
    sd = np.sqrt(shape * (1 - probability)) / probability
    span = rules['grid_laplace_scales'] * scale
    full_view, plain_view = np.empty(len(shape)), np.empty(len(shape))
    for j in range(len(shape)):
        low, high = mean[j] - rules['grid_sd'] * sd[j], mean[j] + rules['grid_sd'] * sd[j]
        grid = np.arange(max(0, math.floor(min(low, noisy[j] - span))), math.ceil(max(high, noisy[j] + span)) + 2, dtype=np.float64)
        weight = log_nb(grid, shape[j], probability)
        full_view[j] = logsumexp(weight + laplace_logpdf(noisy[j] - grid - 1, scale)) - logsumexp(weight + laplace_logpdf(noisy[j] - grid, scale))
        grid = np.arange(max(0, math.floor(min(low, cutoff - span))), math.ceil(max(high, cutoff + span)) + 2, dtype=np.float64)
        weight = log_nb(grid, shape[j], probability)
        tail = laplace_logsf if released[j] else laplace_logcdf
        plain_view[j] = logsumexp(weight + tail(cutoff - grid - 1, scale)) - logsumexp(weight + tail(cutoff - grid, scale))
    return full_view, plain_view


# ---------------------------------------------------------------- published views

def unit_view(units, rows):
    """Released units and unit masses up to one normalizing constant."""
    index = {key: j for j, unit in enumerate(units) for key in unit}
    released, relative = np.zeros(len(units), dtype=bool), np.zeros(len(units))
    for row in rows:
        j = index[row['id']]
        released[j] = True
        if 'weight' in row:
            relative[j] = row['weight'] * len(units[j])
    return released, relative


def recover_counts(relative, released, limit=100000):
    """Integer counts from normalized weights: the smallest scale that makes every ratio integral."""
    if not released.any():
        return np.zeros(len(relative)), None
    ratio = relative[released] / relative[released].min()
    for scale in range(1, limit + 1):
        values = scale * ratio
        if np.all(np.abs(values - np.rint(values)) <= 1e-6 * np.maximum(1, values)):
            counts = np.zeros(len(relative))
            counts[released] = np.rint(values)
            return counts, scale
    require(False, 'count_recovery_failed', 'Weights without noise reveal integer counts up to one scale')


def proportional(relative, mass):
    total = float(mass.sum())
    if not total:
        return not np.any(relative)  # An empty release has no weights.
    return bool(np.allclose(relative, mass / total, rtol=1e-9, atol=1e-15))


def exact_informed(full_mass, contributions, member, relative, released):
    """Releases without noise: which hypothesis, with or without the target, reproduces the actual view."""
    weighted, plain = np.zeros(len(member), dtype=np.int64), np.zeros(len(member), dtype=np.int64)
    full_mass = np.asarray(full_mass, dtype=np.int64)
    for t in range(len(member)):
        start, end = contributions.indptr[t], contributions.indptr[t + 1]
        if start == end:
            continue
        units, values = contributions.indices[start:end], contributions.data[start:end].astype(np.int64)
        outcome = []
        for delta in (1 - int(member[t]), -int(member[t])):  # Data with, then without, the target.
            mass = full_mass.copy()
            mass[units] += delta * values
            outcome.append((proportional(relative, mass), np.array_equal(mass > 0, released)))
        for column, store in ((0, weighted), (1, plain)):
            inside, outside = outcome[0][column], outcome[1][column]
            require(inside or outside, 'informed_view_mismatch', 'One hypothesis reproduces the actual release')
            store[t] = int(inside) - int(outside)
    return weighted, plain


def weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def informed_dp(noisy, cutoff, full_mass, contributions, member, relative, released, scale=SCALE):
    """Exact discrete-Laplace evidence for an adversary who knows every other user and the target."""
    targets = contributions.shape[0]
    rows = np.repeat(np.arange(targets), np.diff(contributions.indptr))
    units = contributions.indices
    x = contributions.data.astype(np.int64)
    others = np.asarray(full_mass, dtype=np.int64)[units] - member[rows].astype(np.int64) * x
    outside = np.asarray(noisy, dtype=np.int64)[units] - others
    noisy_term = (np.abs(outside).astype(np.float64) - np.abs(outside - x).astype(np.float64)) / scale
    margin = cutoff - others
    plain_term = np.where(released[units], discrete_laplace_logsf(margin - x, scale) - discrete_laplace_logsf(margin, scale),
                          discrete_laplace_logcdf(margin - x, scale) - discrete_laplace_logcdf(margin, scale))
    # C+weight: estimate the normalizing constant from released units outside the target's support.
    shown = np.flatnonzero(released)
    ratio = (np.asarray(full_mass, dtype=np.float64)[shown] - cutoff) / relative[shown]
    weighted_term = plain_term.copy()
    estimates = np.full(targets, np.nan)
    for t in np.flatnonzero(np.diff(contributions.indptr)):
        start, end = contributions.indptr[t], contributions.indptr[t + 1]
        keep = ~np.isin(shown, units[start:end])
        if not keep.any():
            continue  # No released unit outside the target's support fixes the constant; keep the C-only evidence.
        estimates[t] = weighted_median(ratio[keep], relative[shown][keep])
        visible = released[units[start:end]]
        guessed = estimates[t] * relative[units[start:end]] + cutoff
        own = others[start:end].astype(np.float64)
        weighted_term[start:end] = np.where(visible, (np.abs(guessed - own) - np.abs(guessed - own - x[start:end])) / scale,
                                            plain_term[start:end])
    sums = lambda values: np.bincount(rows, weights=values, minlength=targets)
    return dict(noisy=sums(noisy_term), plain=sums(plain_term), weighted=sums(weighted_term)), estimates


# ---------------------------------------------------------------- target releases

@dataclass
class TargetRelease:
    arm: str
    domain: str
    bank: object
    raw_full: np.ndarray
    bounded_full: np.ndarray
    population: int
    targets_raw: sparse.csr_matrix
    targets_bounded: sparse.csr_matrix
    labels: np.ndarray
    positions: np.ndarray
    target_known: Knowledge
    reference_raw: np.ndarray
    reference_bounded: np.ndarray
    references: int
    views: dict
    noisy: np.ndarray
    cutoff: int
    records: list
    schedule: list


def target_release(config, banks, groups, schedules, releases, scores, arm, domain):
    bank = banks[arm][domain]
    statistics = load_statistics(config, arm, domain, bank)
    raw, bounded = cohort_matrix(config, arm, domain, 'raw'), cohort_matrix(config, arm, domain, 'bounded')
    known = Knowledge.load(config.stage('cohorts') / arm / f'{domain}-known.npz')
    target = schedules[domain][-1]
    require(target['split'] == 'test', 'invalid_schedule', 'The last scheduled release is the target release')
    positions, labels = np.asarray(target['target_positions']), np.asarray(target['labels'])
    saved = scores[domain, arm, 'DP', 'C+weight']
    target_known = known.take(positions.tolist())
    require(np.array_equal(saved['positions'], positions) and np.array_equal(saved['labels'], labels)
            and np.array_equal(np.bincount(target_known.rows, minlength=len(positions)), saved['known_matches']),
            'target_mismatch', 'Use the targets, labels and partial histories of the attack stage')
    index = {u: i for i, u in enumerate(ordered_users(groups[domain]))}
    references = [index[u] for g in REFERENCE_GROUPS for u in groups[domain][g]]
    require(not set(references) & set(positions.tolist()), 'reference_overlap', 'Attack-reference users are never targets')
    records = releases['records'][arm][domain]
    record = records[-1]
    require(record['release_id'] == target['release_id'], 'invalid_release_records', 'Analyze the target release')
    views = {}
    for method, entry in record['views'].items():
        views[method] = {}
        for view in VIEWS:
            rows = read_view(config, entry, view)
            validate_view(bank, rows, entry['release_id'], view == 'C+weight')
            views[method][view] = rows
    dp = read_json(config.stage('releases') / record['views']['DP']['directory'] / 'dp_mass.json')
    require(dp['release_id'] == record['views']['DP']['release_id'] and len(dp['integer_mass']) == len(bank.units)
            and dp['mechanism']['scale'] == SCALE and dp['mechanism']['epsilon'] == EPSILON,
            'noise_mismatch', 'Use the stored noisy vector of the target release')
    cutoff = releases['thresholds'][arm]
    require(cutoff['integer_scale'] == SCALE, 'threshold_mismatch', 'Use the threshold of the releases')
    return TargetRelease(arm, domain, bank, np.asarray(statistics['raw_mass'], dtype=np.int64),
                         np.asarray(statistics['bounded_mass'], dtype=np.int64), releases['populations'][domain]['construction_users'],
                         raw[positions], bounded[positions], labels, positions, target_known,
                         np.asarray(raw[references].sum(axis=0)).ravel().astype(np.float64),
                         np.asarray(bounded[references].sum(axis=0)).ravel() / Q, len(references), views,
                         np.asarray(dp['integer_mass'], dtype=np.int64), int(cutoff['threshold_integer']), records, schedules[domain])


def verify_views(ctx):
    """Every analyzed view equals the post-processing of its stated vector."""
    checks = {}
    masses = {'Non-private': ctx.raw_full, 'Bounded-non-private': ctx.bounded_full, 'DP': np.maximum(ctx.noisy - ctx.cutoff, 0)}
    for method, mass in masses.items():
        released, _ = unit_view(ctx.bank.units, ctx.views[method]['C-only'])
        shown, relative = unit_view(ctx.bank.units, ctx.views[method]['C+weight'])
        checks[method] = bool(np.array_equal(released, mass > 0) and np.array_equal(shown, mass > 0) and proportional(relative, mass))
    require(all(checks.values()), 'view_mismatch', 'Views are the post-processing of the stored vectors')
    return checks


# ---------------------------------------------------------------- analyses

def reference_attack(ctx, rules):
    probability = (rules['prior_rate'] + ctx.references) / (rules['prior_rate'] + ctx.references + ctx.population)
    raw_shape, bounded_shape = rules['prior_shape'] + ctx.reference_raw, rules['prior_shape'] + ctx.reference_bounded
    units = ctx.bank.units
    released, relative = unit_view(units, ctx.views['Non-private']['C+weight'])
    counts, scale = recover_counts(relative, released)
    # Integrality identifies counts only up to their greatest common divisor.
    divisor = int(np.gcd.reduce(ctx.raw_full[ctx.raw_full > 0])) if released.any() else 1
    dp_released, _ = unit_view(units, ctx.views['DP']['C-only'])
    noisy, plain = noisy_unit_llr(ctx.noisy / Q, dp_released, ctx.cutoff / Q, bounded_shape, probability, rules)
    evidence = {('Non-private', 'C+weight'): count_llr(counts, raw_shape, probability),
                ('Non-private', 'C-only'): release_llr(released, raw_shape, probability),
                ('Bounded-non-private', 'unnormalized'): count_llr(ctx.bounded_full / Q, bounded_shape, probability),
                ('DP', 'unthresholded'): noisy, ('DP', 'C-only'): plain}
    targets = len(ctx.labels)
    scores = {k: target_sums(ctx.target_known.rows, ctx.target_known.units, v, targets) for k, v in evidence.items()}
    matched = np.bincount(ctx.target_known.rows, minlength=targets) > 0
    rows = [dict(method=method, view=view, stratum=stratum, **operating_points(values, ctx.labels, mask))
            for (method, view), values in scores.items()
            for stratum, mask in (('all', None), ('known_match', matched), ('no_known_match', ~matched))]
    facts = dict(references=ctx.references, population=ctx.population, nb_probability=probability,
                 counts_recovered_from_view=bool(np.array_equal(counts.astype(np.int64), ctx.raw_full)),
                 counts_recovered_up_to_divisor=bool(np.array_equal(counts.astype(np.int64) * divisor, ctx.raw_full)),
                 count_common_divisor=divisor, recovery_scale=scale,
                 smallest_released_count=int(counts[released].min()) if released.any() else None,
                 dp_unit_evidence_bound=float(np.max(np.abs(noisy))), user_scale=USER_SCALE)
    return rows, facts, scores


def informed_attack(ctx):
    member = ctx.labels == 1
    units = ctx.bank.units
    scores = {}
    for method, mass, contributions in (('Non-private', ctx.raw_full, ctx.targets_raw), ('Bounded-non-private', ctx.bounded_full, ctx.targets_bounded)):
        released, relative = unit_view(units, ctx.views[method]['C+weight'])
        weighted, plain = exact_informed(mass, contributions, member, relative, released)
        scores[method, 'C+weight'], scores[method, 'C-only'] = weighted.astype(np.float64), plain.astype(np.float64)
    released, relative = unit_view(units, ctx.views['DP']['C+weight'])
    dp, estimates = informed_dp(ctx.noisy, ctx.cutoff, ctx.bounded_full, ctx.targets_bounded, member, relative, released)
    scores['DP', 'unthresholded'], scores['DP', 'C+weight'], scores['DP', 'C-only'] = dp['noisy'], dp['weighted'], dp['plain']
    support = np.diff(ctx.targets_raw.indptr)
    epsilon = np.asarray(ctx.targets_bounded.sum(axis=1)).ravel() / SCALE
    rows = []
    for (method, view), values in scores.items():
        rows.append(dict(method=method, view=view, stratum='all', **operating_points(values, ctx.labels)))
        for name, low, high in K_GROUPS:
            mask = (support >= low) & (support <= (high if high is not None else support.max(initial=0)))
            rows.append(dict(method=method, view=view, stratum='support_' + name,
                             individual_epsilon_max=float(epsilon[mask & (ctx.labels >= 0)].max(initial=0)),
                             **operating_points(values, ctx.labels, mask)))
    shown = np.flatnonzero(released)
    true_scale = float(np.maximum(ctx.noisy - ctx.cutoff, 0)[shown].sum())
    supported = np.isfinite(estimates)
    facts = dict(scale_relative_error_median=float(np.median(np.abs(estimates[supported] / true_scale - 1))) if supported.any() else None,
                 individual_epsilon_max=float(epsilon.max(initial=0)), mechanism_scale=SCALE,
                 targets_with_support={g: int(np.sum((support > 0) & (ctx.labels == v)))
                                       for g, v in (('members', 1), ('test_nonmembers', 0), ('calibration_nonmembers', -1))})
    return rows, facts, scores


def model_scores(models, published, known):
    x, names = unit_features(published, known)
    return dict(aggregate_classifier=classifier_scores(models['aggregate_classifier'], x[:, :len(names)]),
                unit_classifier=classifier_scores(models['unit_classifier'], x))


def placebo(ctx, config, scores, rules):
    """The frozen classifiers on shadow releases that contain no target, each with its own calibrated thresholds."""
    index = KnownIndex(ctx.bank)
    shadows = [r for r in ctx.records if r['split'] == rules['placebo_split']]
    members = {item['release_id']: item['members'] for item in ctx.schedule}
    targets = set(ctx.positions.tolist())
    require(shadows and all(not targets & set(members[r['release_id']]) for r in shadows), 'placebo_contains_target',
            'Placebo releases contain no target')
    rows, replayed = [], {}
    for method in (*NOISELESS, 'DP'):
        for view in VIEWS:
            folder = config.stage('attacks') / ctx.arm / ctx.domain / method / view
            models = {name: read_json(folder / f'{name}.json') for name in ('aggregate_classifier', 'unit_classifier')}
            entry = ctx.records[-1]['views'][method]
            actual = model_scores(models, PublishedView(index, ctx.views[method][view], entry['release_id'], view), ctx.target_known)
            saved = scores[ctx.domain, ctx.arm, method, view]
            replayed[method, view] = all(np.allclose(actual[a], saved[a], rtol=1e-12, atol=1e-12) for a in actual)
            require(replayed[method, view], 'model_replay_mismatch', 'The frozen classifiers reproduce the saved target scores')
            per_release = {a: [] for a in actual}
            for record in shadows:
                entry = record['views'][method]
                published = PublishedView(index, read_view(config, entry, view), entry['release_id'], view)
                for attack, values in model_scores(models, published, ctx.target_known).items():
                    per_release[attack].append(operating_points(values, ctx.labels))
            for attack, results in per_release.items():
                summary = lambda key, point=None: [r[key] if point is None else r[point][key] for r in results]
                rows.append(dict(method=method, view=view, attack=attack, releases=len(results),
                                 auc_mean=float(np.mean(summary('auc'))), auc_min=float(np.min(summary('auc'))), auc_max=float(np.max(summary('auc'))),
                                 **{f'{k}_{p}_{name}': float(f(summary(k, p))) for p in POINTS for k in ('tpr', 'fpr')
                                    for name, f in (('mean', np.mean), ('min', np.min), ('max', np.max))},
                                 **{f'{k}_{p}_sum': int(np.sum(summary(k, p))) for p in POINTS for k in ('tp', 'fp')},
                                 per_release=[dict(auc=r['auc'], **{f'{k}_{p}': r[p][k] for p in POINTS for k in ('tp', 'fp')}) for r in results],
                                 actual=operating_points(actual[attack], ctx.labels)))
    return rows, {f'{m}/{v}': ok for (m, v), ok in replayed.items()}


def blind_history(ctx, rules):
    """How well a classifier on partial histories alone separates members from test nonmembers."""
    targets, dimensions = len(ctx.labels), len(ctx.bank.units)
    known = ctx.target_known
    counts = np.bincount(known.rows, minlength=targets).astype(np.float64)
    mask = sparse.csr_matrix((np.ones(len(known.rows)), (known.rows, known.units)), shape=(targets, dimensions))
    x = sparse.hstack([sparse.csr_matrix(np.column_stack([known.observations, counts])), mask], format='csr')
    test = np.flatnonzero(ctx.labels >= 0)
    y = ctx.labels[test]
    cross_validated, fitted = None, 0
    if min(np.sum(y == 0), np.sum(y == 1)) >= rules['blind_folds']:
        out_of_fold = np.zeros(len(test))
        folds = StratifiedKFold(n_splits=rules['blind_folds'], shuffle=True, random_state=rules['blind_seed'])
        for train, held in folds.split(test, y):
            scaler = StandardScaler(with_mean=False).fit(x[test[train]])
            model = LogisticRegression(solver='liblinear', penalty='l2', C=1, max_iter=500, tol=1e-6, random_state=rules['blind_seed'])
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(scaler.transform(x[test[train]]), y[train])
            out_of_fold[held] = model.decision_function(scaler.transform(x[test[held]]))
            fitted += 1
        cross_validated = score_metrics(out_of_fold[None], y, [None])['auc']
    support = np.diff(ctx.targets_raw.indptr)
    groups = {name: ctx.labels == value for name, value in (('members', 1), ('test_nonmembers', 0), ('calibration_nonmembers', -1))}
    describe = {name: dict(users=int(m.sum()), known_observations_mean=float(known.observations[m].mean()),
                           known_match_share=float(np.mean(counts[m] > 0)), full_support_share=float(np.mean(support[m] > 0)),
                           full_support_units_mean=float(support[m].mean())) for name, m in groups.items()}
    member, nonmember = groups['members'], groups['test_nonmembers']

    def proportion_test(values):
        a, b = values[member], values[nonmember]
        pooled = np.r_[a, b].mean()
        se = math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b))) if 0 < pooled < 1 else 0
        return float(2 * norm.sf(abs(a.mean() - b.mean()) / se)) if se else 1.

    tests = dict(known_observations_mannwhitney_p=float(mannwhitneyu(known.observations[member], known.observations[nonmember]).pvalue),
                 known_match_share_p=proportion_test((counts > 0).astype(float)), full_support_share_p=proportion_test((support > 0).astype(float)))
    return dict(status='cross_validated' if fitted else 'insufficient_users', cross_validated_auc=cross_validated,
                folds=rules['blind_folds'], features=['known_observations', 'known_matching_units', 'known_unit_indicators'],
                populations=describe, tests_member_vs_test_nonmember=tests, models_fitted=fitted)


def match_decomposition(saved, scores):
    """Detections of the trained attacks at their thresholds, split by whether the partial history matches a unit."""
    rows = []
    for (domain, arm, method, view), row in saved.items():
        values = scores[domain, arm, method, view]
        labels, matched = values['labels'], values['known_matches'] > 0
        for attack in ATTACKS:
            cutoff = row['attacks'][attack]['calibration']['threshold']
            predicted = values[attack] > cutoff
            counts = {f'{side}_{stratum}': int(np.sum(predicted & (labels == value) & mask))
                      for side, value in (('tp', 1), ('fp', 0)) for stratum, mask in (('known_match', matched), ('no_known_match', ~matched))}
            rows.append(dict(domain=domain, arm=arm, method=method, view=view, attack=attack, threshold=cutoff,
                             strata=row['attacks'][attack]['strata'], **counts))
    return rows


def attack_audit(config, output):
    rules = config['attack_audit']
    banks = load_banks(config)
    groups, schedules = load_cohorts(config)
    releases = load_releases(config)
    _, saved, scores = saved_scores(config)
    report = dict(status='attack_audit_completed', rules=rules, matches=match_decomposition(saved, scores), domains=[], attack_models=0)
    for arm in ARMS:
        for domain in DOMAINS:
            ctx = target_release(config, banks, groups, schedules, releases, scores, arm, domain)
            checks = verify_views(ctx)
            reference, reference_facts, reference_scores = reference_attack(ctx, rules)
            informed, informed_facts, informed_scores = informed_attack(ctx)
            placebo_rows, replayed = placebo(ctx, config, scores, rules)
            blind = blind_history(ctx, rules)
            report['attack_models'] += blind['models_fitted']
            save_arrays(output / f'{arm}-{domain}-scores.npz', labels=ctx.labels, positions=ctx.positions,
                        **{'reference/' + '/'.join(k): v for k, v in reference_scores.items()},
                        **{'informed/' + '/'.join(k): v for k, v in informed_scores.items()})
            report['domains'].append(dict(arm=arm, domain=domain, view_checks=checks, reference=reference, reference_facts=reference_facts,
                                          informed=informed, informed_facts=informed_facts, placebo=placebo_rows,
                                          frozen_models_replayed=replayed, blind=blind))
    write_json(output / 'manifest.json', report)
    return report
