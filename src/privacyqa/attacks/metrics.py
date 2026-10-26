"""Extended metrics of the trained attacks and approximate simultaneous intervals, from their saved target scores.

Intervals resample members and nonmembers separately and condition on the fitted attacks, thresholds, backgrounds and
the noise of each target release. For the primary attack (C+weight, per-unit classifier, all targets), a paired maxT
procedure covers seven comparisons and ten metrics per domain at level alpha/3.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import beta

from privacyqa.attacks.features import ATTACKS, calibrate_threshold, score_metrics
from privacyqa.attacks.trained import target_scores
from privacyqa.common import ARMS, DOMAINS, VIEWS, require, write_json
from privacyqa.release import METHODS

RATES = (('0_1pct', 1, 1000), ('1pct', 1, 100), ('5pct', 1, 20), ('10pct', 1, 10))
RANK_METRICS = ('auc', 'average_precision', 'eer', 'max_advantage_descriptive', 'pauc_1pct', 'precision_top_1pct', 'precision_top_5pct')
RATE_METRICS = ('tpr', 'fpr', 'precision', 'ppv_prior_10pct', 'ppv_prior_1pct', 'advantage', 'balanced_accuracy', 'mcc', 'f1')
METRICS = RANK_METRICS + tuple(f'{m}_{r}' for r, _, _ in RATES for m in RATE_METRICS)
FAMILY = ('auc', 'average_precision', 'eer', 'pauc_1pct', 'tpr_1pct', 'fpr_1pct', 'advantage_1pct', 'tpr_5pct', 'fpr_5pct', 'advantage_5pct')
COMPARISONS = (
    ('full_vs_unbounded', ('main', 'DP'), ('main', 'Non-private')),
    ('full_vs_bounded', ('main', 'DP'), ('main', 'Bounded-non-private')),
    ('full_vs_random_raw', ('main', 'DP'), ('random', 'Non-private')),
    ('unbounded_vs_random_raw', ('main', 'Non-private'), ('random', 'Non-private')),
    ('bounded_vs_random_raw', ('main', 'Bounded-non-private'), ('random', 'Non-private')),
    ('full_vs_random_dp', ('main', 'DP'), ('random', 'DP')),
    ('bounded_vs_random_bounded', ('main', 'Bounded-non-private'), ('random', 'Bounded-non-private')))


def calibrate_points(scores):
    """Thresholds at every rate target, set on calibration nonmembers (strict >)."""
    ordered = np.sort(scores)
    n = len(ordered)
    return {name: dict(threshold=float(ordered[n - n * num // den - 1]), calibration_nonmembers=n,
                       calibration_fpr=float(np.mean(ordered > ordered[n - n * num // den - 1])), expected_tail=n * num / den,
                       low_resolution=n * num / den < 10) for name, num, den in RATES}


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(denominator), np.nan), where=denominator != 0)


@dataclass
class ExtendedCurve:
    scores: np.ndarray
    labels: np.ndarray
    thresholds: dict

    def __post_init__(self):
        ordered, self.bins = np.unique(-self.scores, return_inverse=True)
        self.cuts = np.array([np.searchsorted(ordered, -self.thresholds[r], side='left') for r, _, _ in RATES])
        self.positive = self.labels == 1
        self.negative = self.labels == 0

    def vector(self, weights):
        p = np.bincount(self.bins, weights=weights * self.positive)
        n = np.bincount(self.bins, weights=weights * self.negative)
        cp, cn = np.r_[0., np.cumsum(p)], np.r_[0., np.cumsum(n)]
        tpr, fpr = cp / cp[-1], cn / cn[-1]
        precision = np.divide(cp[1:], cp[1:] + cn[1:], out=np.zeros(len(p)), where=(cp[1:] + cn[1:]) > 0)
        ap = float(np.dot(p / cp[-1], precision))  # Non-interpolated average precision over whole tied groups.
        crossing = np.searchsorted(fpr + tpr, 1., side='left')
        x = crossing - 1
        eer = fpr[x] + (1 - fpr[x] - tpr[x]) * (fpr[crossing] - fpr[x]) / (fpr[crossing] + tpr[crossing] - fpr[x] - tpr[x])
        stop = np.searchsorted(fpr, .01, side='right')
        raw = np.trapezoid(np.r_[tpr[:stop], np.interp(.01, fpr[stop - 1:stop + 1], tpr[stop - 1:stop + 1])], np.r_[fpr[:stop], .01])
        top = []
        for fraction in (.01, .05):
            k = max(1, math.ceil(fraction * (cp[-1] + cn[-1])))
            top.append(np.interp(k, cp + cn, cp) / k)  # Expected hits under a random order within a tie.
        tp, fp = cp[self.cuts], cn[self.cuts]
        fn, tn = cp[-1] - tp, cn[-1] - fp
        tr, fr = tp / cp[-1], fp / cn[-1]
        rates = np.column_stack((tr, fr, divide(tp, tp + fp), divide(.1 * tr, .1 * tr + .9 * fr), divide(.01 * tr, .01 * tr + .99 * fr),
                                 tr - fr, (1 + tr - fr) / 2, divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
                                 2 * tp / (2 * tp + fp + fn)))
        return np.r_[np.trapezoid(tpr, fpr), ap, eer, np.max(tpr - fpr), .5 * (1 + (raw - .00005) / .00995), top, rates.ravel()]

    def identical_function(self, other, metric):
        if metric in RANK_METRICS:
            return np.array_equal(self.bins, other.bins)
        rate = next(i for i, (r, _, _) in enumerate(RATES) if metric.endswith('_' + r))
        keep = self.positive if metric.startswith('tpr_') else self.negative if metric.startswith('fpr_') else np.ones(len(self.labels), dtype=bool)
        return np.array_equal((self.bins < self.cuts[rate])[keep], (other.bins < other.cuts[rate])[keep])


def interval(values):
    # A resample with no alarm leaves precision undefined; such intervals are not reported.
    return np.quantile(values, [.025, .975], method='linear').tolist() if np.isfinite(values).all() else None


def binomial_interval(k, n):
    return [0. if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1. if k == n else float(beta.ppf(.975, k + 1, n - k))]


def counts_for(curve):
    result = {}
    for i, (name, num, den) in enumerate(RATES):
        predicted = curve.bins < curve.cuts[i]
        tp, fp = int(np.sum(predicted & curve.positive)), int(np.sum(predicted & curve.negative))
        npos, nneg = int(curve.positive.sum()), int(curve.negative.sum())
        result[name] = dict(tp=tp, fp=fp, fn=npos - tp, tn=nneg - fp, alarms=tp + fp, test_expected_tail=nneg * num / den,
                            low_resolution=nneg * num / den < 10, tpr_binomial_reference_95=binomial_interval(tp, npos),
                            fpr_binomial_reference_95=binomial_interval(fp, nneg))
    return result


def common_strata(main, randomized):
    """Targets whose partial history matches a unit of the main bank, of the random bank, of both or of neither."""
    left, right = main > 0, randomized > 0
    return dict(all=np.ones(len(left), dtype=bool), both=left & right, main_only=left & ~right, random_only=~left & right,
                neither=~left & ~right)


def analyze_group(entries, labels, mask, seed, rules, stratum):
    selected = mask & (labels >= 0)
    truth = labels[selected]
    population = dict(members=int(np.sum(truth == 1)), nonmembers=int(np.sum(truth == 0)),
                      calibration_nonmembers=int(np.sum(mask & (labels == -1))), subgroup_threshold_recalibrated=False)
    if set(truth) != {0, 1}:
        return population, [dict(**e['key'], status='insufficient_classes', metrics=None) for e in entries], [], None
    curves = [ExtendedCurve(e['scores'][selected], truth, {r: e['calibration'][r]['threshold'] for r, _, _ in RATES}) for e in entries]
    points = np.stack([c.vector(np.ones(len(truth))) for c in curves])
    samples = np.empty((rules['bootstrap_replicates'], len(curves), len(METRICS)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == label) for label in (0, 1)]
    for b in range(len(samples)):
        weights = np.zeros(len(truth), dtype=np.int64)
        for group in groups:
            weights += np.bincount(rng.choice(group, len(group), replace=True), minlength=len(truth))
        for i, curve in enumerate(curves):
            samples[b, i] = curve.vector(weights)
    results = []
    for i, e in enumerate(entries):
        results.append(dict(**e['key'], status='descriptive',
                            metrics={m: float(v) if np.isfinite(v) else None for m, v in zip(METRICS, points[i])},
                            intervals={m: interval(samples[:, i, j]) for j, m in enumerate(METRICS)},
                            undefined_draws={m: int(np.sum(~np.isfinite(samples[:, i, j]))) for j, m in enumerate(METRICS)
                                             if not np.isfinite(samples[:, i, j]).all()},
                            counts=counts_for(curves[i]), calibration=e['calibration']))
    lookup = {(e['key']['arm'], e['key']['method'], e['key']['view'], e['key']['attack']): i for i, e in enumerate(entries)}
    comparisons, family_rows, family_draws, family_points = [], [], [], []
    for view in VIEWS:
        for attack in ATTACKS:
            for name, left, right in COMPARISONS:
                i, j = lookup[(*left, view, attack)], lookup[(*right, view, attack)]
                delta = points[i] - points[j]
                draws = samples[:, i] - samples[:, j]
                record = dict(comparison=name, left=list(left), right=list(right), view=view, attack=attack,
                              differences={m: float(v) if np.isfinite(v) else None for m, v in zip(METRICS, delta)},
                              intervals={m: interval(draws[:, k]) for k, m in enumerate(METRICS)}, simultaneous={})
                comparisons.append(record)
                if stratum == 'all' and view == 'C+weight' and attack == 'unit_classifier':
                    for metric in FAMILY:
                        k = METRICS.index(metric)
                        family_rows.append((record, metric, curves[i].identical_function(curves[j], metric)))
                        family_draws.append(draws[:, k])
                        family_points.append(delta[k])
    adjusted = None
    if family_rows:
        alpha = rules['familywise_alpha'] / len(DOMAINS)
        values, estimates = np.array(family_draws).T, np.array(family_points)
        valid = np.isfinite(values).all(axis=0) & np.isfinite(estimates)
        sd = np.full(len(family_rows), np.nan)
        sd[valid] = np.std(values[:, valid], axis=0, ddof=1)
        active = valid & (sd > 0)
        critical = (float(np.quantile(np.max(np.abs((values[:, active] - estimates[active]) / sd[active]), axis=1), 1 - alpha, method='linear'))
                    if active.any() else None)
        unavailable = 0
        for k, (row, metric, identical) in enumerate(family_rows):
            if identical and valid[k]:
                require(abs(estimates[k]) < 1e-12, 'identical_metric_mismatch', 'Identical score functions have zero difference')
                item = dict(status='identical_on_saved_users', interval=[0., 0.])
            elif active[k]:
                item = dict(status='approximate_simultaneous_conditional',
                            interval=[float(estimates[k] - critical * sd[k]), float(estimates[k] + critical * sd[k])])
            else:
                item = dict(status='not_estimable', interval=None)
                unavailable += 1
            row['simultaneous'][metric] = item
        adjusted = dict(comparisons=len(COMPARISONS), metrics=list(FAMILY), family_size=len(family_rows), critical_value=critical,
                        per_domain_alpha=alpha, unavailable=unavailable)
    return population, results, comparisons, adjusted


def replay(scores, result):
    """Recompute the thresholds and metrics of the attack stage from the saved scores."""
    labels = scores['labels']
    test = labels >= 0
    for attack in ATTACKS:
        calibration = calibrate_threshold(scores[attack][labels == -1].tolist())
        require(calibration == result['attacks'][attack]['calibration'] and calibration['status'] == 'calibrated'
                and score_metrics(scores[attack][test][None], labels[test], [calibration['threshold']]) == result['attacks'][attack]['metrics'],
                'attack_replay_mismatch', 'Saved scores reproduce the thresholds and metrics of the attack stage')


def saved_scores(config):
    """Target scores of every bank, domain, procedure and view, checked against the attack stage."""
    report = config.finished('attacks', 'attacks_completed')
    by_key = {(r['domain'], r['arm'], r['method'], r['view']): r for r in report['results']}
    require(len(by_key) == len(report['results']) and set(by_key) == {(d, a, m, v) for d in DOMAINS for a in ARMS for m in METHODS for v in VIEWS},
            'incomplete_attacks', 'Every domain, bank, procedure and view was attacked')
    scores = {}
    for (domain, arm, method, view), row in by_key.items():
        values = target_scores(config, arm, domain, method, view)
        require(all(np.isfinite(v).all() for v in values.values()) and set(values['labels']) == {-1, 0, 1}
                and len(np.unique(values['positions'])) == len(values['labels']), 'invalid_scores', 'Scores cover every target once')
        replay(values, row)
        scores[domain, arm, method, view] = values
    for domain in DOMAINS:
        reference = scores[domain, 'main', 'DP', 'C+weight']
        for arm in ARMS:
            matches = scores[domain, arm, 'DP', 'C+weight']['known_matches']
            for method in METHODS:
                for view in VIEWS:
                    values = scores[domain, arm, method, view]
                    require(all(np.array_equal(values[k], reference[k]) for k in ('labels', 'positions'))
                            and np.array_equal(values['known_matches'], matches), 'unpaired_targets',
                            'Every procedure and view scores the same targets with the same partial histories')
    return report, by_key, scores


def attack_metrics(config, output):
    rules = config['attack_metrics']
    _, saved, scores = saved_scores(config)
    results, paired, populations, simultaneous = [], [], {}, {}
    for d, domain in enumerate(DOMAINS):
        labels = scores[domain, 'main', 'DP', 'C+weight']['labels']
        entries = []
        for arm in ARMS:
            for method in METHODS:
                for view in VIEWS:
                    for attack in ATTACKS:
                        values = scores[domain, arm, method, view][attack]
                        calibration = calibrate_points(values[labels == -1])
                        require(calibration['1pct']['threshold'] == saved[domain, arm, method, view]['attacks'][attack]['calibration']['threshold'],
                                'attack_replay_mismatch', 'The 1% operating point is that of the attack stage')
                        entries.append(dict(key=dict(arm=arm, method=method, view=view, attack=attack), scores=values, calibration=calibration))
        masks = common_strata(scores[domain, 'main', 'DP', 'C+weight']['known_matches'], scores[domain, 'random', 'DP', 'C+weight']['known_matches'])
        populations[domain] = {}
        for s, stratum in enumerate(('all', 'both')):
            facts, rows, pairs, adjusted = analyze_group(entries, labels, masks[stratum], rules['bootstrap_seed'] + 100 * d + s, rules, stratum)
            populations[domain][stratum] = facts
            results.extend(dict(domain=domain, stratum=stratum, **r) for r in rows)
            paired.extend(dict(domain=domain, stratum=stratum, **r) for r in pairs)
            if adjusted is not None:
                simultaneous[domain] = adjusted
    report = dict(status='attack_metrics_completed', rules=rules, metrics=list(METRICS), rates=[r for r, _, _ in RATES],
                  comparisons=[dict(name=n, left=list(a), right=list(b)) for n, a, b in COMPARISONS], simultaneous_family=list(FAMILY),
                  results=results, paired=paired, populations=populations, simultaneous=simultaneous)
    write_json(output / 'manifest.json', report)
    return report
