"""Synthetic audit of the implemented mechanism: canaries on synthetic integer vectors; no user data are read.

A canary supports k units with the bounded contribution of a user who supports k units. Each trial releases the
background with or without the canary, at the main-bank threshold. Half of the trials select a test statistic and a
threshold; the other half bound the privacy loss from below with one-sided 97.5% Clopper-Pearson limits for the event
rates under both hypotheses (overall 95%). Two controls check the audit: without noise the loss is unbounded, and at
half the noise scale it is 2 epsilon.
"""
from multiprocessing import get_context

import numpy as np
from scipy.stats import beta

from privacyqa.attacks.audit import discrete_laplace_logcdf, discrete_laplace_logsf
from privacyqa.banks import load_banks
from privacyqa.common import DOMAINS, require, write_json
from privacyqa.mechanism import B, EPSILON, Q, SCALE, add_noise, threshold


def half_scale_measurement():
    """Deliberately misconfigured control: half the noise scale, so the true privacy loss is 2 epsilon."""
    import opendp.prelude as dp
    dp.enable_features('contrib')
    return dp.m.make_laplace(dp.vector_domain(dp.atom_domain(T='i64')), dp.l1_distance(T='i64'), scale=SCALE / 2)


def statistics(noisy, contribution, background, cutoff, scale=SCALE):
    """The auditor's statistics at the claimed scale: on the unthresholded vector and on the C-only release events."""
    outside = noisy - background
    full_view = ((np.abs(outside) - np.abs(outside - contribution)) / scale).sum(axis=1)
    margin = cutoff - background
    plain = np.where(noisy > cutoff, discrete_laplace_logsf(margin - contribution, scale) - discrete_laplace_logsf(margin, scale),
                     discrete_laplace_logcdf(margin - contribution, scale) - discrete_laplace_logcdf(margin, scale))
    return full_view, plain.sum(axis=1)


def trial_chunk(task):
    mechanism, contribution, background, cutoff, trials, member = task
    contribution, background = np.asarray(contribution, dtype=np.int64), np.asarray(background, dtype=np.int64)
    vector = np.tile(background + contribution * int(member), trials).tolist()
    if mechanism == 'opendp':
        noisy, manifest = add_noise(vector)
        require(manifest['scale'] == SCALE and manifest['epsilon'] == EPSILON, 'mechanism_mismatch', 'Audit the implemented mechanism')
    elif mechanism == 'half_scale':
        noisy = half_scale_measurement()(vector)
    else:
        noisy = vector
    noisy = np.asarray(noisy, dtype=np.int64).reshape(trials, len(contribution))
    return statistics(noisy, contribution, background, cutoff)


def epsilon_bounds(positive, negative, cutoffs):
    """Clopper-Pearson lower bounds of ln(P_pos(event) / P_neg(event)) for the events {statistic > t}."""
    positive, negative = np.sort(positive), np.sort(negative)
    k1 = len(positive) - np.searchsorted(positive, cutoffs, side='right')
    k0 = len(negative) - np.searchsorted(negative, cutoffs, side='right')
    lower = np.where(k1 > 0, beta.ppf(.025, np.maximum(k1, 1), len(positive) - k1 + 1), 0.)
    upper = np.where(k0 < len(negative), beta.ppf(.975, k0 + 1, np.maximum(len(negative) - k0, 1)), 1.)
    with np.errstate(divide='ignore'):
        return np.where(lower > 0, np.log(lower) - np.log(upper), -np.inf), k1, k0


def epsilon_lower_bound(members, nonmembers, grid):
    """Select a direction and threshold on one half of the trials; bound epsilon on the untouched half."""
    half_m, half_n = len(members) // 2, len(nonmembers) // 2
    selection = np.r_[members[:half_m], nonmembers[:half_n]]
    tails = np.geomspace(1e-7, .5, grid)
    cutoffs = np.unique(np.quantile(selection, np.r_[tails, 1 - tails]))
    choices = []
    for direction, sign in (('member_high', 1), ('nonmember_low', -1)):
        pos, neg = (members, nonmembers) if sign == 1 else (-nonmembers, -members)
        half_pos, half_neg = (half_m, half_n) if sign == 1 else (half_n, half_m)
        cuts = cutoffs if sign == 1 else -cutoffs
        estimate, _, _ = epsilon_bounds(pos[:half_pos], neg[:half_neg], cuts)
        best = int(np.argmax(estimate))
        choices.append((estimate[best], direction, sign, cuts[best], pos[half_pos:], neg[half_neg:]))
    estimate, direction, sign, cut, pos, neg = max(choices, key=lambda c: c[0])
    bound, k1, k0 = epsilon_bounds(pos, neg, np.array([cut]))
    tpr, fpr = k1[0] / len(pos), k0[0] / len(neg)
    # Without negative events the point ratio is unbounded; only the lower bound is reported then.
    point = float(np.log(tpr) - np.log(fpr)) if tpr > 0 and fpr > 0 else None
    return dict(direction=direction, threshold=float(sign * cut), selection_estimate=float(estimate) if np.isfinite(estimate) else None,
                evaluation_positive=len(pos), evaluation_negative=len(neg), event_positive=int(k1[0]), event_negative=int(k0[0]),
                event_rate_positive=float(tpr), event_rate_negative=float(fpr), epsilon_point=point, epsilon_lower_95=max(0., float(bound[0])))


def mechanism_audit(config, output):
    rules = config['mechanism_audit']
    require(set(rules['trials']) == {str(k) for k in rules['canary_units']}, 'invalid_audit_plan', 'Plan the trials of every canary')
    banks = load_banks(config, arms=('main',))
    dimensions = {d: len(banks['main'][d].units) for d in DOMAINS}
    cutoff = threshold(sum(dimensions.values()), config['release']['familywise_alpha'])['threshold_integer']
    designs = [(f'canary_{k}', 'opendp', k, rules['trials'][str(k)]) for k in rules['canary_units']]
    designs += [('control_noise_free', 'noise_free', B, rules['control_trials']), ('control_half_scale', 'half_scale', B, rules['control_trials'])]
    tasks, layout = [], []
    for name, mechanism, units, trials in designs:
        contribution = [B * Q // max(B, units)] * units
        background = [cutoff - x // 2 for x in contribution]  # Symmetric about the release threshold.
        for member in (True, False):
            remaining = trials
            while remaining:
                size = min(rules['chunk_trials'], remaining)
                tasks.append((mechanism, contribution, background, cutoff, size, member))
                layout.append((name, member))
                remaining -= size
    with get_context('spawn').Pool(rules['workers']) as pool:
        outputs = pool.map(trial_chunk, tasks, chunksize=1)
    results = []
    for name, mechanism, units, trials in designs:
        claimed = units * (B * Q // max(B, units)) / SCALE
        for s, statistic in enumerate(('unthresholded', 'C-only')):
            pick = lambda member: np.concatenate([o[s] for o, (n, m) in zip(outputs, layout) if n == name and m == member])
            members, nonmembers = pick(True), pick(False)
            require(len(members) == len(nonmembers) == trials, 'trial_mismatch', 'Every planned trial is evaluated once')
            results.append(dict(design=name, mechanism=mechanism, canary_units=units, contribution_l1=units * (B * Q // max(B, units)),
                                claimed_epsilon=claimed, true_epsilon=2 * claimed if mechanism == 'half_scale' else (None if mechanism == 'noise_free' else claimed),
                                statistic=statistic, trials_per_hypothesis=trials, **epsilon_lower_bound(members, nonmembers, rules['threshold_grid'])))
    report = dict(status='mechanism_audit_completed', rules=rules, dimensions=dimensions, threshold_integer=cutoff, scale=SCALE, results=results,
                  mechanism_invocations=sum(t[0] == 'opendp' for t in tasks), user_data_read=False)
    write_json(output / 'manifest.json', report)
    return report
