"""What an attacker observes, and the statistics shared by all attacks.

Features come from one published view and the target's partial history only: never from membership labels, complete
histories or supports.
"""
import math
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from privacyqa.common import require

ATTACKS = ('aggregate_classifier', 'unit_classifier', 'matching')
MIN_CALIBRATION = 1000  # Ten expected nonmembers above the 1% threshold.


class KnownIndex:
    """Public catalog rarity of every unit: log(1 + catalog items / complete answers of the core)."""

    def __init__(self, bank):
        self.bank = bank
        self.rarity = np.array([math.log1p(len(bank.catalog.items) / max(1, len(a))) for a in bank.core_answers])


def validate_view(bank, rows, release_id, weighted=False):
    """Units of a view that is exactly a release of this bank: whole units, and in C+weight normalized equal weights."""
    expected = {r['id']: r for r in bank.reconstruct([1] * len(bank.units), release_id)['C-only']}
    seen = set()
    for row in rows:
        plain = {k: v for k, v in row.items() if k != 'weight'}
        require(row['id'] not in seen and plain == expected.get(row['id']) and (not weighted or math.isfinite(row['weight'])),
                'view_mismatch', 'The view consists of rows of this bank with one release identity')
        seen.add(row['id'])
    require(all(not (seen & set(unit)) or set(unit) <= seen for unit in bank.units), 'partial_unit',
            'A view contains every variant of each released unit')
    if weighted:
        require(not rows or math.isclose(math.fsum(r['weight'] for r in rows), 1, rel_tol=0, abs_tol=1e-12),
                'unnormalized_view', 'C+weight weights sum to one')
        weights = {r['id']: r['weight'] for r in rows}
        require(all(len({weights[k] for k in unit if k in weights}) <= 1 for unit in bank.units), 'unequal_unit_weights',
                'Variants of a unit share its weight')
    return seen


class PublishedView:
    """One validated view, scored against the partial histories of many targets."""

    def __init__(self, index, rows, release_id, view):
        require(view in ('C-only', 'C+weight'), 'invalid_view', 'Use C-only or C+weight')
        bank = index.bank
        released = validate_view(bank, rows, release_id, view == 'C+weight')
        self.selected = np.array([bool(released & set(unit)) for unit in bank.units], dtype=np.float64)
        self.rarity, self.empty = index.rarity, not rows
        self.names = ['known_observations', 'known_matching_units', 'released_matching_units', 'match_fraction',
                      'catalog_rarity_match_fraction', 'empty_release']
        self.unit_weight = None
        if view == 'C+weight':
            weights = {r['id']: r['weight'] for r in rows}
            self.unit_weight = np.array([math.fsum(weights.get(k, 0) for k in unit) for unit in bank.units])
            self.names.append('matched_weight')

    def features(self, known):
        size = len(known.observations)
        counts = np.bincount(known.rows, minlength=size)
        matches = np.bincount(known.rows, weights=self.selected[known.units], minlength=size)
        denominator = np.bincount(known.rows, weights=self.rarity[known.units], minlength=size)
        numerator = np.bincount(known.rows, weights=self.rarity[known.units] * self.selected[known.units], minlength=size)
        features = [known.observations, counts, matches, matches / np.maximum(counts, 1),
                    np.divide(numerator, denominator, out=np.zeros(size), where=denominator > 0), np.full(size, self.empty)]
        if self.unit_weight is not None:
            features.append(np.bincount(known.rows, weights=self.unit_weight[known.units], minlength=size))
        return np.column_stack(features).astype(np.float64), self.names


def unit_features(published, known):
    """Aggregate features, then indicators of known units, of known released units and (C+weight) their weights."""
    aggregate, names = published.features(known)
    shape = len(known.observations), len(published.selected)
    mask = sparse.csr_matrix((np.ones(len(known.rows)), (known.rows, known.units)), shape=shape)
    blocks = [sparse.csr_matrix(aggregate), mask, mask.multiply(published.selected)]
    if published.unit_weight is not None:
        blocks.append(mask.multiply(published.unit_weight))
    result = sparse.hstack(blocks, format='csr')
    result.eliminate_zeros()
    return result, names


def fit_classifier(x, y, plan):
    """L2-regularized logistic regression on standardized features (liblinear)."""
    require(set(y) == {0, 1} and np.isfinite(x.data).all(), 'invalid_attack_training', 'Fit finite features with both labels')
    scaler = StandardScaler(with_mean=False).fit(x)
    model = LogisticRegression(solver='liblinear', penalty='l2', C=plan['classifier_C'], max_iter=plan['max_iterations'],
                               tol=plan['tolerance'], random_state=plan['seed'])
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    return dict(solver='liblinear', scale=scaler.scale_.tolist(), coefficient=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]), iterations=int(model.n_iter_[0]), features=x.shape[1], training_rows=x.shape[0])


def classifier_scores(model, x):
    return np.asarray(x @ (np.asarray(model['coefficient']) / np.asarray(model['scale']))).ravel() + model['intercept']


def score_metrics(scores, labels, thresholds, target_weights=None):
    """AUC with tied scores at half credit, and the rates at fixed thresholds (strict >), averaged over releases."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    require(scores.ndim == 2 and scores.shape[1] == len(labels) and len(thresholds) == len(scores) and set(labels) <= {0, 1}
            and np.isfinite(scores).all(), 'invalid_attack_scores', 'One finite score per target and release')
    weights = np.ones(len(labels)) if target_weights is None else np.asarray(target_weights)
    positive, negative = weights * (labels == 1), weights * (labels == 0)
    npos, nneg = positive.sum(), negative.sum()
    if not npos or not nneg:
        return dict(auc=None, actual_fpr=None, tpr_at_frozen_threshold=None, advantage=None)
    auc, fpr, tpr = [], [], []
    for row, cutoff in zip(scores, thresholds):
        _, groups = np.unique(row, return_inverse=True)
        p, n = np.bincount(groups, weights=positive), np.bincount(groups, weights=negative)
        auc.append(float(np.dot(p, np.cumsum(n) - n / 2) / (npos * nneg)))
        fpr.append(float(np.dot(negative, row > cutoff) / nneg) if cutoff is not None else None)
        tpr.append(float(np.dot(positive, row > cutoff) / npos) if cutoff is not None else None)
    average = lambda values: float(np.average(values, weights=np.ones(len(scores))))
    calibrated = all(t is not None for t in thresholds)
    result = dict(auc=average(auc), actual_fpr=average(fpr) if calibrated else None,
                  tpr_at_frozen_threshold=average(tpr) if calibrated else None)
    result['advantage'] = result['tpr_at_frozen_threshold'] - result['actual_fpr'] if calibrated else None
    return result


def calibrate_threshold(nonmember_scores):
    """Threshold for a 1% false-positive rate, set on calibration nonmembers only."""
    n = len(nonmember_scores)
    if n < MIN_CALIBRATION:
        return dict(status='insufficient_calibration_nonmembers', threshold=None, comparator='>', target_fpr=0.01,
                    calibration_nonmembers=n, minimum_nonmembers=MIN_CALIBRATION)
    ordered = sorted(nonmember_scores)
    cutoff = ordered[n - n // 100 - 1]
    return dict(status='calibrated', threshold=cutoff, comparator='>', target_fpr=0.01, calibration_nonmembers=n,
                calibration_fpr=sum(s > cutoff for s in ordered) / n)
