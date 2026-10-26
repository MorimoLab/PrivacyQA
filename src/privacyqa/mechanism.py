"""Contribution bounding, the OpenDP discrete Laplace mechanism, the release threshold and the privacy ledger.

A user who supports k units contributes floor(B*Q / max(B, k)) to each of them, so every user's contribution has l1
norm at most B*Q. Discrete Laplace noise of scale B*Q/epsilon on every coordinate of the fixed bank then gives pure
epsilon-DP for adding or removing one user.
"""
from fractions import Fraction
import importlib.metadata
import math
import sqlite3
import time

from privacyqa.common import require

B = 20
Q = 2**20
EPSILON = 4
SCALE = B * Q // EPSILON
I64_MAX = 2**63 - 1
OPENDP = '0.15.1'


def bounded(support):
    """Integer contributions of one user's binary support vector (numpy int64)."""
    return support * (B * Q // max(B, int(support.sum())))


def laplace_measurement():
    require(importlib.metadata.version('opendp') == OPENDP, 'unreviewed_backend', f'Use OpenDP {OPENDP}')
    import opendp.prelude as dp
    dp.enable_features('contrib')
    measurement = dp.m.make_laplace(dp.vector_domain(dp.atom_domain(T='i64')), dp.l1_distance(T='i64'), scale=B * Q / EPSILON)
    privacy_map = measurement.map(B * Q)
    require(measurement.check(B * Q, float(EPSILON)) and privacy_map <= EPSILON, 'privacy_map_failed',
            'The privacy map of the backend must not exceed epsilon')
    return measurement, dict(backend='OpenDP', backend_version=OPENDP, mechanism='integer_vector_laplace',
                             adjacency='add_remove_one_user', B=B, Q=Q, l1_sensitivity=B * Q, scale=SCALE,
                             epsilon=EPSILON, delta=0, privacy_map=privacy_map)


def add_noise(integer_mass):
    """One measurement: the noisy vector of a complete bounded mass vector. The noise is never saved."""
    require(integer_mass and all(type(v) is int and 0 <= v <= I64_MAX for v in integer_mass), 'invalid_mass',
            'Measure a nonempty vector of nonnegative i64 integers')
    measurement, manifest = laplace_measurement()
    noisy = measurement(list(integer_mass))
    require(len(noisy) == len(integer_mass) and all(type(v) is int for v in noisy), 'invalid_backend_result',
            'The backend returns the complete integer vector')
    return noisy, manifest


def threshold(dimensions, familywise_alpha):
    """Release threshold: with probability at least 1 - alpha, the noise of every coordinate lies within it.

    The union bound gives each of the `dimensions` coordinates the level alpha/dimensions; OpenDP returns the two-sided
    discrete-Laplace tolerance at that level, and the threshold rounds it up.
    """
    require(importlib.metadata.version('opendp') == OPENDP, 'unreviewed_backend', f'Use OpenDP {OPENDP}')
    from opendp.accuracy import discrete_laplacian_scale_to_accuracy
    exact = Fraction(familywise_alpha) / dimensions
    alpha = float(exact)
    if Fraction(alpha) > exact:
        alpha = math.nextafter(alpha, 0.0)
    require(0 < alpha < 1, 'invalid_alpha', 'The per-coordinate level is in (0, 1)')
    accuracy = discrete_laplacian_scale_to_accuracy(float(SCALE), alpha)
    require(math.isfinite(accuracy) and accuracy >= 0, 'invalid_accuracy', 'The backend returns a finite tolerance')
    return dict(dimensions=dimensions, familywise_alpha=str(familywise_alpha), per_coordinate_alpha=str(exact),
                integer_scale=SCALE, threshold_integer=math.ceil(accuracy))


class Ledger:
    """Every planned measurement is reserved before the first one; a failed measurement is never retried or refunded."""

    def __init__(self, path, releases):
        require(not path.exists(), 'ledger_exists', 'Each run starts a new ledger')
        self.conn = sqlite3.connect(path)
        path.chmod(0o600)
        with self.conn:
            self.conn.execute('CREATE TABLE measurements(release_id TEXT PRIMARY KEY, domain TEXT, epsilon REAL, delta REAL, '
                              'status TEXT, updated REAL)')
            self.conn.executemany('INSERT INTO measurements VALUES(?,?,?,?,?,?)',
                                  [(r, d, EPSILON, 0, 'reserved', time.time()) for r, d in releases])

    def mark(self, release_id, status):
        require(status in ('measuring', 'complete', 'failed'), 'invalid_ledger_status', 'Unknown measurement state')
        with self.conn:
            changed = self.conn.execute('UPDATE measurements SET status=?, updated=? WHERE release_id=?',
                                        (status, time.time(), release_id)).rowcount
        require(changed == 1, 'unreserved_measurement', 'Reserve every measurement first')

    def totals(self):
        rows = self.conn.execute('SELECT status, count(*), sum(epsilon) FROM measurements GROUP BY status').fetchall()
        return dict(measurements=sum(r[1] for r in rows), statuses={r[0]: r[1] for r in rows},
                    sequential_epsilon=sum(r[2] for r in rows), delta=0)

    def close(self):
        self.conn.close()
