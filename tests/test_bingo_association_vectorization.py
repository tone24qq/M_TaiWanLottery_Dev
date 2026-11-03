"""Integer-count parity for association kernels against simple causal loops."""

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_associations import (
    _cooccurrence_counts,
    _lag_counts,
    _group_counts,
)


def scalar_cooccurrence(matrix):
    trials = np.zeros(80, dtype=np.int64)
    successes = np.zeros((80, 80), dtype=np.int64)
    for row in matrix:
        numbers = np.flatnonzero(row)
        for a in numbers:
            trials[a] += 1
            for b in numbers:
                if a != b:
                    successes[a, b] += 1
    return trials, successes


def scalar_lag(matrix, steps, lag, start=0, stop=None):
    trials = np.zeros(80, dtype=np.int64)
    successes = np.zeros((80, 80), dtype=np.int64)
    used = 0
    for t in range(
        max(lag, start), min(len(matrix), len(matrix) if stop is None else stop)
    ):
        if not all(steps[t - lag : t]):
            continue
        used += 1
        for a in np.flatnonzero(matrix[t - lag]):
            trials[a] += 1
            for b in np.flatnonzero(matrix[t]):
                successes[a, b] += 1
    return trials, successes, used


def scalar_groups(matrix, steps, ants, cons, lag, start, stop):
    trials = np.zeros(len(ants), dtype=np.int64)
    successes = np.zeros((len(ants), len(cons)), dtype=np.int64)
    for t in range(max(lag, start), min(len(matrix), stop)):
        if not all(steps[t - lag : t]):
            continue
        for a, ant in enumerate(ants):
            if not all(matrix[t - lag, n] for n in ant):
                continue
            trials[a] += 1
            for b, con in enumerate(cons):
                if all(matrix[t, n] for n in con):
                    successes[a, b] += 1
    return trials, successes


def fixture(n=55):
    rng = np.random.default_rng(1984)
    matrix = np.zeros((n, 80))
    for row in matrix:
        row[rng.choice(80, 20, replace=False)] = 1
    steps = np.ones(max(0, n - 1), dtype=bool)
    if n > 20:
        steps[[4, 18]] = False
    return matrix, steps


@pytest.mark.parametrize("lag", range(1, 21))
def test_all_lags_and_disjoint_target_boundaries_equal_scalar_counts(lag):
    matrix, steps = fixture()
    for start, stop in ((0, 55), (17, 39), (39, 55)):
        actual = _lag_counts(matrix, steps, lag, start, stop)
        expected = scalar_lag(matrix, steps, lag, start, stop)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_cooccurrence_and_pair_group_counts_equal_scalar_without_uint8_overflow():
    matrix, steps = fixture()
    for a, b in zip(_cooccurrence_counts(matrix), scalar_cooccurrence(matrix)):
        np.testing.assert_array_equal(a, b)
    ants = [(0, 1), (2, 3), (40, 41)]
    cons = [(0, 2), (6, 7, 8), (21, 22)]
    for lag in range(2, 6):
        actual = _group_counts(matrix, steps, ants, cons, lag, 17, 54)
        expected = scalar_groups(matrix, steps, ants, cons, lag, 17, 54)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    repeated = np.zeros((1000, 80), dtype=np.uint8)
    repeated[:, :20] = 1
    trials, successes = _cooccurrence_counts(repeated)
    assert trials[0] == 1000 and successes[0, 1] == 1000 and successes[0, 0] == 0
    trials, successes, n = _lag_counts(repeated, np.ones(999, dtype=bool), 1)
    assert n == 999 and trials[0] == 999 and successes[0, 1] == 999


def test_future_rows_and_gaps_do_not_change_past_segment_counts():
    matrix, steps = fixture()
    changed = matrix.copy()
    changed[30:] = 1 - changed[30:]
    for lag in range(1, 21):
        before = _lag_counts(matrix, steps, lag, 17, 30)
        after = _lag_counts(changed, steps, lag, 17, 30)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
    no_gaps = np.ones(len(matrix) - 1, dtype=bool)
    assert _lag_counts(matrix, steps, 5)[2] < _lag_counts(matrix, no_gaps, 5)[2]
    empty = np.zeros((0, 80))
    assert _lag_counts(empty, np.zeros(0, dtype=bool), 20)[2] == 0
