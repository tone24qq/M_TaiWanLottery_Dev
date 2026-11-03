"""Pure acceleration parity against PR #1 (eb7ec1c) mathematical definitions.

The reference functions deliberately retain the old full-tensor/loop path. Test
fixtures never enter the live source. Actual all-eight scorers share one fitted
artifact so this checks inference semantics, not a lucky re-training result.
"""

import math
import pickle

import numpy as np
import pytest
from scipy.stats import rankdata

from taiwan_lottery.ml.bingo_models import (
    FEATURE_NAMES,
    MODEL_NAMES,
    WINDOWS,
    FeatureHistory,
    ModelSuite,
    _features_from_matrix,
    draw_matrix,
    features,
    latest_features,
    normalize_scores as model_normalize,
)
from taiwan_lottery.ml.bingo_evaluation import (
    METHODS,
    STARS,
    _performance_summary,
    _rank,
    fuse,
    fuse_many,
    normalize_scores,
)


def history(size, seed=128):
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(80, 20, replace=False).tolist()) for _ in range(size)]


def source(size, seed=128):
    return [[n + 1 for n in row] for row in history(size, seed)]


def reference_scores(suite, history):
    """PR #1 score dispatch using the same already fitted model objects."""
    history = history[-suite.window :]
    matrix = draw_matrix(history)
    row = _features_from_matrix(matrix)[-1]
    result = {}
    for name in suite.enabled:
        if name not in suite.models or name == "ensemble":
            continue
        result[name] = model_normalize(
            suite._score_one(name, history, matrix, row)
        ).tolist()
    if "ensemble" in suite.models and len(result) >= 2:
        result["ensemble"] = model_normalize(
            np.mean(list(result.values()), axis=0)
        ).tolist()
    return result


def reference_fuse(scores, method, stars, performance=None):
    valid = {name: normalize_scores(vector) for name, vector in scores.items()}
    names = sorted(valid)
    matrix = np.asarray([valid[name] for name in names])
    weights = np.ones(len(names), dtype=float)
    summary = _performance_summary(performance, stars)
    if method == "shrink" and all(summary.get(n, (0, 0))[0] >= 30 for n in names):
        null_sd = math.sqrt(stars * 0.25 * 0.75 * (80 - stars) / 79)
        for i, name in enumerate(names):
            n, mean_hits = summary[name]
            effect = np.clip((mean_hits - stars * 0.25) / null_sd, -0.5, 0.5)
            weights[i] = 1 + effect * n / (n + 100)
    weights /= weights.sum()
    support = np.zeros(80, dtype=int)
    for row in matrix:
        support[np.asarray(_rank(row)[:stars]) - 1] += 1
    average = weights @ matrix
    if method == "majority":
        combined = (support + average / (len(names) + 1)) / (len(names) + 1)
    elif method == "borda":
        points = np.asarray(
            [(rankdata(row, method="average") - 1) / 79 for row in matrix]
        )
        combined = points.mean(axis=0)
    else:
        combined = average
    return dict(
        scores=combined.tolist(),
        ranking=_rank(combined),
        support={str(n + 1): int(v) for n, v in enumerate(support)},
        weights={n: float(w) for n, w in zip(names, weights)},
    )


@pytest.mark.parametrize("size", [0, 1, 5, 10, 17, 20, 60, 200, 300, 1000, 10000])
def test_next_row_exactly_matches_original_tensor(size):
    draws = source(size)
    matrix = draw_matrix(draws)
    expected = _features_from_matrix(matrix)[-1]
    np.testing.assert_array_equal(latest_features(matrix), expected)
    np.testing.assert_array_equal(features(draws), expected)


def test_shared_prefix_respects_every_window_boundary_and_target_causality():
    draws = source(1100)
    prepared = FeatureHistory(draws)
    changed = FeatureHistory(draws[:525] + [list(range(1, 21))] * 575)
    for target in (0, 1, 5, 17, 20, 30, 200, 525, 1000, 1100):
        for window in (5, 10, 17, 20, 30, 50, 60, 100, 200, 300, 500, 1000):
            expected = features(draws[max(0, target - window) : target])
            np.testing.assert_array_equal(prepared.row(target, window), expected)
            np.testing.assert_array_equal(
                prepared.matrix_before(target, window),
                draw_matrix(draws[max(0, target - window) : target]),
            )
            if target <= 525:
                np.testing.assert_array_equal(
                    prepared.row(target, window), changed.row(target, window)
                )
    assert not prepared.matrix_before(100).flags.writeable
    with pytest.raises(ValueError):
        prepared.row(-1)
    with pytest.raises(ValueError):
        prepared.row(1101)
    with pytest.raises(ValueError):
        prepared.row(100, 0)


def test_clipped_history_does_not_reuse_appearance_before_window():
    draws = [list(range(1, 21))] * 250 + [list(range(21, 41))] * 3
    cached = FeatureHistory(draws)
    row = cached.row(253, 3)
    assert row[0, 4] == np.float32(3 / 20)
    np.testing.assert_array_equal(row, features(draws[-3:]))


@pytest.mark.parametrize("ties", [False, True])
def test_all_star_votes_and_full_ranks_identical_with_ties_and_realistic_weights(ties):
    rng = np.random.default_rng(77)
    vectors = rng.normal(size=(8, 80))
    if ties:
        vectors = np.round(vectors)
    scores = {name: vector.tolist() for name, vector in zip(MODEL_NAMES, vectors)}
    scores["constant"] = [3] * 80
    performance = [
        dict(model=name, stars=k, samples=100 + i, mean_hits=k * 0.25 + (i - 3) * 0.02)
        for k in STARS
        for i, name in enumerate(scores)
    ]
    all_votes = fuse_many(scores, performance)
    for stars in STARS:
        for method in METHODS:
            expected = reference_fuse(scores, method, stars, performance)
            actual = all_votes[str(stars)][method]
            assert actual == fuse(scores, method, stars, performance)
            for key in expected:
                if key == "scores":
                    np.testing.assert_array_equal(actual[key], expected[key])
                else:
                    assert actual[key] == expected[key]
            assert actual["ranking"][:stars] == expected["ranking"][:stars]


def test_bad_scorer_isolated_once_across_all_votes():
    scores = {"good": list(range(80)), "broken": [float("nan")] * 80}
    result = fuse_many(scores)
    assert all(
        v["excluded"] == {"broken": "模型必須提供 80 個有限分數"}
        for methods in result.values()
        for v in methods.values()
    )


def test_all_eight_same_fitted_artifact_keeps_scores_and_selections():
    draws = source(120)
    suite = ModelSuite(draws, threads=1).fit()
    assert set(suite.scores(draws)) == set(MODEL_NAMES)
    suite = pickle.loads(pickle.dumps(suite))
    for count in (5, 17, 60, 120):
        current = draws[-count:]
        actual = suite.scores(current)
        expected = reference_scores(suite, current)
        assert actual == expected
        cached = FeatureHistory(current)
        assert actual == suite.scores_prepared(cached.matrix_before(), cached.row())
        for stars in STARS:
            for method in METHODS:
                assert (
                    fuse(actual, method, stars)["ranking"]
                    == reference_fuse(expected, method, stars)["ranking"]
                )


def test_real_walk_forward_uses_bounded_feature_blocks_without_future_labels():
    from taiwan_lottery.ml.bingo_evaluation import walk_forward
    from datetime import datetime, timedelta

    draws = [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=datetime(2026, 1, 1) + timedelta(minutes=5 * i),
            numbers=row,
        )
        for i, row in enumerate(source(360))
    ]
    report = walk_forward(draws, count=260, model_names=["frequency", "cdm"])
    assert report["count"] == 260
    # Crossing two block boundaries must preserve the original scalar baseline.
    for index in (0, 127, 128, 255, 256, 259):
        cutoff = 100 + index
        old = reference_scores(
            ModelSuite(
                [d["numbers"] for d in draws[:cutoff]], enabled=["frequency", "cdm"]
            ).fit(),
            [d["numbers"] for d in draws[:cutoff]],
        )
        for model in ("frequency", "cdm"):
            ranking = _rank(normalize_scores(old[model]))
            assert report["folds"][index]["predictions"][model]["5"] == ranking[:5]
