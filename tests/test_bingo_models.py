"""Synthetic fixtures exercise causality/shape; they are never source draws."""

import pickle

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_models import (
    ModelSuite,
    draw_matrix,
    features,
    normalize_scores,
    supervised_samples,
)


def history(size=120):
    rng = np.random.default_rng(23)
    return [
        sorted(rng.choice(np.arange(1, 81), size=20, replace=False).tolist())
        for _ in range(size)
    ]


def test_supervised_features_never_use_target_or_future():
    draws = history()
    x, y = supervised_samples(draws)
    for target in (20, 21, 60, 119):
        np.testing.assert_allclose(x[target - 20], features(draws[:target]))
        np.testing.assert_array_equal(y[target - 20], draw_matrix([draws[target]])[0])
    changed = draws[:60] + [list(range(1, 21))] * 60
    changed_x, _ = supervised_samples(changed)
    np.testing.assert_array_equal(x[:41], changed_x[:41])


def test_scores_standardize_units_and_preserve_ties():
    values = np.repeat(np.arange(20), 4)
    np.testing.assert_array_equal(
        normalize_scores(values), normalize_scores(values * 987 + 100)
    )
    np.testing.assert_array_equal(normalize_scores(np.ones(80)), np.full(80, 0.5))
    with pytest.raises(ValueError):
        normalize_scores(np.full(80, np.nan))


def test_models_fail_closed_on_invalid_draws():
    for bad in ([1] * 20, list(range(20)), list(range(1, 22))):
        with pytest.raises(ValueError):
            ModelSuite([bad])


def test_baselines_deterministic_persistable_and_full_scores():
    draws = history()
    progress = []
    suite = ModelSuite(draws, enabled=["frequency", "cdm", "ensemble"]).fit(
        lambda *record: progress.append(record)
    )
    scores = suite.scores(draws)
    assert set(scores) == {"frequency", "cdm", "ensemble"}
    assert all(
        len(value) == 80 and np.isfinite(value).all() for value in scores.values()
    )
    assert (
        scores == suite.scores(draws) == pickle.loads(pickle.dumps(suite)).scores(draws)
    )
    assert any(record[-1] == "ready" for record in progress)
    assert all(
        record["status"] == "ready"
        for record in suite.statuses()
        if record["name"] in scores
    )


def test_insufficient_data_does_not_invent_predictions():
    suite = ModelSuite(history(5)).fit()
    assert suite.scores(history(5)) == {}
    assert all(record["status"] == "waiting" for record in suite.statuses())


def test_online_duplicate_is_idempotent():
    pytest.importorskip("river")
    draws = history(60)
    suite = ModelSuite(draws[:59], enabled=["river"]).fit()
    suite.update(draws[:59], draws[59])
    before = pickle.dumps(suite)
    suite.update(draws[:59], draws[59])
    assert pickle.dumps(suite) == before
    assert len(suite.scores(draws)["river"]) == 80


def test_optional_model_failure_is_isolated(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fail_lightgbm(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("unavailable", name="lightgbm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_lightgbm)
    draws = history()
    suite = ModelSuite(draws, enabled=["frequency", "lightgbm"]).fit()
    assert set(suite.scores(draws)) == {"frequency"}
    assert (
        next(record for record in suite.statuses() if record["name"] == "lightgbm")[
            "status"
        ]
        == "disabled"
    )


def test_installed_models_really_fit_score_and_restore():
    """Exercise actual optional learners, never mocked model-name placeholders."""
    import importlib.util

    enabled = ["frequency", "cdm"]
    for name in ("river", "lightgbm", "catboost"):
        if importlib.util.find_spec(name):
            enabled.append(name)
    if importlib.util.find_spec("torch"):
        enabled.extend(["transformer", "dqn"])
    enabled.append("ensemble")
    draws = history(120)
    suite = ModelSuite(draws, threads=1, enabled=enabled).fit()
    scores = suite.scores(draws)
    statuses = {record["name"]: record for record in suite.statuses()}
    assert set(scores) == set(enabled), {name: statuses[name] for name in enabled}
    for name in enabled:
        assert statuses[name]["samples"] > 0
        assert statuses[name]["status"] == "ready"
        assert np.ptp(scores[name]) > 0
    assert pickle.loads(pickle.dumps(suite)).scores(draws) == scores


def test_transformer_input_does_not_observe_target_or_future():
    pytest.importorskip("torch")
    from taiwan_lottery.ml.bingo_neural import training_sequences

    draws = history(100)
    matrix = draw_matrix(draws)
    sequences = training_sequences(matrix)
    target = 60
    changed = matrix.copy()
    changed[target:] = 1 - changed[target:]
    changed_sequences = training_sequences(changed)
    # Includes the prediction for target 60; changing answer 60 and every later
    # answer must leave all inputs for predictions <=60 unchanged.
    np.testing.assert_array_equal(
        sequences[: target - 20 + 1], changed_sequences[: target - 20 + 1]
    )
    assert not np.array_equal(
        sequences[target - 20 + 1], changed_sequences[target - 20 + 1]
    )


def test_explicit_ensemble_selection_is_order_independent():
    draws = history(60)
    suite = ModelSuite(
        draws, enabled=["ensemble", "frequency", "cdm", "frequency"]
    ).fit()
    assert suite.enabled == ["frequency", "cdm", "ensemble"]
    assert set(suite.scores(draws)) == {"frequency", "cdm", "ensemble"}
