"""Real score and audit checks for shared adaptive context inference."""

from datetime import datetime, timedelta
import copy

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_models import ModelSuite
from taiwan_lottery.ml.bingo_contexts import context_scores, context_audit


def snapshot(count=150):
    rng = np.random.default_rng(311)
    return [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=datetime(2026, 11, 1, 7, 5) + timedelta(minutes=5 * i),
            numbers=sorted(map(int, rng.choice(np.arange(1, 81), 20, replace=False))),
        )
        for i in range(count)
    ]


def test_fixed_scores_unchanged_and_star_contexts_are_real_causal_scores():
    data = snapshot()
    history = [r["numbers"] for r in data]
    suite = ModelSuite(history, enabled=["frequency", "cdm"]).fit()
    original = suite.scores(history)
    plan = {
        "champions": {"3": {"frequency": 17}, "4": {"cdm": 60}},
        "shadow_windows": [5, 17, 1000],
    }
    result = context_scores(suite, data, plan)
    assert result["scores"] == original
    assert (
        result["by_star_scores"]["3"]["frequency"]
        == suite.scores(history[-17:])["frequency"]
    )
    assert result["by_star_scores"]["4"]["cdm"] == suite.scores(history[-60:])["cdm"]
    assert result["by_star_scores"]["5"] == original
    assert set(result["contexts"]) == {"5", "17", "60", "1000"}
    result.update(
        models=suite.statuses(),
        version="test-artifact",
        training_cutoff=data[-10]["draw_term"],
    )
    audit = context_audit(result, data, "frequency", 3, plan)
    assert audit["history_start"] == data[-17]["draw_term"]
    assert audit["history_count"] == 17
    assert audit["feature_cutoff"] == data[-1]["draw_term"]
    assert audit["training_cutoff"] == data[-1]["draw_term"]
    assert audit["batch_training_cutoff"] == data[-10]["draw_term"]
    assert audit["input_hash"] and audit["feature_hash"]
    assert audit["scoring_cost_seconds"] >= 0


def test_rollback_context_is_computed_and_inputs_not_mutated():
    data = snapshot()
    original = copy.deepcopy(data)
    suite = ModelSuite([r["numbers"] for r in data], enabled=["frequency", "cdm"]).fit()
    plan = {
        "shadow_windows": [5, 10, 1000],
        "by_model": {"frequency": {"required_windows": [200]}},
    }
    result = context_scores(suite, data, plan)
    assert "200" in result["contexts"]
    assert data == original
    for vector in result["scores"].values():
        assert len(vector) == 80 and np.isfinite(vector).all()
    with pytest.raises(ValueError):
        context_scores(suite, data, {"shadow_windows": list(range(1, 100))})
