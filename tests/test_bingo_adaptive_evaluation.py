"""Adaptive historical replay is causal and never accepted as live evidence."""

from datetime import datetime, timedelta
import copy

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_adaptive_evaluation import adaptive_walk_forward
from taiwan_lottery.ml.bingo_adaptive import AdaptiveController


def rows():
    rng = np.random.default_rng(324)
    return [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=datetime(2026, 1, 1, 7, 5) + timedelta(minutes=5 * i),
            numbers=sorted(map(int, rng.choice(np.arange(1, 81), 20, replace=False))),
        )
        for i in range(240)
    ]


def test_context_replay_freezes_before_revealing_current_and_future_results():
    data = rows()
    before = adaptive_walk_forward(data, count=100, model_names=["frequency", "cdm"])
    changed = copy.deepcopy(data)
    changed[140]["numbers"] = list(range(1, 21))
    changed[141]["numbers"] = list(range(61, 81))
    after = adaptive_walk_forward(changed, count=100, model_names=["frequency", "cdm"])
    assert before["folds"][0]["predictions"] == after["folds"][0]["predictions"]
    for x, y in zip(
        before["folds"][0]["model_decisions"], after["folds"][0]["model_decisions"]
    ):
        assert x["scores"] == y["scores"]
    assert before["is_live"] is False and not before["decision"]["allowed_for_live"]
    for fold in before["folds"]:
        assert int(fold["feature_cutoff"]) < int(fold["target_term"])
        assert int(fold["training_cutoff"]) < int(fold["target_term"])
        assert all(len(row["scores"]) == 80 for row in fold["model_decisions"])


def test_historical_adaptive_outcomes_cannot_update_live_controller():
    data = rows()
    report = adaptive_walk_forward(data, count=1, model_names=["frequency", "cdm"])
    fold = report["folds"][0]
    records = report["shadow"]["results"]
    control = AdaptiveController()
    control.plan(fold["feature_cutoff"], history=1000, model_names=["frequency", "cdm"])
    with pytest.raises(ValueError, match="歷史模擬"):
        control.observe(
            fold["target_term"],
            [r for r in records if not r.get("is_shadow")],
            [r for r in records if r.get("is_shadow")],
        )
