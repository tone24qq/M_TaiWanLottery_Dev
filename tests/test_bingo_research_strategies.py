"""Prospective family comparisons, target-day isolation and conservative switching."""

from datetime import datetime, timedelta
import copy

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_research_strategies import ResearchStrategies
from taiwan_lottery.services.bingo_workers import infer_today


def observed(index, day=None):
    date = day or (datetime(2026, 1, 1) + timedelta(days=index // 100))
    at = date.replace(hour=9, minute=0) + timedelta(minutes=5 * (index % 100))
    target = str(115000002 + index)
    rows = []
    for stars in (3, 4, 5):
        for family, hits in [("today", stars), ("historical", 0), ("hybrid", 1)]:
            numbers = list(range(1, hits + 1)) + list(range(41, 41 + stars - hits))
            rows.append(
                dict(
                    model="strategy_" + family,
                    stars=stars,
                    target_term=target,
                    source_term=str(int(target) - 1),
                    created_at=(at - timedelta(minutes=4)).isoformat(),
                    target_time=at.isoformat(),
                    numbers=numbers,
                    scores=[0.5] * 80,
                    actual_numbers=list(range(1, 21)),
                    hit_count=hits,
                    decision_hash=f"{index}:{stars}:{family}",
                    metadata={
                        "history_dates": [at.date().isoformat()]
                        if family == "today"
                        else [],
                        "data_scope": "target_day_only"
                        if family == "today"
                        else "cross_day_history",
                    },
                )
            )
    return target, rows


def test_three_strategies_need_hundreds_of_paired_actual_rows_and_multiple_days():
    tracker = ResearchStrategies()
    for i in range(299):
        target, rows = observed(i)
        tracker.observe(target, rows)
    assert set(tracker.state["champions"].values()) == {"historical"}
    target, rows = observed(299)
    original = copy.deepcopy(rows)
    events = tracker.observe(target, rows)
    assert tracker.state["champions"]["3"] == "today"
    assert any(e["kind"] == "research_strategy_promoted" for e in events)
    assert tracker.status()["stars"]["3"]["paired_samples"] == 300
    assert tracker.status()["stars"]["3"]["days"] == 3
    assert rows == original
    assert tracker.observe(target, rows) == []
    restored = ResearchStrategies(tracker.to_dict())
    assert restored.status() == tracker.status()


def test_hybrid_weights_only_use_preceding_settled_targets_and_never_randomize():
    tracker = ResearchStrategies()
    today = {str(k): list(range(80)) for k in (3, 4, 5)}
    historical = {str(k): list(reversed(range(80))) for k in (3, 4, 5)}
    first = tracker.score(today, historical, 20)
    assert first["3"]["hybrid"]["today_weight"] == 0.5
    assert tracker.score(today, historical, 20) == first
    for i in range(30):
        target, rows = observed(i)
        tracker.observe(target, rows)
    following = tracker.score(today, historical, 20)
    assert following["3"]["hybrid"]["today_weight"] > 0.5
    assert first["3"]["hybrid"]["today_weight"] == 0.5
    assert tracker.score(today, historical, 0)["3"]["hybrid"]["today_weight"] == 0
    assert set(tracker.state["champions"].values()) == {"historical"}


def test_today_training_features_and_early_day_prior_ignore_yesterday(tmp_path):
    prior = [
        dict(
            draw_term="115000100",
            draw_datetime=datetime(2026, 1, 1, 23, 55),
            numbers=list(range(1, 21)),
        )
    ]
    today = [
        dict(
            draw_term=str(115000101 + i),
            draw_datetime=datetime(2026, 1, 2, 7, 5) + timedelta(minutes=5 * i),
            numbers=list(range(41, 61)),
        )
        for i in range(20)
    ]
    zero = infer_today(prior, tmp_path, 1, "2026-01-02")
    assert zero["history_count"] == 0 and zero["history_dates"] == []
    assert zero["warmup"] and set(zero["scores"]["today_prior"]) == {0.25}
    first = infer_today(prior + today, tmp_path, 1, "2026-01-02")
    changed = copy.deepcopy(prior)
    changed[0]["numbers"] = list(range(61, 81))
    second = infer_today(changed + today, tmp_path, 1, "2026-01-02")
    assert first["scores"] == second["scores"]
    assert first["snapshot_hash"] == second["snapshot_hash"]
    assert first["training_cutoff"] == today[-1]["draw_term"]
    assert first["history_start"] == today[0]["draw_term"]
    assert first["history_dates"] == ["2026-01-02"]


def test_foreign_day_and_late_results_are_rejected():
    tracker = ResearchStrategies()
    target, rows = observed(0)
    rows[0]["metadata"]["history_dates"] = ["2025-12-31"]
    with pytest.raises(ValueError, match="其他日期"):
        tracker.observe(target, rows)
    assert tracker.state["observations"] == 0
    target, rows = observed(0)
    rows[0]["created_at"] = rows[0]["target_time"]
    with pytest.raises(ValueError, match="開獎前"):
        tracker.observe(target, rows)
