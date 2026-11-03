"""Controlled numerical fixtures test diagnostics, never production data entry."""

import copy
from datetime import datetime, timedelta
import json

import numpy as np
import pytest

from taiwan_lottery.services.bingo_monitoring import PerformanceMonitor


BASE = datetime(2026, 11, 3, 7, 5)


def observed(
    index,
    *,
    model="frequency",
    stars=3,
    hits=0,
    shift=0,
    flat=False,
    input_hash=None,
    feature_hash=None,
):
    # Explicit test-only settled observations. All metadata obeys source<target
    # and decision time<target time; controlled hits are actual set overlaps.
    target = 115000001 + index
    numbers = [((n + shift) % 80) + 1 for n in range(stars)]
    actual = numbers[:hits] + [n for n in range(1, 81) if n not in numbers][: 20 - hits]
    scores = np.roll(np.arange(80.0, 0.0, -1), shift).tolist()
    if flat:
        scores = [0.5] * 80
    time = BASE + timedelta(minutes=5 * index)
    return {
        "model": model,
        "stars": stars,
        "target_term": str(target),
        "source_term": str(target - 1),
        "source_time": time - timedelta(minutes=5),
        "created_at": time - timedelta(minutes=3),
        "target_time": time,
        "settled_at": time + timedelta(seconds=30),
        "numbers": numbers,
        "scores": scores,
        "version": "model-v1",
        "hit_count": hits,
        "actual_numbers": actual,
        "audit": {
            "input_hash": input_hash or f"input-{index}",
            "feature_hash": feature_hash or f"features-{index}",
            "training_cutoff": str(target - 1),
            "feature_cutoff": str(target - 1),
            "window": 1000,
            "feature_version": "fixed-1",
        },
        "weights": {"frequency": 0.5, "cdm": 0.5},
    }


def feed(monitor, start, end, **kwargs):
    for index in range(start, end):
        row = observed(index, **kwargs)
        report = monitor.observe(row["target_term"], [row])
    return report


def test_no_forced_change_from_zero_full_hits_or_legitimate_repetition():
    monitor = PerformanceMonitor()
    report = feed(monitor, 0, 100, hits=1)
    model = report["models"][0]
    assert model["full_miss_streak"] == 100
    assert model["loss_streak_is_failure"] is False
    assert 0 < model["random_probability_no_full_hit_in_streak"] < 1
    assert model["stability"]["30"]["unchanged_rate"] == 1
    assert model["stability"]["30"]["top_k_jaccard"] == 1
    assert model["stability"]["30"]["rank_displacement"] == 0
    assert model["stability"]["30"]["feature_updates"] == 30
    assert not report["evaluation_requested"]
    assert report["actions"]["force_change"] is False
    assert not report["alerts"]
    assert model["windows"]["100"]["samples"] == 100
    assert model["windows"]["100"]["mean_hits"] == 1
    assert model["windows"]["100"]["random_mean"] == 0.75


def test_rank_movement_topk_overlap_and_consensus_are_numerical():
    monitor = PerformanceMonitor()
    first = observed(0)
    monitor.observe(first["target_term"], [first])
    second = observed(1, shift=1)
    report = monitor.observe(
        second["target_term"], [second, observed(1, model="cdm", shift=30)]
    )
    stability = next(r for r in report["models"] if r["model"] == "frequency")[
        "stability"
    ]["10"]
    assert stability["unchanged_rate"] == 0
    assert stability["top_k_overlap"] == pytest.approx(2 / 3)
    assert stability["top_k_jaccard"] == 0.5
    assert stability["rank_displacement"] > 0
    assert -1 <= stability["rank_spearman"] < 1
    assert report["consensus"]["3"]["models"] == 2
    assert report["consensus"]["3"]["mean_top_k_jaccard"] == 0
    assert report["consensus"]["3"]["disagreement"] == 1


def test_flat_scores_and_stale_feature_inputs_are_diagnosed_without_mutation():
    monitor = PerformanceMonitor()
    original = None
    for index in range(15):
        row = observed(
            index, flat=True, input_hash="stale-input", feature_hash="stale-feature"
        )
        original = copy.deepcopy(row)
        report = monitor.observe(row["target_term"], [row])
        assert row == original
    codes = {a["code"] for a in report["alerts"]}
    assert {"flat_scores", "input_not_refreshed", "feature_cache_suspected"} <= codes
    assert report["models"][0]["score_distribution"][
        "effective_numbers"
    ] == pytest.approx(80)
    assert report["actions"]["repair_data_or_artifact"]
    assert not report["actions"]["force_change"]
    assert not report["evaluation_requested"]


def test_drift_uses_completed_outcomes_and_requests_independent_evaluation():
    monitor = PerformanceMonitor()
    feed(monitor, 0, 100, hits=3)
    reports = []
    for index in range(100, 150):
        row = observed(index, hits=0)
        reports.append(monitor.observe(row["target_term"], [row]))
    assert any(r["evaluation_requested"] for r in reports)
    flagged = [r for r in reports if r["evaluation_requested"]]
    assert all(r["actions"]["force_change"] is False for r in flagged)
    assert any(a["code"] == "performance_regression" for a in reports[-1]["alerts"])
    requested_positions = [
        index for index, r in enumerate(reports) if r["evaluation_requested"]
    ]
    assert all(
        b - a >= 30 for a, b in zip(requested_positions, requested_positions[1:])
    )


def test_unhealthy_data_blocks_challenger_requests_and_reports_failure():
    monitor = PerformanceMonitor()
    feed(monitor, 0, 100, hits=3)
    for index in range(100, 140):
        row = observed(index, hits=0)
        report = monitor.observe(
            row["target_term"],
            [row],
            sync={
                "state": "disconnected",
                "error": "timeout",
                "missing_terms": [{"after": "1", "before": "3"}],
            },
            model_statuses=[
                {"name": "frequency", "status": "failed", "error": "checksum"}
            ],
        )
    codes = {a["code"] for a in report["alerts"]}
    assert {"data_connection", "data_gaps", "model_failure"} <= codes
    assert not report["evaluation_requested"]
    assert report["actions"]["repair_data_or_artifact"]


def test_duplicate_is_idempotent_but_conflicting_replay_and_future_are_rejected():
    monitor = PerformanceMonitor()
    row = observed(0)
    first = monitor.observe(row["target_term"], [row])
    before = monitor.to_dict()
    assert monitor.observe(row["target_term"], [row]) == first
    assert monitor.to_dict() == before
    changed = copy.deepcopy(row)
    changed["version"] = "changed"
    with pytest.raises(ValueError, match="變更"):
        monitor.observe(row["target_term"], [changed])
    invalid = observed(1)
    invalid["audit"]["feature_cutoff"] = invalid["target_term"]
    with pytest.raises(ValueError, match="未來"):
        monitor.observe(invalid["target_term"], [invalid])
    assert monitor.to_dict() == before
    future = observed(1)
    future["created_at"] = future["target_time"]
    with pytest.raises(ValueError, match="開獎後"):
        monitor.observe(future["target_term"], [future])
    assert monitor.to_dict() == before


def test_state_is_bounded_serializable_and_restores_exactly():
    monitor = PerformanceMonitor()
    feed(monitor, 0, 350, hits=1)
    state = monitor.to_dict()
    assert len(state["groups"]["frequency:3"]["history"]) == 300
    assert len(state["reports"]) == 30
    # The monitor stores compact changes rather than full score/rank histories.
    assert all(
        "scores" not in r and "ranking" not in r
        for r in state["groups"]["frequency:3"]["history"]
    )
    encoded = json.dumps(state, allow_nan=False)
    recovered = PerformanceMonitor(json.loads(encoded))
    assert recovered.status() == monitor.status()
    assert recovered.status()["models"][0]["samples"] == 350
    assert recovered.status()["models"][0]["retained_samples"] == 300
    row = observed(350, hits=1)
    assert recovered.observe(row["target_term"], [row]) == monitor.observe(
        row["target_term"], [row]
    )


def test_weight_changes_and_three_star_streams_are_independent():
    monitor = PerformanceMonitor()
    initial = [observed(0, stars=k, hits=k - 1) for k in (3, 4, 5)]
    monitor.observe(initial[0]["target_term"], initial)
    rows = [observed(1, stars=k, hits=1) for k in (3, 4, 5)]
    rows[1]["weights"] = {"frequency": 0.6, "cdm": 0.4}
    rows[1]["audit"]["adjustment_reason"] = "已結算候選驗證通過"
    report = monitor.observe(rows[0]["target_term"], rows)
    assert len(report["models"]) == 3
    assert len(report["weight_changes"]) == 1
    change = report["weight_changes"][0]
    assert change["stars"] == 4 and change["l1_change"] == pytest.approx(0.2)
    assert change["reason"] == "已結算候選驗證通過"
    baseline = {r["stars"]: r["windows"]["10"]["random_mean"] for r in report["models"]}
    assert baseline == {3: 0.75, 4: 1.0, 5: 1.25}


def test_bad_batch_is_atomic_and_invalid_scores_cannot_enter_state():
    monitor = PerformanceMonitor()
    first = observed(0)
    monitor.observe(first["target_term"], [first])
    before = monitor.to_dict()
    good = observed(1)
    bad = observed(1, model="broken")
    bad["scores"][0] = float("nan")
    with pytest.raises(ValueError, match="有限"):
        monitor.observe(good["target_term"], [good, bad])
    assert monitor.to_dict() == before
    unsealed = observed(1)
    unsealed["settled_at"] = unsealed["created_at"]
    with pytest.raises(ValueError, match="結算"):
        monitor.observe(unsealed["target_term"], [unsealed])


def test_expired_training_version_is_checked_before_candidate_evaluation():
    monitor = PerformanceMonitor()
    row = observed(400)
    row["audit"]["training_cutoff"] = "115000001"
    report = monitor.observe(row["target_term"], [row])
    assert any(alert["code"] == "model_stale" for alert in report["alerts"])
    assert report["models"][0]["training_age_draws"] == 399
    assert report["actions"]["repair_data_or_artifact"]
    assert not report["evaluating"]
    assert not report["evaluation_requested"]
