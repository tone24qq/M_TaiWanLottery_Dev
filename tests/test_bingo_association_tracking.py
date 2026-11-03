"""Prospective A/B/C tracking must be causal, paired, durable and descriptive."""

import copy
from datetime import datetime, timedelta
import json
import math

import numpy as np
import pytest

from taiwan_lottery.services.bingo_association_tracking import AssociationTracker


def observed(index=0, *, hits=None, is_live=None, flat=False, expired=False):
    target = str(115000101 + index)
    deadline = datetime(2026, 11, 3, 9) + timedelta(minutes=5 * index)
    rows = []
    for stars in (3, 4, 5):
        for family in ("a", "b", "c"):
            count = (
                hits[family](stars, index)
                if hits and family in hits
                else stars
                if family == "a"
                else 0
                if family == "b"
                else 1
            )
            numbers = list(range(1, count + 1)) + list(range(41, 41 + stars - count))
            row = dict(
                model="association_" + family,
                stars=stars,
                target_term=target,
                source_term=str(int(target) - 1),
                source_time=(deadline - timedelta(minutes=5)).isoformat(),
                created_at=(deadline - timedelta(minutes=4)).isoformat(),
                target_time=deadline.isoformat(),
                settled_at=(deadline + timedelta(seconds=10)).isoformat(),
                numbers=numbers,
                scores=[0.25] * 80
                if flat
                else (np.arange(80, dtype=float) / 79).tolist(),
                actual_numbers=list(range(1, 21)),
                hit_count=count,
                version=f"model-{family}-1",
                decision_hash=f"sealed-{target}-{stars}-{family}",
                metadata=dict(
                    profile="association-v1",
                    rule_version="rules-1",
                    feature_cutoff=str(int(target) - 1),
                    rules=[
                        dict(
                            rule_id="1->2",
                            validated_through=str(int(target) - 1),
                            valid_through_term=str(int(target) - 1)
                            if expired
                            else str(int(target) + 10),
                        )
                    ],
                ),
            )
            if is_live is not None:
                row["is_live"] = is_live
            rows.append(row)
    return target, rows


def lookup(status, stars, family):
    return next(
        row for row in status["stars"][str(stars)]["leaderboard"] if row["id"] == family
    )


def test_complete_nine_rows_preserve_decisions_and_report_real_random_formulas():
    tracker = AssociationTracker()
    target, rows = observed()
    before = copy.deepcopy(rows)
    assert tracker.observe(target, rows) == []
    assert rows == before
    status = tracker.status()
    assert status["observations"] == 1 and not status["automatic_adoption"]
    assert set(status["stars"]) == {"3", "4", "5"}
    for stars in (3, 4, 5):
        a = lookup(status, stars, "a")
        b = lookup(status, stars, "b")
        c = lookup(status, stars, "c")
        assert a["samples"] == 1 and a["full_hits"] == 1 and a["mean_hits"] == stars
        assert b["zero_hits"] == 1 and c["partial_hits"] == 1
        assert a["random_mean"] == stars * 0.25
        assert a["random_full_hit_rate"] == pytest.approx(
            math.comb(20, stars) / math.comb(80, stars)
        )
        assert not a["proven_better_than_random"]
        assert a["cumulative"]["hit_distribution"][str(stars)] == 1
        assert a["recent"]["100"]["samples"] == 1
        assert a["recent"]["300"]["samples"] == 1
        assert a["versions"] == {"model-a-1": 1}


def test_cumulative_and_recent_100_300_are_independent_exact_samples():
    tracker = AssociationTracker()
    hits = {
        "a": lambda stars, index: stars if index < 5 else 0,
        "b": lambda stars, index: index % (stars + 1),
    }
    for index in range(305):
        tracker.observe(*observed(index, hits=hits))
    status = tracker.status()
    for stars in (3, 4, 5):
        a = lookup(status, stars, "a")
        assert a["samples"] == 305 and a["full_hits"] == 5
        assert a["mean_hits"] == pytest.approx(5 * stars / 305)
        assert (
            a["recent"]["100"]["samples"] == 100
            and a["recent"]["100"]["mean_hits"] == 0
        )
        assert (
            a["recent"]["300"]["samples"] == 300
            and a["recent"]["300"]["mean_hits"] == 0
        )
        b = lookup(status, stars, "b")
        assert b["mean_hits"] == pytest.approx(
            np.mean([i % (stars + 1) for i in range(305)])
        )
    assert "champion" not in status["stars"]["3"]


def test_restart_and_repeat_are_idempotent_and_changed_duplicate_is_rejected():
    tracker = AssociationTracker()
    first = observed(0)
    tracker.observe(*first)
    state = tracker.to_dict()
    restored = AssociationTracker(json.loads(json.dumps(state)))
    assert restored.status() == tracker.status()
    assert restored.observe(*first) == []
    assert restored.status() == tracker.status()
    changed = copy.deepcopy(first[1])
    changed[0]["scores"][0] = 0.125
    with pytest.raises(ValueError, match="衝突"):
        restored.observe(first[0], changed)
    assert restored.to_dict() == state
    state["groups"].clear()
    assert restored.status()["observations"] == 1


def test_out_of_order_period_and_time_cannot_revise_existing_evidence():
    tracker = AssociationTracker()
    tracker.observe(*observed(1))
    baseline = tracker.to_dict()
    with pytest.raises(ValueError, match="順序"):
        tracker.observe(*observed(0))
    target, rows = observed(2)
    for row in rows:
        row["source_time"] = "2026-11-03T08:40:00"
        row["created_at"] = "2026-11-03T08:41:00"
        row["target_time"] = "2026-11-03T08:45:00"
        row["settled_at"] = "2026-11-03T08:46:00"
    with pytest.raises(ValueError, match="順序"):
        tracker.observe(target, rows)
    assert tracker.to_dict() == baseline


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r[0].update(source_term=r[0]["target_term"]), "未來"),
        (lambda r: r[0].update(created_at=r[0]["target_time"]), "開獎前"),
        (lambda r: r[0].update(source_time=r[0]["target_time"]), "來源時間"),
        (lambda r: r[0].update(settled_at=r[0]["created_at"]), "結算"),
        (lambda r: r[0]["metadata"].update(feature_cutoff=r[0]["target_term"]), "未來"),
        (
            lambda r: r[0]["metadata"]["rules"][0].update(
                validated_through=r[0]["target_term"]
            ),
            "未來",
        ),
        (lambda r: r[0].update(scores=[float("nan")] * 80), "有限"),
        (lambda r: r[0].update(scores=[0.25] * 79), "80"),
        (lambda r: r[0].update(hit_count=0), "命中"),
        (lambda r: r[0].update(actual_numbers=list(range(1, 20))), "20"),
    ],
)
def test_bad_or_future_rows_are_rejected_before_any_state_mutation(mutate, match):
    tracker = AssociationTracker()
    target, rows = observed()
    before = tracker.to_dict()
    mutate(rows)
    with pytest.raises(ValueError, match=match):
        tracker.observe(target, rows)
    assert tracker.to_dict() == before


def test_incomplete_duplicated_or_cross_answer_comparison_is_rejected():
    tracker = AssociationTracker()
    target, rows = observed()
    with pytest.raises(ValueError, match="九列"):
        tracker.observe(target, rows[:-1])
    with pytest.raises(ValueError, match="重複"):
        tracker.observe(target, rows + [copy.deepcopy(rows[0])])
    changed = copy.deepcopy(rows)
    changed[0]["actual_numbers"] = list(range(2, 22))
    changed[0]["hit_count"] = 2
    with pytest.raises(ValueError, match="結果不一致"):
        tracker.observe(target, changed)
    assert tracker.status()["observations"] == 0
    assert tracker.observe(target, [dict(model="frequency")]) == []


def test_simulation_and_live_state_and_rows_are_strictly_separate():
    target, rows = observed(is_live=False)
    with pytest.raises(ValueError, match="模擬"):
        AssociationTracker().observe(target, rows)
    simulated = AssociationTracker(simulation=True)
    simulated.observe(target, rows)
    assert not simulated.status()["is_live"]
    with pytest.raises(ValueError, match="模擬"):
        AssociationTracker(simulated.to_dict())
    live = AssociationTracker()
    target, rows = observed(is_live=True)
    live.observe(target, rows)
    with pytest.raises(ValueError, match="模擬"):
        AssociationTracker(live.to_dict(), simulation=True)
    with pytest.raises(ValueError, match="模擬"):
        AssociationTracker(simulation=True).observe(target, rows)


def test_expired_rules_and_flat_scores_request_only_independent_revalidation_with_cooldown():
    tracker = AssociationTracker()
    target, rows = observed(flat=True, expired=True)
    original = copy.deepcopy(rows)
    events = tracker.observe(target, rows)
    assert len(events) == 9 and all(not event["automatic_adoption"] for event in events)
    assert all(event["requires_independent_validation"] for event in events)
    assert rows == original
    for index in range(1, 30):
        assert tracker.observe(*observed(index, flat=True, expired=True)) == []
    events = tracker.observe(*observed(30, flat=True, expired=True))
    assert len(events) == 9
    status = tracker.status()
    assert status["observations"] == 31 and not status["automatic_adoption"]
    assert len(status["diagnostics"]) == 30
    for stars in (3, 4, 5):
        group = lookup(status, stars, "a")
        assert group["diagnostics"]["flat_score_forecasts"] == 31
        assert group["diagnostics"]["expired_rule_forecasts"] == 31
        assert group["latest"]["numbers"] == list(range(1, stars + 1))
        assert group["versions"] == {"model-a-1": 31}


def test_bounded_rolling_pairs_and_versions_preserve_all_cumulative_samples():
    tracker = AssociationTracker()
    for index in range(1005):
        target, rows = observed(index)
        if index < 70:
            for row in rows:
                row["version"] = f"artifact-{index}"
        tracker.observe(target, rows)
    state = tracker.to_dict()
    assert len(state["recent_pairs"]) == 1000 and len(state["processed"]) == 1000
    assert state["recent_pairs"][0]["target_term"] == observed(5)[0]
    status = tracker.status()
    a = lookup(status, 3, "a")
    assert a["samples"] == 1005 and a["full_hits"] == 1005
    assert a["older_version_samples"] > 0 and len(a["versions"]) <= 64
    assert sum(a["versions"].values()) + a["older_version_samples"] == 1005
    assert AssociationTracker(json.loads(json.dumps(state))).status() == status


def test_real_official_snapshot_engine_candidates_roundtrip_as_simulation_only():
    """Real observed outcomes and real ABC calculations; never a formal record."""
    import gzip
    from pathlib import Path
    from taiwan_lottery.ml.bingo_associations import association_candidates

    source = (
        Path(__file__).resolve().parents[1]
        / "docs/validation/official-draws-20261103.json.gz"
    )
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    history = data[:200]
    answer = data[200]
    prediction = association_candidates(history, windows=[200], max_lag=2, max_pairs=2)
    before = copy.deepcopy(prediction)
    source_time = datetime.fromisoformat(history[-1]["draw_datetime"])
    target_time = datetime.fromisoformat(answer["draw_datetime"])
    rows = []
    for stars in (3, 4, 5):
        for family in ("a", "b", "c"):
            candidate = prediction["by_star"][str(stars)][family.upper()]
            rows.append(
                dict(
                    model="association_" + family,
                    stars=stars,
                    target_term=answer["draw_term"],
                    source_term=history[-1]["draw_term"],
                    source_time=source_time.isoformat(),
                    created_at=(source_time + timedelta(seconds=1)).isoformat(),
                    target_time=target_time.isoformat(),
                    settled_at=(target_time + timedelta(seconds=1)).isoformat(),
                    numbers=candidate["numbers"],
                    scores=candidate["scores"],
                    actual_numbers=answer["numbers"],
                    hit_count=len(set(candidate["numbers"]) & set(answer["numbers"])),
                    version=prediction["version"],
                    decision_hash=f"simulation:{stars}:{family}",
                    is_live=False,
                    metadata=dict(
                        candidate,
                        training_cutoff=history[-1]["draw_term"],
                        feature_cutoff=history[-1]["draw_term"],
                        rule_version=prediction["version"],
                        rules=[
                            rule
                            for rule in prediction["rules"]
                            if rule["id"] in candidate["evidence"]
                        ],
                    ),
                )
            )
    tracker = AssociationTracker(simulation=True)
    tracker.observe(answer["draw_term"], rows)
    status = tracker.status()
    assert not status["is_live"] and status["observations"] == 1
    for row in rows:
        result = lookup(status, row["stars"], row["model"][-1])
        assert result["mean_hits"] == row["hit_count"]
        assert result["latest"]["numbers"] == row["numbers"]
    assert prediction == before
    with pytest.raises(ValueError, match="模擬"):
        AssociationTracker().observe(answer["draw_term"], rows)
