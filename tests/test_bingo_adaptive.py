"""Independent selection/validation, resource and causal-governance regressions."""

import copy
from datetime import datetime, timedelta
import json

import pytest

from taiwan_lottery.ml.bingo_adaptive import AdaptiveController, CANDIDATE_WINDOWS

START = 115000001
BASE = datetime(2026, 11, 3, 7, 5)


def outcome(i, *, window=1000, hits=0, stars=3, cost=0.001, model="frequency"):
    when = BASE + timedelta(minutes=5 * i)
    return {
        "model": model,
        "stars": stars,
        "window": window,
        "hit_count": hits,
        "target_term": str(START + i),
        "source_term": str(START + i - 1),
        "source_time": when - timedelta(minutes=5),
        "created_at": when - timedelta(minutes=1),
        "target_time": when,
        "training_cutoff_term": str(START + i - 1),
        "feature_cutoff_term": str(START + i - 1),
        "scoring_cost_seconds": cost,
    }


def feed(
    controller,
    first,
    count,
    *,
    champion_hits=0,
    candidate_hits=3,
    candidate_cost=0.001,
    candidates=(5, 10),
    stars=3,
    diagnostics=None,
):
    returned = []
    for i in range(first, first + count):
        plan = controller.plan(
            str(START + i - 1), history=1000, model_names=["frequency"]
        )
        window = plan["champions"][str(stars)]["frequency"]
        live = outcome(i, window=window, hits=champion_hits, stars=stars)
        shadows = [
            outcome(i, window=w, hits=candidate_hits, stars=stars, cost=candidate_cost)
            for w in candidates
            if w != window
        ]
        returned.extend(
            controller.observe(str(START + i), [live], shadows, diagnostics)["events"]
        )
    return returned


def test_minimum_guards_bounded_grid_exploration_and_repeatability():
    for options in (
        {"min_validation": 99},
        {"min_selection": 29},
        {"cooldown": 29},
        {"shadow_budget": 9},
    ):
        with pytest.raises(ValueError):
            AdaptiveController(**options)
    controller = AdaptiveController()
    plan = controller.plan("115000099", history=600, model_names=["frequency"])
    assert plan == controller.plan("115000099", history=600, model_names=["frequency"])
    assert plan["shadow_windows"] == [1000, 5, 10]
    assert plan["champions"]["3"]["frequency"] == 1000
    assert plan["by_model"]["frequency"]["stars"]["3"]["effective_window"] == 600
    observed = set()
    for cohort in range(24):
        controller.state["observation_count"] = cohort * 130
        windows = controller.plan("115000099", history=1000)["shadow_windows"]
        assert len(windows) <= 3 and all(5 <= w <= 1000 for w in windows)
        observed.update(windows)
    assert set(CANDIDATE_WINDOWS).issubset(observed)
    assert observed - set(CANDIDATE_WINDOWS), (
        "bounded neighbourhood exploration is actually used"
    )


def test_selection_and_fresh_holdout_are_disjoint_before_any_promotion():
    controller = AdaptiveController()
    feed(controller, 1, 30)
    status = next(r for r in controller.status()["models"] if r["stars"] == 3)
    assert (
        status["window"] == 1000
        and status["selection_samples"] == 30
        and status["validation_samples"] == 0
    )
    feed(controller, 31, 99)
    assert (
        controller.plan(str(START + 129), history=1000)["champions"]["3"]["frequency"]
        == 1000
    )
    events = feed(controller, 130, 1)
    promotion = next(e for e in events if e["type"] == "window_promoted")
    assert promotion["effective_after_term"] == str(START + 130)
    assert promotion["validation"]["samples"] == 100
    assert promotion["validation"]["first_target"] == str(START + 31)
    assert promotion["validation"]["ci_low"] > 0.03
    assert (
        controller.plan(str(START + 130), history=1000)["champions"]["3"]["frequency"]
        == 5
    )
    assert (
        controller.plan(str(START + 130), history=1000)["champions"]["4"]["frequency"]
        == 1000
    )


def test_no_gain_no_measured_cost_and_excess_cost_cannot_replace_champion():
    for gain, cost in ((0, 0.001), (3, None), (3, 0.01)):
        controller = AdaptiveController()
        feed(controller, 1, 130, candidate_hits=gain, candidate_cost=cost)
        assert (
            controller.plan(str(START + 130), history=1000)["champions"]["3"][
                "frequency"
            ]
            == 1000
        )
        assert not any(
            e["type"] == "window_promoted" for e in controller.status()["recent_events"]
        )


def test_repeated_loss_alone_does_not_force_different_window_or_numbers():
    controller = AdaptiveController()
    feed(controller, 1, 130, champion_hits=0, candidate_hits=0)
    assert all(r["window"] == 1000 for r in controller.status()["models"])
    assert controller.state["revision"] == 0
    assert controller.state["profiles_by_star"]["3"]["method"] == "shrink"


def test_future_answers_late_archive_and_historical_results_are_rejected_atomically():
    controller = AdaptiveController()
    controller.plan(str(START), history=1000, model_names=["frequency"])
    base = outcome(1)
    for modification in (
        {"source_term": str(START + 1)},
        {"created_at": base["target_time"]},
        {"training_cutoff_term": str(START + 1)},
        {"feature_cutoff_term": str(START + 1)},
        {"is_live": False},
    ):
        row = {**base, **modification}
        before = controller.to_dict()
        with pytest.raises(ValueError):
            controller.observe(row["target_term"], [row], [])
        assert controller.to_dict() == before


def test_same_target_replay_restore_and_conflict_have_exactly_once_semantics():
    controller = AdaptiveController()
    controller.plan(str(START), history=1000, model_names=["frequency"])
    row = outcome(1)
    first = controller.observe(row["target_term"], [row], [])
    assert first["updated"]
    restored = AdaptiveController(json.loads(json.dumps(controller.to_dict())))
    assert not restored.observe(row["target_term"], [row], [])["updated"]
    assert restored.to_dict() == controller.to_dict()
    with pytest.raises(ValueError):
        restored.observe(row["target_term"], [{**row, "hit_count": 1}], [])
    with pytest.raises(ValueError):
        restored.observe(str(START), [outcome(0)], [])


def test_unknown_window_cannot_expand_the_search_budget():
    controller = AdaptiveController()
    controller.plan(str(START), history=1000, model_names=["frequency"])
    with pytest.raises(ValueError, match="受限搜尋"):
        controller.observe(str(START + 1), [outcome(1)], [outcome(1, window=777)])
    assert controller.state["observation_count"] == 0


def test_invalid_model_health_blocks_promotion_even_with_extreme_shadow_gain():
    controller = AdaptiveController()
    feed(controller, 1, 130, diagnostics={"data_valid": False})
    assert (
        controller.plan(str(START + 130), history=1000)["champions"]["3"]["frequency"]
        == 1000
    )


def test_previous_verified_window_rolls_back_only_after_fresh_paired_evidence():
    controller = AdaptiveController()
    feed(controller, 1, 130)
    assert controller._context("frequency", 3)["window"] == 5
    # The previous verified fixed baseline is shadow-scored after the promotion.
    for i in range(131, 230):
        controller.plan(str(START + i - 1), history=1000, model_names=["frequency"])
        controller.observe(
            str(START + i),
            [outcome(i, window=5, hits=0)],
            [outcome(i, window=1000, hits=3)],
        )
    assert controller._context("frequency", 3)["window"] == 5
    i = 230
    controller.plan(str(START + i - 1), history=1000, model_names=["frequency"])
    result = controller.observe(
        str(START + i),
        [outcome(i, window=5, hits=0)],
        [outcome(i, window=1000, hits=3)],
    )
    rollback = next(e for e in result["events"] if e["type"] == "window_rollback")
    assert (
        rollback["validation"]["samples"] == 100 and rollback["before"]["window"] == 5
    )
    assert controller._context("frequency", 3)["window"] == 1000


def test_context_change_never_mutates_input_archived_forecasts():
    controller = AdaptiveController()
    live = outcome(1)
    shadows = [outcome(1, window=5, hits=3)]
    original = copy.deepcopy((live, shadows))
    controller.plan(str(START), history=1000, model_names=["frequency"])
    controller.observe(str(START + 1), [live], shadows)
    assert (live, shadows) == original
    state = controller.to_dict()
    state["contexts"]["frequency:3"]["window"] = 777
    assert controller._context("frequency", 3)["window"] == 1000


def test_too_short_source_history_uses_visible_fixed_fallback():
    controller = AdaptiveController()
    feed(controller, 1, 130)
    plan = controller.plan(str(START + 130), history=17, model_names=["frequency"])
    assert plan["champions"]["3"]["frequency"] == 1000
    assert plan["by_model"]["frequency"]["stars"]["3"]["effective_window"] == 17
    assert "不足" in plan["by_model"]["frequency"]["stars"]["3"]["reason"]
