"""Association fixtures are tests only; production always receives real draws."""

from datetime import datetime
import copy
import json
import math
from unittest.mock import patch

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from taiwan_lottery.bingo_clock import next_draw
from taiwan_lottery.ml import bingo_associations as module


def draws(count=240, *, patterned=False, seed=17):
    rng = np.random.default_rng(seed)
    term = "115000001"
    stamp = datetime(2026, 11, 1, 7, 5)
    rows = []
    for _ in range(count):
        numbers = (
            list(range(1, 21))
            if patterned
            else sorted(rng.choice(np.arange(1, 81), 20, replace=False).tolist())
        )
        rows.append({"draw_term": term, "draw_datetime": stamp, "numbers": numbers})
        term, stamp = next_draw(term, stamp)
    return rows


def engine(data, **kwargs):
    with threadpool_limits(1):
        return module.association_candidates(data, **kwargs)


def test_output_is_full_deterministic_bounded_and_structured_on_noise():
    data = draws(80)
    first = engine(data, windows=[80])
    second = engine(data, windows=[80])
    assert first["by_star"] == second["by_star"]
    assert first["rules"] == second["rules"]
    assert first["resources"]["rule_count"] <= 32
    assert first["resources"]["searched_hypotheses"] >= 80 * 80 * 20 + 80 * 79
    for stars in (3, 4, 5):
        for method in ("A", "B", "C"):
            candidate = first["by_star"][str(stars)][method]
            assert len(candidate["scores"]) == 80 and len(candidate["numbers"]) == stars
            assert sorted(candidate["ranking"]) == list(range(1, 81))
            assert candidate["label"] == module.NO_RELIABLE
            assert not candidate["strategy_proven"] and candidate["shadow_only"]
            assert len(candidate["reasons"]) <= 3
            for evidence in candidate["evidence"]:
                assert (
                    set(evidence["consequence"]) & set(candidate["numbers"])
                    or method == "C"
                )
                assert evidence["trigger_count"] >= evidence["success_count"] >= 0
                assert evidence["random_baseline"] > 0
                assert evidence["ci_low"] <= evidence["ci_high"]
                assert not evidence["reliable"]
    assert first["by_star"]["3"]["A"]["evidence"], (
        "exploration must expose actual counts even without proof"
    )
    assert first["by_star"]["3"]["C"]["evidence"] == []
    assert first["by_star"]["3"]["C"]["components"]["validated_rules"] == 0
    json.dumps(first, allow_nan=False)


def test_real_rule_calculation_uses_three_disjoint_target_segments():
    result = engine(draws(240, patterned=True), windows=[240])
    reliable = [rule for rule in result["rules"] if rule["reliable"]]
    assert reliable, (
        "perfect test-only pattern must exercise the genuine confirmation path"
    )
    for rule in reliable:
        ranges = rule["split_history"]
        assert int(ranges["discovery"]["to"]) < int(
            ranges["selection_validation"]["from"]
        )
        assert int(ranges["selection_validation"]["to"]) < int(
            ranges["confirmation"]["from"]
        )
        for stage in ("discovery", "selection_validation", "confirmation"):
            assert rule[stage]["triggers"] >= 20
            assert rule[stage]["approved"]
            assert rule[stage]["p_value"] <= result["resources"]["bonferroni_alpha"]
    assert result["by_star"]["3"]["C"]["components"]["validated_rules"] == 0.05
    assert all(
        rule["kind"] != "cooccurrence" and rule["reliable"]
        for rule in result["by_star"]["3"]["C"]["evidence"]
    )
    assert result["by_star"]["3"]["C"]["evidence"]


def test_confirmation_answers_cannot_select_the_window_or_discover_rules():
    original = draws(240, patterned=True)
    perturbed = copy.deepcopy(original)
    for row in perturbed[180:]:
        row["numbers"] = list(range(61, 81))
    first = engine(original, windows=[240])
    later = engine(perturbed, windows=[240])
    assert first["window_selection"] == later["window_selection"]
    assert [r["id"] for r in first["rules"]] == [r["id"] for r in later["rules"]]
    assert any(r["reliable"] for r in first["rules"])
    assert not any(r["reliable"] for r in later["rules"])
    assert later["by_star"]["3"]["C"]["components"]["validated_rules"] == 0


def test_no_future_draw_is_consumed_and_repeated_source_is_immutable():
    data = draws(100)
    first = engine(data[:80], windows=[80])
    data[80:] = draws(20, patterned=True)
    again = engine(data[:80], windows=[80])
    assert first["snapshot_hash"] == again["snapshot_hash"]
    assert first["by_star"] == again["by_star"]
    with pytest.raises(ValueError, match="遞增"):
        engine(list(reversed(data[:80])), windows=[80])


def test_temporal_group_null_and_same_draw_conditional_are_distinct():
    result = engine(draws(240, patterned=True), windows=[240])
    cooc = [r for r in result["rules"] if r["kind"] == "cooccurrence"]
    groups = [r for r in result["rules"] if r["kind"] == "group"]
    assert cooc and groups
    assert all(r["random_baseline"] == pytest.approx(19 / 79) for r in cooc)
    assert all(
        r["random_baseline"]
        == pytest.approx(
            math.comb(20, len(r["consequence"])) / math.comb(80, len(r["consequence"]))
        )
        for r in groups
    )
    assert all(2 <= r["lag"] <= 5 and len(r["antecedent"]) == 2 for r in groups)
    assert all(not r["next_target_eligible"] for r in cooc)


def test_input_gap_is_not_treated_as_a_normal_lag_and_overnight_is_valid():
    data = draws(210, patterned=True)
    complete = engine(data, windows=[210])
    assert complete["resources"]["term_gaps"] == 0
    missing = data[:80] + data[81:]
    incomplete = engine(missing, windows=[209])
    assert incomplete["resources"]["term_gaps"] == 1
    values = module.draw_matrix([r["numbers"] for r in data[:8]])
    valid = np.ones(7, dtype=bool)
    valid[3] = False
    trials, success, n = module._lag_counts(values, valid, 3)
    assert n == 2
    assert trials[0] == 2 and success[0, 1] == 2


def test_c_uses_actual_weights_and_never_unvalidated_exploration():
    data = draws(80)
    up = np.arange(80, dtype=float)
    down = up[::-1]
    scores = {str(k): {"frequency": up, "river": down} for k in (3, 4, 5)}
    high = {str(k): {"frequency": 0.99, "river": 0.01} for k in (3, 4, 5)}
    low = {str(k): {"frequency": 0.01, "river": 0.99} for k in (3, 4, 5)}
    first = engine(
        data, model_scores_by_star=scores, model_weights_by_star=high, windows=[80]
    )
    second = engine(
        data, model_scores_by_star=scores, model_weights_by_star=low, windows=[80]
    )
    assert (
        first["by_star"]["3"]["C"]["numbers"] != second["by_star"]["3"]["C"]["numbers"]
    )
    assert first["by_star"]["3"]["C"]["model_weights"] == high["3"]
    actual = module._exploratory_a

    def reverse_exploration(*args, **kwargs):
        raw, reasons, cache = actual(*args, **kwargs)
        return raw[::-1], reasons, cache

    with patch.object(module, "_exploratory_a", reverse_exploration):
        changed = engine(
            data, model_scores_by_star=scores, model_weights_by_star=high, windows=[80]
        )
    assert (
        first["by_star"]["3"]["C"]["scores"] == changed["by_star"]["3"]["C"]["scores"]
    )
    assert first["by_star"]["3"]["C"]["components"]["validated_rules"] == 0


def test_new_target_day_has_no_fabricated_today_samples():
    data = draws(100)
    result = engine(data, windows=[100], target_date="2026-11-02")
    assert result["statistics"]["today_samples"] == 0
    assert not any(result["statistics"]["today_frequency"])
    assert not any(result["statistics"]["momentum"])
    assert not any(result["statistics"]["gap"])
    assert not any(result["statistics"]["cooccurrence"])
    with pytest.raises(ValueError, match="早於"):
        engine(data, windows=[100], target_date="2026-10-31")


def test_bad_inputs_and_unbounded_search_are_rejected():
    data = draws(20)
    for options in (
        {"windows": [5, 10, 17, 20]},
        {"max_lag": 21},
        {"max_pairs": 17},
        {"max_rules": 33},
        {"min_triggers": 19},
    ):
        with pytest.raises(ValueError):
            engine(data, **options)
    corrupt = copy.deepcopy(data)
    corrupt[0]["numbers"] = [1] * 20
    with pytest.raises(ValueError):
        engine(corrupt, windows=[20])


def test_multiple_windows_share_unopened_confirmation_targets():
    original = draws(320, patterned=True)
    perturbed = copy.deepcopy(original)
    for row in perturbed[240:]:
        row["numbers"] = list(range(61, 81))
    first = engine(original, windows=[60, 200, 320])
    later = engine(perturbed, windows=[60, 200, 320])
    assert first["window_selection"] == later["window_selection"]
    assert first["validation_segments"] == later["validation_segments"]
    assert (
        first["validation_segments"]["selection_end_term"] == original[239]["draw_term"]
    )
    assert (
        first["validation_segments"]["confirmation_start_term"]
        == original[240]["draw_term"]
    )
    for rule in first["rules"]:
        assert (
            rule["split_history"]["selection_validation"]["to"]
            == original[239]["draw_term"]
        )
        assert (
            rule["split_history"]["confirmation"]["from"] == original[240]["draw_term"]
        )
    assert any(r["reliable"] for r in first["rules"])
    assert not any(r["reliable"] for r in later["rules"])
