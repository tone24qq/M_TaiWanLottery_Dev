"""Independent scalar checks and causal boundaries for neighbour research."""

from copy import deepcopy
from datetime import datetime
import json

import numpy as np
import pytest
from scipy.stats import binom

from taiwan_lottery.bingo_clock import next_draw
from taiwan_lottery.ml.bingo_neighbors import (
    ALPHA, HYPOTHESES, LAGS, MAX_ROWS, NO_EVIDENCE, OFFSETS, WINDOWS,
    _counts, _lookup, _matrix, _rows, _stat, _steps, _targets,
    neighbor_analysis, neighbor_candidate, neighbor_view, prepare_neighbor_bundle,
)


def draws(count=100, seed=101, fixed=False, start="2026-01-01T07:05:00+08:00", term="115000001"):
    rng = np.random.default_rng(seed)
    stamp = datetime.fromisoformat(start)
    result = []
    for _ in range(count):
        numbers = list(range(1, 21)) if fixed else sorted(map(int, rng.choice(np.arange(1, 81), 20, replace=False)))
        result.append(dict(draw_term=term, draw_datetime=stamp.isoformat(), numbers=numbers))
        term, stamp = next_draw(term, stamp)
    return result


@pytest.fixture(scope="module")
def random_rows():
    return draws(140)


@pytest.fixture(scope="module")
def bundle(random_rows):
    return prepare_neighbor_bundle(random_rows)


def test_all_predeclared_dimensions_are_supported():
    assert WINDOWS == (5, 10, 20, 30, 50, 100, 300, 500, 1000)
    assert LAGS == (1, 2, 3, 5, 10, 20)
    assert set(OFFSETS) == {0, -1, 1, -2, 2, -3, 3, -5, 5}
    assert HYPOTHESES == 9 * 6 * 80 * 80 * 4


@pytest.mark.parametrize("lag", LAGS)
def test_vector_counts_equal_independent_scalar_with_gap(random_rows, lag):
    rows = _rows(random_rows[:37] + random_rows[38:])
    matrix, steps = _matrix(rows), _steps(rows)
    trial, hit, pair_count = _counts(matrix, steps, lag, 7, 111)
    expected_trial = np.zeros(80, dtype=int)
    expected_hit = np.zeros((80, 80), dtype=int)
    expected_pairs = 0
    for t in range(7 + lag, 111):
        if not all(steps[t - lag:t]):
            continue
        expected_pairs += 1
        for source in rows[t - lag]["numbers"]:
            expected_trial[source - 1] += 1
            for target in rows[t]["numbers"]:
                expected_hit[source - 1, target - 1] += 1
    np.testing.assert_array_equal(trial, expected_trial)
    np.testing.assert_array_equal(hit, expected_hit)
    assert pair_count == expected_pairs


@pytest.mark.parametrize("snapshot", (0, 5, 10, 20))
def test_all_window_prefix_counts_match_scalar_kernel(random_rows, bundle, snapshot):
    rows = _rows(random_rows[:len(random_rows) - snapshot])
    matrix, steps = _matrix(rows), _steps(rows)
    si = bundle["snapshots"].index(snapshot)
    for wi, window in enumerate(WINDOWS):
        for li, lag in enumerate(LAGS):
            start = max(0, len(rows) - window)
            trial, hit, pairs = _counts(matrix, steps, lag, start, len(rows))
            np.testing.assert_array_equal(bundle["arrays"]["trials"][si, wi, li], trial)
            np.testing.assert_array_equal(bundle["arrays"]["hits"][si, wi, li], hit)
            assert bundle["arrays"]["pairs"][si, wi, li] == pairs


def test_no_boundary_wrap_and_all_sources_all_targets_are_queryable(bundle):
    for source in (1, 80):
        view = neighbor_view(bundle, source=source)
        relations = view["by_source"][0]["relations"]
        assert [r["target"] for r in relations] == list(range(1, 81))
        assert all(r["offset"] == r["target"] - source for r in relations)
    latest = neighbor_view(bundle)
    assert len(latest["by_source"]) == 20
    for source in latest["by_source"]:
        assert all(1 <= r["target"] <= 80 and r["offset"] in OFFSETS for r in source["relations"])


def test_duplicate_identical_rows_idempotent_and_conflicts_rejected(random_rows):
    duplicated = random_rows[:20] + [deepcopy(random_rows[19])] + random_rows[20:]
    assert neighbor_candidate(duplicated) == neighbor_candidate(random_rows)
    bad = deepcopy(random_rows[19])
    bad["numbers"] = list(range(1, 21))
    with pytest.raises(ValueError, match="衝突"):
        neighbor_candidate(random_rows + [bad])


@pytest.mark.parametrize("bad_numbers", ([1] * 20, list(range(0, 20)), list(range(62, 82)), [True] + list(range(2, 21)), [1.0] + list(range(2, 21))))
def test_bad_draw_numbers_rejected(bad_numbers):
    rows = draws(1)
    rows[0]["numbers"] = bad_numbers
    with pytest.raises(ValueError, match="20 個不重複整數"):
        neighbor_candidate(rows)


def test_lookup_matches_exact_tail_and_conservative_intervals():
    table = _lookup(100)
    for n, s in ((0, 0), (10, 0), (20, 5), (50, 25), (100, 100)):
        result = _stat(table, s, n)
        assert result["p_value"] == pytest.approx(binom.sf(s - 1, n, .25) if n else 1)
        assert result["adjusted_p_value"] == pytest.approx(min(1, result["p_value"] * HYPOTHESES))
        assert result["ci_low"] <= result["nominal_ci_low"]
        assert result["ci_high"] >= result["nominal_ci_high"]
        assert 0 <= result["ci_low"] <= result["ci_high"] <= 1
    assert not _stat(table, 19, 19)["passed"]
    assert _stat(table, 100, 100)["passed"]


@pytest.mark.parametrize("lag", LAGS)
def test_inference_pairs_never_share_draws_and_cannot_cross_segment(random_rows, lag):
    rows = _rows(random_rows)
    steps = _steps(rows)
    targets = _targets(steps, lag, 35, 105, independent=True)
    endpoints = [i for t in targets for i in (int(t - lag), int(t))]
    assert len(endpoints) == len(set(endpoints))
    assert all(35 <= i < 105 for i in endpoints)


def test_common_holdout_bounds_across_windows(bundle):
    for snapshot in (0, 5, 10, 20):
        views = [neighbor_view(bundle, window=w, snapshot=snapshot, source=72) for w in WINDOWS]
        segments = [v["validation_segments"] for v in views]
        assert all(s == segments[0] for s in segments)
        # Fixed relation and holdout dates do not change with discovery window.
        proofs = [v["by_source"][0]["relations"][71]["confirmation"] for v in views]
        assert all(proof == proofs[0] for proof in proofs)
        assert int(segments[0]["discovery"]["to"]) < int(segments[0]["selection_validation"]["from"])
        assert int(segments[0]["selection_validation"]["to"]) < int(segments[0]["confirmation"]["from"])


def test_insufficient_short_window_not_approved_by_strong_holdout():
    constant = prepare_neighbor_bundle(draws(320, fixed=True), snapshots=(0,))
    short = neighbor_view(constant, window=5, source=1)["by_source"][0]["relations"][0]
    assert short["confirmation"]["passed"]
    assert not short["discovery"]["passed"]
    assert not short["independently_validated"]
    assert short["status"] == NO_EVIDENCE
    long = neighbor_view(constant, window=300, source=1)["by_source"][0]["relations"][0]
    assert long["independently_validated"]
    assert "策略仍待" in long["status"]
    assert neighbor_candidate(draws(320, fixed=True))["allowed_for_live"] is False


def test_past_source_excludes_later_answers(random_rows):
    source = random_rows[99]["draw_term"]
    first = neighbor_analysis(random_rows, source_term=source, source=72)
    changed = deepcopy(random_rows)
    for row in changed[100:]:
        row["numbers"] = list(range(1, 21))
    second = neighbor_analysis(changed, source_term=source, source=72)
    for view in (first, second):
        view.pop("resources")
    assert first == second
    assert first["source_term"] == source
    assert neighbor_candidate(random_rows[:100]) == neighbor_candidate(changed[:100])


def test_repeat_results_and_candidate_json_are_deterministic(random_rows, bundle):
    other = prepare_neighbor_bundle(random_rows)
    for key in bundle["arrays"]:
        np.testing.assert_array_equal(bundle["arrays"][key], other["arrays"][key])
    a, b = neighbor_candidate(random_rows), neighbor_candidate(random_rows)
    assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    assert len(a["scores"]) == len(a["ranking"]) == 80
    assert set(a["ranking"]) == set(range(1, 81))
    assert set(a["by_star"]) == {"3", "4", "5"}
    assert all(len(a["by_star"][str(k)]["numbers"]) == k for k in (3, 4, 5))
    assert "機率" in a["score_kind"] and a["allowed_for_live"] is False


def test_zero_pairs_keep_missing_rates_and_honest_warning():
    bundle = prepare_neighbor_bundle(draws(2))
    view = neighbor_view(bundle, lag=20, source=80)
    relation = view["by_source"][0]["relations"][0]
    assert relation["trigger_count"] == relation["hit_count"] == 0
    assert relation["conditional_rate"] is None
    assert relation["ci_low"] == 0 and relation["ci_high"] == 1
    assert relation["status"] == NO_EVIDENCE
    missing = neighbor_view(bundle, snapshot=20)
    assert missing["available"] is False and missing["by_source"] == []


def test_cross_night_and_year_are_true_successors():
    rows = draws(6, start="2025-12-31T23:50:00+08:00", term="114074000")
    normalized = _rows(rows)
    assert all(_steps(normalized))
    assert rows[2]["draw_term"] == "115000001"
    trial, hit, pairs = _counts(_matrix(normalized), _steps(normalized), 3, 0, len(rows))
    assert pairs == 3
    assert trial.sum() == 60 and hit.sum() == 1200


def test_memory_history_and_query_budgets_are_bounded():
    rows = draws(1050)
    bundle = prepare_neighbor_bundle(rows)
    assert bundle["resources"]["rows"] == MAX_ROWS
    assert bundle["resources"]["tensor_bytes"] < 80 * 1024 * 1024
    assert bundle["resources"]["peak_prefix_bytes"] < 14 * 1024 * 1024
    assert neighbor_candidate(rows)["history"]["count"] == 1000
    for options in (dict(window=999), dict(lag=4), dict(snapshot=2), dict(source=81), dict(source=True)):
        with pytest.raises(ValueError):
            neighbor_view(bundle, **options)
    with pytest.raises(ValueError):
        prepare_neighbor_bundle(rows, snapshots=(0, 5, 10, 20, 30))


def test_pooled_offset_does_not_claim_20_independent_samples_per_draw(bundle):
    view = neighbor_view(bundle)
    zero = next(r for r in view["offset_statistics"] if r["offset"] == 0)
    assert zero["trigger_count"] == 20 * view["valid_pair_count"]
    assert zero["inferential_trigger_count"] == 20 * zero["independent_pair_count"]
    assert zero["independently_validated"] is False
    assert "不將同一期20來源視為獨立樣本" in zero["ci_method"]
