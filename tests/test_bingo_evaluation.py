"""Evidence checks for causal folds, vote rules and honest sample accounting."""

from datetime import datetime, timedelta
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_evaluation import (
    fuse,
    live_rankings,
    normalize_scores,
    recommend_method,
    summarize_outcomes,
    walk_forward,
)


def draws(n=205):
    rng = np.random.default_rng(7)
    return [
        {
            "draw_term": str(115000001 + i),
            "draw_datetime": datetime(2026, 1, 1) + timedelta(minutes=5 * i),
            "numbers": sorted(rng.choice(np.arange(1, 81), 20, replace=False).tolist()),
        }
        for i in range(n)
    ]


def test_normalization_invalid_models_and_deterministic_votes():
    assert normalize_scores([7] * 80) == [0.5] * 80
    with pytest.raises(ValueError):
        normalize_scores([float("nan")] * 80)
    score = np.linspace(0, 1, 80)
    for method in ("equal", "shrink", "borda", "majority"):
        result = fuse({"a": score, "broken": [0]}, method, 5)
        assert result["ranking"][:5] == [80, 79, 78, 77, 76]
        assert sorted(result["ranking"]) == list(range(1, 81))
        assert result["excluded"] == {"broken": "模型必須提供 80 個有限分數"}
        assert result["weights"] == {"a": 1.0}
        assert result == fuse({"a": score, "broken": [0]}, method, 5)


def test_equal_retains_strength_borda_uses_rank_majority_uses_top_k():
    a = np.zeros(80)
    b = np.zeros(80)
    c = np.zeros(80)
    a[:4] = [1, 0.99, 0.98, 0.97]
    b[:4] = [0.001, 1, 0.002, 0.003]
    c[:4] = [0.001, 1, 0.003, 0.002]
    scores = {"a": a, "b": b, "c": c}
    equal = fuse(scores, "equal", 3)
    borda = fuse(scores, "borda", 3)
    majority = fuse(scores, "majority", 3)
    assert equal["scores"] != borda["scores"]
    assert majority["support"]["2"] == 3
    assert majority["ranking"][0] == 2


def test_shrink_requires_30_for_every_model_and_stays_near_equal():
    scores = {"winner": list(range(80)), "other": list(range(79, -1, -1))}
    early = [
        {"model": "winner", "stars": 3, "samples": 29, "mean_hits": 3},
        {"model": "other", "stars": 3, "samples": 500, "mean_hits": 0},
    ]
    assert fuse(scores, "shrink", 3, early)["weights"] == {"other": 0.5, "winner": 0.5}
    early[0]["samples"] = 30
    weights = fuse(scores, "shrink", 3, early)["weights"]
    assert 0.5 < weights["winner"] < 0.75
    assert sum(weights.values()) == pytest.approx(1.0)


def test_hypergeometric_baseline_and_sample_claims():
    for stars, expected in ((3, 0.75), (4, 1.0), (5, 1.25)):
        stats = summarize_outcomes([{"hit_count": stars}] * 29, stars)
        assert stats["random_mean"] == expected
        assert stats["full_hit_rate"] == 1.0
        assert stats["full_hit_ci_low"] < 1.0
        assert not stats["proven_better_than_random"]
        assert stats["credibility"] == "樣本不足"
    stats = summarize_outcomes([{"hit_count": 1}] * 100, 4)
    assert stats["delta_vs_random"] == 0
    assert stats["ci_low"] < 1 < stats["ci_high"]
    assert not stats["proven_better_than_random"]


def test_gaps_do_not_erase_cumulative_results():
    rows = [
        {"model": "frequency", "stars": 3, "target_term": str(term), "hit_count": 1}
        for term in (101, 102, 103, 105, 106)
    ]
    board = live_rankings(rows)["3"]
    assert board["cumulative"][0]["samples"] == 5
    assert board["consecutive"][0]["samples"] == 2
    with pytest.raises(ValueError):
        live_rankings(rows + [rows[0]])


def test_year_boundary_continuity_requires_verified_times():
    rows = [
        {
            "model": "x",
            "stars": 3,
            "target_term": "114074095",
            "hit_count": 1,
            "draw_datetime": "2025-12-31T23:55:00+08:00",
        },
        {
            "model": "x",
            "stars": 3,
            "target_term": "115000001",
            "hit_count": 0,
            "draw_datetime": "2026-01-01T07:05:00+08:00",
        },
    ]
    assert live_rankings(rows)["3"]["consecutive"][0]["samples"] == 2


class CausalSuite:
    fits = []
    updates = []

    def __init__(self, history, **kwargs):
        self.history = [row[:] for row in history]

    def fit(self, progress=None):
        self.fits.append(self.history)
        return self

    def scores(self, history):
        vector = np.zeros(80)
        vector[np.asarray(history[-1]) - 1] = 1
        return {"causal": vector.tolist()}

    def update(self, previous_history, new_draw):
        self.updates.append((previous_history[-1][:], new_draw[:]))

    def statuses(self):
        return [{"name": "causal", "status": "ready"}]


def test_walk_forward_future_answer_changes_cannot_change_earlier_prediction(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "taiwan_lottery.ml.bingo_models",
        SimpleNamespace(ModelSuite=CausalSuite),
    )
    CausalSuite.fits.clear()
    CausalSuite.updates.clear()
    data = draws()
    report = walk_forward(data, count=105, retrain_every=100)
    assert len(CausalSuite.fits) == 2
    assert len(CausalSuite.fits[0]) == 100
    assert len(CausalSuite.fits[1]) == 200
    assert report["folds"][0]["trained_through"] == data[99]["draw_term"]
    assert report["folds"][100]["trained_through"] == data[199]["draw_term"]
    altered = [dict(row) for row in data]
    altered[100]["numbers"] = list(range(1, 21))
    other = walk_forward(altered, count=105)
    assert report["folds"][0] == other["folds"][0]
    assert (
        report["folds"][0]["predictions"]["causal"]["3"]
        == sorted(data[99]["numbers"])[:3]
    )
    assert report["is_live"] is False
    assert report["leaderboards"]["3"]["cumulative"][0]["samples"] == 105
    assert report["decision"]["validated"] is False


def test_insufficient_training_rejected_and_custom_range(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "taiwan_lottery.ml.bingo_models",
        SimpleNamespace(ModelSuite=CausalSuite),
    )
    with pytest.raises(ValueError, match="資料不足"):
        walk_forward(draws(199), count=100)
    data = draws()
    report = walk_forward(
        data,
        count=5,
        start_term=data[100]["draw_term"],
        end_term=data[104]["draw_term"],
    )
    assert report["count"] == 5
    with pytest.raises(ValueError, match="區間"):
        walk_forward(
            data,
            count=4,
            start_term=data[100]["draw_term"],
            end_term=data[104]["draw_term"],
        )
    assert recommend_method({"results": []})["method"] == "shrink"


def test_strategy_recommendation_requires_all_three_star_paired_histories():
    rows = []
    for index in range(100):
        for stars in (3, 4, 5):
            for method in ("equal", "shrink", "borda", "majority"):
                rows.append(
                    {
                        "target_term": str(1000 + index),
                        "stars": stars,
                        "model": method,
                        "hit_count": stars if method == "equal" else 0,
                    }
                )
    decision = recommend_method({"results": rows})
    assert decision["method"] == "equal"
    assert decision["validated"] is True
    # Missing one star's paired outcomes disqualifies the candidate.
    incomplete = [r for r in rows if not (r["stars"] == 5 and r["model"] == "equal")]
    assert recommend_method({"results": incomplete})["method"] == "shrink"


def test_histogram_statistics_match_raw_without_expanding_lifetime_samples():
    from collections import Counter
    from taiwan_lottery.ml.bingo_evaluation import live_rankings_summary

    for stars in (3, 4, 5):
        hits = [0, 1, 2, stars, 0, 1, stars, 2, 1] * 37
        rows = [{"hit_count": value} for value in hits]
        histogram = [
            {"hit_count": value, "count": count}
            for value, count in Counter(hits).items()
        ]
        raw = summarize_outcomes(rows, stars)
        compact = summarize_outcomes(histogram, stars)
        for key in raw:
            if isinstance(raw[key], float):
                assert compact[key] == pytest.approx(raw[key])
            else:
                assert compact[key] == raw[key]
    # Billions of lifetime outcomes require six histogram bins, not billions of Python objects.
    huge = summarize_outcomes([{"hit_count": 1, "count": 10**9}], 5)
    assert huge["samples"] == 10**9
    assert huge["mean_hits"] == 1.0
    assert huge["ci_low"] < 1 < huge["ci_high"]
    with pytest.raises(ValueError, match="樣本數"):
        summarize_outcomes([{"hit_count": 1, "count": -1}], 3)


def test_summary_leaderboards_preserve_lifetime_and_final_streak_exactly():
    from collections import defaultdict
    from taiwan_lottery.ml.bingo_evaluation import live_rankings_summary

    rows = [
        {
            "model": model,
            "stars": stars,
            "target_term": str(term),
            "hit_count": hit,
            "updated_at": "2026-11-03T12:00:00+08:00",
        }
        for model in ("a", "b")
        for stars in (3, 4, 5)
        for term, hit in ((101, 0), (102, 1), (103, 3), (105, 2), (106, 1))
    ]
    summary = {}
    for kind in ("cumulative", "consecutive"):
        buckets = defaultdict(int)
        for row in rows:
            if kind == "cumulative" or int(row["target_term"]) >= 105:
                buckets[(row["model"], row["stars"], row["hit_count"])] += 1
        summary[kind] = [
            {
                "model": model,
                "stars": stars,
                "hit_count": hit,
                "count": count,
                "updated_at": "2026-11-03T12:00:00+08:00",
            }
            for (model, stars, hit), count in buckets.items()
        ]
    weights = {"3": {"a": 0.6, "b": 0.4}}
    raw = live_rankings(rows, weights=weights)
    compact = live_rankings_summary(summary, weights=weights)
    assert compact == raw
    assert compact["3"]["cumulative"][0]["samples"] == 5
    assert compact["3"]["consecutive"][0]["samples"] == 2


def test_consecutive_terms_with_inconsistent_clock_are_not_a_valid_streak():
    rows = [
        {
            "model": "x",
            "stars": 3,
            "target_term": str(101 + i),
            "hit_count": 1,
            "draw_datetime": at,
        }
        for i, at in enumerate(
            ("2026-11-03T10:00:00+08:00", "2026-11-03T10:10:00+08:00")
        )
    ]
    assert live_rankings(rows)["3"]["cumulative"][0]["samples"] == 2
    assert live_rankings(rows)["3"]["consecutive"][0]["samples"] == 1


def test_missing_term_is_not_reinterpreted_as_the_next_draw():
    from taiwan_lottery.ml.bingo_evaluation import walk_forward
    from datetime import datetime, timedelta

    data = [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=datetime(2026, 1, 1, 7, 5) + timedelta(minutes=5 * i),
            numbers=list(range(1, 21)),
        )
        for i in range(220)
    ]
    del data[140]
    with pytest.raises(ValueError, match="期號有缺漏"):
        walk_forward(data, count=100, model_names=["frequency", "cdm"])
