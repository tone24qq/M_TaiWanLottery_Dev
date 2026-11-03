"""Nested parameter choice must precede independent prospective outcomes."""

from datetime import datetime, timedelta
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

from taiwan_lottery.ml.bingo_evaluation import walk_forward
from taiwan_lottery.ml.bingo_nested import (
    CandidateProfile,
    nested_walk_forward,
    paired_gate,
    profile_scores,
)


def data(n=430):
    random = np.random.default_rng(617)
    return [
        {
            "draw_term": str(115000001 + i),
            "draw_datetime": datetime(2026, 1, 1) + timedelta(minutes=i * 5),
            "numbers": sorted(
                random.choice(np.arange(1, 81), 20, replace=False).tolist()
            ),
        }
        for i in range(n)
    ]


class CausalSuite:
    fits = []

    def __init__(self, history, **kwargs):
        self.history = copy.deepcopy(history)

    def fit(self, progress=None):
        self.fits.append(copy.deepcopy(self.history))
        return self

    def scores(self, history):
        vector = np.zeros(80)
        vector[np.asarray(history[-1]) - 1] = 1
        return {"causal": vector.tolist()}

    def update(self, previous_history, new_draw):
        assert previous_history[-1] != []

    def statuses(self):
        return [{"name": "causal", "status": "ready"}]


def use_fake(monkeypatch):
    # Replace only the learned suite; keep the real causal feature helpers.
    from taiwan_lottery.ml import bingo_models

    monkeypatch.setattr(bingo_models, "ModelSuite", CausalSuite)


def test_fixed_scores_and_votes_identical_to_standalone_original(monkeypatch):
    use_fake(monkeypatch)
    rows = data()
    fixed = walk_forward(rows, count=100)
    nested = nested_walk_forward(rows, count=100)
    assert nested["baseline"]["results"] == [
        dict(row, is_live=False) for row in fixed["results"]
    ]
    for old, new in zip(fixed["folds"], nested["folds"]):
        assert old["target_term"] == new["target_term"]
        assert old["trained_through"] == new["trained_through"]
        assert old["features_through"] == new["features_through"]
        assert old["predictions"] == {
            key: value for key, value in new["predictions"].items() if key != "adaptive"
        }
    assert nested["decision"]["allowed_for_live"] is False
    assert {row["model"] for row in nested["results"]} == {
        "causal",
        "equal",
        "shrink",
        "borda",
        "majority",
        "adaptive",
    }


def test_future_target_perturbation_cannot_change_its_own_fold(monkeypatch):
    use_fake(monkeypatch)
    rows = data()
    before = nested_walk_forward(rows, count=100)
    changed = copy.deepcopy(rows)
    changed[330]["numbers"] = list(range(1, 21))
    changed[331]["numbers"] = list(range(61, 81))
    after = nested_walk_forward(changed, count=100)
    assert before["folds"][0] == after["folds"][0]
    assert before["folds"][0]["features_through"] == rows[329]["draw_term"]
    for fold in before["folds"]:
        target = int(fold["target_term"])
        assert int(fold["trained_through"]) < target
        assert int(fold["features_through"]) < target
        assert fold["decision_hash"]
        for profile in fold["profiles"].values():
            assert int(profile["training_cutoff"]) < target
            assert int(profile["feature_cutoff"]) < target
            assert (
                profile["feature_history"]["count"]
                <= profile["profile"]["context_window"]
            )
            for values in profile["model_scores"].values():
                assert len(values) == 80
                assert np.isfinite(values).all()
            for by_method in profile["fusion_scores"].values():
                assert all(len(scores) == 80 for scores in by_method.values())


def test_inner_and_shadow_splits_do_not_overlap_or_use_current_outer(monkeypatch):
    use_fake(monkeypatch)
    result = nested_walk_forward(data(), count=200)
    assert result["nested"]["selection_count"] == 2
    for selection in result["nested"]["selection_audit"]:
        split = selection["splits"]
        assert split["selection"]["count"] == 30
        assert split["validation"]["count"] == 100
        assert int(split["selection"]["to"]) < int(split["validation"]["from"])
        assert int(split["validation"]["to"]) < int(selection["effective_from"])
        if split["shadow_holdout"]["count"]:
            assert split["shadow_holdout"]["count"] == 100
            assert int(split["validation"]["to"]) < int(split["shadow_holdout"]["from"])
            assert int(split["shadow_holdout"]["to"]) < int(selection["effective_from"])
    for proposal in result["decision"]["proposals"].values():
        assert len(proposal["paired"]) == 100
        assert int(proposal["selection_end_term"]) < int(
            proposal["validation_start_term"]
        )
        assert all(
            int(row["training_cutoff"]) < int(row["target_term"])
            for row in proposal["paired"]
        )
        assert all(row["is_live"] is False for row in proposal["paired"])


def test_last_partial_outer_block_never_borrows_earlier_samples(monkeypatch):
    use_fake(monkeypatch)
    report = nested_walk_forward(data(), count=105)
    assert report["nested"]["selection_count"] == 2
    assert set(report["decision"]["independent_validation_samples"].values()) == {5}
    assert report["decision"]["allowed_for_live"] is False
    assert {
        r["samples"] for r in report["baseline"]["leaderboards"]["3"]["cumulative"]
    } == {105}


def test_guarded_paired_confirmation_and_resource_cost():
    impressive = [{"champion_hits": 0, "challenger_hits": 3} for _ in range(100)]
    assert paired_gate(impressive, 3)["approved"]
    assert not paired_gate(impressive[:99], 3)["approved"]
    assert not paired_gate(impressive, 3, resource_ratio=1.51)["approved"]
    neutral = [{"champion_hits": 1, "challenger_hits": 1} for _ in range(1000)]
    assert not paired_gate(neutral, 3)["approved"]
    # Mean improvement alone cannot conceal a loss in full-hit outcomes.
    full_loss = [
        {"champion_hits": 3 if i % 4 == 0 else 0, "challenger_hits": 2}
        for i in range(1000)
    ]
    assert not paired_gate(full_loss, 3)["approved"]


def test_candidate_budget_and_actual_parameter_validation(monkeypatch):
    use_fake(monkeypatch)
    with pytest.raises(ValueError, match="無界"):
        nested_walk_forward(
            data(),
            count=100,
            profiles=[CandidateProfile("fixed")]
            + [CandidateProfile(f"p{i}") for i in range(4)],
        )
    with pytest.raises(ValueError, match="基準"):
        nested_walk_forward(
            data(),
            count=100,
            profiles=[CandidateProfile("fixed", 500), CandidateProfile("short", 60)],
        )
    with pytest.raises(ValueError, match="資料不足"):
        nested_walk_forward(data(199), count=100)
    with pytest.raises(ValueError):
        CandidateProfile("bad", frequency_mix=(1, 1, 1, 1))
    with pytest.raises(ValueError):
        CandidateProfile("bad", retrain_every=0)


def test_real_scorer_hyperparameters_change_calculation_not_names():
    from taiwan_lottery.ml.bingo_models import ModelSuite

    history = [row["numbers"] for row in data(150)]
    suite = ModelSuite(history, enabled=["frequency", "cdm", "ensemble"]).fit()
    base = profile_scores(suite, history, CandidateProfile("fixed"), baseline=True)
    challenger = profile_scores(
        suite,
        history,
        CandidateProfile("recent", 60, 300, (0.55, 0.25, 0.15, 0.05), 24.0, 200),
    )
    assert set(base) == set(challenger) == {"frequency", "cdm", "ensemble"}
    assert not np.allclose(base["frequency"], challenger["frequency"])
    assert not np.allclose(base["cdm"], challenger["cdm"])
    assert not np.allclose(base["ensemble"], challenger["ensemble"])
