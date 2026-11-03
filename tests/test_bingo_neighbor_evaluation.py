"""Causal replay and frozen-comparator integrity, independent of live artifacts."""

import copy
from datetime import datetime
import math

import numpy as np
import pytest

from taiwan_lottery.bingo_clock import TAIPEI, next_draw
from taiwan_lottery.ml.bingo_neighbor_evaluation import (
    EXPERIMENT_FAMILY,
    _history,
    _partition_evidence,
    audit_comparison_bundle,
    canonical_hash,
    neighbor_evidence,
    neighbor_walk_forward,
)


def history(count=104):
    rng = np.random.default_rng(1781)
    term, stamp = "115000001", datetime(2026, 1, 1, 7, 5, tzinfo=TAIPEI)
    rows = []
    for _ in range(count):
        rows.append(dict(draw_term=term, draw_datetime=stamp.isoformat(),
                         numbers=sorted((rng.choice(80, 20, replace=False) + 1).tolist())))
        term, stamp = next_draw(term, stamp)
    return rows


def frozen_baseline(rows, count=4):
    """Small genuinely fixed prior predictions; outcomes are calculated afresh."""
    folds, results = [], []
    for index in range(len(rows) - count, len(rows)):
        row = rows[index]
        source = rows[index - 1]["draw_term"]
        predictions = {str(star): list(range(1, star + 1)) for star in (3, 4, 5)}
        folds.append(dict(target_term=row["draw_term"], source_term=source,
                          predictions={"frequency": predictions},
                          cutoffs={"frequency": dict(training=source, features=source)}))
        for star in (3, 4, 5):
            numbers = predictions[str(star)]
            results.append(dict(target_term=row["draw_term"], model="frequency", stars=star,
                                predicted_numbers=numbers, actual_numbers=row["numbers"],
                                hit_count=len(set(numbers) & set(row["numbers"])),
                                draw_datetime=row["draw_datetime"], is_live=False))
    return dict(is_live=False, folds=folds, results=results,
                provenance={"kind": "fixed_test_predictions"})


def replay(rows, count=4):
    return neighbor_walk_forward(rows, count=count, threads=1, window=100,
                                 baseline_report=frozen_baseline(rows, count))


def test_repeated_replay_hashes_and_inputs_are_stable():
    rows = history()
    original = copy.deepcopy(rows)
    first, second = replay(rows), replay(rows)
    assert rows == original
    assert first == second
    assert first["decision_chain_sha256"] == second["decision_chain_sha256"]
    for fold in first["folds"]:
        frozen = {key: value for key, value in fold.items() if key != "decision_hash"}
        assert fold["decision_hash"] == canonical_hash(frozen)
        assert int(fold["training_cutoff"]) < int(fold["target_term"])
        assert fold["source_term"] == fold["feature_cutoff"]
        assert fold["history_count"] == 100


def test_mutating_target_and_future_answers_cannot_change_earlier_decision():
    rows = history()
    altered = copy.deepcopy(rows)
    # Change the first target and all later answers, retaining every past row.
    for row in altered[-4:]:
        row["numbers"] = sorted(set(range(1, 81)) - set(row["numbers"]))[:20]
    before, after = replay(rows), replay(altered)
    assert before["folds"][0] == after["folds"][0]
    assert before["results"] != after["results"]
    # A seeded comparator also cannot depend on the answer, even at later folds.
    assert [f["random_ranking"] for f in before["folds"]] == [f["random_ranking"] for f in after["folds"]]


@pytest.mark.parametrize("mutation", ["duplicate", "missing_term", "wrong_time", "missing_time"])
def test_replay_rejects_missing_duplicate_or_inconsistent_schedule(mutation):
    rows = history()
    if mutation == "duplicate":
        rows.insert(50, copy.deepcopy(rows[50]))
    elif mutation == "missing_term":
        rows.pop(50)
    elif mutation == "wrong_time":
        rows[50]["draw_datetime"] = rows[49]["draw_datetime"]
    else:
        rows[50]["draw_datetime"] = None
    with pytest.raises(ValueError):
        replay(rows)


@pytest.mark.parametrize("mutation", [
    "live", "future_training", "future_features", "missing_cutoffs", "empty_cutoffs",
    "missing_feature_cutoff", "mismatched_source", "changed_prediction", "changed_actual",
    "changed_hits", "duplicate_fold", "duplicate_outcome", "missing_outcome", "unknown_model",
    "unsupported_star", "empty_comparison", "unequal_model_samples",
])
def test_frozen_comparator_audit_rejects_tampered_or_incomplete_evidence(mutation):
    rows = history()
    bundle = frozen_baseline(rows)
    fold, outcome = bundle["folds"][0], bundle["results"][0]
    if mutation == "live":
        bundle["is_live"] = True
    elif mutation in ("future_training", "future_features"):
        key = "training" if mutation == "future_training" else "features"
        fold["cutoffs"]["frequency"][key] = fold["target_term"]
    elif mutation == "missing_cutoffs":
        fold.pop("cutoffs")
    elif mutation == "empty_cutoffs":
        fold["cutoffs"]["frequency"] = {}
    elif mutation == "missing_feature_cutoff":
        fold["cutoffs"]["frequency"].pop("features")
    elif mutation == "mismatched_source":
        fold["source_term"] = rows[1]["draw_term"]
    elif mutation == "changed_prediction":
        outcome["predicted_numbers"] = [77, 78, 79]
    elif mutation == "changed_actual":
        outcome["actual_numbers"] = list(range(1, 21))
    elif mutation == "changed_hits":
        outcome["hit_count"] = (outcome["hit_count"] + 1) % 4
    elif mutation == "duplicate_fold":
        bundle["folds"].append(copy.deepcopy(fold))
    elif mutation == "duplicate_outcome":
        bundle["results"].append(copy.deepcopy(outcome))
    elif mutation == "missing_outcome":
        bundle["results"].pop()
    elif mutation == "unknown_model":
        outcome["model"] = "unsealed"
    elif mutation == "unsupported_star":
        outcome["stars"] = 6
    elif mutation == "empty_comparison":
        bundle["folds"], bundle["results"] = [], []
    else:
        # One comparator may not silently omit a target while others retain it.
        for f in bundle["folds"][1:]:
            f["predictions"]["cdm"] = copy.deepcopy(f["predictions"]["frequency"])
            f["cutoffs"]["cdm"] = copy.deepcopy(f["cutoffs"]["frequency"])
        bundle["results"] += [{**copy.deepcopy(r), "model": "cdm"} for r in bundle["results"]
                              if r["target_term"] != fold["target_term"]]
    with pytest.raises(ValueError):
        audit_comparison_bundle(bundle, _history(rows), [r["draw_term"] for r in rows[-4:]])


def test_all_comparators_use_the_same_targets_and_three_stars():
    report = replay(history())
    targets = {f["target_term"] for f in report["folds"]}
    for name in ("frequency", "neighbor", "random"):
        for star in (3, 4, 5):
            outcomes = [r for r in report["results"] if r["model"] == name and r["stars"] == star]
            assert len(outcomes) == len(targets)
            assert {r["target_term"] for r in outcomes} == targets
            assert all(r["is_live"] is False for r in outcomes)
    assert report["decision"]["allowed_for_live"] is False
    assert report["decision"]["validated"] is False


def test_star_statistics_use_combination_baseline_and_declared_family():
    rows = [{"hit_count": n} for n in (0, 1, 2, 3)]
    measured = _partition_evidence(rows, 3, EXPERIMENT_FAMILY)
    assert measured["mean_hits"] == 1.5
    assert measured["random_mean"] == .75
    assert measured["full_hit_rate"] == .25
    assert measured["random_full_hit_rate"] == math.comb(20, 3) / math.comb(80, 3)
    assert measured["mean_clears_random"] is False
    assert EXPERIMENT_FAMILY >= 3 * 3 * 2 * 3


def test_confirmation_cannot_be_replaced_by_best_subperiod_and_never_promotes():
    rows = []
    for index in range(400):
        for star in (3, 4, 5):
            rows.append(dict(model="neighbor", stars=star, target_term=str(115001000 + index),
                             hit_count=star if index < 200 else 0))
    result = neighbor_evidence(rows)
    for star in (3, 4, 5):
        measured = result[str(star)]
        assert measured["discovery"]["mean_clears_random"] is True
        assert measured["confirmation"]["mean_clears_random"] is False
        assert measured["independently_validated"] is False
        assert measured["status"] == "尚未證明優於隨機"


def test_even_overwhelming_research_evidence_cannot_authorize_production_weights():
    # Deliberately artificial deterministic draws exercise the positive evidence
    # branch; they are a test fixture and never presented as official results.
    rows = history(300)
    for row in rows:
        row["numbers"] = list(range(1, 21))
    report = replay(rows, count=200)
    assert all(item["proven_better_than_random"] for item in report["validation"].values())
    assert report["decision"]["allowed_for_live"] is False
    assert report["decision"]["validated"] is False
    assert all(fold["candidate"]["allowed_for_live"] is False for fold in report["folds"])


def test_fresh_two_model_and_abc_replay_integration():
    report = neighbor_walk_forward(history(102), count=2, threads=1, window=100,
                                   model_names=["frequency", "cdm"])
    models = {r["model"] for r in report["results"]}
    assert {"frequency", "cdm", "association_a", "association_b", "association_c", "neighbor", "random"} <= models
    assert report["comparison_provenance"]["kind"] == "fresh_causal_abc_and_eight_model_replay"
    assert report["count"] == 2
    assert report["decision"]["allowed_for_live"] is False
    assert all(r["is_live"] is False for r in report["results"])
    assert all(f["candidate"]["allowed_for_live"] is False for f in report["folds"])
