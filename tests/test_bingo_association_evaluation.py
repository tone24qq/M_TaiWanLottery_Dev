"""Full historical association API evaluation uses causal frozen model folds."""

import copy
import gzip
import json
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from taiwan_lottery.ml.bingo_association_evaluation import association_walk_forward


def history():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/validation/official-draws-20261103.json.gz"
    )
    with gzip.open(path, "rt") as stream:
        return json.load(stream)[:240]


def replay(data):
    with threadpool_limits(1):
        return association_walk_forward(
            data, count=2, threads=1, model_names=["frequency", "cdm"]
        )


def test_current_and_future_answers_cannot_change_presealed_abc_scores_or_rules():
    data = history()
    first = replay(data)
    changed = copy.deepcopy(data)
    changed[-2]["numbers"] = list(range(1, 21))
    changed[-1]["numbers"] = list(range(61, 81))
    later = replay(changed)
    assert first["folds"][0] == later["folds"][0] or {
        k: v for k, v in first["folds"][0].items() if k != "resources"
    } == {k: v for k, v in later["folds"][0].items() if k != "resources"}
    assert first["is_live"] is False and not first["decision"]["allowed_for_live"]
    assert len(first["folds"]) == 2
    for fold in first["folds"]:
        assert int(fold["training_cutoff"]) < int(fold["target_term"])
        assert int(fold["feature_cutoff"]) < int(fold["target_term"])
        assert all(
            len(v["scores"]) == 80
            for group in fold["by_star"].values()
            for v in group.values()
        )
    assert {r["model"] for r in first["results"]} == {
        "frequency",
        "cdm",
        "equal",
        "shrink",
        "borda",
        "majority",
        "association_a",
        "association_b",
        "association_c",
    }
    for star in first["leaderboards"].values():
        assert {r["samples"] for r in star["cumulative"]} == {2}
    json.dumps(first, allow_nan=False)


def test_association_replay_rejects_unbounded_search_and_missing_periods():
    with pytest.raises(ValueError, match="1000"):
        association_walk_forward(history(), count=1001)
    data = history()
    del data[20]
    with pytest.raises(ValueError, match="缺漏"):
        replay(data)
