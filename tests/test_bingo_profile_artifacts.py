"""Real challenger artifact training, causal restoration and controlled fallback."""

from datetime import datetime, timedelta
from pathlib import Path
import json
from unittest.mock import patch

import numpy as np

from taiwan_lottery.ml.bingo_models import ModelSuite
from taiwan_lottery.ml.bingo_nested import CandidateProfile
from taiwan_lottery.ml.bingo_profile_artifacts import (
    train_validated_profiles,
    score_validated_profile,
    profile_key,
    _CACHE,
)


def draws(n=120):
    random = np.random.default_rng(744)
    return [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=datetime(2026, 1, 1, 7, 5) + timedelta(minutes=5 * i),
            numbers=sorted(
                map(int, random.choice(np.arange(1, 81), 20, replace=False))
            ),
        )
        for i in range(n)
    ]


def test_candidate_is_trained_before_use_and_training_parameters_are_real(tmp_path):
    rows = draws()
    profile = CandidateProfile(
        "candidate",
        60,
        100,
        (0.55, 0.25, 0.15, 0.05),
        24,
        10,
        cdm_prior_kind="recent20",
    )
    assert score_validated_profile(profile, rows[:110], tmp_path)["status"] == "pending"
    result = train_validated_profiles(
        rows[:110], tmp_path, [profile], threads=1, enabled=["frequency", "cdm"]
    )
    assert result["trained_count"] == 1
    scored = score_validated_profile(profile, rows[:110], tmp_path)
    assert scored["status"] == "ready"
    assert scored["metadata"]["training_cutoff"] == rows[109]["draw_term"]
    baseline = (
        ModelSuite([d["numbers"] for d in rows[:110]], enabled=["frequency", "cdm"])
        .fit()
        .scores([d["numbers"] for d in rows[50:110]])
    )
    assert scored["scores"]["frequency"] != baseline["frequency"]
    assert len(scored["scores"]["cdm"]) == 80
    with patch.object(
        ModelSuite, "fit", side_effect=AssertionError("inference/refit before cadence")
    ):
        replay = train_validated_profiles(
            rows[:111], tmp_path, [profile], threads=1, enabled=["frequency", "cdm"]
        )
        assert replay["profiles"][0]["reused"]
        assert (
            score_validated_profile(profile, rows[:111], tmp_path)["status"] == "ready"
        )
    _CACHE.clear()
    with patch.object(
        ModelSuite, "fit", side_effect=AssertionError("restore must never fit")
    ):
        assert (
            score_validated_profile(profile, rows[:111], tmp_path)["status"] == "ready"
        )


def test_corruption_and_future_artifacts_cannot_enter_predictions(tmp_path):
    rows = draws()
    profile = CandidateProfile("bounded", 60, 100)
    train_validated_profiles(
        rows[:110], tmp_path, [profile], threads=1, enabled=["frequency", "cdm"]
    )
    assert score_validated_profile(profile, rows[:80], tmp_path)["status"] == "failed"
    manifest = json.loads(
        (tmp_path / "profiles" / profile_key(profile) / "current.json").read_text()
    )
    (tmp_path / "profiles" / profile_key(profile) / manifest["file"]).write_bytes(
        b"corrupt"
    )
    _CACHE.clear()
    result = score_validated_profile(profile, rows[:110], tmp_path)
    assert result["status"] == "failed" and not result["scores"]
