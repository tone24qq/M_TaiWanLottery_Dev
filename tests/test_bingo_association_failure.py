"""Optional association research cannot block the original scoring system."""

import gzip
import json
from pathlib import Path
from unittest.mock import patch

from taiwan_lottery.services.bingo_workers import infer


def test_failed_rule_search_preserves_actual_model_and_today_scores(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "docs/validation/official-draws-20261103.json.gz"
    )
    with gzip.open(source, "rt") as stream:
        data = json.load(stream)[:150]
    plan = {"target_date": data[-1]["draw_datetime"][:10]}
    valid = infer(data, tmp_path, 1, 1000, plan)
    assert valid.get("associations"), valid.get("association_error")
    with patch(
        "taiwan_lottery.ml.bingo_associations.association_candidates",
        side_effect=RuntimeError("rule search unavailable"),
    ):
        failed = infer(data, tmp_path, 1, 1000, plan)
    assert failed["scores"] == valid["scores"]
    assert failed["by_star_scores"] == valid["by_star_scores"]
    assert failed["today"]["scores"] == valid["today"]["scores"]
    assert "associations" not in failed
    assert "rule search unavailable" in failed["association_error"]
    assert all(len(vector) == 80 for vector in failed["scores"].values())
