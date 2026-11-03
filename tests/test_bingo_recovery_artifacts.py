"""Real process recovery and compressed complete research report transport."""

import asyncio
import gzip
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import signal

import numpy as np
import pytest

from taiwan_lottery.services.bingo_workers import WorkerManager


def draws(n=230):
    rng = np.random.default_rng(923)
    return [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=datetime(2026, 10, 20, 7, 5) + timedelta(minutes=5 * i),
            numbers=sorted(map(int, rng.choice(np.arange(1, 81), 20, replace=False))),
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_inference_process_death_recovers_to_real_statistical_scores(tmp_path):
    worker = WorkerManager(tmp_path, threads=1, window=100)
    try:
        first = await asyncio.wait_for(worker.score(draws(40)), 30)
        os.kill(first["pid"], signal.SIGTERM)
        await asyncio.sleep(0.1)
        recovered = await asyncio.wait_for(worker.score(draws(40)), 30)
        assert recovered["pid"] != first["pid"]
        assert worker.inference_quarantined
        assert set(recovered["scores"]) == {"frequency", "cdm"}
        assert all(len(v) == 80 for v in recovered["scores"].values())
        assert recovered["warning"]
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_real_nested_worker_keeps_full_scores_in_gzip_not_http_state(tmp_path):
    worker = WorkerManager(tmp_path, threads=1, window=1000)
    try:
        job = await worker.start(
            "backtest",
            draws(),
            {"mode": "nested", "count": 100, "model_names": ["frequency", "cdm"]},
        )
        await asyncio.wait_for(worker.monitor, 45)
        assert worker.backtest["state"] == "completed", worker.backtest
        report = worker.backtest["report"]
        assert "folds" not in report and "results" not in report
        artifact = tmp_path / "reports" / (job["run_id"] + ".json.gz")
        assert artifact.is_file()
        assert (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            == report["artifact"]["sha256"]
        )
        with gzip.open(artifact, "rt") as stream:
            full = json.load(stream)
        assert len(full["folds"]) == 100
        for fold in full["folds"]:
            assert int(fold["features_through"]) < int(fold["target_term"])
            assert all(
                len(v) == 80 for v in fold["profiles"]["fixed"]["model_scores"].values()
            )
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_today_background_job_rejects_foreign_date_before_spawning(tmp_path):
    worker = WorkerManager(tmp_path, threads=1)
    try:
        with pytest.raises(ValueError, match="其他日期"):
            await worker.start("today_train", draws(20), {"target_date": "2026-10-21"})
        assert worker.process is None
    finally:
        await worker.close()
