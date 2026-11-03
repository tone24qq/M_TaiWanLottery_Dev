"""Real spawn, artifact and incremental inference regression tests."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import queue
import unittest
from unittest.mock import patch

from taiwan_lottery.ml.bingo_models import ModelSuite
from taiwan_lottery.services.bingo_workers import (
    WorkerManager,
    _INFERENCE_CACHE,
    _load_checkpoint,
    _publish,
    _training,
    infer,
)


def draws(count=110):
    base = datetime(2026, 11, 3, 7, 5)
    return [
        {
            "draw_term": str(115000001 + i),
            "draw_datetime": base + timedelta(minutes=i * 5),
            "numbers": sorted({(i + j * 3) % 80 + 1 for j in range(20)}),
        }
        for i in range(count)
    ]


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.data = draws()
        self.suite = ModelSuite(
            [d["numbers"] for d in self.data[:100]],
            window=100,
            enabled=["frequency", "cdm", "river"],
        ).fit()

    def tearDown(self):
        _INFERENCE_CACHE.clear()
        self.directory.cleanup()

    def test_each_publication_immutable_and_checksum_checked(self):
        old = _publish(self.suite, self.data[:100], self.directory.name, "same-job")
        original = (Path(self.directory.name) / old["file"]).read_bytes()
        self.suite.records["frequency"]["samples"] = 101
        new = _publish(self.suite, self.data[:100], self.directory.name, "same-job")
        self.assertNotEqual(old["file"], new["file"])
        self.assertEqual(
            (Path(self.directory.name) / old["file"]).read_bytes(), original
        )
        _load_checkpoint(old, self.data, self.directory.name)
        (Path(self.directory.name) / new["file"]).write_bytes(b"corrupt")
        result = infer(self.data[:100], self.directory.name, 1, 100)
        self.assertEqual(set(result["scores"]), {"frequency", "cdm"})
        self.assertEqual(result["version"], "baseline")
        self.assertIn("校驗失敗", result["warning"])

    def test_online_update_once_survives_restart_and_backward_query(self):
        training = _publish(self.suite, self.data[:100], self.directory.name, "job")
        first = infer(self.data[:101], self.directory.name, 1, 100)
        online_path = Path(self.directory.name) / "online.json"
        online = json.loads(online_path.read_text())
        self.assertEqual(online["source_term"], self.data[100]["draw_term"])
        self.assertEqual(online["base_sha256"], training["sha256"])
        # A repeated score must do zero updates. It reuses the in-process suite.
        with patch.object(
            ModelSuite, "update", side_effect=AssertionError("duplicate online update")
        ):
            again = infer(self.data[:101], self.directory.name, 1, 100)
        self.assertEqual(first["scores"], again["scores"])
        _INFERENCE_CACHE.clear()
        with patch.object(
            ModelSuite, "update", side_effect=AssertionError("replayed after restart")
        ):
            recovered = infer(self.data[:101], self.directory.name, 1, 100)
        self.assertEqual(recovered["scores"], first["scores"])
        # Sliding-window caller omits a training row; suite retains its own past.
        advanced = infer(self.data[1:102], self.directory.name, 1, 100)
        self.assertIn("frequency", advanced["scores"])
        if "river" in first["scores"]:
            self.assertIn("river", advanced["scores"])
            river = next(x for x in advanced["models"] if x["name"] == "river")
            initial = next(x for x in first["models"] if x["name"] == "river")
            self.assertEqual(river["samples"], initial["samples"] + 1)
        newer = online_path.read_bytes()
        historical = infer(self.data[:100], self.directory.name, 1, 100)
        self.assertIn("frequency", historical["scores"])
        self.assertEqual(online_path.read_bytes(), newer)

    def test_partial_retraining_preserves_previous_healthy_scorers(self):
        _publish(self.suite, self.data[:100], self.directory.name, "prior")
        publications = []
        original_publish = _publish

        def capture(suite, *args, **kwargs):
            publications.append(set(suite.models))
            return original_publish(suite, *args, **kwargs)

        with patch(
            "taiwan_lottery.services.bingo_workers._publish", side_effect=capture
        ):
            _training(
                self.data[:101],
                self.directory.name,
                1,
                100,
                "new",
                queue.Queue(),
                enabled=["frequency", "cdm", "river"],
            )
        self.assertTrue(publications)
        self.assertIn("cdm", publications[0])
        if "river" in self.suite.models:
            self.assertIn("river", publications[0])
        result = infer(self.data[:101], self.directory.name, 1, 100)
        self.assertEqual(result["warning"], None)

    def test_future_checkpoint_never_used(self):
        _publish(self.suite, self.data[:100], self.directory.name, "future")
        result = infer(self.data[:80], self.directory.name, 1, 100)
        self.assertEqual(result["version"], "baseline")
        self.assertEqual(set(result["scores"]), {"frequency", "cdm"})


class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.worker = WorkerManager(self.directory.name, threads=1, window=100)

    async def asyncTearDown(self):
        await self.worker.close()
        self.directory.cleanup()

    async def test_actual_spawn_training_and_inference_round_trip(self):
        data = draws(40)
        started = await self.worker.start(
            "train", data, {"enabled": ["frequency", "cdm"]}
        )
        self.assertEqual(started["state"], "running")
        with self.assertRaisesRegex(ValueError, "執行中"):
            await self.worker.start("train", data)
        await asyncio.wait_for(self.worker.monitor, timeout=30)
        self.assertEqual(
            self.worker.training["state"], "completed", self.worker.training
        )
        self.assertFalse(self.worker.busy)
        self.assertTrue((Path(self.directory.name) / "current.json").exists())
        result = await asyncio.wait_for(self.worker.score(data), timeout=30)
        self.assertEqual(set(result["scores"]), {"frequency", "cdm"})
        self.assertTrue(all(len(values) == 80 for values in result["scores"].values()))
        self.assertNotEqual(result["version"], "baseline")

    async def test_stop_is_terminal_and_next_job_can_start(self):
        await self.worker.start("train", draws(100))
        await self.worker.stop()
        self.assertEqual(self.worker.training["state"], "stopped")
        self.assertFalse(self.worker.busy)
        await self.worker.start("train", draws(20), {"enabled": ["frequency", "cdm"]})
        await asyncio.wait_for(self.worker.monitor, timeout=30)
        self.assertEqual(self.worker.training["state"], "completed")
        await self.worker.close()
        with self.assertRaisesRegex(ValueError, "關閉"):
            await self.worker.score(draws(20))
