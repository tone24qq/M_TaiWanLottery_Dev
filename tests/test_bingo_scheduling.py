"""Scheduler regressions with real SQLite history and durable manifests.

Only CPU-job launch, external source delivery and the clock are controlled.
"""

import copy
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from taiwan_lottery.bingo_clock import TAIPEI, now_taipei
from taiwan_lottery.config import settings
from taiwan_lottery.ml.bingo_models import ModelSuite
from taiwan_lottery.services.bingo_runtime import BingoRuntime
from taiwan_lottery.services.bingo_workers import _publish


class Clock:
    def __init__(self):
        self.when = now_taipei()

    def __call__(self):
        return self.when


class ControlledWorker:
    def __init__(self, directory):
        self.directory = str(directory)
        self.busy = False
        self.training = {"state": "waiting"}
        self.backtest = {"state": "waiting", "report": None}
        self.models = []
        self.calls = []
        self.stops = 0

    async def start(self, kind, draws, options=None):
        self.calls.append((kind, copy.deepcopy(draws), copy.deepcopy(options)))
        state = {"state": "running", "run_id": f"regression-{len(self.calls)}"}
        if kind == "train":
            # Simulate a completed CPU job, allowing scheduler decisions to be
            # inspected without creating additional multiprocessing workers.
            self.training.update(state="completed")
        return state

    async def stop(self):
        self.stops += 1
        self.training["state"] = "stopped"

    async def close(self):
        pass


class ControlledSource:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []

    async def fetch_day(self, day):
        self.calls.append(day)
        return [
            copy.deepcopy(row)
            for row in self.rows
            if row["draw_datetime"].date() == day
        ]


def rows(count, *, first=115000001, first_day=date(2026, 9, 1)):
    result = []
    for index in range(count):
        day = first_day + timedelta(days=index // 203)
        when = datetime.combine(day, datetime.min.time()) + timedelta(
            hours=7, minutes=5 + 5 * (index % 203)
        )
        result.append(
            {
                "draw_term": str(first + index),
                "draw_datetime": when,
                "numbers": list(range(1, 21)),
            }
        )
    return result


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.worker = ControlledWorker(Path(self.directory.name) / "artifacts")
        self.source = ControlledSource()
        self.clock = Clock()
        self.options = patch.multiple(
            settings,
            SCRAPER_ENABLED=False,
            AUTO_TRAIN=True,
            HISTORY_MIN_DRAWS=10,
            TRAIN_WINDOW=1000,
            RETRAIN_MIN_DRAWS=36,
            RETRAIN_INTERVAL_SECONDS=10800,
            MAX_BACKFILL_DAYS=31,
        )
        self.options.start()
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.runtime = BingoRuntime(
            self.sessions, source=self.source, clock=self.clock, workers=self.worker
        )
        await self.runtime.initialize()

    async def asyncTearDown(self):
        await self.runtime.close()
        await self.engine.dispose()
        self.options.stop()
        self.directory.cleanup()

    async def seed(self, data):
        await self.runtime.store.ingest(data)
        self.runtime.snapshot = await self.runtime.store.history(limit=10000)

    async def test_restart_restores_ready_training_cursor_without_immediate_refit(self):
        data = rows(120)
        await self.seed(data)
        suite = ModelSuite(
            [d["numbers"] for d in data], enabled=["frequency", "cdm"]
        ).fit()
        _publish(suite, data, self.worker.directory, "ready-before-restart")
        restored_worker = ControlledWorker(self.worker.directory)
        restored = BingoRuntime(
            self.sessions, source=self.source, clock=self.clock, workers=restored_worker
        )
        try:
            await restored.initialize()
            self.assertEqual(restored._last_train_term, data[-1]["draw_term"])
            self.assertTrue(restored_worker.training["restored"])
            await restored._maybe_train()
            self.assertEqual(restored_worker.calls, [])
        finally:
            await restored.close()

    async def test_manual_stop_pauses_automatic_restarts_until_explicit_start(self):
        await self.seed(rows(120))
        await self.runtime.start_training()
        await self.runtime.stop_training()
        self.clock.when += timedelta(hours=4)
        await self.runtime._maybe_train()
        self.assertEqual(len(self.worker.calls), 1)
        self.assertEqual(self.worker.training["state"], "stopped")
        self.assertEqual(self.worker.stops, 1)
        await self.runtime.start_training()
        self.assertFalse(self.runtime._auto_paused)
        self.clock.when += timedelta(hours=4)
        await self.runtime._maybe_train()
        # A clock interval alone must not refit an unchanged data snapshot.
        self.assertEqual(len(self.worker.calls), 2)
        await self.seed(rows(121)[-1:])
        await self.runtime._maybe_train()
        self.assertEqual(len(self.worker.calls), 3)

    async def test_retrain_trigger_counts_new_terms_after_history_hits_10000_cap(self):
        data = rows(10036)
        await self.seed(data[:10000])
        await self.runtime.start_training()
        self.assertEqual(len(self.runtime.snapshot), 10000)
        first_training_term = self.runtime._last_train_term
        await self.seed(data[10000:])
        self.assertEqual(len(self.runtime.snapshot), 10000)
        self.assertNotEqual(self.runtime.snapshot[-1]["draw_term"], first_training_term)
        await self.runtime._maybe_train()
        self.assertEqual(len(self.worker.calls), 2)
        self.assertEqual(
            self.worker.calls[-1][1][-1]["draw_term"], data[-1]["draw_term"]
        )

    async def test_custom_range_backfills_100_past_draws_before_first_target(self):
        self.clock.when = datetime(2026, 11, 3, 12, tzinfo=TAIPEI)
        preceding = rows(203, first_day=date(2026, 11, 1))
        targets = rows(200, first=115000204, first_day=date(2026, 11, 2))
        self.source.rows = preceding
        await self.seed(targets)
        # There are already count+100 rows, yet none precedes start_term. Total
        # size alone must not allow a causally invalid custom-range blind test.
        options = dict(
            count=100,
            start_term=targets[0]["draw_term"],
            end_term=targets[99]["draw_term"],
        )
        result = await self.runtime.start_backtest(options)
        self.assertEqual(self.source.calls, [date(2026, 11, 1)])
        kind, submitted, submitted_options = self.worker.calls[-1]
        self.assertEqual(kind, "backtest")
        self.assertEqual(submitted_options, options)
        prefix = [
            draw
            for draw in submitted
            if int(draw["draw_term"]) < int(options["start_term"])
        ]
        self.assertGreaterEqual(len(prefix), 100)
        persisted = await self.runtime.store.latest_evaluation()
        self.assertEqual(persisted["run_id"], result["run_id"])
        self.assertEqual(persisted["state"], "running")

    async def test_failed_training_retry_waits_at_least_60_seconds(self):
        await self.seed(rows(120))
        await self.runtime.start_training()
        self.worker.training["state"] = "failed"
        self.clock.when += timedelta(seconds=59)
        await self.runtime._maybe_train()
        self.assertEqual(len(self.worker.calls), 1)
        self.clock.when += timedelta(seconds=1)
        await self.runtime._maybe_train()
        self.assertEqual(len(self.worker.calls), 2)

    async def test_failed_today_job_retries_today_without_retraining_historical(self):
        data = rows(120)
        await self.seed(data)
        self.worker.adaptive_scoring = True
        await self.runtime.start_training()
        historical_term = self.runtime._last_train_term
        historical_time = self.runtime._last_train_time
        # Historical is still current. A failed Today job must repair only the
        # date-scoped artifact, even when Historical's old60s retry age passed.
        self.clock.when += timedelta(minutes=5)
        day = data[-1]["draw_datetime"].date().isoformat()
        self.worker.training.update(state="failed", scope="today", target_date=day)
        await self.runtime._maybe_train()
        self.assertEqual(
            [call[0] for call in self.worker.calls], ["train", "today_train"]
        )
        self.assertEqual(self.runtime._last_train_term, historical_term)
        self.assertEqual(self.runtime._last_train_time, historical_time)
        self.assertEqual(self.worker.calls[-1][2], {"target_date": day})
        self.assertTrue(
            all(
                row["draw_datetime"].date().isoformat() == day
                for row in self.worker.calls[-1][1]
            )
        )
        # A repeatedly failing day model has the same bounded60s retry policy,
        # rather than launching again on each45s source poll.
        self.clock.when += timedelta(seconds=59)
        await self.runtime._maybe_train()
        self.assertEqual(len(self.worker.calls), 2)
        self.clock.when += timedelta(seconds=1)
        await self.runtime._maybe_train()
        self.assertEqual(
            [call[0] for call in self.worker.calls],
            ["train", "today_train", "today_train"],
        )

    async def test_historical_repair_backoff_is_not_overwritten_by_today_launch(self):
        await self.seed(rows(120))
        self.worker.adaptive_scoring = True
        await self.runtime.start_training()
        self.worker.training.update(state="failed", scope="historical")
        self.clock.when += timedelta(seconds=59)
        await self.runtime._maybe_train()
        self.assertEqual([call[0] for call in self.worker.calls], ["train"])
        self.clock.when += timedelta(seconds=1)
        await self.runtime._maybe_train()
        self.assertEqual([call[0] for call in self.worker.calls], ["train", "train"])
