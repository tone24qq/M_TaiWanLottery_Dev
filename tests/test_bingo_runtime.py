"""End-to-end coordinator tests with real SQL persistence.

Only external draw delivery, wall clock and expensive scoring are controlled.
Synthetic numbers are confined to these tests; production ingestion is untouched.
"""

import asyncio
import copy
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from taiwan_lottery.bingo_clock import TAIPEI
from taiwan_lottery.config import settings
from taiwan_lottery.db.models.bingo_research import BingoForecast, BingoPredictionBundle
from taiwan_lottery.services.bingo_runtime import BingoRuntime
from taiwan_lottery.services.bingo_workers import snapshot_hash


class ControlledClock:
    def __init__(self, when):
        self.when = when.replace(tzinfo=TAIPEI)

    def __call__(self):
        return self.when


class ControlledSource:
    def __init__(self, rows):
        self.rows = rows
        self.fail_days = set()
        self.calls = []

    async def fetch_day(self, day):
        self.calls.append(day)
        if day in self.fail_days:
            raise ConnectionError("controlled upstream outage")
        return copy.deepcopy(
            [row for row in self.rows if row["draw_datetime"].date() == day]
        )


class CheapWorker:
    """Explicit scoring stub; exercises real runtime fusion and SQL archiving."""

    def __init__(self):
        self.models = []
        self.training = dict(state="waiting", progress=0)
        self.backtest = dict(state="waiting", report=None)
        self.busy = False
        self.calls = []
        self.mutate_input = False
        self.include_bad_model = False

    async def score(self, rows):
        self.calls.append(copy.deepcopy(rows))
        result = dict(
            scores={
                "frequency": list(range(80, 0, -1)),
                "cdm": [float((i * 7) % 13) for i in range(80)],
            },
            snapshot_hash=snapshot_hash(rows),
            version="test-snapshot-v1",
        )
        if self.include_bad_model:
            result["scores"]["broken"] = [float("nan")] * 80
        if self.mutate_input:
            rows[0]["numbers"][0] = 80
        return result

    async def close(self):
        pass


class RuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.options = patch.multiple(
            settings,
            SCRAPER_ENABLED=False,
            AUTO_TRAIN=False,
            HISTORY_MIN_DRAWS=10,
            MAX_BACKFILL_DAYS=3,
        )
        self.options.start()
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.last_time = datetime(2026, 11, 3, 9)
        self.rows = [
            self.draw(115000100 + i, self.last_time - timedelta(minutes=5 * (9 - i)))
            for i in range(10)
        ]
        self.clock = ControlledClock(self.last_time + timedelta(seconds=15))
        self.source = ControlledSource(self.rows)
        self.worker = CheapWorker()
        self.runtime = BingoRuntime(
            self.sessions, source=self.source, clock=self.clock, workers=self.worker
        )
        await self.runtime.initialize()

    async def asyncTearDown(self):
        await self.runtime.close()
        await self.engine.dispose()
        self.options.stop()

    @staticmethod
    def draw(term, when):
        return dict(draw_term=str(term), draw_datetime=when, numbers=list(range(1, 21)))

    async def bundles(self):
        async with self.sessions() as session:
            return (
                (
                    await session.execute(
                        select(BingoPredictionBundle.target_term).order_by(
                            BingoPredictionBundle.target_term
                        )
                    )
                )
                .scalars()
                .all()
            )

    async def test_three_new_draws_complete_continuous_loop_and_restart(self):
        await self.runtime.cycle()
        self.assertEqual(self.runtime.prediction["target_term"], "115000110")
        self.assertFalse(self.runtime.prediction["settled"])
        for i in range(1, 4):
            target = 115000109 + i
            at = self.last_time + timedelta(minutes=i * 5)
            self.rows.append(self.draw(target, at))
            self.clock.when = (at + timedelta(seconds=15)).replace(tzinfo=TAIPEI)
            await self.runtime.cycle()
            self.assertEqual(self.runtime.prediction["target_term"], str(target + 1))
            self.assertEqual(len(self.runtime.outcomes), i * 6 * 3)
            settled = await self.runtime.store.bundle(str(target))
            self.assertTrue(settled["settled"])
            self.assertEqual(len(settled["outcomes"]), 18)
            for stars in ("3", "4", "5"):
                self.assertEqual(
                    {
                        row["samples"]
                        for row in self.runtime.boards[stars]["cumulative"]
                    },
                    {i},
                )
                self.assertEqual(
                    {
                        row["samples"]
                        for row in self.runtime.boards[stars]["consecutive"]
                    },
                    {i},
                )
        before = copy.deepcopy(self.runtime.prediction)
        outcomes = copy.deepcopy(self.runtime.outcomes)
        await self.runtime.cycle()
        self.assertEqual(self.runtime.prediction, before)
        self.assertEqual(self.runtime.outcomes, outcomes)
        self.assertEqual(len(self.worker.calls), 4)
        self.assertEqual(
            await self.bundles(), [str(t) for t in range(115000110, 115000114)]
        )
        restored = BingoRuntime(
            self.sessions, source=self.source, clock=self.clock, workers=CheapWorker()
        )
        await restored.initialize()
        self.assertEqual(restored.prediction, before)
        self.assertEqual(restored.outcomes, outcomes)
        await restored.close()

    async def test_outage_recovery_settles_only_archived_forecast_no_backdating(self):
        await self.runtime.cycle()
        first = copy.deepcopy(self.runtime.prediction)
        self.source.fail_days.add(self.last_time.date())
        self.clock.when += timedelta(minutes=5)
        await self.runtime.cycle()
        self.assertIn(self.runtime.sync["state"], ("disconnected", "degraded"))
        self.assertIn("controlled upstream outage", self.runtime.sync["error"])
        self.assertEqual(self.runtime.prediction, first)
        self.source.fail_days.clear()
        for i in range(1, 4):
            self.rows.append(
                self.draw(115000109 + i, self.last_time + timedelta(minutes=5 * i))
            )
        self.clock.when = (self.last_time + timedelta(minutes=15, seconds=15)).replace(
            tzinfo=TAIPEI
        )
        await self.runtime.cycle()
        self.assertEqual(self.runtime.sync["state"], "connected")
        self.assertEqual(len(self.runtime.snapshot), 13)
        self.assertEqual(await self.bundles(), ["115000110", "115000113"])
        self.assertEqual(
            {row["target_term"] for row in self.runtime.outcomes}, {"115000110"}
        )
        self.assertEqual(len(self.runtime.outcomes), 18)
        self.rows.append(self.draw(115000113, self.last_time + timedelta(minutes=20)))
        self.clock.when += timedelta(minutes=5)
        await self.runtime.cycle()
        for stars in ("3", "4", "5"):
            self.assertEqual(
                {r["samples"] for r in self.runtime.boards[stars]["cumulative"]}, {2}
            )
            self.assertEqual(
                {r["samples"] for r in self.runtime.boards[stars]["consecutive"]}, {1}
            )

    async def test_delayed_results_do_not_create_expired_target_forecast(self):
        self.clock.when += timedelta(minutes=10)
        await self.runtime.cycle()
        self.assertEqual(self.runtime.sync["state"], "delayed")
        self.assertIsNone(self.runtime.prediction)
        self.assertEqual(self.worker.calls, [])
        self.assertEqual(await self.bundles(), [])
        self.assertGreaterEqual(self.runtime.status()["sync"]["delay_seconds"], 600)

    async def test_prediction_snapshot_is_detached_and_repeat_is_immutable(self):
        self.worker.mutate_input = True
        await self.runtime.cycle()
        expected = snapshot_hash(self.runtime.snapshot)
        self.assertEqual(self.runtime.prediction["snapshot_hash"], expected)
        self.assertEqual(self.runtime.snapshot[0]["numbers"], list(range(1, 21)))
        self.assertEqual(
            (await self.runtime.store.history())[0]["numbers"], list(range(1, 21))
        )
        first = copy.deepcopy(self.runtime.prediction)
        repeated = await self.runtime.predict()
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.worker.calls), 1)
        async with self.sessions() as session:
            hashes = (
                (await session.execute(select(BingoForecast.snapshot_hash)))
                .scalars()
                .all()
            )
        self.assertEqual(set(hashes), {expected})

    async def test_concurrent_cycles_do_not_duplicate_scoring_or_archiving(self):
        await asyncio.gather(
            self.runtime.cycle(), self.runtime.cycle(), self.runtime.predict()
        )
        self.assertEqual(len(self.worker.calls), 1)
        self.assertEqual(await self.bundles(), ["115000110"])

    async def test_historical_failure_does_not_block_current_result_lifecycle(self):
        await self.runtime.cycle()
        self.rows.append(self.draw(115000110, self.last_time + timedelta(minutes=5)))
        self.clock.when += timedelta(minutes=5)
        self.source.fail_days.add(self.last_time.date() - timedelta(days=1))
        # The latest day is available; an older backfill outage must not prevent
        # settling the archived target or forecasting the next future target.
        try:
            await self.runtime.cycle()
        except ConnectionError:
            pass
        self.assertEqual(self.runtime.prediction["target_term"], "115000111")
        self.assertTrue((await self.runtime.store.bundle("115000110"))["settled"])
        self.assertEqual(len(self.runtime.outcomes), 18)
        self.assertTrue(self.runtime.sync.get("error"))

    async def test_one_broken_model_does_not_block_healthy_models(self):
        self.worker.include_bad_model = True
        await self.runtime.cycle()
        self.assertEqual(self.runtime.prediction["target_term"], "115000110")
        async with self.sessions() as session:
            names = (
                (await session.execute(select(BingoForecast.model).distinct()))
                .scalars()
                .all()
            )
        self.assertNotIn("broken", names)
        self.assertIn("frequency", names)
        self.assertIn("shrink", names)

    async def test_gap_backfill_does_not_rewrite_already_archived_prediction(self):
        missing = self.rows.pop(4)
        # Keep >=10 training observations while introducing an interior gap.
        earlier = self.draw(
            115000099, self.rows[0]["draw_datetime"] - timedelta(minutes=5)
        )
        self.rows.insert(0, earlier)
        await self.runtime.cycle()
        self.assertTrue(self.runtime.sync["missing_terms"])
        original = copy.deepcopy(self.runtime.prediction)
        archived_hash = original["snapshot_hash"]
        self.rows.append(missing)
        await self.runtime.cycle()
        self.assertEqual(self.runtime.sync["missing_terms"], [])
        self.assertNotEqual(snapshot_hash(self.runtime.snapshot), archived_hash)
        self.assertEqual(self.runtime.prediction, original)
        self.assertEqual(len(self.worker.calls), 1)
        self.assertEqual(await self.bundles(), ["115000110"])
