"""Integration tests exercise the real async SQL ledger, not mock persistence."""

import asyncio
import unittest
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from taiwan_lottery.db.crud.bingo_research import ResearchStore
from taiwan_lottery.db.models.bingo_research import BingoForecast, BingoPredictionBundle


class ResearchStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.store = ResearchStore(self.sessions)
        await self.store.init_schema()
        self.when = datetime(2026, 11, 3, 9)
        self.draw = {
            "draw_term": "115000100",
            "draw_datetime": self.when,
            "numbers": list(range(1, 21)),
        }
        await self.store.ingest([self.draw])
        self.forecasts = [
            {
                "model": "frequency",
                "stars": k,
                "numbers": list(range(1, k + 1)),
                "scores": [0.5] * 80,
                "version": "v1",
                "method": "individual",
            }
            for k in (3, 4, 5)
        ]

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def archive(self, **kwargs):
        arguments = dict(
            target_term="115000101",
            target_time=self.when + timedelta(minutes=5),
            source_term="115000100",
            source_time=self.when,
            payload={"method": "shrink", "by_star": {}},
            forecasts=self.forecasts,
            now=self.when + timedelta(seconds=20),
        )
        arguments.update(kwargs)
        return await self.store.archive(**arguments)

    async def test_duplicate_prediction_and_settlement_survive_restart(self):
        first = await self.archive()
        repeated = await self.archive(
            payload={"method": "equal"}, now=self.when + timedelta(minutes=10)
        )
        self.assertEqual(first, repeated)
        self.assertEqual(await self.store.ingest([self.draw]), 0)
        result = {
            "draw_term": "115000101",
            "draw_datetime": self.when + timedelta(minutes=5),
            "numbers": list(range(1, 21)),
        }
        await self.store.ingest([result])
        self.assertEqual(
            await self.store.settle(now=self.when + timedelta(minutes=4)), 0
        )
        self.assertEqual(
            await self.store.settle(now=self.when + timedelta(minutes=6)), 3
        )
        restarted = ResearchStore(self.sessions)
        self.assertEqual(
            await restarted.settle(now=self.when + timedelta(minutes=10)), 0
        )
        outcomes = await restarted.outcome_rows()
        self.assertEqual([x["hit_count"] for x in outcomes], [3, 4, 5])
        bundle = await restarted.bundle()
        self.assertTrue(bundle["settled"])
        self.assertEqual(bundle["method"], "shrink")

    async def test_reject_late_known_target_and_invalid_snapshot(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            await self.archive(now=self.when + timedelta(minutes=5))
        with self.assertRaisesRegex(ValueError, "Source"):
            await self.archive(source_time=self.when - timedelta(minutes=5))
        await self.store.ingest(
            [
                {
                    "draw_term": "115000101",
                    "draw_datetime": self.when + timedelta(minutes=5),
                    "numbers": list(range(1, 21)),
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "already known"):
            await self.archive()

    async def test_conflicts_reject_whole_batch(self):
        other = dict(
            self.draw,
            draw_term="115000102",
            draw_datetime=self.when + timedelta(minutes=10),
        )
        conflict = dict(self.draw, numbers=list(range(2, 22)))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            await self.store.ingest([other, conflict])
        self.assertEqual(len(await self.store.history()), 1)

    async def test_bundle_and_forecast_tampering_detected(self):
        await self.archive()
        async with self.sessions() as session, session.begin():
            await session.execute(
                update(BingoPredictionBundle).values(payload={"tampered": True})
            )
        with self.assertRaisesRegex(ValueError, "integrity"):
            await self.store.bundle()
        async with self.sessions() as session, session.begin():
            await session.execute(update(BingoForecast).values(numbers=[78, 79, 80]))
        await self.store.ingest(
            [
                {
                    "draw_term": "115000101",
                    "draw_datetime": self.when + timedelta(minutes=5),
                    "numbers": list(range(1, 21)),
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "integrity"):
            await self.store.settle(now=self.when + timedelta(minutes=6))

    async def test_invalid_stars_or_incomplete_scores_rollback_bundle(self):
        bad = dict(self.forecasts[0], stars=6)
        with self.assertRaises(ValueError):
            await self.archive(forecasts=[bad])
        self.assertIsNone(await self.store.bundle())
        with self.assertRaises(ValueError):
            await self.archive(forecasts=[dict(self.forecasts[0], scores=[0.5] * 79)])
        self.assertIsNone(await self.store.bundle())

    async def test_sql_histograms_keep_cumulative_after_gap_and_bound_recent_rows(self):
        await self.archive()
        for index in (101, 102, 103, 104):
            stamp = self.when + timedelta(minutes=5 * (index - 100))
            if index in (103, 104):
                await self.store.archive(
                    str(115000000 + index),
                    stamp,
                    str(115000000 + index - 1),
                    stamp - timedelta(minutes=5),
                    {"method": "shrink"},
                    self.forecasts,
                    now=stamp - timedelta(minutes=4),
                )
            await self.store.ingest(
                [
                    {
                        "draw_term": str(115000000 + index),
                        "draw_datetime": stamp,
                        "numbers": list(range(1, 21)),
                    }
                ]
            )
            await self.store.settle(now=stamp + timedelta(seconds=1))
        summary = await self.store.outcome_summary()
        self.assertEqual(len(summary["cumulative"]), 3)
        self.assertTrue(all(row["count"] == 3 for row in summary["cumulative"]))
        self.assertTrue(all(row["count"] == 2 for row in summary["consecutive"]))
        recent = await self.store.outcome_rows(limit_terms=2)
        self.assertEqual(len(recent), 6)
        self.assertEqual(
            {row["target_term"] for row in recent}, {"115000103", "115000104"}
        )

    async def test_sql_consecutive_histogram_includes_year_rollover(self):
        source = datetime(2026, 12, 31, 23, 50)
        await self.store.ingest(
            [
                {
                    "draw_term": "115074000",
                    "draw_datetime": source,
                    "numbers": list(range(1, 21)),
                }
            ]
        )
        previous_term = "115074000"
        previous_time = source
        for term, stamp in (
            ("115074001", datetime(2026, 12, 31, 23, 55)),
            ("116000001", datetime(2027, 1, 1, 7, 5)),
        ):
            await self.store.archive(
                term,
                stamp,
                previous_term,
                previous_time,
                {"method": "shrink"},
                self.forecasts,
                now=stamp - timedelta(minutes=1),
            )
            await self.store.ingest(
                [
                    {
                        "draw_term": term,
                        "draw_datetime": stamp,
                        "numbers": list(range(1, 21)),
                    }
                ]
            )
            await self.store.settle(now=stamp + timedelta(seconds=1))
            previous_term, previous_time = term, stamp
        summary = await self.store.outcome_summary()
        self.assertTrue(all(row["count"] == 2 for row in summary["consecutive"]))

    async def test_chosen_fusion_weights_restore_without_rewriting_payload(self):
        weights = {"frequency": 0.75, "cdm": 0.25}
        fused = [
            dict(row, model="shrink", method="shrink", weights=weights)
            for row in self.forecasts
        ]
        await self.archive(forecasts=self.forecasts + fused)
        restored = await ResearchStore(self.sessions).bundle("115000101")
        self.assertEqual(
            restored["weights"], {str(stars): weights for stars in (3, 4, 5)}
        )
        async with self.sessions() as session:
            raw = await session.get(BingoPredictionBundle, "115000101")
            self.assertNotIn("weights", raw.payload)

    async def test_backtest_runs_separate_and_persisted(self):
        await self.store.save_evaluation("run1", "running", None)
        await self.store.save_evaluation("run1", "completed", {"samples": 100})
        latest = await ResearchStore(self.sessions).latest_evaluation()
        self.assertEqual(latest["state"], "completed")
        self.assertEqual(latest["report"]["samples"], 100)
        self.assertEqual(await self.store.outcome_rows(), [])
