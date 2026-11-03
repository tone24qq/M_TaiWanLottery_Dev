"""Real transactional SQLite tests for live-only adaptive learning evidence."""

import copy
from datetime import datetime, timedelta
import unittest

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from taiwan_lottery.db.crud.bingo_research import ResearchStore
from taiwan_lottery.db.models.bingo_research import (
    BingoForecast,
    BingoShadowForecast,
    BingoResearchState,
)


class StrategyPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.store = ResearchStore(self.sessions)
        await self.store.init_schema()
        self.when = datetime(2026, 11, 3, 9)
        await self.store.ingest(
            [
                dict(
                    draw_term="115000100",
                    draw_datetime=self.when,
                    numbers=list(range(1, 21)),
                )
            ]
        )
        self.forecasts = [
            dict(
                model="frequency",
                stars=k,
                numbers=list(range(1, k + 1)),
                scores=[i / 79 for i in range(80)],
                version="model-v1",
                method="individual",
            )
            for k in (3, 4, 5)
        ]
        self.metadata = dict(
            training_cutoff="115000100",
            feature_cutoff="115000100",
            history_start="115000084",
            history_end="115000100",
            history_count=17,
            window=17,
            feature_version="causal-v2",
            strategy="fixed",
        )
        self.audit = [
            dict(model="frequency", stars=k, metadata=self.metadata) for k in (3, 4, 5)
        ]
        self.shadows = [
            dict(
                model="frequency",
                stars=k,
                window=w,
                profile=f"window-{w}",
                numbers=list(range(2, k + 2)),
                scores=[i / 79 for i in reversed(range(80))],
                metadata=dict(self.metadata, window=w),
                version="model-v1",
            )
            for w in (17, 1000)
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
            payload={"method": "shrink", "snapshot_hash": "a" * 64},
            forecasts=self.forecasts,
            now=self.when + timedelta(seconds=20),
            audit=self.audit,
            shadows=self.shadows,
        )
        arguments.update(kwargs)
        return await self.store.archive(**arguments)

    async def settle(self):
        await self.store.ingest(
            [
                dict(
                    draw_term="115000101",
                    draw_datetime=self.when + timedelta(minutes=5),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        now = self.when + timedelta(minutes=6)
        await self.store.settle(now=now)
        await self.store.settle_shadow(now=now)

    async def test_atomic_archive_persists_full_audit_and_shadow_and_repeat_is_immutable(
        self,
    ):
        original = await self.archive()
        repeated = await self.archive(
            now=self.when + timedelta(minutes=6),
            shadows=[dict(self.shadows[0], scores=[0.2] * 80)],
        )
        self.assertEqual(original, repeated)
        audit = await self.store.forecast_audit("115000101")
        self.assertEqual(len(audit), 3)
        self.assertEqual(audit[0]["metadata"]["history_count"], 17)
        await self.settle()
        data = await ResearchStore(self.sessions).research_rows()
        target = data["targets"][0]
        self.assertEqual(len(target["live_rows"]), 3)
        self.assertEqual(len(target["shadow_rows"]), 6)
        self.assertTrue(
            all(
                len(r["scores"]) == 80
                for r in target["live_rows"] + target["shadow_rows"]
            )
        )
        self.assertTrue(all(r["is_shadow"] for r in target["shadow_rows"]))
        self.assertEqual({r["window"] for r in target["shadow_rows"]}, {17, 1000})
        self.assertEqual(
            target["live_rows"][0]["metadata"]["training_cutoff"], "115000100"
        )

    async def test_future_audit_or_invalid_shadow_rolls_back_entire_bundle(self):
        audit = copy.deepcopy(self.audit)
        audit[0]["metadata"]["feature_cutoff"] = "115000101"
        with self.assertRaisesRegex(ValueError, "future"):
            await self.archive(audit=audit)
        self.assertIsNone(await self.store.bundle("115000101"))
        with self.assertRaisesRegex(ValueError, "80 finite"):
            await self.archive(shadows=[dict(self.shadows[0], scores=[0.5] * 79)])
        self.assertIsNone(await self.store.bundle("115000101"))
        async with self.sessions() as session:
            self.assertEqual(
                (await session.execute(select(BingoShadowForecast))).scalars().all(), []
            )

    async def test_shadow_settlement_idempotent_and_separate_from_live_rankings(self):
        await self.archive()
        await self.store.ingest(
            [
                dict(
                    draw_term="115000101",
                    draw_datetime=self.when + timedelta(minutes=5),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        self.assertEqual(
            await self.store.settle_shadow(now=self.when + timedelta(minutes=4)), 0
        )
        self.assertEqual(
            await self.store.settle_shadow(now=self.when + timedelta(minutes=6)), 6
        )
        self.assertEqual(
            await self.store.settle_shadow(now=self.when + timedelta(minutes=6)), 0
        )
        self.assertEqual(await self.store.outcome_rows(), [])
        self.assertEqual(
            await self.store.settle(now=self.when + timedelta(minutes=6)), 3
        )
        self.assertEqual(len(await self.store.outcome_rows()), 3)
        summary = await self.store.outcome_summary()
        self.assertTrue(all(r["count"] == 1 for r in summary["cumulative"]))

    async def test_tampered_shadow_or_audit_rejected_before_learning(self):
        await self.archive()
        async with self.sessions() as session, session.begin():
            await session.execute(update(BingoShadowForecast).values(scores=[0.5] * 80))
        await self.store.ingest(
            [
                dict(
                    draw_term="115000101",
                    draw_datetime=self.when + timedelta(minutes=5),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "integrity"):
            await self.store.settle_shadow(now=self.when + timedelta(minutes=6))
        self.assertEqual(
            (await self.store.pending_research_settlements())["targets"], []
        )

    async def test_state_cas_restarts_and_detects_tampering(self):
        self.assertIsNone(await self.store.load_research_state("adaptive"))
        self.assertEqual(
            await self.store.save_research_state(
                "adaptive", {"version": "champion-1"}, expected_version=0
            ),
            1,
        )
        with self.assertRaisesRegex(ValueError, "version conflict"):
            await self.store.save_research_state(
                "adaptive", {"version": "lost"}, expected_version=0
            )
        restored = await ResearchStore(self.sessions).load_research_state("adaptive")
        self.assertEqual(restored["payload"]["version"], "champion-1")
        self.assertEqual(
            await self.store.save_research_state(
                "adaptive", {"version": "champion-2"}, expected_version=1
            ),
            2,
        )
        async with self.sessions() as session, session.begin():
            await session.execute(
                update(BingoResearchState).values(payload={"tampered": True})
            )
        with self.assertRaisesRegex(ValueError, "integrity"):
            await self.store.load_research_state("adaptive")

    async def test_commit_is_atomic_idempotent_persisted_and_excludes_processed_targets(
        self,
    ):
        await self.archive()
        await self.settle()
        pending = await self.store.pending_research_settlements()
        self.assertEqual([r["target_term"] for r in pending["targets"]], ["115000101"])
        states = {
            "adaptive": {"version": "champion-1", "last_target": "115000101"},
            "monitor": {"count": 1},
        }
        diagnostic = {"alerts": [], "target_term": "115000101"}
        events = [
            {
                "kind": "retain",
                "before": {"window": 1000},
                "after": {"window": 1000},
                "validation": {"samples": 1},
                "reason": "insufficient",
            }
        ]
        self.assertTrue(
            await self.store.commit_research_update(
                "115000101", states, diagnostic, events
            )
        )
        self.assertFalse(
            await self.store.commit_research_update(
                "115000101", {"adaptive": {"bad": True}}, diagnostic, events
            )
        )
        self.assertEqual(
            (await self.store.pending_research_settlements())["targets"], []
        )
        self.assertEqual(
            (await self.store.load_research_state("adaptive"))["version"], 1
        )
        self.assertEqual(
            (await self.store.latest_diagnostics())[0]["payload"], diagnostic
        )
        event = (await self.store.strategy_events())[0]
        self.assertEqual(event["kind"], "retain")
        self.assertEqual(event["payload"]["before"], event["payload"]["after"])
        self.assertEqual(len((await self.store.research_rows())["targets"]), 1)

    async def test_unsettled_targets_and_invalid_events_cannot_advance_state(self):
        await self.archive()
        with self.assertRaisesRegex(ValueError, "unsettled"):
            await self.store.commit_research_update(
                "115000101", {"adaptive": {"bad": True}}, {}, []
            )
        self.assertIsNone(await self.store.load_research_state("adaptive"))
        await self.settle()
        with self.assertRaisesRegex(ValueError, "kind"):
            await self.store.commit_research_update(
                "115000101", {"adaptive": {"bad": True}}, {}, [{"kind": ""}]
            )
        self.assertIsNone(await self.store.load_research_state("adaptive"))
        self.assertEqual(await self.store.latest_diagnostics(), [])
        self.assertEqual(
            len((await self.store.pending_research_settlements())["targets"]), 1
        )

    async def test_pending_cursor_keeps_newly_settled_old_target_without_rewinding_state(
        self,
    ):
        import asyncio

        await self.archive()
        await self.store.archive(
            "115000102",
            self.when + timedelta(minutes=10),
            "115000100",
            self.when,
            {},
            self.forecasts,
            now=self.when + timedelta(seconds=30),
        )
        await self.store.ingest(
            [
                dict(
                    draw_term="115000102",
                    draw_datetime=self.when + timedelta(minutes=10),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        await self.store.settle()
        self.assertTrue(
            await self.store.commit_research_update(
                "115000102", {"adaptive": {"version": "newer"}}, {}, []
            )
        )
        self.assertEqual(
            (await self.store.pending_research_settlements())["targets"], []
        )
        await asyncio.sleep(0.002)
        await self.store.ingest(
            [
                dict(
                    draw_term="115000101",
                    draw_datetime=self.when + timedelta(minutes=5),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        await self.store.settle()
        late = (await self.store.pending_research_settlements())["targets"]
        self.assertEqual([target["target_term"] for target in late], ["115000101"])
        with self.assertRaisesRegex(ValueError, "chronological"):
            await self.store.commit_research_update(
                "115000101", {"adaptive": {"version": "older"}}, {}, []
            )
        self.assertTrue(
            await self.store.commit_research_update(
                "115000101",
                {},
                {"late_result": True},
                [{"kind": "late_result_ignored"}],
            )
        )
        self.assertEqual(
            (await self.store.load_research_state("adaptive"))["payload"],
            {"version": "newer"},
        )
        self.assertEqual(
            (await self.store.pending_research_settlements())["targets"], []
        )

    async def test_legacy_v1_forecasts_remain_readable_without_invented_audit(self):
        await self.archive(audit=None, shadows=None)
        await self.settle()
        row = (await self.store.research_rows())["targets"][0]["live_rows"][0]
        self.assertEqual(row["metadata"], {})
        self.assertIsNone(row["window"])
        self.assertEqual(await self.store.forecast_audit("115000101"), [])
        with self.assertRaises(ValueError):
            await self.store.research_rows(limit_terms=0)


def test_real_postgres_v1_upgrade_preserves_ledger_and_enables_shadow_learning():
    """Validate upgrade and new ORM mappings on the actual production dialect."""
    import asyncio
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    import pytest

    pgserver = pytest.importorskip(
        "pgserver", reason="選用 PostgreSQL 整合需要 pgserver==0.1.4"
    )
    Path("work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir="work", prefix="postgres-stage2-"
    ) as directory:
        server = pgserver.get_server(Path(directory) / "pgdata", cleanup_mode="delete")
        uri = server.get_uri()
        environment = {
            **os.environ,
            "DATABASE_URL_SYNC": uri.replace("postgresql://", "postgresql+psycopg2://"),
        }
        async_uri = uri.replace("postgresql://", "postgresql+asyncpg://")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "b90000000001"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            original = asyncio.run(_postgres_legacy_archive(async_uri))
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            asyncio.run(_postgres_stage2_roundtrip(async_uri, original))
        finally:
            server.cleanup()


async def _postgres_legacy_archive(uri):
    engine = create_async_engine(uri)
    store = ResearchStore(async_sessionmaker(engine, expire_on_commit=False))
    at = datetime(2026, 11, 3, 9)
    try:
        await store.ingest(
            [dict(draw_term="115000100", draw_datetime=at, numbers=list(range(1, 21)))]
        )
        return await store.archive(
            "115000101",
            at + timedelta(minutes=5),
            "115000100",
            at,
            {"method": "fixed"},
            [dict(model="frequency", stars=3, numbers=[1, 2, 3], scores=[0.5] * 80)],
            now=at + timedelta(seconds=1),
        )
    finally:
        await engine.dispose()


async def _postgres_stage2_roundtrip(uri, original):
    engine = create_async_engine(uri)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = ResearchStore(sessions)
    at = datetime(2026, 11, 3, 9)
    try:
        await store.init_schema()
        repeated = await store.archive(
            "115000101",
            at + timedelta(minutes=5),
            "115000100",
            at,
            {"wrong": True},
            [],
            now=at + timedelta(minutes=10),
            shadows=[{"invalid": True}],
        )
        assert repeated == original
        assert await store.forecast_audit("115000101") == []
        await store.ingest(
            [
                dict(
                    draw_term="115000101",
                    draw_datetime=at + timedelta(minutes=5),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        assert await store.settle(now=at + timedelta(minutes=6)) == 1
        forecast = dict(
            model="frequency", stars=3, numbers=[1, 2, 3], scores=[0.5] * 80
        )
        metadata = dict(
            training_cutoff="115000101",
            feature_cutoff="115000101",
            history_start="115000100",
            history_end="115000101",
            history_count=2,
            window=17,
        )
        await store.archive(
            "115000102",
            at + timedelta(minutes=10),
            "115000101",
            at + timedelta(minutes=5),
            {},
            [forecast],
            now=at + timedelta(minutes=5, seconds=1),
            audit=[dict(model="frequency", stars=3, metadata=metadata)],
            shadows=[dict(forecast, window=17, profile="window-17", metadata=metadata)],
        )
        await store.ingest(
            [
                dict(
                    draw_term="115000102",
                    draw_datetime=at + timedelta(minutes=10),
                    numbers=list(range(1, 21)),
                )
            ]
        )
        assert await store.settle(now=at + timedelta(minutes=11)) == 1
        assert await store.settle_shadow(now=at + timedelta(minutes=11)) == 1
        targets = (await store.pending_research_settlements())["targets"]
        assert [row["target_term"] for row in targets] == ["115000101", "115000102"]
        assert targets[1]["shadow_rows"][0]["window"] == 17
        assert targets[1]["live_rows"][0]["metadata"]["history_count"] == 2
        assert await store.commit_research_update(
            "115000101", {"adaptive": {"version": "fixed"}}, {"legacy": True}, []
        )
        assert await store.commit_research_update(
            "115000102",
            {"adaptive": {"version": "fixed", "samples": 1}},
            {"warning": "insufficient"},
            [{"kind": "retain", "validation": {"samples": 1}}],
        )
        assert (await store.load_research_state("adaptive"))["version"] == 2
        assert (await store.pending_research_settlements())["targets"] == []
        assert len(await store.outcome_rows()) == 2
    finally:
        await engine.dispose()
