"""HTTP contract and runtime ownership tests without upstream network calls."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import httpx
from sqlalchemy.ext.asyncio import create_async_engine

from taiwan_lottery.main import app
from taiwan_lottery.services.runtime_lock import RuntimeLock


class ControlledRuntime:
    """Explicit API test double; production source and scorers remain real."""

    def __init__(self):
        self.sync = dict(state="connected", last_success="2026-11-03T11:00:00+08:00")
        self.workers = SimpleNamespace(
            busy=True,
            training=dict(
                state="training",
                current_model="transformer",
                completed=3,
                total=6,
                progress=50,
            ),
            backtest=dict(state="completed", report={"results": [{"test_only": True}]}),
        )
        self.options = None
        self.prediction_started = asyncio.Event()
        self.release_prediction = asyncio.Event()
        self.block_prediction = False

    def status(self):
        return dict(
            sync=self.sync, training=self.workers.training, models=[], latest=None
        )

    async def predict(self):
        if self.block_prediction:
            self.prediction_started.set()
            await self.release_prediction.wait()
        return dict(
            target_term="115000101",
            by_star={str(k): dict(numbers=list(range(1, k + 1))) for k in (3, 4, 5)},
            settled=False,
        )

    async def start_training(self):
        if self.workers.busy:
            raise ValueError("背景工作執行中，請等待完成或停止訓練")
        return dict(state="running")

    async def start_backtest(self, options):
        self.options = options
        return dict(state="running", run_id="http-test")


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous = getattr(app.state, "bingo", None)
        self.runtime = ControlledRuntime()
        app.state.bingo = self.runtime
        # ASGITransport intentionally does not run lifespan: these are route
        # contract tests, while runtime tests exercise real SQL lifecycle.
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def asyncTearDown(self):
        self.runtime.release_prediction.set()
        await self.client.aclose()
        if self.previous is None:
            del app.state.bingo
        else:
            app.state.bingo = self.previous

    async def test_only_bingo_routes_and_three_star_sizes_exposed(self):
        for path in (
            "/api/v1/lotto649/",
            "/api/v1/super_lotto/",
            "/api/v1/daily_cash/",
            "/api/v1/ml/predict",
            "/analysis/lotto649",
            "/history/daily_cash",
        ):
            response = await self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
        response = await self.client.post("/api/v1/bingo/predict")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["by_star"]), {"3", "4", "5"})
        self.assertEqual(
            [len(response.json()["by_star"][str(k)]["numbers"]) for k in (3, 4, 5)],
            [3, 4, 5],
        )
        schema = (await self.client.get("/openapi.json")).json()
        self.assertTrue(
            all(
                path.startswith("/api/v1/bingo") or path == "/health"
                for path in schema["paths"]
            )
        )

    async def test_backtest_validation_rejects_unbounded_or_invalid_ranges(self):
        invalid = [
            dict(count=0),
            dict(count=10001),
            dict(start_term="abc"),
            dict(start_term="115000200", end_term="115000100"),
        ]
        for payload in invalid:
            response = await self.client.post("/api/v1/bingo/backtest", json=payload)
            self.assertEqual(response.status_code, 422, payload)
        response = await self.client.post(
            "/api/v1/bingo/backtest",
            json=dict(count=300, start_term="115000001", end_term="115000300"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.runtime.options["count"], 300)
        self.assertEqual(self.runtime.options["start_term"], "115000001")

    async def test_history_query_validation_happens_before_database_work(self):
        # Reject invalid requests without touching any external database.
        for query in ("page=0", "page_size=101", "page_size=0", "date_from=not-a-date"):
            response = await self.client.get("/api/v1/bingo/draws?" + query)
            self.assertEqual(response.status_code, 422, query)

    async def test_duplicate_training_returns_conflict_and_summary_omits_large_report(
        self,
    ):
        response = await self.client.post("/api/v1/bingo/train")
        self.assertEqual(response.status_code, 409)
        self.assertIn("執行中", response.json()["detail"])
        response = await self.client.get("/api/v1/bingo/backtest?summary=true")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["has_report"])
        self.assertNotIn("report", response.json())

    async def test_status_and_health_respond_while_prediction_is_pending(self):
        self.runtime.block_prediction = True
        pending = asyncio.create_task(self.client.post("/api/v1/bingo/predict"))
        try:
            await asyncio.wait_for(self.runtime.prediction_started.wait(), 1)
            response = await asyncio.wait_for(
                self.client.get("/api/v1/bingo/status"), 0.5
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["training"]["progress"], 50)
            health = await asyncio.wait_for(self.client.get("/health"), 0.5)
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["sync_state"], "connected")
            self.assertFalse(pending.done())
        finally:
            self.runtime.release_prediction.set()
            await pending


class RuntimeOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_database_cannot_have_two_owners_even_with_different_artifact_paths(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bingo.db"
            first_engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
            second_engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
            first = RuntimeLock(first_engine, Path(directory) / "one")
            second = RuntimeLock(second_engine, Path(directory) / "two")
            try:
                await first.acquire()
                with self.assertRaisesRegex(RuntimeError, "單一 Uvicorn worker"):
                    await second.acquire()
                await first.close()
                await second.acquire()
            finally:
                await first.close()
                await second.close()
                await first_engine.dispose()
                await second_engine.dispose()
