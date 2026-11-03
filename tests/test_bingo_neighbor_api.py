"""HTTP bounds, independent research resources and production-governance guards."""

import asyncio
import copy
from datetime import datetime, timedelta
import multiprocessing as mp
import os
import queue
import struct
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from taiwan_lottery.main import app
from taiwan_lottery.services.bingo_neighbors import NeighborResearchCache
from taiwan_lottery.services.bingo_runtime import BingoRuntime
from taiwan_lottery.services.bingo_workers import _backtest


def _rows(count=1, start=100, delay=0.0):
    return [
        dict(
            draw_term=str(115000000 + start + i),
            draw_datetime=datetime(2026, 11, 3, 8) + timedelta(minutes=5 * i),
            numbers=list(range(1, 21)),
            test_delay=delay,
        )
        for i in range(count)
    ]


def _controlled_worker(rows, channel):
    """Test-only CPU stand-in; process ownership/IPC remain real."""
    time.sleep(rows[0]["test_delay"])
    channel.send({"bundle": {"term": rows[-1]["draw_term"], "samples": len(rows)}})
    channel.close()


def _partial_worker(rows, channel):
    # Test-only truncated IPC frame: the connection is readable but its complete
    # payload never arrives. A wall-clock budget must also cover this receive.
    os.write(channel.fileno(), struct.pack("!i", 1000000))
    time.sleep(30)
    channel.close()


class _RecordingContext:
    def __init__(self):
        self.real = mp.get_context("spawn")
        self.jobs = []

    def Pipe(self, **kwargs):
        return self.real.Pipe(**kwargs)

    def Process(self, **kwargs):
        self.jobs.append(kwargs["args"][0][-1]["draw_term"])
        return self.real.Process(**kwargs)


class NeighborApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous = getattr(app.state, "bingo", None)
        self.runtime = SimpleNamespace(
            neighbors=Mock(return_value=dict(state="running", analysis=None)),
            start_backtest=AsyncMock(return_value=dict(state="running", run_id="test")),
            status=Mock(return_value=dict(sync=dict(state="connected"), models=[])),
        )
        app.state.bingo = self.runtime
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        if self.previous is None:
            del app.state.bingo
        else:
            app.state.bingo = self.previous

    async def test_neighbor_query_has_bounded_parameters(self):
        for query in ("window=999", "lag=4", "snapshot=1", "source=0", "source=81", "lag=abc"):
            response = await self.client.get("/api/v1/bingo/neighbors?" + query)
            self.assertEqual(response.status_code, 422, query)
        self.runtime.neighbors.assert_not_called()

    async def test_pending_analysis_and_status_return_without_waiting_for_statistics(self):
        response = await asyncio.wait_for(self.client.get("/api/v1/bingo/neighbors?window=20&lag=10&snapshot=5&source=80"), 1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "running")
        self.runtime.neighbors.assert_called_once_with(window=20, lag=10, snapshot=5, source=80)
        self.assertEqual((await self.client.get("/api/v1/bingo/status")).status_code, 200)

    async def test_neighbor_backtest_is_a_separate_explicit_mode(self):
        response = await self.client.post("/api/v1/bingo/backtest", json=dict(mode="neighbors", count=300))
        self.assertEqual(response.status_code, 200)
        self.runtime.start_backtest.assert_awaited_once_with(dict(mode="neighbors", count=300))


class NeighborCacheTests(unittest.IsolatedAsyncioTestCase):
    async def _ready(self, cache, rows):
        for _ in range(200):
            if cache._ready_key is not None and cache._bundle["term"] == rows[-1]["draw_term"]:
                return
            if cache._error:
                self.fail(cache._error)
            await asyncio.sleep(0.025)
        self.fail("Controlled research worker did not complete")

    async def test_one_active_job_latest_pending_only_and_selector_changes_do_not_spawn(self):
        context = _RecordingContext()
        cache = NeighborResearchCache(context=context, worker_target=_controlled_worker)
        first, obsolete, latest = _rows(delay=0.15), _rows(start=101), _rows(start=102)
        try:
            self.assertEqual(cache.read(first)["state"], "running")
            for _ in range(5):
                cache.read(first, window=5, lag=20, snapshot=20)
            cache.read(obsolete)
            cache.read(latest)
            self.assertEqual(len(context.jobs), 1)
            await self._ready(cache, latest)
            self.assertEqual(context.jobs, [first[-1]["draw_term"], latest[-1]["draw_term"]])
            with patch("taiwan_lottery.ml.bingo_neighbors.neighbor_view", return_value={"test_only": True}) as view:
                response = cache.read(latest, source=80)
                self.assertEqual(response["state"], "ready")
                self.assertEqual(response["analysis"], {"test_only": True})
                self.assertFalse(response["allowed_for_live"])
                view.assert_called_once()
            self.assertEqual(len(context.jobs), 2)
        finally:
            await cache.close()

    async def test_cache_history_is_capped_and_timestamp_changes_invalidate_snapshot(self):
        context = _RecordingContext()
        cache = NeighborResearchCache(context=context, worker_target=_controlled_worker)
        rows = _rows(1100)
        try:
            original = copy.deepcopy(rows)
            cache.read(rows)
            await self._ready(cache, rows)
            self.assertEqual(cache._bundle["samples"], 1020)
            changed = copy.deepcopy(rows)
            changed[-1]["draw_datetime"] += timedelta(minutes=5)
            self.assertEqual(cache.read(changed)["state"], "running")
            # A completed result can be visible while its child is still being
            # reaped; the replacement starts after that bounded cleanup.
            for _ in range(100):
                if len(context.jobs) == 2:
                    break
                await asyncio.sleep(0.025)
            self.assertEqual(len(context.jobs), 2)
            self.assertEqual(rows, original)
        finally:
            await cache.close()

    async def test_shutdown_terminates_research_without_waiting_for_long_job(self):
        cache = NeighborResearchCache(worker_target=_controlled_worker)
        cache.read(_rows(delay=30))
        process = cache._process
        await asyncio.wait_for(cache.close(), 3)
        self.assertIsNone(cache._process)
        self.assertTrue(cache._closed)
        self.assertEqual(cache.read(_rows())["state"], "waiting")
        self.assertTrue(process._closed)

    async def test_deadline_stops_failed_snapshot_and_polls_do_not_retry_it(self):
        context = _RecordingContext()
        cache = NeighborResearchCache(context=context, worker_target=_controlled_worker, deadline_seconds=0.1)
        rows = _rows(delay=30)
        try:
            cache.read(rows)
            for _ in range(100):
                if cache._failed_key:
                    break
                await asyncio.sleep(0.025)
            response = cache.read(rows)
            self.assertEqual(response["state"], "failed")
            self.assertIn("時間預算", response["reason"])
            cache.read(rows, source=1)
            self.assertEqual(len(context.jobs), 1)
        finally:
            await cache.close()

    @unittest.skipUnless(os.name == "posix", "POSIX pipe framing fault injection")
    async def test_deadline_covers_partial_pipe_payload(self):
        cache = NeighborResearchCache(worker_target=_partial_worker, deadline_seconds=2.0)
        rows = _rows()
        try:
            cache.read(rows)
            for _ in range(200):
                if cache._failed_key:
                    break
                await asyncio.sleep(0.025)
            response = cache.read(rows)
            self.assertEqual(response["state"], "failed")
            self.assertIn("傳輸超出時間預算", response["reason"])
        finally:
            await asyncio.wait_for(cache.close(), 3)


class NeighborGovernanceTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        runtime = object.__new__(BingoRuntime)
        runtime.store = SimpleNamespace(save_evaluation=AsyncMock(), record_validation_update=AsyncMock())
        runtime.adaptive = Mock()
        runtime.strategy = {"method": "shrink", "status": "existing"}
        runtime._last_train_time = "existing"
        return runtime

    async def test_finished_neighbor_research_cannot_spend_alpha_or_change_formal_strategy(self):
        runtime = self.runtime()
        report = dict(mode="neighbor_walk_forward", decision=dict(allowed_for_live=True))
        await runtime._backtest_done(dict(run_id="research", state="completed", report=report))
        runtime.store.save_evaluation.assert_awaited_once()
        runtime.adaptive.accept_validation.assert_not_called()
        runtime.store.record_validation_update.assert_not_awaited()
        self.assertEqual(runtime.strategy, {"method": "shrink", "status": "existing"})
        self.assertEqual(runtime._last_train_time, "existing")

    async def test_restoring_neighbor_report_preserves_production_governance(self):
        runtime = self.runtime()
        self.assertEqual(runtime._apply_strategy(dict(mode="neighbor_walk_forward"), restore=True), {"events": []})
        self.assertEqual(runtime.strategy["status"], "existing")
        runtime.adaptive.accept_validation.assert_not_called()

    async def test_starting_neighbor_backtest_does_not_advance_formal_validation_cooldown(self):
        runtime = self.runtime()
        rows = _rows(120)
        runtime.store.history = AsyncMock(return_value=rows)
        runtime.store.save_research_state = AsyncMock()
        runtime.workers = SimpleNamespace(busy=False, start=AsyncMock(return_value={"run_id": "research"}))
        runtime.clock = lambda: datetime(2026, 11, 3, 12)
        runtime._validation_at = "existing-time"
        runtime._validation_source = "existing-source"
        await runtime.start_backtest(dict(mode="neighbors", count=1))
        runtime.store.save_research_state.assert_not_awaited()
        self.assertEqual(runtime._validation_at, "existing-time")
        self.assertEqual(runtime._validation_source, "existing-source")
        self.assertEqual(runtime.workers.start.await_args.args[0], "backtest")


class NeighborDispatchTests(unittest.TestCase):
    def test_large_existing_training_window_is_capped_for_neighbor_research(self):
        messages = queue.Queue()
        report = dict(mode="neighbor_walk_forward", count=100,
                      decision=dict(allowed_for_live=False))
        rows = _rows()
        with patch("taiwan_lottery.ml.bingo_neighbor_evaluation.neighbor_walk_forward", return_value=report) as replay:
            _backtest(rows, dict(mode="neighbors", count=100), 1, 10000, messages)
        self.assertEqual(replay.call_count, 1)
        self.assertEqual(replay.call_args.kwargs["window"], 1000)
        self.assertEqual(replay.call_args.args[0], rows)
        self.assertEqual(replay.call_args.kwargs["count"], 100)
        completed = messages.get_nowait()
        self.assertEqual(completed["type"], "done")
        self.assertEqual(completed["report"]["requested_mode"], "neighbors")
        self.assertFalse(completed["report"]["decision"]["allowed_for_live"])
