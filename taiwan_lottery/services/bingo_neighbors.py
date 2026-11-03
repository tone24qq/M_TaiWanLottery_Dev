"""One lazy, low-priority neighbor research process, separate from live workers.

Only the application's verified in-memory draws cross this process boundary. The
cache never opens a database or production model artifact. A full compact bundle
is computed once per source snapshot; HTTP requests only select frozen counts.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import multiprocessing as mp
import os
import time


WINDOWS = (5, 10, 20, 30, 50, 100, 300, 500, 1000)
LAGS = (1, 2, 3, 5, 10, 20)
SNAPSHOTS = (0, 5, 10, 20)
MAX_HISTORY = 1020


def _bundle_worker(draws, channel):
    """Child-only imports keep statistics and numerical work off the HTTP loop."""
    try:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        try:
            os.nice(10)
        except (AttributeError, OSError):
            pass
        from threadpoolctl import threadpool_limits
        from taiwan_lottery.ml.bingo_neighbors import prepare_neighbor_bundle

        with threadpool_limits(limits=1):
            bundle = prepare_neighbor_bundle(draws)
        channel.send({"bundle": bundle})
    except Exception as exc:
        channel.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        channel.close()


def _snapshot(draws):
    rows = copy.deepcopy(draws[-MAX_HISTORY:])
    # Timestamps participate in the key because they define missing-step edges.
    digest = hashlib.sha256(
        json.dumps(
            [(str(row["draw_term"]), str(row["draw_datetime"]), row["numbers"]) for row in rows],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return digest, rows


class NeighborResearchCache:
    """At most one running job, one latest pending snapshot and one result.

    No process is started until a user opens the neighbor analysis. Obsolete
    queued requests are replaced by the newest source snapshot, never appended.
    A failed snapshot is not retried on every poll. Runtime shutdown terminates
    research immediately instead of waiting for an expensive calculation.
    """

    def __init__(self, *, deadline_seconds=45, context=None, worker_target=None):
        self.context = context or mp.get_context("spawn")
        self.deadline_seconds = max(0.1, min(float(deadline_seconds), 120.0))
        self.worker_target = worker_target or _bundle_worker
        self._bundle = None
        self._ready_key = None
        self._active_key = None
        self._wanted = None
        self._failed_key = None
        self._error = None
        self._process = None
        self._channel = None
        self._monitor = None
        self._closed = False
        self._started_at = None

    def _start(self, key, rows):
        receive, send = self.context.Pipe(duplex=False)
        process = self.context.Process(target=self.worker_target, args=(rows, send), daemon=True)
        try:
            process.start()
        except Exception as exc:
            receive.close()
            send.close()
            self._failed_key, self._error = key, f"無法啟動鄰號研究：{type(exc).__name__}: {exc}"
            return
        send.close()
        self._process, self._channel = process, receive
        self._active_key = key
        self._started_at = time.monotonic()
        self._monitor = asyncio.create_task(self._watch(key, process, receive), name="bingo-neighbor-research")

    async def _watch(self, key, process, receive):
        result = None
        try:
            while True:
                if receive.poll():
                    remaining = self.deadline_seconds - (time.monotonic() - self._started_at)
                    try:
                        result = await asyncio.wait_for(asyncio.to_thread(receive.recv), max(0, remaining))
                    except TimeoutError as exc:
                        raise TimeoutError("鄰號研究傳輸超出時間預算，已停止") from exc
                    break
                if not process.is_alive():
                    raise RuntimeError(f"鄰號研究程序退出：{process.exitcode}")
                if time.monotonic() - self._started_at > self.deadline_seconds:
                    raise TimeoutError("鄰號研究超出時間預算，已停止")
                await asyncio.sleep(0.05)
            if result.get("error"):
                raise ValueError(result["error"])
            self._bundle, self._ready_key = result["bundle"], key
            self._failed_key, self._error = None, None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed_key, self._error = key, f"{type(exc).__name__}: {exc}"
        finally:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 1)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 1)
            process.close()
            receive.close()
            self._process, self._channel, self._active_key = None, None, None
            self._started_at = None
            if not self._closed and self._wanted and self._wanted[0] != key:
                latest_key, rows = self._wanted
                self._start(latest_key, rows)

    def read(self, draws, *, window=1000, lag=1, snapshot=0, source=None):
        if window not in WINDOWS or lag not in LAGS or snapshot not in SNAPSHOTS:
            raise ValueError("鄰號研究視窗、跨期距離或比較期數無效")
        if source is not None and not 1 <= source <= 80:
            raise ValueError("來源號碼必須為1～80")
        base = dict(
            state="waiting", analysis=None, research_only=True, allowed_for_live=False,
            source_term=str(draws[-1]["draw_term"]) if draws else None,
            query=dict(window=window, lag=lag, snapshot=snapshot, source=source),
            resources=dict(max_workers=1, threads=1, history_limit=MAX_HISTORY,
                           deadline_seconds=self.deadline_seconds, pending_limit=1),
        )
        if self._closed or not draws:
            return dict(base, reason="研究服務已關閉" if self._closed else "尚無已驗證歷史資料")
        key, rows = _snapshot(draws)
        self._wanted = key, rows
        if key == self._ready_key:
            from taiwan_lottery.ml.bingo_neighbors import neighbor_view

            view = neighbor_view(self._bundle, window=window, lag=lag, snapshot=snapshot, source=source)
            return dict(base, state="ready", analysis=view, reason="歷史條件頻率；尚未證明優於隨機")
        if key == self._failed_key:
            return dict(base, state="failed", reason=self._error)
        if self._process is None:
            self._start(key, rows)
        if key == self._failed_key:
            return dict(base, state="failed", reason=self._error)
        return dict(base, state="running", reason="鄰號研究背景計算中，正式開獎與訓練持續運行")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self._wanted = None
        if self._process and self._process.is_alive():
            self._process.terminate()
        if self._monitor:
            self._monitor.cancel()
            try:
                await self._monitor
            except asyncio.CancelledError:
                pass
        # A task cancelled before its first scheduling turn has no chance to run
        # its own finally block. Retain the same bounded cleanup in that case.
        if self._process is not None:
            process = self._process
            await asyncio.to_thread(process.join, 1)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 1)
            process.close()
            if self._channel is not None:
                self._channel.close()
            self._process, self._channel, self._active_key = None, None, None
        self._bundle = None
