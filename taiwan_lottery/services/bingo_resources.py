"""Nonblocking measured resource telemetry; unavailable observations stay null."""

from __future__ import annotations

import os
import time


class ResourceSampler:
    def __init__(self):
        self.processes = {}
        self.measured = set()
        self.started = time.monotonic()

    def _process(self, pid):
        if pid is None:
            return {"pid": None, "cpu_percent": None, "rss_mb": None, "threads": None}
        try:
            import psutil

            process = self.processes.get(pid)
            if process is None:
                process = psutil.Process(pid)
                self.processes[pid] = process
            with process.oneshot():
                cpu = process.cpu_percent(interval=None)
                result = dict(
                    pid=pid,
                    cpu_percent=cpu if pid in self.measured else None,
                    rss_mb=process.memory_info().rss / 1024**2,
                    threads=process.num_threads(),
                )
            self.measured.add(pid)
            return result
        except (ImportError, OSError, ProcessLookupError):
            return {"pid": pid, "cpu_percent": None, "rss_mb": None, "threads": None}
        except Exception:
            # A worker may exit between the liveness and memory observations.
            return {"pid": pid, "cpu_percent": None, "rss_mb": None, "threads": None}

    def sample(self, workers):
        main = self._process(os.getpid())
        try:
            training_pid = getattr(getattr(workers, "process", None), "pid", None)
        except ValueError:
            training_pid = None
        training = self._process(training_pid)
        inference = self._process(getattr(workers, "inference_pid", None))
        state = getattr(workers, "training", {})
        active = getattr(workers, "active_kind", None)
        training["state"] = (
            state.get("state")
            if active != "backtest"
            else getattr(workers, "backtest", {}).get("state")
        )
        training["kind"] = active
        training["elapsed_seconds"] = (
            max(0, time.monotonic() - workers.job_started)
            if getattr(workers, "job_started", None) and workers.busy
            else None
        )
        active_pids = {
            os.getpid(),
            training_pid,
            getattr(workers, "inference_pid", None),
        }
        self.processes = {
            pid: p for pid, p in self.processes.items() if pid in active_pids
        }
        self.measured.intersection_update(active_pids)
        result = dict(
            main,
            workers={"training": training, "inference": inference},
            training_seconds=getattr(workers, "last_training_seconds", None),
            backtest_seconds=getattr(workers, "last_backtest_seconds", None),
            inference_seconds=getattr(workers, "last_inference_seconds", None),
            uptime_seconds=time.monotonic() - self.started,
            cpu_definition="各程序相對單一 CPU 核心的百分比；第一筆觀測不估算",
        )
        return result
