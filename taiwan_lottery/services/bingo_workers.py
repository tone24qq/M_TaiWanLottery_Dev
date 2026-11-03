"""Bounded CPU workers with immutable checkpoints and persistent online updates.

Only locally generated pickle files are read. Checksums are integrity checks,
not permission to deserialize arbitrary user supplied model files.
"""

import asyncio
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from uuid import uuid4

_INFERENCE_CACHE = {}


def snapshot_hash(draws):
    return hashlib.sha256(
        json.dumps(
            [(d["draw_term"], d["numbers"]) for d in draws], separators=(",", ":")
        ).encode()
    ).hexdigest()


def _limit(threads):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(max(1, threads))


def _atomic(path, data):
    path = Path(path)
    # Different jobs can publish online/training artifacts concurrently.
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _publish(suite, draws, directory, version, *, online=False, base_sha256=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    content = pickle.dumps(suite, protocol=pickle.HIGHEST_PROTOCOL)
    checksum = hashlib.sha256(content).hexdigest()
    # A manifest already read by another process always references immutable bytes.
    filename = f"{'online-' if online else ''}{version}-{checksum}.pkl"
    _atomic(directory / filename, content)
    manifest = dict(
        version=version,
        file=filename,
        sha256=checksum,
        source_term=draws[-1]["draw_term"],
        source_time=str(draws[-1]["draw_datetime"]),
        snapshot_hash=snapshot_hash(draws),
        statuses=suite.statuses(),
        training_history_start=draws[-min(len(draws), len(suite.history))]["draw_term"],
        training_samples=len(suite.history),
        context_limit=suite.window,
        source_dates=sorted(
            {_time(d["draw_datetime"]).date().isoformat() for d in draws}
        ),
    )
    if online:
        manifest["base_sha256"] = base_sha256
    _atomic(
        directory / ("online.json" if online else "current.json"),
        json.dumps(manifest, ensure_ascii=False).encode(),
    )
    _prune_artifacts(directory)
    return manifest


def _prune_artifacts(directory, retain=32):
    """Keep recent versions and referenced checkpoints; give active readers grace."""
    import time

    protected = set()
    for name in ("current.json", "online.json"):
        try:
            protected.add(json.loads((directory / name).read_text())["file"])
        except (OSError, ValueError, KeyError):
            pass
    entries = []
    for path in directory.glob("*.pkl"):
        try:
            entries.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            pass
    for stamp, path in sorted(entries, reverse=True)[retain:]:
        if path.name not in protected and time.time() - stamp > 3600:
            path.unlink(missing_ok=True)


def _training(
    draws, directory, threads, window, version, messages, enabled=None, profiles=None
):
    _limit(threads)
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_models import ModelSuite

    try:
        with threadpool_limits(limits=threads):
            suite = ModelSuite(
                [d["numbers"] for d in draws],
                threads=threads,
                window=window,
                enabled=enabled,
            )
            previous_models, previous_records = {}, {}
            try:
                old_manifest = json.loads(
                    (Path(directory) / "current.json").read_text()
                )
                previous, cutoff = _load_checkpoint(old_manifest, draws, directory)
                for draw in draws[cutoff + 1 :]:
                    history = previous.history
                    previous.update(history, draw["numbers"])
                    previous.history = (history + [list(draw["numbers"])])[
                        -previous.window :
                    ]
                previous_models = {
                    name: model
                    for name, model in previous.models.items()
                    if name in suite.enabled
                    and (name != "river" or previous.window == suite.window)
                }
                previous_records = {
                    name: copy.deepcopy(previous.records[name])
                    for name in previous_models
                }
                for name, record in previous_records.items():
                    record.setdefault(
                        "training_source_term", old_manifest["source_term"]
                    )
                    record.setdefault("training_history_start", draws[0]["draw_term"])
                    record.setdefault("training_history_count", len(previous.history))
                suite.models.update(previous_models)
                suite.records.update(previous_records)
            except Exception:
                # A corrupt or older incompatible artifact must not prevent repair.
                pass

            def progress(model, completed, total, status):
                restored = status in ("failed", "disabled") and model in previous_models
                if restored:
                    error = suite.records[model].get("error")
                    suite.models[model] = previous_models[model]
                    suite.records[model] = dict(
                        previous_records[model],
                        retrain_error=error,
                        error=f"重訓失敗，沿用已驗證模型：{error}",
                    )
                elif status == "ready":
                    suite.records[model].update(
                        training_snapshot=snapshot_hash(draws),
                        training_version=version,
                        training_source_term=draws[-1]["draw_term"],
                        training_history_start=draws[0]["draw_term"],
                        training_history_count=len(draws),
                    )
                messages.put(
                    dict(
                        type="progress",
                        current_model=model,
                        completed=completed,
                        total=total,
                        progress=round(100 * completed / max(total, 1), 2),
                        state=status,
                        models=suite.statuses(),
                    )
                )
                # Healthy prior scorers remain active throughout retraining.
                if (status == "ready" or restored) and suite.models:
                    _publish(suite, draws, directory, version)

            suite.fit(progress=progress)
            if not suite.scores([d["numbers"] for d in draws]):
                raise ValueError("訓練完成但無有效模型")
            _publish(suite, draws, directory, version)
            if profiles:
                from taiwan_lottery.ml.bingo_profile_artifacts import (
                    train_validated_profiles,
                )

                def profile_progress(name, done, total, status):
                    messages.put(
                        dict(
                            type="progress",
                            current_model="profile/" + name,
                            completed=done,
                            total=total,
                            progress=100 * done / max(total, 1),
                            state=status,
                        )
                    )

                profile_report = train_validated_profiles(
                    draws,
                    directory,
                    profiles,
                    threads,
                    profile_progress,
                    enabled=enabled,
                )
                messages.put(dict(type="progress", profiles=profile_report))
            messages.put(dict(type="done", models=suite.statuses()))
    except Exception as exc:
        messages.put(dict(type="error", error=f"{type(exc).__name__}: {exc}"))


def _backtest(draws, options, threads, window, messages, directory=None, run_id=None):
    _limit(threads)
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_evaluation import walk_forward

    def progress(*args, **kwargs):
        completed = kwargs.get("completed", args[1] if len(args) > 1 else 0)
        total = kwargs.get("total", args[2] if len(args) > 2 else 1)
        status = kwargs.get("status", args[3] if len(args) > 3 else "running")
        model = kwargs.get("model", args[0] if args else "walk_forward")
        messages.put(
            dict(
                type="progress",
                current_model=model,
                completed=completed,
                total=total,
                progress=round(100 * completed / max(total, 1), 2),
                state=status,
            )
        )

    try:
        with threadpool_limits(limits=threads):
            options = dict(options)
            mode = options.pop("mode", "fixed")
            if mode == "associations":
                from taiwan_lottery.ml.bingo_association_evaluation import (
                    association_walk_forward,
                )

                report = association_walk_forward(
                    draws, threads=threads, window=window, progress=progress, **options
                )
                report["requested_mode"] = mode
            elif mode == "adaptive":
                from taiwan_lottery.ml.bingo_adaptive_evaluation import (
                    adaptive_walk_forward,
                )

                report = adaptive_walk_forward(
                    draws, threads=threads, window=window, progress=progress, **options
                )
                report["requested_mode"] = mode
            elif mode == "nested":
                from taiwan_lottery.ml.bingo_nested import nested_walk_forward

                report = nested_walk_forward(
                    draws, threads=threads, window=window, progress=progress, **options
                )
                report["requested_mode"] = mode
            else:
                report = walk_forward(
                    draws, threads=threads, window=window, progress=progress, **options
                )
        if directory and run_id and report.get("folds") is not None:
            import gzip

            reports = Path(directory) / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            path = reports / (run_id + ".json.gz")
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, allow_nan=False)
            temporary.replace(path)
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            summary = {
                k: v
                for k, v in report.items()
                if k not in ("folds", "results", "baseline", "shadow")
            }
            for name in ("baseline", "shadow"):
                summary[name] = {
                    k: v
                    for k, v in report.get(name, {}).items()
                    if k not in ("folds", "results")
                }
            summary["artifact"] = dict(
                id=run_id,
                sha256=checksum.hexdigest(),
                bytes=path.stat().st_size,
                format="gzip-json",
                fold_count=report.get("count"),
            )
            summary["artifact_download_url"] = (
                "/api/v1/bingo/backtest/artifact/" + run_id
            )
            report = summary
        messages.put(dict(type="done", report=report))
    except Exception as exc:
        messages.put(dict(type="error", error=f"{type(exc).__name__}: {exc}"))


def _time(value):
    from taiwan_lottery.bingo_clock import to_taipei

    return to_taipei(datetime.fromisoformat(value) if isinstance(value, str) else value)


def _cutoff(manifest, draws):
    index = next(
        (
            i
            for i, draw in enumerate(draws)
            if draw["draw_term"] == manifest.get("source_term")
        ),
        None,
    )
    if index is None or _time(manifest["source_time"]) != _time(
        draws[index]["draw_datetime"]
    ):
        raise ValueError("模型快照截止期不在指定資料內")
    return index


def _load_checkpoint(manifest, draws, directory):
    index = _cutoff(manifest, draws)
    filename = Path(manifest["file"])
    if filename.name != str(filename) or filename.suffix != ".pkl":
        raise ValueError("模型檔案路徑無效")
    content = (Path(directory) / filename).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("模型檔案校驗失敗")
    suite = pickle.loads(content)
    # Verify all overlapping source labels, including the cutoff itself.
    prefix = [d["numbers"] for d in draws[: index + 1]]
    overlap = min(len(prefix), len(suite.history))
    if not overlap or suite.history[-overlap:] != prefix[-overlap:]:
        raise ValueError("模型與正式資料快照不一致")
    return suite, index


def infer(
    draws, directory, threads, window, plan=None, day_scope=None, baseline_only=False
):
    _limit(threads)
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_models import ModelSuite

    if not draws:
        raise ValueError("沒有可評分的資料")
    history = [d["numbers"] for d in draws]
    directory = str(Path(directory).resolve())
    manifest = {}
    suite = None
    warning = "推論程序已隔離，暫用統計模型並等待背景修復" if baseline_only else None
    checkpoint = None
    cutoff = None
    try:
        path = Path(directory) / "current.json"
        if path.exists() and not baseline_only:
            manifest = json.loads(path.read_text())
            if day_scope and manifest.get("source_dates") != [day_scope]:
                raise ValueError("當日模型工件不得含其他日期或不明訓練範圍")
            # Source cutoff must be part of this caller's verified history.
            _cutoff(manifest, draws)
            cached = _INFERENCE_CACHE.get(directory)
            if cached and cached["base_sha256"] == manifest["sha256"]:
                try:
                    cutoff = _cutoff(cached["manifest"], draws)
                    prefix = [draw["numbers"] for draw in draws[: cutoff + 1]]
                    overlap = min(len(prefix), len(cached["suite"].history))
                    if cached["suite"].history[-overlap:] != prefix[-overlap:]:
                        raise ValueError("快取與正式資料快照不一致")
                    suite, checkpoint = cached["suite"], cached["manifest"]
                except ValueError:
                    pass
            if suite is None:
                online = Path(directory) / "online.json"
                if online.exists():
                    try:
                        candidate = json.loads(online.read_text())
                        if candidate.get("base_sha256") == manifest["sha256"]:
                            suite, cutoff = _load_checkpoint(
                                candidate, draws, directory
                            )
                            checkpoint = candidate
                    except (ValueError, OSError, KeyError, pickle.UnpicklingError):
                        pass
            if suite is None:
                suite, cutoff = _load_checkpoint(manifest, draws, directory)
                checkpoint = manifest
    except Exception as exc:
        warning = f"模型恢復失敗，使用統計模型：{type(exc).__name__}: {exc}"
        suite = None
    with threadpool_limits(limits=threads):
        if suite is None:
            suite = ModelSuite(
                history, threads=threads, window=window, enabled=["frequency", "cdm"]
            )
            suite.fit()
            version = "baseline"
        else:
            # The checkpoint retains its complete adaptive window. It is essential
            # when the caller's rolling history has already dropped older rows.
            for draw in draws[cutoff + 1 :]:
                previous = suite.history
                suite.update(previous, draw["numbers"])
                if "river" in suite.models:
                    record = suite.records["river"]
                    record.setdefault(
                        "batch_training_cutoff",
                        record.get("training_source_term", manifest.get("source_term")),
                    )
                    record["training_source_term"] = draw["draw_term"]
                    record["training_history_count"] = (
                        record.get("training_history_count", len(previous)) + 1
                    )
                suite.history = (previous + [list(draw["numbers"])])[-suite.window :]
            version = manifest["version"]
        extra = {}
        if plan is not None:
            from taiwan_lottery.ml.bingo_contexts import context_scores

            safe_plan = copy.deepcopy(plan)
            if baseline_only:
                safe_plan["profiles_by_star"] = {}
            extra = context_scores(
                suite, draws, safe_plan, directory=directory, threads=threads
            )
            scores = extra.pop("scores")
        else:
            scores = suite.scores(history)
        if not scores:
            raise ValueError("無可用模型分數")
    records = suite.statuses()
    if warning:
        for record in records:
            if record["name"] not in scores:
                record.update(status="failed", error=warning)
    if manifest and version != "baseline":
        previous = _INFERENCE_CACHE.get(directory)
        # Backwards inspection must not replace a newer online checkpoint/cache.
        newer = (
            previous
            and previous["base_sha256"] == manifest["sha256"]
            and _time(previous["manifest"]["source_time"])
            > _time(draws[-1]["draw_datetime"])
        )
        if not newer:
            if checkpoint["source_term"] != draws[-1]["draw_term"]:
                checkpoint = _publish(
                    suite,
                    draws,
                    directory,
                    version,
                    online=True,
                    base_sha256=manifest["sha256"],
                )
            _INFERENCE_CACHE[directory] = dict(
                base_sha256=manifest["sha256"], manifest=checkpoint, suite=suite
            )
    result = dict(
        scores=scores,
        models=records,
        version=(version + "-" + (checkpoint or manifest)["sha256"][:12])
        if version != "baseline"
        else version,
        snapshot_hash=snapshot_hash(draws),
        warning=warning,
        training_cutoff=manifest.get("source_term", draws[-1]["draw_term"])
        if version != "baseline"
        else draws[-1]["draw_term"],
        training_history_start=manifest.get(
            "training_history_start", draws[0]["draw_term"]
        ),
        training_samples=manifest.get("training_samples", len(suite.history)),
        pid=os.getpid(),
        context_limit=suite.window,
        **extra,
    )
    if plan is not None:
        result["today"] = infer_today(
            draws,
            directory,
            threads,
            plan.get("target_date"),
            baseline_only=baseline_only,
        )
        try:
            from taiwan_lottery.ml.bingo_associations import association_candidates
            from taiwan_lottery.ml.bingo_evaluation import fuse

            by_star = result.get("by_star_scores") or {
                str(k): scores for k in (3, 4, 5)
            }
            weights = {
                str(k): fuse(
                    by_star[str(k)],
                    plan.get("vote_methods", {}).get(str(k), "shrink"),
                    k,
                    plan.get("association_performance", []),
                )["weights"]
                for k in (3, 4, 5)
            }
            requested_windows = sorted(
                {
                    int(w)
                    for row in plan.get("champions", {}).values()
                    for w in row.values()
                    if int(w) >= 100
                }
            )
            with threadpool_limits(limits=threads):
                result["associations"] = association_candidates(
                    draws,
                    by_star,
                    windows=sorted({200, 500, max(requested_windows or [window])}),
                    model_weights_by_star=weights,
                    target_date=plan.get("target_date"),
                )
        except Exception as exc:
            # A failed research engine cannot prevent the original forecasts.
            result["association_error"] = f"{type(exc).__name__}: {exc}"
    return result


def infer_today(draws, directory, threads, target_date, *, baseline_only=False):
    """A separate date-scoped artifact; previous-day weights are never consulted."""
    from taiwan_lottery.ml.bingo_models import draw_matrix

    target_date = (
        target_date or _time(draws[-1]["draw_datetime"]).date().isoformat()
    )
    today = [
        d
        for d in draws
        if _time(d["draw_datetime"]).date().isoformat() == target_date
    ]
    started = time.perf_counter()
    if len(today) < 10:
        matrix = draw_matrix([d["numbers"] for d in today])
        # Beta(2,6) prior, updated only by known target-day observations.
        # At n=0 this is an explicit uniform .25 prior, never fabricated training.
        probabilities = ((matrix.sum(axis=0) + 2) / (len(today) + 8)).tolist()
        result = dict(
            scores={"today_prior": probabilities},
            models=[],
            version="today-beta-prior-v1",
            training_cutoff=None,
            training_history_start=None,
            training_samples=len(today),
            warning=None,
        )
        reason = "當日樣本少於10期，使用明示的當日Beta先驗，尚未訓練其他模型"
    else:
        path = Path(directory) / "today" / target_date
        result = infer(
            today,
            str(path),
            threads,
            max(100, min(1000, len(today))),
            day_scope=target_date,
            baseline_only=baseline_only,
        )
        reason = "所有模型工件、特徵與增量更新均限制於目標當天"
    used = min(len(today), result.get("context_limit", len(today)))
    result.update(
        target_date=target_date,
        available_count=len(today),
        history_count=used,
        history_start=today[-used]["draw_term"] if used else None,
        history_end=today[-1]["draw_term"] if today else None,
        history_dates=[target_date] if today else [],
        snapshot_hash=snapshot_hash(today),
        warmup=len(today) < 10,
        scoring_cost_seconds=time.perf_counter() - started,
        reason=reason,
    )
    # Date namespaces cannot leave an unbounded process-local cache after months.
    day_keys = [key for key in _INFERENCE_CACHE if "/today/" in key]
    for key in sorted(day_keys)[:-3]:
        _INFERENCE_CACHE.pop(key, None)
    return result


class WorkerManager:
    adaptive_scoring = True

    def __init__(self, directory, threads=2, window=1000, on_backtest=None):
        self.directory = str(directory)
        self.threads = max(1, min(threads, os.cpu_count() or 1, 8))
        self.window = max(100, min(window, 10000))
        self.context = mp.get_context("spawn")
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=self.context)
        self.process = None
        self.messages = None
        self.monitor = None
        self.on_backtest = on_backtest
        self.training = dict(state="waiting", progress=0, completed=0, total=0)
        self.backtest = dict(
            state="waiting", progress=0, completed=0, total=0, report=None
        )
        self.models = []
        self.inference_pid = None
        self.inference_quarantined = False
        self.job_started = None
        self.active_kind = None
        self.last_training_seconds = None
        self.last_backtest_seconds = None
        self.last_inference_seconds = None
        self._stopping = False
        self._closed = False
        manifest = Path(directory) / "current.json"
        if manifest.exists():
            try:
                self.models = json.loads(manifest.read_text())["statuses"]
            except (ValueError, KeyError, OSError):
                self.training.update(state="failed", error="模型資訊無法恢復，等待重訓")

    @property
    def busy(self):
        return self.process is not None

    async def score(self, draws, plan=None):
        if self._closed:
            raise ValueError("背景工作已關閉")
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                infer,
                draws,
                self.directory,
                self.threads,
                self.window,
                plan,
                None,
                self.inference_quarantined,
            )
        except BrokenProcessPool:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = ProcessPoolExecutor(max_workers=1, mp_context=self.context)
            self.inference_quarantined = True
            result = await loop.run_in_executor(
                self.executor,
                infer,
                draws,
                self.directory,
                self.threads,
                self.window,
                plan,
                None,
                True,
            )
        self.last_inference_seconds = time.perf_counter() - started
        self.inference_pid = result.get("pid")
        if not self.busy:
            self.models = result["models"]
        return result

    async def start(self, kind, draws, options=None):
        if self._closed:
            raise ValueError("背景工作已關閉")
        if kind not in ("train", "today_train", "backtest"):
            raise ValueError("未知背景工作")
        if self.busy:
            raise ValueError("背景工作執行中，請等待完成或停止訓練")
        if len(draws) < 10:
            raise ValueError("至少需要 10 期已驗證資料")
        state = self.training if kind in ("train", "today_train") else self.backtest
        state.clear()
        state.update(
            state="running",
            progress=0,
            completed=0,
            total=0,
            error=None,
            scope="today"
            if kind == "today_train"
            else "historical"
            if kind == "train"
            else "validation",
        )
        version = uuid4().hex
        state["run_id"] = version
        # A manager queue cannot be corrupted by killing a child halfway through
        # writing a large backtest report. Its small server consumes no model CPU.
        options = options or {}
        if kind == "today_train":
            from datetime import date

            day = date.fromisoformat(options["target_date"]).isoformat()
            state["target_date"] = day
            if any(_time(d["draw_datetime"]).date().isoformat() != day for d in draws):
                state.update(state="failed", error="當日訓練資料混入其他日期，已拒絕")
                raise ValueError("Today Only 訓練不能使用其他日期資料")
        manager = self.context.Manager()
        messages = manager.Queue()
        args = (
            (
                draws,
                str(Path(self.directory) / "today" / str(options["target_date"]))
                if kind == "today_train"
                else self.directory,
                self.threads,
                self.window,
                version,
                messages,
                options.get("enabled"),
                options.get("validated_profiles"),
            )
            if kind in ("train", "today_train")
            else (
                draws,
                options,
                self.threads,
                self.window,
                messages,
                self.directory,
                version,
            )
        )
        process = self.context.Process(
            target=_training if kind in ("train", "today_train") else _backtest,
            args=args,
            daemon=True,
        )
        try:
            process.start()
        except Exception:
            manager.shutdown()
            state.update(state="failed", error="無法啟動背景工作")
            raise
        self.process, self.messages, self._stopping = process, messages, False
        self.job_started = time.monotonic()
        self.active_kind = kind
        self.monitor = asyncio.create_task(
            self._monitor(kind, state, process, messages, manager)
        )
        return dict(state)

    async def _monitor(self, kind, state, process, messages, manager):
        terminal = False
        try:
            while True:
                while True:
                    try:
                        item = messages.get_nowait()
                    except queue.Empty:
                        break
                    event = item.pop("type")
                    if item.get("models") is not None:
                        if kind == "today_train":
                            self.today_models = item["models"]
                        else:
                            self.models = item["models"]
                    if not self._stopping:
                        state.update(item)
                        if event in ("done", "error"):
                            state["state"] = (
                                "completed" if event == "done" else "failed"
                            )
                            terminal = True
                if not process.is_alive():
                    if self._stopping:
                        state["state"] = "stopped"
                    elif not terminal:
                        state.update(
                            state="failed", error=f"工作程序退出：{process.exitcode}"
                        )
                    break
                await asyncio.sleep(0.1)
        except Exception as exc:
            if not self._stopping:
                state.update(
                    state="failed", error=f"工作監控失敗：{type(exc).__name__}: {exc}"
                )
        finally:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 2)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 1)
            process.close()
            await asyncio.to_thread(manager.shutdown)
            elapsed = time.monotonic() - self.job_started if self.job_started else None
            if kind in ("train", "today_train"):
                self.last_training_seconds = elapsed
                if kind == "train" and state.get("state") == "completed":
                    self.inference_quarantined = False
            else:
                self.last_backtest_seconds = elapsed
            state["elapsed_seconds"] = elapsed
            self.process, self.messages = None, None
            self.active_kind = None
            if kind == "backtest" and self.on_backtest:
                try:
                    await self.on_backtest(dict(state))
                except Exception as exc:
                    state["error"] = f"回測結果儲存失敗：{type(exc).__name__}: {exc}"

    async def stop(self):
        if self.process:
            self._stopping = True
            self.process.terminate()
            if self.monitor:
                await self.monitor

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.stop()
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
