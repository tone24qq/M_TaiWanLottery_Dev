"""Automatic sync -> settle -> immutable next prediction -> ranking loop."""

import asyncio
import copy
from collections import Counter
from datetime import timedelta
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from taiwan_lottery.bingo_clock import (
    now_taipei,
    to_taipei,
    next_draw,
    is_successor,
    latest_scheduled_draw,
)
from taiwan_lottery.config import settings
from taiwan_lottery.db.crud.bingo_research import ResearchStore
from taiwan_lottery.ml.bingo_evaluation import (
    fuse,
    live_rankings,
    live_rankings_summary,
    normalize_scores,
    recommend_method,
)
from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper
from taiwan_lottery.services.bingo_workers import WorkerManager


class BingoRuntime:
    def __init__(self, session_factory, source=None, clock=now_taipei, workers=None):
        self.store = ResearchStore(session_factory)
        self.source = source or BingoScraper()
        self.clock = clock
        self.workers = workers or WorkerManager(
            settings.MODEL_ARTIFACTS_DIR / "bingo",
            settings.CPU_THREADS,
            settings.TRAIN_WINDOW,
            self._backtest_done,
        )
        self.snapshot = []
        self.lock = asyncio.Lock()
        self.task = None
        self.sync = dict(
            state="waiting",
            last_success=None,
            last_attempt=None,
            error=None,
            missing_terms=[],
        )
        self.prediction = None
        self.target = None
        self.outcomes = []
        self.boards = {str(k): dict(cumulative=[], consecutive=[]) for k in (3, 4, 5)}
        self.strategy = dict(
            method="shrink", status="暫定策略，尚未完成樣本外比較", comparison=None
        )
        self._last_train_size = 0
        self._last_train_time = None
        self._last_train_term = None
        self._auto_paused = False
        self._weights = {}
        self._outcome_summary = {"cumulative": [], "consecutive": []}
        self._ledger_dirty = True
        self._bootstrapping = True
        self._closed = False
        self._complete_days = set()

    async def initialize(self):
        await self.store.init_schema()
        self.snapshot = await self.store.history(limit=10000)
        self._bootstrapping = not bool(self.snapshot)
        await self.store.settle(now=self.clock())
        await self._refresh()
        previous = await self.store.latest_evaluation()
        if previous:
            self.workers.backtest.update(previous)
            if previous.get("report"):
                self._apply_strategy(previous["report"])
        manifest = (
            Path(
                getattr(
                    self.workers, "directory", settings.MODEL_ARTIFACTS_DIR / "bingo"
                )
            )
            / "current.json"
        )
        if manifest.exists():
            try:
                checkpoint = json.loads(manifest.read_text())
                if any(
                    d["draw_term"] == checkpoint["source_term"] for d in self.snapshot
                ) and all(
                    r["status"] in ("ready", "disabled") for r in checkpoint["statuses"]
                ):
                    self._last_train_term = checkpoint["source_term"]
                    self.workers.training.update(
                        state="completed",
                        restored=True,
                        progress=100,
                        completed=1,
                        total=1,
                    )
                    updates = [
                        datetime.fromisoformat(r["updated_at"])
                        for r in checkpoint["statuses"]
                        if r.get("updated_at")
                    ]
                    self._last_train_time = (
                        to_taipei(max(updates)) if updates else self.clock()
                    )
            except (OSError, ValueError, KeyError):
                logger.warning("無法恢復訓練排程，將使用已驗證快照重訓")
        if settings.SCRAPER_ENABLED:
            self.task = asyncio.create_task(self._loop(), name="bingo-sync")

    async def _loop(self):
        while True:
            started = asyncio.get_running_loop().time()
            try:
                await self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("賓果同步週期失敗")
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.1, settings.SYNC_INTERVAL_SECONDS - elapsed))

    def _gap_days(self):
        days = set()
        gaps = []
        for previous, current in zip(self.snapshot, self.snapshot[1:]):
            if not is_successor(
                previous["draw_term"],
                previous["draw_datetime"],
                current["draw_term"],
                current["draw_datetime"],
            ):
                day = previous["draw_datetime"].date()
                while day <= current["draw_datetime"].date():
                    days.add(day)
                    day += timedelta(days=1)
                gaps.append(
                    dict(after=previous["draw_term"], before=current["draw_term"])
                )
        return days, gaps

    async def cycle(self):
        if self.lock.locked():
            return dict(self.sync)
        async with self.lock:
            at = self.clock()
            self.sync.update(state="syncing", last_attempt=at.isoformat(), error=None)
            try:
                today = at.date()
                gap_days, _ = self._gap_days()
                start = today - timedelta(days=2)
                if self.snapshot:
                    start = min(start, self.snapshot[-1]["draw_datetime"].date())
                # Each attempt performs bounded work, older gaps remain visible.
                start = max(
                    start, today - timedelta(days=settings.MAX_BACKFILL_DAYS - 1)
                )
                days = {
                    start + timedelta(days=i) for i in range((today - start).days + 1)
                }
                days = {d for d in days if d not in self._complete_days}
                days.add(today)
                days |= {
                    d
                    for d in gap_days
                    if d >= today - timedelta(days=settings.MAX_BACKFILL_DAYS - 1)
                }
                inserted = 0
                # Newest day first; recover current operation promptly, then backfill.
                failures = []
                latest_day_ok = False
                for day in sorted(days, reverse=True):
                    try:
                        draws = await self.source.fetch_day(day)
                        inserted += await self.store.ingest(draws)
                        if day < today and len(draws) == 203:
                            self._complete_days.add(day)
                        if day == today:
                            latest_day_ok = True
                            # Current results must settle and archive before slow
                            # historical retries consume the next draw deadline.
                            self.snapshot = await self.store.history(limit=10000)
                            settled = await self.store.settle(now=self.clock())
                            self._ledger_dirty |= bool(settled)
                            await self._refresh()
                            if not self._bootstrapping:
                                await self._predict_locked()
                    except Exception as exc:
                        failures.append(f"{day}：{type(exc).__name__}: {exc}")
                self.snapshot = await self.store.history(limit=10000)
                extra = today - timedelta(days=3)
                earliest = today - timedelta(days=settings.MAX_BACKFILL_DAYS - 1)
                while (
                    len(self.snapshot) < settings.HISTORY_MIN_DRAWS
                    and extra >= earliest
                ):
                    try:
                        historical = await self.source.fetch_day(extra)
                        inserted += await self.store.ingest(historical)
                        if len(historical) == 203:
                            self._complete_days.add(extra)
                    except Exception as exc:
                        failures.append(f"{extra}：{type(exc).__name__}: {exc}")
                    self.snapshot = await self.store.history(limit=10000)
                    extra -= timedelta(days=1)
                _, missing = self._gap_days()
                self.sync.update(
                    state="degraded" if failures else "connected",
                    missing_terms=missing,
                    inserted=inserted,
                    error="；".join(failures) or None,
                )
                if latest_day_ok:
                    self.sync["last_success"] = self.clock().isoformat()
                if len(self.snapshot) < settings.HISTORY_MIN_DRAWS:
                    self.sync.update(
                        state="insufficient",
                        error=f"擴充歷史後仍只有 {len(self.snapshot)} 期，至少需要 {settings.HISTORY_MIN_DRAWS} 期",
                    )
                settled = await self.store.settle(now=self.clock())
                self._ledger_dirty |= bool(settled)
                await self._refresh()
                self._bootstrapping = False
                await self._predict_locked()
                await self._maybe_train()
            except Exception as exc:
                # Committed source data and predictions survive transient failures.
                self.sync.update(
                    state="disconnected", error=f"{type(exc).__name__}: {exc}"
                )
                logger.warning("賓果同步中斷：{}", self.sync["error"])
                raise
            return dict(self.sync)

    async def _refresh(self):
        if self._ledger_dirty:
            self.outcomes = await self.store.outcome_rows(limit_terms=1000)
            self._outcome_summary = await self.store.outcome_summary()
            self._ledger_dirty = False
        self.boards = live_rankings_summary(
            self._outcome_summary, weights=self._weights
        )
        self.prediction = await self.store.bundle()
        if self.prediction and self.prediction.get("weights"):
            self._weights = self.prediction["weights"]
            self.boards = live_rankings_summary(
                self._outcome_summary, weights=self._weights
            )
        if self.snapshot:
            term, at = next_draw(
                self.snapshot[-1]["draw_term"], self.snapshot[-1]["draw_datetime"]
            )
            self.target = dict(draw_term=term, draw_datetime=at.isoformat())

    async def predict(self):
        async with self.lock:
            return await self._predict_locked()

    async def _predict_locked(self):
        if len(self.snapshot) < 10:
            return None
        latest = self.snapshot[-1]
        term, at = next_draw(latest["draw_term"], latest["draw_datetime"])
        existing = await self.store.bundle(term)
        if existing:
            self.prediction = existing
            return existing
        if at <= self.clock():
            # Late official results may never be labelled as next-draw foresight.
            self.sync["state"] = "delayed"
            return None
        snapshot = copy.deepcopy(
            self.snapshot[-max(settings.TRAIN_WINDOW + 500, 1500) :]
        )
        result = await self.workers.score(snapshot)
        if result.get("warning") and not self.workers.busy:
            self.workers.training.update(state="failed", error=result["warning"])
            logger.warning(result["warning"])
        score_map = {}
        for name, vector in result["scores"].items():
            try:
                score_map[name] = normalize_scores(vector)
            except (ValueError, TypeError) as exc:
                logger.warning("排除無效模型 {}：{}", name, exc)
        if not score_map:
            raise ValueError("無可用模型評分，尚未封存")
        method = self.strategy["method"]
        forecasts, by_star = [], {}
        for stars in (3, 4, 5):
            for name, vector in score_map.items():
                ranking = sorted(range(1, 81), key=lambda n: (-vector[n - 1], n))
                forecasts.append(
                    dict(
                        model=name,
                        stars=stars,
                        numbers=ranking[:stars],
                        scores=vector,
                        version=result["version"],
                        method="individual",
                    )
                )
            selected = None
            for voting in ("equal", "shrink", "borda", "majority"):
                fused = fuse(score_map, voting, stars, self.outcomes)
                forecasts.append(
                    dict(
                        model=voting,
                        stars=stars,
                        numbers=fused["ranking"][:stars],
                        scores=fused["scores"],
                        weights=fused["weights"],
                        version=result["version"],
                        method=voting,
                    )
                )
                if voting == method:
                    selected = fused
            self._weights[str(stars)] = selected["weights"]
            numbers = selected["ranking"][:stars]
            by_star[str(stars)] = dict(
                numbers=numbers,
                scores=[selected["scores"][n - 1] for n in numbers],
                weights=selected["weights"],
                confidence=sum(selected["scores"][n - 1] for n in numbers) / stars,
                support=[selected["support"][str(n)] for n in numbers],
                ranking=[
                    dict(
                        number=n,
                        score=selected["scores"][n - 1],
                        support=selected["support"][str(n)],
                    )
                    for n in selected["ranking"]
                ],
            )
            if stars == 3:
                full_ranking = [
                    dict(
                        number=n,
                        score=selected["scores"][n - 1],
                        support=selected["support"][str(n)],
                    )
                    for n in selected["ranking"]
                ]
        payload = dict(
            method=method,
            weights=copy.deepcopy(self._weights),
            by_star=by_star,
            ranking=full_ranking,
            snapshot_hash=result["snapshot_hash"],
            source_term=latest["draw_term"],
            target_time=at.isoformat(),
            model_version=result["version"],
        )
        self.prediction = await self.store.archive(
            term,
            at,
            latest["draw_term"],
            latest["draw_datetime"],
            payload,
            forecasts,
            now=self.clock(),
        )
        self.boards = live_rankings_summary(
            self._outcome_summary, weights=self._weights
        )
        return self.prediction

    async def _maybe_train(self):
        if (
            not settings.AUTO_TRAIN
            or self._auto_paused
            or self.workers.busy
            or len(self.snapshot) < settings.HISTORY_MIN_DRAWS
        ):
            return
        now = self.clock()
        if (
            self._last_train_time
            and (now - self._last_train_time).total_seconds()
            < settings.RETRAIN_INTERVAL_SECONDS
        ):
            cutoffs = [
                i
                for i, d in enumerate(self.snapshot)
                if d["draw_term"] == self._last_train_term
            ]
            new_draws = (
                len(self.snapshot) - cutoffs[0] - 1 if cutoffs else len(self.snapshot)
            )
            failed_retry = (
                self.workers.training.get("state") == "failed"
                and (now - self._last_train_time).total_seconds() >= 60
            )
            if new_draws < settings.RETRAIN_MIN_DRAWS and not failed_retry:
                return
        await self.start_training()

    async def start_training(self):
        snapshot = copy.deepcopy(self.snapshot[-settings.TRAIN_WINDOW :])
        result = await self.workers.start("train", snapshot)
        self._last_train_size = len(self.snapshot)
        self._last_train_time = self.clock()
        self._last_train_term = self.snapshot[-1]["draw_term"]
        self._auto_paused = False
        return result

    async def stop_training(self):
        self._auto_paused = True
        await self.workers.stop()
        self.workers.training["state"] = "stopped"
        return self.workers.training

    async def start_backtest(self, options):
        # A 1000-fold run needs a separate >=100-draw causal warmup.
        if self.workers.busy:
            raise ValueError("背景工作執行中，請等待完成或停止訓練")
        requested = options.get("count", 100) + 100
        draws = await self.store.history(
            limit=100000
            if options.get("start_term") or options.get("end_term")
            else min(100000, requested + settings.TRAIN_WINDOW)
        )
        day = (
            draws[0]["draw_datetime"].date() - timedelta(days=1)
            if draws
            else self.clock().date()
        )
        lower = self.clock().date() - timedelta(days=settings.MAX_BACKFILL_DAYS - 1)

        def needs_range():
            if not options.get("start_term"):
                return False
            return (
                sum(int(d["draw_term"]) < int(options["start_term"]) for d in draws)
                < 100
            )

        while (len(draws) < requested or needs_range()) and day >= lower:
            await self.store.ingest(await self.source.fetch_day(day))
            day -= timedelta(days=1)
            draws = await self.store.history(
                limit=100000
                if options.get("start_term") or options.get("end_term")
                else min(100000, requested + settings.TRAIN_WINDOW)
            )
        self.snapshot = await self.store.history(limit=10000)
        result = await self.workers.start("backtest", copy.deepcopy(draws), options)
        await self.store.save_evaluation(result["run_id"], "running", None)
        return result

    async def _backtest_done(self, state):
        report = state.get("report")
        await self.store.save_evaluation(
            state["run_id"], state["state"], report, state.get("error")
        )
        if report:
            self._apply_strategy(report)

    def _apply_strategy(self, report):
        decision = recommend_method(report)
        self.strategy = dict(
            method=decision["method"],
            status=decision["reason"],
            comparison=report.get("method_comparison"),
            validated=decision["validated"],
        )

    def status(self):
        latest = self.snapshot[-1] if self.snapshot else None
        sync = dict(self.sync)
        sync["delay_seconds"] = (
            max(
                0.0,
                (
                    latest_scheduled_draw(self.clock())
                    - to_taipei(latest["draw_datetime"])
                ).total_seconds(),
            )
            if latest
            else None
        )
        return dict(
            latest={
                **latest,
                "draw_datetime": to_taipei(latest["draw_datetime"]).isoformat(),
            }
            if latest
            else None,
            sync=sync,
            target=self.target,
            prediction=self.prediction,
            models=self.workers.models,
            training=self.workers.training,
            leaderboards=self.boards,
            strategy=self.strategy,
        )

    def analysis(self):
        from itertools import combinations

        draws = self.snapshot[-settings.TRAIN_WINDOW :]
        count = Counter(n for d in draws for n in d["numbers"])
        gaps = {n: len(draws) for n in range(1, 81)}
        for i, d in enumerate(reversed(draws)):
            for n in d["numbers"]:
                if gaps[n] == len(draws):
                    gaps[n] = i
        pairs = Counter(pair for d in draws for pair in combinations(d["numbers"], 2))
        return dict(
            frequency=[
                dict(
                    number=n,
                    count=count[n],
                    rate=count[n] / len(draws) if draws else 0,
                    gap=gaps[n],
                )
                for n in range(1, 81)
            ],
            ranking={
                k: v.get("ranking", [])
                for k, v in self.prediction.get("by_star", {}).items()
            }
            if self.prediction
            else {},
            cooccurrence=[
                dict(numbers=list(pair), count=c) for pair, c in pairs.most_common(20)
            ],
            size_odd_even=dict(
                odd=sum(n % 2 for d in draws for n in d["numbers"]),
                even=sum(n % 2 == 0 for d in draws for n in d["numbers"]),
                small=sum(n <= 40 for d in draws for n in d["numbers"]),
                large=sum(n > 40 for d in draws for n in d["numbers"]),
            ),
            history=[
                {**d, "draw_datetime": to_taipei(d["draw_datetime"]).isoformat()}
                for d in reversed(draws[-100:])
            ],
            samples=len(draws),
        )

    async def close(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        await self.workers.close()
