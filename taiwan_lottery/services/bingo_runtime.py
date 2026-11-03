"""Automatic sync -> settle -> immutable next prediction -> ranking loop."""

import asyncio
import copy
from collections import Counter
from datetime import timedelta
import json
import hashlib
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
    fuse_many,
    live_rankings,
    live_rankings_summary,
    normalize_scores,
    recommend_method,
    _performance_summary,
)
from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper
from taiwan_lottery.services.bingo_workers import WorkerManager
from taiwan_lottery.ml.bingo_adaptive import AdaptiveController
from taiwan_lottery.ml.bingo_research_strategies import ResearchStrategies
from taiwan_lottery.ml.bingo_models import MODEL_NAMES
from taiwan_lottery.ml.bingo_contexts import context_audit
from taiwan_lottery.services.bingo_monitoring import PerformanceMonitor
from taiwan_lottery.services.bingo_resources import ResourceSampler
from taiwan_lottery.services.bingo_association_tracking import AssociationTracker
from taiwan_lottery.services.bingo_neighbors import NeighborResearchCache


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
        self.adaptive = AdaptiveController(
            baseline_window=settings.TRAIN_WINDOW,
            shadow_budget=settings.ADAPTIVE_SHADOW_BUDGET,
        )
        self.performance = PerformanceMonitor()
        self.families = ResearchStrategies()
        self.association_tracker = AssociationTracker()
        self.neighbor_research = NeighborResearchCache()
        self._monitor_view = self.performance.status()
        self.resources = ResourceSampler()
        self._strategy_events = []
        self._diagnostics = []
        self._validation_at = None
        self._validation_source = None
        self._profile_status = {}
        self._last_today_train_time = None
        self._last_today_train_date = None

    async def initialize(self):
        await self.store.init_schema()
        self.snapshot = await self.store.history(limit=10000)
        self._bootstrapping = not bool(self.snapshot)
        for namespace in (
            "adaptive",
            "monitor",
            "families",
            "associations",
            "validation",
        ):
            saved = await self.store.load_research_state(namespace)
            if not saved:
                continue
            if namespace == "adaptive":
                if saved["payload"].get("config", {}).get("simulation"):
                    raise ValueError("歷史模擬策略狀態不可載入正式流程")
                self.adaptive = AdaptiveController(saved["payload"])
            elif namespace == "monitor":
                self.performance = PerformanceMonitor(saved["payload"])
            elif namespace == "families":
                if saved["payload"].get("is_live") is False:
                    raise ValueError("歷史模擬不可載入並行實戰")
                self.families = ResearchStrategies(saved["payload"])
            elif namespace == "associations":
                if saved["payload"].get("is_live") is False:
                    raise ValueError("歷史模擬不可載入關聯實戰")
                self.association_tracker = AssociationTracker(saved["payload"])
            else:
                self._validation_at = saved["payload"].get("last_evaluation_at")
                self._validation_source = saved["payload"].get("source_term")
        self._monitor_view = self.performance.status()
        if self.snapshot:
            self.adaptive.plan(
                self.snapshot[-1]["draw_term"], history=len(self.snapshot)
            )
        await self.store.settle(now=self.clock())
        await self._refresh()
        previous = await self.store.latest_evaluation()
        if previous:
            self.workers.backtest.update(previous)
            if previous.get("report"):
                self._apply_strategy(previous["report"], restore=True)
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
                await self._maybe_evaluate()
            except Exception as exc:
                # Committed source data and predictions survive transient failures.
                self.sync.update(
                    state="disconnected", error=f"{type(exc).__name__}: {exc}"
                )
                logger.warning("賓果同步中斷：{}", self.sync["error"])
                raise
            return dict(self.sync)

    async def _refresh(self):
        await self.store.settle_shadow(now=self.clock())
        await self._process_research()
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
            self.sync["state"] = "delayed"
            return None
        snapshot = copy.deepcopy(
            self.snapshot[-max(settings.TRAIN_WINDOW + 500, 1500) :]
        )
        plan = self.adaptive.plan(latest["draw_term"], history=len(snapshot))
        plan["target_date"] = at.date().isoformat()
        # Only compact, already settled model statistics cross the process boundary.
        plan["association_performance"] = [
            dict(model=name, stars=k, samples=count, mean_hits=mean)
            for k in (3, 4, 5)
            for name, (count, mean) in _performance_summary(self.outcomes, k).items()
            if name in MODEL_NAMES
        ]
        if not settings.ADAPTIVE_ENABLED:
            plan["shadow_windows"] = []
            plan["champions"] = {
                str(k): {n: settings.TRAIN_WINDOW for n in MODEL_NAMES}
                for k in (3, 4, 5)
            }
            plan["profiles_by_star"] = {}
        if getattr(self.workers, "adaptive_scoring", False):
            result = await self.workers.score(snapshot, plan=plan)
        else:
            result = await self.workers.score(snapshot)
        if result.get("warning") and not self.workers.busy:
            self.workers.training.update(
                state="failed", scope="historical", error=result["warning"]
            )
            self.workers.training.pop("target_date", None)
        self._profile_status = result.get("profile_status", {})
        forecasts, shadows, audits, by_star = [], [], [], {}
        historical_scores = {}
        historical_supports = {}
        default_scores = result["scores"]
        for stars in (3, 4, 5):
            key = str(stars)
            valid = {}
            for name, vector in (
                result.get("by_star_scores", {}).get(key, default_scores).items()
            ):
                try:
                    valid[name] = normalize_scores(vector)
                except (ValueError, TypeError) as exc:
                    logger.warning("排除無效模型 {}：{}", name, exc)
            if not valid:
                raise ValueError("無可用模型評分，尚未封存")
            metadata = {}
            for name, vector in valid.items():
                ranking = sorted(range(1, 81), key=lambda n: (-vector[n - 1], n))
                audit = context_audit(result, snapshot, name, stars, plan)
                metadata[name] = audit
                forecasts.append(
                    dict(
                        model=name,
                        stars=stars,
                        numbers=ranking[:stars],
                        scores=vector,
                        version=audit["model_version"],
                        method="individual",
                    )
                )
                audits.append(dict(model=name, stars=stars, metadata=audit))
            pending = self._profile_status.get(key, {}).get("status") in (
                "pending",
                "failed",
            )
            method = (
                "shrink"
                if pending
                else plan.get("vote_methods", {}).get(key, self.strategy["method"])
            )
            fused_many = fuse_many(valid, self.outcomes)[key]
            selected = fused_many[method]
            historical_scores[key] = selected["scores"]
            historical_supports[key] = selected["support"]
            common = dict(
                training_cutoff=max(
                    (r["training_cutoff"] for r in metadata.values()), key=int
                ),
                feature_cutoff=latest["draw_term"],
                history_start=min(
                    (r["history_start"] for r in metadata.values()), key=int
                ),
                history_end=latest["draw_term"],
                history_count=max(r["history_count"] for r in metadata.values()),
                model_version=result["version"],
                feature_version=result.get(
                    "feature_version", "bingo-features-causal-v1"
                ),
                policy_version=plan["version"],
                model_windows={n: r["window"] for n, r in metadata.items()},
                input_hash=result["snapshot_hash"],
                scoring_cost_seconds=sum(
                    (r.get("scoring_cost_seconds") or 0) for r in metadata.values()
                )
                / len(metadata),
                strategy=method,
            )
            for voting, fused in fused_many.items():
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
                audits.append(
                    dict(
                        model=voting,
                        stars=stars,
                        metadata={**common, "vote_strategy": voting},
                    )
                )
            self._weights[key] = selected["weights"]
            numbers = selected["ranking"][:stars]
            ranking = [
                dict(
                    number=n,
                    score=selected["scores"][n - 1],
                    support=selected["support"][str(n)],
                )
                for n in selected["ranking"]
            ]
            by_star[key] = dict(
                numbers=numbers,
                scores=[selected["scores"][n - 1] for n in numbers],
                weights=selected["weights"],
                confidence=sum(selected["scores"][n - 1] for n in numbers) / stars,
                support=[selected["support"][str(n)] for n in numbers],
                ranking=ranking,
                method=method,
                model_windows=common["model_windows"],
            )
            if stars == 3:
                full_ranking = ranking
            # Each alternative context is independently scored before the answer.
            for window, context in (
                result.get("contexts_by_star", {})
                .get(key, result.get("contexts", {}))
                .items()
            ):
                w = int(window)
                required = set(plan.get("shadow_windows", [])) | {
                    r
                    for entry in plan.get("by_model", {}).values()
                    for r in entry.get("rollback_windows", [])
                }
                if w not in required:
                    continue
                for name, vector in context["scores"].items():
                    if name not in valid or metadata[name]["window"] == w:
                        continue
                    order = sorted(range(1, 81), key=lambda n: (-vector[n - 1], n))
                    audit = context_audit(result, snapshot, name, stars, plan, window=w)
                    shadows.append(
                        dict(
                            model=name,
                            stars=stars,
                            window=w,
                            profile="window-" + str(w),
                            numbers=order[:stars],
                            scores=vector,
                            metadata=audit,
                            version=audit["model_version"],
                            method="shadow",
                        )
                    )
            previous = result.get("previous_policy_scores", {}).get(key)
            if previous and not pending:
                current = plan["profiles_by_star"][key]
                shadows.append(
                    dict(
                        model="policy",
                        stars=stars,
                        window=current["context_window"],
                        profile="policy:" + current["id"],
                        numbers=numbers,
                        scores=selected["scores"],
                        metadata={**common, "policy_profile_id": current["id"]},
                        version=plan["version"],
                        method=method,
                    )
                )
                prior = previous["profile"]
                prior_fusion = fuse(
                    previous["scores"], prior["method"], stars, self.outcomes
                )
                shadows.append(
                    dict(
                        model="policy",
                        stars=stars,
                        window=prior["context_window"],
                        profile="policy:" + prior["id"],
                        numbers=prior_fusion["ranking"][:stars],
                        scores=prior_fusion["scores"],
                        metadata={
                            **common,
                            "policy_profile_id": prior["id"],
                            "scoring_cost_seconds": previous["scoring_cost_seconds"],
                        },
                        version=prior["id"],
                        method=prior["method"],
                    )
                )
        research_predictions = {}
        today = result.get("today")
        if today is not None:
            today_fusions = {
                str(k): fuse(today["scores"], "equal", k) for k in (3, 4, 5)
            }
            today_scores = {k: v["scores"] for k, v in today_fusions.items()}
            research_predictions = self.families.score(
                today_scores, historical_scores, today["history_count"]
            )
            historic_snapshot = snapshot[-settings.TRAIN_WINDOW :]
            dates = sorted(
                {
                    to_taipei(d["draw_datetime"]).date().isoformat()
                    for d in historic_snapshot
                }
            )
            today_audit = dict(
                data_scope="target_day_only",
                target_date=today["target_date"],
                history_dates=today["history_dates"],
                history_start=today["history_start"],
                history_end=today["history_end"],
                history_count=today["history_count"],
                training_cutoff=today.get("training_cutoff"),
                feature_cutoff=today["history_end"],
                model_version=today["version"],
                feature_version="target-day-features-v1",
                input_hash=today["snapshot_hash"],
                warmup_prior=today["warmup"],
                models_used=sorted(today["scores"]),
                scoring_cost_seconds=today["scoring_cost_seconds"],
            )
            history_audit = dict(
                data_scope="cross_day_history",
                history_dates=dates,
                history_start=historic_snapshot[0]["draw_term"],
                history_end=latest["draw_term"],
                history_count=len(historic_snapshot),
                training_cutoff=result.get("training_cutoff", latest["draw_term"]),
                feature_cutoff=latest["draw_term"],
                model_version=result["version"],
                feature_version=result.get("feature_version"),
                input_hash=result["snapshot_hash"],
            )
            for stars in (3, 4, 5):
                key = str(stars)
                for name, pred in research_predictions[key].items():
                    info = dict(
                        today_audit if name == "today" else history_audit,
                        research_strategy=name,
                        today_weight=pred["today_weight"],
                        history_weight=pred["history_weight"],
                        weight_reason=pred["reason"],
                    )
                    if name == "hybrid":
                        info.update(
                            today_input_hash=today["snapshot_hash"],
                            historical_input_hash=result["snapshot_hash"],
                        )
                    info["window"] = info["history_count"]
                    shadows.append(
                        dict(
                            model="strategy_" + name,
                            stars=stars,
                            window=max(1, info["history_count"]),
                            profile="research-" + name,
                            numbers=pred["numbers"],
                            scores=pred["scores"],
                            metadata=info,
                            version="prospective-strategies-v1:"
                            + str(info["model_version"]),
                            method="prospective_research",
                        )
                    )
                champion = self.families.state["champions"][key]
                if champion != "historical":
                    chosen = research_predictions[key][champion]
                    production_info = dict(
                        today_audit if champion == "today" else history_audit,
                        production_strategy=champion,
                    )
                    if champion == "hybrid":
                        production_info.update(
                            today_input_hash=today["snapshot_hash"],
                            historical_input_hash=result["snapshot_hash"],
                        )
                    forecasts.append(
                        dict(
                            model="production",
                            stars=stars,
                            numbers=chosen["numbers"],
                            scores=chosen["scores"],
                            version="prospective-strategies-v1",
                            method="strategy_" + champion,
                            weights={
                                "today": chosen["today_weight"],
                                "historical": chosen["history_weight"],
                            },
                        )
                    )
                    audits.append(
                        dict(
                            model="production",
                            stars=stars,
                            metadata=production_info,
                        )
                    )
                    by_star[key].update(
                        numbers=chosen["numbers"],
                        scores=[chosen["scores"][n - 1] for n in chosen["numbers"]],
                        confidence=sum(
                            chosen["scores"][n - 1] for n in chosen["numbers"]
                        )
                        / stars,
                        ranking=[
                            dict(number=n, score=chosen["scores"][n - 1], support=None)
                            for n in chosen["ranking"]
                        ],
                        method="strategy_" + champion,
                        support=[
                            today_fusions[key]["support"][str(n)]
                            if champion == "today"
                            else int(n in research_predictions[key]["today"]["numbers"])
                            + int(
                                n in research_predictions[key]["historical"]["numbers"]
                            )
                            for n in chosen["numbers"]
                        ],
                        support_kind="模型" if champion == "today" else "策略",
                    )
                    if stars == 3:
                        full_ranking = by_star[key]["ranking"]
        associations = result.get("associations")
        if associations:
            for stars in (3, 4, 5):
                for label, candidate in associations["by_star"][str(stars)].items():
                    history = candidate.get("history", {})
                    info = dict(
                        candidate,
                        training_cutoff=latest["draw_term"],
                        feature_cutoff=latest["draw_term"],
                        model_version=result["version"],
                        feature_version=associations["feature_version"],
                        policy_version=associations["version"],
                        input_hash=result["snapshot_hash"],
                        history_start=history.get(
                            "start_term", history.get("from", snapshot[0]["draw_term"])
                        ),
                        history_end=latest["draw_term"],
                        history_count=history.get("count", min(1000, len(snapshot))),
                        association_label=label,
                        rule_version=associations["version"],
                        models_used=sorted(candidate.get("model_weights", {}))
                        if label == "C"
                        else [],
                        model_training_cutoffs={
                            r["model"]: r["metadata"]["training_cutoff"]
                            for r in audits
                            if r["stars"] == stars
                            and r["model"] in candidate.get("model_weights", {})
                        }
                        if label == "C"
                        else {},
                        model_versions={
                            r["model"]: r["metadata"]["model_version"]
                            for r in audits
                            if r["stars"] == stars
                            and r["model"] in candidate.get("model_weights", {})
                        }
                        if label == "C"
                        else {},
                    )
                    shadows.append(
                        dict(
                            model="association_" + label.lower(),
                            stars=stars,
                            window=max(1, int(info["history_count"])),
                            profile=associations["version"],
                            numbers=candidate["numbers"],
                            scores=candidate["scores"],
                            metadata=info,
                            version=associations["version"]
                            + ":"
                            + hashlib.sha256(
                                json.dumps(
                                    info["model_versions"] or result["version"],
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest()[:24],
                            method="association_research",
                        )
                    )
        methods = {r["method"] for r in by_star.values()}
        payload = dict(
            method=next(iter(methods)) if len(methods) == 1 else "per_star",
            methods_by_star={k: r["method"] for k, r in by_star.items()},
            weights=copy.deepcopy(self._weights),
            by_star=by_star,
            ranking=full_ranking,
            snapshot_hash=result["snapshot_hash"],
            source_term=latest["draw_term"],
            target_time=at.isoformat(),
            model_version=result["version"],
            feature_version=result.get("feature_version", "bingo-features-causal-v1"),
            policy_version=plan["version"],
            profile_status=self._profile_status,
            shadow_count=len(shadows),
            research_strategies=research_predictions,
            associations=associations,
            association_error=result.get("association_error"),
            today_status={
                k: v
                for k, v in today.items()
                if k not in ("scores", "contexts", "by_star_scores")
            }
            if today is not None
            else None,
        )
        self.prediction = await self.store.archive(
            term,
            at,
            latest["draw_term"],
            latest["draw_datetime"],
            payload,
            forecasts,
            now=self.clock(),
            audit=audits,
            shadows=shadows,
        )
        self.boards = live_rankings_summary(
            self._outcome_summary, weights=self._weights
        )
        return self.prediction

    async def _process_research(self):
        pending = await self.store.pending_research_settlements(limit_terms=30)
        for target in pending["targets"]:
            term = target["target_term"]
            if self.adaptive.state["last_target"] and int(term) <= int(
                self.adaptive.state["last_target"]
            ):
                await self.store.commit_research_update(
                    term,
                    {},
                    dict(
                        target_term=term,
                        late_result=True,
                        reason="遲到舊結果僅保留診斷，不回寫已推進的策略狀態",
                    ),
                    [],
                )
                continue
            backup = (
                self.adaptive.to_dict(),
                self.performance.to_dict(),
                self.families.to_dict(),
                self.association_tracker.to_dict(),
            )
            try:
                if self.snapshot:
                    self.adaptive.plan(
                        self.snapshot[-1]["draw_term"], history=len(self.snapshot)
                    )
                report = self.performance.observe(
                    term,
                    target["live_rows"]
                    + [
                        r
                        for r in target["shadow_rows"]
                        if r["model"].startswith(("strategy_", "association_"))
                    ],
                    sync=self.sync,
                    model_statuses=self.workers.models,
                )
                live = []
                for row in target["live_rows"]:
                    if row["model"] not in MODEL_NAMES:
                        continue
                    value = {
                        **row,
                        "window": row.get("window")
                        or self.adaptive.config["baseline_window"],
                    }
                    live.append(value)
                candidates = [
                    r for r in target["shadow_rows"] if r["model"] in MODEL_NAMES
                ]
                update = self.adaptive.observe(
                    term, live, candidates, diagnostics=report
                )
                events = update.get(
                    "events", []
                ) + self.adaptive.observe_profile_outcomes(term, target["shadow_rows"])
                events += self.families.observe(term, target["shadow_rows"])
                events += self.association_tracker.observe(term, target["shadow_rows"])
                compact = {k: v for k, v in report.items() if k != "models"}
                compact["models"] = [
                    {
                        k: r.get(k)
                        for k in (
                            "model",
                            "name",
                            "stars",
                            "drift",
                            "score_distribution",
                            "window",
                            "repetition_reason",
                        )
                    }
                    for r in report.get("models", [])
                ]
                await self.store.commit_research_update(
                    term,
                    {
                        "adaptive": self.adaptive.to_dict(),
                        "monitor": self.performance.to_dict(),
                        "families": self.families.to_dict(),
                        "associations": self.association_tracker.to_dict(),
                    },
                    compact,
                    events,
                )
                self._monitor_view = self.performance.status()
            except Exception:
                self.adaptive = AdaptiveController(backup[0])
                self.performance = PerformanceMonitor(backup[1])
                self.families = ResearchStrategies(backup[2])
                self.association_tracker = AssociationTracker(backup[3])
                raise
        self._strategy_events = await self.store.strategy_events(limit=30)
        self._diagnostics = await self.store.latest_diagnostics(limit=30)

    async def _maybe_evaluate(self):
        if (
            not settings.AUTO_VALIDATE
            or not settings.SCRAPER_ENABLED
            or not getattr(self.workers, "adaptive_scoring", False)
            or self.workers.busy
            or len(self.snapshot) < 300
        ):
            return
        if self._validation_at:
            since = (
                self.clock() - to_taipei(datetime.fromisoformat(self._validation_at))
            ).total_seconds()
            if since < settings.VALIDATION_COOLDOWN_SECONDS:
                return
        # A fixed bootstrap comparison, followed by monitored bounded re-evaluation.
        status = self.performance.status()
        if self._validation_source and not status.get("evaluation_requested"):
            return
        await self.start_backtest(
            {
                "count": min(
                    settings.VALIDATION_COUNT, max(100, len(self.snapshot) - 230)
                ),
                "mode": "nested",
            }
        )

    def intelligence(self):
        adaptive = self.adaptive.status()
        statuses = {r["name"]: r for r in self.workers.models}
        for item in adaptive["models"]:
            item["training_window"] = statuses.get(item["name"], {}).get(
                "training_history_count"
            )
            phase = self._profile_status.get(str(item["stars"]), {})
            if phase.get("status") in ("pending", "failed"):
                item["reason"] = phase.get("reason", "候選工件尚未就緒，固定基準接手")
        monitor = copy.deepcopy(self._monitor_view)
        evaluating = bool(
            self.workers.busy
            and getattr(self.workers, "active_kind", None) == "backtest"
        )
        monitor["evaluating"] = evaluating
        return dict(
            adaptive=adaptive,
            monitor=monitor,
            events=self._strategy_events,
            diagnostics=self._diagnostics,
            research_strategies={
                **self.families.status(),
                "predictions": (self.prediction or {}).get("research_strategies", {}),
                "today_status": (self.prediction or {}).get("today_status"),
            },
            resources=self.resources.sample(self.workers),
            jobs=dict(
                training=self.workers.training,
                backtest={
                    k: v for k, v in self.workers.backtest.items() if k != "report"
                },
                evaluation={"state": "running" if evaluating else "waiting"},
            ),
        )

    def neighbors(self, *, window=1000, lag=1, snapshot=0, source=None):
        """Schedule one lazy research bundle; never await statistics in HTTP."""
        return self.neighbor_research.read(
            self.snapshot, window=window, lag=lag, snapshot=snapshot, source=source
        )

    def associations(self):
        """Serve only the immutable bundle; never mine rules in an HTTP request."""
        prediction = self.prediction or {}
        frozen = copy.deepcopy(prediction.get("associations") or {})
        tracked = self.association_tracker.status()
        boards = {}
        for stars, group in tracked["stars"].items():
            rows = group["leaderboard"]
            boards[stars] = {
                "cumulative": rows,
                **{
                    "recent" + str(window): [
                        {
                            **r["recent"][str(window)],
                            "model": r["model"],
                            "name": r["name"],
                            "id": r["id"],
                        }
                        for r in rows
                    ]
                    for window in (100, 300)
                },
            }
        return dict(
            frozen,
            source_term=prediction.get("source_term"),
            target_term=prediction.get("target_term"),
            target_time=prediction.get("target_time"),
            sealed_at=prediction.get("created_at"),
            leaderboards=boards,
            observations=tracked["observations"],
            diagnostics=tracked["diagnostics"],
            revalidation_requests=tracked["revalidation_requests"],
            error=prediction.get("association_error"),
            model_comparison=self.boards,
            automatic_adoption=False,
            state="sealed" if frozen else "waiting",
        )

    async def _maybe_train(self):
        if (
            not settings.AUTO_TRAIN
            or self._auto_paused
            or self.workers.busy
            or len(self.snapshot) < settings.HISTORY_MIN_DRAWS
        ):
            return
        now = self.clock()
        # Today Only has its own artifact namespace. Its failure must not make
        # the healthy cross-day model enter the Historical repair loop.
        historical_failed = (
            self.workers.training.get("state") == "failed"
            and self.workers.training.get("scope") != "today"
        )
        if (
            self._last_train_term == self.snapshot[-1]["draw_term"]
            and self._last_train_time
            and not historical_failed
        ):
            await self._maybe_today_train()
            return
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
                historical_failed
                and (now - self._last_train_time).total_seconds() >= 60
            )
            if new_draws < settings.RETRAIN_MIN_DRAWS and not failed_retry:
                # Preserve a pending Historical repair during its60s backoff;
                # a Today job must not overwrite that failure's scope.
                if not historical_failed:
                    await self._maybe_today_train()
                return
        await self.start_training()

    async def _maybe_today_train(self):
        if self.workers.busy or not getattr(self.workers, "adaptive_scoring", False):
            return
        _, target_at = next_draw(
            self.snapshot[-1]["draw_term"], self.snapshot[-1]["draw_datetime"]
        )
        day = target_at.date().isoformat()
        today_failed = (
            self.workers.training.get("state") == "failed"
            and self.workers.training.get("scope") == "today"
            and self.workers.training.get("target_date", day) == day
        )
        if (
            today_failed
            and self._last_today_train_date == day
            and self._last_today_train_time is not None
            and (self.clock() - self._last_today_train_time).total_seconds() < 60
        ):
            return
        today = [
            d
            for d in self.snapshot
            if to_taipei(d["draw_datetime"]).date().isoformat() == day
        ]
        if len(today) < 10:
            return
        path = Path(self.workers.directory) / "today" / day / "current.json"
        train = True
        if path.exists():
            try:
                old = json.loads(path.read_text())
                indexes = [
                    i
                    for i, d in enumerate(today)
                    if d["draw_term"] == old["source_term"]
                ]
                seen = indexes[0] + 1 if indexes else 0
                maturity = any(seen < n <= len(today) for n in (10, 32, 80, 96))
                train = (
                    today_failed
                    or maturity
                    or len(today) - seen >= 36
                    or old.get("context_limit") != self.workers.window
                    or old.get("source_dates") != [day]
                )
            except (ValueError, KeyError, OSError):
                pass
        if train:
            self._last_today_train_time = self.clock()
            self._last_today_train_date = day
            await self.workers.start(
                "today_train", copy.deepcopy(today), {"target_date": day}
            )

    async def start_training(self):
        snapshot = copy.deepcopy(self.snapshot[-settings.TRAIN_WINDOW :])
        profiles = [
            v
            for v in self.adaptive.state["profiles_by_star"].values()
            if v.get("id") not in ("fixed", "fixed-v1")
        ]
        if profiles and getattr(self.workers, "adaptive_scoring", False):
            result = await self.workers.start(
                "train", snapshot, {"validated_profiles": profiles}
            )
        else:
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
        mode = options.get("mode", "fixed")
        if mode not in ("fixed", "adaptive", "nested", "associations", "neighbors"):
            raise ValueError("未知回測模式")
        if mode != "fixed" and options.get("count", 100) > 1000:
            raise ValueError("巢狀比較每次最多1000期")
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
        if mode != "neighbors":
            self._validation_at = self.clock().isoformat()
            self._validation_source = draws[-1]["draw_term"] if draws else None
            await self.store.save_research_state(
                "validation",
                {
                    "last_evaluation_at": self._validation_at,
                    "source_term": self._validation_source,
                },
            )
        await self.store.save_evaluation(result["run_id"], "running", None)
        return result

    async def _backtest_done(self, state):
        report = state.get("report")
        await self.store.save_evaluation(
            state["run_id"], state["state"], report, state.get("error")
        )
        if report and report.get("mode") != "neighbor_walk_forward" and report.get("requested_mode") != "neighbors":
            async with self.lock:
                before = self.adaptive.to_dict()
                old_strategy = copy.deepcopy(self.strategy)
                last_train = self._last_train_time
                try:
                    outcome = self._apply_strategy(report)
                    await self.store.record_validation_update(
                        self.adaptive.state["known_source_term"],
                        {"adaptive": self.adaptive.to_dict()},
                        outcome.get("events", []),
                    )
                    self._strategy_events = await self.store.strategy_events(limit=30)
                except Exception:
                    self.adaptive = AdaptiveController(before)
                    self.strategy = old_strategy
                    self._last_train_time = last_train
                    raise

    def _apply_strategy(self, report, restore=False):
        if report.get("mode") == "neighbor_walk_forward" or report.get("requested_mode") == "neighbors":
            # Research must not spend the production controller's alpha budget,
            # advance its cooldown, change its status or acquire fusion weights.
            return {"events": []}
        # Old fixed reports remain inspectable; they cannot bypass phase-two
        # nested confirmation, cooldown, repeated-search or shadow safeguards.
        outcome = {"events": []}
        if not restore:
            outcome = self.adaptive.accept_validation(report)
            if any(e["kind"] == "profile_validated" for e in outcome["events"]):
                self._last_train_time = None
        self.strategy = dict(
            method="shrink",
            status=report.get("decision", {}).get(
                "reason", "固定基準維持正式策略；候選尚待獨立驗證"
            ),
            comparison=report.get("method_comparison"),
            validated=False,
        )
        return outcome

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
            # Rule evidence has its own slower, on-demand endpoint. Avoid
            # retransmitting the complete rule tree on every5s status poll.
            prediction={k: v for k, v in self.prediction.items() if k != "associations"}
            if self.prediction
            else None,
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
        await self.neighbor_research.close()
        await self.workers.close()
