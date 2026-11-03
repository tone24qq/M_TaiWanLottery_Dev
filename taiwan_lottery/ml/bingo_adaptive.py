"""Causal, bounded champion/challenger governance for Bingo history windows.

A window can be *scored* from any past prefix, but it cannot become a live
champion merely by winning a search. Selection samples and prospective shadow
validation samples are disjoint. All decisions affect only later targets.

This module never changes numbers, archived records, or a model artifact. It
chooses deterministic contexts and records evidence; the runtime persists that
state beside its separate live and shadow ledgers.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

POLICY_VERSION = "adaptive-window-1"
FEATURE_VERSION = "bingo-features-causal-v1"
CANDIDATE_WINDOWS = (5, 10, 17, 20, 30, 50, 60, 100, 200, 300, 500, 1000)
STARS = (3, 4, 5)
MODEL_NAMES = (
    "frequency",
    "cdm",
    "river",
    "lightgbm",
    "catboost",
    "transformer",
    "dqn",
    "ensemble",
)
_DEFAULTS = {
    "min_selection": 30,
    "min_validation": 100,
    "cooldown": 30,
    "shadow_budget": 3,
    "min_improvement": 0.03,
    "max_overhead_ratio": 1.5,
    "max_extra_seconds": 0.25,
    "alpha": 0.05,
    "baseline_window": 1000,
    "min_history": 100,
    "simulation": False,
}


def _term(value: Any) -> str:
    text = str(value)
    if not text.isdigit() or not 0 < int(text) or len(text) > 20:
        raise ValueError("期號必須為正整數字串")
    return text


def _stamp(value: Any) -> datetime:
    stamp = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if not isinstance(stamp, datetime):
        raise ValueError("必須提供封存時間與目標開獎時間")
    return (
        stamp.replace(tzinfo=ZoneInfo("Asia/Taipei")) if stamp.tzinfo is None else stamp
    )


def _window(value: Any) -> int:
    try:
        valid = (
            not isinstance(value, bool)
            and int(value) == value
            and 5 <= int(value) <= 10000
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("歷史視窗必須為 5～10000 的整數")
    return int(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _mean(values):
    return float(np.mean(values)) if values else 0.0


class AdaptiveController:
    """Serializable, deterministic per-model/per-star context controller.

    ``plan`` is called before generating the next forecast. ``observe`` receives
    only rows whose decisions were archived before their target was revealed.
    First 30 paired samples select a candidate; at least 100 *later* paired
    samples validate it. Hoeffding confidence bounds spend a summable alpha
    budget over all attempted comparisons, guarding repeated searches.
    """

    def __init__(self, state: dict | None = None, **options):
        unknown = set(options) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"未知自適應設定：{sorted(unknown)}")
        config = {**_DEFAULTS, **(state or {}).get("config", {}), **options}
        if (
            config["min_selection"] < 30
            or config["min_validation"] < 100
            or config["cooldown"] < 30
        ):
            raise ValueError("至少需要 30 期選擇、100 期獨立驗證及 30 期冷卻")
        if not 2 <= config["shadow_budget"] <= 3:
            raise ValueError("影子視窗預算必須為 2 或 3")
        if not 0 < config["alpha"] <= 0.05 or not 0.0 < config["min_improvement"] <= 1:
            raise ValueError("統計門檻無效")
        if (
            not 1 <= config["max_overhead_ratio"] <= 2
            or config["max_extra_seconds"] <= 0
        ):
            raise ValueError("資源成本限制無效")
        _window(config["baseline_window"])
        if state and state.get("policy_version") != POLICY_VERSION:
            raise ValueError("自適應狀態版本不相容")
        self.config = config
        self.state = (
            copy.deepcopy(state)
            if state
            else {
                "policy_version": POLICY_VERSION,
                "feature_version": FEATURE_VERSION,
                "config": config,
                "observation_count": 0,
                "selection_count": 0,
                "comparison_count": 0,
                "revision": 0,
                "last_target": None,
                "last_target_time": None,
                "known_source_term": None,
                "available_history": None,
                "contexts": {},
                "recent_events": [],
                "processed": {},
                "accepted_reports": [],
                "profiles_by_star": {str(k): self._fixed_profile() for k in STARS},
            }
        )
        self.state["config"] = copy.deepcopy(config)
        self.state.setdefault("reviewed_reports", [])
        self.state.setdefault("profile_previous_by_star", {str(k): [] for k in STARS})
        self.state.setdefault(
            "profile_last_change_index", {str(k): -config["cooldown"] for k in STARS}
        )
        self.state.setdefault("profile_validation_ranges", {str(k): [] for k in STARS})

    def _fixed_profile(self):
        return {
            "id": "fixed-v1",
            "context_window": self.config["baseline_window"],
            "train_window": self.config["baseline_window"],
            "method": "shrink",
            "frequency_mix": [0.35, 0.30, 0.20, 0.15],
            "cdm_prior_kind": "uniform",
            "cdm_prior": 8.0,
            "retrain_every": 36,
            "validated_baseline": True,
        }

    def _context(self, model, stars):
        if model not in MODEL_NAMES or stars not in STARS:
            raise ValueError("未知模型或星數")
        key = f"{model}:{stars}"
        if key not in self.state["contexts"]:
            self.state["contexts"][key] = {
                "model": model,
                "stars": stars,
                "window": self.config["baseline_window"],
                "version": "fixed-v1",
                "reason": "固定視窗基準；尚無獨立驗證優勢證據",
                "last_change": None,
                "last_change_index": -self.config["cooldown"],
                "previous": [],
                "arms": {},
                "rollback": [],
            }
        return self.state["contexts"][key]

    @property
    def version(self):
        return f"{POLICY_VERSION}.{self.state['revision']}"

    def _cohort(self):
        size = self.config["min_selection"] + self.config["min_validation"]
        return self.state["observation_count"] // size

    def _windows(self):
        baseline = self.config["baseline_window"]
        values = [w for w in CANDIDATE_WINDOWS if w != baseline]
        slots = self.config["shadow_budget"] - 1
        cohort = self._cohort()
        chosen = [values[(cohort * slots + i) % len(values)] for i in range(slots)]
        # Explore one deterministic interior neighbour only after the fixed grid
        # has received a full bounded search cycle. Never launch an open search.
        if cohort >= math.ceil(len(values) / slots):
            base = chosen[-1]
            position = CANDIDATE_WINDOWS.index(base)
            other = CANDIDATE_WINDOWS[min(position + 1, len(CANDIDATE_WINDOWS) - 1)]
            neighbour = int(round((base + other) / 2))
            if neighbour != base and neighbour not in chosen:
                chosen[-1] = neighbour
        return [baseline] + chosen

    def plan(self, source_term, history=None, model_names=None, **metadata):
        """Return bounded contexts without looking at an unrevealed target.

        ``window`` is requested context length; ``effective_window`` is limited
        by available source history. Training window is separately recorded by
        the model artifact and is never silently re-labelled as context length.
        """
        source = _term(source_term)
        previous = self.state.get("known_source_term")
        if previous and int(source) < int(previous):
            raise ValueError("不能使用較新的策略狀態回頭規劃過去期數")
        self.state["known_source_term"] = source
        if history is not None:
            available = len(history) if not isinstance(history, int) else history
            if available < 0:
                raise ValueError("歷史期數不可為負")
            self.state["available_history"] = available
        available = self.state.get("available_history")
        models = list(MODEL_NAMES if model_names is None else model_names)
        if len(set(models)) != len(models) or any(m not in MODEL_NAMES for m in models):
            raise ValueError("模型名稱重複或無效")
        windows = self._windows()
        champions = {str(k): {} for k in STARS}
        by_model = {}
        for model in models:
            current, previous_windows = [], []
            details = {}
            for stars in STARS:
                context = self._context(model, stars)
                requested = context["window"]
                if available is not None and available < self.config["min_history"]:
                    requested = self.config["baseline_window"]
                champions[str(stars)][model] = requested
                current.append(requested)
                if context["previous"]:
                    previous_windows.append(context["previous"][-1]["window"])
                details[str(stars)] = {
                    "window": requested,
                    "effective_window": min(requested, available)
                    if available is not None
                    else requested,
                    "reason": (
                        "歷史資料不足，回退固定基準"
                        if available is not None
                        and available < self.config["min_history"]
                        else context["reason"]
                    ),
                    "version": context["version"],
                }
            by_model[model] = {
                "stars": details,
                "shadow_windows": list(windows),
                "rollback_windows": sorted(set(previous_windows)),
                "required_windows": sorted(set(windows + current + previous_windows)),
            }
        return {
            "source_term": source,
            "version": self.version,
            "feature_version": FEATURE_VERSION,
            "cohort": self._cohort(),
            "champions": champions,
            "shadow_windows": windows,
            "by_model": by_model,
            "selection_count": self.state["selection_count"],
            "profiles_by_star": copy.deepcopy(self.state["profiles_by_star"]),
            "previous_profiles_by_star": {
                k: copy.deepcopy(v[-1]) if v else None
                for k, v in self.state["profile_previous_by_star"].items()
            },
            "vote_methods": {
                k: v["method"] for k, v in self.state["profiles_by_star"].items()
            },
            "feature_weights": {
                k: v["frequency_mix"] for k, v in self.state["profiles_by_star"].items()
            },
            "reason": "候選視窗僅影子評估；通過分離選擇／驗證、資源與冷卻門檻後才更動未來預測",
            "bounds": {
                "shadow_contexts_per_model": self.config["shadow_budget"],
                "maximum_contexts_per_model": 9,
            },
        }

    def _canonical(self, row, target, *, shadow):
        value = {
            **(row.get("metadata") or {}),
            **(row.get("audit") or {}),
            **{key: item for key, item in row.items() if item is not None},
        }
        model, stars = value.get("model", value.get("name")), int(value["stars"])
        if model not in MODEL_NAMES or stars not in STARS:
            raise ValueError("自適應結果包含未知模型或星數")
        if _term(value.get("target_term", target)) != target:
            raise ValueError("結算期號與觀測目標不一致")
        source = _term(value["source_term"])
        known = self.state.get("known_source_term")
        if known is not None and int(source) > int(known):
            raise ValueError("預測來源超出當時已知資料截止期")
        if int(source) >= int(target):
            raise ValueError("預測來源不得包含當期或未來答案")
        created = _stamp(value.get("created_at", value.get("archived_at")))
        when = _stamp(value.get("target_time", value.get("draw_datetime")))
        if created >= when:
            raise ValueError("只能使用開獎前封存的候選")
        if (
            value.get("source_time") is not None
            and _stamp(value["source_time"]) >= when
        ):
            raise ValueError("特徵來源時間不得晚於目標")
        for key in (
            "training_cutoff_term",
            "trained_through",
            "feature_cutoff_term",
            "features_through",
            "training_cutoff",
            "feature_cutoff",
        ):
            if value.get(key) is not None and int(_term(value[key])) > int(source):
                raise ValueError("訓練或特徵截止期包含未來資料")
        if not self.config["simulation"] and (
            value.get("is_live") is False
            or value.get("mode")
            in (
                "backtest",
                "simulation",
                "strict_walk_forward",
                "nested_walk_forward",
            )
        ):
            raise ValueError("歷史模擬不可寫入正式自適應實績")
        hits = value.get("hit_count")
        if (
            isinstance(hits, bool)
            or not isinstance(hits, (int, np.integer))
            or not 0 <= hits <= stars
        ):
            raise ValueError("命中數無效")
        if shadow and "window" not in value and "context_window" not in value:
            raise ValueError("影子候選必須記錄實際視窗")
        window = _window(
            value.get(
                "window", value.get("context_window", self.config["baseline_window"])
            )
        )
        if value.get("numbers") is not None and value.get("actual_numbers") is not None:
            if (
                len(value["numbers"]) != stars
                or len(set(value["numbers"])) != stars
                or len(value["actual_numbers"]) != 20
            ):
                raise ValueError("封存選號或答案長度無效")
            if len(set(value["numbers"]) & set(value["actual_numbers"])) != hits:
                raise ValueError("結算命中數與封存選號不一致")
        cost = value.get("scoring_cost_seconds", value.get("cost_seconds"))
        if cost is not None:
            cost = float(cost)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("推論資源測量必須為非負有限數")
        return {
            "model": model,
            "stars": stars,
            "window": window,
            "hit_count": int(hits),
            "target_term": target,
            "source_term": source,
            "created_at": created.isoformat(),
            "target_time": when.isoformat(),
            "scoring_cost_seconds": cost,
        }

    @staticmethod
    def _blocked(model, diagnostics):
        if not diagnostics:
            return None
        if (
            diagnostics.get("data_valid") is False
            or diagnostics.get("data_ok") is False
        ):
            return "資料驗證未通過，維持既有策略"
        if diagnostics.get("actions", {}).get("repair_data_or_artifact"):
            return "資料或模型工件需先修復，暫停切換"
        for alert in diagnostics.get("alerts", []):
            if alert.get("severity") == "error" and alert.get("model") in (None, model):
                return "資料或模型監控發現錯誤，暫停切換"
        for key in ("blocked_models", "stale_models", "invalid_models"):
            if model in diagnostics.get(key, []):
                return "模型或資料狀態異常，暫停切換"
        return None

    def _pair(self, candidate, champion):
        return {
            "target_term": candidate["target_term"],
            "candidate_hits": candidate["hit_count"],
            "champion_hits": champion["hit_count"],
            "delta": candidate["hit_count"] - champion["hit_count"],
            "candidate_cost": candidate["scoring_cost_seconds"],
            "champion_cost": champion["scoring_cost_seconds"],
        }

    def _evidence(self, pairs, stars, *, spend=False):
        if spend:
            self.state["comparison_count"] += 1
        attempt = max(1, self.state["comparison_count"])
        # Sum_{i>=1} alpha/(i*(i+1)) = alpha; unlimited repeated attempts
        # cannot silently keep using the same nominal 5% significance threshold.
        alpha = self.config["alpha"] / (attempt * (attempt + 1))
        n = len(pairs)
        differences = [r["delta"] for r in pairs]
        mean = _mean(differences)
        width = (
            2 * stars * math.sqrt(math.log(2 / alpha) / (2 * n)) if n else float("inf")
        )
        candidate_full = _mean([r["candidate_hits"] == stars for r in pairs])
        champion_full = _mean([r["champion_hits"] == stars for r in pairs])
        known_cost = bool(pairs) and all(
            r["candidate_cost"] is not None and r["champion_cost"] is not None
            for r in pairs
        )
        candidate_cost = (
            _mean([r["candidate_cost"] for r in pairs]) if known_cost else None
        )
        champion_cost = (
            _mean([r["champion_cost"] for r in pairs]) if known_cost else None
        )
        ratio = (candidate_cost / max(champion_cost, 1e-12)) if known_cost else None
        half = max(1, n // 2)
        stable = (
            n >= 2
            and min(_mean(differences[:half]), _mean(differences[half:]))
            > self.config["min_improvement"]
        )
        return {
            "samples": n,
            "mean_delta": mean,
            "ci_low": mean - width,
            "ci_high": mean + width,
            "delta": mean,
            "lower_bound": mean - width,
            "overhead_ratio": ratio,
            "alpha": alpha,
            "comparison_count": attempt,
            "candidate_mean_hits": _mean([r["candidate_hits"] for r in pairs]),
            "champion_mean_hits": _mean([r["champion_hits"] for r in pairs]),
            "candidate_full_hit_rate": candidate_full,
            "champion_full_hit_rate": champion_full,
            "random_mean": stars * 0.25,
            "stable_halves": stable,
            "resource_measured": known_cost,
            "resource_ratio": ratio,
            "candidate_seconds": candidate_cost,
            "champion_seconds": champion_cost,
            "resource_allowed": known_cost
            and ratio <= self.config["max_overhead_ratio"]
            and candidate_cost - champion_cost <= self.config["max_extra_seconds"],
            "first_target": pairs[0]["target_term"] if pairs else None,
            "last_target": pairs[-1]["target_term"] if pairs else None,
        }

    def _event(self, kind, context, target, **fields):
        event = {
            "type": kind,
            "model": context["model"],
            "stars": context["stars"],
            "target_term": target,
            "effective_after_term": target,
            "policy_version": self.version,
            "feature_version": FEATURE_VERSION,
            **fields,
        }
        event["id"] = _digest(event)
        self.state["recent_events"].append(event)
        self.state["recent_events"] = self.state["recent_events"][-100:]
        return event

    def _promote(self, context, window, target, when, evidence, *, rollback=False):
        old = {
            "window": context["window"],
            "version": context["version"],
            "reason": context["reason"],
            "validated": True,
        }
        if rollback:
            previous = context["previous"].pop()
            window = previous["window"]
        else:
            context["previous"].append(old)
            context["previous"] = context["previous"][-5:]
        self.state["revision"] += 1
        context.update(
            window=window,
            version=self.version,
            last_change=when,
            last_change_index=self.state["observation_count"],
            rollback=[],
            arms={},
            reason=(
                "經獨立已封存影子比較確認退化，回退上一已驗證版本"
                if rollback
                else "獨立影子驗證及資源／冷卻門檻通過，僅適用其後期數"
            ),
        )
        return self._event(
            "window_rollback" if rollback else "window_promoted",
            context,
            target,
            before=old,
            after={"window": window, "version": context["version"]},
            validation=evidence,
            reason=context["reason"],
        )

    def observe(self, settled_target, live_rows, shadow_rows, diagnostics=None):
        """Apply one revealed target exactly once, never learning a future label.

        Persist ``state`` and all returned ``events`` in the same runtime update
        transaction. Replaying identical outcomes after a restart is a no-op;
        conflicting or out-of-order outcomes are rejected.
        """
        target = _term(
            settled_target.get("target_term")
            if isinstance(settled_target, dict)
            else settled_target
        )
        rows = [self._canonical(r, target, shadow=False) for r in live_rows]
        rows += [self._canonical(r, target, shadow=True) for r in shadow_rows]
        rows.sort(key=lambda r: (r["model"], r["stars"], r["window"]))
        unique = {}
        for row in rows:
            identity = (row["model"], row["stars"], row["window"])
            if identity in unique and unique[identity] != row:
                raise ValueError("同一模型／星數／視窗有衝突結果")
            unique[identity] = row
        rows = list(unique.values())
        fingerprint = _digest(rows)
        if target in self.state["processed"]:
            if self.state["processed"][target] != fingerprint:
                raise ValueError("已結算目標出現互相衝突的自適應結果")
            return {
                "updated": False,
                "duplicate": True,
                "events": [],
                "state": self.to_dict(),
            }
        previous = self.state["last_target"]
        if previous is not None and int(target) <= int(previous):
            raise ValueError("自適應結果必須依真實期號逐期前進")
        if not rows:
            return {
                "updated": False,
                "events": [],
                "reason": "無已封存可觀測結果",
                "state": self.to_dict(),
            }
        last_when = self.state.get("last_target_time")
        when = rows[0]["target_time"]
        if any(_stamp(r["target_time"]) != _stamp(when) for r in rows):
            raise ValueError("相同期號的開獎時間不一致")
        if last_when and _stamp(when) <= _stamp(last_when):
            raise ValueError("自適應觀測時間必須遞增")
        grouped = {}
        for row in rows:
            identity = (row["model"], row["stars"], row["window"])
            if identity in grouped:
                if grouped[identity] != row:
                    raise ValueError("同一模型／星數／視窗有衝突結果")
                continue
            grouped[identity] = row
        current_windows = set(self._windows())
        for row in grouped.values():
            context = self._context(row["model"], row["stars"])
            allowed = (
                current_windows
                | {context["window"]}
                | {r["window"] for r in context["previous"][-1:]}
            )
            if row["window"] not in allowed:
                raise ValueError("候選視窗超過當時受限搜尋計畫")
        events = []
        cohort = self._cohort()
        self.state["observation_count"] += 1
        self.state["last_target"] = target
        self.state["last_target_time"] = when
        self.state["known_source_term"] = target
        for key in sorted(self.state["contexts"]):
            context = self.state["contexts"][key]
            model, stars = context["model"], context["stars"]
            champion = grouped.get((model, stars, context["window"]))
            if champion is None:
                continue
            blocked = self._blocked(model, diagnostics)
            if blocked:
                context["reason"] = blocked
            elapsed = self.state["observation_count"] - context["last_change_index"]
            if context["previous"]:
                previous_window = context["previous"][-1]["window"]
                old = grouped.get((model, stars, previous_window))
                if old:
                    context["rollback"].append(self._pair(old, champion))
                    context["rollback"] = context["rollback"][
                        -self.config["min_validation"] :
                    ]
                    if (
                        len(context["rollback"]) >= self.config["min_validation"]
                        and elapsed >= self.config["cooldown"]
                        and not blocked
                    ):
                        evidence = self._evidence(
                            context["rollback"], stars, spend=True
                        )
                        if (
                            evidence["ci_low"] > self.config["min_improvement"]
                            and evidence["resource_allowed"]
                        ):
                            events.append(
                                self._promote(
                                    context,
                                    previous_window,
                                    target,
                                    when,
                                    evidence,
                                    rollback=True,
                                )
                            )
                            continue
            for window in sorted(current_windows - {context["window"]}):
                candidate = grouped.get((model, stars, window))
                if candidate is None:
                    continue
                arm = context["arms"].get(str(window))
                if (
                    arm is None
                    or arm["cohort"] != cohort
                    or arm.get("champion_window") != context["window"]
                ):
                    arm = {
                        "window": window,
                        "cohort": cohort,
                        "champion_window": context["window"],
                        "phase": "selection",
                        "selection": [],
                        "validation": [],
                        "reason": "收集選擇期資料",
                    }
                    context["arms"][str(window)] = arm
                pair = self._pair(candidate, champion)
                if arm["phase"] == "selection":
                    arm["selection"].append(pair)
                    if len(arm["selection"]) >= self.config["min_selection"]:
                        self.state["selection_count"] += 1
                        arm["selection_mean_delta"] = _mean(
                            [r["delta"] for r in arm["selection"]]
                        )
                        arm["selection_end_term"] = target
                        if arm["selection_mean_delta"] > self.config["min_improvement"]:
                            arm.update(
                                phase="validation",
                                reason="選擇期已凍結；等待其後獨立影子驗證",
                            )
                            events.append(
                                self._event(
                                    "candidate_selected",
                                    context,
                                    target,
                                    candidate_window=window,
                                    selection_samples=len(arm["selection"]),
                                    selection_end_term=target,
                                    reason=arm["reason"],
                                )
                            )
                        else:
                            arm.update(
                                phase="rejected",
                                reason="選擇期未見最小改善，維持原策略",
                            )
                    continue
                if arm["phase"] != "validation":
                    continue
                # This branch cannot receive the selection-ending target; it is
                # reached only for later already-archived, revealed targets.
                if int(target) <= int(arm["selection_end_term"]):
                    raise ValueError("選擇與驗證樣本不可重疊")
                arm["validation"].append(pair)
                arm["validation"] = arm["validation"][-self.config["min_validation"] :]
                if len(arm["validation"]) < self.config["min_validation"] or blocked:
                    continue
                evidence = self._evidence(arm["validation"], stars, spend=True)
                arm["evidence"] = evidence
                if elapsed < self.config["cooldown"]:
                    arm["reason"] = "仍在冷卻期，維持原策略"
                    continue
                passes = (
                    evidence["ci_low"] > self.config["min_improvement"]
                    and evidence["stable_halves"]
                    and evidence["candidate_full_hit_rate"]
                    >= evidence["champion_full_hit_rate"]
                    and evidence["resource_allowed"]
                )
                if passes:
                    events.append(
                        self._promote(context, window, target, when, evidence)
                    )
                    break
                arm.update(
                    phase="rejected",
                    reason="獨立驗證未通過保守統計／全中率／資源門檻，維持原策略",
                )
                events.append(
                    self._event(
                        "candidate_rejected",
                        context,
                        target,
                        candidate_window=window,
                        selection_end_term=arm["selection_end_term"],
                        validation=evidence,
                        reason=arm["reason"],
                    )
                )
            # Bound old cohort state; exact ledger evidence remains in SQL.
            context["arms"] = {
                w: a for w, a in context["arms"].items() if a["cohort"] >= cohort - 1
            }
        self.state["processed"][target] = fingerprint
        while len(self.state["processed"]) > 256:
            self.state["processed"].pop(next(iter(self.state["processed"])))
        return {"updated": True, "events": events, "state": self.to_dict()}

    def accept_validation(self, report):
        """Independently audit a completed nested proposal for future-only use.

        Historical evidence can authorize a candidate, but it never contributes
        to live samples. Repeated report reads do not spend another search.
        """
        from taiwan_lottery.ml.bingo_nested import CandidateProfile, paired_gate

        decision = report.get("decision", {})
        fingerprint = _digest(decision)
        if fingerprint in self.state["reviewed_reports"]:
            return {"updated": False, "events": [], "duplicate": True}
        self.state["reviewed_reports"].append(fingerprint)
        self.state["reviewed_reports"] = self.state["reviewed_reports"][-100:]
        self.state["selection_count"] += 1
        events = []
        for stars in STARS:
            key = str(stars)
            proposal = decision.get("proposals", {}).get(key)
            reason = "獨立歷史驗證未證明優勢；正式策略維持不變"
            evidence = None
            accepted = False
            if (
                report.get("mode") == "nested_walk_forward"
                and report.get("is_live") is False
                and decision.get("allowed_for_live")
                and proposal
            ):
                profile = CandidateProfile(**proposal["profile"])
                current = self.state["profiles_by_star"][key]
                pairs = proposal.get("paired", [])
                terms = [int(_term(row["target_term"])) for row in pairs]
                end = proposal.get("selection_end_term")
                valid = bool(
                    terms
                    and end
                    and terms == sorted(set(terms))
                    and int(end) < terms[0]
                )
                known = self.state.get("known_source_term")
                valid &= known is not None and (not terms or terms[-1] <= int(known))
                for row in pairs:
                    source = int(_term(row["source_term"]))
                    target = int(_term(row["target_term"]))
                    valid &= (
                        source < target
                        and int(_term(row["training_cutoff"])) <= source
                        and int(_term(row["feature_cutoff"])) <= source
                    )
                for prior in self.state["profile_validation_ranges"][key]:
                    if terms and not (terms[-1] < prior[0] or terms[0] > prior[1]):
                        valid = False
                valid &= current["id"] in (
                    "fixed",
                    "fixed-v1",
                    proposal.get("baseline_profile_id"),
                )
                valid &= (
                    self.state["observation_count"]
                    - self.state["profile_last_change_index"][key]
                    >= self.config["cooldown"]
                )
                inner = proposal.get("inner_validation", {})
                valid &= (
                    bool(inner.get("approved"))
                    and inner.get("samples", 0) >= self.config["min_validation"]
                )
                evidence = paired_gate(
                    pairs,
                    stars,
                    comparison_count=max(1, int(proposal.get("comparison_count", 1))),
                    attempt=self.state["selection_count"],
                    minimum_samples=self.config["min_validation"],
                    minimum_improvement=self.config["min_improvement"],
                    resource_ratio=float(proposal.get("resource_ratio", float("inf"))),
                )
                if valid and evidence["approved"] and profile.id != "fixed":
                    previous = copy.deepcopy(current)
                    self.state["profile_previous_by_star"][key].append(previous)
                    self.state["profile_previous_by_star"][key] = self.state[
                        "profile_previous_by_star"
                    ][key][-3:]
                    self.state["profiles_by_star"][key] = profile.payload()
                    self.state["profile_last_change_index"][key] = self.state[
                        "observation_count"
                    ]
                    self.state["profile_validation_ranges"][key].append(
                        [terms[0], terms[-1]]
                    )
                    self.state["profile_validation_ranges"][key] = self.state[
                        "profile_validation_ranges"
                    ][key][-100:]
                    self.state["revision"] += 1
                    for model in MODEL_NAMES:
                        context = self._context(model, stars)
                        context.update(
                            window=profile.context_window,
                            arms={},
                            rollback=[],
                            version=self.version,
                            last_change_index=self.state["observation_count"],
                            reason="獨立巢狀驗證通過；等待相符工件背景就緒後套用未來預測",
                        )
                    accepted = True
                    reason = (
                        "獨立巢狀驗證與資源門檻通過；已封存預測不變，候選工件背景準備"
                    )
                elif not valid:
                    reason = "驗證切分、截止期、重複選擇或冷卻條件不符；維持正式策略"
            event = dict(
                kind="profile_validated" if accepted else "profile_retained",
                stars=stars,
                target_term=self.state.get("known_source_term"),
                reason=reason,
                before=current if accepted else self.state["profiles_by_star"][key],
                after=self.state["profiles_by_star"][key],
                validation=evidence,
                report_hash=fingerprint,
                selection_count=self.state["selection_count"],
            )
            events.append(event)
        self.state["recent_events"] = (self.state["recent_events"] + events)[-50:]
        return {"updated": True, "events": events}

    def observe_profile_outcomes(self, target_term, rows):
        """Rollback only after new prearchived paired evidence favors the prior version."""
        from taiwan_lottery.ml.bingo_nested import paired_gate

        tracks = self.state.setdefault("profile_rollback_pairs", {})
        events = []
        for stars in STARS:
            key = str(stars)
            prior = self.state["profile_previous_by_star"][key]
            if not prior:
                continue
            current = self.state["profiles_by_star"][key]
            by_id = {
                row.get("metadata", {}).get("policy_profile_id"): row
                for row in rows
                if row.get("model") == "policy" and row["stars"] == stars
            }
            old, new = by_id.get(prior[-1]["id"]), by_id.get(current["id"])
            if not old or not new:
                continue
            for row in (old, new):
                if _stamp(row["created_at"]) >= _stamp(row["target_time"]) or int(
                    row["source_term"]
                ) >= int(target_term):
                    raise ValueError("回退證據必須事前封存")
            track = tracks.setdefault(key, [])
            if track and track[-1]["target_term"] == str(target_term):
                continue
            track.append(
                dict(
                    target_term=str(target_term),
                    champion_hits=new["hit_count"],
                    challenger_hits=old["hit_count"],
                )
            )
            del track[: -self.config["min_validation"]]
            if (
                len(track) < self.config["min_validation"]
                or self.state["observation_count"]
                - self.state["profile_last_change_index"][key]
                < self.config["cooldown"]
            ):
                continue
            self.state["comparison_count"] += 1
            costold = old.get("metadata", {}).get("scoring_cost_seconds")
            costnew = new.get("metadata", {}).get("scoring_cost_seconds")
            ratio = (
                float(costold) / max(float(costnew), 1e-12)
                if costold is not None and costnew is not None
                else float("inf")
            )
            gate = paired_gate(
                track,
                stars,
                comparison_count=3,
                attempt=self.state["comparison_count"],
                minimum_samples=self.config["min_validation"],
                minimum_improvement=self.config["min_improvement"],
                resource_ratio=ratio,
            )
            if not gate["approved"]:
                continue
            previous = copy.deepcopy(current)
            restored = prior.pop()
            self.state["profiles_by_star"][key] = restored
            self.state["profile_last_change_index"][key] = self.state[
                "observation_count"
            ]
            self.state["revision"] += 1
            for model in MODEL_NAMES:
                self._context(model, stars).update(
                    window=restored["context_window"],
                    arms={},
                    rollback=[],
                    reason="新事前影子驗證證明退化，回退上一已驗證策略",
                    version=self.version,
                )
            tracks[key] = []
            events.append(
                dict(
                    kind="profile_rollback",
                    stars=stars,
                    target_term=str(target_term),
                    before=previous,
                    after=restored,
                    validation=gate,
                    rollback_version=restored["id"],
                    reason="獨立事前配對影子驗證通過，回退僅影響未來預測",
                )
            )
        self.state["recent_events"] = (self.state["recent_events"] + events)[-50:]
        return events

    def status(self):
        models = []
        available = self.state.get("available_history")
        for context in self.state["contexts"].values():
            arms = list(context["arms"].values())
            active = [a for a in arms if a["phase"] in ("selection", "validation")]
            arm = max(
                active or arms,
                key=lambda a: (len(a["validation"]), len(a["selection"])),
                default=None,
            )
            models.append(
                {
                    "model": context["model"],
                    "name": context["model"],
                    "stars": context["stars"],
                    "window": context["window"],
                    "requested_window": context["window"],
                    "effective_window": min(context["window"], available)
                    if available is not None
                    else context["window"],
                    "training_window": None,
                    "reason": context["reason"],
                    "phase": arm["phase"]
                    if arm
                    else ("fixed" if context["version"] == "fixed-v1" else "champion"),
                    "candidate_window": arm["window"] if arm else None,
                    "selection_samples": len(arm["selection"]) if arm else 0,
                    "validation_samples": len(arm["validation"]) if arm else 0,
                    "last_change": context["last_change"],
                    "version": context["version"],
                    "rollback_version": context["previous"][-1]["version"]
                    if context["previous"]
                    else None,
                }
            )
        return {
            "version": self.version,
            "feature_version": FEATURE_VERSION,
            "last_target": self.state["last_target"],
            "observation_count": self.state["observation_count"],
            "selection_count": self.state["selection_count"],
            "comparison_count": self.state["comparison_count"],
            "min_selection": self.config["min_selection"],
            "min_validation": self.config["min_validation"],
            "cooldown": self.config["cooldown"],
            "shadow_budget": self.config["shadow_budget"],
            "evaluating": any(
                r["phase"] in ("selection", "validation") for r in models
            ),
            "models": sorted(models, key=lambda r: (r["model"], r["stars"])),
            "profiles_by_star": copy.deepcopy(self.state["profiles_by_star"]),
            "recent_events": copy.deepcopy(self.state["recent_events"][-20:]),
            "reason": "只有獨立樣本外及資源證據足夠時才切換；連續未全中不代表模型故障",
        }

    def to_dict(self):
        return copy.deepcopy(self.state)
