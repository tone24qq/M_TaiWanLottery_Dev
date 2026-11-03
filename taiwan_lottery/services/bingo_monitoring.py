"""Bounded diagnostics from already archived and settled Bingo predictions.

A diagnostic is an observation, not a strategy approval. Nothing in this module
changes candidates, adds random jitter, promotes a challenger, or rewrites the
live ledger. Evidence gates in the adaptive controller remain independent.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from typing import Any

import numpy as np

from taiwan_lottery.bingo_clock import to_taipei
from taiwan_lottery.ml.bingo_evaluation import summarize_outcomes

VERSION = "bingo-monitor-1"
HISTORY_LIMIT = 300
REPORT_LIMIT = 30
WINDOWS = (10, 30, 100, 300)
FUSION_NAMES = frozenset(("equal", "shrink", "borda", "majority"))


def _stamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return to_taipei(value)


def _iso(value: Any) -> str | None:
    parsed = _stamp(value)
    return parsed.isoformat() if parsed is not None else None


def _alert(
    code: str, message: str, *, model=None, stars=None, severity="notice", **details
):
    return {
        "code": code,
        "message": message,
        "severity": severity,
        "model": model,
        "stars": stars,
        "details": details,
    }


def _selection_metrics(scores: np.ndarray, numbers: list[int], previous: dict | None):
    """Stable ties and numeric rank comparisons; no score perturbation."""
    ranking = np.lexsort((np.arange(80), -scores))
    inverse = np.empty(80, dtype=np.int16)
    inverse[ranking] = np.arange(80)
    span = float(np.ptp(scores))
    normalized = (scores - float(scores.min())) / span if span > 1e-12 else np.ones(80)
    total = float(normalized.sum())
    mass = normalized / total if total > 0 else np.full(80, 1 / 80)
    positive = mass > 0
    concentration = float(np.dot(mass, mass))
    histogram = (
        np.histogram(
            normalized if span > 1e-12 else np.full(80, 0.5), bins=8, range=(0.0, 1.0)
        )[0]
        / 80
    )
    metrics = {
        "score_span": span,
        "score_std": float(normalized.std() * span) if span else 0.0,
        "flat_scores": span <= 1e-12,
        "relative_concentration": concentration,
        "effective_numbers": 1 / concentration,
        "relative_entropy": float(
            -np.dot(mass[positive], np.log(mass[positive])) / math.log(80)
        ),
        "histogram": histogram.tolist(),
        "ranking": (ranking + 1).tolist(),
        "unchanged": None,
        "top_k_overlap": None,
        "top_k_jaccard": None,
        "rank_displacement": None,
        "rank_spearman": None,
    }
    if previous:
        before = set(previous["numbers"])
        after = set(numbers)
        intersection = len(before & after)
        metrics.update(
            unchanged=before == after,
            top_k_overlap=intersection / len(after),
            top_k_jaccard=intersection / len(before | after),
        )
        old = np.asarray(previous["ranking"], dtype=np.int16) - 1
        old_inverse = np.empty(80, dtype=np.int16)
        old_inverse[old] = np.arange(80)
        difference = inverse.astype(float) - old_inverse
        metrics.update(
            rank_displacement=float(np.abs(difference).mean()),
            rank_spearman=float(
                1 - 6 * np.dot(difference, difference) / (80 * (80**2 - 1))
            ),
        )
    return metrics


def _js_divergence(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    mixture = (a + b) / 2
    divergence = 0.0
    for vector in (a, b):
        mask = vector > 0
        divergence += 0.5 * float(
            np.dot(vector[mask], np.log2(vector[mask] / mixture[mask]))
        )
    return max(0.0, min(1.0, divergence))


def _safe_row(row: dict, target: str):
    """Reject leaked/mismatched observations before touching monitor state."""
    if str(row.get("target_term", target)) != target:
        raise ValueError("監控資料的目標期與結算期不一致")
    stars = int(row["stars"])
    if stars not in (3, 4, 5):
        raise ValueError("監控僅接受三星、四星、五星")
    model = str(row.get("model", row.get("name", "")))
    if not model:
        raise ValueError("監控缺少模型名稱")
    raw_numbers = row.get("numbers", row.get("predicted_numbers", []))
    if (
        len(raw_numbers) != stars
        or len(set(raw_numbers)) != stars
        or any(type(n) is not int or not 1 <= n <= 80 for n in raw_numbers)
    ):
        raise ValueError("監控選號不是有效的不重複星數號碼")
    hits = row.get("hit_count")
    if type(hits) is not int or not 0 <= hits <= stars:
        raise ValueError("監控僅接受已結算的有效命中數")
    scores = np.asarray(row["scores"], dtype=float)
    if scores.shape != (80,) or not np.isfinite(scores).all():
        raise ValueError("監控分數必須是 80 個有限值")
    if not math.isfinite(float(scores.max()) - float(scores.min())):
        raise ValueError("監控分數範圍超出有限數值界線")
    source_term = row.get("source_term")
    if source_term is not None and (
        not str(source_term).isdigit() or int(source_term) >= int(target)
    ):
        raise ValueError("監控拒絕當期或未來資料作為預測來源")
    created, source_time, target_time = (
        _stamp(row.get(name)) for name in ("created_at", "source_time", "target_time")
    )
    if target_time and (
        (created and created >= target_time)
        or (source_time and source_time >= target_time)
    ):
        raise ValueError("監控拒絕開獎後才封存的預測")
    audit = dict(row.get("audit") or row.get("metadata") or {})
    # First-stage rows already contain a durable snapshot hash, even though
    # feature audit metadata was added later. Preserve that known evidence.
    if not audit.get("input_hash") and row.get("snapshot_hash"):
        audit["input_hash"] = row["snapshot_hash"]
    for key in ("training_age_draws", "max_model_age_draws", "retrain_every"):
        if key in audit and audit[key] is not None:
            value = audit[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < (1 if key == "retrain_every" else 0)
            ):
                raise ValueError("模型訓練年齡／排程參數無效")
            audit[key] = int(value)
    for key in ("input_hash", "feature_hash"):
        if audit.get(key) is not None and not isinstance(audit[key], str):
            raise ValueError("監控摘要必須是字串")
    for key in ("training_cutoff", "feature_cutoff"):
        cutoff = audit.get(key)
        if cutoff is not None:
            if not str(cutoff).isdigit() or int(cutoff) >= int(target):
                raise ValueError("監控拒絕使用當期或未來資料的特徵／模型")
            if source_term is not None and int(cutoff) > int(source_term):
                raise ValueError("模型／特徵截止期晚於已知來源期")
    settled = _stamp(row.get("settled_at", row.get("updated_at")))
    if settled and target_time and settled < target_time:
        raise ValueError("監控拒絕開獎前的結算結果")
    weights = {
        str(key): float(value) for key, value in (row.get("weights") or {}).items()
    }
    if any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in weights.values()
    ):
        raise ValueError("監控權重必須是 0～1 的有限數")
    actual = row.get("actual_numbers")
    if actual is not None and len(set(raw_numbers) & set(actual)) != hits:
        raise ValueError("監控命中數與實際開獎結果不符")
    return {
        "model": model,
        "stars": stars,
        "target_term": target,
        "source_term": str(source_term) if source_term is not None else None,
        "created_at": _iso(row.get("created_at")),
        "target_time": _iso(row.get("target_time")),
        "updated_at": _iso(
            row.get("settled_at", row.get("updated_at", row.get("target_time")))
        ),
        "numbers": sorted(raw_numbers),
        "hit_count": hits,
        "scores": scores.tolist(),
        "version": str(row.get("version", "unknown")),
        "audit": audit,
        "weights": weights,
    }


def _digest(rows):
    return hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
        ).encode()
    ).hexdigest()


class PerformanceMonitor:
    """Causal performance, repetition and freshness diagnostics.

    ``observe`` accepts each target once, after its pre-draw decision is settled.
    A duplicate returns the existing report; changed duplicate data is rejected.
    Stored history is capped at 300 compact records per model/star. Only the last
    ranking is retained in full; adjacent movement statistics are sufficient for
    rolling stability summaries and avoid persisting 300 full score matrices.
    """

    def __init__(self, state: dict | None = None):
        self.state = (
            copy.deepcopy(state)
            if state
            else {
                "version": VERSION,
                "groups": {},
                "reports": [],
                "last_target": None,
                "last_digest": None,
                "last_report": None,
                "target_count": 0,
                "last_evaluation_request": -30,
            }
        )
        if self.state.get("version") != VERSION:
            raise ValueError("未知的監控狀態版本")
        self.state.setdefault("target_count", 0)
        self.state.setdefault("last_evaluation_request", -30)
        # A restored oversized old state cannot bypass the memory boundary.
        self.state["reports"] = self.state.get("reports", [])[-REPORT_LIMIT:]
        for group in self.state.get("groups", {}).values():
            group["history"] = group.get("history", [])[-HISTORY_LIMIT:]

    def observe(self, target_term, rows, *, sync=None, model_statuses=None):
        target = str(target_term)
        if not target.isdigit():
            raise ValueError("監控期號無效")
        # Canonicalize all observations first: a bad row cannot partially update
        # another model's state, and external input dictionaries stay untouched.
        clean = sorted(
            (_safe_row(row, target) for row in rows),
            key=lambda r: (r["model"], r["stars"]),
        )
        identities = [(r["model"], r["stars"]) for r in clean]
        if len(set(identities)) != len(identities):
            raise ValueError("同一期監控含重複模型／星數")
        fingerprint = _digest(clean)
        previous_target = self.state.get("last_target")
        if previous_target == target:
            if fingerprint != self.state.get("last_digest"):
                raise ValueError("已監控的同一期資料被變更")
            return copy.deepcopy(self.state["last_report"])
        if previous_target is not None and int(target) < int(previous_target):
            raise ValueError("監控只能按照真實期號順序更新")
        if not clean:
            raise ValueError("監控沒有已結算資料")
        alerts = []
        weight_changes = []
        for row in clean:
            key = f"{row['model']}:{row['stars']}"
            group = self.state["groups"].setdefault(
                key,
                {
                    "model": row["model"],
                    "stars": row["stars"],
                    "history": [],
                    "samples": 0,
                    "full_miss_streak": 0,
                    "mean_fraction": 0.0,
                    "ewma_fraction": None,
                    "ph_downward": 0.0,
                    "same_input_streak": 0,
                    "same_feature_streak": 0,
                },
            )
            group_alerts, changes = self._update(group, row)
            alerts.extend(group_alerts)
            weight_changes.extend(changes)
        # Keep inactive models visible with their own actual last observation;
        # a missing current-period model must not erase its recent history.
        comparisons = max(1, len(self.state["groups"]) * len(WINDOWS))
        summaries = [
            self._summarize(group, comparisons)
            for _, group in sorted(self.state["groups"].items())
        ]
        sync = copy.deepcopy(sync or {})
        if sync.get("state") in ("disconnected", "failed", "insufficient", "degraded"):
            alerts.append(
                _alert(
                    "data_connection",
                    "資料同步異常，先修復資料來源；不以失準為由更換選號",
                    severity="warning",
                    state=sync.get("state"),
                    error=sync.get("error"),
                )
            )
        if sync.get("missing_terms"):
            alerts.append(
                _alert(
                    "data_gaps",
                    "已知歷史有缺期，請先補齊或確認可用快照",
                    severity="warning",
                    gaps=sync["missing_terms"],
                )
            )
        for model in model_statuses or []:
            if model.get("status") == "failed":
                alerts.append(
                    _alert(
                        "model_failure",
                        "模型回報失敗，保留其他健康模型並檢查版本與工件",
                        model=model.get("name"),
                        severity="warning",
                        error=model.get("error"),
                    )
                )
        # A loss streak alone is deliberately absent from the evaluation triggers.
        candidate_signal = any(
            a["code"]
            in (
                "performance_drift",
                "performance_regression",
                "score_distribution_shift",
            )
            for a in alerts
        )
        data_unhealthy = any(
            a["code"]
            in (
                "data_connection",
                "data_gaps",
                "model_failure",
                "model_stale",
                "input_not_refreshed",
                "feature_cache_suspected",
            )
            for a in alerts
        )
        count = self.state["target_count"] + 1
        cooled = count - self.state["last_evaluation_request"] >= 30
        request = bool(candidate_signal and not data_unhealthy and cooled)
        if request:
            self.state["last_evaluation_request"] = count
        stamp = (
            max((r["updated_at"] or r["target_time"] or "" for r in clean), default="")
            or None
        )
        report = {
            "version": VERSION,
            "target_term": target,
            "updated_at": stamp,
            "models": summaries,
            "alerts": alerts,
            "consensus": self._consensus(clean),
            "weight_changes": weight_changes,
            "evaluation_requested": request,
            # A signal does not mean a background validation job has started.
            # The runtime reports the actual worker state separately.
            "evaluating": False,
            "evaluation_reason": (
                "健康資料上的退化訊號，請在獨立影子驗證中比較候選"
                if request
                else "資料／工件異常需先修復"
                if data_unhealthy
                else "偵測到探索性訊號，仍在評估冷卻期間"
                if candidate_signal and not cooled
                else "沒有足夠退化訊號；維持原策略"
            ),
            "actions": {
                "repair_data_or_artifact": data_unhealthy,
                "evaluate_challengers": request,
                "force_change": False,
            },
            "disclaimer": "監控僅描述已封存實績；連續未全中不等同故障，重複號碼不會被強制改動",
        }
        summary = {
            k: copy.deepcopy(report[k])
            for k in (
                "target_term",
                "updated_at",
                "alerts",
                "evaluation_requested",
                "evaluation_reason",
                "weight_changes",
            )
        }
        self.state.update(
            last_target=target,
            last_digest=fingerprint,
            last_report=report,
            target_count=count,
        )
        self.state["reports"] = (self.state["reports"] + [summary])[-REPORT_LIMIT:]
        return copy.deepcopy(report)

    def _update(self, group, row):
        model, stars = row["model"], row["stars"]
        history = group["history"]
        previous = group.get("last_decision")
        metrics = _selection_metrics(
            np.asarray(row["scores"]), row["numbers"], previous
        )
        alerts, changes = [], []
        audit = row["audit"]
        input_hash = audit.get("input_hash", audit.get("snapshot_hash"))
        feature_hash = audit.get("feature_hash")
        cutoff, source = audit.get("training_cutoff"), row["source_term"]
        age = audit.get("training_age_draws")
        if age is None and cutoff and source and str(cutoff)[:3] == str(source)[:3]:
            age = max(0, int(source) - int(cutoff))
        maximum_age = int(
            audit.get(
                "max_model_age_draws",
                max(300, 3 * int(audit.get("retrain_every", 100))),
            )
        )
        group["training_age_draws"] = age
        if audit.get("model_expired") or (age is not None and age > maximum_age):
            alerts.append(
                _alert(
                    "model_stale",
                    "模型已超過已設定的訓練年齡，請先檢查背景訓練與版本恢復",
                    model=model,
                    stars=stars,
                    severity="warning",
                    training_age_draws=age,
                    maximum_age_draws=maximum_age,
                )
            )
        source_advanced = bool(
            previous and row["source_term"] != previous.get("source_term")
        )
        group["same_input_streak"] = (
            group["same_input_streak"] + 1
            if source_advanced and input_hash and input_hash == group.get("input_hash")
            else 0
        )
        group["same_feature_streak"] = (
            group["same_feature_streak"] + 1
            if source_advanced
            and feature_hash
            and feature_hash == group.get("feature_hash")
            else 0
        )
        if group["same_input_streak"] >= 3:
            alerts.append(
                _alert(
                    "input_not_refreshed",
                    "來源期已更新但輸入摘要持續相同，請檢查快取與快照鍵",
                    model=model,
                    stars=stars,
                    severity="warning",
                    consecutive=group["same_input_streak"],
                )
            )
        if group["same_feature_streak"] >= 10:
            alerts.append(
                _alert(
                    "feature_cache_suspected",
                    "輸入期已更新但特徵摘要持續相同，需檢查快取或確認特徵不變的計算依據",
                    model=model,
                    stars=stars,
                    severity="warning",
                    consecutive=group["same_feature_streak"],
                )
            )
        if metrics["flat_scores"]:
            alerts.append(
                _alert(
                    "flat_scores",
                    "完整 80 號分數沒有區別，選號由固定同分規則決定，請檢查模型輸出",
                    model=model,
                    stars=stars,
                    severity="warning",
                )
            )
        if metrics["effective_numbers"] < 2:
            alerts.append(
                _alert(
                    "concentrated_scores",
                    "相對分數高度集中，請檢查模型數值及版本；這不代表中獎信心",
                    model=model,
                    stars=stars,
                    effective_numbers=metrics["effective_numbers"],
                )
            )
        histogram = metrics.pop("histogram")
        old_hist = group.get("score_histogram_ewma")
        divergence = (
            _js_divergence(histogram, old_hist) if old_hist is not None else 0.0
        )
        if old_hist is not None and group["samples"] >= 30 and divergence > 0.25:
            alerts.append(
                _alert(
                    "score_distribution_shift",
                    "模型相對分數分布明顯變化，需以獨立候選驗證診斷原因",
                    model=model,
                    stars=stars,
                    divergence=divergence,
                )
            )
        group["score_histogram_ewma"] = (
            np.asarray(histogram)
            if old_hist is None
            else 0.05 * np.asarray(histogram) + 0.95 * np.asarray(old_hist)
        ).tolist()
        fraction = row["hit_count"] / stars
        group["samples"] += 1
        count = group["samples"]
        old_mean = group["mean_fraction"]
        group["mean_fraction"] = old_mean + (fraction - old_mean) / count
        group["ewma_fraction"] = (
            fraction
            if group["ewma_fraction"] is None
            else 0.05 * fraction + 0.95 * group["ewma_fraction"]
        )
        group["ph_downward"] = max(
            0.0, group["ph_downward"] + group["mean_fraction"] - fraction - 0.02
        )
        if count >= 30 and group["ph_downward"] > 6:
            alerts.append(
                _alert(
                    "performance_drift",
                    "有界 Page-Hinkley 觀察到命中比例下降；屬探索訊號，需獨立驗證",
                    model=model,
                    stars=stars,
                    statistic=group["ph_downward"],
                )
            )
        group["full_miss_streak"] = (
            group["full_miss_streak"] + 1 if row["hit_count"] < stars else 0
        )
        if previous:
            before_weights = previous.get("weights", {})
            if row["weights"] != before_weights:
                keys = before_weights.keys() | row["weights"].keys()
                changes.append(
                    {
                        "model": model,
                        "stars": stars,
                        "before": before_weights,
                        "after": row["weights"],
                        "l1_change": sum(
                            abs(row["weights"].get(k, 0) - before_weights.get(k, 0))
                            for k in keys
                        ),
                        "reason": audit.get(
                            "adjustment_reason",
                            "依已封存策略參數更新；監控不宣稱調整有效",
                        ),
                    }
                )
        record = {
            "target_term": row["target_term"],
            "hit_count": row["hit_count"],
            "numbers": row["numbers"],
            "version": row["version"],
            "source_term": row["source_term"],
            "updated_at": row["updated_at"],
            "input_changed": bool(
                previous and input_hash and input_hash != group.get("input_hash")
            ),
            "feature_changed": bool(
                previous and feature_hash and feature_hash != group.get("feature_hash")
            ),
            "version_changed": bool(
                previous and row["version"] != previous.get("version")
            ),
            **{k: v for k, v in metrics.items() if k != "ranking"},
        }
        group["history"] = (history + [record])[-HISTORY_LIMIT:]
        group.update(
            last_decision={
                "numbers": row["numbers"],
                "ranking": metrics["ranking"],
                "source_term": row["source_term"],
                "version": row["version"],
                "weights": row["weights"],
            },
            input_hash=input_hash,
            feature_hash=feature_hash,
            audit=copy.deepcopy(audit),
            last_metrics=metrics,
            score_divergence=divergence,
        )
        if len(group["history"]) >= 60:
            recent = np.asarray(
                [r["hit_count"] for r in group["history"][-30:]], dtype=float
            )
            prior = np.asarray(
                [r["hit_count"] for r in group["history"][-130:-30]], dtype=float
            )
            # Bounded independent-sample Hoeffding envelope; no optimistic normal
            # approximation after a long zero-variance winning or losing run.
            width = stars * (
                math.sqrt(math.log(40) / (2 * len(recent)))
                + math.sqrt(math.log(40) / (2 * len(prior)))
            )
            upper = float(recent.mean() - prior.mean() + width)
            group["recent_regression"] = {
                "recent_samples": len(recent),
                "prior_samples": len(prior),
                "mean_difference": float(recent.mean() - prior.mean()),
                "difference_ci_upper": upper,
                "evidence": upper < 0,
            }
            if upper < 0:
                alerts.append(
                    _alert(
                        "performance_regression",
                        "最近 30 期相較先前區間出現保守下降證據，需以未參與選擇的資料驗證候選",
                        model=model,
                        stars=stars,
                        difference_ci_upper=upper,
                    )
                )
        return alerts, changes

    @staticmethod
    def _summarize(group, comparison_count=1):
        history, stars = group["history"], group["stars"]
        windows = {}
        stability = {}
        for window in WINDOWS:
            recent = history[-window:]
            stats = summarize_outcomes(recent, stars, comparison_count=comparison_count)
            stats.update(requested_window=window, observed_window=len(recent))
            windows[str(window)] = stats
            if window != 300:
                transitions = [r for r in recent[1:] if r["unchanged"] is not None]

                def average(key):
                    return (
                        float(np.mean([r[key] for r in transitions]))
                        if transitions
                        else None
                    )

                stability[str(window)] = {
                    "samples": len(recent),
                    "transitions": len(transitions),
                    "unchanged_rate": average("unchanged"),
                    "top_k_overlap": average("top_k_overlap"),
                    "top_k_jaccard": average("top_k_jaccard"),
                    "rank_displacement": average("rank_displacement"),
                    "rank_spearman": average("rank_spearman"),
                    "input_updates": sum(r["input_changed"] for r in recent),
                    "feature_updates": sum(r["feature_changed"] for r in recent),
                    "version_updates": sum(r["version_changed"] for r in recent),
                }
        probability = math.comb(20, stars) / math.comb(80, stars)
        streak = group["full_miss_streak"]
        repeat = stability["30"]["unchanged_rate"]
        return {
            "model": group["model"],
            "name": group["model"],
            "stars": stars,
            "samples": group["samples"],
            "last_target_term": history[-1]["target_term"],
            "updated_at": history[-1]["updated_at"],
            "retained_samples": len(history),
            "windows": windows,
            "stability": stability,
            "full_miss_streak": streak,
            "random_full_hit_rate": probability,
            "random_probability_no_full_hit_in_streak": math.exp(
                streak * math.log1p(-probability)
            ),
            "loss_streak_is_failure": False,
            "ewma_hit_fraction": group["ewma_fraction"],
            "drift": {
                "page_hinkley_downward": group["ph_downward"],
                "exploratory_signal": group["samples"] >= 30
                and group["ph_downward"] > 6,
                "recent_regression": group.get("recent_regression"),
            },
            "score_distribution": {
                k: group["last_metrics"][k]
                for k in (
                    "score_span",
                    "score_std",
                    "flat_scores",
                    "relative_concentration",
                    "effective_numbers",
                    "relative_entropy",
                )
            },
            "score_divergence": group["score_divergence"],
            "audit_available": bool(
                group.get("input_hash") and group.get("feature_hash")
            ),
            "audit_missing": [
                name for name in ("input_hash", "feature_hash") if not group.get(name)
            ],
            "input_hash": group.get("input_hash"),
            "feature_hash": group.get("feature_hash"),
            "window": group.get("audit", {}).get("window"),
            "feature_version": group.get("audit", {}).get("feature_version"),
            "model_version": group["last_decision"]["version"],
            "training_age_draws": group.get("training_age_draws"),
            "repetition_reason": (
                "選號持續相同；若輸入、特徵與分數均正常更新，允許維持相同結果"
                if repeat is not None and repeat >= 0.9
                else "選號依實際評分更新，不加入隨機擾動"
            ),
        }

    @staticmethod
    def _consensus(rows):
        result = {}
        for stars in (3, 4, 5):
            sources = [
                r
                for r in rows
                if r["stars"] == stars and r["model"] not in FUSION_NAMES
            ]
            count = len(sources)
            if count < 2:
                result[str(stars)] = {
                    "models": count,
                    "mean_top_k_jaccard": None,
                    "mean_rank_correlation": None,
                }
                continue
            masks = np.zeros((count, 80), dtype=np.int16)
            ranks = []
            for i, row in enumerate(sources):
                masks[i, np.asarray(row["numbers"]) - 1] = 1
                order = np.lexsort((np.arange(80), -np.asarray(row["scores"])))
                inverse = np.empty(80, dtype=float)
                inverse[order] = np.arange(80)
                ranks.append(inverse)
            intersection = masks @ masks.T
            union = 2 * stars - intersection
            ids = np.triu_indices(count, k=1)
            normalized = (np.asarray(ranks) - 39.5) / math.sqrt((80**2 - 1) / 12)
            correlations = normalized @ normalized.T / 80
            result[str(stars)] = {
                "models": count,
                "mean_top_k_jaccard": float((intersection / union)[ids].mean()),
                "mean_rank_correlation": float(
                    np.clip(correlations[ids], -1, 1).mean()
                ),
                "disagreement": float(1 - (intersection / union)[ids].mean()),
            }
        return result

    def status(self):
        report = copy.deepcopy(self.state.get("last_report"))
        if report is None:
            return {
                "version": VERSION,
                "models": [],
                "alerts": [],
                "last_report": None,
                "evaluating": False,
                "evaluation_requested": False,
                "updated_at": None,
                "reports": copy.deepcopy(self.state["reports"]),
            }
        # Model/window details already occur at the top level. Do not duplicate
        # the same large arrays under last_report on every browser status poll.
        summary = {
            key: copy.deepcopy(report[key])
            for key in (
                "target_term",
                "updated_at",
                "alerts",
                "evaluation_requested",
                "evaluation_reason",
                "actions",
                "consensus",
                "weight_changes",
            )
        }
        return {
            **report,
            "last_report": summary,
            "reports": copy.deepcopy(self.state["reports"]),
        }

    def to_dict(self):
        return copy.deepcopy(self.state)
