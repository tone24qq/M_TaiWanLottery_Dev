"""Causal bingo evaluation and score fusion.

All score vectors have 80 entries indexed by number minus one. Scores are
relative min-max scores, never calibrated probabilities. Outcome rows use
``model, stars, target_term, hit_count, draw_datetime, updated_at``. Historical
walk-forward results are simulations and must never enter the live archive.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import math
from typing import Callable

import numpy as np
from scipy.stats import rankdata, t as student_t

STARS = (3, 4, 5)
METHODS = ("equal", "shrink", "borda", "majority")


def normalize_scores(scores) -> list[float]:
    """Map finite 80-number scores to a common [0, 1] scale, preserving ties.

    A constant vector carries no preference and becomes 0.5 everywhere. A
    deterministic number-order tie break is used only when selecting candidates.
    """
    vector = np.asarray(scores, dtype=np.float64)
    if vector.shape != (80,) or not np.isfinite(vector).all():
        raise ValueError("模型必須提供 80 個有限分數")
    span = float(np.ptp(vector))
    if span <= np.finfo(float).eps:
        return [0.5] * 80
    return ((vector - vector.min()) / span).tolist()


def _rank(scores) -> list[int]:
    return (np.lexsort((np.arange(80), -np.asarray(scores))) + 1).tolist()


def _performance_summary(performance, stars: int):
    summaries = {}
    raw = defaultdict(list)
    for row in performance or []:
        if int(row.get("stars", stars)) != stars:
            continue
        name = row.get("model", row.get("name"))
        if not name:
            continue
        if "samples" in row and "mean_hits" in row:
            summaries[name] = (int(row["samples"]), float(row["mean_hits"]))
        elif row.get("hit_count") is not None:
            raw[name].append(float(row["hit_count"]))
    for name, hits in raw.items():
        summaries[name] = (len(hits), float(np.mean(hits)))
    return summaries


class FusionScores:
    """Normalize, rank and count model support once for all four methods.

    A snapshot is reused across star counts. No persistent cache is keyed merely
    by term or model name, so a new score matrix always receives a fresh result.
    """

    def __init__(self, scores: dict):
        valid, self.excluded = {}, {}
        for name, vector in scores.items():
            try:
                valid[name] = normalize_scores(vector)
            except (TypeError, ValueError) as exc:
                self.excluded[name] = str(exc)
        if not valid:
            raise ValueError("目前沒有可用的完整模型分數")
        self.names = sorted(valid)
        self.matrix = np.asarray([valid[name] for name in self.names])
        # Stable ascending number order is precisely _rank's tie rule.
        self.order = np.argsort(-self.matrix, axis=1, kind="stable")
        self.borda = None
        self.support = {}

    def calculate(self, method="shrink", stars=3, performance=None, summary=None):
        if stars not in STARS or method not in METHODS:
            raise ValueError("僅支援三星、四星、五星及四種指定投票方法")
        names, matrix = self.names, self.matrix
        weights = np.ones(len(names), dtype=float)
        summary = (
            _performance_summary(performance, stars) if summary is None else summary
        )
        if method == "shrink" and all(summary.get(n, (0, 0))[0] >= 30 for n in names):
            null_sd = math.sqrt(stars * 0.25 * 0.75 * (80 - stars) / 79)
            for i, name in enumerate(names):
                n, mean_hits = summary[name]
                effect = np.clip((mean_hits - stars * 0.25) / null_sd, -0.5, 0.5)
                weights[i] = 1 + effect * n / (n + 100)
        weights /= weights.sum()
        if stars not in self.support:
            self.support[stars] = np.bincount(
                self.order[:, :stars].ravel(), minlength=80
            )
        support = self.support[stars]
        average = weights @ matrix
        if method == "majority":
            combined = (support + average / (len(names) + 1)) / (len(names) + 1)
        elif method == "borda":
            if self.borda is None:
                self.borda = (
                    (rankdata(matrix, method="average", axis=1) - 1) / 79
                ).mean(axis=0)
            combined = self.borda
        else:
            combined = average
        return {
            "method": method,
            "stars": stars,
            "scores": combined.tolist(),
            "ranking": _rank(combined),
            "weights": {n: float(w) for n, w in zip(names, weights)},
            "support": {str(n + 1): int(v) for n, v in enumerate(support)},
            "excluded": dict(self.excluded),
            "score_kind": "相對排名分數，非開獎機率",
        }


def fuse(
    scores: dict,
    method: str = "shrink",
    stars: int = 3,
    performance: list[dict] | None = None,
) -> dict:
    """Independent deterministic full ranking; score validation isolates failures.

    Shrink exclusively consumes already settled observations. The original
    public contract and scalar floating-point reduction order are preserved.
    """
    if stars not in STARS or method not in METHODS:
        raise ValueError("僅支援三星、四星、五星及四種指定投票方法")
    return FusionScores(scores).calculate(method, stars, performance)


def fuse_many(scores: dict, performance: list[dict] | None = None) -> dict:
    """All star/method combinations from one normalized immutable score matrix."""
    prepared = FusionScores(scores)
    result = {}
    for stars in STARS:
        summary = _performance_summary(performance, stars)
        result[str(stars)] = {
            method: prepared.calculate(method, stars, summary=summary)
            for method in METHODS
        }
    return result


def _wilson(successes: int, n: int, z: float = 1.95996398454):
    if not n:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - width), min(1.0, center + width)


def summarize_outcomes(rows: list[dict], stars: int, comparison_count: int = 1) -> dict:
    """Observed hit metrics and analytic hypergeometric random baseline.

    Rows may be raw outcomes or SQL histogram rows with integer ``count``.
    Histogram arithmetic preserves the exact full sample without expanding it.

    ``hit_rate`` is mean fraction of selected numbers hit; ``full_hit_rate`` is
    the fraction of draws hitting all k. Mean intervals are Bonferroni-adjusted
    t intervals; they are descriptive and assume independent draws. A minimum of
    100 observations and an adjusted lower bound above random is required before
    reporting evidence of an advantage. Repeated monitoring remains exploratory.
    """
    if stars not in STARS:
        raise ValueError("僅支援三星、四星、五星")
    observed = [r for r in rows if r.get("hit_count") is not None]
    hits = np.asarray([int(r["hit_count"]) for r in observed], dtype=np.int64)
    counts = np.asarray([int(r.get("count", 1)) for r in observed], dtype=np.int64)
    if (
        any(int(r.get("count", 1)) != r.get("count", 1) for r in observed)
        or (counts < 0).any()
    ):
        raise ValueError("彙總樣本數必須為非負整數")
    if ((hits < 0) | (hits > stars)).any():
        raise ValueError("命中數超出星數範圍")
    n = int(counts.sum())
    full = int(counts[hits == stars].sum())
    partial = int(counts[(hits > 0) & (hits < stars)].sum())
    mean = float(np.dot(hits, counts) / n) if n else 0.0
    if n >= 2:
        quantile = float(student_t.ppf(1 - 0.025 / max(1, comparison_count), n - 1))
        variance = float(np.dot(counts, (hits - mean) ** 2) / (n - 1))
        width = quantile * math.sqrt(variance / n)
        # Zero observed variance does not mean zero uncertainty for future draws.
        width = max(
            width, stars * math.sqrt(math.log(40 * max(1, comparison_count)) / (2 * n))
        )
        low, high = max(0.0, mean - width), min(float(stars), mean + width)
    else:
        low, high = 0.0, float(stars)
    full_low, full_high = _wilson(full, n)
    baseline = stars * 0.25
    credible = n >= 100 and low > baseline
    return {
        "stars": stars,
        "samples": n,
        "full_hits": full,
        "partial_hits": partial,
        "zero_hits": int(counts[hits == 0].sum()),
        "mean_hits": mean,
        "average_hits": mean,
        "hit_rate": mean / stars,
        "full_hit_rate": full / n if n else 0.0,
        "full_hit_ci_low": full_low,
        "full_hit_ci_high": full_high,
        "ci_low": low,
        "ci_high": high,
        "random_mean": baseline,
        "random_full_hit_rate": math.comb(20, stars) / math.comb(80, stars),
        "random_variance": stars * 0.25 * 0.75 * (80 - stars) / 79,
        "delta_vs_random": mean - baseline if n else None,
        "hit_distribution": {
            str(k): int(counts[hits == k].sum()) for k in range(stars + 1)
        },
        "proven_better_than_random": credible,
        "credibility": "樣本不足" if n < 30 else ("初步觀察" if n < 100 else "可比較"),
        "evidence": "觀察區間高於隨機基準，仍需持續樣本外驗證"
        if credible
        else "尚未證明優於隨機",
        "interval_method": "保守 95% 區間：多重比較校正 t 區間與 Hoeffding 界取較寬者；假設各期獨立",
        "valid": bool(n),
    }


def _parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if value:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return None


def _consecutive(a: dict, b: dict) -> bool:
    try:
        before, after = (
            _parse_datetime(a.get("draw_datetime")),
            _parse_datetime(b.get("draw_datetime")),
        )
        if int(b["target_term"]) == int(a["target_term"]) + 1:
            if before is None or after is None:
                return True
            elapsed = (after - before).total_seconds()
            return elapsed == 300 or (
                elapsed == 25800
                and (before.hour, before.minute) == (23, 55)
                and (after.hour, after.minute) == (7, 5)
            )
        # Taiwan draw sequence restarts at 000001 in a new ROC year.
        old, new = str(a["target_term"]), str(b["target_term"])
        return bool(
            len(old) == len(new) == 9
            and int(new[:3]) == int(old[:3]) + 1
            and new[3:] == "000001"
            and before
            and after
            and after.date() == before.date() + timedelta(days=1)
            and (before.month, before.day) == (12, 31)
            and (after.month, after.day) == (1, 1)
            and (before.hour, before.minute) == (23, 55)
            and (after.hour, after.minute) == (7, 5)
        )
    except (TypeError, ValueError):
        return False


def live_rankings(rows: list[dict], weights: dict | None = None) -> dict:
    """Return per-star cumulative and final consecutive-run leaderboards.

    Gaps reset only the consecutive view. Older legal outcomes always remain in
    the cumulative view. Models are ranked on mean hits, with full hits as a
    secondary key; sample counts and uncertainty stay visible.
    """
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("hit_count") is None or int(row.get("stars", 0)) not in STARS:
            continue
        name = row.get("model", row.get("name"))
        identity = (name, int(row["stars"]), str(row["target_term"]))
        if identity in seen:
            raise ValueError("排行榜發現重複結算")
        seen.add(identity)
        groups[(name, int(row["stars"]))].append(row)
    result = {str(k): {"cumulative": [], "consecutive": []} for k in STARS}
    comparisons = max(1, len({name for name, _ in groups}))
    for (name, stars), values in groups.items():
        ordered = sorted(values, key=lambda r: int(r["target_term"]))
        streak = []
        for row in ordered:
            if streak and not _consecutive(streak[-1], row):
                streak = []
            streak.append(row)
        for kind, observations in (("cumulative", ordered), ("consecutive", streak)):
            stats = summarize_outcomes(observations, stars, comparisons)
            weight = (weights or {}).get(str(stars), {}).get(name)
            stats.update(
                {
                    "name": name,
                    "model": name,
                    "weight": weight,
                    "updated_at": str(ordered[-1].get("updated_at", "")),
                }
            )
            result[str(stars)][kind].append(stats)
    for boards in result.values():
        for board in boards.values():
            board.sort(key=lambda r: (-r["mean_hits"], -r["full_hit_rate"], r["name"]))
            for position, row in enumerate(board, 1):
                row["rank"] = position
    return result


def live_rankings_summary(summary: dict, weights: dict | None = None) -> dict:
    """Build exact lifetime and trailing-run boards from bounded SQL histograms.

    ``summary`` has cumulative/consecutive lists. Each row contains model,
    stars, hit_count, count and updated_at. Only at most (stars+1) rows are
    needed per model/star, independent of the lifetime number of predictions.
    """
    result = {str(k): {"cumulative": [], "consecutive": []} for k in STARS}
    comparisons = max(
        1, len({r.get("model", r.get("name")) for r in summary.get("cumulative", [])})
    )
    for kind in ("cumulative", "consecutive"):
        groups = defaultdict(list)
        for row in summary.get(kind, []):
            stars = int(row.get("stars", 0))
            name = row.get("model", row.get("name"))
            if name and stars in STARS and row.get("hit_count") is not None:
                groups[(name, stars)].append(row)
        for (name, stars), rows in groups.items():
            stats = summarize_outcomes(rows, stars, comparisons)
            stats.update(
                {
                    "name": name,
                    "model": name,
                    "weight": (weights or {}).get(str(stars), {}).get(name),
                    "updated_at": max(str(r.get("updated_at", "")) for r in rows),
                }
            )
            result[str(stars)][kind].append(stats)
    for boards in result.values():
        for board in boards.values():
            board.sort(key=lambda r: (-r["mean_hits"], -r["full_hit_rate"], r["name"]))
            for position, row in enumerate(board, 1):
                row["rank"] = position
    return result


def _validate_draws(draws):
    clean = []
    seen = set()
    for draw in draws:
        term = str(draw["draw_term"])
        nums = list(draw["numbers"])
        if not term.isdigit() or term in seen:
            raise ValueError("期號重複或無效")
        if (
            len(nums) != 20
            or len(set(nums)) != 20
            or any(type(n) is not int or not 1 <= n <= 80 for n in nums)
        ):
            raise ValueError("每期必須包含 20 個不重複的 1～80 號碼")
        seen.add(term)
        clean.append({**draw, "draw_term": term, "numbers": sorted(nums)})
    clean.sort(key=lambda d: int(d["draw_term"]))
    for before, after in zip(clean, clean[1:]):
        if int(after["draw_term"]) != int(before["draw_term"]) + 1:
            a = {
                "target_term": before["draw_term"],
                "draw_datetime": before.get("draw_datetime"),
            }
            b = {
                "target_term": after["draw_term"],
                "draw_datetime": after.get("draw_datetime"),
            }
            if not _consecutive(a, b):
                raise ValueError("歷史期號有缺漏，請先補齊再執行逐期回測")
    timestamps = [_parse_datetime(d.get("draw_datetime")) for d in clean]
    for a, b in zip(timestamps, timestamps[1:]):
        if a is not None and b is not None and a >= b:
            raise ValueError("期號與開獎時間順序不一致")
    return clean


def recommend_method(report: dict) -> dict:
    """Choose a method for future draws using completed validation only.

    No recommendation changes a prediction in the report. A non-Shrink winner
    needs >=100 paired folds for each star and a positive, multiplicity-adjusted
    paired mean interval against Shrink. Otherwise Shrink remains provisional.
    """
    rows = report.get("results", [])
    indexed = {
        (str(r["target_term"]), int(r["stars"]), r["model"]): r["hit_count"]
        for r in rows
    }
    terms = sorted({term for term, _, _ in indexed}, key=int)
    counts = []
    candidates = []
    for method in METHODS:
        if method == "shrink":
            continue
        lower, counts_method = [], []
        for stars in STARS:
            paired = [
                indexed[(term, stars, method)] - indexed[(term, stars, "shrink")]
                for term in terms
                if (term, stars, method) in indexed
                and (term, stars, "shrink") in indexed
            ]
            n = len(paired)
            counts_method.append(n)
            if n < 100:
                lower.append(-math.inf)
                continue
            differences = np.asarray(paired, dtype=float)
            # Bounded paired differences lie in [-stars, stars].
            width = 2 * stars * math.sqrt(math.log(2 * 9 / 0.05) / (2 * n))
            lower.append(float(differences.mean()) - width)
        counts.extend(counts_method)
        if min(lower) > 0:
            candidates.append((float(np.mean(lower)), method))
    if candidates:
        method = max(candidates)[1]
        return {
            "method": method,
            "validated": True,
            "samples": min(counts),
            "reason": "三種星數各至少 100 期，配對比較保守區間皆優於 Shrink；僅供其後新期採用",
        }
    enough = bool(counts) and min(counts) >= 100
    return {
        "method": "shrink",
        "validated": False,
        "samples": min(counts) if counts else 0,
        "reason": (
            "完整樣本外比較未證明其他方法穩定優於 Shrink；維持暫定策略，未宣稱優於隨機"
            if enough
            else "有效樣本不足 100 期；Shrink 為暫定策略，尚無可靠優勢證據"
        ),
    }


def walk_forward(
    draws: list[dict],
    count: int = 100,
    start_term=None,
    end_term=None,
    threads: int = 2,
    window: int = 1000,
    progress: Callable | None = None,
    model_names: list[str] | None = None,
    retrain_every: int = 100,
) -> dict:
    """Fresh causal walk-forward simulation, never using saved live models.

    Each fit sees only the prefix before its target; tree/neural models retrain
    at the configured cadence. Online models update after the outcome. Every
    model and all four voting strategies are evaluated at each available fold.
    Fusion weights use only already settled earlier folds. ``folds`` permanently
    records the causal cutoff and candidate sets in the returned report.
    """
    from taiwan_lottery.ml.bingo_models import ModelSuite

    if not 1 <= count <= 10000 or window < 100 or retrain_every < 1:
        raise ValueError("回測期數需為 1～10000，訓練視窗至少 100 期")
    data = _validate_draws(draws)
    eligible = [
        i
        for i, draw in enumerate(data)
        if (start_term is None or int(draw["draw_term"]) >= int(start_term))
        and (end_term is None or int(draw["draw_term"]) <= int(end_term))
    ]
    if start_term is None and end_term is None:
        eligible = eligible[-count:]
    elif len(eligible) > count:
        raise ValueError("自訂區間超過設定回測上限，請增加 count")
    if (
        not eligible
        or eligible[0] < 100
        or (start_term is None and end_term is None and len(eligible) < count)
    ):
        raise ValueError("資料不足：每個目標期之前至少需要 100 期訓練資料")
    results, folds, warnings = [], [], []
    feature_history = None
    feature_origin = feature_end = 0
    if hasattr(ModelSuite, "scores_prepared"):
        from taiwan_lottery.ml.bingo_models import FeatureHistory
    totals = defaultdict(lambda: [0, 0])
    suite = None
    latest_fit_index = None
    for step, index in enumerate(eligible):
        history = [d["numbers"] for d in data[max(0, index - window) : index]]
        if suite is None or step % retrain_every == 0:
            kwargs = {"threads": threads, "window": window}
            if model_names is not None:
                kwargs["enabled"] = model_names

            def fit_progress(name, completed, total, status):
                if progress:
                    progress(
                        f"walk_forward/{name} {completed}/{total}",
                        step,
                        len(eligible),
                        "training",
                    )

            suite = ModelSuite(history, **kwargs).fit(progress=fit_progress)
            latest_fit_index = index - 1
        effective_window = min(window, getattr(suite, "window", window))
        if hasattr(suite, "scores_prepared"):
            # A bounded block plus its context prevents memory growing with the
            # entire stored database. Each query is still an exclusive prefix.
            if feature_history is None or index >= feature_end:
                feature_origin = max(0, index - effective_window)
                feature_end = min(len(data), index + 128)
                feature_history = FeatureHistory(
                    [d["numbers"] for d in data[feature_origin:feature_end]]
                )
            offset = index - feature_origin
            current_row = feature_history.row(offset, effective_window)
            score_map = suite.scores_prepared(
                feature_history.matrix_before(offset, effective_window), current_row
            )
        else:
            current_row = None
            score_map = suite.scores(history)
        if model_names is not None:
            score_map = {n: s for n, s in score_map.items() if n in model_names}
        if not score_map:
            raise ValueError("回測中無可用模型")
        # Strictly freeze all candidate sets before reading this target's answer.
        predictions = {}
        for name, vector in score_map.items():
            try:
                ranking = _rank(normalize_scores(vector))
                predictions[name] = {str(k): ranking[:k] for k in STARS}
            except (ValueError, TypeError) as exc:
                warnings.append(f"{data[index]['draw_term']} / {name}：{exc}")
        past_performance = [
            {"model": name, "stars": stars, "samples": n, "mean_hits": total / n}
            for (name, stars), (n, total) in totals.items()
            if n
        ]
        fusions = fuse_many(score_map, past_performance)
        for method in METHODS:
            predictions[method] = {
                str(stars): fusions[str(stars)][method]["ranking"][:stars]
                for stars in STARS
            }
        fold = {
            "target_term": data[index]["draw_term"],
            "trained_through": data[latest_fit_index]["draw_term"],
            "features_through": data[index - 1]["draw_term"],
            "predictions": predictions,
            "history_from": data[max(0, index - effective_window)]["draw_term"],
            "history_to": data[index - 1]["draw_term"],
            "history_count": min(index, effective_window),
            "training_history_from": data[max(0, latest_fit_index + 1 - window)][
                "draw_term"
            ],
            "training_history_to": data[latest_fit_index]["draw_term"],
            "training_history_count": min(latest_fit_index + 1, window),
            "feature_version": "bingo-features-causal-v1",
            "model_version": getattr(suite, "history_digest", "fixed-causal-suite-v1"),
            "model_scores": {name: list(vector) for name, vector in score_map.items()},
            "fusion_scores": {
                str(k): {
                    method: fusions[str(k)][method]["scores"] for method in METHODS
                }
                for k in STARS
            },
            "fusion_weights": {
                str(k): {
                    method: fusions[str(k)][method]["weights"] for method in METHODS
                }
                for k in STARS
            },
            "is_live": False,
        }
        folds.append(fold)
        actual = set(data[index]["numbers"])
        for name, by_star in predictions.items():
            for key, candidates in by_star.items():
                results.append(
                    {
                        "target_term": data[index]["draw_term"],
                        "model": name,
                        "stars": int(key),
                        "predicted_numbers": list(candidates),
                        "actual_numbers": sorted(actual),
                        "hit_count": len(actual.intersection(candidates)),
                        "draw_datetime": str(data[index].get("draw_datetime", "")),
                        "updated_at": str(data[index].get("draw_datetime", "")),
                    }
                )
                totals[(name, int(key))][0] += 1
                totals[(name, int(key))][1] += results[-1]["hit_count"]
        if hasattr(suite, "scores_prepared"):
            suite.update(history, data[index]["numbers"], feature_row=current_row)
        else:
            suite.update(history, data[index]["numbers"])
        # If custom intervals skip dates, rebuild from the complete known prefix.
        if step + 1 < len(eligible) and eligible[step + 1] != index + 1:
            suite = None
        if progress:
            progress("walk_forward", step + 1, len(eligible), "running")
    boards = live_rankings(results)
    report = {
        "mode": "strict_walk_forward",
        "is_live": False,
        "count": len(folds),
        "train_size": min(window, eligible[0]),
        "window": window,
        "retrain_every": retrain_every,
        "folds": folds,
        "results": results,
        "leaderboards": boards,
        "method_comparison": {
            str(k): [r for r in boards[str(k)]["cumulative"] if r["model"] in METHODS]
            for k in STARS
        },
        "model_statuses": suite.statuses(),
        "warnings": sorted(set(warnings)),
        "score_kind": "相對排名分數，非開獎機率",
        "methodology": "每期候選先凍結，再揭露結果；僅用過去特徵、已結算權重及當時可用模型；歷史模擬與正式實戰分離",
    }
    decision = recommend_method(report)
    report.update(
        {
            "selected_method": decision["method"],
            "selection_reason": decision["reason"],
            "decision": decision,
        }
    )
    return report
