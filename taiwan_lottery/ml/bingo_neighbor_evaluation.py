"""Prefix-only neighbor research replay; never reads or writes live artifacts.

The candidate configuration is fixed before evaluation. Every decision (including
the seeded random comparator) is hashed before the target outcome is revealed.
An optional, fully audited frozen comparison bundle avoids retraining historical
models when reproducing the checked-in official-data experiment.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math

import numpy as np
from scipy.stats import binomtest, norm

from taiwan_lottery.bingo_clock import is_successor
from taiwan_lottery.ml.bingo_evaluation import _validate_draws, live_rankings

STARS = (3, 4, 5)
RANDOM_SEED = "bingo-neighbor-random-v1"
# Three published sample sizes, three stars, mean/full-hit and both temporal
# partitions plus the overall interval. This budget is fixed before examining
# any experiment result (three sample sizes x stars x metrics x partitions).
EXPERIMENT_FAMILY = 3 * 3 * 2 * 3


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _time(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _history(draws):
    data = _validate_draws(draws)
    for before, after in zip(data, data[1:]):
        if not is_successor(before["draw_term"], _time(before["draw_datetime"]),
                            after["draw_term"], _time(after["draw_datetime"])):
            raise ValueError("歷史排程有缺漏或期號時間不符，請先補齊再回測")
    return data


def _targets(data, count, start_term, end_term):
    eligible = [i for i, row in enumerate(data)
                if (start_term is None or int(row["draw_term"]) >= int(start_term))
                and (end_term is None or int(row["draw_term"]) <= int(end_term))]
    if start_term is None and end_term is None:
        eligible = eligible[-count:]
        if len(eligible) != count:
            raise ValueError("資料不足：無法完成指定回測期數")
    if not eligible or eligible[0] < 100 or len(eligible) > count:
        raise ValueError("資料不足或區間過大：每個目標期之前至少需100期")
    return eligible


def _random_ranking(target):
    seed = int.from_bytes(hashlib.sha256((RANDOM_SEED + ":" + target).encode()).digest()[:8], "big")
    return (np.random.default_rng(seed).permutation(80) + 1).tolist()


def _comparison_bundle(report):
    """Normalize freshly calculated ABC replay to the frozen comparison contract."""
    by_target = {f["target_term"]: f for f in report["folds"]}
    folds = {}
    for row in report["results"]:
        term, name = str(row["target_term"]), row["model"]
        source = by_target[term]
        fold = folds.setdefault(term, dict(target_term=term, source_term=source["source_term"],
                                          predictions={}, cutoffs={}))
        fold["predictions"].setdefault(name, {})[str(row["stars"])] = list(row["predicted_numbers"])
        training = (source["rule_training_cutoff"] if name.startswith("association_") else
                    source["model_training_cutoffs"].get(name, source["batch_training_cutoff"]))
        fold["cutoffs"][name] = dict(training=training, features=source["feature_cutoff"])
    return dict(is_live=False, folds=list(folds.values()), results=report["results"],
                model_statuses=report.get("model_statuses", []),
                provenance=dict(kind="fresh_causal_abc_and_eight_model_replay"))


def audit_comparison_bundle(bundle, data, terms):
    """Reject future cutoffs, mismatched outcomes and unsealed comparator rows."""
    if bundle.get("is_live") is not False:
        raise ValueError("比較資料必須是獨立歷史模擬")
    if not bundle.get("folds") or not bundle.get("results"):
        raise ValueError("比較資料必須包含凍結候選與結算")
    index = {r["draw_term"]: i for i, r in enumerate(data)}
    folds = {}
    for fold in bundle["folds"]:
        term = str(fold["target_term"])
        if term in folds:
            raise ValueError("比較資料有重複目標期")
        folds[term] = fold
    if set(folds) != set(terms):
        raise ValueError("比較資料必須涵蓋完全相同的目標期")
    for term, fold in folds.items():
        at = index[term]
        if fold["source_term"] != data[at - 1]["draw_term"]:
            raise ValueError("比較資料的來源不是目標真正上一期")
        predictions_by_model = fold.get("predictions", {})
        cutoffs_by_model = fold.get("cutoffs", {})
        if not predictions_by_model or set(cutoffs_by_model) != set(predictions_by_model):
            raise ValueError("比較資料必須包含每個模型的訓練與特徵截止期")
        for name, predictions in fold["predictions"].items():
            if set(cutoffs_by_model[name]) != {"training", "features"}:
                raise ValueError("比較模型必須明確包含training/features截止期")
            for cutoff in fold["cutoffs"][name].values():
                if cutoff not in index or index[cutoff] >= at:
                    raise ValueError("比較模型訓練或特徵截止期洩漏未來答案")
            if set(predictions) != {"3", "4", "5"}:
                raise ValueError("比較候選必須包含三種星數")
            for key, numbers in predictions.items():
                if (len(numbers) != int(key) or len(set(numbers)) != len(numbers)
                        or any(type(n) is not int or not 1 <= n <= 80 for n in numbers)):
                    raise ValueError("比較候選號碼格式錯誤")
    rows, seen, model_sets = [], set(), []
    for row in bundle["results"]:
        term, name, star = str(row["target_term"]), row["model"], int(row["stars"])
        if term not in folds or name in {"neighbor", "random"}:
            raise ValueError("比較資料包含不相同期數或保留模型名稱")
        if star not in STARS or name not in folds[term]["predictions"]:
            raise ValueError("比較資料的模型或星數沒有對應凍結候選")
        identity = (term, name, star)
        if identity in seen:
            raise ValueError("比較資料有重複結算")
        seen.add(identity)
        actual = sorted(data[index[term]]["numbers"])
        numbers = folds[term]["predictions"][name][str(star)]
        hits = len(set(numbers) & set(actual))
        if (list(row["predicted_numbers"]) != numbers or sorted(row["actual_numbers"]) != actual
                or row["hit_count"] != hits):
            raise ValueError("比較資料的凍結候選或實際答案不一致")
        rows.append({**row, "is_live": False})
    for term in terms:
        expected = {(term, name, star) for name in folds[term]["predictions"] for star in STARS}
        if not expected <= seen:
            raise ValueError("比較資料缺少已封存的候選結算")
        model_sets.append(set(folds[term]["predictions"]))
    if any(names != model_sets[0] for names in model_sets):
        raise ValueError("比較模型必須使用相同目標樣本")
    return rows


def _partition_evidence(rows, stars, family):
    values = np.asarray([r["hit_count"] for r in rows], dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else None
    width = stars * math.sqrt(math.log(2 * family / .05) / (2 * n)) if n else stars
    baseline = math.comb(20, stars) / math.comb(80, stars)
    full = int(np.sum(values == stars))
    p_value = float(binomtest(full, n, baseline, alternative="greater").pvalue) if n else 1.
    # Simultaneous Wilson intervals are descriptive; the exact binomial p value
    # and bounded mean interval determine the predeclared experimental gate.
    z = float(norm.ppf(1 - .025 / family))
    if n:
        p = full / n
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        interval = [max(0., center-radius), min(1., center+radius)]
    else:
        interval = [0., 1.]
    return dict(samples=n, mean_hits=mean, mean_ci_low=max(0., mean-width) if n else 0.,
                mean_ci_high=min(float(stars), mean+width) if n else float(stars),
                random_mean=stars*.25, full_hits=full, full_hit_rate=full/n if n else None,
                random_full_hit_rate=baseline, full_hit_ci=interval,
                adjusted_full_hit_p=min(1., p_value*family),
                mean_clears_random=bool(n >= 100 and mean-width > stars*.25))


def neighbor_evidence(rows):
    """Fixed family and disjoint evaluation halves; no best-segment selection."""
    evidence = {}
    comparators = sorted({r["model"] for r in rows} - {"neighbor"})
    paired_family = max(EXPERIMENT_FAMILY, 3 * 3 * 2 * max(1, len(comparators)))
    for star in STARS:
        own = sorted([r for r in rows if r["model"] == "neighbor" and r["stars"] == star],
                     key=lambda r: int(r["target_term"]))
        split = len(own)//2
        first, last = own[:split], own[split:]
        discovery = _partition_evidence(first, star, EXPERIMENT_FAMILY)
        confirmation = _partition_evidence(last, star, EXPERIMENT_FAMILY)
        proven = discovery["mean_clears_random"] and confirmation["mean_clears_random"]
        paired = {}
        for name in comparators:
            other = {r["target_term"]: r["hit_count"] for r in rows
                     if r["model"] == name and r["stars"] == star}
            diff = [r["hit_count"] - other[r["target_term"]] for r in own]
            n = len(diff)
            delta = float(np.mean(diff)) if n else 0.
            width = 2*star*math.sqrt(math.log(2*paired_family/.05)/(2*n)) if n else 2*star
            paired[name] = dict(samples=n, mean_delta=delta, ci_low=delta-width,
                                ci_high=delta+width, family_size=paired_family,
                                evidence="配對保守區間；探索性比較，未授權正式調權")
        evidence[str(star)] = dict(
            overall=_partition_evidence(own, star, EXPERIMENT_FAMILY),
            discovery=discovery, confirmation=confirmation,
            confirmation_from=last[0]["target_term"] if last else None,
            independently_validated=proven, proven_better_than_random=proven,
            status="獨立樣本外平均命中證據通過，仍限研究" if proven else "尚未證明優於隨機",
            paired_comparisons=paired,
        )
    return evidence


def neighbor_walk_forward(draws, count=100, start_term=None, end_term=None,
                          threads=2, window=1000, progress=None, model_names=None,
                          *, baseline_report=None):
    """Replay the fixed neighbor candidate versus ABC, eight models and random.

    ``baseline_report`` is only for reproducible local experiments; it must
    contain audited, frozen per-target predictions and pre-target cutoffs.
    The HTTP API never accepts external reports.
    """
    from taiwan_lottery.ml.bingo_neighbors import neighbor_candidate, VERSION

    if not 1 <= count <= 1000 or not 100 <= window <= 1000 or not 1 <= threads <= 8:
        raise ValueError("鄰號回測最多1000期，視窗100～1000，執行緒1～8")
    data = _history(draws)
    eligible = _targets(data, count, start_term, end_term)
    terms = [data[i]["draw_term"] for i in eligible]
    if baseline_report is None:
        from taiwan_lottery.ml.bingo_association_evaluation import association_walk_forward
        base = association_walk_forward(data, count=count, start_term=start_term,
                                       end_term=end_term, threads=threads, window=window,
                                       progress=progress, model_names=model_names)
        baseline_report = _comparison_bundle(base)
    comparison = audit_comparison_bundle(baseline_report, data, terms)
    folds, outcomes = [], []
    for step, index in enumerate(eligible):
        history = data[max(0, index-window):index]
        calculated = neighbor_candidate(history)
        random = _random_ranking(data[index]["draw_term"])
        frozen = dict(target_term=data[index]["draw_term"], source_term=history[-1]["draw_term"],
                      training_cutoff=history[-1]["draw_term"], feature_cutoff=history[-1]["draw_term"],
                      history_from=history[0]["draw_term"], history_count=len(history),
                      history_sha256=canonical_hash([(r["draw_term"], r["numbers"]) for r in history]),
                      engine_version=VERSION, candidate=copy.deepcopy(calculated),
                      random_ranking=random, random_seed=RANDOM_SEED, is_live=False)
        frozen["decision_hash"] = canonical_hash(frozen)
        folds.append(frozen)
        # This is deliberately the first read of the target outcome in this loop.
        actual = sorted(data[index]["numbers"])
        for star in STARS:
            for name, numbers in (("neighbor", calculated["by_star"][str(star)]["numbers"]),
                                  ("random", random[:star])):
                outcomes.append(dict(target_term=frozen["target_term"], model=name, stars=star,
                                     predicted_numbers=list(numbers), actual_numbers=actual,
                                     hit_count=len(set(numbers) & set(actual)),
                                     draw_datetime=str(data[index]["draw_datetime"]), is_live=False))
        if progress:
            progress("neighbor_walk_forward", step+1, len(eligible), "running")
    rows = comparison + outcomes
    return dict(mode="neighbor_walk_forward", is_live=False, count=len(folds), folds=folds,
                results=rows, leaderboards=live_rankings(rows), validation=neighbor_evidence(rows),
                model_statuses=baseline_report.get("model_statuses", []), selected_method="shrink",
                selection_reason="鄰號為獨立研究候選，歷史回測不改正式融合權重",
                decision=dict(allowed_for_live=False, validated=False,
                              reason="研究結果未授權正式調權；需另行嚴格前瞻驗證與治理"),
                comparison_provenance=baseline_report.get("provenance", {}),
                comparison_sha256=canonical_hash(baseline_report),
                decision_chain_sha256=canonical_hash([f["decision_hash"] for f in folds]),
                score_kind="歷史條件頻率與相對研究分數，均非下一期中獎機率",
                methodology="固定1000期上限／lag1／0±1±2±3±5；每期獨占過去前綴，先凍結再揭答案；完整區間不挑段；研究與實戰分離",
                experiment_family=EXPERIMENT_FAMILY)
