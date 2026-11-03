"""Bounded, causal A/B/C association research; no language-model explanations.

Discovery, window selection and final confirmation use disjoint target ranges.
Same-draw association is descriptive: it never contributes a supposedly known
future antecedent to C. An association test is not a winning-strategy proof.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
import time

import numpy as np
from scipy.stats import binom, norm

from taiwan_lottery.bingo_clock import is_successor, to_taipei
from taiwan_lottery.ml.bingo_models import draw_matrix

VERSION = "bingo-associations-2"
FEATURE_VERSION = "bingo-features-causal-v1"
NO_RELIABLE = "目前沒有可靠關聯，僅供探索"


def _normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (80,) or not np.isfinite(values).all():
        raise ValueError("關聯分數必須包含 80 個有限數值")
    span = float(np.ptp(values))
    return (
        np.full(80, 0.5)
        if span <= np.finfo(float).eps
        else (values - values.min()) / span
    )


def _rank(values):
    return (np.lexsort((np.arange(80), -np.asarray(values))) + 1).tolist()


def _lag_counts(matrix, valid_steps, lag, target_start=0, target_stop=None):
    """Exact integer counts; a lag crossing any missing term is excluded.

    Matrix shape is (draws,80); valid_steps[i] connects draws i and i+1.
    Target boundaries concern answers, so an earlier known antecedent may lie
    before target_start. This is necessary for causal validation at split edges.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    valid_steps = np.asarray(valid_steps, dtype=bool)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 80
        or valid_steps.shape != (max(0, len(matrix) - 1),)
    ):
        raise ValueError("二元矩陣或期號連續性維度無效")
    if lag < 1 or int(lag) != lag:
        raise ValueError("延遲必須為正整數")
    stop = len(matrix) if target_stop is None else min(len(matrix), int(target_stop))
    targets = np.arange(
        max(int(target_start), lag), max(max(int(target_start), lag), stop)
    )
    if not len(targets):
        return np.zeros(80, dtype=np.int64), np.zeros((80, 80), dtype=np.int64), 0
    gaps = np.r_[0, np.cumsum(~valid_steps)]
    targets = targets[gaps[targets] == gaps[targets - lag]]
    before, after = matrix[targets - lag], matrix[targets]
    return (
        np.rint(before.sum(axis=0)).astype(np.int64),
        np.rint(before.T @ after).astype(np.int64),
        len(targets),
    )


def _cooccurrence_counts(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    successes = np.rint(matrix.T @ matrix).astype(np.int64)
    np.fill_diagonal(successes, 0)
    return np.rint(matrix.sum(axis=0)).astype(np.int64), successes


def _group_counts(matrix, valid_steps, antecedents, consequents, lag, start, stop):
    targets = np.arange(max(lag, start), max(max(lag, start), min(len(matrix), stop)))
    gaps = np.r_[0, np.cumsum(~np.asarray(valid_steps, dtype=bool))]
    targets = targets[gaps[targets] == gaps[targets - lag]]
    if not len(targets):
        return np.zeros(len(antecedents), dtype=np.int64), np.zeros(
            (len(antecedents), len(consequents)), dtype=np.int64
        )
    before = np.stack(
        [matrix[targets - lag][:, list(group)].prod(axis=1) for group in antecedents],
        axis=1,
    )
    after = np.stack(
        [matrix[targets][:, list(group)].prod(axis=1) for group in consequents], axis=1
    )
    return np.rint(before.sum(axis=0)).astype(np.int64), np.rint(
        before.T @ after
    ).astype(np.int64)


def _wilson(successes, trials, alpha):
    if not trials:
        return [0.0, 1.0]
    z = float(norm.isf(alpha / 2))
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    width = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return [max(0.0, center - width), min(1.0, center + width)]


def _evidence(successes, trials, p0, alpha, min_triggers):
    p = float(binom.sf(successes - 1, trials, p0)) if trials else 1.0
    low, high = _wilson(successes, trials, alpha)
    estimate = (successes + 8 * p0) / (trials + 8)
    return {
        "successes": int(successes),
        "triggers": int(trials),
        "sample_count": int(trials),
        "random_baseline": float(p0),
        "smoothed_rate": float(estimate),
        "lift": float(estimate / p0),
        "p_value": p,
        "adjusted_alpha": alpha,
        "ci_low": low,
        "ci_high": high,
        "approved": bool(trials >= min_triggers and p <= alpha and low > p0),
    }


def _rule(kind, lag, antecedent, consequence, window, successes, trials, p0):
    identity = f"{kind}:{lag}:{antecedent}:{consequence}:{window}"
    return {
        "id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "family": "B" if kind == "group" else "A",
        "kind": kind,
        "lag": int(lag),
        "antecedent": [int(i) + 1 for i in antecedent],
        "consequence": [int(i) + 1 for i in consequence],
        "window": int(window),
        "random_baseline": float(p0),
        "discovery_successes": int(successes),
        "discovery_triggers": int(trials),
    }


def _order_rules(rules):
    def score(rule):
        n = rule["discovery_triggers"]
        p0 = rule["random_baseline"]
        s = rule["discovery_successes"]
        z = (s - n * p0) / math.sqrt(max(n * p0 * (1 - p0), 1))
        return (-z, -n, rule["lag"], rule["antecedent"], rule["consequence"])

    return sorted(rules, key=score)


def _discover(matrix, steps, window, split, max_lag, max_pairs, max_rules):
    temporal = []
    same = []
    group_rules = []
    hypotheses = 0
    for lag in range(1, min(max_lag, len(matrix) - 1) + 1):
        trials, successes, _ = _lag_counts(matrix, steps, lag, 0, split)
        hypotheses += 80 * 80
        z = (successes - trials[:, None] * 0.25) / np.sqrt(
            np.maximum(trials[:, None] * 0.25 * 0.75, 1)
        )
        eligible = np.flatnonzero(np.broadcast_to(trials[:, None] > 0, (80, 80)))
        # A bounded shortlist per lag; every searched cell still counts in alpha.
        best = eligible[np.argsort(-z.ravel()[eligible], kind="stable")[:max_rules]]
        for flat in best:
            a, b = divmod(int(flat), 80)
            temporal.append(
                _rule(
                    "temporal",
                    lag,
                    (a,),
                    (b,),
                    window,
                    successes[a, b],
                    trials[a],
                    0.25,
                )
            )
    trials, cooc = _cooccurrence_counts(matrix[:split])
    hypotheses += 80 * 79
    z = (cooc - trials[:, None] * (19 / 79)) / np.sqrt(
        np.maximum(trials[:, None] * (19 / 79) * (60 / 79), 1)
    )
    np.fill_diagonal(z, -np.inf)
    eligible = np.flatnonzero(
        np.broadcast_to(trials[:, None] > 0, (80, 80)) & ~np.eye(80, dtype=bool)
    )
    for flat in eligible[
        np.argsort(-z.ravel()[eligible], kind="stable")[: max(1, max_rules // 4)]
    ]:
        a, b = divmod(int(flat), 80)
        same.append(
            _rule("cooccurrence", 0, (a,), (b,), window, cooc[a, b], trials[a], 19 / 79)
        )
    # Antecedent pairs and consequence groups are chosen on discovery only.
    a, b = np.triu_indices(80, 1)
    pair_order = np.lexsort((b, a, -cooc[a, b]))
    pairs = [(int(a[i]), int(b[i])) for i in pair_order[:max_pairs]]
    frequencies = matrix[:split].sum(axis=0)
    number_order = np.lexsort((np.arange(80), -frequencies))
    groups = list(pairs[: max(1, max_pairs // 2)])
    for pair in pairs[: max(1, max_pairs // 2)]:
        third = next(int(i) for i in number_order if i not in pair)
        groups.append(tuple(sorted((*pair, third))))
    groups = list(dict.fromkeys(groups))[:max_pairs]
    hypotheses += math.comb(80, 2)
    for lag in range(2, min(5, len(matrix) - 1) + 1):
        counts, successes = _group_counts(matrix, steps, pairs, groups, lag, 0, split)
        hypotheses += len(pairs) * len(groups)
        for ai, antecedent in enumerate(pairs):
            if counts[ai] == 0:
                continue
            for bi, consequence in enumerate(groups):
                p0 = math.comb(20, len(consequence)) / math.comb(80, len(consequence))
                group_rules.append(
                    _rule(
                        "group",
                        lag,
                        antecedent,
                        consequence,
                        window,
                        successes[ai, bi],
                        counts[ai],
                        p0,
                    )
                )
    per_family = max(1, max_rules // 2)
    # Reserve temporal capacity so descriptive same-draw facts cannot crowd out
    # all genuinely next-target hypotheses on a highly correlated test history.
    a_rules = (
        _order_rules(temporal)[: max(1, per_family - max(1, per_family // 4))]
        + _order_rules(same)[: max(1, per_family // 4)]
    )
    return a_rules[:per_family] + _order_rules(group_rules)[:per_family], hypotheses


def _rule_counts(rule, matrix, steps, start, stop):
    ant = [i - 1 for i in rule["antecedent"]]
    con = [i - 1 for i in rule["consequence"]]
    if rule["kind"] == "cooccurrence":
        triggers = matrix[start:stop, ant[0]]
        return int(np.dot(triggers, matrix[start:stop, con[0]])), int(triggers.sum())
    trials, successes = _group_counts(
        matrix, steps, [tuple(ant)], [tuple(con)], rule["lag"], start, stop
    )
    return int(successes[0, 0]), int(trials[0])


def _active(rule, matrix, steps):
    lag = rule["lag"]
    if lag < 1 or lag > len(matrix):
        return False
    # The next target is defined as the successor; only the preceding known
    # transitions need verification. A source gap must not be treated as a lag.
    if lag > 1 and not np.all(steps[len(matrix) - lag :]):
        return False
    return bool(np.all(matrix[-lag, np.array(rule["antecedent"]) - 1]))


def _predictive_contribution(rules, matrix, steps):
    values = np.zeros(80)
    active = []
    for rule in rules:
        if (
            rule.get("reliable")
            and rule["kind"] != "cooccurrence"
            and _active(rule, matrix, steps)
        ):
            strength = max(
                0.0, rule["confirmation"]["ci_low"] - rule["random_baseline"]
            )
            values[np.array(rule["consequence"]) - 1] += strength
            active.append(rule)
    return values, active


def _exploratory_a(matrix, steps, max_lag):
    values = np.zeros(80)
    total = 0.0
    explanations = []
    count_cache = {}
    for lag in range(1, min(max_lag, len(matrix)) + 1):
        trials, successes, _ = _lag_counts(matrix, steps, lag)
        count_cache[lag] = (trials, successes)
        if lag > 1 and not np.all(steps[len(matrix) - lag :]):
            continue
        active = np.flatnonzero(matrix[-lag])
        rates = (successes + 2) / (trials[:, None] + 8)
        weight = math.exp(-(lag - 1) / 10)
        values += weight * rates[active].mean(axis=0)
        total += weight
        for a in active:
            if trials[a]:
                b = int(np.argmax(rates[a]))
                explanations.append(
                    (
                        float(rates[a, b] - 0.25),
                        lag,
                        int(a) + 1,
                        b + 1,
                        int(successes[a, b]),
                        int(trials[a]),
                    )
                )
    reasons = []
    for _, lag, a, b, s, n in sorted(explanations, reverse=True)[:3]:
        reasons.append(
            f"近資料中，前 {lag} 期出現 {a:02d} 號後，{b:02d} 號出現 {s}／{n} 次；屬探索統計"
        )
    return values / max(total, 1e-12), reasons, count_cache


def _evidence_detail(rule, matrix, steps, adjusted_alpha, min_triggers):
    detail = dict(rule)
    successes, trials = _rule_counts(rule, matrix, steps, 0, len(matrix))
    measured = _evidence(
        successes, trials, rule["random_baseline"], adjusted_alpha, min_triggers
    )
    detail.update(
        trigger_count=trials,
        success_count=successes,
        conditional_frequency=successes / trials if trials else None,
        smoothed_rate=measured["smoothed_rate"],
        lift=measured["lift"],
        ci_low=measured["ci_low"],
        ci_high=measured["ci_high"],
        effective_samples=trials,
        independently_validated=bool(rule.get("reliable")),
        confidence="通過分離資料確認；不是策略勝率證明"
        if rule.get("reliable")
        else "未通過獨立確認；僅供探索",
    )
    return detail


def _a_explanatory_rules(
    matrix,
    steps,
    window,
    selected,
    max_lag,
    adjusted_alpha,
    min_triggers,
    count_cache=None,
):
    details = []
    for lag in range(1, min(max_lag, len(matrix)) + 1):
        if lag > 1 and not np.all(steps[len(matrix) - lag :]):
            continue
        if count_cache and lag in count_cache:
            trials, successes = count_cache[lag]
        else:
            trials, successes, _ = _lag_counts(matrix, steps, lag)
        for b in selected:
            active = np.flatnonzero((matrix[-lag] > 0) & (trials > 0))
            if not len(active):
                continue
            rates = (successes[active, b - 1] + 2) / (trials[active] + 8)
            chosen = int(active[np.lexsort((active, -trials[active], -rates))[0]])
            rule = _rule(
                "temporal",
                lag,
                (chosen,),
                (b - 1,),
                window,
                successes[chosen, b - 1],
                trials[chosen],
                0.25,
            )
            rule.update(
                reliable=False,
                predictive=True,
                active_for_next=True,
                next_target_eligible=False,
                history={"count": len(matrix)},
                stage="all_known_descriptive",
            )
            details.append(
                _evidence_detail(rule, matrix, steps, adjusted_alpha, min_triggers)
            )
    details.sort(
        key=lambda r: (-r["lift"], -r["trigger_count"], r["lag"], r["consequence"])
    )
    chosen = []
    for detail in details:
        if detail["consequence"] not in [r["consequence"] for r in chosen]:
            chosen.append(detail)
        if len(chosen) >= len(selected):
            break
    return chosen


def _rule_reasons(details):
    reasons = []
    for rule in details[:3]:
        status = (
            "已通過分離資料確認，策略仍待實戰"
            if rule.get("reliable")
            else "未通過獨立確認，僅供探索"
        )
        reasons.append(
            f"前 {rule['lag']} 期 {'、'.join(map(str, rule['antecedent']))} 號後，{'、'.join(map(str, rule['consequence']))} 號組合出現 {rule['success_count']}／{rule['trigger_count']} 次；{status}"
        )
    return reasons


def _models_for_star(model_scores_by_star, star, model_weights_by_star):
    raw = (model_scores_by_star or {}).get(
        str(star), (model_scores_by_star or {}).get(star, {})
    )
    if isinstance(raw, (list, tuple, np.ndarray)):
        raw = {"provided_fusion": raw}
    weights_raw = (model_weights_by_star or {}).get(
        str(star), (model_weights_by_star or {}).get(star, {})
    )
    matrices = []
    weights = []
    names = []
    excluded = []
    for name, value in sorted(raw.items()):
        try:
            vector = value.get("scores") if isinstance(value, dict) else value
            weight = weights_raw.get(
                name, value.get("weight", 1.0) if isinstance(value, dict) else 1.0
            )
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("投票權重無效")
            matrices.append(_normalize(vector))
            weights.append(weight)
            names.append(name)
        except (TypeError, ValueError):
            excluded.append(name)
    if not matrices:
        return np.zeros(80), {}, excluded
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights[:] = 1
    weights /= weights.sum()
    return weights @ np.stack(matrices), dict(zip(names, map(float, weights))), excluded


def _history(rows):
    return (
        {"from": rows[0]["draw_term"], "to": rows[-1]["draw_term"], "count": len(rows)}
        if rows
        else {"from": None, "to": None, "count": 0}
    )


def _candidate(
    values,
    stars,
    label,
    reasons,
    history,
    evidence,
    *,
    model_weights=None,
    components=None,
):
    scores = _normalize(values)
    ranking = _rank(scores)
    return {
        "scores": scores.tolist(),
        "ranking": ranking,
        "numbers": ranking[:stars],
        "label": label,
        "reasons": reasons[:3],
        "validated": any(
            r.get("reliable") and r.get("predictive") and r.get("active_for_next")
            for r in evidence
        ),
        "association_evidence_validated": any(
            r.get("reliable") and r.get("predictive") and r.get("active_for_next")
            for r in evidence
        ),
        "strategy_proven": False,
        "shadow_only": True,
        "history": history,
        "evidence": evidence,
        "rules": evidence,
        "evidence_ids": [r["id"] for r in evidence],
        "model_weights": model_weights or {},
        "components": components or {},
        "relative_confidence": float(scores[np.array(ranking[:stars]) - 1].mean()),
        "score_kind": "相對分數，非下一期中獎機率",
        "version": VERSION,
        "feature_version": FEATURE_VERSION,
    }


def association_candidates(
    draws,
    model_scores_by_star=None,
    *,
    windows=None,
    max_history=1000,
    model_weights_by_star=None,
    target_date=None,
    alpha=0.05,
    min_triggers=20,
    max_lag=20,
    max_pairs=16,
    max_rules=32,
):
    """Return nine reproducible, genuinely computed A/B/C shadow candidates.

    C is a research composite: base models .60, today-frequency .15, momentum
    .10, gap .05, descriptive cooccurrence .05, confirmed predictive rules .05.
    If no independently confirmed active rule exists, its .05 is reassigned to
    model fusion (or frequency when no model is available); unvalidated rules
    always contribute exactly zero to C.
    """
    started = time.perf_counter()
    cpu_started = time.process_time()
    if (
        not 1 <= max_history <= 10000
        or not 1 <= max_lag <= 20
        or not 1 <= max_pairs <= 16
        or not 2 <= max_rules <= 32
        or min_triggers < 20
        or not 0 < alpha <= 0.05
    ):
        raise ValueError(
            "關聯搜尋必須受限：最多10000期、20延遲、16群組、32規則及至少20次觸發"
        )
    rows = [
        {
            "draw_term": str(d["draw_term"]),
            "draw_datetime": d.get("draw_datetime"),
            "numbers": list(d["numbers"]),
        }
        for d in draws[-max_history:]
    ]
    if not rows:
        raise ValueError("沒有已知歷史資料可供關聯分析")
    terms = [int(r["draw_term"]) for r in rows]
    if any(t <= 0 for t in terms) or any(b <= a for a, b in zip(terms, terms[1:])):
        raise ValueError("歷史資料必須按照不重複的真實期號遞增")
    matrix = draw_matrix([r["numbers"] for r in rows]).astype(np.float64)
    steps = []
    for before, after in zip(rows, rows[1:]):
        a, b = before["draw_datetime"], after["draw_datetime"]
        if a is not None and b is not None:
            a = datetime.fromisoformat(a) if isinstance(a, str) else a
            b = datetime.fromisoformat(b) if isinstance(b, str) else b
            if to_taipei(b) <= to_taipei(a):
                raise ValueError("開獎時間必須遞增")
            try:
                steps.append(
                    is_successor(before["draw_term"], a, after["draw_term"], b)
                )
            except ValueError:
                steps.append(False)
        else:
            steps.append(int(after["draw_term"]) == int(before["draw_term"]) + 1)
    steps = np.array(steps, dtype=bool)
    proposed = (
        [min(len(rows), 200), min(len(rows), 500), min(len(rows), 1000)]
        if windows is None
        else list(windows)
    )
    if (
        not proposed
        or len(proposed) > 3
        or any(
            isinstance(w, bool)
            or not isinstance(w, (int, np.integer))
            or w < 5
            or w > 10000
            for w in proposed
        )
    ):
        raise ValueError("最多允許三個5～10000期候選視窗")
    windows = sorted(set(min(int(w), len(rows), max_history) for w in proposed))
    all_rules = []
    per_window = {}
    hypotheses = 0
    # Every candidate sees the SAME unopened confirmation targets. Using each
    # window's own quarter would leak a short window's selection labels into a
    # long window's supposed holdout. Only discovery reference length varies.
    global_discovery = max(1, len(rows) // 2)
    global_selection = max(global_discovery, min(len(rows) - 1, (len(rows) * 3) // 4))
    for window in windows:
        origin = max(0, global_discovery - window)
        values = matrix[origin:]
        valid = steps[origin:] if len(values) > 1 else steps[:0]
        discovery = global_discovery - origin
        selection = global_selection - origin
        rules, count = _discover(
            values, valid, window, discovery, max_lag, max_pairs, max_rules
        )
        context = matrix[-window:]
        context_steps = steps[-(window - 1) :] if window > 1 else steps[:0]
        per_window[window] = {
            "matrix": values,
            "steps": valid,
            "discovery": discovery,
            "selection": selection,
            "origin": origin,
            "rules": rules,
            "context_matrix": context,
            "context_steps": context_steps,
        }
        hypotheses += count
        all_rules.extend(rules)
    adjusted = alpha / max(1, hypotheses)
    selections = {}
    for rule in all_rules:
        data = per_window[rule["window"]]
        values = data["matrix"]
        valid = data["steps"]
        discovery, selection = data["discovery"], data["selection"]
        s, n = _rule_counts(rule, values, valid, 0, discovery)
        rule["discovery"] = _evidence(
            s, n, rule["random_baseline"], adjusted, min_triggers
        )
        s, n = _rule_counts(rule, values, valid, discovery, selection)
        rule["selection_validation"] = _evidence(
            s, n, rule["random_baseline"], adjusted, min_triggers
        )
        rule["reliable"] = False
        rule["predictive"] = rule["kind"] != "cooccurrence"
    # Select each family's window before exposing any final confirmation label.
    for family in ("A", "B"):
        metrics = []
        for window in windows:
            candidates = [
                r for r in all_rules if r["window"] == window and r["family"] == family
            ]
            passed = [
                r
                for r in candidates
                if r["discovery"]["approved"] and r["selection_validation"]["approved"]
            ]
            metric = sum(
                max(0.0, r["selection_validation"]["ci_low"] - r["random_baseline"])
                for r in passed
            )
            metrics.append((metric, window))
        chosen = max(metrics)[1]
        data = per_window[chosen]
        segment_rows = rows[data["origin"] :]
        selections[family] = {
            "window": chosen,
            "candidate_windows": windows,
            "selection_metrics": [{"window": w, "score": m} for m, w in metrics],
            "selection_end_term": segment_rows[data["selection"] - 1]["draw_term"]
            if data["selection"]
            else None,
            "confirmation_start_term": segment_rows[data["selection"]]["draw_term"]
            if data["selection"] < len(segment_rows)
            else None,
            "reason": "僅用發現與選擇驗證段選視窗；最終確認段未參與視窗挑選",
        }
    retained = []
    for family in ("A", "B"):
        chosen = selections[family]["window"]
        data = per_window[chosen]
        for rule in data["rules"]:
            if rule["family"] != family:
                continue
            s, n = _rule_counts(
                rule,
                data["matrix"],
                data["steps"],
                data["selection"],
                len(data["matrix"]),
            )
            rule["confirmation"] = _evidence(
                s, n, rule["random_baseline"], adjusted, min_triggers
            )
            rule["reliable"] = bool(
                rule["discovery"]["approved"]
                and rule["selection_validation"]["approved"]
                and rule["confirmation"]["approved"]
            )
            rule["window_selection"] = selections[family]
            evidence_rows = rows[data["origin"] :]
            rule["history"] = _history(evidence_rows)
            rule["context_history"] = _history(rows[-chosen:])
            rule["evidence_history"] = _history(evidence_rows)
            rule["split_history"] = {
                "discovery": _history(evidence_rows[: data["discovery"]]),
                "selection_validation": _history(
                    evidence_rows[data["discovery"] : data["selection"]]
                ),
                "confirmation": _history(evidence_rows[data["selection"] :]),
            }
            rule["active_for_next"] = _active(
                rule, data["context_matrix"], data["context_steps"]
            )
            rule["next_target_eligible"] = bool(
                rule["reliable"] and rule["predictive"] and rule["active_for_next"]
            )
            retained.append(rule)
    # Selected families have at most max_rules combined, including exploratory rows.
    retained = retained[:max_rules]
    a_data = per_window[selections["A"]["window"]]
    b_data = per_window[selections["B"]["window"]]
    a_raw, a_reasons, a_count_cache = _exploratory_a(
        a_data["context_matrix"], a_data["context_steps"], max_lag
    )
    b_raw = np.zeros(80)
    b_reasons = []
    b_details = []
    for rule in retained:
        if rule["family"] != "B" or not _active(
            rule, b_data["context_matrix"], b_data["context_steps"]
        ):
            continue
        s, n = _rule_counts(
            rule,
            b_data["context_matrix"],
            b_data["context_steps"],
            0,
            len(b_data["context_matrix"]),
        )
        posterior = (s + 8 * rule["random_baseline"]) / (n + 8)
        b_raw[np.array(rule["consequence"]) - 1] += max(
            0.0, posterior - rule["random_baseline"]
        )
        b_details.append(
            _evidence_detail(
                rule,
                b_data["context_matrix"],
                b_data["context_steps"],
                adjusted,
                min_triggers,
            )
        )
        b_reasons.append(
            f"前 {rule['lag']} 期的 {'、'.join(map(str, rule['antecedent']))} 號組合後，{'、'.join(map(str, rule['consequence']))} 號組合出現 {s}／{n} 次；屬探索統計"
        )
    verified_a, active_a = _predictive_contribution(
        [r for r in retained if r["family"] == "A"],
        a_data["context_matrix"],
        a_data["context_steps"],
    )
    verified_b, active_b = _predictive_contribution(
        [r for r in retained if r["family"] == "B"],
        b_data["context_matrix"],
        b_data["context_steps"],
    )
    # Once a confirmed rule is active, A/B use its contribution rather than
    # mixing unvalidated associations into a candidate labelled confirmed.
    if active_a:
        a_raw = verified_a
        a_details = [
            _evidence_detail(
                r,
                a_data["context_matrix"],
                a_data["context_steps"],
                adjusted,
                min_triggers,
            )
            for r in active_a
        ]
    else:
        a_details = _a_explanatory_rules(
            a_data["context_matrix"],
            a_data["context_steps"],
            selections["A"]["window"],
            _rank(_normalize(a_raw))[:5],
            max_lag,
            adjusted,
            min_triggers,
            a_count_cache,
        )
    if active_b:
        b_raw = verified_b
        b_details = [
            _evidence_detail(
                r,
                b_data["context_matrix"],
                b_data["context_steps"],
                adjusted,
                min_triggers,
            )
            for r in active_b
        ]
    verified_details = [
        _evidence_detail(
            r,
            per_window[r["window"]]["context_matrix"],
            per_window[r["window"]]["context_steps"],
            adjusted,
            min_triggers,
        )
        for r in active_a + active_b
    ]
    # Today is explicitly the next target date, when the runtime supplies it.

    dates = []
    for row in rows:
        value = row["draw_datetime"]
        value = datetime.fromisoformat(value) if isinstance(value, str) else value
        dates.append(to_taipei(value).date() if value is not None else None)
    if isinstance(target_date, datetime):
        target_day = to_taipei(target_date).date()
    elif isinstance(target_date, str):
        target_day = date.fromisoformat(target_date)
    else:
        target_day = target_date or dates[-1]
    if target_day is not None and dates[-1] is not None and target_day < dates[-1]:
        raise ValueError("目標日期不可早於已知來源日期")
    today = np.array([target_day is not None and d == target_day for d in dates])
    today_matrix = matrix[today]
    frequency = today_matrix.mean(axis=0) if len(today_matrix) else np.zeros(80)
    short = (
        today_matrix[-min(10, len(today_matrix)) :].mean(axis=0)
        if len(today_matrix)
        else np.zeros(80)
    )
    older = today_matrix[-min(30, len(today_matrix)) : -min(10, len(today_matrix))]
    momentum = short - (older.mean(axis=0) if len(older) else frequency)
    if len(today_matrix):
        last_seen = np.max(
            np.where(today_matrix > 0, np.arange(len(today_matrix))[:, None], -1),
            axis=0,
        )
        gap = np.minimum((len(today_matrix) - 1 - last_seen) / 20, 2)
    else:
        gap = np.zeros(80)
    today_trials, today_cooc = _cooccurrence_counts(today_matrix)
    cooc = ((today_cooc + 0.0) / (np.maximum(today_trials, 1)[:, None]))[
        np.flatnonzero(matrix[-1])
    ].mean(axis=0)
    verified = verified_a + verified_b
    active_rules = active_a + active_b
    by_star = {}
    excluded = {}
    for star in (3, 4, 5):
        model, weights, invalid = _models_for_star(
            model_scores_by_star, star, model_weights_by_star
        )
        excluded[str(star)] = invalid
        components = {
            "models": 0.60,
            "today_frequency": 0.15,
            "momentum": 0.10,
            "gap": 0.05,
            "cooccurrence": 0.05,
            "validated_rules": 0.05 if active_rules else 0.0,
        }
        if not weights:
            components["today_frequency"] += components["models"]
            components["models"] = 0.0
        if not active_rules:
            components["models" if weights else "today_frequency"] += 0.05
        c_raw = (
            components["models"] * _normalize(model)
            + components["today_frequency"] * _normalize(frequency)
            + components["momentum"] * _normalize(momentum)
            + components["gap"] * _normalize(gap)
            + components["cooccurrence"] * _normalize(cooc)
            + components["validated_rules"]
            * (_normalize(verified) if active_rules else np.zeros(80))
        )
        association_label = "關聯通過獨立資料檢查；策略仍僅供影子驗證，非中獎保證"
        c_reasons = [
            f"實際融合 {len(weights)} 個可用模型；今日頻率採 {int(today.sum())} 期資料",
            f"只納入 {len(active_rules)} 條獨立確認且當前觸發的預測關聯；未驗證規則貢獻為零",
            f"動能、間隔與同期共現均由已知資料計算；來源截止 {rows[-1]['draw_term']} 期",
        ]
        a_numbers = set(_rank(_normalize(a_raw))[:star])
        b_numbers = set(_rank(_normalize(b_raw))[:star])
        a_evidence = [r for r in a_details if set(r["consequence"]) & a_numbers][:3]
        b_evidence = [r for r in b_details if set(r["consequence"]) & b_numbers][:3]
        by_star[str(star)] = {
            "A": _candidate(
                a_raw,
                star,
                association_label if active_a else NO_RELIABLE,
                _rule_reasons(a_evidence) or ["歷史觸發樣本不足；維持確定性探索排序"],
                _history(rows[a_data["origin"] :]),
                a_evidence,
            ),
            "B": _candidate(
                b_raw,
                star,
                association_label if active_b else NO_RELIABLE,
                _rule_reasons(b_evidence) or ["目前沒有觸發的已知組合；不加入隨機擾動"],
                _history(rows[b_data["origin"] :]),
                b_evidence,
            ),
            "C": _candidate(
                c_raw,
                star,
                association_label if active_rules else NO_RELIABLE,
                c_reasons,
                _history(rows),
                verified_details,
                model_weights=weights,
                components=components,
            ),
        }
    for star in ("3", "4", "5"):
        for family in ("A", "B"):
            data = per_window[selections[family]["window"]]
            candidate = by_star[star][family]
            candidate["context_history"] = _history(
                rows[-selections[family]["window"] :]
            )
            candidate["training_history"] = _history(
                rows[data["origin"] : global_discovery]
            )
            candidate["validation_history"] = {
                "selection_validation": _history(
                    rows[global_discovery:global_selection]
                ),
                "confirmation": _history(rows[global_selection:]),
            }
        by_star[star]["C"]["context_history"] = _history(rows)
        by_star[star]["C"]["validation_history"] = {
            "selection_validation": _history(rows[global_discovery:global_selection]),
            "confirmation": _history(rows[global_selection:]),
        }
    digest = hashlib.sha256(
        json.dumps(
            [(r["draw_term"], r["numbers"]) for r in rows], separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "version": VERSION,
        "feature_version": FEATURE_VERSION,
        "source_term": rows[-1]["draw_term"],
        "snapshot_hash": digest,
        "by_star": by_star,
        "rules": retained,
        "window_selection": selections,
        "statistics": {
            "target_date": target_day.isoformat() if target_day is not None else None,
            "today_samples": int(today.sum()),
            "today_frequency": frequency.tolist(),
            "momentum": momentum.tolist(),
            "gap": gap.tolist(),
            "cooccurrence": cooc.tolist(),
            "same_draw_random_conditional": 19 / 79,
        },
        "history": _history(rows),
        "validation_segments": {
            "discovery_end_term": rows[global_discovery - 1]["draw_term"],
            "selection_end_term": rows[global_selection - 1]["draw_term"],
            "confirmation_start_term": rows[global_selection]["draw_term"]
            if global_selection < len(rows)
            else None,
        },
        "warning": NO_RELIABLE
        if not active_rules
        else "關聯檢查不等同策略實戰優勢；先維持影子模式",
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "matrix_bytes": int(matrix.nbytes),
            "windows": windows,
            "searched_hypotheses": int(hypotheses),
            "bonferroni_alpha": adjusted,
            "rule_count": len(retained),
            "independently_confirmed_rules": sum(r["reliable"] for r in retained),
            "active_confirmed_predictive_rules": len(active_rules),
            "max_lag": max_lag,
            "max_pairs": max_pairs,
            "max_rules": max_rules,
            "excluded_models": excluded,
            "term_gaps": int((~steps).sum()),
        },
        "methodology": "目標期不作輸入；發現、選擇驗證、最終確認的目標區間分離；所有搜尋假設納入Bonferroni；同日共現僅描述，不當作已知未來條件",
        "statistical_assumption": "各期公平獨立抽取20/80；條件次數採二項近似、群組隨機機率採不放回超幾何；尚非策略勝率證明",
    }
