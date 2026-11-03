"""Bounded nested, causal walk-forward evaluation.

All candidate scores are computed before a target answer is read. Inner selection
and validation end before the independent outer block. A fixed baseline remains
available in every fold, and a final outer block is a confirmation set, never a
place to select whichever profile happened to win. This module never writes the
formal live prediction ledger.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import time
from typing import Callable

import numpy as np

from taiwan_lottery.ml.bingo_evaluation import (
    METHODS,
    STARS,
    _rank,
    _validate_draws,
    fuse,
    fuse_many,
    live_rankings,
    normalize_scores,
    summarize_outcomes,
)

FEATURE_VERSION = "bingo-features-causal-v1"
NESTED_VERSION = "nested-walk-forward-v1"


@dataclass(frozen=True)
class CandidateProfile:
    """Only parameters that actually change computation are exposed."""

    id: str
    context_window: int = 1000
    train_window: int = 1000
    frequency_mix: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)
    cdm_prior: float = 8.0
    retrain_every: int = 100
    method: str = "shrink"
    cdm_prior_kind: str = "uniform"

    def __post_init__(self):
        object.__setattr__(self, "frequency_mix", tuple(self.frequency_mix))
        if self.cdm_prior_kind not in ("uniform", "recent20"):
            raise ValueError("CDM 先驗種類無效")
        if not self.id or len(self.id) > 40:
            raise ValueError("候選策略版本無效")
        if (
            not 5 <= self.context_window <= 10000
            or not 100 <= self.train_window <= 10000
        ):
            raise ValueError("特徵視窗需為 5～10000，訓練視窗至少 100 期")
        if not 10 <= self.retrain_every <= 1000 or self.method not in METHODS:
            raise ValueError("重訓頻率或投票方法無效")
        values = np.asarray(self.frequency_mix, dtype=float)
        if (
            values.shape != (4,)
            or not np.isfinite(values).all()
            or (values < 0).any()
            or abs(float(values.sum()) - 1) > 1e-8
        ):
            raise ValueError("頻率特徵權重必須為四個非負且總和為 1 的數值")
        if not math.isfinite(self.cdm_prior) or not 0 < self.cdm_prior <= 1000:
            raise ValueError("CDM 先驗強度無效")

    def payload(self, method=None):
        result = asdict(self)
        result["frequency_mix"] = list(self.frequency_mix)
        if method is not None:
            result["method"] = method
        return result


def default_profiles(window=1000):
    """Three suites bound search and memory, rather than a Cartesian grid."""
    return (
        CandidateProfile("fixed", window, window),
        CandidateProfile(
            "recent",
            min(60, window),
            min(300, window),
            (0.55, 0.25, 0.15, 0.05),
            24.0,
            200,
            cdm_prior_kind="recent20",
        ),
        CandidateProfile(
            "stable",
            min(200, window),
            window,
            (0.15, 0.25, 0.30, 0.30),
            16.0,
            100,
            cdm_prior_kind="recent20",
        ),
    )


def profile_scores(
    suite, history, profile: CandidateProfile, *, baseline=False, matrix=None, row=None
):
    """Score one real fitted suite with a causal, bounded feature context.

    Tree/neural parameters are fitted only on the profile's training prefix.
    Frequency decay and CDM prior are genuine scorer hyperparameters; changing
    either also recomputes Ensemble, rather than just changing a display label.
    """
    context = history[-min(profile.context_window, profile.train_window) :]
    if hasattr(suite, "scores_prepared"):
        from taiwan_lottery.ml.bingo_models import draw_matrix, latest_features

        matrix = draw_matrix(context) if matrix is None else matrix
        row = latest_features(matrix) if row is None else row
        score_map = suite.scores_prepared(matrix, row)
    else:
        score_map = suite.scores(context)
    if baseline:
        return score_map
    if not ("frequency" in score_map or "cdm" in score_map):
        return score_map
    from taiwan_lottery.ml.bingo_models import draw_matrix, latest_features

    matrix = draw_matrix(context) if matrix is None else matrix
    row = latest_features(matrix) if row is None else row
    changed = False
    if "frequency" in score_map:
        rate = row[:, :4] @ np.asarray(profile.frequency_mix)
        values = 0.7 * rate + 0.2 * row[:, 5] + 0.025 * row[:, 4]
        score_map["frequency"] = normalize_scores(values)
        changed = True
    if "cdm" in score_map:
        # A uniform prior's strength is erased by min-max normalization. Only
        # genuinely nonuniform, past-derived prior alternatives merit search.
        prior = (
            matrix[-min(20, len(matrix)) :].mean(axis=0)
            if profile.cdm_prior_kind == "recent20"
            else 0.25
        )
        values = (matrix.sum(axis=0) + profile.cdm_prior * prior) / (
            len(matrix) + profile.cdm_prior
        )
        score_map["cdm"] = normalize_scores(values)
        changed = True
    sources = [values for name, values in score_map.items() if name != "ensemble"]
    if changed and "ensemble" in score_map and len(sources) >= 2:
        score_map["ensemble"] = normalize_scores(np.mean(sources, axis=0))
    return score_map


def paired_gate(
    pairs,
    stars,
    *,
    comparison_count=1,
    attempt=1,
    minimum_samples=100,
    minimum_improvement=0.03,
    resource_ratio=1.0,
):
    """Prespecified, multiplicity/spending-adjusted paired holdout gate."""
    n = len(pairs)
    alpha = 0.05 / max(1, comparison_count) / (max(1, attempt) * (max(1, attempt) + 1))
    differences = np.asarray(
        [float(row["challenger_hits"]) - float(row["champion_hits"]) for row in pairs]
    )
    full_differences = np.asarray(
        [
            float(row["challenger_hits"] == stars)
            - float(row["champion_hits"] == stars)
            for row in pairs
        ]
    )
    delta = float(differences.mean()) if n else 0.0
    full_delta = float(full_differences.mean()) if n else 0.0
    width = 2 * stars * math.sqrt(math.log(2 / alpha) / (2 * n)) if n else math.inf
    full_width = 2 * math.sqrt(math.log(2 / alpha) / (2 * n)) if n else math.inf
    lower = delta - width
    full_lower = full_delta - full_width
    approved = (
        n >= minimum_samples
        and lower > minimum_improvement
        and full_lower >= -0.02
        and math.isfinite(resource_ratio)
        and resource_ratio <= 1.5
    )
    return {
        "samples": n,
        "delta": delta,
        "lower_bound": lower if n else None,
        "full_hit_delta": full_delta,
        "full_hit_lower_bound": full_lower if n else None,
        "alpha": alpha,
        "comparison_count": comparison_count,
        "minimum_improvement": minimum_improvement,
        "resource_ratio": resource_ratio if math.isfinite(resource_ratio) else None,
        "approved": bool(approved),
        "reason": "通過獨立配對驗證"
        if approved
        else "樣本、保守改善、全中率或資源門檻未通過；維持正式基準",
    }


def _interval(data, indexes):
    indexes = list(indexes)
    return {
        "from": data[indexes[0]]["draw_term"] if indexes else None,
        "to": data[indexes[-1]]["draw_term"] if indexes else None,
        "count": len(indexes),
    }


def _summary(totals):
    return [
        {"model": name, "stars": stars, "samples": count, "mean_hits": total / count}
        for (name, stars), (count, total) in totals.items()
        if count
    ]


def _pairs(records, profile_id, method, stars, indexes):
    pairs = []
    for index in indexes:
        baseline = records.get((index, "fixed", "shrink", stars))
        candidate = records.get((index, profile_id, method, stars))
        if baseline is not None and candidate is not None:
            pairs.append(
                {
                    "index": index,
                    "champion_hits": baseline,
                    "challenger_hits": candidate,
                }
            )
    return pairs


def _select_before_block(
    data,
    records,
    profiles,
    index,
    outer_start,
    selection_count,
    selection_size,
    validation_size,
    previous_champions,
    costs,
    minimum_improvement,
):
    # The last closed outer block is reserved as independent shadow confirmation
    # and is excluded from the current inner search/validation period.
    shadow_indexes = list(range(max(outer_start, index - validation_size), index))
    inner_end = shadow_indexes[0] if shadow_indexes else index
    validation_indexes = list(range(max(100, inner_end - validation_size), inner_end))
    selection_end = validation_indexes[0] if validation_indexes else inner_end
    selection_indexes = list(
        range(max(100, selection_end - selection_size), selection_end)
    )
    train_end = selection_indexes[0] if selection_indexes else selection_end
    comparisons = max(1, (len(profiles) - 1) * len(METHODS) * len(STARS))
    output = {
        "effective_from": data[index]["draw_term"],
        "selection_count": selection_count,
        "splits": {
            "train": _interval(data, range(train_end)),
            "selection": _interval(data, selection_indexes),
            "validation": _interval(data, validation_indexes),
            "shadow_holdout": _interval(data, shadow_indexes),
        },
        "by_star": {},
    }
    for stars in STARS:
        choices = []
        for profile in profiles[1:]:
            for method in METHODS:
                pairs = _pairs(records, profile.id, method, stars, selection_indexes)
                if len(pairs) < selection_size:
                    continue
                improvements = np.asarray(
                    [r["challenger_hits"] - r["champion_hits"] for r in pairs]
                )
                # Selection reward includes observed stability; confirmation below
                # still uses an untouched, conservative paired test.
                quality = float(improvements.mean() - 0.1 * improvements.std())
                choices.append((quality, profile.id, method))
        choice = (
            sorted(choices, key=lambda x: (-x[0], x[1], x[2]))[0]
            if choices
            else (0.0, "fixed", "shrink")
        )
        _, profile_id, method = choice
        base_cost = costs.get("fixed", 0.0)
        ratio = costs.get(profile_id, 0.0) / base_cost if base_cost > 0 else 1.0
        inner = paired_gate(
            _pairs(records, profile_id, method, stars, validation_indexes),
            stars,
            comparison_count=comparisons,
            attempt=selection_count,
            minimum_improvement=minimum_improvement,
            resource_ratio=ratio,
        )
        shadow = paired_gate(
            _pairs(records, profile_id, method, stars, shadow_indexes),
            stars,
            comparison_count=comparisons,
            attempt=selection_count,
            minimum_improvement=minimum_improvement,
            resource_ratio=ratio,
        )
        current = copy.deepcopy(previous_champions[str(stars)])
        rollback_pairs = _pairs(
            records, current["profile"], current["method"], stars, shadow_indexes
        )
        rollback = paired_gate(
            [
                {
                    **row,
                    "champion_hits": row["challenger_hits"],
                    "challenger_hits": row["champion_hits"],
                }
                for row in rollback_pairs
            ],
            stars,
            comparison_count=comparisons,
            attempt=selection_count,
            minimum_improvement=minimum_improvement,
            resource_ratio=base_cost / costs.get(current["profile"], base_cost)
            if costs.get(current["profile"], base_cost) > 0
            else 1.0,
        )
        rolled_back = current["profile"] != "fixed" and rollback["approved"]
        if rolled_back:
            current = {
                "profile": "fixed",
                "method": "shrink",
                "validated": True,
                "effective_from": data[index]["draw_term"],
            }
        promoted = (
            not rolled_back
            and profile_id != "fixed"
            and inner["approved"]
            and shadow["approved"]
        )
        if promoted:
            current = {
                "profile": profile_id,
                "method": method,
                "validated": True,
                "effective_from": data[index]["draw_term"],
            }
        output["by_star"][str(stars)] = {
            "candidate": {"profile": profile_id, "method": method},
            "champion": current,
            "promoted": promoted,
            "rolled_back": rolled_back,
            "rollback_validation": rollback,
            "inner_validation": inner,
            "shadow_validation": shadow,
            "reason": "已驗證候選在先前獨立影子期數持續退化，僅未來期數回退固定基準"
            if rolled_back
            else (
                "候選通過先前獨立影子期數，僅變更其後期數"
                if promoted
                else inner["reason"]
            ),
        }
    return output


def nested_walk_forward(
    draws: list[dict],
    count=100,
    start_term=None,
    end_term=None,
    threads=2,
    window=1000,
    progress: Callable | None = None,
    model_names=None,
    profiles=None,
    outer_block_size=100,
    inner_selection_size=30,
    inner_validation_size=100,
    minimum_improvement=0.03,
):
    """Run actual bounded nested evaluation, keeping fixed and shadow ledgers apart.

    The final independent outer block is not used to choose its candidate. Only
    that preselected candidate can be proposed for later live validation. Raw
    highest-scoring outer results never authorize retroactive parameter changes.
    """
    from taiwan_lottery.ml import bingo_models as model_module

    ModelSuite = model_module.ModelSuite
    if not 1 <= count <= 1000 or not 100 <= window <= 10000:
        raise ValueError("回測期數或訓練視窗超出範圍")
    if (
        outer_block_size < 100
        or inner_selection_size < 30
        or inner_validation_size < 100
    ):
        raise ValueError("巢狀驗證至少需要 30 期選擇、100 期驗證及 100 期外層區塊")
    data = _validate_draws(draws)
    feature_cache = (
        model_module.FeatureHistory([item["numbers"] for item in data])
        if hasattr(model_module, "FeatureHistory")
        else None
    )
    eligible = [
        i
        for i, row in enumerate(data)
        if (start_term is None or int(row["draw_term"]) >= int(start_term))
        and (end_term is None or int(row["draw_term"]) <= int(end_term))
    ]
    if start_term is None and end_term is None:
        eligible = eligible[-count:]
    elif len(eligible) > count:
        raise ValueError("自訂區間超過設定回測上限")
    if (
        not eligible
        or eligible[0] < 100
        or (start_term is None and end_term is None and len(eligible) < count)
    ):
        raise ValueError("資料不足：外層目標之前至少需要 100 期訓練資料")
    requested_profiles = (
        default_profiles(window)
        if profiles is None
        else tuple(
            p if isinstance(p, CandidateProfile) else CandidateProfile(**p)
            for p in profiles
        )
    )
    if (
        not 2 <= len(requested_profiles) <= 4
        or requested_profiles[0].id != "fixed"
        or len({p.id for p in requested_profiles}) != len(requested_profiles)
    ):
        raise ValueError("需固定基準及 1～3 個不重複候選，禁止無界參數搜尋")
    # Baseline parameters are immutable even when callers customize challengers.
    if requested_profiles[0] != CandidateProfile("fixed", window, window):
        raise ValueError("正式固定基準不得透過候選設定修改")
    by_id = {p.id: p for p in requested_profiles}
    outer_start = eligible[0]
    begin = max(100, outer_start - inner_selection_size - inner_validation_size)
    outer_indexes = set(eligible)
    suites, trained_through, fit_counts, costs = (
        {},
        {},
        defaultdict(int),
        defaultdict(float),
    )
    totals = {p.id: defaultdict(lambda: [0, 0]) for p in requested_profiles}
    records, outcomes, shadow_outcomes, folds, audits, warnings = {}, [], [], [], [], []
    champions = {
        str(s): {"profile": "fixed", "method": "shrink", "validated": False}
        for s in STARS
    }
    last_selection = None
    started, cpu_started = time.perf_counter(), time.process_time()
    for index in range(begin, eligible[-1] + 1):
        outer_step = len(folds)
        is_outer = index in outer_indexes
        if index == outer_start:
            # Keep the fixed baseline identical to a stand-alone stage-one run.
            suites.clear()
            totals = {p.id: defaultdict(lambda: [0, 0]) for p in requested_profiles}
        if is_outer and (not audits or outer_step % outer_block_size == 0):
            last_selection = _select_before_block(
                data,
                records,
                requested_profiles,
                index,
                outer_start,
                len(audits) + 1,
                inner_selection_size,
                inner_validation_size,
                champions,
                costs,
                minimum_improvement,
            )
            audits.append(last_selection)
            champions = {
                key: copy.deepcopy(value["champion"])
                for key, value in last_selection["by_star"].items()
            }
        profile_outputs, profile_audits = {}, {}
        for profile in requested_profiles:
            history_start = max(0, index - profile.train_window)
            history = [row["numbers"] for row in data[history_start:index]]
            anchor = outer_start if index >= outer_start else begin
            before_cpu = time.process_time()
            if (
                profile.id not in suites
                or (index - anchor) % profile.retrain_every == 0
            ):
                kwargs = {"threads": threads, "window": profile.train_window}
                if model_names is not None:
                    kwargs["enabled"] = model_names

                def fit_progress(name, done, total, status):
                    if progress:
                        progress(
                            f"nested/{profile.id}/{name} {done}/{total}",
                            outer_step,
                            len(eligible),
                            "training",
                        )

                suites[profile.id] = ModelSuite(history, **kwargs).fit(
                    progress=fit_progress
                )
                trained_through[profile.id] = index - 1
                fit_counts[profile.id] += 1
            suite = suites[profile.id]
            prepared_window = min(profile.context_window, profile.train_window)
            matrix = (
                feature_cache.matrix_before(index, prepared_window)
                if feature_cache
                else None
            )
            row = feature_cache.row(index, prepared_window) if feature_cache else None
            score_map = profile_scores(
                suite,
                history,
                profile,
                baseline=profile.id == "fixed",
                matrix=matrix,
                row=row,
            )
            if model_names is not None:
                score_map = {
                    name: vector
                    for name, vector in score_map.items()
                    if name in model_names
                }
            if not score_map:
                raise ValueError(f"候選 {profile.id} 沒有可用模型")
            past = _summary(totals[profile.id])
            fused = fuse_many(score_map, past)
            costs[profile.id] += time.process_time() - before_cpu
            profile_outputs[profile.id] = {"models": score_map, "fusions": fused}
            context_start = max(history_start, index - profile.context_window)
            profile_audits[profile.id] = {
                "profile": profile.payload(),
                "training_cutoff": data[trained_through[profile.id]]["draw_term"],
                "feature_cutoff": data[index - 1]["draw_term"],
                "training_history": _interval(
                    data,
                    range(
                        max(0, trained_through[profile.id] + 1 - profile.train_window),
                        trained_through[profile.id] + 1,
                    ),
                ),
                "feature_history": _interval(data, range(context_start, index)),
                "model_version": getattr(
                    suite,
                    "history_digest",
                    f"{profile.id}:{data[trained_through[profile.id]]['draw_term']}",
                ),
                "feature_version": FEATURE_VERSION,
                "model_scores": copy.deepcopy(score_map),
                "fusion_scores": {
                    key: {method: row["scores"] for method, row in methods.items()}
                    for key, methods in fused.items()
                },
                "fusion_weights": {
                    key: {method: row["weights"] for method, row in methods.items()}
                    for key, methods in fused.items()
                },
            }
        predictions = {}
        base = profile_outputs["fixed"]
        for name, vector in base["models"].items():
            ranking = _rank(normalize_scores(vector))
            predictions[name] = {str(s): ranking[:s] for s in STARS}
        for method in METHODS:
            predictions[method] = {
                str(s): base["fusions"][str(s)][method]["ranking"][:s] for s in STARS
            }
        predictions["adaptive"] = {}
        for stars in STARS:
            choice = champions[str(stars)]
            predictions["adaptive"][str(stars)] = profile_outputs[choice["profile"]][
                "fusions"
            ][str(stars)][choice["method"]]["ranking"][:stars]
        # The immutable decision hash is computed before this target's answer.
        frozen = {
            "target_term": data[index]["draw_term"],
            "trained_through": data[trained_through["fixed"]]["draw_term"],
            "features_through": data[index - 1]["draw_term"],
            "history_from": data[max(0, index - window)]["draw_term"],
            "history_to": data[index - 1]["draw_term"],
            "history_count": min(index, window),
            "feature_version": FEATURE_VERSION,
            "strategy_version": NESTED_VERSION,
            "champions": copy.deepcopy(champions),
            "predictions": copy.deepcopy(predictions),
            "profiles": profile_audits,
            "selection_count": len(audits),
            "is_live": False,
        }
        frozen["decision_hash"] = hashlib.sha256(
            json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if is_outer:
            folds.append(frozen)
        # Only here is the target answer revealed; no selection/scoring above it.
        actual = set(data[index]["numbers"])
        for profile in requested_profiles:
            output = profile_outputs[profile.id]
            for name, vector in output["models"].items():
                ranking = _rank(normalize_scores(vector))
                for stars in STARS:
                    hits = len(actual.intersection(ranking[:stars]))
                    totals[profile.id][(name, stars)][0] += 1
                    totals[profile.id][(name, stars)][1] += hits
            for stars in STARS:
                for method in METHODS:
                    candidates = output["fusions"][str(stars)][method]["ranking"][
                        :stars
                    ]
                    hits = len(actual.intersection(candidates))
                    records[(index, profile.id, method, stars)] = hits
                    if is_outer and profile.id != "fixed":
                        shadow_outcomes.append(
                            {
                                "target_term": data[index]["draw_term"],
                                "model": f"{profile.id}/{method}",
                                "profile_id": profile.id,
                                "method": method,
                                "stars": stars,
                                "hit_count": hits,
                                "predicted_numbers": list(candidates),
                                "actual_numbers": sorted(actual),
                                "draw_datetime": str(
                                    data[index].get("draw_datetime", "")
                                ),
                                "updated_at": str(data[index].get("draw_datetime", "")),
                                "is_live": False,
                            }
                        )
            history = [
                row["numbers"]
                for row in data[max(0, index - profile.train_window) : index]
            ]
            before_cpu = time.process_time()
            if feature_cache and hasattr(suites[profile.id], "scores_prepared"):
                suites[profile.id].update(
                    history,
                    data[index]["numbers"],
                    feature_row=feature_cache.row(index, profile.train_window),
                )
            else:
                suites[profile.id].update(history, data[index]["numbers"])
            costs[profile.id] += time.process_time() - before_cpu
        if is_outer:
            for name, by_star in predictions.items():
                for key, candidates in by_star.items():
                    outcomes.append(
                        {
                            "target_term": data[index]["draw_term"],
                            "model": name,
                            "stars": int(key),
                            "predicted_numbers": list(candidates),
                            "actual_numbers": sorted(actual),
                            "hit_count": len(actual.intersection(candidates)),
                            "draw_datetime": str(data[index].get("draw_datetime", "")),
                            "updated_at": str(data[index].get("draw_datetime", "")),
                            "is_live": False,
                        }
                    )
            if progress:
                progress("nested", len(folds), len(eligible), "running")
    baseline_rows = [row for row in outcomes if row["model"] != "adaptive"]
    baseline_boards = live_rankings(baseline_rows)
    boards = live_rankings(outcomes)
    comparison_count = max(1, (len(requested_profiles) - 1) * len(METHODS) * len(STARS))
    final_start = max(
        outer_start, eligible[-1] + 1 - min(outer_block_size, len(eligible))
    )
    # Confirmation belongs to the actual last block's preselected profile. If
    # a custom count ends with a short block it cannot borrow earlier outcomes.
    final_start = int(
        next(
            i for i in eligible if data[i]["draw_term"] == audits[-1]["effective_from"]
        )
    )
    final_indexes = [i for i in eligible if i >= final_start]
    proposals, confirmation = {}, {}
    for stars in STARS:
        chosen = audits[-1]["by_star"][str(stars)]["candidate"]
        profile = by_id[chosen["profile"]]
        pairs = _pairs(records, profile.id, chosen["method"], stars, final_indexes)
        base_cost = costs.get("fixed", 0.0)
        ratio = costs.get(profile.id, 0.0) / base_cost if base_cost else 1.0
        gate = paired_gate(
            pairs,
            stars,
            comparison_count=comparison_count,
            attempt=len(audits),
            minimum_improvement=minimum_improvement,
            resource_ratio=ratio,
        )
        confirmation[str(stars)] = gate
        audit_pairs = []
        for pair in pairs:
            i = pair["index"]
            fold = folds[eligible.index(i)]
            metadata = fold["profiles"][profile.id]
            audit_pairs.append(
                {
                    "target_term": data[i]["draw_term"],
                    "champion_hits": pair["champion_hits"],
                    "challenger_hits": pair["challenger_hits"],
                    "source_term": data[i - 1]["draw_term"],
                    "feature_cutoff": metadata["feature_cutoff"],
                    "training_cutoff": metadata["training_cutoff"],
                    "target_time": str(data[i].get("draw_datetime", "")),
                    "is_live": False,
                }
            )
        proposals[str(stars)] = {
            "profile": profile.payload(chosen["method"]),
            "baseline_profile_id": "fixed",
            "paired": audit_pairs,
            "selection_end_term": audits[-1]["splits"]["validation"]["to"],
            "validation_start_term": data[final_indexes[0]]["draw_term"],
            "validation_end_term": data[final_indexes[-1]]["draw_term"],
            "resource_ratio": ratio,
            "comparison_count": comparison_count,
            "selection_count": len(audits),
            "inner_validation": audits[-1]["by_star"][str(stars)]["inner_validation"],
        }
    allowed = all(
        confirmation[str(s)]["approved"]
        and audits[-1]["by_star"][str(s)]["inner_validation"]["approved"]
        and proposals[str(s)]["profile"]["id"] != "fixed"
        for s in STARS
    )
    decision = {
        "method": "shrink",
        "validated": bool(allowed),
        "allowed_for_live": bool(allowed),
        "profiles_by_star": {key: value["profile"] for key, value in proposals.items()},
        "independent_validation_samples": {
            key: value["samples"] for key, value in confirmation.items()
        },
        "proposals": proposals,
        "selection_end_term": audits[-1]["splits"]["validation"]["to"],
        "validation_start_term": data[final_indexes[0]]["draw_term"],
        "validation_end_term": data[final_indexes[-1]]["draw_term"],
        "outer_start_term": data[eligible[0]]["draw_term"],
        "outer_end_term": data[eligible[-1]]["draw_term"],
        "resource_ratio": max(
            value["resource_ratio"] for value in confirmation.values()
        ),
        "comparison_count": comparison_count,
        "mean_deltas_by_star": {
            key: value["delta"] for key, value in confirmation.items()
        },
        "ci_lows_by_star": {
            key: value["lower_bound"] for key, value in confirmation.items()
        },
        "full_hit_deltas_by_star": {
            key: value["full_hit_delta"] for key, value in confirmation.items()
        },
        "reason": "預選候選通過最終獨立外層確認，僅可供未來策略重新稽核"
        if allowed
        else "獨立樣本外證據不足；固定 Shrink 維持正式策略，候選僅為影子比較",
    }
    return {
        "mode": "nested_walk_forward",
        "is_live": False,
        "count": len(folds),
        "train_size": min(window, outer_start),
        "window": window,
        "retrain_every": 100,
        "folds": folds,
        "results": outcomes,
        "leaderboards": boards,
        "method_comparison": {
            str(s): [
                row
                for row in baseline_boards[str(s)]["cumulative"]
                if row["model"] in METHODS
            ]
            for s in STARS
        },
        "baseline": {"leaderboards": baseline_boards, "results": baseline_rows},
        "shadow": {
            "leaderboards": live_rankings(shadow_outcomes),
            "results": shadow_outcomes,
        },
        "nested": {
            "inner_selection_size": inner_selection_size,
            "inner_validation_size": inner_validation_size,
            "outer_block_size": outer_block_size,
            "profile_budget": len(requested_profiles),
            "profiles": [profile.payload() for profile in requested_profiles],
            "selection_count": len(audits),
            "selection_audit": audits,
            "confirmation": confirmation,
            "split_policy": "內層選擇與驗證排除最近已封閉外層影子區塊；最後外層區塊候選在開獎前固定，絕不以外層最佳名次回選",
        },
        "model_statuses": suites["fixed"].statuses(),
        "warnings": warnings,
        "selected_method": "shrink",
        "selection_reason": decision["reason"],
        "decision": decision,
        "score_kind": "相對排名分數，非開獎機率",
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "profile_cpu_seconds": dict(costs),
            "fit_counts": dict(fit_counts),
        },
        "methodology": "逐期先凍結完整分數與候選，再揭露結果；內外層時間分離；歷史模擬不得加入正式實戰",
    }
