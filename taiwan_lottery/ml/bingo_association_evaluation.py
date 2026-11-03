"""Causal A/B/C replay with fresh historical models, isolated from live state."""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json

from taiwan_lottery.bingo_clock import next_draw
from taiwan_lottery.ml.bingo_associations import association_candidates
from taiwan_lottery.ml.bingo_evaluation import (
    _validate_draws,
    live_rankings,
    walk_forward,
)


def association_walk_forward(
    draws,
    count=100,
    start_term=None,
    end_term=None,
    threads=2,
    window=1000,
    progress=None,
    model_names=None,
):
    if not 1 <= count <= 1000:
        raise ValueError("關聯比較每次最多1000期")
    data = _validate_draws(draws)

    def base_progress(model, done, total, state):
        if progress:
            progress(model, done, 2 * total, state)

    # These immutable folds come from newly fitted causal prefixes. No current
    # production artifact is ever used to score an earlier target.
    base = walk_forward(
        data,
        count=count,
        start_term=start_term,
        end_term=end_term,
        threads=threads,
        window=window,
        progress=base_progress,
        model_names=model_names,
    )
    index_by_term = {r["draw_term"]: i for i, r in enumerate(data)}
    results, folds = [], []
    for step, previous in enumerate(base["folds"]):
        target = previous["target_term"]
        index = index_by_term[target]
        history = data[max(0, index - min(1000, window)) : index]
        stamp = history[-1]["draw_datetime"]
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp)
        expected, target_at = next_draw(history[-1]["draw_term"], stamp)
        if expected != target:
            raise ValueError("關聯回測的目標必須是來源期之後的真正下一期")
        calculated = association_candidates(
            history,
            {str(k): previous["model_scores"] for k in (3, 4, 5)},
            model_weights_by_star={
                str(k): previous["fusion_weights"][str(k)]["shrink"] for k in (3, 4, 5)
            },
            target_date=target_at.date(),
        )
        decision = dict(
            target_term=target,
            source_term=history[-1]["draw_term"],
            training_cutoff=history[-1]["draw_term"],
            rule_training_cutoff=history[-1]["draw_term"],
            batch_training_cutoff=previous["trained_through"],
            model_training_cutoffs={
                name: previous["features_through"]
                if name in ("frequency", "cdm", "river", "ensemble")
                else previous["trained_through"]
                for name in previous["model_scores"]
            },
            feature_cutoff=previous["features_through"],
            model_version=previous["model_version"],
            feature_version=calculated["feature_version"],
            association_version=calculated["version"],
            history=calculated["history"],
            by_star=copy.deepcopy(calculated["by_star"]),
            rules=copy.deepcopy(calculated["rules"]),
            window_selection=calculated["window_selection"],
            is_live=False,
        )
        decision["decision_hash"] = hashlib.sha256(
            json.dumps(
                decision,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        # Reveal the current target only after all nine scores and explanations
        # have been frozen. Subsequent rules may use this now-known result.
        actual = data[index]["numbers"]
        for star, candidates in decision["by_star"].items():
            for label, candidate in candidates.items():
                results.append(
                    dict(
                        target_term=target,
                        model="association_" + label.lower(),
                        stars=int(star),
                        predicted_numbers=candidate["numbers"],
                        actual_numbers=actual,
                        hit_count=len(set(candidate["numbers"]) & set(actual)),
                        draw_datetime=str(data[index]["draw_datetime"]),
                        is_live=False,
                    )
                )
        folds.append(dict(decision, resources=calculated["resources"]))
        if progress:
            progress(
                "association_walk_forward",
                len(base["folds"]) + step + 1,
                2 * len(base["folds"]),
                "running",
            )
    comparison = [{**r, "is_live": False} for r in base["results"]] + results
    return dict(
        mode="association_walk_forward",
        is_live=False,
        count=len(folds),
        folds=folds,
        results=comparison,
        leaderboards=live_rankings(comparison),
        model_statuses=base["model_statuses"],
        baseline={k: v for k, v in base.items() if k not in ("folds", "results")},
        selected_method="shrink",
        selection_reason="ABC持續影子驗證，不改寫正式策略",
        decision=dict(
            allowed_for_live=False, reason="歷史關聯比較不計入正式實戰；未證明策略優勢"
        ),
        methodology="ABC與原八模型及四投票比較相同期數；僅用當折已知前綴、凍結模型分數及過去已結算權重",
    )
