"""Shared, metered context scoring from causal immutable model artifacts."""

from __future__ import annotations

import copy
import hashlib
import time
import numpy as np

from taiwan_lottery.ml.bingo_models import features

FEATURE_VERSION = "bingo-features-causal-v1"


def _score(suite, draws, window):
    recent = [d["numbers"] for d in draws[-window:]]
    # The fitted artifact stays immutable. A context length is an explicit
    # inference parameter, independently audited from the training window.
    scorer = copy.copy(suite)
    scorer.window = window
    started, cpu = time.perf_counter(), time.process_time()
    scores = scorer.scores(recent)
    row = features(recent)
    return dict(
        scores=scores,
        scoring_cost_seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        window=window,
        history_count=len(recent),
        history_start=draws[-len(recent)]["draw_term"],
        history_end=draws[-1]["draw_term"],
        input_hash=hashlib.sha256(
            np.asarray(recent, dtype=np.uint8).tobytes()
        ).hexdigest(),
        feature_hash=hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest(),
        feature_cutoff=draws[-1]["draw_term"],
        feature_version=FEATURE_VERSION,
    )


def context_scores(suite, draws, plan=None, *, directory=None, threads=2):
    if not draws:
        raise ValueError("沒有可評分的已知資料")
    plan = plan or {}
    fixed = int(suite.window)
    champions = plan.get("champions", {})
    required = {fixed}
    required.update(int(w) for models in champions.values() for w in models.values())
    for entry in plan.get("by_model", {}).values():
        required.update(int(w) for w in entry.get("rollback_windows", []))
    requested = required | {int(w) for w in plan.get("shadow_windows", [])}
    for entry in plan.get("by_model", {}).values():
        requested.update(int(w) for w in entry.get("required_windows", []))
    if any(not 1 <= w <= 10000 for w in requested) or len(requested) > 64:
        raise ValueError("歷史視窗搜尋超出受限預算")
    contexts = {}
    extra_started = None
    skipped = []
    for window in [fixed] + sorted(requested - {fixed}):
        if extra_started is None and window != fixed:
            extra_started = time.perf_counter()
        if (
            window not in required
            and extra_started
            and time.perf_counter() - extra_started > 0.25
        ):
            skipped.append(window)
            continue
        contexts[str(window)] = _score(suite, draws, window)
    fixed_scores = contexts[str(fixed)]["scores"]
    by_star, effective, star_contexts, profile_status, prior_scores = {}, {}, {}, {}, {}
    profile_cache = {}
    for stars in (3, 4, 5):
        key = str(stars)
        star_contexts[key] = dict(contexts)
        profile = plan.get("profiles_by_star", {}).get(key, {})
        pending = False
        if (
            profile
            and profile.get("id") not in ("fixed", "fixed-v1")
            and directory is not None
        ):
            from taiwan_lottery.ml.bingo_profile_artifacts import (
                score_validated_profile,
                profile_key,
            )

            for window, context in contexts.items():
                cache = (profile_key(profile), window)
                if cache not in profile_cache:
                    profile_cache[cache] = score_validated_profile(
                        profile, draws, directory, threads, context_window=int(window)
                    )
                result = profile_cache[cache]
                if result["status"] != "ready":
                    pending = True
                    profile_status[key] = {
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "profile": profile,
                    }
                    break
                star_contexts[key][window] = {
                    **context,
                    **result["metadata"],
                    "scores": result["scores"],
                    "model_version": result["version"],
                    "profile": profile,
                }
            if not pending:
                profile_status[key] = {"status": "ready", "profile": profile}
        if pending:
            star_contexts[key] = dict(contexts)
        selected = {} if pending else champions.get(key, {})
        effective[key] = {
            name: int(selected.get(name, fixed))
            if str(int(selected.get(name, fixed))) in star_contexts[key]
            and name
            in star_contexts[key][str(int(selected.get(name, fixed)))]["scores"]
            else fixed
            for name in fixed_scores
        }
        by_star[key] = {
            name: star_contexts[key][str(effective[key][name])]["scores"].get(
                name, vector
            )
            for name, vector in fixed_scores.items()
        }
        previous = plan.get("previous_profiles_by_star", {}).get(key)
        if previous:
            if previous.get("id") in ("fixed", "fixed-v1"):
                prior_scores[key] = dict(
                    scores=fixed_scores,
                    profile=previous,
                    scoring_cost_seconds=contexts[str(fixed)]["scoring_cost_seconds"],
                )
            elif directory:
                from taiwan_lottery.ml.bingo_profile_artifacts import (
                    score_validated_profile,
                )

                result = score_validated_profile(previous, draws, directory, threads)
                if result["status"] == "ready":
                    prior_scores[key] = dict(
                        scores=result["scores"],
                        profile=previous,
                        scoring_cost_seconds=result["metadata"]["scoring_cost_seconds"],
                    )
    return dict(
        scores=fixed_scores,
        by_star_scores=by_star,
        contexts=contexts,
        contexts_by_star=star_contexts,
        fixed_window=fixed,
        effective_champions=effective,
        profile_status=profile_status,
        previous_policy_scores=prior_scores,
        skipped_shadow_windows=skipped,
        feature_version=FEATURE_VERSION,
    )


def context_audit(result, draws, model, stars, plan=None, *, window=None):
    plan = plan or {}
    fixed = result.get("fixed_window", 1000)
    requested = int(
        window
        if window is not None
        else plan.get("champions", {}).get(str(stars), {}).get(model, fixed)
    )
    selected = (
        requested
        if window is not None
        else result.get("effective_champions", {})
        .get(str(stars), {})
        .get(model, requested)
    )
    context = (
        result.get("contexts_by_star", {})
        .get(str(stars), result.get("contexts", {}))
        .get(str(selected), {})
    )
    statuses = {r["name"]: r for r in result.get("models", [])}
    record = statuses.get(model, {})
    count = context.get("history_count", min(selected, len(draws)))
    model_version = context.get(
        "model_version",
        record.get("training_version", result.get("version", "baseline")),
    )
    audit = dict(
        training_cutoff=context.get(
            "training_cutoff",
            record.get(
                "training_source_term",
                result.get("training_cutoff", draws[-1]["draw_term"]),
            ),
        ),
        training_history_start=context.get(
            "training_history_start",
            record.get("training_history_start", result.get("training_history_start")),
        ),
        training_history_count=context.get(
            "training_history_count",
            record.get(
                "training_history_count", result.get("training_samples", len(draws))
            ),
        ),
        feature_cutoff=draws[-1]["draw_term"],
        history_start=context.get("history_start", draws[-count]["draw_term"]),
        history_end=draws[-1]["draw_term"],
        history_count=count,
        window=selected,
        requested_window=requested,
        effective_feature_count=min(
            count,
            16
            if model == "transformer"
            else 200
            if model in ("frequency", "river", "lightgbm", "catboost", "dqn")
            else 10000,
        ),
        model_version=model_version,
        feature_version=result.get("feature_version", FEATURE_VERSION),
        vote_strategy=plan.get("vote_methods", {}).get(str(stars), "shrink"),
        policy_version=plan.get("version", "fixed-baseline"),
        feature_hash=context.get("feature_hash"),
        input_hash=context.get("input_hash"),
        scoring_cost_seconds=context.get("scoring_cost_seconds"),
        source_term=draws[-1]["draw_term"],
    )
    audit["batch_training_cutoff"] = record.get(
        "batch_training_cutoff",
        record.get("training_source_term", result.get("training_cutoff")),
    )
    if model in ("frequency", "cdm", "ensemble"):
        audit["training_cutoff"] = draws[-1]["draw_term"]
        audit["training_history_start"] = audit["history_start"]
        audit["training_history_count"] = audit["history_count"]
        audit["update_action"] = "recomputed_from_known_context"
    elif model == "river":
        audit["update_action"] = "online_update_after_settlement"
    else:
        audit["update_action"] = "scheduled_batch_parameters_with_current_features"
    if audit["training_history_start"] is None:
        audit.pop("training_history_start")
    return audit
