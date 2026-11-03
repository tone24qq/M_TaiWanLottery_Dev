"""Walk-forward replay of the actual online window controller, separate from live."""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta
import copy
import hashlib
import json

from taiwan_lottery.bingo_clock import to_taipei
from taiwan_lottery.ml.bingo_adaptive import AdaptiveController
from taiwan_lottery.ml.bingo_contexts import context_scores, context_audit
from taiwan_lottery.ml.bingo_evaluation import (
    _validate_draws,
    _rank,
    fuse,
    live_rankings,
)
from taiwan_lottery.ml.bingo_models import ModelSuite


def adaptive_walk_forward(
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
        raise ValueError("智慧視窗每次比較1～1000期")
    data = _validate_draws(draws)
    eligible = [
        i
        for i, d in enumerate(data)
        if (start_term is None or int(d["draw_term"]) >= int(start_term))
        and (end_term is None or int(d["draw_term"]) <= int(end_term))
    ]
    if start_term is None and end_term is None:
        eligible = eligible[-count:]
    if not eligible or eligible[0] < 100 or len(eligible) > count:
        raise ValueError("智慧視窗需足夠歷史及目標前100期資料")
    control = AdaptiveController(baseline_window=window, simulation=True)
    suite = None
    fit_index = None
    last_index = None
    folds = []
    results = []
    shadow = []
    events = []
    performance = []
    fixed_performance = []
    for step, index in enumerate(eligible):
        snapshot = data[max(0, index - window) : index]
        history = [d["numbers"] for d in snapshot]
        if (
            suite is None
            or step % 100 == 0
            or (last_index is not None and index != last_index + 1)
        ):
            suite = ModelSuite(
                history, threads=threads, window=window, enabled=model_names
            ).fit()
            fit_index = index - 1
        names = [name for name in suite.enabled if name in suite.models]
        plan = control.plan(
            data[index - 1]["draw_term"], history=len(history), model_names=names
        )
        scored = context_scores(suite, snapshot, plan)
        scored.update(
            version=suite.history_digest,
            models=suite.statuses(),
            training_cutoff=data[fit_index]["draw_term"],
        )
        baseline = {}
        predictions = {}
        rows = []
        source_time = to_taipei(
            datetime.fromisoformat(str(data[index - 1]["draw_datetime"]))
        )
        target_time = to_taipei(
            datetime.fromisoformat(str(data[index]["draw_datetime"]))
        )
        for stars in (3, 4, 5):
            key = str(stars)
            baseline[key] = fuse(scored["scores"], "shrink", stars, fixed_performance)
            chosen = fuse(scored["by_star_scores"][key], "shrink", stars, performance)
            predictions[key] = dict(
                numbers=chosen["ranking"][:stars],
                scores=chosen["scores"],
                weights=chosen["weights"],
            )
            for name, vector in scored["by_star_scores"][key].items():
                audit = context_audit(scored, snapshot, name, stars, plan)
                rows.append(
                    dict(
                        model=name,
                        stars=stars,
                        target_term=data[index]["draw_term"],
                        source_term=data[index - 1]["draw_term"],
                        source_time=source_time.isoformat(),
                        target_time=target_time.isoformat(),
                        created_at=(source_time + timedelta(seconds=1)).isoformat(),
                        numbers=_rank(vector)[:stars],
                        scores=vector,
                        window=audit["window"],
                        metadata=audit,
                        is_live=False,
                    )
                )
            for w, context in scored["contexts"].items():
                for name, vector in context["scores"].items():
                    if int(w) == plan["champions"][key][name]:
                        continue
                    rows.append(
                        dict(
                            model=name,
                            stars=stars,
                            target_term=data[index]["draw_term"],
                            source_term=data[index - 1]["draw_term"],
                            source_time=source_time.isoformat(),
                            target_time=target_time.isoformat(),
                            created_at=(source_time + timedelta(seconds=1)).isoformat(),
                            numbers=_rank(vector)[:stars],
                            scores=vector,
                            window=int(w),
                            metadata=context_audit(
                                scored, snapshot, name, stars, plan, window=int(w)
                            ),
                            is_shadow=True,
                            is_live=False,
                        )
                    )
        frozen = dict(
            target_term=data[index]["draw_term"],
            training_cutoff=data[fit_index]["draw_term"],
            feature_cutoff=data[index - 1]["draw_term"],
            history_start=snapshot[0]["draw_term"],
            history_end=snapshot[-1]["draw_term"],
            history_count=len(snapshot),
            feature_version=scored["feature_version"],
            model_version=suite.history_digest,
            policy_version=plan["version"],
            champion_windows=plan["champions"],
            predictions=predictions,
            model_decisions=copy.deepcopy(rows),
            is_live=False,
        )
        # Resource telemetry is preserved but is not part of score reproducibility.
        deterministic = copy.deepcopy(frozen)
        for row in deterministic["model_decisions"]:
            row["metadata"].pop("scoring_cost_seconds", None)
        frozen["decision_hash"] = hashlib.sha256(
            json.dumps(deterministic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        actual = set(data[index]["numbers"])
        for row in rows:
            row.update(
                hit_count=len(actual & set(row["numbers"])),
                actual_numbers=sorted(actual),
            )
        live = [r for r in rows if not r.get("is_shadow")]
        candidate = [r for r in rows if r.get("is_shadow")]
        update = control.observe(data[index]["draw_term"], live, candidate)
        events.extend(update.get("events", []))
        shadow.extend(rows)
        for stars in (3, 4, 5):
            key = str(stars)
            for name, picks in [
                ("fixed", baseline[key]["ranking"][:stars]),
                ("adaptive", predictions[key]["numbers"]),
            ]:
                results.append(
                    dict(
                        model=name,
                        stars=stars,
                        target_term=data[index]["draw_term"],
                        hit_count=len(actual & set(picks)),
                        predicted_numbers=picks,
                        actual_numbers=sorted(actual),
                        draw_datetime=str(data[index]["draw_datetime"]),
                        is_live=False,
                    )
                )
        for stars in (3, 4, 5):
            for name, vector in scored["scores"].items():
                fixed_performance.append(
                    dict(
                        model=name,
                        stars=stars,
                        hit_count=len(actual & set(_rank(vector)[:stars])),
                    )
                )
        performance.extend(
            [
                {"model": r["model"], "stars": r["stars"], "hit_count": r["hit_count"]}
                for r in live
            ]
        )
        # Online targets are learned after all this fold's decisions are frozen.
        suite.update(history, data[index]["numbers"])
        suite.history = (history + [data[index]["numbers"]])[-window:]
        folds.append(frozen)
        last_index = index
        if progress:
            progress("adaptive_window", step + 1, len(eligible), "running")
    return dict(
        mode="adaptive_window_walk_forward",
        is_live=False,
        count=len(folds),
        folds=folds,
        results=results,
        leaderboards=live_rankings(results),
        shadow={"results": shadow},
        events=events,
        adaptive=control.status(),
        model_statuses=suite.statuses(),
        decision={
            "allowed_for_live": False,
            "method": "shrink",
            "reason": "智慧視窗歷史模擬僅供比較；正式切換仍須獨立事前影子實戰驗證",
        },
    )
