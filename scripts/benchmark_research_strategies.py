"""Causal actual-data comparison of Today Only/Historical/Adaptive Hybrid.

Historical scores reuse the already frozen strict nested baseline audit rather
than fitting a current artifact backwards. Today suites are genuinely fit from
that target date's earlier draws, with scheduled/maturity-triggered refits.
Historical simulation is never written into the live strategy ledger.
"""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def run(count, threads=2):
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    import numpy as np
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.bingo_clock import to_taipei
    from taiwan_lottery.ml.bingo_models import ModelSuite, draw_matrix
    from taiwan_lottery.ml.bingo_evaluation import fuse, summarize_outcomes
    from taiwan_lottery.ml.bingo_research_strategies import ResearchStrategies
    from datetime import datetime, timedelta

    output = Path("work/stage2/parallel")
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open("docs/validation/official-draws-20261103.json.gz", "rt") as f:
        data = json.load(f)
    path = Path(f"work/stage2/adaptive/nested-walk-forward-{count}.json.gz")
    with gzip.open(path, "rt") as f:
        fixed = json.load(f)
    by_term = {r["draw_term"]: i for i, r in enumerate(data)}
    state = ResearchStrategies(simulation=True)
    suite = None
    current_day = None
    last_fit = 0
    folds = []
    results = []
    fits = 0
    started, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(limits=threads):
        for step, historical in enumerate(fixed["folds"]):
            target = historical["target_term"]
            index = by_term[target]
            when = to_taipei(datetime.fromisoformat(data[index]["draw_datetime"]))
            day = when.date().isoformat()
            earlier = [
                r
                for r in data[:index]
                if datetime.fromisoformat(r["draw_datetime"]).date() == when.date()
            ]
            history = [r["numbers"] for r in earlier]
            if current_day != day:
                suite = None
                current_day = day
                last_fit = 0
            mature = any(last_fit < n <= len(history) for n in (10, 32, 80, 96))
            if len(history) >= 10 and (
                suite is None or mature or len(history) - last_fit >= 36
            ):
                suite = ModelSuite(history, threads=threads, window=1000).fit()
                last_fit = len(history)
                fits += 1
            if len(history) < 10:
                matrix = draw_matrix(history)
                day_scores = {
                    "today_prior": (
                        (matrix.sum(axis=0) + 2) / (len(history) + 8)
                    ).tolist()
                }
            else:
                day_scores = suite.scores(history)
            today = {str(k): fuse(day_scores, "equal", k)["scores"] for k in (3, 4, 5)}
            historic = {
                str(k): historical["profiles"]["fixed"]["fusion_scores"][str(k)][
                    "shrink"
                ]
                for k in (3, 4, 5)
            }
            predictions = state.score(today, historic, len(history))
            before_source = data[index - 1]
            sealed_at = to_taipei(
                datetime.fromisoformat(before_source["draw_datetime"])
            ) + timedelta(seconds=1)
            audit = dict(
                target_term=target,
                source_term=before_source["draw_term"],
                sealed_at=sealed_at.isoformat(),
                target_time=when.isoformat(),
                today_history_start=earlier[0]["draw_term"] if earlier else None,
                today_history_end=earlier[-1]["draw_term"] if earlier else None,
                today_history_count=len(earlier),
                today_training_cutoff=earlier[last_fit - 1]["draw_term"]
                if last_fit
                else None,
                today_history_dates=[day] if earlier else [],
                today_model_scores=day_scores,
                historical_decision_hash=historical["decision_hash"],
                predictions=predictions,
                is_live=False,
            )
            assert int(before_source["draw_term"]) < int(target) and sealed_at < when
            # This immutable hash precedes reading the target answer below.
            audit["decision_hash"] = hashlib.sha256(
                json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            actual = set(data[index]["numbers"])
            rows = []
            for stars in (3, 4, 5):
                for family, prediction in predictions[str(stars)].items():
                    row = dict(
                        model="strategy_" + family,
                        stars=stars,
                        target_term=target,
                        source_term=before_source["draw_term"],
                        created_at=sealed_at.isoformat(),
                        target_time=when.isoformat(),
                        numbers=prediction["numbers"],
                        scores=prediction["scores"],
                        actual_numbers=sorted(actual),
                        hit_count=len(actual & set(prediction["numbers"])),
                        decision_hash=audit["decision_hash"] + family,
                        metadata={"history_dates": [day] if earlier else []},
                        is_live=False,
                    )
                    rows.append(row)
                    results.append(row)
            state.observe(target, rows)
            folds.append(audit)
            if suite is not None:
                suite.update(history, data[index]["numbers"])
                suite.history = history + [data[index]["numbers"]]
    report = dict(
        mode="parallel_strategies_walk_forward",
        is_live=False,
        count=count,
        folds=folds,
        results=results,
        summary=state.status(),
        resource=dict(
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            today_refits=fits,
            threads=threads,
        ),
        historical_source=str(path),
        scope="Today模型只看目標當天較早期數；Historical重用已凍結的嚴格逐期工件結果；Hybrid只用先前已結算配對實績",
    )
    destination = output / f"parallel-strategies-{count}.json.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, allow_nan=False)
    summary = {k: v for k, v in report.items() if k not in ("folds", "results")}
    summary["artifact_bytes"] = destination.stat().st_size
    (output / f"parallel-summary-{count}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "count": count,
                "resource": report["resource"],
                "champions": {
                    k: v["champion"] for k, v in state.status()["stars"].items()
                },
                "path": str(destination),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, choices=(100, 300, 1000), default=100)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    run(a.count, a.threads)
