"""A/B/C causal research replay plus exact scalar/matrix kernel benchmarks.

C consumes the already frozen eight-model scores from nested walk-forward folds,
never a currently fitted model applied backwards. This script writes simulation
files only. It does not connect to, or append to, a formal live prediction ledger.
"""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _source(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _steps(rows):
    from datetime import datetime
    from taiwan_lottery.bingo_clock import is_successor

    return [
        is_successor(
            a["draw_term"],
            datetime.fromisoformat(a["draw_datetime"]),
            b["draw_term"],
            datetime.fromisoformat(b["draw_datetime"]),
        )
        for a, b in zip(rows, rows[1:])
    ]


def _scalar(matrix, steps):
    import numpy as np

    trials = np.zeros(80, dtype=np.int64)
    cooc = np.zeros((80, 80), dtype=np.int64)
    for row in matrix:
        values = np.flatnonzero(row)
        for a in values:
            trials[a] += 1
            for b in values:
                if a != b:
                    cooc[a, b] += 1
    result = [trials.tolist(), cooc.tolist()]
    for lag in range(1, 21):
        count = np.zeros(80, dtype=np.int64)
        hits = np.zeros((80, 80), dtype=np.int64)
        valid = 0
        for t in range(lag, len(matrix)):
            if not all(steps[t - lag : t]):
                continue
            valid += 1
            after = np.flatnonzero(matrix[t])
            for a in np.flatnonzero(matrix[t - lag]):
                count[a] += 1
                for b in after:
                    hits[a, b] += 1
        result.append([count.tolist(), hits.tolist(), valid])
    return result


def _kernels(job):
    import resource
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_models import draw_matrix
    from taiwan_lottery.ml.bingo_associations import _cooccurrence_counts, _lag_counts

    rows = _source(job["source"])[-job["count"] :]
    matrix = draw_matrix([r["numbers"] for r in rows])
    steps = _steps(rows)
    with threadpool_limits(limits=2):
        started, cpu = time.perf_counter(), time.process_time()
        if job["variant"] == "scalar":
            output = _scalar(matrix, steps)
        else:
            a, b = _cooccurrence_counts(matrix)
            output = [a.tolist(), b.tolist()]
            for lag in range(1, 21):
                a, b, n = _lag_counts(matrix, steps, lag)
                output.append([a.tolist(), b.tolist(), n])
        wall = time.perf_counter() - started
        cpu_time = time.process_time() - cpu
    return dict(
        count=job["count"],
        variant=job["variant"],
        wall_seconds=wall,
        cpu_seconds=cpu_time,
        cpu_percent=cpu_time / max(wall, 1e-12) * 100,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        target_rows_per_second=job["count"] / max(wall, 1e-12),
        result_sha256=_hash(output),
        method="all same-draw80x80 counts plus all20 lagged80x80 counts; gap crossings excluded",
    )


def _replay(job):
    import resource
    import math
    import numpy as np
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_associations import association_candidates, VERSION
    import inspect
    from datetime import datetime
    from taiwan_lottery.bingo_clock import next_draw

    engine_file = Path(inspect.getsourcefile(association_candidates))
    engine_sha = hashlib.sha256(engine_file.read_bytes()).hexdigest()
    from taiwan_lottery.ml.bingo_evaluation import live_rankings

    rows = _source(job["source"])
    source_index = {r["draw_term"]: i for i, r in enumerate(rows)}
    nested = _source(job["nested"])
    folds = nested["folds"]
    if len(folds) != job["count"]:
        raise ValueError("Nested folded sample count mismatch")
    outcomes = []
    costs = []
    confirmed = 0
    active = 0
    proof = []
    audit_path = Path(job["audit"])
    total_started, total_cpu = time.perf_counter(), time.process_time()
    with (
        threadpool_limits(limits=2),
        gzip.open(audit_path, "wt", encoding="utf-8") as audit,
    ):
        for step, fold in enumerate(folds, 1):
            target = str(fold["target_term"])
            index = source_index[target]
            fixed = fold["profiles"]["fixed"]
            cutoff = str(rows[index - 1]["draw_term"])
            if (
                not int(fixed["training_cutoff"]) < int(target)
                or fixed["feature_cutoff"] != cutoff
            ):
                raise ValueError("Frozen model or feature cutoff is not before target")
            models = fixed["model_scores"]
            if len(models) != 8 or any(len(v) != 80 for v in models.values()):
                raise ValueError(
                    "C requires the eight actual frozen model score vectors"
                )
            weights = {
                str(k): fixed["fusion_weights"][str(k)]["shrink"] for k in (3, 4, 5)
            }
            history = rows[max(0, index - 1000) : index]
            expected_target, target_at = next_draw(
                history[-1]["draw_term"],
                datetime.fromisoformat(history[-1]["draw_datetime"]),
            )
            if expected_target != target:
                raise ValueError(
                    "Target schedule and frozen source are not immediate successors"
                )
            calculated = association_candidates(
                history,
                {str(k): models for k in (3, 4, 5)},
                model_weights_by_star=weights,
                target_date=target_at.date(),
            )
            segments = calculated["validation_segments"]
            for rule in calculated["rules"]:
                split = rule["split_history"]
                d, v, c = (
                    split[name]
                    for name in ("discovery", "selection_validation", "confirmation")
                )
                if d["count"] and int(d["to"]) > int(segments["discovery_end_term"]):
                    raise ValueError("Discovery crossed the common validation boundary")
                if v["count"] and (
                    int(v["from"]) <= int(segments["discovery_end_term"])
                    or int(v["to"]) > int(segments["selection_end_term"])
                ):
                    raise ValueError("Selection crossed a common holdout boundary")
                if c["count"] and int(c["from"]) < int(
                    segments["confirmation_start_term"]
                ):
                    raise ValueError("Confirmation overlaps any window selection data")
            # Freeze all nine full score vectors and ranks before touching answer t.
            frozen = dict(
                target_term=target,
                training_cutoff=fixed["training_cutoff"],
                feature_cutoff=cutoff,
                history_from=history[0]["draw_term"],
                history_to=cutoff,
                history_count=len(history),
                model_version=fixed["model_version"],
                feature_version=calculated["feature_version"],
                association_version=VERSION,
                engine_source_sha256=engine_sha,
                by_star=calculated["by_star"],
                rules=calculated["rules"],
                window_selection=calculated["window_selection"],
                validation_segments=segments,
                is_live=False,
            )
            for star, candidates in frozen["by_star"].items():
                if set(candidates) != set("ABC"):
                    raise ValueError("Incomplete A/B/C candidates")
                for label, prediction in candidates.items():
                    if (
                        len(prediction["scores"]) != 80
                        or len(prediction["ranking"]) != 80
                        or len(prediction["numbers"]) != int(star)
                    ):
                        raise ValueError("Incomplete archived80-score/star candidate")
                    if label == "C" and set(prediction["model_weights"]) != set(models):
                        raise ValueError(
                            "C did not include all available frozen models"
                        )
            frozen["decision_hash"] = _hash(frozen)
            # The simulated answer is revealed only after the decision freezes.
            actual = rows[index]["numbers"]
            settled = []
            for star, candidates in frozen["by_star"].items():
                for label, prediction in candidates.items():
                    item = dict(
                        target_term=target,
                        model=f"association_{label.lower()}",
                        stars=int(star),
                        predicted_numbers=prediction["numbers"],
                        actual_numbers=actual,
                        hit_count=len(set(prediction["numbers"]) & set(actual)),
                        draw_datetime=rows[index]["draw_datetime"],
                        is_live=False,
                    )
                    outcomes.append(item)
                    settled.append(item)
            audit.write(
                json.dumps(
                    dict(
                        decision=frozen,
                        outcomes=settled,
                        resources=calculated["resources"],
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            costs.append(calculated["resources"]["wall_seconds"])
            confirmed += calculated["resources"]["independently_confirmed_rules"]
            active += calculated["resources"]["active_confirmed_predictive_rules"]
            proof.append(
                dict(
                    target_term=target,
                    decision_hash=frozen["decision_hash"],
                    training_cutoff=fixed["training_cutoff"],
                    feature_cutoff=cutoff,
                )
            )
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "count": job["count"],
                            "completed": step,
                            "seconds": time.perf_counter() - total_started,
                        }
                    ),
                    flush=True,
                )
    if hashlib.sha256(engine_file.read_bytes()).hexdigest() != engine_sha:
        raise RuntimeError(
            "Association engine changed during validation; discard incomplete report"
        )
    baseline = nested.get("baseline", {}).get(
        "results", [r for r in nested["results"] if r["model"] != "adaptive"]
    )
    valid_names = {
        "frequency",
        "cdm",
        "river",
        "lightgbm",
        "catboost",
        "transformer",
        "dqn",
        "ensemble",
        "equal",
        "shrink",
        "borda",
        "majority",
    }
    baseline = [r for r in baseline if r["model"] in valid_names]
    boards = live_rankings(baseline + outcomes)
    comparisons = {}
    for stars in (3, 4, 5):
        base = {
            r["target_term"]: r["hit_count"]
            for r in baseline
            if r["model"] == "shrink" and r["stars"] == stars
        }
        comparisons[str(stars)] = {}
        for label in "ABC":
            paired = [
                r["hit_count"] - base[r["target_term"]]
                for r in outcomes
                if r["model"] == f"association_{label.lower()}" and r["stars"] == stars
            ]
            n = len(paired)
            difference = float(np.mean(paired))
            width = 2 * stars * math.sqrt(math.log(2 * 9 / 0.05) / (2 * n))
            comparisons[str(stars)][label] = dict(
                samples=n,
                delta_vs_frozen_shrink=difference,
                conservative_paired_ci_low=difference - width,
                conservative_paired_ci_high=difference + width,
                clears_exploratory_paired_gate=bool(
                    n >= 100 and difference - width > 0
                ),
                strategy_proven=False,
                shadow_only=True,
            )
    elapsed = time.perf_counter() - total_started
    cpu_elapsed = time.process_time() - total_cpu
    return dict(
        mode="causal_abc_shadow_replay",
        is_live=False,
        count=job["count"],
        version=VERSION,
        engine_source_sha256=engine_sha,
        first_target=folds[0]["target_term"],
        last_target=folds[-1]["target_term"],
        wall_seconds=elapsed,
        cpu_seconds=cpu_elapsed,
        cpu_percent=100 * cpu_elapsed / max(elapsed, 1e-12),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        throughput_targets_per_second=job["count"] / max(elapsed, 1e-12),
        per_target_wall_median=float(np.median(costs)),
        per_target_wall_p95=float(np.quantile(costs, 0.95)),
        independently_confirmed_rule_occurrences=confirmed,
        active_confirmed_rule_occurrences=active,
        leaderboards=boards,
        paired_comparisons=comparisons,
        decision_log=audit_path.name,
        decision_chain_sha256=_hash(proof),
        source_nested_sha256=hashlib.sha256(
            Path(job["nested"]).read_bytes()
        ).hexdigest(),
        methodology="Prefix only; frozen historical nested8-model scores and prior fusionweights for C; all9 decisions frozen before outcome. Simulations never enter live leaderboard. No current model backcasting.",
        limitations="Exploratory paired comparisons; no promotion from this script; association tests do not establish a betting or winning strategy.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path("docs/validation/official-draws-20261103.json.gz"),
    )
    p.add_argument(
        "--nested-directory", type=Path, default=Path("work/stage2/adaptive")
    )
    p.add_argument("--output", type=Path, default=Path("work/stage2/associations"))
    p.add_argument("--counts", type=int, nargs="+", default=[100, 300, 1000])
    p.add_argument("--worker")
    args = p.parse_args()
    if args.worker:
        job = json.loads(args.worker)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "2"
        result = _kernels(job) if job["kind"] == "kernels" else _replay(job)
        Path(job["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("leaderboards", "paired_comparisons")
                },
                ensure_ascii=False,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    root = script.parents[1]
    expected_engine_sha = hashlib.sha256(
        (root / "taiwan_lottery/ml/bingo_associations.py").read_bytes()
    ).hexdigest()
    for count in args.counts:
        for variant in ("scalar", "vectorized"):
            result = args.output / f"kernels-{count}-{variant}.json"
            job = dict(
                kind="kernels",
                source=str(args.source.resolve()),
                count=count,
                variant=variant,
                result=str(result.resolve()),
            )
            subprocess.run(
                [sys.executable, str(script), "--worker", json.dumps(job)],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                check=True,
            )
        before = json.loads((args.output / f"kernels-{count}-scalar.json").read_text())
        after = json.loads(
            (args.output / f"kernels-{count}-vectorized.json").read_text()
        )
        if before["result_sha256"] != after["result_sha256"]:
            raise ValueError("Scalar/matrix integer counts changed")
        job = dict(
            kind="replay",
            source=str(args.source.resolve()),
            count=count,
            nested=str(
                (
                    args.nested_directory / f"nested-walk-forward-{count}.json.gz"
                ).resolve()
            ),
            audit=str((args.output / f"abc-decisions-{count}.jsonl.gz").resolve()),
            result=str((args.output / f"abc-summary-{count}.json").resolve()),
        )
        subprocess.run(
            [sys.executable, str(script), "--worker", json.dumps(job)],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            check=True,
        )
        completed = json.loads(Path(job["result"]).read_text())
        if completed["engine_source_sha256"] != expected_engine_sha:
            raise RuntimeError(
                "Engine changed between sample sizes; discard mixed-version benchmark"
            )


if __name__ == "__main__":
    main()
