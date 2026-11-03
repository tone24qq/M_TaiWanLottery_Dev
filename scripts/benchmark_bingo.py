"""Reproducible before/after CPU benchmarks, separate from prediction accuracy.

Use a detached PR #1 worktree as --baseline. Each measurement is a fresh process
with identical data, seeds, artifacts and bounded thread settings. Only the
10,000-row feature microbenchmark uses an explicitly synthetic size fixture.
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


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _worker(job):
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(job["threads"])
    import pickle
    import resource
    import numpy as np
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_models import ModelSuite, features
    from taiwan_lottery.ml.bingo_evaluation import fuse, walk_forward

    with gzip.open(job["source"], "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    n = job.get("size", 1000)
    if n > len(data):
        rng = np.random.default_rng(20261103)
        history = [
            sorted(rng.choice(np.arange(1, 81), 20, replace=False).tolist())
            for _ in range(n)
        ]
        source_kind = "synthetic size-only feature fixture"
    else:
        history = [d["numbers"] for d in data[-n:]]
        source_kind = "official actual history"
    outputs = None
    with threadpool_limits(limits=job["threads"]):
        if job["kind"] == "artifact":
            suite = ModelSuite(
                [d["numbers"] for d in data[:200]], threads=job["threads"]
            ).fit()
            if len(suite.scores(history)) != 8:
                raise RuntimeError(suite.statuses())
            Path(job["artifact"]).write_bytes(
                pickle.dumps(suite, pickle.HIGHEST_PROTOCOL)
            )
            return dict(kind="artifact", models=suite.statuses())
        if job["kind"] in ("inference", "fusion"):
            suite = pickle.loads(Path(job["artifact"]).read_bytes())
            score_map = suite.scores(history)
        if job["kind"] == "feature_stream" and job["variant"] == "optimized":
            from taiwan_lottery.ml.bingo_models import FeatureHistory
        if job["kind"] == "fusion" and job["variant"] == "optimized":
            from taiwan_lottery.ml.bingo_evaluation import fuse_many
        wall = time.perf_counter()
        cpu = time.process_time()
        repeats = job.get("repeats", 1)
        if job["kind"] == "features":
            for _ in range(repeats):
                outputs = features(history).tolist()
        elif job["kind"] == "feature_stream":
            count = job["size"]
            rows = [d["numbers"] for d in data[: count + 100]]
            checksum = hashlib.sha256()
            if job["variant"] == "optimized":
                prepared = FeatureHistory(rows)
                for index in range(100, len(rows)):
                    checksum.update(prepared.row(index, 1000).tobytes())
            else:
                for index in range(100, len(rows)):
                    checksum.update(
                        features(rows[max(0, index - 1000) : index]).tobytes()
                    )
            outputs = checksum.hexdigest()
        elif job["kind"] == "training":
            suite = ModelSuite(history, threads=job["threads"]).fit()
            outputs = suite.scores(history)
            if len(outputs) != 8:
                raise RuntimeError(suite.statuses())
        elif job["kind"] == "inference":
            for _ in range(repeats):
                outputs = suite.scores(history)
        elif job["kind"] == "fusion":
            for _ in range(repeats):
                if job["variant"] == "optimized":
                    outputs = fuse_many(score_map)
                else:
                    outputs = {
                        str(k): {
                            method: fuse(score_map, method, k)
                            for method in ("equal", "shrink", "borda", "majority")
                        }
                        for k in (3, 4, 5)
                    }
        elif job["kind"] == "walk_forward":
            report = walk_forward(data, count=job["size"], threads=job["threads"])
            outputs = {
                key: report[key] for key in ("folds", "results", "selected_method")
            }
            destination = Path(job["report"])
            with gzip.open(destination, "wt", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False)
        else:
            raise ValueError(job["kind"])
        elapsed = time.perf_counter() - wall
        cpu_seconds = time.process_time() - cpu
    if job["kind"] in ("inference", "training"):
        Path(job["values"]).write_text(json.dumps(outputs))
    return dict(
        kind=job["kind"],
        variant=job["variant"],
        size=n,
        threads=job["threads"],
        repetitions=repeats,
        wall_seconds=elapsed,
        cpu_seconds=cpu_seconds,
        cpu_percent=100 * cpu_seconds / max(elapsed, 1e-12),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        operations_per_second=(
            job["size"]
            if job["kind"] in ("feature_stream", "walk_forward")
            else repeats
        )
        / max(elapsed, 1e-12),
        result_sha256=_digest(outputs),
        source_kind=source_kind,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("work/stage1-baseline"))
    parser.add_argument("--output", type=Path, default=Path("work/stage2/performance"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip heavy training and full all-model walk-forward",
    )
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(json.loads(args.worker)), ensure_ascii=False))
        return
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "docs/validation/official-draws-20261103.json.gz"
    artifact = output / "same-stage1-model.pkl"
    script = Path(__file__).resolve()
    results = []

    def run(kind, size, variant, repeats=1):
        repo = baseline if variant == "baseline" else root
        job = dict(
            kind=kind,
            size=size,
            threads=args.threads,
            variant=variant,
            repeats=repeats,
            source=str(source),
            artifact=str(artifact),
            values=str(output / f"{kind}-{size}-{variant}-scores.json"),
            report=str(output / f"fixed-wf-{size}-{variant}.json.gz"),
        )
        env = {**os.environ, "PYTHONPATH": str(repo)}
        result = subprocess.run(
            [sys.executable, str(script), "--worker", json.dumps(job)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-6000:])
        row = json.loads(result.stdout.strip().splitlines()[-1])
        if kind != "artifact":
            results.append(row)
        (output / "benchmark-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    run("artifact", 1000, "baseline")
    for n in (100, 300, 1000, 10000):
        repeats = max(10, 30000 // n)
        for variant in ("baseline", "optimized"):
            run("features", n, variant, repeats)
    for n in (100, 300, 1000):
        for variant in ("baseline", "optimized"):
            run("feature_stream", n, variant)
        for variant in ("baseline", "optimized"):
            run("inference", n, variant, 50)
    for variant in ("baseline", "optimized"):
        run("fusion", 1000, variant, 200)
    if not args.quick:
        for n in (100, 300, 1000):
            for variant in ("baseline", "optimized"):
                run("training", n, variant)
        for n in (100, 300, 1000):
            for variant in ("baseline", "optimized"):
                run("walk_forward", n, variant)
    pairs = {}
    for row in results:
        pairs.setdefault((row["kind"], row["size"]), {})[row["variant"]] = row
    summary = []
    for (kind, n), pair in pairs.items():
        before, after = pair["baseline"], pair["optimized"]
        equivalent = before["result_sha256"] == after["result_sha256"]
        summary.append(
            dict(
                kind=kind,
                size=n,
                bitwise_output_identical=equivalent,
                wall_speedup=before["wall_seconds"] / after["wall_seconds"],
                cpu_speedup=before["cpu_seconds"] / after["cpu_seconds"],
                before=before,
                after=after,
            )
        )
    (output / "benchmark-comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    if not all(row["bitwise_output_identical"] for row in summary):
        raise RuntimeError(
            "Optimization changed a benchmark result; inspect exact full scores/ranks before accepting"
        )


if __name__ == "__main__":
    main()
