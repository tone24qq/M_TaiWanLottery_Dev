"""Reproduce the actual window-controller replay without writing to live SQL."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(100, 300, 1000), default=100)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/validation/official-draws-20261103.json.gz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/stage2/window-replay")
    )
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(args.threads)
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_adaptive_evaluation import adaptive_walk_forward

    with gzip.open(args.input, "rt") as stream:
        data = json.load(stream)
    wall, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(args.threads):
        report = adaptive_walk_forward(data, count=args.count, threads=args.threads)
    report["resource"] = dict(
        wall_seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        threads=args.threads,
    )
    report["data_sha256"] = hashlib.sha256(
        json.dumps(data, separators=(",", ":")).encode()
    ).hexdigest()
    original = Path(f"docs/validation/real-walk-forward-{args.count}.json.gz")
    if original.exists():
        with gzip.open(original, "rt") as stream:
            before = json.load(stream)
        old = {
            (r["target_term"], r["stars"]): r["predicted_numbers"]
            for r in before["results"]
            if r["model"] == "shrink"
        }
        fixed = {
            (r["target_term"], r["stars"]): r["predicted_numbers"]
            for r in report["results"]
            if r["model"] == "fixed"
        }
        if fixed != old:
            raise AssertionError("Fixed baseline changed during adaptive replay")
        report["fixed_stage1_exact_parity"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"adaptive-window-{args.count}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False)
    summary = {
        k: v for k, v in report.items() if k not in ("folds", "results", "shadow")
    }
    summary.update(report=str(path), report_bytes=path.stat().st_size)
    (args.output / f"window-summary-{args.count}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            dict(count=args.count, resource=report["resource"], report=str(path))
        )
    )


if __name__ == "__main__":
    main()
