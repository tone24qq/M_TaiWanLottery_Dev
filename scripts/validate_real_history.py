"""Reproduce genuine historical comparisons, without touching the live ledger.

Run: python -m scripts.validate_real_history --count 1000 --output work/validation
The checked-in snapshot contains actual official outcomes, not generated data.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="官方歷史資料嚴格樣本外驗證")
    parser.add_argument("--count", type=int, choices=(100, 300, 1000), default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/validation/official-draws-20261103.json.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("work/validation"))
    options = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(max(1, min(8, options.threads)))
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_evaluation import walk_forward

    with gzip.open(options.input, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    started = time.perf_counter()
    with threadpool_limits(limits=options.threads):
        report = walk_forward(data, count=options.count, threads=options.threads)
    report["elapsed_seconds"] = time.perf_counter() - started
    options.output.mkdir(parents=True, exist_ok=True)
    destination = options.output / f"real-walk-forward-{options.count}.json.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False)
    print(
        json.dumps(
            {
                "file": str(destination),
                "count": report["count"],
                "seconds": report["elapsed_seconds"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
