"""Real official-data nested comparisons in fresh bounded CPU processes.

This only writes research files; it never opens or seeds the live database.
Run after pure performance profiling to avoid resource contention:
  python -m scripts.benchmark_adaptation --counts 100 300 1000 --threads 2
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description="實際官方資料巢狀樣本外比較")
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/validation/official-draws-20261103.json.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("work/stage2/adaptive"))
    parser.add_argument("--model-names", nargs="+")
    parser.add_argument("--worker", type=int)
    return parser.parse_args()


def worker(options):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(max(1, min(options.threads, 4)))
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_nested import nested_walk_forward

    with gzip.open(options.input, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    wall, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(limits=options.threads):
        report = nested_walk_forward(
            data,
            count=options.worker,
            threads=options.threads,
            model_names=options.model_names,
        )
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    report["benchmark"] = {
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "mean_cpu_percent": cpu / wall * 100,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "outer_draws_per_second": options.worker / wall,
        "threads": options.threads,
        "data_sha256": hashlib.sha256(
            json.dumps(data, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    options.output.mkdir(parents=True, exist_ok=True)
    destination = options.output / f"nested-walk-forward-{options.worker}.json.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False)
    original = Path(f"docs/validation/real-walk-forward-{options.worker}.json.gz")
    parity = None
    if original.exists() and options.model_names is None:
        with gzip.open(original, "rt", encoding="utf-8") as stream:
            before = json.load(stream)
        after = [
            {key: value for key, value in row.items() if key != "is_live"}
            for row in report["baseline"]["results"]
        ]
        parity = after == before["results"]
        if not parity:
            raise AssertionError(f"固定基準 {options.worker} 期與第一階段封存比較不同")
    summary = {
        **report["benchmark"],
        "count": options.worker,
        "stage1_baseline_exact_parity": parity,
        "decision": {
            key: value
            for key, value in report["decision"].items()
            if key != "proposals"
        },
        "fit_counts": report["resource"]["fit_counts"],
        "selection_count": report["nested"]["selection_count"],
        "model_statuses": report["model_statuses"],
        "by_star": {
            key: value["cumulative"] for key, value in report["leaderboards"].items()
        },
        "shadow_by_star": {
            key: value["cumulative"]
            for key, value in report["shadow"]["leaderboards"].items()
        },
        "report": str(destination),
        "report_bytes": destination.stat().st_size,
    }
    (options.output / f"nested-summary-{options.worker}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "count": options.worker,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "rss_mib": summary["peak_rss_mib"],
                "baseline_parity": parity,
                "allowed_for_live": summary["decision"]["allowed_for_live"],
                "report": str(destination),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    options = arguments()
    if options.worker is not None:
        worker(options)
        return
    if any(count not in (100, 300, 1000) for count in options.counts):
        raise ValueError("驗收比較使用 100、300、1000 期")
    options.output.mkdir(parents=True, exist_ok=True)
    for count in options.counts:
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_adaptation",
            "--worker",
            str(count),
            "--threads",
            str(options.threads),
            "--input",
            str(options.input),
            "--output",
            str(options.output),
        ]
        if options.model_names:
            command += ["--model-names", *options.model_names]
        subprocess.run(command, check=True)
    summaries = [
        json.loads((options.output / f"nested-summary-{count}.json").read_text())
        for count in options.counts
    ]
    (options.output / "nested-summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2)
    )
    lines = [
        "# 第二階段真實資料巢狀 Walk-forward 比較",
        "",
        "固定基準與第一階段完全相同；候選另以受限影子程序比較，不能用外層答案回選當折參數。",
        "候選包含真正的歷史視窗、頻率特徵權重、CDM 先驗種類／強度、投票方法與重訓頻率改動。",
        "",
        "| 期數 | 壁鐘秒 | CPU 秒 | 平均 CPU | 峰值 MiB | 吞吐（期／秒） | 基準逐筆一致 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['count']} | {summary['wall_seconds']:.3f} | {summary['cpu_seconds']:.3f} | {summary['mean_cpu_percent']:.1f}% | {summary['peak_rss_mib']:.1f} | {summary['outer_draws_per_second']:.3f} | {summary['stage1_baseline_exact_parity']} |"
        )
    for summary in summaries:
        lines += [
            "",
            f"## {summary['count']} 期結果",
            "",
            "| 星數 | 固定 Shrink 平均命中 | 安全自適應平均命中 | 隨機平均 | 已允許替換正式策略 |",
            "|---:|---:|---:|---:|---|",
        ]
        for stars, rows in summary["by_star"].items():
            fixed = next(row for row in rows if row["model"] == "shrink")
            adaptive = next(row for row in rows if row["model"] == "adaptive")
            lines.append(
                f"| {stars} | {fixed['mean_hits']:.4f} | {adaptive['mean_hits']:.4f} | {int(stars) * 0.25:.2f} | {summary['decision']['allowed_for_live']} |"
            )
        lines += [
            "",
            summary["decision"]["reason"],
            f"完成 {summary['selection_count']} 次事前內層選擇，實際模型訓練次數：{summary['fit_counts']}。",
            "統計門檻採配對有界差異、最少 100 期、多重比較及序列 alpha 分配；全中率不得惡化且資源比須在 1.5 內。",
            "數字是單機特定快照測量，不是速度或準確度保證。完整 80 號分數、期號截止與預選候選包含於 gzip 報告。",
        ]
    (options.output / "nested-comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
