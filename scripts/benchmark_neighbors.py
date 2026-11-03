"""Reproduce official-data neighbor walk-forward without a live database.

Default comparisons reuse audited historical predictions already committed to
the repository. Neighbor and random decisions are newly replayed, prefix-only,
for every target. --fresh-comparators refits all eight models and ABC instead.
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


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_sha(path):
    return hashlib.sha256(gzip.decompress(Path(path).read_bytes())).hexdigest()


def frozen_comparators(root, count, source):
    """Verify immutable historical decisions, retaining their actual versions."""
    from taiwan_lottery.ml.bingo_neighbor_evaluation import canonical_hash

    baseline_path = root / f"docs/validation/real-walk-forward-{count}.json.gz"
    abc_path = root / f"docs/validation/stage2/associations/abc-decisions-{count}.jsonl.gz"
    manifest = json.loads((abc_path.parent / "manifest.json").read_text())
    if _sha(abc_path) != manifest["files"][abc_path.name]:
        raise ValueError("歷史ABC決策檔雜湊不符")
    base = _read(baseline_path)
    if base["source_snapshot_sha256"] != _source_sha(source):
        raise ValueError("歷史模型比較的官方來源快照不同")
    folds = {}
    for fold in base["folds"]:
        term = fold["target_term"]
        predictions = fold["predictions"]
        if set(predictions) != {"frequency", "cdm", "river", "lightgbm", "catboost",
                                "transformer", "dqn", "ensemble", "equal", "shrink", "borda", "majority"}:
            raise ValueError("歷史比較缺少八模型或四投票")
        folds[term] = dict(
            target_term=term, source_term=fold["features_through"], predictions=predictions,
            cutoffs={name: dict(training=fold["features_through"] if name in
                                {"frequency", "cdm", "river", "ensemble"} else fold["trained_through"],
                                features=fold["features_through"]) for name in predictions})
    results = base["results"]
    seen = set()
    with gzip.open(abc_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            decision = item["decision"]
            if decision["decision_hash"] != canonical_hash({k: v for k, v in decision.items() if k != "decision_hash"}):
                raise ValueError("歷史ABC逐期決策雜湊不符")
            if decision["engine_source_sha256"] != manifest["engine_source_sha256"]:
                raise ValueError("歷史ABC引擎版本雜湊不一致")
            term = decision["target_term"]
            if term in seen or term not in folds or decision["is_live"] is not False:
                raise ValueError("歷史ABC比較目標或研究身分錯誤")
            seen.add(term)
            for star, group in decision["by_star"].items():
                for label, candidate in group.items():
                    name = "association_" + label.lower()
                    folds[term]["predictions"].setdefault(name, {})[star] = candidate["numbers"]
                    folds[term]["cutoffs"][name] = dict(training=decision["feature_cutoff"],
                                                         features=decision["feature_cutoff"])
            results.extend(item["outcomes"])
    if seen != set(folds):
        raise ValueError("歷史ABC與八模型的目標期不一致")
    return dict(is_live=False, folds=list(folds.values()), results=results,
                model_statuses=base["model_statuses"], provenance=dict(
                    kind="audited_frozen_historical_comparators",
                    source_snapshot_sha256=_source_sha(source), source_file_sha256=_sha(source),
                    baseline_file=str(baseline_path.relative_to(root)),
                    baseline_sha256=_sha(baseline_path), abc_file=str(abc_path.relative_to(root)),
                    abc_sha256=_sha(abc_path), abc_engine_sha256=manifest["engine_source_sha256"],
                    note="既有歷史凍結比較；本次重播鄰號與隨機，不使用目前正式模型倒推"))


def _write_gzip(path, value):
    # A fixed mtime and empty embedded filename keep byte-level reproducibility.
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/validation/official-draws-20261103.json.gz"))
    parser.add_argument("--output", type=Path, default=Path("work/neighbors"))
    parser.add_argument("--counts", type=int, nargs="+", choices=(100, 300, 1000), default=[100, 300, 1000])
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fresh-comparators", action="store_true")
    options = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(options.threads)
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_neighbor_evaluation import neighbor_walk_forward, canonical_hash
    from taiwan_lottery.ml.bingo_neighbors import prepare_neighbor_bundle, neighbor_view

    root = Path(__file__).resolve().parents[1]
    data = _read(options.source)
    options.output.mkdir(parents=True, exist_ok=True)
    hashes, summaries = {}, {}
    with threadpool_limits(options.threads):
        for count in options.counts:
            base = None if options.fresh_comparators else frozen_comparators(root, count, options.source)
            started, cpu = time.perf_counter(), time.process_time()
            report = neighbor_walk_forward(data, count=count, threads=options.threads, baseline_report=base)
            # Replay the entire specified sample twice. Resource measurements are
            # outside sealed decisions and cannot alter the reproducibility hash.
            again = neighbor_walk_forward(data, count=count, threads=options.threads, baseline_report=base)
            if canonical_hash(report) != canonical_hash(again):
                raise AssertionError("相同輸入重跑產生不同回測結果")
            elapsed, cpu_elapsed = time.perf_counter()-started, time.process_time()-cpu
            report["source_snapshot_sha256"] = _source_sha(options.source)
            path = options.output / f"neighbor-walk-forward-{count}.json.gz"
            _write_gzip(path, report)
            hashes[path.name] = _sha(path)
            summary = {k: v for k, v in report.items() if k not in {"folds", "results"}}
            summary.update(reproducibility=dict(repeated_runs=2, identical=True,
                                                report_sha256=canonical_hash(again)),
                           resources=dict(threads=options.threads, wall_seconds_two_replays=elapsed,
                                          cpu_seconds=cpu_elapsed,
                                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            summary_path = options.output / f"neighbor-summary-{count}.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            hashes[summary_path.name] = _sha(summary_path)
            summaries[str(count)] = summary
            print(json.dumps(dict(count=count, seconds=round(elapsed, 3),
                                  decision_chain_sha256=report["decision_chain_sha256"],
                                  proven={k:v["proven_better_than_random"] for k,v in report["validation"].items()})), flush=True)
        # All descriptive windows/lags/snapshots are also built on real data.
        started = time.perf_counter()
        bundle = prepare_neighbor_bundle(data)
        snapshots = {str(s): neighbor_view(bundle, window=1000, lag=1, snapshot=s) for s in (0,5,10,20)}
        _write_gzip(options.output / "neighbor-latest-snapshots.json.gz", snapshots)
        hashes["neighbor-latest-snapshots.json.gz"] = _sha(options.output / "neighbor-latest-snapshots.json.gz")
        manifest = dict(source_snapshot_sha256=_source_sha(options.source), source_file_sha256=_sha(options.source),
                        counts=options.counts, files=hashes,
                        engine_files={p:_sha(root/p) for p in ("taiwan_lottery/ml/bingo_neighbors.py",
                                      "taiwan_lottery/ml/bingo_neighbor_evaluation.py", "scripts/benchmark_neighbors.py")},
                        bundle_wall_seconds=time.perf_counter()-started, bundle_resources=bundle.get("resources"),
                        is_live=False, writes_live_ledger=False, modifies_production_weights=False)
        (options.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
