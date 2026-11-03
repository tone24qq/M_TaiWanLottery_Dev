"""Bounded neighbour/temporal research with separate descriptive and test samples.

All 80 by 80 transitions are counted, even when the interface only displays the
nine predeclared displacements.  Nothing in this module writes model weights,
prediction ledgers or a database.  Historical rates are not forecast odds.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import time

import numpy as np
from scipy.stats import binom, norm

from taiwan_lottery.bingo_clock import is_successor, next_draw, to_taipei

VERSION = "bingo-neighbors-1"
WINDOWS = (5, 10, 20, 30, 50, 100, 300, 500, 1000)
LAGS = (1, 2, 3, 5, 10, 20)
OFFSETS = (-5, -3, -2, -1, 0, 1, 2, 3, 5)
SNAPSHOTS = (0, 5, 10, 20)
BASELINE = 0.25
ALPHA = 0.05
MIN_TRIGGERS = 20
# Looking up arbitrary targets must not quietly escape the correction family.
HYPOTHESES = len(WINDOWS) * len(LAGS) * 80 * 80 * len(SNAPSHOTS)
NO_EVIDENCE = "尚未證明優於隨機"
SCORE_KIND = "相對研究分數，非下一期中獎機率"
MAX_ROWS = max(WINDOWS) + max(SNAPSHOTS)


def _time(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime):
        raise ValueError("鄰號研究需要真實開獎時間，不能用資料列位置代替期數")
    return to_taipei(value)


def _rows(draws, source_term=None):
    """Validate before clipping; duplicate identical results are idempotent."""
    rows = []
    seen = {}
    for item in draws:
        term = str(item["draw_term"])
        if not term.isdigit() or int(term) <= 0:
            raise ValueError("開獎期號無效")
        stamp = _time(item.get("draw_datetime"))
        raw = list(item["numbers"])
        if (
            len(raw) != 20
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) for n in raw)
            or len(set(raw)) != 20
            or any(n < 1 or n > 80 for n in raw)
        ):
            raise ValueError("每期必須包含 1～80 中的 20 個不重複整數")
        numbers = sorted(map(int, raw))
        # Also verifies an isolated row is on the official five-minute schedule.
        next_draw(term, stamp)
        identity = (stamp, tuple(numbers))
        if term in seen:
            if seen[term] != identity:
                raise ValueError("同一期號出現衝突資料")
            continue
        if rows and (stamp <= rows[-1]["draw_datetime"] or int(term) <= int(rows[-1]["draw_term"])):
            raise ValueError("歷史資料必須按真實期號和時間遞增")
        seen[term] = identity
        rows.append(dict(draw_term=term, draw_datetime=stamp, numbers=numbers))
    if source_term is not None:
        positions = [i for i, row in enumerate(rows) if row["draw_term"] == str(source_term)]
        if not positions:
            raise ValueError("來源期號不在已知歷史資料內")
        rows = rows[: positions[0] + 1]
    if not rows:
        raise ValueError("沒有已知歷史資料可供鄰號分析")
    return rows[-MAX_ROWS:]


def _matrix(rows):
    result = np.zeros((len(rows), 80), dtype=np.uint8)
    for index, row in enumerate(rows):
        result[index, np.asarray(row["numbers"]) - 1] = 1
    return result


def _steps(rows):
    return np.asarray(
        [is_successor(a["draw_term"], a["draw_datetime"], b["draw_term"], b["draw_datetime"])
         for a, b in zip(rows, rows[1:])],
        dtype=bool,
    )


def _history(rows):
    return dict(
        **{"from": rows[0]["draw_term"] if rows else None},
        to=rows[-1]["draw_term"] if rows else None,
        count=len(rows),
    )


def _targets(steps, lag, start, stop, independent=False):
    """Both ends lie in [start,stop); no path may cross a missing draw."""
    values = np.arange(start + lag, stop, lag + 1 if independent else 1)
    gaps = np.r_[0, np.cumsum(~steps)]
    return values[gaps[values] == gaps[values - lag]]


def _counts(matrix, steps, lag, start, stop, independent=False):
    targets = _targets(steps, lag, start, stop, independent)
    before = matrix[targets - lag].astype(np.float64)
    after = matrix[targets].astype(np.float64)
    return (
        np.rint(before.sum(axis=0)).astype(np.uint16),
        np.rint(before.T @ after).astype(np.uint16),
        len(targets),
    )


def _prefix(matrix, steps, lag):
    """One at most 13 MiB integer prefix tensor, reused for all windows."""
    n = len(matrix)
    hits = np.zeros((n + 1, 80, 80), dtype=np.uint16)
    trials = np.zeros((n + 1, 80), dtype=np.uint16)
    pairs = np.zeros(n + 1, dtype=np.uint16)
    targets = _targets(steps, lag, 0, n)
    if len(targets):
        hits[targets + 1] = matrix[targets - lag, :, None] * matrix[targets, None, :]
        trials[targets + 1] = matrix[targets - lag]
        pairs[targets + 1] = 1
    np.cumsum(hits, axis=0, dtype=np.uint16, out=hits)
    np.cumsum(trials, axis=0, dtype=np.uint16, out=trials)
    np.cumsum(pairs, dtype=np.uint16, out=pairs)
    return trials, hits, pairs


def _lookup(max_trials):
    """Precompute statistical evidence in the worker, never in an HTTP view."""
    n, s = np.indices((max_trials + 1, max_trials + 1))
    valid = (n > 0) & (s <= n)
    table = np.zeros((max_trials + 1, max_trials + 1, 5), dtype=np.float64)
    table[..., 1] = 1
    table[..., 3] = 1
    table[..., 4] = 1
    trials, hits = n[valid], s[valid]
    rates = hits / trials
    for low_index, high_index, alpha in ((0, 1, ALPHA / HYPOTHESES), (2, 3, ALPHA)):
        z = float(norm.isf(alpha / 2))
        den = 1 + z * z / trials
        center = (rates + z * z / (2 * trials)) / den
        width = z * np.sqrt(rates * (1 - rates) / trials + z * z / (4 * trials**2)) / den
        table[..., low_index][valid] = np.maximum(0, center - width)
        table[..., high_index][valid] = np.minimum(1, center + width)
    table[..., 4][valid] = binom.sf(hits - 1, trials, BASELINE)
    return table


def _stat(table, hits, trials):
    hits, trials = int(hits), int(trials)
    low, high, nominal_low, nominal_high, p = map(float, table[trials, hits])
    adjusted = min(1.0, p * HYPOTHESES)
    return dict(
        trigger_count=trials,
        hit_count=hits,
        conditional_rate=hits / trials if trials else None,
        random_baseline=BASELINE,
        delta=hits / trials - BASELINE if trials else None,
        ci_low=low,
        ci_high=high,
        nominal_ci_low=nominal_low,
        nominal_ci_high=nominal_high,
        p_value=p,
        adjusted_p_value=adjusted,
        passed=bool(trials >= MIN_TRIGGERS and adjusted <= ALPHA and low > BASELINE),
    )


def _offset_summary(matrix, steps, lag, start, stop, offset):
    """Pooled counts are descriptive; 20 sources are not 20 independent draws."""
    targets = _targets(steps, lag, start, stop)
    independent = _targets(steps, lag, start, stop, True)
    sources = np.arange(max(0, -offset), min(80, 80 - offset))
    consequences = sources + offset

    def counted(indices):
        before = matrix[indices - lag][:, sources]
        after = matrix[indices][:, consequences]
        per_trial = before.sum(axis=1, dtype=np.int64)
        return int(per_trial.sum()), int((before * after).sum()), per_trial

    trials, hits, _ = counted(targets)
    infer_trials, infer_hits, per_trial = counted(independent)
    rate = infer_hits / infer_trials if infer_trials else None
    # Conditional on the independent source draws, each answer contributes a
    # bounded [0,m_i] count. Hoeffding permits within-answer dependence.
    width = (
        math.sqrt(0.5 * float(np.square(per_trial).sum()) * math.log(2 * HYPOTHESES / ALPHA)) / infer_trials
        if infer_trials else 1.0
    )
    return dict(
        offset=offset,
        direction=_direction(offset),
        trigger_count=trials,
        hit_count=hits,
        conditional_rate=hits / trials if trials else None,
        random_baseline=BASELINE,
        delta=hits / trials - BASELINE if trials else None,
        inferential_trigger_count=infer_trials,
        inferential_hit_count=infer_hits,
        inferential_rate=rate,
        independent_pair_count=len(independent),
        ci_low=max(0.0, rate - width) if rate is not None else 0.0,
        ci_high=min(1.0, rate + width) if rate is not None else 1.0,
        ci_method="非重疊配對、每期聚合計數Hoeffding保守區間；不將同一期20來源視為獨立樣本",
        p_value=None,
        adjusted_p_value=None,
        testing_note="彙總只供描述；不以個別號碼二項檢定推論整體位移優勢",
        independently_validated=False,
        status=NO_EVIDENCE,
    )


def _direction(offset):
    if offset == 0:
        return "原號重開／重複"
    return ("上升" if offset > 0 else "下降") + ("／鄰接連號" if abs(offset) == 1 else "")


def prepare_neighbor_bundle(draws, *, snapshots=SNAPSHOTS):
    """Build bounded numeric tensors for cached, cheap parameter selection.

    A snapshot excludes every row after its source. Windows change only the
    discovery context; all windows share disjoint selection/confirmation dates.
    Statistical tests additionally thin lag pairs so no draw is shared between
    pairs. This avoids treating overlapping temporal observations as independent.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    rows = _rows(draws)
    snapshots = tuple(snapshots)
    if not snapshots or len(snapshots) > 4 or len(set(snapshots)) != len(snapshots) or any(s not in SNAPSHOTS for s in snapshots):
        raise ValueError("僅允許最新、前5、10、20期快照")
    matrix, steps = _matrix(rows), _steps(rows)
    shape = (len(snapshots), len(WINDOWS), len(LAGS))
    hits = np.zeros((*shape, 80, 80), dtype=np.uint16)
    trials = np.zeros((*shape, 80), dtype=np.uint16)
    infer_hits, infer_trials = np.zeros_like(hits), np.zeros_like(trials)
    stage_hits = np.zeros((*shape, 3, 80, 80), dtype=np.uint16)
    stage_trials = np.zeros((*shape, 3, 80), dtype=np.uint16)
    pairs = np.zeros(shape, dtype=np.uint16)
    metadata, summaries = [], {}
    origins = []
    for offset in snapshots:
        stop = len(rows) - offset
        if stop <= 0:
            metadata.append(dict(snapshot=offset, available=False, source_term=None, source_numbers=[]))
            origins.append(None)
            continue
        origin = max(0, stop - 1000)
        n = stop - origin
        discovery_end = origin + n // 2
        selection_end = origin + (n * 3) // 4
        context = rows[origin:stop]
        metadata.append(dict(
            snapshot=offset,
            available=True,
            source_term=rows[stop - 1]["draw_term"],
            source_numbers=rows[stop - 1]["numbers"],
            source_datetime=rows[stop - 1]["draw_datetime"].isoformat(),
            history=_history(context),
            window_histories={str(w): _history(context[-w:]) for w in WINDOWS},
            snapshot_hash=hashlib.sha256(json.dumps(
                [(r["draw_term"], r["draw_datetime"].isoformat(), r["numbers"]) for r in context],
                separators=(",", ":"),
            ).encode()).hexdigest(),
            validation_segments=dict(
                discovery=_history(rows[origin:discovery_end]),
                selection_validation=_history(rows[discovery_end:selection_end]),
                confirmation=_history(rows[selection_end:stop]),
                common_boundaries=True,
                pair_endpoints_disjoint=True,
            ),
            term_gaps=int((~steps[origin:max(origin, stop - 1)]).sum()),
        ))
        origins.append((origin, discovery_end, selection_end, stop))
    for li, lag in enumerate(LAGS):
        prefix_trials, prefix_hits, prefix_pairs = _prefix(matrix, steps, lag)
        for si, boundaries in enumerate(origins):
            if boundaries is None:
                continue
            origin, d_end, s_end, stop = boundaries
            selection = _counts(matrix, steps, lag, d_end, s_end, True)
            confirmation = _counts(matrix, steps, lag, s_end, stop, True)
            for wi, window in enumerate(WINDOWS):
                start = max(origin, stop - window)
                first = min(stop, start + lag)
                hits[si, wi, li] = prefix_hits[stop] - prefix_hits[first]
                trials[si, wi, li] = prefix_trials[stop] - prefix_trials[first]
                pairs[si, wi, li] = prefix_pairs[stop] - prefix_pairs[first]
                inf = _counts(matrix, steps, lag, start, stop, True)
                infer_trials[si, wi, li], infer_hits[si, wi, li] = inf[:2]
                discovery = _counts(matrix, steps, lag, max(origin, d_end - window), d_end, True)
                for stage, result in enumerate((discovery, selection, confirmation)):
                    stage_trials[si, wi, li, stage], stage_hits[si, wi, li, stage] = result[:2]
                summaries[(si, wi, li)] = [_offset_summary(matrix, steps, lag, start, stop, o) for o in OFFSETS]
        del prefix_trials, prefix_hits, prefix_pairs
    max_trials = int(max(trials.max(initial=0), infer_trials.max(initial=0), stage_trials.max(initial=0)))
    lookup = _lookup(max_trials)
    arrays = dict(hits=hits, trials=trials, infer_hits=infer_hits, infer_trials=infer_trials,
                  stage_hits=stage_hits, stage_trials=stage_trials, pairs=pairs, lookup=lookup)
    digest = hashlib.sha256(json.dumps(
        [(r["draw_term"], r["draw_datetime"].isoformat(), r["numbers"]) for r in rows],
        separators=(",", ":"),
    ).encode()).hexdigest()
    return dict(
        version=VERSION,
        windows=WINDOWS, lags=LAGS, offsets=OFFSETS, snapshots=snapshots,
        metadata=metadata, arrays=arrays, offset_summaries=summaries,
        snapshot_hash=digest,
        resources=dict(
            wall_seconds=time.perf_counter() - wall,
            cpu_seconds=time.process_time() - cpu,
            tensor_bytes=sum(a.nbytes for a in arrays.values()),
            peak_prefix_bytes=(len(rows) + 1) * (80 * 80 + 80 + 1) * 2,
            rows=len(rows), max_rows=MAX_ROWS, searched_hypotheses=HYPOTHESES,
            bonferroni_alpha=ALPHA / HYPOTHESES,
        ),
    )


def neighbor_view(bundle, *, window=1000, lag=1, snapshot=0, source=None):
    """Select already computed evidence; no scipy calls or matrix operations."""
    if bundle.get("version") != VERSION:
        raise ValueError("鄰號快取版本無效")
    if any(isinstance(value, (bool, np.bool_)) for value in (window, lag, snapshot)) or window not in WINDOWS or lag not in LAGS or snapshot not in bundle["snapshots"]:
        raise ValueError("不支援的鄰號視窗、跨期距離或快照")
    if source is not None and (isinstance(source, bool) or not isinstance(source, (int, np.integer)) or not 1 <= source <= 80):
        raise ValueError("來源號碼必須在1～80")
    si, wi, li = bundle["snapshots"].index(snapshot), WINDOWS.index(window), LAGS.index(lag)
    meta = bundle["metadata"][si]
    if not meta["available"]:
        return dict(version=VERSION, **meta, by_source=[], offset_statistics=[], warning="歷史資料不足，" + NO_EVIDENCE)
    meta = dict(meta)
    meta["validation_history"] = meta["history"]
    meta["history"] = meta.pop("window_histories")[str(window)]
    arrays = bundle["arrays"]
    lookup = arrays["lookup"]
    active = set(meta["source_numbers"])
    sources = [int(source)] if source is not None else sorted(active)
    by_source = []
    for a in sources:
        ai = a - 1
        consequences = range(1, 81) if source is not None else [a + o for o in OFFSETS if 1 <= a + o <= 80]
        relations = []
        for b in consequences:
            bi = b - 1
            raw = _stat(lookup, arrays["hits"][si, wi, li, ai, bi], arrays["trials"][si, wi, li, ai])
            inference = _stat(lookup, arrays["infer_hits"][si, wi, li, ai, bi], arrays["infer_trials"][si, wi, li, ai])
            stages = {
                name: _stat(lookup, arrays["stage_hits"][si, wi, li, stage, ai, bi], arrays["stage_trials"][si, wi, li, stage, ai])
                for stage, name in enumerate(("discovery", "selection_validation", "confirmation"))
            }
            validated = all(stage["passed"] for stage in stages.values())
            # Raw counts use every valid pair. Inferential intervals/test use
            # non-overlapping pairs; the two sample sizes are exposed explicitly.
            raw.update(
                source=a, target=b, offset=b - a, direction=_direction(b - a),
                ci_low=inference["ci_low"], ci_high=inference["ci_high"],
                nominal_ci_low=inference["nominal_ci_low"], nominal_ci_high=inference["nominal_ci_high"],
                p_value=inference["p_value"], adjusted_p_value=inference["adjusted_p_value"],
                inferential_trigger_count=inference["trigger_count"], inferential_hit_count=inference["hit_count"],
                inferential_rate=inference["conditional_rate"],
                independently_validated=validated,
                status="通過分離樣本的關聯檢定；策略仍待獨立樣本外驗證" if validated else NO_EVIDENCE,
                score_kind="歷史條件頻率，非下一期中獎機率",
                **stages,
            )
            raw.pop("passed")
            relations.append(raw)
        by_source.append(dict(source=a, active=a in active, relations=relations))
    return dict(
        version=VERSION, **meta,
        parameters=dict(window=window, lag=lag, source=source),
        supported_windows=list(WINDOWS), supported_lags=list(LAGS), supported_offsets=list(OFFSETS),
        supported_snapshots=list(SNAPSHOTS),
        by_source=by_source,
        offset_statistics=bundle["offset_summaries"][(si, wi, li)],
        valid_pair_count=int(arrays["pairs"][si, wi, li]),
        validation=dict(
            min_triggers=MIN_TRIGGERS, family_hypotheses=HYPOTHESES,
            correction="Bonferroni：9視窗×6距離×80來源×80後果×4快照；不只校正顯示的鄰號",
            confidence_level=1 - ALPHA,
            ci_method="不共享開獎資料的配對；各段內Wilson家族同時區間；各驗證段兩端都在各自區間內",
            independent_pair_stride=lag + 1,
            descriptive_warning="歷史全配對條件頻率與獨立配對檢定樣本不同；皆不是下一期中獎機率",
        ),
        resources=bundle["resources"],
        warning=NO_EVIDENCE,
        allowed_for_live=False,
        shadow_only=True,
    )


def neighbor_analysis(draws, *, window=1000, lag=1, source_term=None, include_all=False, source=None):
    """Convenience entry point for offline inspection of one causal snapshot."""
    rows = _rows(draws, source_term=source_term)
    bundle = prepare_neighbor_bundle(rows, snapshots=(0,))
    if include_all:
        view = neighbor_view(bundle, window=window, lag=lag, source=1)
        view["by_source"] = [neighbor_view(bundle, window=window, lag=lag, source=a)["by_source"][0] for a in range(1, 81)]
        view["parameters"]["source"] = None
        view["parameters"]["include_all"] = True
        return view
    return neighbor_view(bundle, window=window, lag=lag, source=source)


def neighbor_candidate(draws):
    """Predeclared fixed research candidate; never picks a winning subperiod.

    Beta(2,6) smooths the historical counts. The mean over active neighbouring
    sources is converted into relative scores, not a calibrated probability.
    No hypothesis evidence or holdout label selects this configuration.
    """
    rows = _rows(draws)[-1000:]
    matrix, steps = _matrix(rows), _steps(rows)
    trials, hits, pair_count = _counts(matrix, steps, 1, 0, len(rows))
    active = np.flatnonzero(matrix[-1])
    target = np.arange(80)
    allowed = np.isin(target[None, :] - active[:, None], OFFSETS)
    rates = (hits[active].astype(float) + 2) / (trials[active, None] + 8)
    contributors = allowed.sum(axis=0)
    raw = np.divide((rates * allowed).sum(axis=0), contributors,
                    out=np.full(80, BASELINE), where=contributors > 0)
    span = float(np.ptp(raw))
    scores = (raw - raw.min()) / span if span > np.finfo(float).eps else np.full(80, 0.5)
    ranking = (np.lexsort((target, -scores)) + 1).tolist()
    config = dict(window=1000, lag=1, offsets=list(OFFSETS), prior_alpha=2, prior_beta=6,
                  aggregation="當期20來源中符合預先指定位移者的平滑條件頻率平均；無來源時0.25")
    by_star = {
        str(k): dict(numbers=ranking[:k], ranking=ranking, scores=scores.tolist(),
                     score_kind=SCORE_KIND, shadow_only=True, strategy_proven=False)
        for k in (3, 4, 5)
    }
    return dict(
        version=VERSION, source_term=rows[-1]["draw_term"], source_numbers=rows[-1]["numbers"],
        history=_history(rows), config=config, scores=scores.tolist(), ranking=ranking,
        by_star=by_star, valid_pair_count=pair_count, term_gaps=int((~steps).sum()),
        score_kind=SCORE_KIND, warning=NO_EVIDENCE,
        shadow_only=True, allowed_for_live=False, strategy_proven=False,
    )
