"""Validated profile artifacts trained only in the bounded background worker.

Inference never fits a challenger. Missing, mismatched or future artifacts leave
its previous champion active until a matching causal artifact becomes ready.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import time

from taiwan_lottery.ml.bingo_nested import CandidateProfile, profile_scores

_CACHE = {}


def canonical_profile(profile):
    if isinstance(profile, CandidateProfile):
        return profile
    allowed = {field.name for field in fields(CandidateProfile)}
    return CandidateProfile(
        **{key: value for key, value in profile.items() if key in allowed}
    )


def profile_key(profile):
    value = canonical_profile(profile).payload()
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train_validated_profiles(
    draws, directory, profiles, threads=2, progress=None, *, enabled=None
):
    from threadpoolctl import threadpool_limits
    from taiwan_lottery.ml.bingo_models import ModelSuite
    from taiwan_lottery.services.bingo_workers import _publish, _atomic, snapshot_hash

    unique = {profile_key(p): canonical_profile(p) for p in profiles}
    if len(unique) > 3:
        raise ValueError("每次最多準備三個已驗證策略工件")
    results = []
    with threadpool_limits(limits=threads):
        for key, profile in sorted(unique.items()):
            snapshot = draws[-profile.train_window :]
            if len(snapshot) < 100:
                results.append(
                    dict(
                        profile_key=key,
                        status="pending",
                        reason="候選訓練至少需要100期",
                    )
                )
                continue
            path = Path(directory) / "profiles" / key
            try:
                previous = json.loads((path / "current.json").read_text())
                indexes = [
                    i
                    for i, row in enumerate(draws)
                    if row["draw_term"] == previous["source_term"]
                ]
                if indexes and len(draws) - indexes[0] - 1 < profile.retrain_every:
                    load_profile_suite(profile, draws, directory)
                    results.append(
                        dict(
                            profile_key=key,
                            status="ready",
                            reused=True,
                            version=previous["version"],
                            training_cutoff=previous["source_term"],
                        )
                    )
                    continue
            except (FileNotFoundError, ValueError, KeyError, OSError):
                pass
            history = [row["numbers"] for row in snapshot]
            suite = ModelSuite(
                history, threads=threads, window=profile.train_window, enabled=enabled
            )

            def update(name, done, total, status):
                if status == "ready":
                    suite.records[name].update(
                        training_source_term=snapshot[-1]["draw_term"],
                        training_history_start=snapshot[0]["draw_term"],
                        training_history_count=len(snapshot),
                        profile_key=key,
                    )
                if progress:
                    progress(f"{profile.id}/{name}", done, total, status)

            started = time.perf_counter()
            suite.fit(progress=update)
            if not suite.scores(history):
                results.append(
                    dict(profile_key=key, status="failed", reason="候選無有效模型")
                )
                continue
            manifest = _publish(suite, snapshot, path, "profile-" + key[:16])
            manifest.update(
                profile=profile.payload(),
                profile_key=key,
                training_seconds=time.perf_counter() - started,
            )
            _atomic(
                path / "current.json", json.dumps(manifest, ensure_ascii=False).encode()
            )
            _CACHE.pop(str(path.resolve()), None)
            results.append(
                dict(
                    profile_key=key,
                    status="ready",
                    version=manifest["version"],
                    training_cutoff=snapshot[-1]["draw_term"],
                    snapshot_hash=snapshot_hash(snapshot),
                    training_seconds=manifest["training_seconds"],
                )
            )
    return dict(
        profiles=results, trained_count=sum(r["status"] == "ready" for r in results)
    )


def load_profile_suite(profile, draws, directory):
    from taiwan_lottery.services.bingo_workers import (
        _load_checkpoint,
        _publish,
        _cutoff,
    )

    profile = canonical_profile(profile)
    key = profile_key(profile)
    path = Path(directory) / "profiles" / key
    manifest = json.loads((path / "current.json").read_text())
    if (
        manifest.get("profile_key") != key
        or manifest.get("profile") != profile.payload()
    ):
        raise ValueError("候選工件與已驗證參數不一致")
    cache_key = str(path.resolve())
    cached = _CACHE.get(cache_key)
    suite = None
    if cached and cached["base"] == manifest["sha256"]:
        try:
            index = _cutoff(cached["online"], draws)
            prefix = [d["numbers"] for d in draws[: index + 1]]
            overlap = min(len(prefix), len(cached["suite"].history))
            if cached["suite"].history[-overlap:] != prefix[-overlap:]:
                raise ValueError("候選快取資料不一致")
            suite = cached["suite"]
        except ValueError:
            pass
    if suite is None:
        candidate = manifest
        online = path / "online.json"
        if online.exists():
            data = json.loads(online.read_text())
            if data.get("base_sha256") == manifest["sha256"]:
                try:
                    _cutoff(data, draws)
                    candidate = data
                except ValueError:
                    pass
        try:
            suite, index = _load_checkpoint(candidate, draws, path)
        except (ValueError, OSError, KeyError):
            suite, index = _load_checkpoint(manifest, draws, path)
    changed = index < len(draws) - 1
    for draw in draws[index + 1 :]:
        previous = suite.history
        suite.update(previous, draw["numbers"])
        suite.history = (previous + [list(draw["numbers"])])[-suite.window :]
    if (
        cached
        and cached["base"] == manifest["sha256"]
        and int(cached["online"]["source_term"]) > int(draws[-1]["draw_term"])
    ):
        return suite, dict(manifest, artifact_directory=str(path))
    checkpoint = (
        _publish(
            suite,
            draws,
            path,
            manifest["version"],
            online=True,
            base_sha256=manifest["sha256"],
        )
        if changed
        else (cached["online"] if cached and suite is cached["suite"] else manifest)
    )
    _CACHE[cache_key] = dict(suite=suite, online=checkpoint, base=manifest["sha256"])
    return suite, dict(manifest, artifact_directory=str(path))


def score_validated_profile(
    profile, draws, directory, threads=2, *, context_window=None
):
    try:
        profile = canonical_profile(profile)
        suite, manifest = load_profile_suite(profile, draws, directory)
        active = (
            replace(profile, context_window=context_window)
            if context_window is not None
            else profile
        )
        started = time.perf_counter()
        scores = profile_scores(suite, [r["numbers"] for r in draws], active)
        import numpy as np
        from taiwan_lottery.ml.bingo_models import features

        count = min(len(draws), active.context_window, active.train_window)
        known = [r["numbers"] for r in draws[-count:]]
        feature_hash = hashlib.sha256(
            np.ascontiguousarray(features(known)).tobytes()
        ).hexdigest()
        input_hash = hashlib.sha256(
            np.asarray(known, dtype=np.uint8).tobytes()
        ).hexdigest()
        return dict(
            status="ready",
            scores=scores,
            version=manifest["version"] + "-" + manifest["sha256"][:12],
            metadata=dict(
                training_cutoff=manifest["source_term"],
                training_history_start=manifest.get("training_history_start"),
                training_history_count=manifest.get("training_samples"),
                profile_key=profile_key(profile),
                profile=profile.payload(),
                history_count=count,
                history_start=draws[-count]["draw_term"],
                history_end=draws[-1]["draw_term"],
                feature_hash=feature_hash,
                input_hash=input_hash,
                scoring_cost_seconds=time.perf_counter() - started,
            ),
            models=suite.statuses(),
        )
    except FileNotFoundError:
        return dict(status="pending", scores={}, reason="候選工件尚未完成背景訓練")
    except Exception as exc:
        return dict(
            status="failed",
            scores={},
            reason=f"候選工件無法使用：{type(exc).__name__}: {exc}",
        )
