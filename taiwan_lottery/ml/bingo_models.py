"""Bounded, deterministic 80-number scorers for the Bingo research pipeline.

Outputs are *relative scores*, not calibrated probabilities of a win. Every
supervised row for draw t is computed from draws strictly before t. Artifacts are
pickleable; optional libraries are imported only inside their model's fit path.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable

import numpy as np

MODEL_NAMES = (
    "frequency",
    "cdm",
    "river",
    "lightgbm",
    "catboost",
    "transformer",
    "dqn",
    "ensemble",
)
VERSION = "bingo-causal-1"
WINDOWS = (5, 20, 60, 200)
FEATURE_NAMES = (
    "frequency5",
    "frequency20",
    "frequency60",
    "frequency200",
    "gap",
    "momentum",
    "last",
    "number",
    "sin",
    "cos",
)


def draw_matrix(history: list[list[int]]) -> np.ndarray:
    """Validate source records rather than silently accepting corrupt labels."""
    matrix = np.zeros((len(history), 80), dtype=np.float32)
    for i, draw in enumerate(history):
        if (
            len(draw) != 20
            or len(set(draw)) != 20
            or any(
                isinstance(n, bool)
                or not isinstance(n, (int, np.integer))
                or not 1 <= n <= 80
                for n in draw
            )
        ):
            raise ValueError(f"第 {i} 筆資料必須包含 20 個不重複的 1～80 整數")
        matrix[i, np.asarray(draw, dtype=int) - 1] = 1
    return matrix


def normalize_scores(values) -> np.ndarray:
    """Map to [0, 1], preserving score spacing; these are not probabilities.

    Equal fusion therefore preserves magnitude information, whereas Borda uses
    only ranks. Constant outputs are neutral (.5), never random perturbations.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (80,) or not np.isfinite(values).all():
        raise ValueError("模型必須輸出 80 個有限分數")
    low, high = float(values.min()), float(values.max())
    if high - low <= np.finfo(float).eps * max(1, abs(high), abs(low)):
        return np.full(80, 0.5)
    return (values - low) / (high - low)


def _features_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Features at all t (including next prediction), using a prefix scan."""
    n = len(matrix)
    cumulative = np.concatenate([np.zeros((1, 80)), np.cumsum(matrix, axis=0)], axis=0)
    times = np.arange(n + 1)
    result = np.zeros((n + 1, 80, len(FEATURE_NAMES)), dtype=np.float32)
    for column, window in enumerate(WINDOWS):
        starts = np.maximum(0, times - window)
        result[:, :, column] = (cumulative[times] - cumulative[starts]) / np.maximum(
            1, times - starts
        )[:, None]
    # Last appearance before t. At t=0 no draw has been observed.
    last_seen = (
        np.maximum.accumulate(np.where(matrix > 0, np.arange(n)[:, None], -1), axis=0)
        if n
        else np.empty((0, 80))
    )
    last_before = np.concatenate([np.full((1, 80), -1), last_seen], axis=0)
    result[:, :, 4] = np.minimum((times[:, None] - 1 - last_before) / 20, 2)
    result[:, :, 5] = result[:, :, 0] - result[:, :, 1]
    if n:
        result[1:, :, 6] = matrix
    numbers = np.arange(1, 81)
    result[:, :, 7] = numbers / 80
    result[:, :, 8] = np.sin(2 * np.pi * numbers / 80)
    result[:, :, 9] = np.cos(2 * np.pi * numbers / 80)
    return result


def latest_features(matrix: np.ndarray) -> np.ndarray:
    """Compute just the next causal feature row, without an O(n*80*10) tensor.

    Binary counts are exact in float32 at the supported history limit. Casting
    each frequency and momentum at the same point as the original training
    tensor preserves scores, ranking ties and serialized model semantics.
    """
    n = len(matrix)
    row = np.zeros((80, len(FEATURE_NAMES)), dtype=np.float32)
    for column, window in enumerate(WINDOWS):
        count = min(n, window)
        if count:
            row[:, column] = matrix[-count:].sum(axis=0) / count
    if n:
        seen = matrix.any(axis=0)
        reverse_index = matrix[::-1].argmax(axis=0)
        row[:, 4] = np.minimum(np.where(seen, reverse_index, n) / 20, 2)
        row[:, 6] = matrix[-1]
    row[:, 5] = row[:, 0] - row[:, 1]
    numbers = np.arange(1, 81)
    row[:, 7] = numbers / 80
    row[:, 8] = np.sin(2 * np.pi * numbers / 80)
    row[:, 9] = np.cos(2 * np.pi * numbers / 80)
    return row


class FeatureHistory:
    """Read-only reusable binary/prefix snapshot for causal fold calculations.

    ``index`` is an exclusive target boundary. Rows after it cannot affect a
    result. A clipped window resets unseen-number gaps exactly like constructing
    a fresh history slice, including when an appearance predates that slice.
    """

    def __init__(self, history: list[list[int]]):
        self.matrix = draw_matrix(history)
        self.matrix.flags.writeable = False
        self.prefix = np.concatenate(
            [np.zeros((1, 80)), np.cumsum(self.matrix, axis=0)], axis=0
        )
        n = len(self.matrix)
        self.last_seen = (
            np.maximum.accumulate(
                np.where(self.matrix > 0, np.arange(n)[:, None], -1), axis=0
            )
            if n
            else np.empty((0, 80), dtype=int)
        )
        self.prefix.flags.writeable = False
        self.last_seen.flags.writeable = False

    def _bounds(self, index, window):
        index = len(self.matrix) if index is None else index
        if not isinstance(index, (int, np.integer)) or not 0 <= index <= len(
            self.matrix
        ):
            raise ValueError("特徵目標邊界超出資料範圍")
        if window is not None and (
            not isinstance(window, (int, np.integer)) or window < 1
        ):
            raise ValueError("特徵視窗必須為正整數")
        return max(0, index - window) if window is not None else 0, index

    def matrix_before(self, index=None, window=None):
        start, index = self._bounds(index, window)
        return self.matrix[start:index]

    def row(self, index=None, window=None):
        start, index = self._bounds(index, window)
        row = np.zeros((80, len(FEATURE_NAMES)), dtype=np.float32)
        for column, width in enumerate(WINDOWS):
            begin = max(start, index - width)
            row[:, column] = (self.prefix[index] - self.prefix[begin]) / max(
                1, index - begin
            )
        if index > start:
            last = np.maximum(self.last_seen[index - 1], start - 1)
            row[:, 4] = np.minimum((index - 1 - last) / 20, 2)
            row[:, 6] = self.matrix[index - 1]
        row[:, 5] = row[:, 0] - row[:, 1]
        numbers = np.arange(1, 81)
        row[:, 7] = numbers / 80
        row[:, 8] = np.sin(2 * np.pi * numbers / 80)
        row[:, 9] = np.cos(2 * np.pi * numbers / 80)
        return row


def features(history: list[list[int]]) -> np.ndarray:
    return latest_features(draw_matrix(history))


def supervised_samples(
    history: list[list[int]], min_context: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    matrix = draw_matrix(history)
    return _features_from_matrix(matrix)[min_context:-1], matrix[min_context:]


def _digest(history) -> str:
    return hashlib.sha256(np.asarray(history, dtype=np.uint8).tobytes()).hexdigest()


class ModelSuite:
    """Independent scorers trained from one immutable, validated history snapshot.

    Frequency/CDM are vectorized versions of the existing frequency/gap/momentum
    and posterior formulas. They have no fitted future-dependent parameter and
    can refresh cheaply at each new draw. Heavy models retain their scheduled
    training weights while features adapt to the current validated snapshot.
    """

    def __init__(
        self,
        history: list[list[int]],
        threads: int = 2,
        window: int = 1000,
        enabled: list[str] | None = None,
    ):
        self.window = max(100, min(int(window), 10000))
        self.history = [list(draw) for draw in history[-self.window :]]
        draw_matrix(self.history)
        self.threads = max(1, min(int(threads), 4))
        requested = MODEL_NAMES if enabled is None else enabled
        if any(name not in MODEL_NAMES for name in requested):
            raise ValueError("未知模型")
        # Canonical order prevents an explicitly selected Ensemble from fitting
        # before its selected sources, and deduplicates repeated model names.
        self.enabled = [name for name in MODEL_NAMES if name in requested]
        self.models: dict = {}
        self.records = {
            name: {
                "name": name,
                "status": "waiting" if name in self.enabled else "disabled",
                "samples": 0,
                "version": VERSION,
                "error": None,
                "completed": 0,
                "total": 0,
                "percent": 0,
                "updated_at": None,
            }
            for name in MODEL_NAMES
        }
        self.history_digest = _digest(self.history)
        self.river_seen = self.history_digest
        self.feature_importance: dict = {}

    def _report(self, callback, name, done, total, status):
        record = self.records[name]
        record.update(
            status=status,
            completed=done,
            total=total,
            percent=round(100 * done / total, 2) if total else 0,
        )
        if callback:
            callback(name, done, total, status)

    def fit(self, progress: Callable | None = None) -> "ModelSuite":
        history = self.history
        x, y = supervised_samples(history)
        matrix = draw_matrix(history)
        minima = {
            "frequency": 10,
            "cdm": 10,
            "river": 32,
            "lightgbm": 80,
            "catboost": 80,
            "transformer": 96,
            "dqn": 96,
            "ensemble": 10,
        }
        for name in self.enabled:
            if len(history) < minima[name]:
                self.records[name].update(
                    status="waiting", error=f"至少需要 {minima[name]} 期有效資料"
                )
                continue
            self._report(progress, name, 0, 1, "training")
            try:
                if name in ("frequency", "cdm"):
                    self.models[name] = True
                elif name == "river":
                    from river import linear_model, optim

                    learners = [
                        linear_model.LogisticRegression(
                            optimizer=optim.SGD(0.025), l2=0.005
                        )
                        for _ in range(80)
                    ]
                    for t, (row, label) in enumerate(zip(x, y), 1):
                        for number, learner in enumerate(learners):
                            learner.learn_one(
                                dict(zip(FEATURE_NAMES, map(float, row[number]))),
                                bool(label[number]),
                            )
                        if t % 20 == 0 or t == len(x):
                            self._report(progress, name, t, len(x), "training")
                    self.models[name] = learners
                elif name == "lightgbm":
                    import lightgbm as lgb

                    def completed_round(env):
                        self._report(progress, name, env.iteration + 1, 40, "training")

                    model = lgb.LGBMClassifier(
                        n_estimators=40,
                        num_leaves=15,
                        max_depth=5,
                        learning_rate=0.05,
                        min_child_samples=80,
                        n_jobs=self.threads,
                        deterministic=True,
                        force_col_wise=True,
                        random_state=17,
                        verbosity=-1,
                    )
                    model.fit(
                        x.reshape(-1, x.shape[-1]),
                        y.reshape(-1).astype(int),
                        callbacks=[completed_round],
                    )
                    self.models[name] = model
                    self.feature_importance[name] = dict(
                        zip(FEATURE_NAMES, map(float, model.feature_importances_))
                    )
                elif name == "catboost":
                    from catboost import CatBoostClassifier

                    suite = self

                    class Progress:
                        def after_iteration(self, info):
                            suite._report(
                                progress, name, info.iteration, 40, "training"
                            )
                            return True

                    model = CatBoostClassifier(
                        iterations=40,
                        depth=4,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        thread_count=self.threads,
                        random_seed=17,
                        verbose=False,
                        allow_writing_files=False,
                    )
                    model.fit(
                        x.reshape(-1, x.shape[-1]),
                        y.reshape(-1).astype(int),
                        callbacks=[Progress()],
                    )
                    self.models[name] = model
                    self.feature_importance[name] = dict(
                        zip(FEATURE_NAMES, map(float, model.feature_importances_))
                    )
                elif name in ("transformer", "dqn"):
                    from taiwan_lottery.ml.bingo_neural import fit_neural

                    self.models[name] = fit_neural(
                        name,
                        matrix,
                        x,
                        y,
                        self.threads,
                        lambda done, total: self._report(
                            progress, name, done, total, "training"
                        ),
                    )
                elif name == "ensemble":
                    if len(self.models) < 2:
                        raise ValueError("至少需要兩個已訓練基礎模型")
                    self.models[name] = True
                samples = len(history) if name in ("frequency", "cdm") else len(x)
                if name == "dqn":
                    samples = max(0, len(x) - 1)
                if name == "ensemble":
                    samples = min(
                        self.records[source]["samples"]
                        for source in self.models
                        if source != "ensemble"
                    )
                self.records[name].update(
                    samples=samples,
                    error=None,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                # Validate the entire 80-element output before admitting a scorer.
                if name != "ensemble":
                    normalize_scores(
                        self._score_one(
                            name, history, matrix, _features_from_matrix(matrix)[-1]
                        )
                    )
                total = max(1, self.records[name]["total"])
                self._report(progress, name, total, total, "ready")
            except ImportError as exc:
                self.models.pop(name, None)
                self.records[name].update(error=f"缺少選用依賴：{exc.name}")
                self._report(progress, name, 0, 0, "disabled")
            except Exception as exc:
                self.models.pop(name, None)
                self.records[name].update(error=f"{type(exc).__name__}: {exc}")
                self._report(progress, name, 0, 0, "failed")
        return self

    def _score_one(self, name, history, matrix, row):
        if name == "frequency":
            # Existing frequency + gap + short momentum, without the former
            # full-history auto-tuning leakage or pair-combination traversal.
            rate = row[:, :4] @ np.array([0.35, 0.30, 0.20, 0.15])
            momentum = row[:, 5]
            gap = row[:, 4]
            return 0.7 * rate + 0.2 * momentum + 0.025 * gap
        if name == "cdm":
            # Same posterior predictive idea as CDMModel: a uniform Dirichlet
            # prior shrinks finite-sample frequency toward the random baseline.
            count = matrix.sum(axis=0)
            return (count + 8 * 0.25) / (len(matrix) + 8)
        if name == "river":
            return np.array(
                [
                    learner.predict_proba_one(
                        dict(zip(FEATURE_NAMES, map(float, row[i])))
                    ).get(True, 0.5)
                    for i, learner in enumerate(self.models[name])
                ]
            )
        if name == "catboost":
            return self.models[name].predict_proba(row, thread_count=self.threads)[:, 1]
        if name == "lightgbm":
            return self.models[name].predict_proba(row)[:, 1]
        if name in ("transformer", "dqn"):
            from taiwan_lottery.ml.bingo_neural import neural_scores

            return neural_scores(name, self.models[name], matrix, row)
        raise ValueError(name)

    def scores(self, history: list[list[int]]) -> dict[str, list[float]]:
        history = history[-self.window :]
        matrix = draw_matrix(history)
        if not len(matrix):
            return {}
        return self.scores_prepared(matrix)

    def scores_prepared(
        self, matrix: np.ndarray, row: np.ndarray | None = None
    ) -> dict[str, list[float]]:
        """Score a validated shared snapshot; no history rescan or feature tensor.

        Public ``scores`` remains the validating boundary. Walk-forward and the
        runtime may reuse ``FeatureHistory`` matrix views for this trusted path.
        """
        matrix = matrix[-self.window :]
        if not len(matrix):
            return {}
        if row is None:
            row = latest_features(matrix)
        if row.shape != (80, len(FEATURE_NAMES)) or not np.isfinite(row).all():
            raise ValueError("模型特徵必須為 80 號完整有限矩陣")
        history = None
        result = {}
        for name in self.enabled:
            if name not in self.models or name == "ensemble":
                continue
            try:
                result[name] = normalize_scores(
                    self._score_one(name, history, matrix, row)
                ).tolist()
            except Exception as exc:
                self.records[name].update(
                    status="failed", error=f"評分失敗：{type(exc).__name__}: {exc}"
                )
        if "ensemble" in self.models and len(result) >= 2:
            result["ensemble"] = normalize_scores(
                np.mean(list(result.values()), axis=0)
            ).tolist()
        return result

    def update(
        self,
        previous_history: list[list[int]],
        new_draw: list[int],
        *,
        feature_row=None,
    ):
        """Learn exactly one newly observed draw; duplicate calls are idempotent.

        The caller supplies the history ending immediately before new_draw.
        Batch catch-up calls this in chronological order after settlement.
        """
        previous = previous_history[-self.window :]
        draw_matrix([new_draw])
        after = (previous + [list(new_draw)])[-self.window :]
        after_digest = _digest(after)
        if after_digest == self.river_seen:
            return
        if "river" in self.models:
            if _digest(previous) != self.river_seen:
                # Training worker may finish behind live data. Never silently
                # skip unknown labels or double-count an overlapping snapshot.
                self.records["river"].update(
                    status="waiting", error="線上快照不連續，等待從完整快照重新訓練"
                )
                self.models.pop("river", None)
            else:
                row = features(previous) if feature_row is None else feature_row
                for number, learner in enumerate(self.models["river"]):
                    learner.learn_one(
                        dict(zip(FEATURE_NAMES, map(float, row[number]))),
                        number + 1 in new_draw,
                    )
                self.records["river"]["samples"] += 1
                self.records["river"]["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
        self.river_seen = after_digest
        for name in ("frequency", "cdm"):
            if name in self.models:
                self.records[name].update(
                    samples=len(after),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
        if "ensemble" in self.models:
            self.records["ensemble"].update(
                samples=min(
                    self.records[source]["samples"]
                    for source in self.models
                    if source != "ensemble"
                ),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    def statuses(self) -> list[dict]:
        return [dict(self.records[name]) for name in MODEL_NAMES]
