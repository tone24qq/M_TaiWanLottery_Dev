"""Real score functions and SQL test the whole phase-two research feedback loop."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import copy

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from taiwan_lottery.bingo_clock import TAIPEI
from taiwan_lottery.config import settings
from taiwan_lottery.ml.bingo_models import ModelSuite
from taiwan_lottery.ml.bingo_contexts import context_scores
from taiwan_lottery.services.bingo_workers import infer_today, snapshot_hash
from taiwan_lottery.services.bingo_runtime import BingoRuntime


class Clock:
    when = datetime(2026, 11, 3, 7, 50, 10, tzinfo=TAIPEI)

    def __call__(self):
        return self.when


class Source:
    def __init__(self, rows):
        self.rows = rows

    async def fetch_day(self, day):
        return copy.deepcopy([r for r in self.rows if r["draw_datetime"].date() == day])


class Scorer:
    adaptive_scoring = True
    busy = False
    process = None
    active_kind = None

    def __init__(self, directory):
        self.directory = str(directory)
        self.models = []
        self.training = {"state": "waiting"}
        self.backtest = {"state": "waiting"}

    async def score(self, rows, plan=None):
        suite = ModelSuite(
            [r["numbers"] for r in rows], enabled=["frequency", "cdm"]
        ).fit()
        result = context_scores(suite, rows, plan, directory=self.directory)
        result.update(
            snapshot_hash=snapshot_hash(rows),
            version="real-score-test",
            models=suite.statuses(),
            training_cutoff=rows[-1]["draw_term"],
        )
        result["today"] = infer_today(rows, self.directory, 1, plan["target_date"])
        from taiwan_lottery.ml.bingo_associations import association_candidates

        result["associations"] = association_candidates(rows, result["by_star_scores"])
        self.models = suite.statuses()
        return result

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_all_three_families_archive_settle_learn_and_restore_without_fake_backfill(
    tmp_path,
):
    day = datetime(2026, 11, 2, 7, 5)
    rows = [
        dict(
            draw_term=str(115000001 + i),
            draw_datetime=day + timedelta(minutes=5 * i),
            numbers=list(range(1, 21)),
        )
        for i in range(203)
    ]
    rows += [
        dict(
            draw_term=str(115000204 + i),
            draw_datetime=datetime(2026, 11, 3, 7, 5) + timedelta(minutes=5 * i),
            numbers=list(range(41, 61)),
        )
        for i in range(10)
    ]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    clock = Clock()
    source = Source(rows)
    scorer = Scorer(tmp_path / "models")
    with patch.multiple(
        settings,
        SCRAPER_ENABLED=False,
        AUTO_TRAIN=False,
        AUTO_VALIDATE=False,
        HISTORY_MIN_DRAWS=10,
    ):
        runtime = BingoRuntime(sessions, source=source, clock=clock, workers=scorer)
        try:
            await runtime.initialize()
            await runtime.cycle()
            first = copy.deepcopy(runtime.prediction)
            assert len(first["research_strategies"]) == 3
            assert all(len(v) == 3 for v in first["research_strategies"].values())
            assert first["today_status"]["history_dates"] == ["2026-11-03"]
            assert first["today_status"]["history_count"] == 10
            assert first["today_status"]["history_start"] == "115000204"
            assert set(first["associations"]["by_star"]) == {"3", "4", "5"}
            assert all(
                set(v) == {"A", "B", "C"}
                for v in first["associations"]["by_star"].values()
            )
            frozen_associations = copy.deepcopy(first["associations"])
            assert (await runtime.predict()) == first
            assert runtime.associations()["by_star"] == frozen_associations["by_star"]
            import httpx
            from taiwan_lottery.main import app

            previous_runtime = getattr(app.state, "bingo", None)
            app.state.bingo = runtime
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.get("/api/v1/bingo/associations")
                assert response.status_code == 200
                assert response.json()["by_star"] == frozen_associations["by_star"]
                assert response.json()["state"] == "sealed"
            finally:
                if previous_runtime is None:
                    del app.state.bingo
                else:
                    app.state.bingo = previous_runtime
            target = runtime.target["draw_term"]
            rows.append(
                dict(
                    draw_term=target,
                    draw_datetime=datetime(2026, 11, 3, 7, 55),
                    numbers=list(range(41, 61)),
                )
            )
            clock.when = datetime(2026, 11, 3, 7, 55, 10, tzinfo=TAIPEI)
            await runtime.cycle()
            assert runtime.families.state["observations"] == 1
            assert runtime.association_tracker.state["observations"] == 1
            assert all(
                row["samples"] == 1
                for board in runtime.associations()["leaderboards"].values()
                for row in board["cumulative"]
            )
            assert runtime.performance.status()["last_report"]["target_term"] == target
            for star in ("3", "4", "5"):
                group = runtime.families.status()["stars"][star]
                assert (
                    group["paired_samples"] == 1 and group["champion"] == "historical"
                )
                assert {r["samples"] for r in group["leaderboard"]} == {1}
            audit = await runtime.store.forecast_audit(target)
            assert audit
            original = copy.deepcopy(runtime.prediction)
            await runtime.cycle()
            assert runtime.families.state["observations"] == 1
            restored = BingoRuntime(
                sessions,
                source=source,
                clock=clock,
                workers=Scorer(tmp_path / "models"),
            )
            await restored.initialize()
            assert restored.families.status() == runtime.families.status()
            assert (
                restored.association_tracker.status()
                == runtime.association_tracker.status()
            )
            assert restored.prediction == original
            await restored.close()
        finally:
            await runtime.close()
            await engine.dispose()
