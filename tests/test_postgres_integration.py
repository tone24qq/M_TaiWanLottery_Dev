"""Optional genuine PostgreSQL integration, no Docker/existing DB required.

Install pgserver==0.1.4 on Linux to run; PostgreSQL binaries live in the wheel.
All temporary database files and the process are removed after the test.
"""

import asyncio
from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import tempfile

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from taiwan_lottery.db.crud.bingo_research import ResearchStore
from taiwan_lottery.services.runtime_lock import RuntimeLock


def test_real_postgres_migrations_lock_and_ledger():
    pgserver = pytest.importorskip(
        "pgserver", reason="選用 PostgreSQL 整合需要 pgserver==0.1.4"
    )
    Path("work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir="work", prefix="postgres-validation-"
    ) as directory:
        server = pgserver.get_server(Path(directory) / "pgdata", cleanup_mode="delete")
        uri = server.get_uri()
        sync_uri = uri.replace("postgresql://", "postgresql+psycopg2://")
        async_uri = uri.replace("postgresql://", "postgresql+asyncpg://")
        try:
            result = subprocess.run(
                [__import__("sys").executable, "-m", "alembic", "upgrade", "head"],
                env={**os.environ, "DATABASE_URL_SYNC": sync_uri},
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            asyncio.run(_ledger(async_uri, directory))
        finally:
            server.cleanup()


async def _ledger(uri, directory):
    engine = create_async_engine(uri)
    store = ResearchStore(async_sessionmaker(engine, expire_on_commit=False))
    try:
        await store.init_schema()
        lock = RuntimeLock(engine, directory)
        second = RuntimeLock(engine, directory)
        await lock.acquire()
        try:
            with pytest.raises(RuntimeError):
                await second.acquire()
        finally:
            await lock.close()
        at = datetime(2026, 11, 3, 10)
        source = dict(
            draw_term="115001000", draw_datetime=at, numbers=list(range(1, 21))
        )
        assert await store.ingest([source]) == 1
        assert await store.ingest([source]) == 0
        forecast = dict(
            model="frequency", stars=3, numbers=[1, 2, 3], scores=[0.5] * 80
        )
        payload = await store.archive(
            "115001001",
            at + timedelta(minutes=5),
            "115001000",
            at,
            {},
            [forecast],
            now=at + timedelta(seconds=1),
        )
        assert not payload["settled"]
        await store.ingest(
            [
                {
                    **source,
                    "draw_term": "115001001",
                    "draw_datetime": at + timedelta(minutes=5),
                }
            ]
        )
        assert await store.settle(now=at + timedelta(minutes=6)) == 1
        assert await store.settle(now=at + timedelta(minutes=6)) == 0
        assert (await store.bundle("115001001"))["settled"]
        summary = await store.outcome_summary()
        assert summary["cumulative"][0]["count"] == 1
        assert summary["consecutive"][0]["count"] == 1
    finally:
        await engine.dispose()
