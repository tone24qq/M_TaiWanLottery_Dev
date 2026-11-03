"""Fail fast if another API process owns the same runtime storage.

Single scheduler/worker owner per database. PostgreSQL uses an advisory session
lock; SQLite uses an OS lock, automatically released after a crash.
"""

from pathlib import Path
from sqlalchemy import text


class RuntimeLock:
    def __init__(self, engine, artifacts):
        self.engine = engine
        self.directory = Path(artifacts)
        self.connection = None
        self.handle = None

    async def acquire(self):
        if self.engine.dialect.name == "postgresql":
            self.connection = await self.engine.connect()
            acquired = await self.connection.scalar(
                text("SELECT pg_try_advisory_lock(741169542105)")
            )
            if not acquired:
                await self.connection.close()
                self.connection = None
                raise RuntimeError(
                    "此資料庫已有賓果主程序執行，請使用單一 Uvicorn worker"
                )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Lock database path so different artifact paths cannot duplicate it.
            database = self.engine.url.database
            path = (
                Path(database + ".runtime.lock")
                if database and database != ":memory:"
                else self.directory / "runtime.lock"
            )
            self.handle = path.open("a+b")
            try:
                if __import__("os").name == "nt":
                    import msvcrt

                    self.handle.seek(0)
                    self.handle.write(b"0")
                    self.handle.flush()
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.handle.close()
                self.handle = None
                raise RuntimeError(
                    "此資料庫已有賓果主程序執行，請使用單一 Uvicorn worker"
                ) from exc

    async def close(self):
        if self.connection:
            await self.connection.execute(
                text("SELECT pg_advisory_unlock(741169542105)")
            )
            await self.connection.close()
        if self.handle:
            self.handle.close()
