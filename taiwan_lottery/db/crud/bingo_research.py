"""Transactional, causal live prediction ledger; never backfills live predictions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import (
    select,
    update,
    text,
    func,
    case,
    cast,
    BigInteger,
    and_,
    or_,
    extract,
    literal,
    union_all,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from taiwan_lottery.db.base import Base
from taiwan_lottery.db.models.bingo import BingoDraw
from taiwan_lottery.db.models.bingo_research import (
    BingoEvaluationRun,
    BingoForecast,
    BingoPredictionBundle,
)

TAIPEI = ZoneInfo("Asia/Taipei")
DECISION_FIELDS = (
    "target_term",
    "target_time",
    "source_term",
    "source_time",
    "created_at",
    "model",
    "stars",
    "numbers",
    "scores",
    "weights",
    "snapshot_hash",
    "version",
    "method",
)


def local_time(value: datetime | None = None) -> datetime:
    value = value or datetime.now(TAIPEI)
    if value.tzinfo is not None:
        value = value.astimezone(TAIPEI).replace(tzinfo=None)
    return value


def _json_default(value):
    if isinstance(value, datetime):
        return local_time(value).isoformat(timespec="microseconds")
    raise TypeError(f"Unsupported ledger value: {type(value).__name__}")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            default=_json_default,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def decision_digest(row) -> str:
    return digest(
        {
            key: row[key] if isinstance(row, dict) else getattr(row, key)
            for key in DECISION_FIELDS
        }
    )


def valid_numbers(values, count=20) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"Expected {count} bingo numbers")
    if (
        any(
            isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 80
            for x in values
        )
        or len(set(values)) != count
    ):
        raise ValueError("Bingo numbers must be unique integers in 1..80")
    return sorted(values)


class ResearchStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    @staticmethod
    def _insert(session, model):
        return (
            pg_insert(model)
            if session.bind.dialect.name == "postgresql"
            else sqlite_insert(model)
        )

    @staticmethod
    async def _lock(session, term):
        if session.bind.dialect.name == "postgresql":
            # Shared namespace for ingestion and archival. Hash is deterministic
            # across processes, unlike Python's randomized built-in hash().
            key = int.from_bytes(
                hashlib.sha256(f"bingo:{term}".encode()).digest()[:8],
                "big",
                signed=True,
            )
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": key}
            )

    async def init_schema(self):
        async with self.session_factory() as session:
            async with session.bind.begin() as connection:
                await connection.run_sync(
                    lambda conn: Base.metadata.create_all(
                        conn,
                        tables=[
                            BingoDraw.__table__,
                            BingoForecast.__table__,
                            BingoPredictionBundle.__table__,
                            BingoEvaluationRun.__table__,
                        ],
                    )
                )

    async def history(self, limit=1000):
        if not isinstance(limit, int) or not 1 <= limit <= 100000:
            raise ValueError("History limit must be 1..100000")
        async with self.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(BingoDraw)
                        .order_by(
                            BingoDraw.draw_datetime.desc(), BingoDraw.draw_term.desc()
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return [
                {
                    "draw_term": row.draw_term,
                    "draw_datetime": row.draw_datetime,
                    "numbers": list(row.numbers),
                }
                for row in reversed(rows)
            ]

    async def ingest(self, draws):
        canonical = {}
        for draw in draws:
            term = str(draw["draw_term"])
            if not term.isdigit() or len(term) > 20:
                raise ValueError("Invalid draw term")
            numbers = valid_numbers(draw["numbers"])
            when = local_time(draw["draw_datetime"])
            row = {
                "draw_term": term,
                "draw_datetime": when,
                "numbers": numbers,
                "sum_total": sum(numbers),
                "odd_count": sum(x % 2 for x in numbers),
            }
            row.update(
                {
                    f"sector_{i + 1}_count": sum(
                        i * 20 < x <= (i + 1) * 20 for x in numbers
                    )
                    for i in range(4)
                }
            )
            if term in canonical and canonical[term] != row:
                raise ValueError(f"Conflicting source rows for term {term}")
            canonical[term] = row
        if not canonical:
            return 0
        inserted = 0
        terms = sorted(canonical)
        async with self.session_factory() as session, session.begin():
            if session.bind.dialect.name == "postgresql":
                keys = sorted(
                    int.from_bytes(
                        hashlib.sha256(f"bingo:{term}".encode()).digest()[:8],
                        "big",
                        signed=True,
                    )
                    for term in terms
                )
                # One network round trip, globally consistent lock order.
                await session.execute(
                    text(
                        "SELECT pg_advisory_xact_lock(k) FROM (SELECT unnest(CAST(:keys AS BIGINT[])) AS k ORDER BY k) AS ordered_keys"
                    ),
                    {"keys": keys},
                )
            existing = {}
            for offset in range(0, len(terms), 400):
                rows = (
                    await session.execute(
                        select(
                            BingoDraw.draw_term,
                            BingoDraw.numbers,
                            BingoDraw.draw_datetime,
                        ).where(BingoDraw.draw_term.in_(terms[offset : offset + 400]))
                    )
                ).all()
                existing.update({row.draw_term: row for row in rows})
            for term, old in existing.items():
                row = canonical[term]
                if (
                    sorted(old.numbers) != row["numbers"]
                    or old.draw_datetime != row["draw_datetime"]
                ):
                    raise ValueError(
                        f"Official draw conflicts with stored result: {term}"
                    )
            pending = [canonical[term] for term in terms if term not in existing]
            # Bounded batches also stay below SQLite's parameter limit.
            for offset in range(0, len(pending), 100):
                stmt = (
                    self._insert(session, BingoDraw)
                    .values(pending[offset : offset + 100])
                    .on_conflict_do_nothing(index_elements=["draw_term"])
                )
                result = await session.execute(stmt)
                inserted += result.rowcount
            # Check winners when another client inserted a duplicate concurrently.
            pending_terms = [row["draw_term"] for row in pending]
            for offset in range(0, len(pending_terms), 400):
                saved_rows = (
                    await session.execute(
                        select(
                            BingoDraw.draw_term,
                            BingoDraw.numbers,
                            BingoDraw.draw_datetime,
                        ).where(
                            BingoDraw.draw_term.in_(
                                pending_terms[offset : offset + 400]
                            )
                        )
                    )
                ).all()
                for saved in saved_rows:
                    row = canonical[saved.draw_term]
                    if (
                        sorted(saved.numbers) != row["numbers"]
                        or saved.draw_datetime != row["draw_datetime"]
                    ):
                        raise ValueError(
                            f"Concurrent conflicting draw result: {saved.draw_term}"
                        )
        return inserted

    async def archive(
        self,
        target_term,
        target_time,
        source_term,
        source_time,
        payload,
        forecasts,
        now=None,
    ):
        target_term, source_term = str(target_term), str(source_term)
        target_time, source_time, now = (
            local_time(target_time),
            local_time(source_time),
            local_time(now),
        )
        payload = copy.deepcopy(payload)
        async with self.session_factory() as session, session.begin():
            await self._lock(session, target_term)
            old = await session.get(BingoPredictionBundle, target_term)
            if old is not None:
                return self._bundle_payload(old)
            if (
                not target_term.isdigit()
                or not source_term.isdigit()
                or int(source_term) >= int(target_term)
            ):
                raise ValueError("Target term must follow a valid source term")
            if (
                not source_time < target_time
                or not now < target_time
                or source_time > now
            ):
                raise ValueError(
                    "Live prediction deadline passed; historical predictions cannot be archived"
                )
            if (
                await session.execute(
                    select(BingoDraw.id).where(BingoDraw.draw_term == target_term)
                )
            ).first():
                raise ValueError(
                    "Target draw already known; refusing a late live prediction"
                )
            source = (
                await session.execute(
                    select(BingoDraw).where(BingoDraw.draw_term == source_term)
                )
            ).scalar_one_or_none()
            if source is None or source.draw_datetime != source_time:
                raise ValueError("Source draw/time not found in validated snapshot")
            if not forecasts:
                raise ValueError("Cannot archive an empty prediction bundle")
            snapshot_hash = payload.get("snapshot_hash") or digest(
                {
                    "source_term": source_term,
                    "source_time": source_time,
                    "numbers": source.numbers,
                }
            )
            identities = set()
            prepared = []
            for forecast in forecasts:
                stars = forecast["stars"]
                model = str(forecast["model"])
                if (
                    stars not in (3, 4, 5)
                    or not model
                    or len(model) > 50
                    or (model, stars) in identities
                ):
                    raise ValueError("Invalid or duplicated model/star forecast")
                identities.add((model, stars))
                numbers = valid_numbers(forecast["numbers"], stars)
                scores = [float(x) for x in forecast["scores"]]
                if len(scores) != 80 or any(not math.isfinite(x) for x in scores):
                    raise ValueError("Every model must archive 80 finite scores")
                row = dict(
                    target_term=target_term,
                    target_time=target_time,
                    source_term=source_term,
                    source_time=source_time,
                    created_at=now,
                    model=model,
                    stars=stars,
                    numbers=numbers,
                    scores=scores,
                    weights=copy.deepcopy(forecast.get("weights", {})),
                    snapshot_hash=snapshot_hash,
                    version=str(forecast.get("version", "unknown")),
                    method=str(forecast.get("method", "individual")),
                )
                row["decision_hash"] = decision_digest(row)
                prepared.append(row)
            payload.update(
                target_term=target_term,
                created_at=now.replace(tzinfo=TAIPEI).isoformat(),
                settled=False,
                snapshot_hash=snapshot_hash,
            )
            stmt = (
                self._insert(session, BingoPredictionBundle)
                .values(
                    target_term=target_term,
                    created_at=now,
                    payload=payload,
                    payload_hash=digest(payload),
                )
                .on_conflict_do_nothing(index_elements=["target_term"])
            )
            result = await session.execute(stmt)
            if not result.rowcount:
                # Another writer won. Its transaction includes all forecasts.
                return self._bundle_payload(
                    await session.get(BingoPredictionBundle, target_term)
                )
            await session.execute(self._insert(session, BingoForecast).values(prepared))
            return payload

    @staticmethod
    def _bundle_payload(row):
        if row.payload_hash != digest(row.payload):
            raise ValueError(f"Prediction bundle integrity failure: {row.target_term}")
        return copy.deepcopy(row.payload)

    async def settle(self, now=None):
        now = local_time(now)
        settled = 0
        async with self.session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(BingoForecast, BingoDraw)
                    .join(BingoDraw, BingoForecast.target_term == BingoDraw.draw_term)
                    .where(
                        BingoForecast.hit_count.is_(None),
                        BingoForecast.target_time <= now,
                    )
                    .order_by(BingoForecast.target_term)
                )
            ).all()
            for forecast, draw in rows:
                if forecast.decision_hash != decision_digest(forecast):
                    raise ValueError(
                        f"Archived forecast integrity failure: {forecast.target_term}/{forecast.model}/{forecast.stars}"
                    )
                if not (
                    forecast.source_time < forecast.target_time
                    and forecast.created_at < forecast.target_time
                    and draw.draw_datetime == forecast.target_time
                ):
                    raise ValueError(
                        f"Invalid chronological forecast/result: {forecast.target_term}"
                    )
                actual = valid_numbers(draw.numbers)
                hits = len(set(forecast.numbers).intersection(actual))
                result = await session.execute(
                    update(BingoForecast)
                    .where(
                        BingoForecast.id == forecast.id,
                        BingoForecast.hit_count.is_(None),
                    )
                    .values(hit_count=hits, actual_numbers=actual, settled_at=now)
                )
                settled += result.rowcount
        return settled

    async def bundle(self, term=None):
        async with self.session_factory() as session:
            query = select(BingoPredictionBundle)
            query = (
                query.where(BingoPredictionBundle.target_term == str(term))
                if term
                else query.order_by(BingoPredictionBundle.created_at.desc()).limit(1)
            )
            row = (await session.execute(query)).scalar_one_or_none()
            if row is None:
                return None
            payload = self._bundle_payload(row)
            # Restore the exact chosen fusion weights after a process restart.
            # These are derived from the immutable decision rows, never from
            # today's performance; the original payload/hash stays untouched.
            weighted = (
                await session.execute(
                    select(BingoForecast.stars, BingoForecast.weights).where(
                        BingoForecast.target_term == row.target_term,
                        BingoForecast.model == payload.get("method", "shrink"),
                    )
                )
            ).all()
            payload["weights"] = {
                str(item.stars): copy.deepcopy(item.weights) for item in weighted
            }
            # Result state is derived; decision payload itself stays immutable.
            settled = (
                (
                    await session.execute(
                        select(BingoForecast).where(
                            BingoForecast.target_term == row.target_term,
                            BingoForecast.hit_count.is_not(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
            payload["settled"] = bool(settled)
            if settled:
                payload["outcomes"] = [
                    {
                        "model": x.model,
                        "stars": x.stars,
                        "hit_count": x.hit_count,
                        "numbers": x.numbers,
                        "actual_numbers": x.actual_numbers,
                    }
                    for x in settled
                ]
            return payload

    async def outcome_rows(self, limit_terms=1000):
        if not isinstance(limit_terms, int) or not 1 <= limit_terms <= 10000:
            raise ValueError("Recent outcome limit must be 1..10000 terms")
        async with self.session_factory() as session:
            recent_terms = (
                select(BingoForecast.target_term)
                .where(BingoForecast.hit_count.is_not(None))
                .distinct()
                .order_by(BingoForecast.target_term.desc())
                .limit(limit_terms)
            )
            rows = (
                await session.execute(
                    select(
                        BingoForecast.model,
                        BingoForecast.stars,
                        BingoForecast.target_term,
                        BingoForecast.hit_count,
                        BingoForecast.target_time,
                        BingoForecast.settled_at,
                        BingoForecast.created_at,
                        BingoForecast.decision_hash,
                    )
                    .where(
                        BingoForecast.hit_count.is_not(None),
                        BingoForecast.target_term.in_(recent_terms),
                    )
                    .order_by(
                        BingoForecast.target_time,
                        BingoForecast.model,
                        BingoForecast.stars,
                    )
                )
            ).all()
            return [
                {
                    "model": x.model,
                    "stars": x.stars,
                    "target_term": x.target_term,
                    "hit_count": x.hit_count,
                    "draw_datetime": x.target_time.isoformat(),
                    "updated_at": x.settled_at.replace(tzinfo=TAIPEI).isoformat(),
                    "created_at": x.created_at.isoformat(),
                    "decision_hash": x.decision_hash,
                }
                for x in rows
            ]

    async def outcome_summary(self):
        """All-time and latest uninterrupted streak histograms, bounded in Python.

        Window functions find the final gap independently for every model/star.
        Only at most (models * 3 * 6 * 2) aggregate rows leave the database; no
        score arrays or historical ORM instances are materialized.
        """
        columns = (
            BingoForecast.model,
            BingoForecast.stars,
            BingoForecast.hit_count,
            BingoForecast.target_term,
            BingoForecast.target_time,
            BingoForecast.settled_at,
        )
        settled = (
            select(*columns)
            .where(BingoForecast.hit_count.is_not(None))
            .subquery("settled_outcomes")
        )
        grouping = (settled.c.model, settled.c.stars)
        lagged = select(
            settled,
            func.lag(settled.c.target_term)
            .over(partition_by=grouping, order_by=settled.c.target_time)
            .label("previous_term"),
            func.lag(settled.c.target_time)
            .over(partition_by=grouping, order_by=settled.c.target_time)
            .label("previous_time"),
        ).subquery("previous_outcomes")
        current_term = cast(lagged.c.target_term, BigInteger)
        previous_term = cast(lagged.c.previous_term, BigInteger)
        difference = extract("epoch", lagged.c.target_time) - extract(
            "epoch", lagged.c.previous_time
        )
        overnight = and_(
            difference == 25800,
            extract("hour", lagged.c.previous_time) == 23,
            extract("minute", lagged.c.previous_time) == 55,
            extract("hour", lagged.c.target_time) == 7,
            extract("minute", lagged.c.target_time) == 5,
        )
        regular = and_(
            current_term == previous_term + 1, or_(difference == 300, overnight)
        )
        new_year = and_(
            overnight,
            current_term % 1000000 == 1,
            current_term // 1000000 == previous_term // 1000000 + 1,
            extract("year", lagged.c.target_time)
            == extract("year", lagged.c.previous_time) + 1,
            extract("month", lagged.c.target_time) == 1,
            extract("day", lagged.c.target_time) == 1,
            extract("month", lagged.c.previous_time) == 12,
            extract("day", lagged.c.previous_time) == 31,
        )
        gap = case((or_(regular, new_year), None), else_=lagged.c.target_time)
        islands = select(
            lagged,
            func.max(gap)
            .over(partition_by=(lagged.c.model, lagged.c.stars))
            .label("last_gap"),
        ).subquery("final_streak")
        cumulative = select(
            literal("cumulative").label("kind"),
            settled.c.model,
            settled.c.stars,
            settled.c.hit_count,
            func.count().label("count"),
            func.max(settled.c.settled_at).label("updated_at"),
        ).group_by(settled.c.model, settled.c.stars, settled.c.hit_count)
        consecutive = (
            select(
                literal("consecutive").label("kind"),
                islands.c.model,
                islands.c.stars,
                islands.c.hit_count,
                func.count().label("count"),
                func.max(islands.c.settled_at).label("updated_at"),
            )
            .where(islands.c.target_time >= islands.c.last_gap)
            .group_by(islands.c.model, islands.c.stars, islands.c.hit_count)
        )
        async with self.session_factory() as session:
            rows = (await session.execute(union_all(cumulative, consecutive))).all()
        result = {"cumulative": [], "consecutive": []}
        for row in rows:
            result[row.kind].append(
                dict(
                    model=row.model,
                    stars=row.stars,
                    hit_count=row.hit_count,
                    count=row._mapping["count"],
                    updated_at=row.updated_at.replace(tzinfo=TAIPEI).isoformat(),
                )
            )
        return result

    async def save_evaluation(self, run_id, status, report, error=None):
        now = local_time()
        async with self.session_factory() as session, session.begin():
            row = await session.get(BingoEvaluationRun, str(run_id))
            if row is None:
                session.add(
                    BingoEvaluationRun(
                        run_id=str(run_id),
                        status=status,
                        report=report,
                        error=error,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                row.status, row.report, row.error, row.updated_at = (
                    status,
                    report,
                    error,
                    now,
                )

    async def latest_evaluation(self):
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    select(BingoEvaluationRun)
                    .order_by(BingoEvaluationRun.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "run_id": row.run_id,
                "state": row.status,
                "report": copy.deepcopy(row.report),
                "error": row.error,
                "created_at": row.created_at.replace(tzinfo=TAIPEI).isoformat(),
                "updated_at": row.updated_at.replace(tzinfo=TAIPEI).isoformat(),
            }
