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
    delete,
    inspect,
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
    BingoForecastAudit,
    BingoShadowForecast,
    BingoResearchState,
    BingoStrategyEvent,
    BingoDiagnostic,
    BingoResearchPending,
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


SHADOW_FIELDS = DECISION_FIELDS + ("window", "profile", "audit")


def json_value(value):
    """Canonical JSON copies prevent mutation and serialize timestamp metadata."""
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def shadow_digest(row):
    return digest(
        {
            key: row[key] if isinstance(row, dict) else getattr(row, key)
            for key in SHADOW_FIELDS
        }
    )


def _validate_metadata(metadata, source_term, target_term):
    metadata = json_value(metadata or {})
    if not isinstance(metadata, dict):
        raise ValueError("Forecast audit metadata must be an object")
    for key in (
        "training_cutoff",
        "training_cutoff_term",
        "trained_through",
        "feature_cutoff",
        "feature_cutoff_term",
        "features_through",
        "history_start",
        "history_end",
        "history_start_term",
        "history_end_term",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        if (
            not str(value).isdigit()
            or int(value) > int(source_term)
            or int(value) >= int(target_term)
        ):
            raise ValueError(f"Audit cutoff {key} includes future data")
    start = metadata.get("history_start", metadata.get("history_start_term"))
    end = metadata.get("history_end", metadata.get("history_end_term"))
    if start is not None and end is not None and int(start) > int(end):
        raise ValueError("Audit history range is reversed")
    for key in ("history_count", "history_size", "window", "training_samples"):
        value = metadata.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"Audit {key} must be a nonnegative integer")
    return metadata


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
                pending_existed = await connection.run_sync(
                    lambda conn: inspect(conn).has_table(
                        BingoResearchPending.__tablename__
                    )
                )
                await connection.run_sync(
                    lambda conn: Base.metadata.create_all(
                        conn,
                        tables=[
                            BingoDraw.__table__,
                            BingoForecast.__table__,
                            BingoPredictionBundle.__table__,
                            BingoEvaluationRun.__table__,
                            BingoForecastAudit.__table__,
                            BingoShadowForecast.__table__,
                            BingoResearchState.__table__,
                            BingoStrategyEvent.__table__,
                            BingoDiagnostic.__table__,
                            BingoResearchPending.__table__,
                        ],
                    )
                )
                if not pending_existed:
                    # One-time upgrade/bootstrap, not a full scan on every poll.
                    seed = (
                        select(
                            BingoForecast.target_term,
                            func.min(BingoForecast.target_time),
                            literal(local_time()),
                        )
                        .outerjoin(
                            BingoDiagnostic,
                            BingoForecast.target_term == BingoDiagnostic.target_term,
                        )
                        .where(
                            BingoForecast.hit_count.is_not(None),
                            BingoDiagnostic.target_term.is_(None),
                        )
                        .group_by(BingoForecast.target_term)
                    )
                    await connection.execute(
                        self._insert(session, BingoResearchPending)
                        .from_select(
                            ["target_term", "target_time", "enqueued_at"], seed
                        )
                        .on_conflict_do_nothing(index_elements=["target_term"])
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

    @staticmethod
    def _prepare_audits(audit, forecasts, source_term, target_term, now):
        if audit is None:
            audit = [
                dict(
                    model=row["model"],
                    stars=row["stars"],
                    metadata=row.get("audit", {}),
                )
                for row in forecasts
                if row.get("audit")
            ]
        if isinstance(audit, dict):
            audit = [
                dict(
                    model=key.rsplit(":", 1)[0],
                    stars=int(key.rsplit(":", 1)[1]),
                    metadata=value,
                )
                for key, value in audit.items()
            ]
        identities = {(row["model"], row["stars"]): row for row in forecasts}
        seen, prepared = set(), []
        for item in audit or []:
            identity = (str(item["model"]), int(item["stars"]))
            if identity not in identities or identity in seen:
                raise ValueError("Audit has an unknown or duplicated forecast identity")
            seen.add(identity)
            metadata = _validate_metadata(
                item.get("metadata", item.get("audit", {})), source_term, target_term
            )
            value = dict(
                target_term=target_term,
                model=identity[0],
                stars=identity[1],
                audit=metadata,
                created_at=now,
            )
            value["payload_hash"] = digest(
                {
                    "metadata": metadata,
                    "decision_hash": identities[identity]["decision_hash"],
                }
            )
            prepared.append(value)
        return prepared

    @staticmethod
    def _prepare_shadows(shadows, template, source_term, target_term):
        prepared, seen = [], set()
        for shadow in shadows or []:
            model, stars = str(shadow["model"]), shadow["stars"]
            window, profile = shadow["window"], str(shadow.get("profile", "window"))
            identity = (model, stars, window, profile)
            if (
                not model
                or len(model) > 50
                or stars not in (3, 4, 5)
                or isinstance(window, bool)
                or not isinstance(window, int)
                or not 1 <= window <= 10000
                or not profile
                or len(profile) > 100
                or identity in seen
            ):
                raise ValueError("Invalid or duplicated shadow identity")
            seen.add(identity)
            scores = [float(x) for x in shadow["scores"]]
            if len(scores) != 80 or any(not math.isfinite(x) for x in scores):
                raise ValueError("Every shadow forecast must archive 80 finite scores")
            metadata = _validate_metadata(
                shadow.get("metadata", shadow.get("audit", {})),
                source_term,
                target_term,
            )
            row = {key: copy.deepcopy(template[key]) for key in DECISION_FIELDS}
            row.update(
                model=model,
                stars=stars,
                window=window,
                profile=profile,
                numbers=valid_numbers(shadow["numbers"], stars),
                scores=scores,
                weights=json_value(shadow.get("weights", {})),
                audit=metadata,
                version=str(shadow.get("version", template["version"])),
                method=str(shadow.get("method", "shadow")),
            )
            row["decision_hash"] = shadow_digest(row)
            prepared.append(row)
        return prepared

    async def archive(
        self,
        target_term,
        target_time,
        source_term,
        source_time,
        payload,
        forecasts,
        now=None,
        *,
        audit=None,
        shadows=None,
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
            if audit is None:
                audit = [
                    dict(
                        model=row["model"],
                        stars=row["stars"],
                        metadata=row.get("metadata", row.get("audit", {})),
                    )
                    for row in forecasts
                    if row.get("metadata") or row.get("audit")
                ]
            audits = self._prepare_audits(
                audit, prepared, source_term, target_term, now
            )
            shadow_rows = self._prepare_shadows(
                shadows, prepared[0], source_term, target_term
            )
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
            if audits:
                await session.execute(
                    self._insert(session, BingoForecastAudit).values(audits)
                )
            # Avoid exceeding SQLite's parameter limit on rich shadow batches.
            for offset in range(0, len(shadow_rows), 30):
                await session.execute(
                    self._insert(session, BingoShadowForecast).values(
                        shadow_rows[offset : offset + 30]
                    )
                )
            return payload

    @staticmethod
    def _bundle_payload(row):
        if row.payload_hash != digest(row.payload):
            raise ValueError(f"Prediction bundle integrity failure: {row.target_term}")
        return copy.deepcopy(row.payload)

    async def _enqueue_research(self, session, targets):
        if not targets:
            return
        processed = set(
            (
                await session.execute(
                    select(BingoDiagnostic.target_term).where(
                        BingoDiagnostic.target_term.in_(targets)
                    )
                )
            )
            .scalars()
            .all()
        )
        rows = [
            dict(target_term=term, target_time=stamp, enqueued_at=local_time())
            for term, stamp in sorted(targets.items())
            if term not in processed
        ]
        for offset in range(0, len(rows), 300):
            await session.execute(
                self._insert(session, BingoResearchPending)
                .values(rows[offset : offset + 300])
                .on_conflict_do_nothing(index_elements=["target_term"])
            )

    async def settle(self, now=None):
        now = local_time(now)
        settled = 0
        targets = {}
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
                if result.rowcount:
                    targets[forecast.target_term] = forecast.target_time
            await self._enqueue_research(session, targets)
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
            if not payload.get("weights"):
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

    async def settle_shadow(self, now=None):
        """Reveal challenger outcomes only after their real archived target closes."""
        now = local_time(now)
        settled = 0
        targets = {}
        async with self.session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(BingoShadowForecast, BingoDraw)
                    .join(
                        BingoDraw,
                        BingoShadowForecast.target_term == BingoDraw.draw_term,
                    )
                    .where(
                        BingoShadowForecast.hit_count.is_(None),
                        BingoShadowForecast.target_time <= now,
                    )
                    .order_by(BingoShadowForecast.target_time, BingoShadowForecast.id)
                )
            ).all()
            for forecast, draw in rows:
                if forecast.decision_hash != shadow_digest(forecast):
                    raise ValueError(
                        f"Shadow forecast integrity failure: {forecast.target_term}/{forecast.model}/{forecast.stars}"
                    )
                if not (
                    forecast.source_time <= forecast.created_at < forecast.target_time
                    and draw.draw_datetime == forecast.target_time
                ):
                    raise ValueError(
                        f"Invalid chronological shadow/result: {forecast.target_term}"
                    )
                actual = valid_numbers(draw.numbers)
                result = await session.execute(
                    update(BingoShadowForecast)
                    .where(
                        BingoShadowForecast.id == forecast.id,
                        BingoShadowForecast.hit_count.is_(None),
                    )
                    .values(
                        hit_count=len(set(forecast.numbers).intersection(actual)),
                        actual_numbers=actual,
                        settled_at=now,
                    )
                )
                settled += result.rowcount
                if result.rowcount:
                    targets[forecast.target_term] = forecast.target_time
            await self._enqueue_research(session, targets)
        return settled

    @staticmethod
    def _research_limit(limit_terms):
        if (
            isinstance(limit_terms, bool)
            or not isinstance(limit_terms, int)
            or not 1 <= limit_terms <= 1000
        ):
            raise ValueError("Research history limit must be 1..1000 terms")

    @staticmethod
    def _serialize_forecast(row, metadata=None):
        data = {key: copy.deepcopy(getattr(row, key)) for key in DECISION_FIELDS}
        for key in ("source_time", "target_time", "created_at"):
            data[key] = data[key].replace(tzinfo=TAIPEI).isoformat()
        data.update(
            hit_count=row.hit_count,
            actual_numbers=copy.deepcopy(row.actual_numbers),
            settled_at=row.settled_at.replace(tzinfo=TAIPEI).isoformat()
            if row.settled_at
            else None,
            draw_datetime=row.target_time.replace(tzinfo=TAIPEI).isoformat(),
            updated_at=row.settled_at.replace(tzinfo=TAIPEI).isoformat()
            if row.settled_at
            else None,
            decision_hash=row.decision_hash,
            metadata=copy.deepcopy(metadata or {}),
        )
        if isinstance(row, BingoShadowForecast):
            data.update(window=row.window, profile=row.profile, is_shadow=True)
        else:
            data.update(
                window=(metadata or {}).get(
                    "window", (metadata or {}).get("history_count")
                ),
                is_shadow=False,
            )
        return data

    async def _research_rows(self, *, after_term=None, limit_terms=300, pending=False):
        self._research_limit(limit_terms)
        if after_term is not None and not str(after_term).isdigit():
            raise ValueError("Research cursor must be a valid term")
        query = select(BingoForecast.target_term).where(
            BingoForecast.hit_count.is_not(None)
        )
        if after_term is not None:
            query = query.where(
                cast(BingoForecast.target_term, BigInteger) > int(after_term)
            )
        if pending:
            query = query.outerjoin(
                BingoDiagnostic,
                BingoForecast.target_term == BingoDiagnostic.target_term,
            ).where(BingoDiagnostic.target_term.is_(None))
        async with self.session_factory() as session:
            if pending:
                # Only queued targets are inspected; the queue survives a crash
                # and drained batches cannot strand late historical settlements.
                query = (
                    select(BingoResearchPending.target_term)
                    .outerjoin(
                        BingoDiagnostic,
                        BingoResearchPending.target_term == BingoDiagnostic.target_term,
                    )
                    .where(
                        BingoDiagnostic.target_term.is_(None),
                        select(BingoForecast.id)
                        .where(
                            BingoForecast.target_term
                            == BingoResearchPending.target_term,
                            BingoForecast.hit_count.is_not(None),
                        )
                        .exists(),
                    )
                    .order_by(
                        BingoResearchPending.target_time,
                        BingoResearchPending.target_term,
                    )
                    .limit(limit_terms)
                )
            else:
                query = (
                    query.distinct()
                    .order_by(BingoForecast.target_term)
                    .limit(limit_terms)
                )
            terms = list((await session.execute(query)).scalars().all())
            if not terms:
                return {"targets": []}
            targets = {
                term: dict(target_term=term, live_rows=[], shadow_rows=[])
                for term in terms
            }
            live = (
                await session.execute(
                    select(BingoForecast, BingoForecastAudit)
                    .outerjoin(
                        BingoForecastAudit,
                        and_(
                            BingoForecast.target_term == BingoForecastAudit.target_term,
                            BingoForecast.model == BingoForecastAudit.model,
                            BingoForecast.stars == BingoForecastAudit.stars,
                        ),
                    )
                    .where(
                        BingoForecast.target_term.in_(terms),
                        BingoForecast.hit_count.is_not(None),
                    )
                    .order_by(
                        BingoForecast.target_time,
                        BingoForecast.model,
                        BingoForecast.stars,
                    )
                )
            ).all()
            for row, audit in live:
                if row.decision_hash != decision_digest(row):
                    raise ValueError(
                        f"Forecast integrity failure in research: {row.target_term}"
                    )
                metadata = audit.audit if audit else {}
                if audit and audit.payload_hash != digest(
                    {"metadata": metadata, "decision_hash": row.decision_hash}
                ):
                    raise ValueError(
                        f"Forecast audit integrity failure: {row.target_term}/{row.model}/{row.stars}"
                    )
                targets[row.target_term]["live_rows"].append(
                    self._serialize_forecast(row, metadata)
                )
            shadows = (
                (
                    await session.execute(
                        select(BingoShadowForecast)
                        .where(
                            BingoShadowForecast.target_term.in_(terms),
                            BingoShadowForecast.hit_count.is_not(None),
                        )
                        .order_by(
                            BingoShadowForecast.target_time,
                            BingoShadowForecast.model,
                            BingoShadowForecast.stars,
                            BingoShadowForecast.window,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in shadows:
                if row.decision_hash != shadow_digest(row):
                    raise ValueError(
                        f"Shadow forecast integrity failure in research: {row.target_term}"
                    )
                targets[row.target_term]["shadow_rows"].append(
                    self._serialize_forecast(row, row.audit)
                )
        return {"targets": list(targets.values())}

    async def research_rows(self, after_term=None, limit_terms=300):
        """Causal settled decisions with full audit, bounded by distinct targets.

        With a cursor this streams oldest-first; without a cursor the caller can
        rebuild bounded monitor state from the latest targets, in causal order.
        """
        if after_term is None:
            self._research_limit(limit_terms)
            async with self.session_factory() as session:
                recent = list(
                    (
                        await session.execute(
                            select(BingoForecast.target_term)
                            .where(BingoForecast.hit_count.is_not(None))
                            .distinct()
                            .order_by(BingoForecast.target_term.desc())
                            .limit(limit_terms + 1)
                        )
                    )
                    .scalars()
                    .all()
                )
            if len(recent) > limit_terms:
                after_term = recent[-1]
        return await self._research_rows(after_term=after_term, limit_terms=limit_terms)

    async def pending_research_settlements(self, limit_terms=300):
        """Consume only after settle() AND settle_shadow(); update via commit API."""
        return await self._research_rows(limit_terms=limit_terms, pending=True)

    @staticmethod
    def _state_value(row):
        if row.payload_hash != digest(row.payload):
            raise ValueError(f"Research state integrity failure: {row.namespace}")
        return dict(
            payload=copy.deepcopy(row.payload),
            version=row.version,
            updated_at=row.updated_at.replace(tzinfo=TAIPEI).isoformat(),
        )

    async def load_research_state(self, namespace):
        async with self.session_factory() as session:
            row = await session.get(BingoResearchState, str(namespace))
            return self._state_value(row) if row else None

    @staticmethod
    def _namespace(namespace):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 80:
            raise ValueError("Research state namespace must contain 1..80 characters")
        return namespace

    async def _save_state(self, session, namespace, payload, expected_version=None):
        namespace = self._namespace(namespace)
        payload = json_value(payload)
        if not isinstance(payload, dict):
            raise ValueError("Research state must be an object")
        old = await session.get(BingoResearchState, namespace)
        if old is not None:
            self._state_value(old)
        actual = old.version if old else 0
        if expected_version is not None and expected_version != actual:
            raise ValueError(
                f"Research state version conflict: {namespace}; expected {expected_version}, found {actual}"
            )
        version = actual + 1
        values = dict(
            payload=payload,
            payload_hash=digest(payload),
            version=version,
            updated_at=local_time(),
        )
        if old:
            result = await session.execute(
                update(BingoResearchState)
                .where(
                    BingoResearchState.namespace == namespace,
                    BingoResearchState.version == actual,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                raise ValueError(
                    f"Concurrent research state version conflict: {namespace}"
                )
        else:
            result = await session.execute(
                self._insert(session, BingoResearchState)
                .values(namespace=namespace, **values)
                .on_conflict_do_nothing(index_elements=["namespace"])
            )
            if result.rowcount != 1:
                raise ValueError(
                    f"Concurrent research state version conflict: {namespace}"
                )
        return version

    async def save_research_state(self, namespace, payload, expected_version=None):
        async with self.session_factory() as session, session.begin():
            await self._lock(session, f"research-state:{namespace}")
            return await self._save_state(session, namespace, payload, expected_version)

    async def commit_research_update(self, target_term, states, diagnostic, events):
        """Commit monitor/controller state exactly once with the closed target.

        No diagnostic or strategy state may be advanced by an unclosed target.
        A single transaction binds the processing marker, state, and event log.
        """
        target_term = str(target_term)
        if not target_term.isdigit():
            raise ValueError("Invalid research target term")
        states = copy.deepcopy(states or {})
        diagnostic = json_value(diagnostic or {})
        if not isinstance(diagnostic, dict):
            raise ValueError("Diagnostic must be an object")
        events = json_value(events or [])
        now = local_time()
        async with self.session_factory() as session, session.begin():
            await self._lock(session, "research-update")
            prior = await session.get(BingoDiagnostic, target_term)
            if prior:
                if prior.payload_hash != digest(prior.payload):
                    raise ValueError(f"Diagnostic integrity failure: {target_term}")
                await session.execute(
                    delete(BingoResearchPending).where(
                        BingoResearchPending.target_term == target_term
                    )
                )
                return False
            closed = (
                await session.execute(
                    select(BingoForecast.id)
                    .join(BingoDraw, BingoForecast.target_term == BingoDraw.draw_term)
                    .where(
                        BingoForecast.target_term == target_term,
                        BingoForecast.hit_count.is_not(None),
                        BingoForecast.target_time <= now,
                    )
                    .limit(1)
                )
            ).first()
            if closed is None:
                raise ValueError(
                    "Cannot update research from an unknown or unsettled target"
                )
            # A late old result can be diagnosed, but must not overwrite state
            # already advanced using a later target. Callers must process causally.
            latest = (
                await session.execute(
                    select(BingoDiagnostic.target_term)
                    .order_by(BingoDiagnostic.target_term.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if latest is not None and int(target_term) <= int(latest) and states:
                raise ValueError(
                    "Research targets must be processed in chronological order"
                )
            marker = (
                self._insert(session, BingoDiagnostic)
                .values(
                    target_term=target_term,
                    payload=diagnostic,
                    payload_hash=digest(diagnostic),
                    created_at=now,
                )
                .on_conflict_do_nothing(index_elements=["target_term"])
            )
            result = await session.execute(marker)
            if not result.rowcount:
                return False
            for namespace, payload in states.items():
                await self._save_state(session, namespace, payload)
            for event in events:
                kind = str(event.get("kind", event.get("type", "diagnostic")))
                if not kind or len(kind) > 50:
                    raise ValueError(
                        "Strategy event kind must contain 1..50 characters"
                    )
                event_payload = json_value(event)
                event_id = str(
                    event.get("event_id")
                    or digest(dict(target_term=target_term, event=event_payload))
                )
                if len(event_id) > 64 or not event_id:
                    raise ValueError("Invalid strategy event identity")
                old = await session.get(BingoStrategyEvent, event_id)
                if old:
                    if old.target_term != target_term or old.payload_hash != digest(
                        event_payload
                    ):
                        raise ValueError(
                            "Strategy event identity conflicts with immutable event"
                        )
                    continue
                session.add(
                    BingoStrategyEvent(
                        id=event_id,
                        target_term=target_term,
                        kind=kind,
                        payload=event_payload,
                        payload_hash=digest(event_payload),
                        created_at=now,
                    )
                )
            await session.execute(
                delete(BingoResearchPending).where(
                    BingoResearchPending.target_term == target_term
                )
            )
        return True

    async def record_validation_update(self, source_term, states, events):
        """Persist a completed validation decision and its state in one transaction."""
        source_term = str(source_term)
        async with self.session_factory() as session, session.begin():
            await self._lock(session, "research-update")
            source = (
                await session.execute(
                    select(BingoDraw.id).where(BingoDraw.draw_term == source_term)
                )
            ).first()
            if source is None:
                raise ValueError("Validation decision requires a known source cutoff")
            for namespace, payload in states.items():
                await self._save_state(session, namespace, payload)
            for event in events:
                value = json_value(event)
                event_id = digest({"source_term": source_term, "event": value})
                prior = await session.get(BingoStrategyEvent, event_id)
                if prior:
                    if prior.payload_hash != digest(value):
                        raise ValueError("Validation event integrity failure")
                    continue
                session.add(
                    BingoStrategyEvent(
                        id=event_id,
                        target_term=source_term,
                        kind=str(value.get("kind", "profile_retained")),
                        payload=value,
                        payload_hash=digest(value),
                        created_at=local_time(),
                    )
                )

    async def latest_diagnostics(self, limit=30):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 300
        ):
            raise ValueError("Diagnostic limit must be 1..300")
        async with self.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(BingoDiagnostic)
                        .order_by(BingoDiagnostic.target_term.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                if row.payload_hash != digest(row.payload):
                    raise ValueError(f"Diagnostic integrity failure: {row.target_term}")
            return [
                dict(
                    target_term=row.target_term,
                    payload=copy.deepcopy(row.payload),
                    created_at=row.created_at.replace(tzinfo=TAIPEI).isoformat(),
                )
                for row in rows
            ]

    async def strategy_events(self, limit=30):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 300
        ):
            raise ValueError("Event limit must be 1..300")
        async with self.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(BingoStrategyEvent)
                        .order_by(
                            BingoStrategyEvent.created_at.desc(),
                            BingoStrategyEvent.id.desc(),
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                if row.payload_hash != digest(row.payload):
                    raise ValueError(f"Strategy event integrity failure: {row.id}")
            return [
                dict(
                    event_id=row.id,
                    target_term=row.target_term,
                    kind=row.kind,
                    payload=copy.deepcopy(row.payload),
                    created_at=row.created_at.replace(tzinfo=TAIPEI).isoformat(),
                )
                for row in rows
            ]

    async def forecast_audit(self, term):
        """Drill down into a saved decision without synthesizing missing audits."""
        term = str(term)
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(BingoForecastAudit, BingoForecast.decision_hash)
                    .join(
                        BingoForecast,
                        and_(
                            BingoForecastAudit.target_term == BingoForecast.target_term,
                            BingoForecastAudit.model == BingoForecast.model,
                            BingoForecastAudit.stars == BingoForecast.stars,
                        ),
                    )
                    .where(BingoForecastAudit.target_term == term)
                    .order_by(BingoForecastAudit.model, BingoForecastAudit.stars)
                )
            ).all()
            output = []
            for audit, decision_hash in rows:
                if audit.payload_hash != digest(
                    {"metadata": audit.audit, "decision_hash": decision_hash}
                ):
                    raise ValueError(
                        f"Forecast audit integrity failure: {term}/{audit.model}/{audit.stars}"
                    )
                output.append(
                    dict(
                        target_term=term,
                        model=audit.model,
                        stars=audit.stars,
                        metadata=copy.deepcopy(audit.audit),
                    )
                )
            return output
