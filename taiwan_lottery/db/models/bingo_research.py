"""Durable, separate ledgers for live bingo decisions and historical evaluations.

Times in this schema are naive Asia/Taipei, matching ``bingo_draws``. A decision
is inserted once; only its result columns may change after the draw is known.
"""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column

from taiwan_lottery.db.base import Base


class BingoForecast(Base):
    __tablename__ = "bingo_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "target_term", "model", "stars", name="uq_bingo_forecast_target_model_stars"
        ),
        Index("ix_bingo_forecast_board", "model", "stars", "target_time"),
        CheckConstraint("stars IN (3,4,5)", name="ck_bingo_forecast_stars"),
        CheckConstraint(
            "created_at < target_time AND source_time < target_time",
            name="ck_bingo_forecast_causal_time",
        ),
        CheckConstraint(
            "hit_count IS NULL OR (hit_count >= 0 AND hit_count <= stars)",
            name="ck_bingo_forecast_hits",
        ),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_term: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    target_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source_term: Mapped[str] = mapped_column(String(20), nullable=False)
    source_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    stars: Mapped[int] = mapped_column(Integer, nullable=False)
    numbers: Mapped[list] = mapped_column(JSON, nullable=False)
    scores: Mapped[list] = mapped_column(JSON, nullable=False)
    weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(30), nullable=False)
    hit_count: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    actual_numbers: Mapped[list | None] = mapped_column(JSON, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class BingoPredictionBundle(Base):
    __tablename__ = "bingo_prediction_bundles"
    target_term: Mapped[str] = mapped_column(String(20), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class BingoEvaluationRun(Base):
    __tablename__ = "bingo_evaluation_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class BingoForecastAudit(Base):
    """Immutable causal metadata beside, rather than altering, the v1 ledger."""

    __tablename__ = "bingo_forecast_audits"
    __table_args__ = (CheckConstraint("stars IN (3,4,5)", name="ck_bingo_audit_stars"),)
    target_term: Mapped[str] = mapped_column(String(20), primary_key=True)
    model: Mapped[str] = mapped_column(String(50), primary_key=True)
    stars: Mapped[int] = mapped_column(Integer, primary_key=True)
    audit: Mapped[dict] = mapped_column("metadata", JSON, key="audit", nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BingoShadowForecast(Base):
    """Pre-draw challenger decisions; never counted as formal live outcomes."""

    __tablename__ = "bingo_shadow_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "target_term",
            "model",
            "stars",
            "window",
            "profile",
            name="uq_bingo_shadow_identity",
        ),
        Index("ix_bingo_shadow_settlement", "hit_count", "target_time"),
        Index("ix_bingo_shadow_research", "target_term", "model", "stars"),
        CheckConstraint("stars IN (3,4,5)", name="ck_bingo_shadow_stars"),
        CheckConstraint(
            "history_window >= 1 AND history_window <= 10000",
            name="ck_bingo_shadow_window",
        ),
        CheckConstraint(
            "created_at < target_time AND source_time < target_time",
            name="ck_bingo_shadow_causal_time",
        ),
        CheckConstraint(
            "hit_count IS NULL OR (hit_count >= 0 AND hit_count <= stars)",
            name="ck_bingo_shadow_hits",
        ),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_term: Mapped[str] = mapped_column(String(20), nullable=False)
    target_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source_term: Mapped[str] = mapped_column(String(20), nullable=False)
    source_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    stars: Mapped[int] = mapped_column(Integer, nullable=False)
    window: Mapped[int] = mapped_column(
        "history_window", Integer, key="window", nullable=False
    )
    profile: Mapped[str] = mapped_column(String(100), nullable=False)
    numbers: Mapped[list] = mapped_column(JSON, nullable=False)
    scores: Mapped[list] = mapped_column(JSON, nullable=False)
    weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    audit: Mapped[dict] = mapped_column("metadata", JSON, key="audit", nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(30), nullable=False)
    hit_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_numbers: Mapped[list | None] = mapped_column(JSON, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class BingoResearchState(Base):
    """Optimistically versioned monitor/controller snapshots."""

    __tablename__ = "bingo_research_states"
    __table_args__ = (CheckConstraint("version >= 1", name="ck_bingo_state_version"),)
    namespace: Mapped[str] = mapped_column(String(80), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BingoStrategyEvent(Base):
    """Append-only promotion/retention/rollback evidence, separate from forecasts."""

    __tablename__ = "bingo_strategy_events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_term: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)


class BingoDiagnostic(Base):
    """A target is processed exactly once with all controller states/events."""

    __tablename__ = "bingo_diagnostics"
    target_term: Mapped[str] = mapped_column(String(20), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)


class BingoResearchPending(Base):
    """Durable small work queue; chronological catch-up never scans score history."""

    __tablename__ = "bingo_research_pending"
    __table_args__ = (
        Index("ix_bingo_pending_chronology", "target_time", "target_term"),
    )
    target_term: Mapped[str] = mapped_column(String(20), primary_key=True)
    target_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
