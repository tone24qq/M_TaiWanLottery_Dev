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
