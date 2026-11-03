"""Separate causal audit, shadow, diagnostics and versioned strategy state.

Only adds tables. Existing draws and immutable v1 forecasts remain untouched.
Revision ID: b90000000002
Revises: b90000000001
"""

from alembic import op
import sqlalchemy as sa

revision = "b90000000002"
down_revision = "b90000000001"
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    sa.Table(
        "bingo_forecast_audits",
        metadata,
        sa.Column("target_term", sa.String(20), primary_key=True),
        sa.Column("model", sa.String(50), primary_key=True),
        sa.Column("stars", sa.Integer(), primary_key=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("stars IN (3,4,5)", name="ck_bingo_audit_stars"),
    )
    shadow = sa.Table(
        "bingo_shadow_forecasts",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("target_term", sa.String(20), nullable=False),
        sa.Column("target_time", sa.DateTime(), nullable=False),
        sa.Column("source_term", sa.String(20), nullable=False),
        sa.Column("source_time", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("model", sa.String(50), nullable=False),
        sa.Column("stars", sa.Integer(), nullable=False),
        sa.Column("history_window", sa.Integer(), nullable=False),
        sa.Column("profile", sa.String(100), nullable=False),
        sa.Column("numbers", sa.JSON(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("weights", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("snapshot_hash", sa.String(64), nullable=False),
        sa.Column("decision_hash", sa.String(64), nullable=False),
        sa.Column("version", sa.String(100), nullable=False),
        sa.Column("method", sa.String(30), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=True),
        sa.Column("actual_numbers", sa.JSON(), nullable=True),
        sa.Column("settled_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "target_term",
            "model",
            "stars",
            "history_window",
            "profile",
            name="uq_bingo_shadow_identity",
        ),
        sa.CheckConstraint("stars IN (3,4,5)", name="ck_bingo_shadow_stars"),
        sa.CheckConstraint(
            "history_window >= 1 AND history_window <= 10000",
            name="ck_bingo_shadow_window",
        ),
        sa.CheckConstraint(
            "created_at < target_time AND source_time < target_time",
            name="ck_bingo_shadow_causal_time",
        ),
        sa.CheckConstraint(
            "hit_count IS NULL OR (hit_count >= 0 AND hit_count <= stars)",
            name="ck_bingo_shadow_hits",
        ),
    )
    sa.Index("ix_bingo_shadow_settlement", shadow.c.hit_count, shadow.c.target_time)
    sa.Index(
        "ix_bingo_shadow_research", shadow.c.target_term, shadow.c.model, shadow.c.stars
    )
    sa.Table(
        "bingo_research_states",
        metadata,
        sa.Column("namespace", sa.String(80), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("version >= 1", name="ck_bingo_state_version"),
    )
    sa.Table(
        "bingo_strategy_events",
        metadata,
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("target_term", sa.String(20), nullable=False, index=True),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
    )
    sa.Table(
        "bingo_diagnostics",
        metadata,
        sa.Column("target_term", sa.String(20), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
    )
    pending = sa.Table(
        "bingo_research_pending",
        metadata,
        sa.Column("target_term", sa.String(20), primary_key=True),
        sa.Column("target_time", sa.DateTime(), nullable=False),
        sa.Column("enqueued_at", sa.DateTime(), nullable=False),
    )
    sa.Index(
        "ix_bingo_pending_chronology", pending.c.target_time, pending.c.target_term
    )
    metadata.create_all(op.get_bind(), checkfirst=True)
    original = sa.Table(
        "bingo_forecasts",
        sa.MetaData(),
        sa.Column("target_term", sa.String(20)),
        sa.Column("target_time", sa.DateTime()),
        sa.Column("hit_count", sa.Integer()),
    )
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Taipei")).replace(tzinfo=None)
    seed = (
        sa.select(
            original.c.target_term, sa.func.min(original.c.target_time), sa.literal(now)
        )
        .where(original.c.hit_count.is_not(None))
        .group_by(original.c.target_term)
    )
    if op.get_bind().dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    op.get_bind().execute(
        insert(pending)
        .from_select(["target_term", "target_time", "enqueued_at"], seed)
        .on_conflict_do_nothing(index_elements=["target_term"])
    )
    # Index-only addition to the original immutable forecast table.
    ledger = sa.Table(
        "bingo_forecasts", sa.MetaData(), sa.Column("settled_at", sa.DateTime())
    )
    sa.Index("ix_bingo_forecast_settlement_time", ledger.c.settled_at).create(
        op.get_bind(), checkfirst=True
    )


def downgrade():
    op.drop_index("ix_bingo_forecast_settlement_time", table_name="bingo_forecasts")
    for name in (
        "bingo_research_pending",
        "bingo_diagnostics",
        "bingo_strategy_events",
        "bingo_research_states",
        "bingo_shadow_forecasts",
        "bingo_forecast_audits",
    ):
        op.drop_table(name)
