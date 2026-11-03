"""Add immutable bingo live ledger and separate evaluation records.

Existing draw/result/model tables are preserved, including legacy predictions.
Revision ID: b90000000001
Revises: ac41ae7f9661
"""

from alembic import op
import sqlalchemy as sa

revision = "b90000000001"
down_revision = "ac41ae7f9661"
branch_labels = None
depends_on = None


def upgrade():
    # checkfirst supports upgrading an installation initialized by app startup.
    metadata = sa.MetaData()
    sa.Table(
        "bingo_forecasts",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("target_term", sa.String(20), nullable=False, index=True),
        sa.Column("target_time", sa.DateTime(), nullable=False),
        sa.Column("source_term", sa.String(20), nullable=False),
        sa.Column("source_time", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("model", sa.String(50), nullable=False, index=True),
        sa.Column("stars", sa.Integer(), nullable=False),
        sa.Column("numbers", sa.JSON(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("weights", sa.JSON(), nullable=False),
        sa.Column("snapshot_hash", sa.String(64), nullable=False),
        sa.Column("decision_hash", sa.String(64), nullable=False),
        sa.Column("version", sa.String(100), nullable=False),
        sa.Column("method", sa.String(30), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=True, index=True),
        sa.Column("actual_numbers", sa.JSON(), nullable=True),
        sa.Column("settled_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "target_term", "model", "stars", name="uq_bingo_forecast_target_model_stars"
        ),
        sa.CheckConstraint("stars IN (3,4,5)", name="ck_bingo_forecast_stars"),
        sa.CheckConstraint(
            "created_at < target_time AND source_time < target_time",
            name="ck_bingo_forecast_causal_time",
        ),
        sa.CheckConstraint(
            "hit_count IS NULL OR (hit_count >= 0 AND hit_count <= stars)",
            name="ck_bingo_forecast_hits",
        ),
    )
    sa.Table(
        "bingo_prediction_bundles",
        metadata,
        sa.Column("target_term", sa.String(20), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
    )
    sa.Table(
        "bingo_evaluation_runs",
        metadata,
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("report", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    sa.Index(
        "ix_bingo_forecast_board",
        metadata.tables["bingo_forecasts"].c.model,
        metadata.tables["bingo_forecasts"].c.stars,
        metadata.tables["bingo_forecasts"].c.target_time,
    )
    metadata.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    # Explicit downgrade removes only new research tables.
    op.drop_table("bingo_evaluation_runs")
    op.drop_table("bingo_prediction_bundles")
    op.drop_table("bingo_forecasts")
