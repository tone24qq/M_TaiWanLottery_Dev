"""賓果賓果 (Bingo Bingo) ORM model."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, ARRAY, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column

from taiwan_lottery.db.base import Base


class BingoDraw(Base):
    """賓果賓果開獎記錄 — 從80個號碼中開出20個，每5分鐘一期."""

    __tablename__ = "bingo_draws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draw_term: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )
    draw_datetime: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )

    # 20 drawn numbers stored as array
    numbers: Mapped[list[int]] = mapped_column(
        JSON().with_variant(ARRAY(Integer), "postgresql"), nullable=False
    )

    # Pre-computed features
    sum_total: Mapped[int] = mapped_column(Integer, nullable=False)
    odd_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Sector counts (1-20, 21-40, 41-60, 61-80)
    sector_1_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sector_2_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sector_3_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sector_4_count: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("ix_bingo_numbers", "numbers", postgresql_using="gin"),)

    def __repr__(self) -> str:
        return f"<BingoDraw term={self.draw_term} count={len(self.numbers)}>"
