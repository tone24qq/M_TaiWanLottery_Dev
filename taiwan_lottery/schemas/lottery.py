"""Pydantic schemas for lottery data."""

from datetime import date, datetime
from pydantic import BaseModel
from typing import Literal


class BingoDrawSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    draw_term: str
    draw_datetime: datetime
    numbers: list[int]
    sum_total: int
    odd_count: int
    sector_1_count: int
    sector_2_count: int
    sector_3_count: int
    sector_4_count: int


# --- Paginated response ---

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Scraper ---

class ScrapeRequest(BaseModel):
    game_type: Literal["bingo"] = "bingo"


class ScrapeStatusSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    game_type: Literal["bingo"]
    status: str
    records_found: int
    records_inserted: int
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None


class BackfillRequest(BaseModel):
    game_type: Literal["bingo"]
    year_start: int = 103  # 民國年
    year_end: int | None = None  # None = current year
