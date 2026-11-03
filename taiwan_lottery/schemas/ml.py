"""Pydantic schemas for ML models and predictions."""

from datetime import datetime
from pydantic import BaseModel
from typing import Literal


class TrainRequest(BaseModel):
    game_type: Literal["bingo"]
    model_type: str = "frequency"  # frequency / lstm / dqn / ensemble
    pick_count: Literal[3,4,5] | None = None


class TrainResponse(BaseModel):
    model_id: int
    game_type: Literal["bingo"]
    model_type: str
    version: str
    metrics: dict | None = None
    message: str


class ConfidenceDetail(BaseModel):
    number: int
    raw_probability: float
    lift: float          # prob / (1/max_num), relative to random
    percentile: float    # rank percentile among all numbers (0-100)
    normalized: float    # min-max scaled score (0-100)


class PredictionResponse(BaseModel):
    game_type: Literal["bingo"]
    model_type: str
    predicted_numbers: list[int]
    confidence_scores: list[float] | None = None
    confidence_details: list[ConfidenceDetail] | None = None
    max_num: int | None = None
    pick_count: int | None = None
    expected_random_hit: float | None = None
    created_at: datetime


class ModelInfo(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    game_type: Literal["bingo"]
    model_type: str
    version: str
    is_active: bool
    metrics: dict | None = None
    trained_at: datetime


class EvaluateRequest(BaseModel):
    game_type: Literal["bingo"]
    model_type: str | None = None


class EvaluateResponse(BaseModel):
    game_type: Literal["bingo"]
    model_type: str
    total_predictions: int
    average_hits: float
    hit_distribution: dict[str, int]  # {"0": 10, "1": 5, "2": 2, ...}


class BacktestRequest(BaseModel):
    game_type: Literal["bingo"]
    model_type: str = "ensemble"
    test_size: int = 100
    compare_all: bool = False  # If True, backtest all models and compare
    pick_count: Literal[3,4,5] | None = None


class BacktestSummary(BaseModel):
    model_type: str
    test_size: int
    train_size: int
    max_num: int
    pick_count: int
    total_hits: int
    average_hits: float
    median_hits: float
    std_hits: float
    max_hits: int
    min_hits: int
    expected_random: float
    lift_vs_random: float
    hit_distribution: dict[str, int]
    hit_rate_nonzero: float
    best_hit_streak: int
    worst_miss_streak: int
    rolling_avg_last5: list[float]
    top_predicted_numbers: list[dict]
    # Phase 1: statistical significance fields
    monte_carlo_avg: float
    monte_carlo_std: float
    p_value: float
    effect_size: float
    confidence_interval_95: list[float]
    quarterly_performance: list[float]
    is_significant: bool


class BacktestResponse(BaseModel):
    game_type: Literal["bingo"]
    backtest: BacktestSummary | None = None
    comparison: list[dict] | None = None
    message: str
