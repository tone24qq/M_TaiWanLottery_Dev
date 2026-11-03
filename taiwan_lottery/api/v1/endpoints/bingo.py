"""Bingo-only API. All CPU work runs in bounded worker processes."""

from datetime import date
from pathlib import Path
import re
from fastapi.responses import FileResponse
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from taiwan_lottery.api.deps import get_db
from taiwan_lottery.db.crud import bingo as crud
from taiwan_lottery.schemas.lottery import BingoDrawSchema, PaginatedResponse

router = APIRouter()


def runtime(request: Request):
    return request.app.state.bingo


class BacktestRequest(BaseModel):
    mode: Literal["fixed", "adaptive", "nested", "associations", "neighbors"] = "fixed"
    count: int = Field(100, ge=1, le=10000)
    start_term: str | None = Field(None, pattern=r"^\d{1,20}$")
    end_term: str | None = Field(None, pattern=r"^\d{1,20}$")

    @model_validator(mode="after")
    def ordered(self):
        if (
            self.start_term
            and self.end_term
            and int(self.start_term) > int(self.end_term)
        ):
            raise ValueError("起始期號不可晚於結束期號")
        return self


@router.get("/status")
async def status(request: Request):
    return runtime(request).status()


@router.get("/intelligence")
async def intelligence(request: Request):
    return runtime(request).intelligence()


@router.get("/associations")
async def associations(request: Request):
    return runtime(request).associations()


@router.get("/neighbors")
async def neighbors(
    request: Request,
    window: int = Query(1000),
    lag: int = Query(1),
    snapshot: int = Query(0),
    source: int | None = Query(None, ge=1, le=80),
):
    from taiwan_lottery.services.bingo_neighbors import WINDOWS, LAGS, SNAPSHOTS

    if window not in WINDOWS or lag not in LAGS or snapshot not in SNAPSHOTS:
        raise HTTPException(422, "鄰號研究視窗、跨期距離或比較期數無效")
    return runtime(request).neighbors(window=window, lag=lag, snapshot=snapshot, source=source)


@router.get("/audit/{term}")
async def forecast_audit(term: str, request: Request):
    if not term.isdigit():
        raise HTTPException(422, "期號格式無效")
    return await runtime(request).store.forecast_audit(term)


@router.post("/predict")
async def predict(request: Request):
    try:
        result = await runtime(request).predict()
        if result is None:
            raise HTTPException(409, "資料不足或下一期封存截止時間已過，等待同步")
        return result
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/sync")
async def sync(request: Request):
    try:
        return await runtime(request).cycle()
    except Exception as exc:
        raise HTTPException(503, f"同步失敗：{exc}") from exc


@router.post("/train")
async def train(request: Request):
    try:
        return await runtime(request).start_training()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/train/stop")
async def stop_training(request: Request):
    return await runtime(request).stop_training()


@router.get("/analysis")
async def analysis(request: Request):
    return runtime(request).analysis()


@router.post("/backtest")
async def backtest(body: BacktestRequest, request: Request):
    try:
        options = body.model_dump(exclude_none=True)
        if options["mode"] == "fixed":
            options.pop("mode")
        return await runtime(request).start_backtest(options)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/backtest")
async def backtest_status(request: Request, summary: bool = False):
    state = runtime(request).workers.backtest
    if summary:
        return {
            **{k: v for k, v in state.items() if k != "report"},
            "has_report": bool(state.get("report")),
        }
    return state


@router.get("/backtest/artifact/{run_id}")
async def backtest_artifact(run_id: str, request: Request):
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(404, "報告不存在")
    path = Path(runtime(request).workers.directory) / "reports" / (run_id + ".json.gz")
    if not path.is_file():
        raise HTTPException(404, "報告檔案不存在或尚未完成")
    return FileResponse(
        path,
        media_type="application/gzip",
        filename="bingo-walk-forward-" + run_id + ".json.gz",
    )


@router.get("/latest", response_model=BingoDrawSchema)
async def latest(db: AsyncSession = Depends(get_db)):
    draw = await crud.get_latest(db)
    if not draw:
        raise HTTPException(404, "尚無開獎資料")
    return draw


@router.get("/draws", response_model=PaginatedResponse)
async def draws(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    date_from: date | None = None,
    date_to: date | None = None,
    db: AsyncSession = Depends(get_db),
):
    records, total = await crud.get_draws(
        db, page=page, page_size=page_size, date_from=date_from, date_to=date_to
    )
    return PaginatedResponse(
        items=[BingoDrawSchema.model_validate(d) for d in records],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math_ceil(total, page_size),
    )


def math_ceil(total, size):
    return (total + size - 1) // size


@router.get("/draws/{term}", response_model=BingoDrawSchema)
async def by_term(term: str, db: AsyncSession = Depends(get_db)):
    draw = await crud.get_by_term(db, term)
    if not draw:
        raise HTTPException(404, "查無此期開獎資料")
    return draw
