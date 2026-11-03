"""Traditional Chinese Bingo Bingo research application."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from taiwan_lottery.config import settings
from taiwan_lottery.db.engine import engine, async_session_factory
from taiwan_lottery.services.bingo_runtime import BingoRuntime

logger.remove()
logger.add(sys.stderr, level="DEBUG" if settings.DEBUG else "INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="INFO")
    from taiwan_lottery.services.runtime_lock import RuntimeLock

    owner = RuntimeLock(engine, settings.MODEL_ARTIFACTS_DIR)
    await owner.acquire()
    runtime = BingoRuntime(async_session_factory)
    app.state.bingo = runtime
    try:
        await runtime.initialize()
        yield
    finally:
        await runtime.close()
        await owner.close()
        await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="僅支援賓果三星、四星、五星；相對評分並非中獎保證。",
    lifespan=lifespan,
)
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
from taiwan_lottery.api.v1.router import api_router

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health(request: Request):
    state = request.app.state.bingo.sync
    return dict(
        status="ok", sync_state=state["state"], last_success=state["last_success"]
    )


@app.get("/", include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/bingo", include_in_schema=False)
@app.get("/predictions", include_in_schema=False)
async def bingo_page():
    return RedirectResponse("/")


@app.get("/analysis/{game}", include_in_schema=False)
@app.get("/history/{game}", include_in_schema=False)
async def analysis_page(request: Request, game: str):
    if game != "bingo":
        raise HTTPException(404, "僅支援賓果賓果")
    return templates.TemplateResponse(
        request=request, name="analysis.html", context={"game": "bingo"}
    )
