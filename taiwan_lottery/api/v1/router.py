"""Only Bingo Bingo is exposed. Legacy offline research code is retained."""

from fastapi import APIRouter
from taiwan_lottery.api.v1.endpoints.bingo import router

api_router = APIRouter()
api_router.include_router(router, prefix="/bingo", tags=["賓果賓果"])
