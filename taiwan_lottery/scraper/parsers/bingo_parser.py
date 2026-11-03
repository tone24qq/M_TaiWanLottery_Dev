"""Validated, TLS-verified official Taiwan Lottery Bingo data ingestion.

A day is returned atomically: a failed/malformed page never becomes a successful
partial day. All dates use Asia/Taipei. Existing PostgreSQL columns store naive
Taipei wall times, so timezone information is removed only at the ORM boundary.
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Callable

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.bingo_clock import DRAWS_PER_DAY, TAIPEI, now_taipei, scheduled_time
from taiwan_lottery.db.crud import bingo as crud
from taiwan_lottery.scraper.base import BaseScraper

BINGO_API_URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/BingoResult"
PAGE_SIZE = 100
BINGO_FIRST_DRAW_HOUR = 7
BINGO_FIRST_DRAW_MINUTE = 5
BINGO_INTERVAL_MINUTES = 5
HEADERS = {
    "User-Agent": "TaiwanBingoResearch/2.0",
    "Accept": "application/json",
    "Referer": "https://www.taiwanlottery.com/",
    "Origin": "https://www.taiwanlottery.com",
}


class BingoSourceError(RuntimeError):
    """A day could not be completely retrieved or validated."""


def _compute_sectors(numbers: list[int]) -> tuple[int, int, int, int]:
    return tuple(sum(low <= n <= low + 19 for n in numbers) for low in (1, 21, 41, 61))


def _valid_term(raw: object, draw_date: date) -> str:
    if isinstance(raw, bool) or not str(raw).isdigit():
        raise BingoSourceError("官方資料含無效期號")
    term = str(raw)
    prefix = str(draw_date.year - 1911)
    if (
        len(term) != len(prefix) + 6
        or not term.startswith(prefix)
        or int(term[-6:]) <= 0
    ):
        raise BingoSourceError("官方期號與開獎年度不符")
    return term


def _official_datetime(item: dict, draw_date: date) -> datetime | None:
    # dDate is presently 0001-01-01; date-only values do not establish draw time.
    for key in ("drawDateTime", "drawDatetime", "openDateTime", "dDate"):
        raw = item.get(key)
        if not isinstance(raw, str) or not raw or raw.startswith("0001-"):
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BingoSourceError(f"官方 {key} 時間格式無效") from exc
        stamp = (
            parsed.replace(tzinfo=TAIPEI)
            if parsed.tzinfo is None
            else parsed.astimezone(TAIPEI)
        )
        if stamp.hour == stamp.minute == stamp.second == 0:
            continue
        if (
            stamp.date() != draw_date
            or stamp.minute % 5
            or stamp.second
            or stamp.microsecond
            or not (7, 5) <= (stamp.hour, stamp.minute) <= (23, 55)
        ):
            raise BingoSourceError("官方開獎時間不在當日排程內")
        return stamp
    return None


def _parse_api_result(item: dict, draw_date: date, first_term: int) -> dict:
    """Parse one result; first_term MUST come from a complete validated day."""
    if not isinstance(item, dict):
        raise BingoSourceError("官方開獎紀錄格式無效")
    term = _valid_term(item.get("drawTerm"), draw_date)
    raw_numbers = item.get("bigShowOrder")
    if not isinstance(raw_numbers, list) or len(raw_numbers) != 20:
        raise BingoSourceError(f"第 {term} 期開獎號碼不是 20 個")
    if any(isinstance(n, bool) or not str(n).isdigit() for n in raw_numbers):
        raise BingoSourceError(f"第 {term} 期含非整數號碼")
    numbers = sorted(int(n) for n in raw_numbers)
    if len(set(numbers)) != 20 or numbers[0] < 1 or numbers[-1] > 80:
        raise BingoSourceError(f"第 {term} 期含重複或超出 1～80 號碼")
    position = int(term) - first_term
    try:
        inferred = scheduled_time(draw_date, position)
    except ValueError as exc:
        raise BingoSourceError("官方當日期號範圍超出賓果排程") from exc
    stamp = _official_datetime(item, draw_date) or inferred
    # Complete day chronology gives an independent cross-check of official time.
    if stamp != inferred:
        raise BingoSourceError(f"第 {term} 期時間與完整當日期序不符")
    sectors = _compute_sectors(numbers)
    return {
        "draw_term": term,
        "draw_datetime": stamp.replace(tzinfo=None),
        "numbers": numbers,
        "sum_total": sum(numbers),
        "odd_count": sum(n % 2 for n in numbers),
        **{f"sector_{i + 1}_count": count for i, count in enumerate(sectors)},
    }


class BingoScraper(BaseScraper):
    game_type = "bingo"

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        attempts: int = 3,
        request_timeout: float = 12.0,
        retry_delay: float = 0.5,
        sleep: Callable = asyncio.sleep,
    ):
        if not 1 <= attempts <= 5:
            raise ValueError("重試次數必須介於 1～5")
        self.transport = transport
        self.attempts = attempts
        self.request_timeout = request_timeout
        self.retry_delay = retry_delay
        self.sleep = sleep

    async def _request_page(
        self, client: httpx.AsyncClient, target_date: date, page: int
    ) -> tuple[int, list[dict]]:
        response = await client.get(
            BINGO_API_URL,
            params={
                "openDate": target_date.isoformat(),
                "pageNum": page,
                "pageSize": PAGE_SIZE,
            },
        )
        response.raise_for_status()
        try:
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("rtCode") != 0:
                raise BingoSourceError("官方 API 回報錯誤或缺少成功代碼")
            content = payload.get("content")
            if not isinstance(content, dict):
                raise BingoSourceError("官方 API 缺少 content")
            total = content.get("totalSize")
            results = content.get("bingoQueryResult")
            if (
                isinstance(total, bool)
                or not isinstance(total, int)
                or not 0 <= total <= DRAWS_PER_DAY
                or not isinstance(results, list)
                or len(results) > PAGE_SIZE
            ):
                raise BingoSourceError("官方分頁資料格式無效")
            return total, results
        except (ValueError, TypeError) as exc:
            raise BingoSourceError("官方 API 未傳回有效 JSON") from exc

    async def _fetch_once(
        self, client: httpx.AsyncClient, target_date: date
    ) -> list[dict]:
        total, items = await self._request_page(client, target_date, 1)
        if total == 0:
            if items:
                raise BingoSourceError("官方分頁總數與內容不符")
            return []
        page_count = (total + PAGE_SIZE - 1) // PAGE_SIZE
        for page in range(2, page_count + 1):
            page_total, page_items = await self._request_page(client, target_date, page)
            if page_total != total:
                raise BingoSourceError("抓取期間官方期數改變，重新取得完整快照")
            items.extend(page_items)
        if len(items) != total:
            raise BingoSourceError("官方分頁未完整取得")
        terms = [
            _valid_term(item.get("drawTerm"), target_date)
            if isinstance(item, dict)
            else ""
            for item in items
        ]
        if "" in terms or len(set(terms)) != total:
            raise BingoSourceError("官方分頁含重複或無效期號")
        ordered = sorted(int(term) for term in terms)
        if ordered != list(range(ordered[0], ordered[0] + total)):
            raise BingoSourceError("官方當日期數不連續，拒絕推算錯誤開獎時間")
        # A nonempty past day must contain its full scheduled day. For a current
        # day, totalSize defines the complete set of already published results.
        if target_date < now_taipei().date() and total != DRAWS_PER_DAY:
            raise BingoSourceError(
                f"{target_date} 歷史資料僅 {total}/{DRAWS_PER_DAY} 期，等待補齊"
            )
        parsed = sorted(
            (_parse_api_result(item, target_date, ordered[0]) for item in items),
            key=lambda item: item["draw_datetime"],
        )
        if parsed[-1]["draw_datetime"].replace(
            tzinfo=TAIPEI
        ) > now_taipei() + timedelta(seconds=30):
            raise BingoSourceError("官方資料包含尚未到達的開獎時間")
        return parsed

    async def _fetch_day(self, target_date: date) -> list[dict]:
        if target_date > now_taipei().date():
            raise BingoSourceError("不能查詢未來開獎資料")
        last_error: Exception | None = None
        # Verify certificates and respect the managed proxy/CA configuration.
        async with httpx.AsyncClient(
            headers=HEADERS,
            timeout=httpx.Timeout(self.request_timeout),
            transport=self.transport,
            trust_env=True,
            follow_redirects=False,
        ) as client:
            for attempt in range(self.attempts):
                try:
                    # Total bound includes all pages, in addition to per-I/O bounds.
                    async with asyncio.timeout(self.request_timeout * 4):
                        return await self._fetch_once(client, target_date)
                except (httpx.HTTPError, BingoSourceError, TimeoutError) as exc:
                    last_error = exc
                    if attempt + 1 < self.attempts:
                        await self.sleep(self.retry_delay * 2**attempt)
        raise BingoSourceError(
            f"{target_date} 官方資料同步失敗（已重試 {self.attempts} 次）：{last_error}"
        ) from last_error

    fetch_day = _fetch_day

    async def fetch_days(self, start: date, end: date) -> list[dict]:
        if start > end or (end - start).days > 366:
            raise ValueError("查詢區間必須為 1～367 天")
        all_draws: list[dict] = []
        current = start
        while current <= end:
            all_draws.extend(await self._fetch_day(current))
            current += timedelta(days=1)
        return all_draws

    async def fetch_latest(self, session: AsyncSession) -> int:
        today = now_taipei().date()
        draws = await self._fetch_day(today)
        if not draws:
            draws = await self._fetch_day(today - timedelta(days=1))
        return await crud.bulk_upsert(session, draws)

    async def fetch_by_month(self, session: AsyncSession, year: int, month: int) -> int:
        start = date(year + 1911, month, 1)
        next_month = date(start.year + (month == 12), month % 12 + 1, 1)
        end = min(next_month - timedelta(days=1), now_taipei().date())
        if start > end:
            return 0
        # Commit policy belongs to the caller; errors propagate for rollback.
        return await crud.bulk_upsert(session, await self.fetch_days(start, end))
