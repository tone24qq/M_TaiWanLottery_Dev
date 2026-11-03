import asyncio
from datetime import date, datetime

import httpx
import pytest

from taiwan_lottery.bingo_clock import (
    TAIPEI,
    is_successor,
    latest_scheduled_draw,
    next_draw,
)
from taiwan_lottery.scraper.parsers.bingo_parser import (
    BingoScraper,
    BingoSourceError,
    _parse_api_result,
)

TODAY = date(2026, 11, 3)


def item(term, numbers=None, **kwargs):
    return {
        "drawTerm": term,
        "bigShowOrder": numbers or list(range(1, 21)),
        "dDate": "0001-01-01T00:00:00",
        **kwargs,
    }


def response(total, items):
    return {"rtCode": 0, "content": {"totalSize": total, "bingoQueryResult": items}}


def source(handler, **kwargs):
    return BingoScraper(
        transport=httpx.MockTransport(handler), attempts=1, retry_delay=0, **kwargs
    )


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(
        "taiwan_lottery.scraper.parsers.bingo_parser.now_taipei",
        lambda: datetime(2026, 11, 3, 23, 59, tzinfo=TAIPEI),
    )


def test_complete_day_chronology_uses_all_pages():
    def handler(request):
        page = int(request.url.params["pageNum"])
        terms = list(range(115062101, 115062204))[::-1]
        rows = [item(term) for term in terms[(page - 1) * 100 : page * 100]]
        return httpx.Response(200, json=response(103, rows))

    draws = asyncio.run(source(handler)._fetch_day(TODAY))
    assert len(draws) == 103
    assert draws[0]["draw_datetime"] == datetime(2026, 11, 3, 7, 5)
    assert draws[-1]["draw_datetime"] == datetime(2026, 11, 3, 15, 35)
    assert draws[0]["numbers"] == list(range(1, 21))


def test_failed_second_page_never_returns_partial_day():
    def handler(request):
        if request.url.params["pageNum"] == "2":
            return httpx.Response(503)
        return httpx.Response(
            200, json=response(101, [item(115062201 - i) for i in range(100)])
        )

    with pytest.raises(BingoSourceError, match="已重試"):
        asyncio.run(source(handler)._fetch_day(TODAY))


@pytest.mark.parametrize(
    "numbers",
    [
        list(range(1, 20)),
        [1] * 20,
        list(range(62, 82)),
        [True] + list(range(2, 21)),
        [1.0] + list(range(2, 21)),
    ],
)
def test_invalid_draw_is_not_silently_dropped(numbers):
    with pytest.raises(BingoSourceError):
        _parse_api_result(item(115062101, numbers), TODAY, 115062101)


def test_inconsistent_snapshot_retries_whole_day():
    calls = []

    async def no_wait(_):
        pass

    def handler(request):
        calls.append(request.url.params["pageNum"])
        if len(calls) == 2:
            return httpx.Response(200, json=response(102, [item(115062101)]))
        page = int(request.url.params["pageNum"])
        terms = list(range(115062101, 115062202))[::-1]
        return httpx.Response(
            200,
            json=response(101, [item(t) for t in terms[(page - 1) * 100 : page * 100]]),
        )

    scraper = BingoScraper(
        transport=httpx.MockTransport(handler), attempts=2, retry_delay=0, sleep=no_wait
    )
    assert len(asyncio.run(scraper._fetch_day(TODAY))) == 101
    assert calls == ["1", "2", "1", "2"]


def test_gap_duplicate_error_and_past_partial_day():
    for terms in ([115062101, 115062103], [115062101, 115062101]):
        scraper = source(
            lambda req: httpx.Response(200, json=response(2, [item(t) for t in terms]))
        )
        with pytest.raises(BingoSourceError):
            asyncio.run(scraper._fetch_day(TODAY))
    scraper = source(
        lambda req: httpx.Response(200, json=response(1, [item(115061999)]))
    )
    with pytest.raises(BingoSourceError, match="歷史資料僅"):
        asyncio.run(scraper._fetch_day(date(2026, 11, 2)))


def test_empty_results_and_api_error_are_distinct():
    assert (
        asyncio.run(
            source(lambda req: httpx.Response(200, json=response(0, [])))._fetch_day(
                TODAY
            )
        )
        == []
    )
    for payload in ({}, {"rtCode": 1}, {"rtCode": 0, "content": {}}):
        with pytest.raises(BingoSourceError):
            asyncio.run(
                source(lambda req: httpx.Response(200, json=payload))._fetch_day(TODAY)
            )


def test_year_overnight_and_early_day_clock():
    term, stamp = next_draw("115074095", datetime(2026, 12, 31, 23, 55))
    assert term == "116000001"
    assert stamp == datetime(2027, 1, 1, 7, 5, tzinfo=TAIPEI)
    assert next_draw("115062118", datetime(2026, 11, 2, 23, 55)) == (
        "115062119",
        datetime(2026, 11, 3, 7, 5, tzinfo=TAIPEI),
    )
    assert is_successor(
        "115062118",
        datetime(2026, 11, 2, 23, 55),
        "115062119",
        datetime(2026, 11, 3, 7, 5),
    )
    assert latest_scheduled_draw(datetime(2026, 11, 3, 6, 59)) == datetime(
        2026, 11, 2, 23, 55, tzinfo=TAIPEI
    )
    assert latest_scheduled_draw(datetime(2026, 11, 3, 11, 49)) == datetime(
        2026, 11, 3, 11, 45, tzinfo=TAIPEI
    )


def test_official_time_must_match_complete_day_position():
    with pytest.raises(BingoSourceError, match="期序不符"):
        _parse_api_result(
            item(115062101, drawDateTime="2026-11-03T12:00:00+08:00"), TODAY, 115062101
        )


def test_future_draw_answer_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "taiwan_lottery.scraper.parsers.bingo_parser.now_taipei",
        lambda: datetime(2026, 11, 3, 7, 3, tzinfo=TAIPEI),
    )
    scraper = source(
        lambda req: httpx.Response(200, json=response(1, [item(115062119)]))
    )
    with pytest.raises(BingoSourceError, match="尚未到達"):
        asyncio.run(scraper._fetch_day(TODAY))
