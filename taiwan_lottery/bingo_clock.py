"""Official Bingo schedule, using Taipei wall-clock times at database boundaries.

Bingo terms increase across nights and restart at ROC-year ``000001`` on Jan 1.
The result API currently returns a sentinel ``dDate``; inferred timestamps must
therefore be based on a complete day response, never on a partial page.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
FIRST_DRAW = time(7, 5)
LAST_DRAW = time(23, 55)
DRAW_INTERVAL = timedelta(minutes=5)
DRAWS_PER_DAY = 203


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


def to_taipei(value: datetime) -> datetime:
    """Naive historical DB timestamps are Taipei wall clock, not UTC."""
    return (
        value.replace(tzinfo=TAIPEI)
        if value.tzinfo is None
        else value.astimezone(TAIPEI)
    )


def first_draw_on(day: date) -> datetime:
    return datetime.combine(day, FIRST_DRAW, TAIPEI)


def scheduled_time(day: date, position: int) -> datetime:
    if not 0 <= position < DRAWS_PER_DAY:
        raise ValueError("開獎日內期數超出 0～202")
    return first_draw_on(day) + position * DRAW_INTERVAL


def next_draw(term: str, drawn_at: datetime) -> tuple[str, datetime]:
    """Return the immediate next scheduled term/time, including overnight/year."""
    current = to_taipei(drawn_at)
    if (
        current.time().replace(tzinfo=None) < FIRST_DRAW
        or current.time().replace(tzinfo=None) > LAST_DRAW
        or current.minute % 5
        or current.second
        or current.microsecond
    ):
        raise ValueError("開獎時間不在賓果排程內")
    if not str(term).isdigit() or int(term) <= 0:
        raise ValueError("開獎期號無效")
    if current.time().replace(tzinfo=None) == LAST_DRAW:
        upcoming = first_draw_on(current.date() + timedelta(days=1))
    else:
        upcoming = current + DRAW_INTERVAL
    next_term = (
        f"{upcoming.year - 1911}000001"
        if upcoming.year != current.year
        else str(int(term) + 1)
    )
    return next_term, upcoming


def latest_scheduled_draw(at: datetime | None = None) -> datetime:
    current = to_taipei(at or now_taipei())
    first = first_draw_on(current.date())
    if current < first:
        return datetime.combine(current.date() - timedelta(days=1), LAST_DRAW, TAIPEI)
    last = datetime.combine(current.date(), LAST_DRAW, TAIPEI)
    if current >= last:
        return last
    steps = int((current - first) // DRAW_INTERVAL)
    return first + steps * DRAW_INTERVAL


def is_successor(
    previous_term: str, previous_time: datetime, target_term: str, target_time: datetime
) -> bool:
    term, stamp = next_draw(previous_term, previous_time)
    return term == str(target_term) and stamp == to_taipei(target_time)
