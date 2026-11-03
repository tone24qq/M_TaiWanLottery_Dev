"""Historical data backfill — import from ROC year 93 (2004) onwards."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper
from taiwan_lottery.bingo_clock import now_taipei
SCRAPERS = {"bingo": BingoScraper()}
GAME_START_YEAR = {"bingo": 103}



async def backfill_game(
    session: AsyncSession,
    game_type: str,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict:
    """Backfill historical data for a specific game.

    Args:
        session: Database session
        game_type: Only bingo
        year_start: ROC year to start from (default: game start year)
        year_end: ROC year to end at (default: current ROC year)

    Returns:
        dict with total_inserted and details
    """
    scraper = SCRAPERS.get(game_type)
    if not scraper:
        raise ValueError(f"Unknown game type: {game_type}")

    if year_start is None:
        year_start = GAME_START_YEAR.get(game_type, 93)
    if year_end is None:
        year_end = now_taipei().year - 1911

    logger.info(
        "Backfilling {} from ROC {} to {}",
        game_type, year_start, year_end,
    )

    total_inserted = 0
    errors = []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # Skip future months
            now = now_taipei()
            if year + 1911 > now.year or (year + 1911 == now.year and month > now.month):
                continue

            try:
                inserted = await scraper.fetch_by_month(session, year, month)
                total_inserted += inserted
                if inserted > 0:
                    logger.info(
                        "[{}] {}/{}: {} records",
                        game_type, year, month, inserted,
                    )
                # Commit periodically
                await session.commit()
            except Exception as e:
                logger.warning(
                    "[{}] Error for {}/{}: {}",
                    game_type, year, month, e,
                )
                errors.append(f"{year}/{month}: {e}")
                await session.rollback()

    return {
        "game_type": game_type,
        "total_inserted": total_inserted,
        "year_range": f"{year_start}-{year_end}",
        "errors": errors,
    }
