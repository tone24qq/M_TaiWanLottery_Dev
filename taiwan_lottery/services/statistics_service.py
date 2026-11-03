"""Statistics service — frequency, hot/cold, gaps, pairs analysis."""

from collections import Counter
from itertools import combinations

from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import bingo
from taiwan_lottery.schemas.statistics import (
    NumberFrequency,
    HotColdAnalysis,
    GapAnalysis,
    PairFrequency,
)

# Game configuration
GAME_CONFIG = {
    "bingo": {"max_num": 80, "pick_count": 20, "get_numbers": lambda d: d.numbers},
}

CRUD_MAP = {
    "bingo": bingo,
}


async def _get_all_draws(session: AsyncSession, game_type: str) -> list:
    crud_module = CRUD_MAP[game_type]
    return await crud_module.get_all_sorted_by_date(session)


def _extract_all_numbers(draws: list, game_type: str) -> list[list[int]]:
    """Extract sorted number lists from all draws."""
    get_numbers = GAME_CONFIG[game_type]["get_numbers"]
    return [get_numbers(d) for d in draws]


async def get_frequency(
    session: AsyncSession, game_type: str, window: int | None = None
) -> list[NumberFrequency]:
    """Get number frequency counts."""
    draws = await _get_all_draws(session, game_type)
    if not draws:
        return []

    number_lists = _extract_all_numbers(draws, game_type)
    if window:
        number_lists = number_lists[-window:]

    max_num = GAME_CONFIG[game_type]["max_num"]
    counter = Counter()
    for nums in number_lists:
        counter.update(nums)

    total_draws = len(number_lists)
    result = []
    for num in range(1, max_num + 1):
        count = counter.get(num, 0)
        result.append(NumberFrequency(
            number=num,
            count=count,
            percentage=round(count / total_draws * 100, 2) if total_draws > 0 else 0,
        ))

    return sorted(result, key=lambda x: x.count, reverse=True)


async def get_hot_cold(
    session: AsyncSession, game_type: str, window: int = 30
) -> HotColdAnalysis:
    """Get hot and cold numbers based on recent window."""
    freq = await get_frequency(session, game_type, window=window)
    if not freq:
        return HotColdAnalysis(hot_numbers=[], cold_numbers=[], window_size=window)

    pick_count = GAME_CONFIG[game_type]["pick_count"]
    hot = freq[:pick_count]
    cold = freq[-pick_count:]

    return HotColdAnalysis(
        hot_numbers=hot,
        cold_numbers=list(reversed(cold)),
        window_size=window,
    )


async def get_gaps(
    session: AsyncSession, game_type: str
) -> list[GapAnalysis]:
    """Get gap analysis (遺漏值) for each number."""
    draws = await _get_all_draws(session, game_type)
    if not draws:
        return []

    number_lists = _extract_all_numbers(draws, game_type)
    max_num = GAME_CONFIG[game_type]["max_num"]
    total_draws = len(number_lists)

    result = []
    for num in range(1, max_num + 1):
        # Find all gaps for this number
        gaps = []
        last_seen = -1

        for i, nums in enumerate(number_lists):
            if num in nums:
                if last_seen >= 0:
                    gaps.append(i - last_seen)
                last_seen = i

        # Current gap (draws since last appearance)
        current_gap = total_draws - 1 - last_seen if last_seen >= 0 else total_draws

        avg_gap = sum(gaps) / len(gaps) if gaps else float(total_draws)
        max_gap = max(gaps) if gaps else current_gap
        gap_ratio = round(current_gap / avg_gap, 2) if avg_gap > 0 else 0

        result.append(GapAnalysis(
            number=num,
            current_gap=current_gap,
            average_gap=round(avg_gap, 2),
            max_gap=max_gap,
            gap_ratio=gap_ratio,
        ))

    return sorted(result, key=lambda x: x.current_gap, reverse=True)


async def get_pairs(
    session: AsyncSession, game_type: str, top_n: int = 20
) -> list[PairFrequency]:
    """Get most common number pairs."""
    draws = await _get_all_draws(session, game_type)
    if not draws:
        return []

    number_lists = _extract_all_numbers(draws, game_type)
    total_draws = len(number_lists)

    pair_counter = Counter()
    for nums in number_lists:
        for pair in combinations(sorted(nums), 2):
            pair_counter[pair] += 1

    result = []
    for pair, count in pair_counter.most_common(top_n):
        result.append(PairFrequency(
            pair=list(pair),
            count=count,
            percentage=round(count / total_draws * 100, 2) if total_draws > 0 else 0,
        ))

    return result


async def get_bias_report(
    session: AsyncSession, game_type: str
) -> dict:
    """Run full bias detection analysis."""
    from taiwan_lottery.ml.bias_detector import BiasDetector

    draws = await _get_all_draws(session, game_type)
    if not draws:
        return {
            "game_type": game_type,
            "total_draws": 0,
            "chi_square_results": {},
            "significant_biases": [],
            "overall_uniformity_p": 1.0,
            "runs_test_results": {},
            "positional_bias": {},
            "temporal_bias": None,
        }

    config = GAME_CONFIG[game_type]
    number_lists = _extract_all_numbers(draws, game_type)

    detector = BiasDetector(config["max_num"], config["pick_count"])
    report = detector.full_report(number_lists)
    report["game_type"] = game_type
    report["total_draws"] = len(number_lists)

    return report
