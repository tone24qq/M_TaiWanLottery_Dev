"""Reproduce neighbor-panel browser checks without a database or live writes.

Uses the real analysis template, CSS, neighbor JavaScript and official-data core
bundle. --all-components loads all existing scripts with the repository's saved
read-only API fixtures. The optional backtest fixture tests rendering only and
is never a statistical result. A system Chromium is sufficient.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from itertools import combinations
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

from taiwan_lottery.ml.bingo_neighbors import prepare_neighbor_bundle, neighbor_view

ROOT = Path(__file__).resolve().parents[1]


def _report_fixture():
    models = ["neighbor", "random", "frequency", "cdm", "river", "lightgbm", "catboost",
              "transformer", "dqn", "ensemble", "association_a", "association_b",
              "association_c", "equal", "shrink", "borda", "majority"]
    boards = {}
    evidence = {}
    for star in (3, 4, 5):
        # Explicitly controlled UI contract values, never used for conclusions.
        boards[str(star)] = [dict(name=name, samples=100, mean_hits=star * .25,
                                 ci_low=0., ci_high=float(star), full_hits=0,
                                 full_hit_rate=0., full_hit_ci_low=0., full_hit_ci_high=.04,
                                 random_mean=star * .25,
                                 random_full_hit_rate=math.comb(20, star) / math.comb(80, star),
                                 delta_vs_random=0.) for name in models]
        partition = dict(samples=50, mean_hits=star * .25, mean_ci_low=0.,
                         mean_ci_high=float(star), adjusted_full_hit_p=1.)
        evidence[str(star)] = dict(status="尚未證明優於隨機", independently_validated=False,
                                   discovery=partition, confirmation=partition,
                                   confirmation_from="115001001")
    return dict(mode="neighbor_walk_forward", count=100, is_live=False,
                selection_reason="UI 測試資料，非真實回測證據。",
                leaderboards=dict(cumulative=boards), validation=evidence,
                decision=dict(allowed_for_live=False))


def _analysis_fixture(draws, status):
    """The current analysis response shape, calculated without opening a DB."""
    rows = draws[-1000:]
    counts = Counter(n for row in rows for n in row["numbers"])
    gaps = {n: len(rows) for n in range(1, 81)}
    for index, row in enumerate(reversed(rows)):
        for n in row["numbers"]:
            if gaps[n] == len(rows):
                gaps[n] = index
    pairs = Counter(pair for row in rows for pair in combinations(row["numbers"], 2))
    return dict(samples=len(rows),
                frequency=[dict(number=n, count=counts[n], rate=counts[n] / len(rows), gap=gaps[n])
                           for n in range(1, 81)],
                ranking={star: entry.get("ranking", [])
                         for star, entry in status.get("prediction", {}).get("by_star", {}).items()},
                cooccurrence=[dict(numbers=list(pair), count=count) for pair, count in pairs.most_common(20)],
                size_odd_even=dict(odd=sum(n % 2 for row in rows for n in row["numbers"]),
                                   even=sum(n % 2 == 0 for row in rows for n in row["numbers"]),
                                   small=sum(n <= 40 for row in rows for n in row["numbers"]),
                                   large=sum(n > 40 for row in rows for n in row["numbers"])),
                history=list(reversed(rows[-100:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/validation/official-draws-20261103.json.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "work/neighbors-ui")
    parser.add_argument("--report", type=Path, help="Optional real neighbor walk-forward JSON or gzip report")
    parser.add_argument("--all-components", action="store_true", help="Run the existing bingo, intelligence and association scripts too")
    parser.add_argument("--chromium", default=shutil.which("chromium") or shutil.which("chromium-browser"))
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.source, "rt", encoding="utf-8") as stream:
        draws = json.load(stream)
    existing = {}
    if args.all_components:
        for name in ("status", "intelligence", "associations"):
            with gzip.open(ROOT / f"docs/validation/stage2/live/{name}.json.gz", "rt", encoding="utf-8") as stream:
                existing[name] = json.load(stream)
        existing["analysis"] = _analysis_fixture(draws, existing["status"])
        existing["latest"] = draws[-1]
        existing["draws"] = dict(items=list(reversed(draws[-20:])), total=len(draws),
                                 page=1, page_size=20, total_pages=math.ceil(len(draws) / 20))
    bundle = prepare_neighbor_bundle(draws[-1020:])
    if args.report:
        opener = gzip.open if args.report.suffix == ".gz" else open
        with opener(args.report, "rt", encoding="utf-8") as stream:
            report = json.load(stream)
        if report.get("mode") != "neighbor_walk_forward":
            raise ValueError("Expected a neighbor_walk_forward report")
    else:
        report = _report_fixture()
    insufficient_confirmation = any(item.get("confirmation", {}).get("samples", 0) < 100
                                    for item in report.get("validation", {}).values())
    environment = Environment(loader=FileSystemLoader(ROOT / "taiwan_lottery/templates"),
                              autoescape=select_autoescape(["html"]))
    html = environment.get_template("analysis.html").render()
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium,
                                             args=["--no-sandbox", "--disable-dev-shm-usage"])
        try:
            for width in (320, 390, 1440):
                context = browser.new_context(viewport={"width": width, "height": 1000})
                page = context.new_page()
                errors, queries = [], []
                checks = 0

                def check(condition, detail=""):
                    nonlocal checks
                    checks += 1
                    if not condition:
                        raise AssertionError(detail or f"UI check {checks} failed at width {width}")

                phase = dict(initial=True, failed=False)
                page.on("pageerror", lambda error: errors.append(str(error)))

                def route_request(route):
                    url = urlparse(route.request.url)
                    path = url.path
                    if path == "/analysis/bingo":
                        route.fulfill(status=200, content_type="text/html", body=html)
                    elif not args.all_components and path in ("/static/js/bingo.js", "/static/js/intelligence.js", "/static/js/associations.js"):
                        route.fulfill(status=200, content_type="application/javascript", body="/* Other components are isolated in this test. */")
                    elif path.startswith("/static/"):
                        file = ROOT / "taiwan_lottery" / path.removeprefix("/")
                        if file.is_file():
                            route.fulfill(status=200, content_type="text/css" if file.suffix == ".css" else "application/javascript", body=file.read_text())
                        else:
                            route.fulfill(status=404, body="")
                    elif path == "/api/v1/bingo/neighbors":
                        query = parse_qs(url.query)
                        queries.append(query)
                        if phase["initial"]:
                            phase["initial"] = False
                            response = dict(state="waiting", analysis=None, reason="測試等待狀態")
                        elif phase["failed"]:
                            response = dict(state="failed", analysis=None, reason="測試背景失敗，稍後重試")
                        else:
                            view = neighbor_view(bundle, window=int(query.get("window", [1000])[0]),
                                                 lag=int(query.get("lag", [1])[0]),
                                                 snapshot=int(query.get("snapshot", [0])[0]),
                                                 source=int(query["source"][0]) if query.get("source") else None)
                            view["warning"] = '<img src=x onerror="window.neighborXss=true"> 僅作文字呈現'
                            response = dict(state="ready", source_term=bundle["metadata"][0]["source_term"],
                                            analysis=view, research_only=True, allowed_for_live=False)
                        route.fulfill(status=200, content_type="application/json", body=json.dumps(response, ensure_ascii=False))
                    elif path == "/api/v1/bingo/backtest":
                        response = dict(state="completed", run_id="ui-fixture", finished_at="ui-fixture",
                                        has_report=True)
                        if not url.query:
                            response["report"] = report
                        route.fulfill(status=200, content_type="application/json", body=json.dumps(response, ensure_ascii=False))
                    elif path.removeprefix("/api/v1/bingo/") in existing:
                        route.fulfill(status=200, content_type="application/json",
                                      body=json.dumps(existing[path.removeprefix("/api/v1/bingo/")], ensure_ascii=False))
                    else:
                        route.fulfill(status=404, body="")

                page.route("**/*", route_request)
                page.goto("http://neighbor-ui.test/analysis/bingo")
                page.wait_for_function("document.getElementById('neighbor-state').textContent === '等待歷史資料'")
                check(page.locator("#neighbor-relations .empty").count() == 1)
                page.locator("#neighbor-form button").click()
                page.wait_for_function("document.getElementById('neighbor-state').textContent === '統計已就緒'")
                page.wait_for_selector("#neighbor-relations .neighbor-card")
                check(page.locator("#neighbor-numbers .neighbor-number").count() == 20)
                check(page.locator("#neighbor-relations .neighbor-card").count() == 20)
                check(page.locator("#neighbor-offset-statistics tbody tr").count() == 9)
                check(page.locator("#neighbor-relations").inner_text().count("獨立配對條件率") > 0)
                check("95% 家族同時區間" in page.locator("#neighbor-relations").inner_text())
                check("25%" in page.locator("#neighbor-relations").inner_text())
                check("尚未證明優於隨機" in page.locator(".neighbor-warning").inner_text())
                check("不是下一期中獎機率" in page.locator(".neighbor-warning").inner_text())
                check(page.locator("#neighbor-form option[value='80']").count() == 1)
                check(page.locator("#backtest-form option[value='neighbors']").count() == 1)
                for identifier in ("backtest-results", "frequency", "full-ranking", "history", "distribution"):
                    check(page.locator("#" + identifier).count() == 1)
                if args.all_components:
                    page.wait_for_function("document.querySelectorAll('#frequency tr').length === 80 && document.querySelectorAll('#association-cards .association-card').length === 3")
                    check(page.locator("#frequency tr").count() == 80)
                    check(page.locator("#full-ranking tr").count() == 80)
                    check(page.locator("#history tr").count() == 100)
                    check(page.locator("#intelligence-models .intelligence-model").count() > 0)
                    check(page.locator("#association-cards .association-card").count() == 3)
                    check(not page.locator("#intelligence-error").is_visible())
                    check(not page.locator("#association-error").is_visible())
                    check(not page.locator("#message").is_visible())
                    page.locator('[data-intelligence-star="4"]').click()
                    check(page.locator('[data-intelligence-star="4"]').get_attribute("aria-pressed") == "true")
                    page.locator('[data-association-star="4"]').click()
                    check(page.locator('[data-association-star="4"]').get_attribute("aria-pressed") == "true")
                    check(page.locator("#association-cards .association-card").count() == 3)
                page.wait_for_selector(".neighbor-report-star", timeout=15000)
                check(page.locator(".neighbor-report-star").count() == 3)
                check(page.locator("#neighbor-backtest-results tbody tr").count() == 51)
                if report["count"] == 100:
                    check("後段獨立確認 50 期" in page.locator("#neighbor-backtest-results").inner_text())
                check("可重現隨機選號" in page.locator("#neighbor-backtest-results").inner_text())
                check("ABC 關聯 A" in page.locator("#neighbor-backtest-results").inner_text())
                check(("確認樣本不足 100 期" in page.locator("#neighbor-backtest-results").inner_text()) == insufficient_confirmation)
                check(page.locator("#neighbor-methodology img").count() == 0)
                check(page.evaluate("window.neighborXss === undefined"))
                check("<img src=x" in page.locator("#neighbor-methodology").text_content())
                check(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
                page.locator("#neighbors").evaluate("node => node.scrollIntoView({block: 'start'})")
                page.screenshot(path=str(args.output / f"neighbors-{width}.png"))

                page.select_option("#neighbor-source", "80")
                page.wait_for_function("document.querySelectorAll('#neighbor-relations tbody tr').length === 80")
                targets = page.locator("#neighbor-relations tbody tr td:first-child").all_text_contents()
                check(targets == [f"{n:02d}" for n in range(1, 81)])
                check("來源 80" in page.locator("#neighbor-relations summary").inner_text())
                page.select_option("#neighbor-snapshot", "5")
                page.wait_for_function("!document.getElementById('neighbor-changes').hidden && document.getElementById('neighbor-changes').textContent.includes('最新新增')")
                check("最新−觀察期" in page.locator("#neighbor-relations").inner_text())
                page.select_option("#neighbor-window", "5")
                page.select_option("#neighbor-lag", "20")
                page.wait_for_function("document.getElementById('neighbor-context').textContent.includes('最近 5 期視窗 · 後 20 期')")
                check(page.locator("#neighbor-relations tbody tr").count() == 80)
                check("25%" in page.locator("#neighbor-relations").inner_text())
                check(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
                check(any(q.get("source") == ["80"] and q.get("snapshot") == ["5"] and q.get("lag") == ["20"] for q in queries))
                phase["failed"] = True
                page.locator("#neighbor-form button").click()
                page.wait_for_function("document.getElementById('neighbor-state').textContent === '分析暫不可用'")
                check(page.locator("#neighbor-error").is_visible())
                check(page.locator("#neighbor-relations .neighbor-card").count() == 0)
                check(not errors, errors)
                results.append(dict(width=width, checks=checks, page_errors=errors, body_overflow=False,
                                    requests=len(queries), source_targets=80, source_cards=20,
                                    all_components=args.all_components))
                context.close()
        finally:
            browser.close()
    summary = dict(scope=("Real analysis template, CSS and all component scripts; official neighbor data, real backtest and saved read-only existing API fixtures"
                          if args.all_components else "Real analysis template, CSS and neighbor JS; read-only API fixture; other component scripts isolated"),
                   source=str(args.source.relative_to(ROOT)) if args.source.is_relative_to(ROOT) else args.source.name,
                   backtest_source=str(args.report) if args.report else "Controlled UI contract fixture, not statistical evidence",
                   widths=results, checks=sum(item["checks"] for item in results), passed=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
