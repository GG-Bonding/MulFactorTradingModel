from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gold_signal.europe import append_europe, classify_europe, fetch_europe_snapshot
from gold_signal.fred import as_record, fetch_us_real_yield_10y
from gold_signal.hypothesis import evaluate_hypothesis, historical_validation, load_hypothesis
from gold_signal.observation import Recorder
from gold_signal.tape import GROUP_LABELS, GROUP_ORDER, fetch_tape
from gold_signal.jin10 import SHANGHAI, Jin10Client, Jin10Error, parse_time
from gold_signal.market import fetch_binance_bars, fetch_binance_price, snapshot
from gold_signal.models import (
    Bar,
    EuropeVerdict,
    FlashNews,
    MarketSnapshot,
    PricedQuote,
    PolymarketBoard,
    SignalResult,
    SignalSide,
    Thresholds,
)
from gold_signal.news import classify_btc_news, classify_gold_news, news_age_seconds, news_weight
from gold_signal.polymarket import PolymarketFeed
from gold_signal.products import SIGNAL_PRODUCTS, fetch_signal_board, impact_for_product, pick_product_news
from gold_signal.signal import EventLog, SignalEngine, SignalStore

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
DATA_PATH = ROOT / "data" / "signals.jsonl"
EVENTS_PATH = ROOT / "data" / "events.jsonl"
HISTORY_PATH = ROOT / "data" / "history" / "live.jsonl"
EUROPE_PATH = ROOT / "data" / "europe.jsonl"
CONSOLE = Console()


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Gold Realtime Signal Engine V0")
    parser.add_argument("--mode", choices=("replay", "live", "europe", "tape", "polymarket", "stats", "yield", "hypothesis", "backtest"), required=True)
    parser.add_argument("--strategy", type=Path, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--book", choices=("gold", "btc", "all"), default="gold")
    parser.add_argument("--interval", type=int, default=None, help="seconds between ticks")
    parser.add_argument("--ticks", type=int, default=0, help="ticks then exit; 0 = forever (live) or one shot (europe)")
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)

    store = SignalStore(args.output)
    engine = SignalEngine()
    interval = args.interval
    if interval is None:
        interval = 60 if args.mode in ("europe", "tape", "polymarket") else Thresholds.POLL_SECONDS

    try:
        if args.mode == "replay":
            return run_replay(engine, store)
        if args.mode == "europe":
            return run_europe(interval=interval, ticks=args.ticks or 1)
        if args.mode == "tape":
            return run_tape(interval=interval, ticks=args.ticks or 1)
        if args.mode == "polymarket":
            return run_polymarket(interval=interval, ticks=args.ticks or 1)
        if args.mode == "stats":
            return run_stats()
        if args.mode == "yield":
            return run_yield()
        if args.mode == "hypothesis":
            if args.strategy is None:
                raise RuntimeError("--strategy is required")
            return run_hypothesis(args.strategy)
        if args.mode == "backtest":
            if args.strategy is None or not args.start or not args.end:
                raise RuntimeError("--strategy, --start, and --end are required")
            return run_backtest(args.strategy, args.start, args.end)
        return run_live(engine, store, interval=interval, ticks=args.ticks, book=args.book)
    except (Jin10Error, RuntimeError) as exc:
        CONSOLE.print(f"[red]{exc}[/red]")
        return 1


def run_replay(engine: SignalEngine, store: SignalStore) -> int:
    cases = load_replay_cases()
    CONSOLE.rule("Gold Signal Engine V0 / REPLAY")
    seen: dict[str, int] = {side.value: 0 for side in SignalSide}
    last_results: list[SignalResult] = []

    for case in cases:
        market = snapshot(
            case["xau"],
            case["xag"],
            case["eurusd"],
            as_of=case["now"],
        )
        result = engine.evaluate(case["news"], market, now=case["now"])
        future = case.get("future_xau") or {}
        if result.signal in (SignalSide.BUY, SignalSide.SELL) and future:
            result.return_1m = _future_return(result.xau_price, future.get("1m"))
            result.return_5m = _future_return(result.xau_price, future.get("5m"))
            result.return_15m = _future_return(result.xau_price, future.get("15m"))
            result.return_30m = _future_return(result.xau_price, future.get("30m"))
        store.append(result)
        EventLog(EVENTS_PATH).upsert(result)
        seen[result.signal.value] += 1
        last_results.append(result)
        CONSOLE.print(render_signal_card(result, market))
        expected = case["expected"]
        ok = _expected_ok(expected, result.signal)
        CONSOLE.print(f"expected={expected} actual={result.signal.value} {'OK' if ok else 'FAIL'}")
        CONSOLE.print()
        if not ok:
            raise SystemExit(f"replay fixture {case['name']} failed: expected {expected}, got {result.signal.value}")

    CONSOLE.rule("Replay summary")
    CONSOLE.print(f"BUY={seen['BUY']} SELL={seen['SELL']} HOLD={seen['HOLD']}")
    CONSOLE.print(f"signals written to {store.path}")
    CONSOLE.print(render_dashboard(market, case["news"], result, last_results, mode="replay"))
    return 0


def run_live(
    engine: SignalEngine,
    store: SignalStore,
    interval: int,
    ticks: int,
    book: str = "gold",
) -> int:
    token = os.getenv("JIN10_TOKEN") or os.getenv("JIN10_BEARER_TOKEN")
    if not token:
        raise Jin10Error(
            "Live mode needs JIN10_TOKEN. Get one at https://mcp.jin10.com/app/ then put it in .env"
        )
    url = os.getenv("JIN10_MCP_URL") or os.getenv("JIN10_MCP_SERVER_URL") or "https://mcp.jin10.com/mcp"
    client = Jin10Client(token=token, url=url)
    poly_feed: PolymarketFeed | None = None
    try:
        init = client.connect()
        CONSOLE.print(f"Jin10 connected. tools={client.tool_names()} book={book}")
        CONSOLE.print(f"initialize protocol={init.get('protocolVersion')}")
        if book == "all":
            return _run_all_products(client, engine, store, interval, ticks)
        pending: list[dict[str, Any]] = []
        history: list[SignalResult] = []
        tape = QuoteTape()
        tick = 0
        last_market = None
        europe_verdict: EuropeVerdict | None = None
        last_europe_ts = 0.0
        polymarket_board: PolymarketBoard | None = None
        if book == "gold":
            poly_feed = PolymarketFeed()
            try:
                polymarket_board = poly_feed.start()
            except RuntimeError as exc:
                CONSOLE.print(f"[yellow]polymarket feed failed: {exc}[/yellow]")
                poly_feed.stop()
                poly_feed = None
        with Live(console=CONSOLE, refresh_per_second=4) as live:
            while True:
                tick += 1
                news_list = client.list_flash()
                if book == "btc":
                    keywords = ("比特币",)
                else:
                    keywords = ("黄金", "美联储", "国债", "美债")
                for keyword in keywords:
                    try:
                        news_list = merge_news(news_list, client.search_flash(keyword))
                    except Jin10Error as exc:
                        CONSOLE.print(f"[yellow]search_flash({keyword}) failed: {exc}[/yellow]")
                now = datetime.now(tz=SHANGHAI)
                if book == "btc":
                    news = pick_news(news_list, now, classifier=classify_btc_news)
                else:
                    news = pick_news(news_list, now)
                market = fetch_market(client, tape, book=book)
                last_market = market
                now = market.as_of
                impact = classify_btc_news(news.text) if book == "btc" and news else None
                result = engine.evaluate(news, market, now=now, impact=impact)
                store.append(result)
                EventLog(EVENTS_PATH).upsert(result)
                Recorder(HISTORY_PATH).append(
                    "observation",
                    {
                        "factor": f"market.{market.xau.code}.close",
                        "value": market.xau.price,
                        "observed_at": market.as_of,
                        "available_at": market.as_of,
                        "ingested_at": datetime.now(tz=SHANGHAI),
                        "source": "live",
                    },
                )
                if news is not None:
                    Recorder(HISTORY_PATH).append(
                        "event",
                        {
                            "event_id": news.event_id,
                            "published_at": news.published_at,
                            "available_at": news.published_at,
                            "ingested_at": datetime.now(tz=SHANGHAI),
                            "title": news.title,
                            "source": "jin10",
                        },
                    )
                if result.signal in (SignalSide.BUY, SignalSide.SELL) and result.is_primary:
                    pending.append(
                        {
                            "event_id": result.event_id,
                            "timestamp": result.timestamp.isoformat(),
                            "entry": result.xau_price,
                            "t0": result.timestamp,
                        }
                    )
                history.append(result)
                update_pending_returns(store, pending, now, market.xau.price)
                if book == "gold" and time.time() - last_europe_ts >= 60:
                    try:
                        europe_verdict = classify_europe(fetch_europe_snapshot(client))
                        append_europe(EUROPE_PATH, europe_verdict)
                    except (Jin10Error, RuntimeError) as exc:
                        CONSOLE.print(f"[yellow]europe fetch failed: {exc}[/yellow]")
                    last_europe_ts = time.time()
                if poly_feed is not None:
                    polymarket_board = poly_feed.snapshot()
                live.update(
                    render_dashboard(
                        market,
                        news,
                        result,
                        history[-10:],
                        mode=f"live/{book}",
                        europe=europe_verdict,
                        polymarket=polymarket_board,
                    )
                )
                if ticks and tick >= ticks:
                    break
                time.sleep(interval)
        if history and last_market is not None:
            CONSOLE.print(render_signal_card(history[-1], last_market))
    finally:
        if poly_feed is not None:
            poly_feed.stop()
        client.close()
    return 0


def _run_all_products(
    client: Jin10Client,
    engine: SignalEngine,
    store: SignalStore,
    interval: int,
    ticks: int,
) -> int:
    tick = 0
    while True:
        tick += 1
        news_list = client.list_flash()
        for keyword in ("黄金", "美联储", "国债", "美债", "原油", "欧元", "纳指", "比特币"):
            try:
                news_list = merge_news(news_list, client.search_flash(keyword))
            except Jin10Error as exc:
                CONSOLE.print(f"[yellow]search_flash({keyword}) failed: {exc}[/yellow]")
        markets, missing = fetch_signal_board(client)
        for note in missing:
            CONSOLE.print(f"[yellow]{note}[/yellow]")
        now = datetime.now(tz=SHANGHAI)
        CONSOLE.rule(f"product signals {now.strftime('%H:%M:%S')}")
        for spec, market in markets:
            news = pick_product_news(news_list, now, spec.family)
            impact = impact_for_product(news.text, spec.family) if news else None
            result = engine.evaluate(news, market, now=now, impact=impact)
            store.append(result)
            EventLog(EVENTS_PATH).upsert(result)
            CONSOLE.print(render_signal_card(result, market))
        if not markets:
            CONSOLE.print("[yellow]no product had 1m bars[/yellow]")
        CONSOLE.print(f"products={len(SIGNAL_PRODUCTS)} priced={len(markets)}")
        if ticks and tick >= ticks:
            break
        time.sleep(interval)
    return 0


def run_europe(interval: int, ticks: int) -> int:
    token = os.getenv("JIN10_TOKEN") or os.getenv("JIN10_BEARER_TOKEN")
    url = os.getenv("JIN10_MCP_URL") or os.getenv("JIN10_MCP_SERVER_URL") or "https://mcp.jin10.com/mcp"
    client: Jin10Client | None = None
    try:
        if token:
            try:
                client = Jin10Client(token=token, url=url)
                client.connect()
                CONSOLE.print(f"Jin10 connected for FX. tools={client.tool_names()}")
            except Jin10Error as exc:
                CONSOLE.print(f"[yellow]Jin10 unavailable ({exc}); FX from Yahoo[/yellow]")
                client = None
        else:
            CONSOLE.print("[yellow]No JIN10_TOKEN — EURUSD/GBPUSD fall back to Yahoo[/yellow]")
        tick = 0
        last = None
        while True:
            tick += 1
            snap = fetch_europe_snapshot(client)
            verdict = classify_europe(snap)
            append_europe(EUROPE_PATH, verdict)
            last = verdict
            CONSOLE.print(render_europe_panel(verdict))
            if ticks and tick >= ticks:
                break
            time.sleep(interval)
        if last is not None:
            CONSOLE.print(f"europe verdict written to {EUROPE_PATH}")
    finally:
        if client is not None:
            client.close()
    return 0


def run_stats() -> int:
    rows = load_events(EVENTS_PATH)
    CONSOLE.print(f"events={len(rows)} file={EVENTS_PATH}")
    table = Table(title="Win rate after the news")
    table.add_column("horizon")
    table.add_column("status")
    table.add_column("samples")
    table.add_column("waiting")
    table.add_column("wins")
    table.add_column("win rate")
    table.add_column("avg return")
    for row in win_rate_table(rows):
        rate = "INSUFFICIENT" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        average = "INSUFFICIENT" if row["avg_return"] is None else f"{row['avg_return']:+.2%}"
        table.add_row(row["horizon"], row["status"], str(row["samples"]), str(row["waiting"]), str(row["wins"]), rate, average)
    CONSOLE.print(table)
    return 0


def run_hypothesis(strategy: Path) -> int:
    spec = load_hypothesis(strategy)
    CONSOLE.print(f"{spec.id} {spec.asset} entry={spec.entry}")
    for case in load_replay_cases():
        market = snapshot(case["xau"], case["xag"], case["eurusd"], as_of=case["now"])
        decision = evaluate_hypothesis(spec, case["news"], market, case["now"])
        CONSOLE.print(f"{case['name']} expected={case['expected']} side={decision.side} {' '.join(decision.reasons)}")
    return 0


def run_backtest(strategy: Path, start: str, end: str) -> int:
    spec = load_hypothesis(strategy)
    report = historical_validation(spec, start, end)
    CONSOLE.print(f"{report['hypothesis']} {report['start']} → {report['end']}")
    CONSOLE.print(f"status={report['status']} samples={report['samples']}")
    for item in report["missing"]:
        CONSOLE.print(f"missing: {item}")
    return 0


def run_yield() -> int:
    try:
        point = fetch_us_real_yield_10y()
    except RuntimeError as exc:
        CONSOLE.print(f"[red]{exc}[/red]")
        return 1
    record = as_record(point)
    if record.get("value") is None:
        CONSOLE.print(f"US 10Y real yield MISSING  [{record['source']}]")
        return 0
    CONSOLE.print(
        f"US 10Y real yield {record['value']:.2f}%  date {record['timestamp']}  [{record['source']}]\n"
        "This is not a gold BUY or SELL. Do not use the print before its observation date."
    )
    return 0


def run_polymarket(interval: int, ticks: int) -> int:
    feed = PolymarketFeed()
    try:
        CONSOLE.print(render_polymarket(feed.start()))
        deadline = time.time() + 8
        while time.time() < deadline and feed.updates == 0:
            time.sleep(0.2)
        board = feed.snapshot()
        CONSOLE.print(render_polymarket(board))
        CONSOLE.print(f"clob updates={feed.updates}")
        if feed.error:
            CONSOLE.print(f"[yellow]clob: {feed.error}[/yellow]")
        if ticks <= 1:
            return 0
        seen = feed.updates
        tick = 1
        while True:
            time.sleep(interval)
            tick += 1
            if feed.updates != seen:
                seen = feed.updates
                CONSOLE.print(render_polymarket(feed.snapshot()))
            if ticks and tick >= ticks:
                break
    finally:
        feed.stop()
    return 0


def run_tape(interval: int, ticks: int) -> int:
    token = os.getenv("JIN10_TOKEN") or os.getenv("JIN10_BEARER_TOKEN")
    url = os.getenv("JIN10_MCP_URL") or os.getenv("JIN10_MCP_SERVER_URL") or "https://mcp.jin10.com/mcp"
    client: Jin10Client | None = None
    try:
        if token:
            try:
                client = Jin10Client(token=token, url=url)
                client.connect()
                CONSOLE.print(f"Jin10 connected for oil/FX. tools={client.tool_names()}")
            except Jin10Error as exc:
                CONSOLE.print(f"[yellow]Jin10 unavailable ({exc}); oil and FX will be missing[/yellow]")
                client = None
        else:
            CONSOLE.print("[yellow]No JIN10_TOKEN — oil and FX will be missing[/yellow]")
        tick = 0
        while True:
            tick += 1
            CONSOLE.print(render_tape(fetch_tape(client)))
            if ticks and tick >= ticks:
                break
            time.sleep(interval)
    finally:
        if client is not None:
            client.close()
    return 0


def fetch_market(
    client: Jin10Client,
    tape: "QuoteTape | None" = None,
    book: str = "gold",
) -> MarketSnapshot:
    wall = datetime.now(tz=SHANGHAI)
    count = Thresholds.KLINE_MINUTES + 1
    if book == "btc":
        btc_bars = fetch_binance_bars("BTCUSDT", count=count)
        eth_bars = fetch_binance_bars("ETHUSDT", count=count)
        btc_price = fetch_binance_price("BTCUSDT")
        eth_price = fetch_binance_price("ETHUSDT")
        eurusd_quote = client.get_quote("EURUSD")
        eurusd_bars = client.get_kline("EURUSD", count=count)
        eurusd_bars = _ensure_bars("EURUSD", eurusd_bars, eurusd_quote.price, wall, count)
        return snapshot(
            btc_bars,
            eth_bars,
            eurusd_bars,
            xau_price=btc_price,
            xag_price=eth_price,
            eurusd_price=eurusd_quote.price,
            as_of=wall,
            primary_code="BTCUSDT",
            confirm_code="ETHUSDT",
            dollar_code="EURUSD",
        )

    xau_quote = client.get_quote("XAUUSD")
    xag_quote = client.get_quote("XAGUSD")
    eurusd_quote = client.get_quote("EURUSD")
    if tape is not None:
        tape.add("XAUUSD", wall, xau_quote.price)
        tape.add("XAGUSD", wall, xag_quote.price)
        tape.add("EURUSD", wall, eurusd_quote.price)
    xau_bars = client.get_kline("XAUUSD", count=count)
    xag_bars = client.get_kline("XAGUSD", count=count)
    eurusd_bars = client.get_kline("EURUSD", count=count)
    if min(len(xau_bars), len(xag_bars), len(eurusd_bars)) < 2:
        if tape is not None:
            xau_bars = tape.bars("XAUUSD") or xau_bars
            xag_bars = tape.bars("XAGUSD") or xag_bars
            eurusd_bars = tape.bars("EURUSD") or eurusd_bars
    xau_bars = _ensure_bars("XAUUSD", xau_bars, xau_quote.price, wall, count)
    xag_bars = _ensure_bars("XAGUSD", xag_bars, xag_quote.price, wall, count)
    eurusd_bars = _ensure_bars("EURUSD", eurusd_bars, eurusd_quote.price, wall, count)
    return snapshot(
        xau_bars,
        xag_bars,
        eurusd_bars,
        xau_price=xau_quote.price,
        xag_price=xag_quote.price,
        eurusd_price=eurusd_quote.price,
        as_of=wall,
    )


def _ensure_bars(code: str, bars: list[Bar], price: float, end: datetime, count: int) -> list[Bar]:
    if len(bars) >= 2:
        return bars
    start = end - timedelta(minutes=count - 1)
    return [Bar(ts=start + timedelta(minutes=i), close=float(price)) for i in range(count)]


class QuoteTape:
    def __init__(self) -> None:
        self.points: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    def add(self, code: str, ts: datetime, price: float) -> None:
        series = self.points[code]
        series.append((ts, price))
        cutoff = ts - timedelta(minutes=40)
        self.points[code] = [item for item in series if item[0] >= cutoff]

    def bars(self, code: str) -> list[Bar]:
        buckets: dict[datetime, float] = {}
        for ts, price in self.points.get(code, []):
            minute = ts.replace(second=0, microsecond=0)
            buckets[minute] = price
        return [Bar(ts=ts, close=price) for ts, price in sorted(buckets.items())]


def merge_news(left: list[FlashNews], right: list[FlashNews]) -> list[FlashNews]:
    seen: set[str] = set()
    out: list[FlashNews] = []
    for item in left + right:
        if item.event_id in seen:
            continue
        seen.add(item.event_id)
        out.append(item)
    return out


def pick_news(
    items: list[FlashNews],
    now: datetime,
    classifier=classify_gold_news,
) -> FlashNews | None:
    ranked: list[tuple[int, float, FlashNews]] = []
    for item in items:
        impact = classifier(item.text)
        if impact.importance <= 0:
            continue
        age = news_age_seconds(item, now)
        if news_weight(age) <= 0:
            continue
        ranked.append((impact.importance, -age, item))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return ranked[0][2]


def update_pending_returns(
    store: SignalStore,
    pending: list[dict[str, Any]],
    now: datetime,
    price: float,
) -> None:
    if not pending:
        return
    still: list[dict[str, Any]] = []
    for item in pending:
        entry = float(item["entry"])
        elapsed = (now - item["t0"]).total_seconds()
        ret = price / entry - 1 if entry else None
        kwargs: dict[str, float] = {}
        if elapsed >= 60 and "done_1m" not in item:
            kwargs["return_1m"] = ret
            item["done_1m"] = True
        if elapsed >= 300 and "done_5m" not in item:
            kwargs["return_5m"] = ret
            item["done_5m"] = True
        if elapsed >= 900 and "done_15m" not in item:
            kwargs["return_15m"] = ret
            item["done_15m"] = True
        if elapsed >= 1800 and "done_30m" not in item:
            kwargs["return_30m"] = ret
            item["done_30m"] = True
        if kwargs:
            store.update_returns(item["event_id"], item["timestamp"], **kwargs)
        if "done_30m" not in item:
            still.append(item)
    pending[:] = still


def load_replay_cases() -> list[dict[str, Any]]:
    import json

    cases = []
    paths = sorted(FIXTURE_DIR.glob("case*.json"))
    if not paths:
        raise FileNotFoundError(f"no replay fixtures in {FIXTURE_DIR}")
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        now = parse_time(raw["now"])
        news_raw = raw["news"]
        news = FlashNews(
            event_id=news_raw.get("event_id") or "",
            title=news_raw.get("title") or "",
            content=news_raw.get("content") or news_raw.get("title") or "",
            published_at=parse_time(news_raw["published_at"]),
        )
        if not news.event_id:
            from gold_signal.news import ensure_event_id

            news.event_id = ensure_event_id(news)
        cases.append(
            {
                "name": path.stem,
                "expected": raw["expected"],
                "now": now,
                "news": news,
                "xau": _bars_from_closes(raw["xau_closes"], now),
                "xag": _bars_from_closes(raw["xag_closes"], now),
                "eurusd": _bars_from_closes(raw["eurusd_closes"], now),
                "future_xau": raw.get("future_xau") or {},
            }
        )
    return cases


def _bars_from_closes(closes: list[float], end: datetime) -> list[Bar]:
    start = end - timedelta(minutes=len(closes) - 1)
    return [
        Bar(ts=start + timedelta(minutes=i), close=float(price))
        for i, price in enumerate(closes)
    ]


def _future_return(entry: float, future_price: float | None) -> float | None:
    if future_price is None or not entry:
        return None
    return float(future_price) / entry - 1


def render_signal_card(result: SignalResult, market: MarketSnapshot) -> Panel:
    b = result.breakdown
    body = Text()
    body.append(f"{market.xau.code} SIGNAL\n", style="bold")
    body.append(result.timestamp.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    body.append(f"Signal:      {result.signal.value}\n", style=_signal_style(result.signal))
    body.append(f"Score:       {result.score:+d}\n")
    body.append(f"Strength:    {result.strength}/100\n\n")
    body.append("News:\n")
    body.append((result.news or "(none)") + "\n\n")
    body.append(f"News Impact:\n{result.news_impact_label} {b.news:+d}\n\n")
    body.append("After the news, from the price at the headline:\n")
    anchor = "no anchor" if result.anchor_price is None else f"anchor {result.anchor_price:.4f}"
    body.append(f"{anchor}\n")
    body.append(f"{market.xau.code} +15s   {_fmt_ret(result.reaction_15s)}\n")
    body.append(f"{market.xau.code} +30s   {_fmt_ret(result.reaction_30s)}\n")
    body.append(f"{market.xau.code} +1m    {_fmt_ret(result.reaction_1m)}   {b.gold_1m:+d}\n")
    body.append(f"{market.xau.code} +3m    {_fmt_ret(result.reaction_3m)}   {b.gold_3m:+d}\n")
    body.append(f"{market.xau.code} +5m    {_fmt_ret(result.reaction_5m)}\n\n")
    body.append("Confirmation, same window:\n")
    body.append(f"{market.xag.code} +1m    {_fmt_ret(result.xag_1m) if result.reaction_1m is not None else 'waiting'}   {b.silver:+d}\n")
    if market.eurusd.code == "FLAT":
        body.append("第二确认   无\n\n")
    else:
        third = None if result.reaction_1m is None else result.eurusd_1m
        body.append(f"{market.eurusd.code} +1m    {_fmt_ret(third)}   {b.eurusd:+d}\n\n")
    body.append("Reason:\n")
    body.append(result.reason + "\n")
    if result.return_1m is not None:
        body.append(
            "\nForward: "
            f"1m={result.return_1m:+.2%} "
            f"5m={_fmt_ret(result.return_5m)} "
            f"15m={_fmt_ret(result.return_15m)} "
            f"30m={_fmt_ret(result.return_30m)}\n"
        )
    return Panel(body, title="==========", subtitle="==========")


def render_tape(rows: list[PricedQuote]) -> Panel:
    body = Text()
    body.append("Prices only. These do not create a gold BUY/SELL.\n\n", style="bold")
    by_group: dict[str, list[PricedQuote]] = {group: [] for group in GROUP_ORDER}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)
    for group in GROUP_ORDER:
        body.append(GROUP_LABELS.get(group, group) + "\n", style="bold")
        for row in by_group.get(group, []):
            if row.price is None:
                body.append(f"  {row.name:<8} {row.code:<8} MISSING  {row.missing or ''}\n")
                continue
            ts = row.ts.strftime("%H:%M:%S") if row.ts else "no-ts"
            body.append(
                f"  {row.name:<8} {row.code:<8} {row.price:.4f}  1d {_fmt_signed(row.change_1d)}  {ts}  [{row.source}]\n"
            )
        body.append("\n")
    return Panel(body, title="Priced tape", subtitle="missing stays missing")


def render_polymarket(board: PolymarketBoard) -> Panel:
    body = Text()
    body.append("Prediction odds. These do not create a BUY or SELL.\n\n", style="bold")
    current = ""
    for row in board.contracts:
        if row.topic != current:
            current = row.topic
            body.append(f"{row.topic}: {row.event}\n", style="bold")
        if row.yes is None:
            body.append(f"  {row.question[:72]}  MISSING  {row.missing or ''}\n")
            continue
        updated = (row.updated_at or "no-ts")[:19]
        channel = "clob" if "CLOB" in row.source else "gamma"
        body.append(
            f"  Yes {row.yes:.1%}  {channel}  {updated}  {row.question[:60]}\n"
        )
    if board.missing:
        body.append("\nMissing\n", style="bold")
        for item in board.missing:
            body.append(f"  • {item}\n")
    return Panel(body, title="Polymarket", subtitle="odds are not a trade")


def render_europe_panel(verdict: EuropeVerdict) -> Panel:
    snap = verdict.snapshot
    body = Text()
    body.append(f"{snap.as_of.strftime('%Y-%m-%d %H:%M:%S %Z')}\n", style="bold")
    body.append(f"Stage:  {verdict.stage}\n", style=_europe_stage_style(verdict.stage))
    body.append(f"Driver: {verdict.driver}\n\n")

    body.append("Yields (source on each line)\n", style="bold")
    body.append(_yield_line("OAT 10Y", snap.oat) + "\n")
    body.append(_yield_line("Bund 10Y", snap.bund) + "\n")
    body.append(_yield_line("Italy 10Y", snap.italy) + "\n")
    body.append(_yield_line("Gilt 10Y", snap.gilt) + "\n")
    body.append(
        f"OAT-Bund   {_fmt_bp(snap.oat_bund_bp)}   Italy-Bund {_fmt_bp(snap.italy_bund_bp)}\n\n"
    )

    body.append("FX / banks\n", style="bold")
    body.append(
        f"EURUSD { _fmt_px(snap.eurusd) }  1d {_fmt_signed(snap.eurusd_1d)}\n"
        f"GBPUSD { _fmt_px(snap.gbpusd) }  1d {_fmt_signed(snap.gbpusd_1d)}\n"
        f"EURGBP { _fmt_px(snap.eurgbp) }  1d {_fmt_signed(snap.eurgbp_1d)}\n"
        f"CAC40  1d {_fmt_signed(snap.cac40_1d)}   "
        f"FR banks 1d {_fmt_signed(snap.french_banks_1d)}   "
        f"banks vs CAC {_fmt_signed(snap.banks_vs_cac_1d)}\n\n"
    )

    body.append("Separate transmission — not one gold SELL\n", style="bold")
    body.append(f"EUR              {verdict.eur}\n")
    body.append(f"GBP vs EUR       {verdict.gbp_vs_eur}\n")
    body.append(f"GBP vs USD       {verdict.gbp_vs_usd}\n")
    body.append(f"French domestic  {verdict.french_domestic}\n")
    body.append(f"French banks     {verdict.french_banks}\n")
    body.append(f"French exporters {verdict.french_exporters}\n")
    body.append(f"Gold bias        {verdict.gold}\n")
    body.append("Gold BUY/SELL still waits for XAUUSD confirmation.\n\n")

    body.append("Evidence\n", style="bold")
    if verdict.evidence:
        for item in verdict.evidence:
            body.append(f"  • {item}\n")
    else:
        body.append("  (none)\n")
    body.append("\nNot proven / missing\n", style="bold")
    holes = list(verdict.not_proven) + [f"missing: {m}" for m in snap.missing]
    if holes:
        for item in holes:
            body.append(f"  • {item}\n")
    else:
        body.append("  (none)\n")
    return Panel(body, title="Europe credit / fragmentation", subtitle="numbers without a source are not used")


def _yield_line(label: str, point) -> str:
    if point is None:
        return f"{label:<10} MISSING"
    ts = point.ts.strftime("%H:%M:%S") if point.ts else "no-ts"
    chg = f"{point.change_bp:+.1f}bp" if point.change_bp is not None else "chg n/a"
    return f"{label:<10} {point.yield_pct:.3f}%  {chg}  {ts}  [{point.source}]"


def _fmt_bp(value: float | None) -> str:
    return "MISSING" if value is None else f"{value:.1f}bp"


def _fmt_px(value: float | None) -> str:
    return "MISSING" if value is None else f"{value:.5f}"


def _fmt_signed(value: float | None) -> str:
    return "MISSING" if value is None else f"{value:+.2%}"


def _europe_stage_style(stage: str) -> str:
    if stage in ("EZ_FRAGMENTATION", "FRANCE_STRESS"):
        return "bold red"
    if stage == "FRANCE_REPRICING":
        return "bold yellow"
    if stage == "INSUFFICIENT":
        return "bold magenta"
    return "bold green"


def render_dashboard(
    market: MarketSnapshot,
    news: FlashNews | None,
    result: SignalResult,
    history: list[SignalResult],
    mode: str,
    europe: EuropeVerdict | None = None,
    polymarket: PolymarketBoard | None = None,
) -> Table:
    layout = Table.grid(expand=True)
    header = Text(f"Gold Signal Engine V0  [{mode}]", style="bold")
    prices = Table(show_header=False, box=None)
    prices.add_row(market.xau.code, f"{market.xau.price:.2f}")
    prices.add_row("+1m", _fmt_ret(result.reaction_1m))
    prices.add_row("+3m", _fmt_ret(result.reaction_3m))
    prices.add_row(market.xag.code, _fmt_ret(None if result.reaction_1m is None else result.xag_1m))
    prices.add_row(market.eurusd.code, _fmt_ret(None if result.reaction_1m is None else result.eurusd_1m))

    event = Text()
    if news:
        age = news_age_seconds(news, result.timestamp)
        event.append((news.text[:160] or "") + "\n")
        event.append(f"Gold Impact: {result.news_impact_label}\n")
        event.append(f"Age: {int(age)} sec")
    else:
        event.append("无有效相关新闻")

    scores = Table(show_header=False, box=None)
    scores.add_row("News", f"{result.breakdown.news:+d}")
    scores.add_row(f"{market.xau.code} 1m", f"{result.breakdown.gold_1m:+d}")
    scores.add_row(f"{market.xau.code} 3m", f"{result.breakdown.gold_3m:+d}")
    scores.add_row(market.xag.code, f"{result.breakdown.silver:+d}")
    scores.add_row(market.eurusd.code, f"{result.breakdown.eurusd:+d}")
    scores.add_row("TOTAL", f"{result.breakdown.total:+d}")

    sig = Text(
        f"{result.signal.value}\nStrength {result.strength}/100",
        style=_signal_style(result.signal),
        justify="center",
    )

    recent = Table(title="最近10个信号")
    recent.add_column("time")
    recent.add_column("signal")
    recent.add_column("score")
    recent.add_column("news")
    for item in history[-10:]:
        recent.add_row(
            item.timestamp.strftime("%H:%M:%S"),
            item.signal.value,
            f"{item.score:+d}",
            (item.news or "")[:40],
        )

    layout.add_row(header)
    layout.add_row(prices)
    layout.add_row(Panel(event, title="Latest Event"))
    layout.add_row(Panel(scores, title="Score"))
    layout.add_row(Panel(sig, title="SIGNAL"))
    if europe is not None:
        compact = Text()
        compact.append(f"{europe.stage} / {europe.driver}\n", style=_europe_stage_style(europe.stage))
        compact.append(
            f"OAT-Bund {_fmt_bp(europe.snapshot.oat_bund_bp)}  "
            f"Italy-Bund {_fmt_bp(europe.snapshot.italy_bund_bp)}  "
            f"EUR {europe.eur}  banks {europe.french_banks}  gold {europe.gold}\n"
        )
        if europe.not_proven:
            compact.append("Not proven: " + europe.not_proven[0])
        layout.add_row(Panel(compact, title="Europe (not a gold BUY/SELL)"))
    if polymarket is not None and polymarket.contracts:
        odds = Text()
        for row in polymarket.contracts[:6]:
            yes = "MISSING" if row.yes is None else f"{row.yes:.0%}"
            odds.append(f"{row.topic} {yes}  {row.question[:48]}\n")
        layout.add_row(Panel(odds, title="Polymarket (not a trade)"))
    layout.add_row(recent)
    return layout


def _signal_style(signal: SignalSide) -> str:
    if signal == SignalSide.BUY:
        return "bold green"
    if signal == SignalSide.SELL:
        return "bold red"
    return "yellow"


def _expected_ok(expected: str, actual: SignalSide) -> bool:
    if expected == "HOLD_OR_SELL":
        return actual in (SignalSide.HOLD, SignalSide.SELL)
    if expected == "HOLD_OR_BUY":
        return actual in (SignalSide.HOLD, SignalSide.BUY)
    return actual.value == expected


def _fmt_ret(value: float | None) -> str:
    return f"{value:+.2%}" if value is not None else "waiting"


if __name__ == "__main__":
    raise SystemExit(main())
