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

from gold_signal.jin10 import SHANGHAI, Jin10Client, Jin10Error, parse_time
from gold_signal.market import fetch_binance_bars, fetch_binance_price, snapshot
from gold_signal.models import Bar, FlashNews, MarketSnapshot, SignalResult, SignalSide, Thresholds
from gold_signal.news import classify_news, news_age_seconds, news_weight
from gold_signal.signal import SignalEngine, SignalStore

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
DATA_PATH = ROOT / "data" / "signals.jsonl"
CONSOLE = Console()


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Gold Realtime Signal Engine V0")
    parser.add_argument("--mode", choices=("replay", "live"), required=True)
    parser.add_argument("--book", choices=("gold", "btc"), default="gold")
    parser.add_argument("--interval", type=int, default=Thresholds.POLL_SECONDS)
    parser.add_argument("--ticks", type=int, default=0, help="live ticks then exit; 0 = run forever")
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)

    store = SignalStore(args.output)
    engine = SignalEngine()

    try:
        if args.mode == "replay":
            return run_replay(engine, store)
        return run_live(engine, store, interval=args.interval, ticks=args.ticks, book=args.book)
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
    try:
        init = client.connect()
        CONSOLE.print(f"Jin10 connected. tools={client.tool_names()} book={book}")
        CONSOLE.print(f"initialize protocol={init.get('protocolVersion')}")
        pending: list[dict[str, Any]] = []
        history: list[SignalResult] = []
        tape = QuoteTape()
        tick = 0
        last_market = None
        with Live(console=CONSOLE, refresh_per_second=4) as live:
            while True:
                tick += 1
                news_list = client.list_flash()
                if book == "btc":
                    try:
                        news_list = merge_news(news_list, client.search_flash("比特币"))
                    except Jin10Error as exc:
                        CONSOLE.print(f"[yellow]search_flash(比特币) failed: {exc}[/yellow]")
                news = pick_news(news_list, datetime.now(tz=SHANGHAI))
                market = fetch_market(client, tape, book=book)
                last_market = market
                now = market.as_of
                result = engine.evaluate(news, market, now=now)
                store.append(result)
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
                live.update(render_dashboard(market, news, result, history[-10:], mode=f"live/{book}"))
                if ticks and tick >= ticks:
                    break
                time.sleep(interval)
        if history and last_market is not None:
            CONSOLE.print(render_signal_card(history[-1], last_market))
    finally:
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


def pick_news(items: list[FlashNews], now: datetime) -> FlashNews | None:
    ranked: list[tuple[int, float, FlashNews]] = []
    for item in items:
        impact = classify_news(item.text)
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
    body.append(f"Confidence:  {int(result.confidence * 100)}%\n\n")
    body.append("News:\n")
    body.append((result.news or "(none)") + "\n\n")
    body.append(f"News Impact:\n{result.news_impact_label} {b.news:+d}\n\n")
    body.append("Market Reaction:\n")
    body.append(f"{market.xau.code} 1m     {market.xau.return_1m:+.2%}   {b.gold_1m:+d}\n")
    body.append(f"{market.xau.code} 3m     {market.xau.return_3m:+.2%}   {b.gold_3m:+d}\n\n")
    body.append("Confirmation:\n")
    body.append(f"{market.xag.code} 1m     {market.xag.return_1m:+.2%}   {b.silver:+d}\n")
    body.append(f"{market.eurusd.code} 1m     {market.eurusd.return_1m:+.2%}   {b.eurusd:+d}\n\n")
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


def render_dashboard(
    market: MarketSnapshot,
    news: FlashNews | None,
    result: SignalResult,
    history: list[SignalResult],
    mode: str,
) -> Table:
    layout = Table.grid(expand=True)
    header = Text(f"Gold Signal Engine V0  [{mode}]", style="bold")
    prices = Table(show_header=False, box=None)
    prices.add_row(market.xau.code, f"{market.xau.price:.2f}")
    prices.add_row("1m", f"{market.xau.return_1m:+.2%}")
    prices.add_row("3m", f"{market.xau.return_3m:+.2%}")
    prices.add_row(market.xag.code, f"{market.xag.return_1m:+.2%}")
    prices.add_row(market.eurusd.code, f"{market.eurusd.return_1m:+.2%}")

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
        f"{result.signal.value}\nConfidence {int(result.confidence * 100)}%",
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
    return f"{value:+.2%}" if value is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
