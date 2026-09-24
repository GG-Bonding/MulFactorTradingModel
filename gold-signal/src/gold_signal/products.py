from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from gold_signal.jin10 import Jin10Client, Jin10Error
from gold_signal.market import fetch_binance_bars, fetch_binance_price, snapshot
from gold_signal.models import Bar, FlashNews, MarketSnapshot, NewsImpact, Thresholds
from gold_signal.news import news_age_seconds, news_weight
from gold_signal.transmission import apply_transmission

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

@dataclass(frozen=True)
class ProductSpec:
    code: str
    name: str
    family: str
    source: str
    confirm: str
    confirm_source: str
    third: str | None = None
    third_source: str | None = None


# One signal per product. Confirm legs must rise when this product rises.
SIGNAL_PRODUCTS: tuple[ProductSpec, ...] = (
    ProductSpec("XAUUSD", "黄金", "metal", "jin10", "XAGUSD", "jin10", "EURUSD", "jin10"),
    ProductSpec("XAGUSD", "白银", "metal", "jin10", "XAUUSD", "jin10", "EURUSD", "jin10"),
    ProductSpec("EURUSD", "欧元", "eur", "jin10", "GBPUSD", "jin10", "AUDUSD", "jin10"),
    ProductSpec("USDJPY", "美日", "dollar", "jin10", "USDCHF", "jin10", "USDCAD", "jin10"),
    ProductSpec("USOIL", "WTI", "oil", "jin10", "UKOIL", "jin10", "EURUSD", "jin10"),
    ProductSpec("UKOIL", "布伦特", "oil", "jin10", "USOIL", "jin10", "EURUSD", "jin10"),
    ProductSpec("NQ=F", "纳指", "nasdaq", "yahoo", "ES=F", "yahoo", "YM=F", "yahoo"),
    ProductSpec("BTCUSDT", "比特币", "crypto", "binance", "ETHUSDT", "binance"),
)


def impact_for_product(text: str, family: str) -> NewsImpact:
    return apply_transmission(text, family)


def pick_product_news(items: list[FlashNews], now: datetime, family: str) -> FlashNews | None:
    ranked: list[tuple[int, float, FlashNews]] = []
    for item in items:
        impact = impact_for_product(item.text, family)
        if impact.importance <= 0 or impact.direction == 0:
            continue
        age = news_age_seconds(item, now)
        if news_weight(age) <= 0:
            continue
        ranked.append((impact.importance, -age, item))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return ranked[0][2]


def fetch_signal_board(
    jin10: Jin10Client | None,
) -> tuple[list[tuple[ProductSpec, MarketSnapshot]], list[str]]:
    book = _QuoteBook(jin10)
    markets: list[tuple[ProductSpec, MarketSnapshot]] = []
    missing: list[str] = []
    now = datetime.now(tz=timezone.utc)
    for spec in SIGNAL_PRODUCTS:
        try:
            primary_bars, primary_px = book.load(spec.code, spec.source, now)
            confirm_bars, confirm_px = book.load(spec.confirm, spec.confirm_source, now)
            if spec.third and spec.third_source:
                third_bars, third_px = book.load(spec.third, spec.third_source, now)
                third_code = spec.third
            else:
                third_bars = _flat_bars(primary_px, now)
                third_px = primary_px
                third_code = "FLAT"
            if min(len(primary_bars), len(confirm_bars), len(third_bars)) < 2:
                missing.append(f"{spec.code} 1m bars missing")
                continue
            markets.append(
                (
                    spec,
                    snapshot(
                        primary_bars,
                        confirm_bars,
                        third_bars,
                        xau_price=primary_px,
                        xag_price=confirm_px,
                        eurusd_price=third_px,
                        as_of=now,
                        primary_code=spec.code,
                        confirm_code=spec.confirm,
                        dollar_code=third_code,
                    ),
                )
            )
        except (Jin10Error, RuntimeError, ValueError) as exc:
            missing.append(f"{spec.code}: {exc}")
    return markets, missing


class _QuoteBook:
    def __init__(self, jin10: Jin10Client | None) -> None:
        self.jin10 = jin10
        self._bars: dict[tuple[str, str], list[Bar]] = {}
        self._px: dict[tuple[str, str], float] = {}

    def load(self, code: str, source: str, now: datetime) -> tuple[list[Bar], float]:
        key = (source, code)
        if key not in self._bars:
            bars, price = _load_series(self.jin10, code, source, now)
            self._bars[key] = bars
            self._px[key] = price
        return self._bars[key], self._px[key]


def _load_series(
    jin10: Jin10Client | None,
    code: str,
    source: str,
    now: datetime,
) -> tuple[list[Bar], float]:
    count = Thresholds.KLINE_MINUTES + 1
    if source == "jin10":
        if jin10 is None:
            raise RuntimeError(f"{code} needs Jin10")
        quote = jin10.get_quote(code)
        bars = jin10.get_kline(code, count=count)
        if len(bars) < 2:
            bars = _flat_bars(quote.price, now, count)
        return bars, quote.price
    if source == "binance":
        return fetch_binance_bars(code, count=count), fetch_binance_price(code)
    if source == "yahoo":
        bars = fetch_yahoo_minutes(code, count=count)
        if len(bars) < 2:
            raise RuntimeError(f"yahoo 1m bars missing for {code}")
        return bars, bars[-1].close
    raise RuntimeError(f"unknown source {source}")


def fetch_yahoo_minutes(symbol: str, count: int = 31) -> list[Bar]:
    response = httpx.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={"range": "1d", "interval": "1m"},
        headers={"User-Agent": "Mozilla/5.0 gold-signal/0.2"},
        timeout=20.0,
        follow_redirects=True,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"yahoo HTTP {response.status_code} {symbol}")
    return parse_yahoo_minutes(response.json(), symbol)[-count:]


def parse_yahoo_minutes(payload: dict, code: str) -> list[Bar]:
    result = (payload.get("chart") or {}).get("result") if isinstance(payload, dict) else None
    if not result:
        raise RuntimeError(f"yahoo 1m payload empty for {code}")
    stamps = result[0].get("timestamp") or []
    quote = ((result[0].get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    bars: list[Bar] = []
    for ts, close in zip(stamps, closes):
        if close is None or ts is None:
            continue
        bars.append(Bar(ts=datetime.fromtimestamp(int(ts), tz=timezone.utc), close=float(close)))
    bars.sort(key=lambda bar: bar.ts)
    return bars


def _flat_bars(price: float, end: datetime, count: int = 31) -> list[Bar]:
    start = end - timedelta(minutes=count - 1)
    return [Bar(ts=start + timedelta(minutes=i), close=float(price)) for i in range(count)]
