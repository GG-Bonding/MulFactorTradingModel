from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from gold_signal.jin10 import Jin10Client, Jin10Error
from gold_signal.market import fetch_binance_bars, fetch_binance_price, snapshot
from gold_signal.models import Bar, FlashNews, MarketSnapshot, NewsImpact, Thresholds
from gold_signal.news import classify_btc_news, classify_gold_news, news_age_seconds, news_weight

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

_WAR = re.compile(r"(战争|开战|空袭|导弹|军事冲突|冲突升级|地缘政治|袭击|遭袭|霍尔木兹)")
_US_MACRO = ("非农", "CPI", "PCE", "美联储", "FOMC", "Powell", "鲍威尔", "美债", "美国国债", "实际利率", "TIPS", "降息", "加息", "鹰派", "鸽派")
_FRANCE_YIELD_UP = re.compile(r"(法国国债|OAT|欧债).{0,16}(收益率)?.{0,8}(上涨|上升|走高|飙升|上行)")
_FRANCE_YIELD_DOWN = re.compile(r"(法国国债|OAT|欧债).{0,16}(收益率)?.{0,8}(下跌|回落|走低|下行)")
_OIL_UP = re.compile(r"(原油|WTI|布伦特|石油|OPEC).{0,16}(上涨|减产|紧缺|中断|飙升|供应紧张)")
_OIL_DOWN = re.compile(r"(原油|WTI|布伦特|石油|OPEC).{0,16}(下跌|增产|过剩|需求疲软|累库|供应增加)")


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
    raw = (text or "").strip()
    if family == "metal":
        return classify_gold_news(raw)
    if family == "oil":
        return _oil_impact(raw)
    if family == "eur":
        return _eur_impact(raw)
    if family == "dollar":
        return _dollar_impact(raw)
    if family == "nasdaq":
        return _nasdaq_impact(raw)
    if family == "crypto":
        return _crypto_impact(raw)
    return _none(f"未知产品族 {family}")


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


def _oil_impact(text: str) -> NewsImpact:
    if not text:
        return _none("empty")
    if _OIL_DOWN.search(text):
        return _hit(-1, "原油供应增加或需求走弱")
    if _OIL_UP.search(text) or _WAR.search(text):
        return _hit(1, "原油供应收紧或地缘冲击供给")
    return _none("这条新闻没有原油方向")


def _eur_impact(text: str) -> NewsImpact:
    if not text:
        return _none("empty")
    if _WAR.search(text):
        return _hit(-1, "风险事件偏美元，欧元承压")
    if _FRANCE_YIELD_UP.search(text):
        return _hit(-1, "法国或欧债溢价上升，欧元承压")
    if _FRANCE_YIELD_DOWN.search(text):
        return _hit(1, "法国或欧债溢价回落，欧元压力减轻")
    if _has_us_macro(text):
        gold = classify_gold_news(text)
        if gold.direction != 0:
            return _hit(gold.direction, "美元宏观：欧元与黄金同向于美元强弱")
    return _none("这条新闻没有欧元方向")


def _dollar_impact(text: str) -> NewsImpact:
    if not text or _WAR.search(text) or not _has_us_macro(text):
        return _none("这条新闻没有美元兑日元方向")
    gold = classify_gold_news(text)
    if gold.direction == 0:
        return _none("美国宏观方向不确定，不映射美日")
    return _hit(-gold.direction, "美元宏观：美日与黄金方向相反")


def _nasdaq_impact(text: str) -> NewsImpact:
    if not text:
        return _none("empty")
    if _WAR.search(text):
        return _hit(-1, "风险事件偏空股指")
    if _has_us_macro(text):
        gold = classify_gold_news(text)
        if gold.direction != 0:
            return _hit(gold.direction, "美国利率新闻：股指与黄金同向于宽松或收紧")
    return _none("这条新闻没有纳指方向")


def _crypto_impact(text: str) -> NewsImpact:
    return classify_btc_news(text)


def _has_us_macro(text: str) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in _US_MACRO)


def _hit(direction: int, reason: str) -> NewsImpact:
    return NewsImpact(direction=direction, importance=2, confidence=0.75, reason=reason)


def _none(reason: str) -> NewsImpact:
    return NewsImpact(direction=0, importance=0, confidence=0.0, reason=reason)
