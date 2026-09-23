from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gold_signal.europe import _http_json
from gold_signal.jin10 import SHANGHAI, Jin10Client, Jin10Error
from gold_signal.models import PricedQuote, Quote

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Jin10 quote://codes has oil and FX majors. It has no Nasdaq and no US shares.
# Nasdaq and the US basket come from Yahoo. The basket is liquid mega-caps, not a live heat ranking.
BOARD: tuple[tuple[str, str, str, str], ...] = (
    ("NQ=F", "纳指100期货", "nasdaq", "yahoo"),
    ("USOIL", "WTI原油", "oil", "jin10"),
    ("UKOIL", "布伦特原油", "oil", "jin10"),
    ("EURUSD", "欧元/美元", "fx", "jin10"),
    ("GBPUSD", "英镑/美元", "fx", "jin10"),
    ("USDJPY", "美元/日元", "fx", "jin10"),
    ("AUDUSD", "澳元/美元", "fx", "jin10"),
    ("USDCNH", "美元/人民币", "fx", "jin10"),
    ("USDCHF", "美元/瑞郎", "fx", "jin10"),
    ("NZDUSD", "纽元/美元", "fx", "jin10"),
    ("USDCAD", "美元/加元", "fx", "jin10"),
    ("NVDA", "英伟达", "us_equity", "yahoo"),
    ("AAPL", "苹果", "us_equity", "yahoo"),
    ("MSFT", "微软", "us_equity", "yahoo"),
    ("AMZN", "亚马逊", "us_equity", "yahoo"),
    ("GOOGL", "谷歌", "us_equity", "yahoo"),
    ("META", "Meta", "us_equity", "yahoo"),
    ("TSLA", "特斯拉", "us_equity", "yahoo"),
)

GROUP_ORDER = ("nasdaq", "oil", "fx", "us_equity")
GROUP_LABELS = {
    "nasdaq": "纳指",
    "oil": "石油",
    "fx": "外汇",
    "us_equity": "美股（固定流动性篮子，不是当日热度榜）",
}


def ups_percent_to_fraction(value: Any) -> float | None:
    """Jin10 ups_percent is already in percent points: '-0.307' means -0.307%."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("%", "").replace("+", "")) / 100.0
    except (TypeError, ValueError):
        return None


def quote_from_jin10(spec: tuple[str, str, str, str], quote: Quote) -> PricedQuote:
    code, name, group, _source = spec
    return PricedQuote(
        code=code,
        name=str(quote.raw.get("name") or name),
        group=group,
        price=quote.price,
        change_1d=ups_percent_to_fraction(quote.raw.get("ups_percent")),
        ts=quote.ts,
        source=f"jin10:{code}",
    )


def missing_quote(spec: tuple[str, str, str, str], reason: str) -> PricedQuote:
    code, name, group, source = spec
    return PricedQuote(
        code=code,
        name=name,
        group=group,
        source=source,
        missing=reason,
    )


def fetch_tape(jin10: Jin10Client | None = None) -> list[PricedQuote]:
    rows: list[PricedQuote] = []
    for spec in BOARD:
        code, _name, _group, source = spec
        if source == "jin10":
            rows.append(_fetch_jin10(spec, jin10))
        else:
            rows.append(_fetch_yahoo(spec, code))
    return rows


def _fetch_jin10(spec: tuple[str, str, str, str], jin10: Jin10Client | None) -> PricedQuote:
    code = spec[0]
    if jin10 is None:
        return missing_quote(spec, "no JIN10_TOKEN")
    try:
        return quote_from_jin10(spec, jin10.get_quote(code))
    except Jin10Error as exc:
        return missing_quote(spec, str(exc))


def _fetch_yahoo(spec: tuple[str, str, str, str], symbol: str) -> PricedQuote:
    code, name, group, _source = spec
    try:
        payload = _http_json(YAHOO_CHART_URL.format(symbol=symbol), {"range": "5d", "interval": "1d"})
    except RuntimeError as exc:
        return missing_quote(spec, str(exc))
    result = (payload.get("chart") or {}).get("result") if isinstance(payload, dict) else None
    if not result:
        return missing_quote(spec, f"yahoo:{symbol} empty")
    meta = result[0].get("meta") or {}
    price = _as_float(meta.get("regularMarketPrice"))
    prev = _as_float(meta.get("chartPreviousClose") or meta.get("previousClose"))
    change = None
    if price is not None and prev not in (None, 0):
        change = price / prev - 1
    ts = _unix_time(meta.get("regularMarketTime"))
    if price is None:
        return missing_quote(spec, f"yahoo:{symbol} missing price")
    return PricedQuote(
        code=code,
        name=name,
        group=group,
        price=price,
        change_1d=change,
        ts=ts,
        source=f"yahoo:{symbol}",
    )


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unix_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone(SHANGHAI)
    except (TypeError, ValueError, OSError):
        return None
