from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import httpx

from gold_signal.models import AssetMove, Bar, EventReaction, MarketSnapshot, Thresholds

BINANCE_VISION = "https://data-api.binance.vision"


def compute_returns(bars: list[Bar], current_price: float | None = None) -> dict[str, float]:
    if len(bars) < 2:
        raise ValueError(f"need at least 2 bars to compute returns, got {len(bars)}")
    ordered = sorted(bars, key=lambda b: b.ts)
    current = float(current_price if current_price is not None else ordered[-1].close)
    return {
        "return_1m": _return_vs(ordered, current, minutes=1),
        "return_3m": _return_vs(ordered, current, minutes=3),
        "return_5m": _return_vs(ordered, current, minutes=5),
    }


def rolling_std_1m(bars: list[Bar]) -> float:
    ordered = sorted(bars, key=lambda b: b.ts)
    rets = minute_returns(ordered)
    if len(rets) < 2:
        return 0.0
    return float(statistics.stdev(rets))


def minute_returns(bars: list[Bar]) -> list[float]:
    ordered = sorted(bars, key=lambda b: b.ts)
    out: list[float] = []
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.close == 0:
            raise ValueError(f"zero close at {prev.ts}")
        out.append(cur.close / prev.close - 1)
    return out


def normalized_move(return_1m: float, std_1m: float) -> float:
    if std_1m == 0:
        return 0.0
    return return_1m / std_1m


def obvious_direction(value: float, z_score: float, threshold: float) -> int:
    if abs(value) >= threshold or abs(z_score) >= Thresholds.Z_OBVIOUS:
        if value > 0:
            return 1
        if value < 0:
            return -1
    return 0


def asset_move(code: str, bars: list[Bar], current_price: float | None = None) -> AssetMove:
    if not bars:
        raise ValueError(f"{code} has no bars")
    ordered = sorted(bars, key=lambda b: b.ts)
    price = float(current_price if current_price is not None else ordered[-1].close)
    rets = compute_returns(ordered, price)
    std = rolling_std_1m(ordered)
    z_1m = normalized_move(rets["return_1m"], std)
    return AssetMove(
        code=code,
        price=price,
        return_1m=rets["return_1m"],
        return_3m=rets["return_3m"],
        return_5m=rets["return_5m"],
        std_1m=std,
        z_1m=z_1m,
        obvious_1m=obvious_direction(rets["return_1m"], z_1m, Thresholds.RETURN_1M),
        obvious_3m=obvious_direction(
            rets["return_3m"],
            normalized_move(rets["return_3m"], std),
            Thresholds.RETURN_3M,
        ),
    )


def price_asof(points: list[Bar], when: datetime) -> Bar | None:
    eligible = [bar for bar in points if bar.ts <= when]
    if not eligible:
        return None
    return max(eligible, key=lambda bar: bar.ts)


def event_reaction(
    bars: list[Bar],
    published_at: datetime,
    now: datetime,
    *,
    extra: list[Bar] | None = None,
    max_anchor_gap: float = 90.0,
) -> EventReaction:
    """R(horizon) = P(news + horizon) / P(news) - 1.

    A horizon stays empty until that clock time has arrived and a later print
    exists inside the window. A rally that finished before the news is not a reaction.
    """
    points = sorted([*(bars or []), *(extra or [])], key=lambda bar: bar.ts)
    anchor = price_asof(points, published_at)
    if anchor is None or anchor.close == 0:
        return EventReaction()
    gap = (published_at - anchor.ts).total_seconds()
    if gap > max_anchor_gap:
        return EventReaction(
            anchor_price=anchor.close,
            anchor_ts=anchor.ts,
            anchor_gap_seconds=gap,
        )
    return EventReaction(
        anchor_price=anchor.close,
        anchor_ts=anchor.ts,
        anchor_gap_seconds=gap,
        return_15s=_horizon_return(points, anchor, published_at, now, 15),
        return_30s=_horizon_return(points, anchor, published_at, now, 30),
        return_1m=_horizon_return(points, anchor, published_at, now, 60),
        return_3m=_horizon_return(points, anchor, published_at, now, 180),
        return_5m=_horizon_return(points, anchor, published_at, now, 300),
    )


def _horizon_return(
    points: list[Bar],
    anchor: Bar,
    published_at: datetime,
    now: datetime,
    horizon_seconds: int,
) -> float | None:
    deadline = published_at + timedelta(seconds=horizon_seconds)
    if now < deadline:
        return None
    inside = [bar for bar in points if anchor.ts < bar.ts <= deadline]
    if not inside:
        return None
    return inside[-1].close / anchor.close - 1


def snapshot(
    xau_bars: list[Bar],
    xag_bars: list[Bar],
    eurusd_bars: list[Bar],
    *,
    xau_price: float | None = None,
    xag_price: float | None = None,
    eurusd_price: float | None = None,
    as_of: datetime | None = None,
    primary_code: str = "XAUUSD",
    confirm_code: str = "XAGUSD",
    dollar_code: str = "EURUSD",
) -> MarketSnapshot:
    xau = asset_move(primary_code, xau_bars, xau_price)
    return MarketSnapshot(
        xau=xau,
        xag=asset_move(confirm_code, xag_bars, xag_price),
        eurusd=asset_move(dollar_code, eurusd_bars, eurusd_price),
        as_of=as_of or max(bar.ts for bar in xau_bars),
        xau_bars=list(xau_bars),
        xag_bars=list(xag_bars),
        eurusd_bars=list(eurusd_bars),
    )


def parse_binance_klines(rows: list, code: str = "BTCUSDT") -> list[Bar]:
    bars: list[Bar] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            raise ValueError(f"{code} kline row is invalid: {row!r}")
        ts_ms = int(row[0])
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        bars.append(
            Bar(
                ts=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
            )
        )
    bars.sort(key=lambda b: b.ts)
    return bars


def fetch_binance_bars(symbol: str, count: int = 31) -> list[Bar]:
    payload = _binance_get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": "1m", "limit": count},
    )
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Binance klines for {symbol} were empty: {payload!r}")
    return parse_binance_klines(payload, symbol)


def fetch_binance_price(symbol: str) -> float:
    payload = _binance_get("/api/v3/ticker/price", {"symbol": symbol})
    if not isinstance(payload, dict) or "price" not in payload:
        raise RuntimeError(f"Binance ticker for {symbol} missing price: {payload!r}")
    return float(payload["price"])


def _binance_get(path: str, params: dict) -> object:
    url = BINANCE_VISION + path
    try:
        response = httpx.get(url, params=params, timeout=20.0)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Binance request failed {url}: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"Binance HTTP {response.status_code} {url}: {response.text[:400]}")
    return response.json()


def _return_vs(bars: list[Bar], current: float, minutes: int) -> float:
    if current == 0:
        raise ValueError("current price is 0")
    if len(bars) <= minutes:
        previous = bars[0].close
    else:
        previous = bars[-(minutes + 1)].close
    if previous == 0:
        raise ValueError("previous close is 0")
    return current / previous - 1
