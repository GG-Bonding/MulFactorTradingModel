from __future__ import annotations

import statistics
from datetime import datetime

from gold_signal.models import AssetMove, Bar, MarketSnapshot, Thresholds


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


def snapshot(
    xau_bars: list[Bar],
    xag_bars: list[Bar],
    eurusd_bars: list[Bar],
    *,
    xau_price: float | None = None,
    xag_price: float | None = None,
    eurusd_price: float | None = None,
    as_of: datetime | None = None,
) -> MarketSnapshot:
    xau = asset_move("XAUUSD", xau_bars, xau_price)
    return MarketSnapshot(
        xau=xau,
        xag=asset_move("XAGUSD", xag_bars, xag_price),
        eurusd=asset_move("EURUSD", eurusd_bars, eurusd_price),
        as_of=as_of or max(bar.ts for bar in xau_bars),
    )


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
