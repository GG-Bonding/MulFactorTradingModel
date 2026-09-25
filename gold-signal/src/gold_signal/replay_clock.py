from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from gold_signal.observation import CoverageManifest, EventRecord, FactorResolver, Observation


@dataclass(frozen=True)
class DecisionRecord:
    event_id: str
    hypothesis_id: str
    evaluated_at: datetime
    side: str
    entry_at: datetime | None
    entry_price: float | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostModel:
    """Fixed round-trip hurdle subtracted from signed gross return.

    The numbers are a pessimistic constant, not a spread fitted to the sample.
    """

    spread_cost: float = 0.0002
    slippage_cost: float = 0.0003
    commission: float = 0.0

    @property
    def total(self) -> float:
        return self.spread_cost + self.slippage_cost + self.commission


@dataclass(frozen=True)
class TradeOutcome:
    """PnL from the entry print. The news anchor is not an entry.

    return_from_entry is the raw price change. gross_return and net_return
    are signed in the trade direction, and net_return is after costs.
    """

    return_from_entry: float | None
    gross_return: float | None = None
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    net_return: float | None = None
    mfe: float | None = None
    mae: float | None = None
    horizon: str | None = None


def raw_return(entry_price: float, price: float) -> float:
    return price / entry_price - 1


def signed_return(side: str, entry_price: float, price: float) -> float:
    change = raw_return(entry_price, price)
    if side in ("SHORT", "SELL"):
        return -change
    return change


def fill_price(
    side: str,
    price: float | None,
    observed_at: datetime | None,
    ready_at: datetime,
) -> tuple[datetime | None, float | None]:
    """The fill is the print at the decision. A price from before the minute is not an entry."""
    if side not in ("LONG", "SHORT", "BUY", "SELL"):
        return None, None
    if price is None or observed_at is None or observed_at < ready_at:
        return None, None
    return observed_at, price


def measure_outcome(
    *,
    side: str,
    entry_price: float,
    entry_at: datetime,
    path: list[tuple[datetime, float]],
    exit_price: float | None,
    costs: CostModel | None = None,
    horizon: str | None = None,
) -> TradeOutcome:
    costs = costs or CostModel()
    if exit_price is None or entry_price == 0:
        return TradeOutcome(
            return_from_entry=None,
            gross_return=None,
            spread_cost=costs.spread_cost,
            slippage_cost=costs.slippage_cost,
            commission=costs.commission,
            net_return=None,
            mfe=None,
            mae=None,
            horizon=horizon,
        )
    gross = signed_return(side, entry_price, exit_price)
    signed_path = [signed_return(side, entry_price, price) for stamp, price in path if stamp > entry_at]
    if not signed_path:
        signed_path = [gross]
    return TradeOutcome(
        return_from_entry=raw_return(entry_price, exit_price),
        gross_return=gross,
        spread_cost=costs.spread_cost,
        slippage_cost=costs.slippage_cost,
        commission=costs.commission,
        net_return=gross - costs.total,
        mfe=max(signed_path),
        mae=min(signed_path),
        horizon=horizon,
    )


def reaction_1m(resolver: FactorResolver, factor: str, published_at: datetime, now: datetime) -> float | None:
    anchor = resolver.price_at(factor, published_at, now)
    later = resolver.price_at(factor, published_at + timedelta(minutes=1), now)
    if anchor is None or later is None or anchor.value == 0:
        return None
    if later.observed_at < published_at + timedelta(minutes=1):
        return None
    return later.value / anchor.value - 1


def replay_gold_confirmation(
    event: EventRecord,
    prices: list[Observation],
    *,
    hypothesis_id: str = "gold_v0",
    factor: str = "market.XAUUSD.close",
) -> list[tuple[datetime, DecisionRecord | None, TradeOutcome | None]]:
    """Step the clock. A 1-minute reaction cannot be used before that minute exists."""
    resolver = FactorResolver(prices)
    ready_at = event.published_at + timedelta(minutes=1)
    marks = [
        event.published_at + timedelta(seconds=59),
        ready_at,
        ready_at + timedelta(minutes=1),
    ]
    rows: list[tuple[datetime, DecisionRecord | None, TradeOutcome | None]] = []
    decision: DecisionRecord | None = None
    for now in marks:
        if now < event.available_at:
            rows.append((now, None, None))
            continue
        change = reaction_1m(resolver, factor, event.published_at, now)
        if change is None or change <= 0:
            rows.append((now, None, None))
            continue
        if decision is None:
            entry = resolver.price_at(factor, now, now)
            entry_at, entry_price = fill_price(
                "LONG",
                None if entry is None else entry.value,
                None if entry is None else entry.observed_at,
                ready_at,
            )
            if entry_price is None or entry_at is None:
                rows.append((now, None, None))
                continue
            decision = DecisionRecord(
                event_id=event.event_id,
                hypothesis_id=hypothesis_id,
                evaluated_at=now,
                side="LONG",
                entry_at=entry_at,
                entry_price=entry_price,
            )
        outcome = None
        if decision.entry_price and decision.entry_at is not None:
            last = resolver.price_at(factor, now, now)
            if last is not None and last.observed_at > decision.entry_at:
                path = [
                    (row.observed_at, row.value)
                    for row in resolver.visible(factor, now)
                    if decision.entry_at < row.observed_at <= now
                ]
                outcome = measure_outcome(
                    side=decision.side,
                    entry_price=decision.entry_price,
                    entry_at=decision.entry_at,
                    path=path,
                    exit_price=last.value,
                )
        rows.append((now, decision, outcome))
    return rows


def coverage_status(manifest: CoverageManifest, factors: list[str], start: datetime, end: datetime) -> dict:
    gaps = manifest.missing(factors, start, end)
    if gaps:
        return {"status": "INSUFFICIENT_HISTORY", "samples": 0, "missing": gaps}
    return {"status": "OK", "samples": 0, "missing": []}
