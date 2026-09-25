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


@dataclass(frozen=True)
class TradeOutcome:
    """PnL from the entry print. The news anchor is not an entry."""

    return_from_entry: float | None


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
            if entry is None or entry.observed_at < ready_at:
                rows.append((now, None, None))
                continue
            decision = DecisionRecord(
                event_id=event.event_id,
                hypothesis_id=hypothesis_id,
                evaluated_at=now,
                side="LONG",
                entry_at=entry.observed_at,
                entry_price=entry.value,
            )
        outcome = None
        if decision.entry_price:
            last = resolver.price_at(factor, now, now)
            if last is not None and last.observed_at > decision.entry_at:
                outcome = TradeOutcome(return_from_entry=last.value / decision.entry_price - 1)
        rows.append((now, decision, outcome))
    return rows


def coverage_status(manifest: CoverageManifest, factors: list[str], start: datetime, end: datetime) -> dict:
    gaps = manifest.missing(factors, start, end)
    if gaps:
        return {"status": "INSUFFICIENT_HISTORY", "samples": 0, "missing": gaps}
    return {"status": "OK", "samples": 0, "missing": []}
