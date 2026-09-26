from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from gold_signal.models import Bar, MarketSnapshot
from gold_signal.observation import EventRecord, FactorResolver, Observation
from gold_signal.research import observations_from_bars


def evaluation_now(event: EventRecord) -> datetime:
    """The first instant the one-minute reaction is allowed to exist."""
    ready = event.published_at + timedelta(minutes=1)
    if event.available_at > ready:
        return event.available_at
    return ready


class EvaluationContext:
    """Data for one decision. The engine must not care which subclass this is."""

    now: datetime

    def resolver(self) -> FactorResolver:
        raise NotImplementedError


@dataclass(frozen=True)
class ReplayEvaluationContext(EvaluationContext):
    observations: tuple[Observation, ...]
    now: datetime

    def resolver(self) -> FactorResolver:
        return FactorResolver(list(self.observations))


@dataclass(frozen=True)
class LiveEvaluationContext(EvaluationContext):
    observations: tuple[Observation, ...]
    now: datetime

    def resolver(self) -> FactorResolver:
        return FactorResolver(list(self.observations))

    @classmethod
    def from_market(cls, market: MarketSnapshot, event: EventRecord) -> LiveEvaluationContext:
        rows: list[Observation] = []
        for asset, bars in (
            (market.xau, market.xau_bars),
            (market.xag, market.xag_bars),
            (market.eurusd, market.eurusd_bars),
        ):
            if not asset.code or asset.code == "FLAT" or not bars:
                continue
            rows.extend(
                observations_from_bars(
                    f"market.{asset.code}.close",
                    list(bars),
                    symbol=asset.code,
                    source="live",
                    timestamp_kind="close",
                )
            )
        return cls(tuple(rows), evaluation_now(event))


def bars_as_observations(factor: str, bars: list[Bar], symbol: str) -> list[Observation]:
    return observations_from_bars(factor, bars, symbol=symbol, source="replay", timestamp_kind="close")
