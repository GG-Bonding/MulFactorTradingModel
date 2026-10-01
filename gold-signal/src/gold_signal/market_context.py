"""Market regime from visible factors. A missing series stays unknown."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gold_signal.hypothesis import matching_triggers
from gold_signal.observation import EventRecord, FactorResolver, Observation

_FLAT = 0.0005


def context_from_default_archive() -> dict:
    """Regime at the last stored bar. Missing factors stay unknown."""
    from gold_signal.archive import default_archive

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    factors = [
        "market.XAUUSD.close",
        "market.XAGUSD.close",
        "market.USOIL.close",
        "market.DXY.close",
        "market.VIX.close",
        "market.DFII10.close",
    ]
    events, observations, _gaps = default_archive().load(start, end, factors)
    if not observations:
        return build_market_context([], events, end)
    now = max(row.available_at for row in observations)
    return build_market_context(observations, events, now)


def build_market_context(
    observations: list[Observation],
    events: list[EventRecord],
    now: datetime,
) -> dict:
    resolver = FactorResolver(observations)
    gold = _change(resolver, "market.XAUUSD.close", now)
    silver = _change(resolver, "market.XAGUSD.close", now)
    oil = _change(resolver, "market.USOIL.close", now)
    dollar = _change(resolver, "market.DXY.close", now)
    vix = _change(resolver, "market.VIX.close", now)
    real_yield = _change(resolver, "market.DFII10.close", now)
    return {
        "timestamp": now.isoformat(),
        "gold_trend": _trend(gold),
        "silver": _silver(gold, silver),
        "oil": _rising(oil),
        "usd": _trend(dollar),
        "real_yield": _rising(real_yield),
        "risk": _risk(vix),
        "geopolitical": _geopolitical(events, now),
    }


def _change(resolver: FactorResolver, factor: str, now: datetime) -> float | None:
    rows = resolver.visible(factor, now)
    if len(rows) < 2 or rows[-2].value == 0:
        return None
    return rows[-1].value / rows[-2].value - 1


def _trend(change: float | None) -> str:
    if change is None:
        return "UNKNOWN"
    if change > _FLAT:
        return "BULLISH"
    if change < -_FLAT:
        return "BEARISH"
    return "FLAT"


def _rising(change: float | None) -> str:
    if change is None:
        return "UNKNOWN"
    if change > _FLAT:
        return "RISING"
    if change < -_FLAT:
        return "FALLING"
    return "FLAT"


def _silver(gold: float | None, silver: float | None) -> str:
    if gold is None or silver is None:
        return "UNKNOWN"
    if abs(gold) <= _FLAT or abs(silver) <= _FLAT:
        return "FLAT"
    if (gold > 0 and silver > 0) or (gold < 0 and silver < 0):
        return "CONFIRMING"
    return "DIVERGING"


def _risk(vix: float | None) -> str:
    if vix is None:
        return "UNKNOWN"
    if vix > _FLAT:
        return "RISK_OFF"
    if vix < -_FLAT:
        return "RISK_ON"
    return "FLAT"


def _geopolitical(events: list[EventRecord], now: datetime) -> str:
    if not events:
        return "UNKNOWN"
    recent = now - timedelta(hours=24)
    for event in events:
        if event.published_at < recent or event.published_at > now:
            continue
        text = f"{event.title} {event.content}"
        if event.event_type == "GEOPOLITICAL_ESCALATION" or "GEOPOLITICAL_ESCALATION" in matching_triggers(text):
            return "ELEVATED"
    return "QUIET"
