from datetime import datetime, timedelta, timezone

from gold_signal.market_context import build_market_context
from gold_signal.observation import EventRecord, Observation


def _utc(hour: int) -> datetime:
    return datetime(2026, 9, 28, hour, tzinfo=timezone.utc)


def _close(factor: str, symbol: str, hour: int, price: float) -> Observation:
    stamp = _utc(hour)
    return Observation(factor, price, stamp, stamp, stamp, "fixture", symbol)


def test_context_uses_only_series_that_are_already_visible():
    now = _utc(12)
    observations = [
        _close("market.XAUUSD.close", "XAUUSD", 10, 100.0),
        _close("market.XAUUSD.close", "XAUUSD", 11, 101.0),
        _close("market.XAGUSD.close", "XAGUSD", 10, 40.0),
        _close("market.XAGUSD.close", "XAGUSD", 11, 40.5),
    ]
    events = [
        EventRecord(
            "strike",
            "GEOPOLITICAL_ESCALATION",
            now - timedelta(hours=2),
            now - timedelta(hours=2),
            now - timedelta(hours=2),
            "办公楼遭袭",
            "办公楼遭袭",
            "jin10",
        )
    ]
    context = build_market_context(observations, events, now)
    assert context["gold_trend"] == "BULLISH"
    assert context["silver"] == "CONFIRMING"
    assert context["oil"] == "UNKNOWN"
    assert context["usd"] == "UNKNOWN"
    assert context["real_yield"] == "UNKNOWN"
    assert context["risk"] == "UNKNOWN"
    assert context["geopolitical"] == "ELEVATED"


def test_packaged_archive_context_leaves_missing_factors_unknown():
    from gold_signal.market_context import context_from_default_archive

    context = context_from_default_archive()
    assert context["gold_trend"] in {"BULLISH", "BEARISH", "FLAT"}
    assert context["oil"] == "UNKNOWN"
    assert context["usd"] == "UNKNOWN"
    assert context["risk"] == "UNKNOWN"
    assert context["geopolitical"] in {"ELEVATED", "QUIET"}


def test_silver_divergence_and_a_stale_headline_are_not_elevated():
    now = _utc(12)
    observations = [
        _close("market.XAUUSD.close", "XAUUSD", 10, 100.0),
        _close("market.XAUUSD.close", "XAUUSD", 11, 101.0),
        _close("market.XAGUSD.close", "XAGUSD", 10, 40.0),
        _close("market.XAGUSD.close", "XAGUSD", 11, 39.0),
    ]
    events = [
        EventRecord(
            "old",
            "GEOPOLITICAL_ESCALATION",
            now - timedelta(days=3),
            now - timedelta(days=3),
            now - timedelta(days=3),
            "空袭",
            "空袭",
            "jin10",
        )
    ]
    context = build_market_context(observations, events, now)
    assert context["silver"] == "DIVERGING"
    assert context["geopolitical"] == "QUIET"
