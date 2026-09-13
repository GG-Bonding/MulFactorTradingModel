from datetime import datetime, timedelta, timezone

from gold_signal.market import snapshot
from gold_signal.models import Bar, FlashNews, SignalSide
from gold_signal.news import classify_news
from gold_signal.signal import SignalEngine


def _bars(closes: list[float], end: datetime) -> list[Bar]:
    start = end - timedelta(minutes=len(closes) - 1)
    return [Bar(ts=start + timedelta(minutes=i), close=c) for i, c in enumerate(closes)]


def _news(text: str, now: datetime, age_sec: int = 60, event_id: str = "e1") -> FlashNews:
    return FlashNews(
        event_id=event_id,
        title=text,
        content=text,
        published_at=now - timedelta(seconds=age_sec),
    )


def _quiet_then(start: float, tail: list[float]) -> list[float]:
    return [start] * (31 - len(tail)) + tail


def _market(now: datetime, xau: list[float], xag: list[float], eurusd: list[float]):
    return snapshot(_bars(xau, now), _bars(xag, now), _bars(eurusd, now), as_of=now)


def test_buy_score():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3604.0, 3608.0, 3614.0]),
        _quiet_then(42.0, [42.00, 42.04, 42.08, 42.16]),
        _quiet_then(1.10, [1.1000, 1.1010, 1.1020, 1.1040]),
    )
    result = SignalEngine().evaluate(_news("美国8月非农就业人数低于预期", now), market, now)
    assert result.breakdown.news == 2
    assert result.breakdown.gold_1m == 2
    assert result.breakdown.gold_3m == 1
    assert result.breakdown.silver == 1
    assert result.breakdown.eurusd == 1
    assert result.score >= 5
    assert result.signal == SignalSide.BUY


def test_sell_score():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3596.0, 3592.0, 3586.0]),
        _quiet_then(42.0, [42.00, 41.96, 41.92, 41.84]),
        _quiet_then(1.10, [1.1000, 1.0990, 1.0980, 1.0960]),
    )
    result = SignalEngine().evaluate(_news("美国CPI超预期升温", now), market, now)
    assert result.breakdown.news == -2
    assert result.score <= -5
    assert result.signal == SignalSide.SELL


def test_conflict_hold():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3604.0, 3608.0, 3614.0]),
        _quiet_then(42.0, [42.00, 41.96, 41.92, 41.84]),
        _quiet_then(1.10, [1.1000, 1.0990, 1.0980, 1.0960]),
    )
    result = SignalEngine().evaluate(_news("美国8月非农就业人数低于预期", now), market, now)
    assert result.signal == SignalSide.HOLD
    assert abs(result.score) < 5 or result.breakdown.silver == -1


def test_bullish_news_price_down_cannot_buy():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3596.0, 3592.0, 3586.0]),
        _quiet_then(42.0, [42.0, 42.0, 42.0, 42.0]),
        _quiet_then(1.10, [1.10, 1.10, 1.10, 1.10]),
    )
    result = SignalEngine().evaluate(_news("美国8月非农就业人数低于预期", now), market, now)
    assert result.signal != SignalSide.BUY
    assert result.rejected_by_market == "Bullish news rejected by market"


def test_bearish_news_price_up_cannot_sell():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3604.0, 3608.0, 3614.0]),
        _quiet_then(42.0, [42.0, 42.0, 42.0, 42.0]),
        _quiet_then(1.10, [1.10, 1.10, 1.10, 1.10]),
    )
    result = SignalEngine().evaluate(_news("美国CPI超预期升温", now), market, now)
    assert result.signal != SignalSide.SELL
    assert result.rejected_by_market == "Bearish news rejected by market"


def test_expired_news_hold():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3604.0, 3608.0, 3614.0]),
        _quiet_then(42.0, [42.00, 42.04, 42.08, 42.16]),
        _quiet_then(1.10, [1.1000, 1.1010, 1.1020, 1.1040]),
    )
    result = SignalEngine().evaluate(
        _news("美国8月非农就业人数低于预期", now, age_sec=20 * 60),
        market,
        now,
    )
    assert result.signal == SignalSide.HOLD
    assert result.breakdown.news == 0


def test_dedup_same_news_only_one_primary_buy():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    market = _market(
        now,
        _quiet_then(3600.0, [3600.0, 3604.0, 3608.0, 3614.0]),
        _quiet_then(42.0, [42.00, 42.04, 42.08, 42.16]),
        _quiet_then(1.10, [1.1000, 1.1010, 1.1020, 1.1040]),
    )
    engine = SignalEngine()
    news = _news("美国8月非农就业人数低于预期", now, event_id="same-event")
    first = engine.evaluate(news, market, now)
    second = engine.evaluate(news, market, now + timedelta(seconds=10))
    assert first.signal == SignalSide.BUY
    assert first.is_primary is True
    assert second.signal == SignalSide.HOLD
    assert second.is_primary is False


def test_classify_used_by_engine():
    assert classify_news("美国8月非农就业人数低于预期").direction == 1
