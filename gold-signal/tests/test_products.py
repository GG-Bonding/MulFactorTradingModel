from datetime import datetime, timedelta, timezone

from gold_signal.market import snapshot
from gold_signal.models import Bar, FlashNews, SignalSide
from gold_signal.products import impact_for_product, parse_yahoo_minutes
from gold_signal.signal import SignalEngine


def _bars(closes: list[float], end: datetime) -> list[Bar]:
    start = end - timedelta(minutes=len(closes) - 1)
    return [Bar(ts=start + timedelta(minutes=i), close=c) for i, c in enumerate(closes)]


def _quiet_then(start: float, tail: list[float]) -> list[float]:
    return [start] * (31 - len(tail)) + tail


def _rising(now: datetime, code: str, confirm: str, third: str):
    return snapshot(
        _bars(_quiet_then(100.0, [100.0, 100.2, 100.4, 100.8]), now),
        _bars(_quiet_then(100.0, [100.0, 100.2, 100.4, 100.8]), now),
        _bars(_quiet_then(100.0, [100.0, 100.2, 100.4, 100.8]), now),
        as_of=now,
        primary_code=code,
        confirm_code=confirm,
        dollar_code=third,
    )


def test_cpi_does_not_copy_onto_oil_or_bitcoin():
    text = "美国CPI超预期升温"
    assert impact_for_product(text, "metal").direction == -1
    assert impact_for_product(text, "eur").direction == -1
    assert impact_for_product(text, "nasdaq").direction == -1
    assert impact_for_product(text, "dollar").direction == 1
    assert impact_for_product(text, "oil").direction == 0
    assert impact_for_product(text, "crypto").direction == 0


def test_hormuz_splits_by_product():
    text = "霍尔木兹海峡遭袭"
    assert impact_for_product(text, "metal").direction == 1
    assert impact_for_product(text, "oil").direction == 1
    assert impact_for_product(text, "eur").direction == -1
    assert impact_for_product(text, "nasdaq").direction == -1
    assert impact_for_product(text, "dollar").direction == 0
    assert impact_for_product(text, "crypto").direction == 0


def test_french_yield_up_hits_gold_and_eur_only():
    text = "法国国债收益率上涨"
    assert impact_for_product(text, "metal").direction == -1
    assert impact_for_product(text, "eur").direction == -1
    assert impact_for_product(text, "oil").direction == 0
    assert impact_for_product(text, "nasdaq").direction == 0
    assert impact_for_product(text, "dollar").direction == 0


def test_opec_cut_is_oil_not_gold():
    text = "OPEC宣布减产"
    assert impact_for_product(text, "oil").direction == 1
    assert impact_for_product(text, "metal").direction == 0


def test_bitcoin_headline_stays_on_bitcoin():
    text = "比特币突破前高并持续上涨"
    assert impact_for_product(text, "crypto").direction == 1
    assert impact_for_product(text, "metal").direction == 0
    assert impact_for_product(text, "oil").direction == 0
    assert impact_for_product(text, "eur").direction == 0
    assert impact_for_product(text, "nasdaq").direction == 0


def test_same_headline_can_signal_two_products():
    now = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)
    news = FlashNews(
        event_id="war-1",
        title="霍尔木兹海峡遭袭",
        content="霍尔木兹海峡遭袭",
        published_at=now - timedelta(seconds=70),
    )
    engine = SignalEngine()
    gold = engine.evaluate(
        news,
        _rising(now, "XAUUSD", "XAGUSD", "EURUSD"),
        now,
        impact=impact_for_product(news.text, "metal"),
    )
    oil = engine.evaluate(
        news,
        _rising(now, "USOIL", "UKOIL", "EURUSD"),
        now,
        impact=impact_for_product(news.text, "oil"),
    )
    assert gold.signal == SignalSide.BUY
    assert gold.product == "XAUUSD"
    assert oil.signal == SignalSide.BUY
    assert oil.product == "USOIL"
    assert oil.is_primary is True


def test_yahoo_minute_parser_skips_null_closes():
    from datetime import datetime, timedelta, timezone

    bars = parse_yahoo_minutes(
        {
            "chart": {
                "result": [
                    {
                        "timestamp": [1_700_000_000, 1_700_000_060, 1_700_000_120],
                        "indicators": {"quote": [{"close": [100.0, None, 101.0]}]},
                    }
                ]
            }
        },
        "NQ=F",
    )
    assert [bar.close for bar in bars] == [100.0, 101.0]
    assert bars[0].ts == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc) + timedelta(minutes=1)
    assert bars[1].ts == datetime.fromtimestamp(1_700_000_120, tz=timezone.utc) + timedelta(minutes=1)
