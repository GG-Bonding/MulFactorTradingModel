import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gold_signal.market import snapshot
from gold_signal.models import Bar, FlashNews, SignalSide
from gold_signal.signal import SignalEngine, SignalStore


def test_store_writes_hold_and_buy_and_updates_forward_returns(tmp_path: Path):
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)

    def bars(closes):
        t0 = now - timedelta(minutes=len(closes) - 1)
        return [Bar(ts=t0 + timedelta(minutes=i), close=c) for i, c in enumerate(closes)]

    quiet = [3600.0] * 27 + [3600.0, 3604.0, 3608.0, 3614.0]
    xag = [42.0] * 27 + [42.00, 42.04, 42.08, 42.16]
    eurusd = [1.10] * 27 + [1.1000, 1.1010, 1.1020, 1.1040]
    market = snapshot(bars(quiet), bars(xag), bars(eurusd), as_of=now)
    news = FlashNews(
        event_id="persist-1",
        title="美国8月非农就业人数低于预期",
        content="美国8月非农就业人数低于预期",
        published_at=now,
    )
    engine = SignalEngine()
    buy = engine.evaluate(news, market, now)
    assert buy.signal == SignalSide.BUY
    path = tmp_path / "signals.jsonl"
    store = SignalStore(path)
    store.append(buy)
    hold_news = FlashNews(
        event_id="persist-hold",
        title="某公司发布新产品",
        content="某公司发布新产品",
        published_at=now,
    )
    hold = engine.evaluate(hold_news, market, now)
    assert hold.signal == SignalSide.HOLD
    store.append(hold)
    store.update_returns(
        buy.event_id,
        buy.timestamp.isoformat(),
        return_1m=0.0005,
        return_5m=0.0014,
        return_15m=0.0022,
        return_30m=0.0016,
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    signals = {row["event_id"]: row for row in rows}
    assert signals["persist-1"]["signal"] == "BUY"
    assert signals["persist-1"]["return_1m"] == 0.0005
    assert signals["persist-1"]["return_30m"] == 0.0016
    assert signals["persist-hold"]["signal"] == "HOLD"
    assert signals["persist-hold"]["return_1m"] is None
