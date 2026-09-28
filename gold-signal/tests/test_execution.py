from datetime import datetime, timedelta, timezone

from gold_signal.execution import ExitPolicy, portfolio_result, simulate_trade
from gold_signal.observation import Observation
from gold_signal.replay_clock import DecisionRecord


def _utc(minute: int) -> datetime:
    return datetime(2026, 9, 4, 14, minute, tzinfo=timezone.utc)


def _bars(*prices: float) -> list[Observation]:
    return [
        Observation("market.XAUUSD.close", price, _utc(30 + index), _utc(30 + index), None, "fixture", "XAUUSD")
        for index, price in enumerate(prices)
    ]


def _decision(price: float, minute: int, event_id: str = "e1") -> DecisionRecord:
    return DecisionRecord(
        event_id=event_id,
        hypothesis_id="spec",
        evaluated_at=_utc(minute),
        side="LONG",
        entry_at=_utc(minute),
        entry_price=price,
        reasons=("filled",),
    )


def test_take_profit_exits_before_the_hold_limit():
    trade = simulate_trade(
        _decision(100.0, 31),
        "XAUUSD",
        _bars(100.0, 100.0, 100.8, 101.0, 102.0, 103.0, 104.0),
        ExitPolicy(max_hold=timedelta(minutes=5), take_profit=0.005),
    )
    assert trade is not None
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.exit_at == _utc(32)
    assert trade.exit_price == 100.8


def test_stop_exits_when_the_close_goes_against_the_position():
    trade = simulate_trade(
        _decision(100.0, 31),
        "XAUUSD",
        _bars(100.0, 100.0, 99.4),
        ExitPolicy(max_hold=timedelta(minutes=5), stop_loss=0.005),
    )
    assert trade is not None
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == 99.4


def test_max_hold_is_the_exit_when_neither_barrier_hits():
    trade = simulate_trade(
        _decision(100.0, 31),
        "XAUUSD",
        _bars(100.0, 100.0, 100.1, 100.1, 100.1, 100.1, 100.2),
        ExitPolicy(max_hold=timedelta(minutes=5)),
    )
    assert trade is not None
    assert trade.exit_reason == "MAX_HOLD"
    assert trade.exit_at == _utc(36)


def test_a_second_signal_does_not_stack_while_the_position_is_open():
    first = simulate_trade(
        _decision(100.0, 31, "e1"),
        "XAUUSD",
        _bars(100.0, 100.0, 100.1, 100.1, 100.1, 100.1, 100.2),
        ExitPolicy(),
    )
    second = simulate_trade(
        _decision(100.0, 33, "e2"),
        "XAUUSD",
        _bars(100.0, 100.0, 100.1, 100.1, 100.1, 100.1, 100.2, 100.3, 100.4),
        ExitPolicy(),
    )
    assert first is not None and second is not None
    result = portfolio_result([first, second])
    assert result["executed"] == 1
    assert result["skipped"] == 1
    assert result["end_equity"] == 1.0 + first.net_return
