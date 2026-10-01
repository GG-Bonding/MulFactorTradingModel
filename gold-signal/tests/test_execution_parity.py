"""The same bars must close the same way in research and in paper."""

from datetime import datetime, timedelta, timezone

from gold_signal.execution import ExitPolicy, paper_execution, simulate_trade
from gold_signal.observation import Observation
from gold_signal.replay_clock import DecisionRecord


def _utc(minute: int) -> datetime:
    return datetime(2026, 9, 4, 14, minute, tzinfo=timezone.utc)


def _bars() -> list[Observation]:
    prices = (100.0, 100.0, 100.4, 99.2, 99.0, 98.5, 98.0)
    return [
        Observation(
            "market.XAUUSD.close",
            price,
            _utc(30 + index),
            _utc(30 + index),
            None,
            "fixture",
            "XAUUSD",
        )
        for index, price in enumerate(prices)
    ]


def _decision() -> DecisionRecord:
    return DecisionRecord(
        event_id="parity",
        hypothesis_id="spec",
        evaluated_at=_utc(31),
        side="LONG",
        entry_at=_utc(31),
        entry_price=100.0,
        reasons=("filled",),
    )


def _same(policy: ExitPolicy) -> None:
    bars = _bars()
    decision = _decision()
    historical = simulate_trade(decision, "XAUUSD", bars, policy)
    paper = paper_execution(decision, "XAUUSD", bars, policy)
    assert historical is not None and paper is not None
    assert paper.entry_price == historical.entry_price
    assert paper.entry_at == historical.entry_at
    assert paper.exit_price == historical.exit_price
    assert paper.exit_at == historical.exit_at
    assert paper.exit_reason == historical.exit_reason
    assert paper.outcome.gross_return == historical.outcome.gross_return
    assert paper.outcome.spread_cost == historical.outcome.spread_cost
    assert paper.outcome.slippage_cost == historical.outcome.slippage_cost
    assert paper.outcome.net_return == historical.outcome.net_return
    assert paper.outcome.mfe == historical.outcome.mfe
    assert paper.outcome.mae == historical.outcome.mae


def test_take_profit_matches():
    closed = simulate_trade(_decision(), "XAUUSD", _bars(), ExitPolicy(take_profit=0.003, max_hold=timedelta(minutes=15)))
    assert closed is not None
    assert closed.exit_reason == "TAKE_PROFIT"
    assert closed.exit_at == _utc(32)
    _same(ExitPolicy(take_profit=0.003, max_hold=timedelta(minutes=15)))


def test_stop_matches():
    _same(ExitPolicy(stop_loss=0.005, max_hold=timedelta(minutes=15)))


def test_max_hold_matches():
    _same(ExitPolicy(max_hold=timedelta(minutes=5)))
