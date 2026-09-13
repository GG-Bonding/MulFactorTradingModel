from gold_signal.main import load_replay_cases
from gold_signal.market import snapshot
from gold_signal.models import SignalSide
from gold_signal.signal import SignalEngine


def test_replay_cases_cover_buy_sell_hold():
    engine = SignalEngine()
    signals = []
    for case in load_replay_cases():
        market = snapshot(case["xau"], case["xag"], case["eurusd"], as_of=case["now"])
        result = engine.evaluate(case["news"], market, now=case["now"])
        signals.append(result.signal)
        if case["expected"] == "HOLD_OR_SELL":
            assert result.signal in (SignalSide.HOLD, SignalSide.SELL)
            assert result.signal != SignalSide.BUY
        else:
            assert result.signal.value == case["expected"]
    assert SignalSide.BUY in signals
    assert SignalSide.SELL in signals
    assert SignalSide.HOLD in signals
