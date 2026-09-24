from gold_signal.outcomes import win_rate, win_rate_table
from gold_signal.fred import parse_fred_csv
from gold_signal.transmission import apply_transmission


def test_war_directions_come_from_the_table():
    text = "霍尔木兹海峡遭袭"
    assert apply_transmission(text, "metal").direction == 1
    assert apply_transmission(text, "oil").direction == 1
    assert apply_transmission(text, "eur").direction == -1
    assert apply_transmission(text, "nasdaq").direction == -1
    assert apply_transmission(text, "dollar").direction == 0
    assert apply_transmission(text, "crypto").direction == 0


def test_cpi_does_not_become_an_oil_signal():
    text = "美国CPI超预期升温"
    assert apply_transmission(text, "oil").direction == 0
    assert apply_transmission(text, "dollar").direction == 1
    assert apply_transmission(text, "nasdaq").direction == -1


def test_win_rate_does_not_invent_a_rate_without_samples():
    result = win_rate([
        {"signal": "BUY", "reaction_5m": None},
        {"signal": "HOLD", "reaction_5m": 0.01},
    ])
    assert result["status"] == "INSUFFICIENT"
    assert result["win_rate"] is None
    assert result["waiting"] == 1


def test_win_rate_counts_only_ready_buy_and_sell():
    result = win_rate([
        {"signal": "BUY", "reaction_5m": 0.01},
        {"signal": "SELL", "reaction_5m": -0.02},
        {"signal": "BUY", "reaction_5m": -0.01},
        {"signal": "SELL", "reaction_5m": None},
    ])
    assert result["status"] == "OK"
    assert result["samples"] == 3
    assert result["wins"] == 2
    assert abs(result["win_rate"] - 2 / 3) < 1e-12
    assert result["waiting"] == 1
    assert len(win_rate_table([])) == 5


def test_fred_skips_missing_prints_and_keeps_the_observation_date():
    point = parse_fred_csv("DATE,DFII10\n2026-09-20,1.80\n2026-09-21,.\n2026-09-22,1.87\n")
    assert point is not None
    assert point.yield_pct == 1.87
    assert point.source == "FRED DFII10"
    assert point.ts is not None
    assert point.ts.date().isoformat() == "2026-09-22"
    assert parse_fred_csv("DATE,DFII10\n2026-09-21,.\n") is None
