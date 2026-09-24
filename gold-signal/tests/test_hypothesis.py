from gold_signal.hypothesis import evaluate_hypothesis, historical_validation, parse_hypothesis
from gold_signal.main import load_replay_cases
from gold_signal.market import snapshot


def _decision(text: str, case_name: str):
    spec = parse_hypothesis(text)
    case = next(item for item in load_replay_cases() if item["name"] == case_name)
    market = snapshot(case["xau"], case["xag"], case["eurusd"], as_of=case["now"])
    return evaluate_hypothesis(spec, case["news"], market, case["now"])


GOLD = """
id: gold_v0
name: gold confirm
asset: XAUUSD
family: metal
entry: FOLLOW_NEWS
horizons:
  - 1m
confirmations:
  - factor: XAUUSD.reaction_1m
    operator: same_sign_abs_gte
    value: 0.0008
"""

STRICT = GOLD.replace("value: 0.0008", "value: 0.05")


def test_yaml_longs_when_post_news_price_confirms():
    assert _decision(GOLD, "case1_bullish_confirmed").side == "LONG"


def test_yaml_does_not_long_when_price_rejects_the_news():
    assert _decision(GOLD, "case3_bullish_rejected").side == "FLAT"


def test_changing_only_the_yaml_threshold_blocks_the_same_case():
    assert _decision(STRICT, "case1_bullish_confirmed").side == "FLAT"


def test_backtest_reports_missing_history_instead_of_a_win_rate():
    spec = parse_hypothesis(
        """
id: iran_gold_001
name: iran
asset: XAUUSD
family: metal
entry: FOLLOW_NEWS
confirmations:
  - factor: USOIL.reaction_1m
    operator: ">"
    value: 0.0015
  - factor: DFII10.change
    operator: "<="
    value: 0
"""
    )
    report = historical_validation(spec, "2025-01-01", "2026-09-01")
    assert report["status"] == "INSUFFICIENT"
    assert report["samples"] == 0
    assert report["win_rate"] is None
    assert report["avg_return"] is None
    joined = " ".join(report["missing"])
    assert "news archive" in joined
    assert "DFII10" in joined
    assert "USOIL.reaction_1m" in joined
