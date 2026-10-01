from gold_signal.hypothesis import parse_hypothesis
from gold_signal.thesis import HypothesisReading, build_thesis, mechanism_of

IRAN = """
id: iran_gold
name: Oil and silver confirm a geopolitical gold long
asset: XAUUSD
family: metal
entry: LONG
confirmations:
  - factor: XAUUSD.reaction_1bar
    operator: ">"
    value: 0.0008
  - factor: USOIL.reaction_1bar
    operator: ">"
    value: 0.0015
  - factor: DFII10.change
    operator: "<="
    value: 0
"""

NFP = """
id: nfp_gold
name: 非农低于预期后做多XAUUSD
asset: XAUUSD
family: metal
entry: LONG
trigger: NFP_BELOW_EXPECTATION
confirmations:
  - factor: XAUUSD.reaction_1bar
    operator: ">"
    value: 0.0008
  - factor: XAGUSD.reaction_1bar
    operator: ">"
    value: 0.0008
"""

DOLLAR = """
id: dollar_gold
name: Dollar weak gold long
asset: XAUUSD
family: metal
entry: LONG
confirmations:
  - factor: DXY.change
    operator: "<"
    value: 0
"""

SILVER = """
id: silver_gold
name: Silver confirms gold
asset: XAUUSD
family: metal
entry: LONG
confirmations:
  - factor: XAGUSD.reaction_1bar
    operator: ">"
    value: 0.0008
"""

BREAKOUT = """
id: gold_break
name: Gold breakout
asset: XAUUSD
family: metal
entry: LONG
confirmations:
  - factor: XAUUSD.reaction_1bar
    operator: ">"
    value: 0.0008
"""


def _pass(spec_text: str, side: str = "LONG", status: str = "PASS") -> HypothesisReading:
    spec = parse_hypothesis(spec_text)
    return HypothesisReading(spec.asset, spec.id, spec.name, mechanism_of(spec), side, status)


def test_each_hypothesis_votes_in_one_mechanism():
    assert mechanism_of(parse_hypothesis(IRAN)) == "Event"
    assert mechanism_of(parse_hypothesis(NFP)) == "Macro"
    assert mechanism_of(parse_hypothesis(DOLLAR)) == "FX"
    assert mechanism_of(parse_hypothesis(SILVER)) == "Commodity"
    assert mechanism_of(parse_hypothesis(BREAKOUT)) == "Price"


def test_four_passing_channels_count_as_four_not_an_average():
    thesis = build_thesis([
        _pass(IRAN),
        _pass(NFP),
        _pass(DOLLAR),
        _pass(SILVER),
    ])[0]
    assert thesis["asset"] == "XAUUSD"
    assert thesis["bias"] == "LONG"
    assert thesis["aligned_mechanisms"] == 4
    assert "score" not in thesis


def test_two_event_hypotheses_are_one_mechanism():
    second = """
id: iran_gold_follow
name: Another geopolitical gold long
asset: XAUUSD
family: metal
entry: LONG
trigger: GEOPOLITICAL_ESCALATION
confirmations:
  - factor: XAUUSD.reaction_1bar
    operator: ">"
    value: 0.0008
"""
    thesis = build_thesis([_pass(IRAN), _pass(second)])[0]
    assert thesis["aligned_mechanisms"] == 1
    event = thesis["mechanisms"][0]
    assert event["mechanism"] == "Event"
    assert [row["status"] for row in event["hypotheses"]] == ["PASS", "PASS"]


def test_opposite_mechanisms_conflict_instead_of_averaging():
    thesis = build_thesis([
        _pass(IRAN, "LONG"),
        _pass(DOLLAR, "SHORT"),
    ])[0]
    assert thesis["bias"] == "CONFLICT"
    assert thesis["aligned_mechanisms"] == 0


def test_latest_signal_is_the_only_vote(tmp_path):
    from gold_signal.persistence.service import create_from_idea
    from gold_signal.persistence.store import ProductStore
    from gold_signal.thesis import thesis_for_store

    store = ProductStore(tmp_path / "thesis.sqlite")
    agent, version = create_from_idea(store, "如果非农高于预期，而且黄金一分钟下跌，我做空黄金。")
    assert thesis_for_store(store)[0]["bias"] == "NO_SETUP"
    store.insert_signal(
        {
            "id": "sig-1",
            "agent_id": agent.id,
            "version_id": version.id,
            "event_id": "nfp",
            "side": "SHORT",
            "entry_price": 1.0,
            "reason": "passed",
            "evaluated_at": "2026-09-04T12:31:00+00:00",
            "created_at": "2026-09-04T12:31:00+00:00",
        }
    )
    thesis = thesis_for_store(store)[0]
    assert thesis["bias"] == "SHORT"
    assert thesis["aligned_mechanisms"] == 1
    assert thesis["as_of"] == "2026-09-04T12:31:00+00:00"
    assert thesis["mechanisms"][0]["mechanism"] == "Macro"


def test_a_failed_hypothesis_does_not_vote():
    thesis = build_thesis([
        _pass(IRAN, "FLAT", "FAIL"),
        _pass(BREAKOUT, "LONG", "PASS"),
    ])[0]
    assert thesis["bias"] == "LONG"
    assert thesis["aligned_mechanisms"] == 1
    assert thesis["mechanisms"][0]["side"] == "NONE"
