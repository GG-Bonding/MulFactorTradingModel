from gold_signal.opportunity import build_opportunities
from gold_signal.thesis import HypothesisReading, build_thesis


def _reading(hypothesis_id: str, mechanism: str, side: str = "LONG", status: str = "PASS") -> HypothesisReading:
    return HypothesisReading("XAUUSD", hypothesis_id, hypothesis_id, mechanism, side, status)


def test_aligned_mechanisms_are_a_setup_without_a_score():
    thesis = build_thesis([
        _reading("iran", "Event"),
        _reading("yield", "Macro"),
        _reading("dollar", "FX"),
        _reading("silver", "Commodity"),
    ])
    opportunity = build_opportunities(thesis, {"iran": "INSUFFICIENT"})[0]
    assert opportunity["status"] == "LONG_SETUP"
    assert opportunity["aligned_mechanisms"] == 4
    assert opportunity["note"] == ""
    assert "score" not in opportunity
    iran = next(
        item
        for row in opportunity["mechanisms"]
        for item in row["hypotheses"]
        if item["id"] == "iran"
    )
    assert iran["evidence"] == "INSUFFICIENT"
    assert next(item["evidence"] for row in opportunity["mechanisms"] for item in row["hypotheses"] if item["id"] == "yield") is None


def test_support_without_a_price_trigger_is_a_watch():
    thesis = build_thesis([
        _reading("liquidity", "Macro"),
        _reading("breakout", "Price", "FLAT", "FAIL"),
    ])
    opportunity = build_opportunities(thesis)[0]
    assert opportunity["status"] == "WATCH"
    assert opportunity["note"] == "Price not triggered"
    price = next(row for row in opportunity["mechanisms"] if row["mechanism"] == "Price")
    assert price["hypotheses"][0]["evidence"] is None


def test_opposing_mechanisms_are_not_a_setup():
    thesis = build_thesis([
        _reading("rates", "Macro", "LONG"),
        _reading("price", "Price", "SHORT"),
    ])
    opportunity = build_opportunities(thesis)[0]
    assert opportunity["status"] == "NO_SETUP"
    assert opportunity["note"] == "Conflict"
    assert opportunity["aligned_mechanisms"] == 0


def test_home_shows_a_setup_and_leaves_missing_evidence_blank(tmp_path):
    from fastapi.testclient import TestClient

    from gold_signal.api.app import create_app
    from gold_signal.persistence.store import ProductStore

    store = ProductStore(tmp_path / "opp.sqlite")
    client = TestClient(create_app(store))
    created = client.post("/api/agents", json={"idea": "如果非农高于预期，而且黄金一分钟下跌，我做空黄金。"})
    agent_id = created.json()["agent"]["id"]
    version_id = created.json()["version"]["id"]
    assert client.get("/api/opportunities").json()[0]["status"] == "NO_SETUP"
    assert client.post(f"/api/agents/{agent_id}/backtests").status_code == 201
    store.insert_signal(
        {
            "id": "sig-home",
            "agent_id": agent_id,
            "version_id": version_id,
            "event_id": "nfp",
            "side": "SHORT",
            "entry_price": 4448.0,
            "reason": "passed",
            "evaluated_at": "2026-09-04T12:31:00+00:00",
            "created_at": "2026-09-04T12:31:00+00:00",
        }
    )
    opportunity = client.get("/api/opportunities").json()[0]
    assert opportunity["status"] == "SHORT_SETUP"
    assert opportunity["aligned_mechanisms"] == 1
    hypothesis = opportunity["mechanisms"][0]["hypotheses"][0]
    assert hypothesis["evidence"] == "INSUFFICIENT"
    page = client.get("/")
    assert page.status_code == 200
    assert "Trading Agents" in page.text
    assert "SHORT" in page.text
    assert "2026-09-04 12:31" in page.text


def test_setups_sort_ahead_of_conflicts():
    long = build_thesis([_reading("gold", "Event")])[0]
    clash = build_thesis([
        HypothesisReading("EURGBP", "rates", "rates", "Macro", "LONG", "PASS"),
        HypothesisReading("EURGBP", "price", "price", "Price", "SHORT", "PASS"),
    ])[0]
    rows = build_opportunities([clash, long])
    assert [row["asset"] for row in rows] == ["XAUUSD", "EURGBP"]
    assert rows[0]["status"] == "LONG_SETUP"
    assert rows[1]["status"] == "NO_SETUP"
