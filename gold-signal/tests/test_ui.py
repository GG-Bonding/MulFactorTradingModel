from fastapi.testclient import TestClient

from gold_signal.api.app import create_app
from gold_signal.persistence.store import ProductStore

MISS = "如果非农低于预期，黄金和白银一分钟上涨，我做多黄金。"
SHORT = "如果非农高于预期，而且黄金一分钟下跌，我做空黄金。"


def test_pages_walk_compile_backtest_paper_and_keep_v1(tmp_path):
    store = ProductStore(tmp_path / "ui.sqlite")
    client = TestClient(create_app(store))
    assert client.get("/agents").status_code == 200
    preview = client.post("/agents/new", data={"idea": SHORT, "action": "preview"})
    assert preview.status_code == 200
    assert "NFP_ABOVE_EXPECTATION" in preview.text
    assert "XAUUSD.reaction_1bar" in preview.text
    refused = client.post("/agents/new", data={"idea": "今天天气不错", "action": "create"})
    assert refused.status_code == 400
    created = client.post("/agents/new", data={"idea": SHORT, "action": "create"}, follow_redirects=False)
    assert created.status_code == 303
    detail = client.get(created.headers["location"])
    assert detail.status_code == 200
    assert "DRAFT" in detail.text
    agent_url = detail.url.path
    too_soon = client.post(agent_url + "/deploy-paper")
    assert too_soon.status_code == 409
    backtest = client.post(agent_url + "/backtest", follow_redirects=True)
    assert backtest.status_code == 200
    assert "Historical backtest cannot run." not in backtest.text
    assert "Historical Evidence Available" not in backtest.text
    assert "INSUFFICIENT" in backtest.text
    assert "RECONSTRUCTED" in backtest.text
    assert "Events" in backtest.text
    assert "oos" in backtest.text
    home = client.get(agent_url)
    assert "BACKTESTED" in home.text
    paper = client.post(agent_url + "/deploy-paper", follow_redirects=True)
    assert paper.status_code == 200
    assert "PAPER" in paper.text
    arrived = client.post(
        agent_url + "/ticks",
        data={
            "stage": "arrived",
            "headline": "美国8月非农远高于预期",
            "published_at": "2026-03-24T14:30:00+00:00",
        },
        follow_redirects=True,
    )
    assert "还没有模拟成交" in arrived.text
    minute = client.post(
        agent_url + "/ticks",
        data={
            "stage": "minute",
            "headline": "美国8月非农远高于预期",
            "published_at": "2026-03-24T14:30:00+00:00",
        },
        follow_redirects=True,
    )
    assert "SHORT" in minute.text
    assert "4448" in minute.text
    settled = client.post(
        agent_url + "/ticks",
        data={
            "stage": "settle",
            "headline": "美国8月非农远高于预期",
            "published_at": "2026-03-24T14:30:00+00:00",
        },
        follow_redirects=True,
    )
    assert "还没有模拟成交" not in settled.text
    versioned = client.post(agent_url + "/versions", data={"threshold_percent": "0.12"}, follow_redirects=True)
    assert "v2" in versioned.text
    assert "4448" in versioned.text
    activity = client.get(agent_url + "/activity")
    assert activity.status_code == 200
    for kind in ("AGENT_CREATED", "BACKTEST_FINISHED", "DEPLOYED", "SIGNAL_CREATED", "OUTCOME_UPDATED"):
        assert kind in activity.text
    store.close()


def test_zero_trade_backtest_is_not_evidence_and_cannot_paper(tmp_path):
    store = ProductStore(tmp_path / "flat.sqlite")
    client = TestClient(create_app(store))
    created = client.post("/agents/new", data={"idea": MISS, "action": "create"}, follow_redirects=False)
    agent_url = created.headers["location"]
    backtest = client.post(agent_url + "/backtest", follow_redirects=True)
    assert "NO_TRADES" in backtest.text
    assert "COMPLETED" in backtest.text
    assert "Historical Evidence Available" not in backtest.text
    assert "RECONSTRUCTED" in backtest.text
    home = client.get(agent_url)
    assert "cannot go to paper" in home.text
    refused = client.post(agent_url + "/deploy-paper")
    assert refused.status_code == 409
    assert "NO_TRADES" in refused.text
    store.close()
