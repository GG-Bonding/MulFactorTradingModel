from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from gold_signal.api.app import create_app
from gold_signal.domain.models import AgentVersion
from gold_signal.persistence.service import (
    add_hypothesis_version,
    create_from_idea,
    record_paper_trade,
    record_signal,
    run_backtest,
)
from gold_signal.persistence.store import ProductStore, VersionImmutable

IDEA = "如果非农低于预期，黄金和白银一分钟上涨，我做多黄金。"


def _now() -> datetime:
    return datetime(2026, 3, 24, 9, 0, tzinfo=timezone.utc)


def test_restart_keeps_v1_backtest_signal_and_trade(tmp_path):
    path = tmp_path / "product.sqlite"
    store = ProductStore(path)
    agent, version = create_from_idea(store, IDEA, now=_now())
    backtest = run_backtest(store, agent.id, now=_now())
    assert backtest["status"] == "OK"
    assert backtest["version_id"] == version.id
    assert backtest["report"]["counts"]["events"] == 1
    assert backtest["report"]["windows"]["oos"]["status"] == "OK"
    signal = record_signal(
        store,
        agent.id,
        version.id,
        {"event_id": "nfp-1", "side": "LONG", "entry_price": 3810.0, "reason": "2/2 confirmations passed"},
        now=_now(),
    )
    record_paper_trade(
        store,
        {
            "agent_id": agent.id,
            "version_id": version.id,
            "signal_id": signal["id"],
            "side": "LONG",
            "entry_price": 3810.0,
            "spec_sha256": version.hypothesis_sha256,
            "net_return": 0.0007,
        },
        now=_now(),
    )
    edited = version.hypothesis_yaml.replace("0.0008", "0.0012")
    second = add_hypothesis_version(store, agent.id, edited, now=_now())
    assert second.version == 2
    assert second.hypothesis_sha256 != version.hypothesis_sha256
    with pytest.raises(VersionImmutable):
        store.save_version(
            AgentVersion(
                version.id,
                version.agent_id,
                version.version,
                "changed: true\n",
                "deadbeef",
                version.created_at,
            )
        )
    store.close()

    reopened = ProductStore(path)
    kept = reopened.get_version(version.id)
    assert kept is not None
    assert kept.hypothesis_yaml == version.hypothesis_yaml
    assert kept.hypothesis_sha256 == version.hypothesis_sha256
    runs = reopened.list_backtests(agent.id)
    assert len(runs) == 1
    assert runs[0]["version_id"] == version.id
    assert reopened.list_signals(agent.id)[0]["version_id"] == version.id
    assert reopened.list_paper_trades(agent.id)[0]["version_id"] == version.id
    assert reopened.list_versions(agent.id)[1].id == second.id
    saved = reopened.get_agent(agent.id)
    assert saved is not None
    assert saved.status.value == "DRAFT"
    assert saved.active_version_id == second.id
    reopened.close()


def test_api_compile_create_backtest_and_keep_v1(tmp_path):
    store = ProductStore(tmp_path / "api.sqlite")
    client = TestClient(create_app(store))
    compiled = client.post("/api/agents/compile", json={"idea": IDEA})
    assert compiled.status_code == 200
    body = compiled.json()
    assert body["asset"] == "XAUUSD"
    assert body["entry"] == "LONG"
    assert body["trigger"] == "NFP_BELOW_EXPECTATION"
    refused = client.post("/api/agents/compile", json={"idea": "今天天气不错"})
    assert refused.status_code == 400

    created = client.post("/api/agents", json={"idea": IDEA})
    assert created.status_code == 201
    agent_id = created.json()["agent"]["id"]
    version_id = created.json()["version"]["id"]
    assert created.json()["agent"]["status"] == "DRAFT"

    blocked = client.post(f"/api/agents/{agent_id}/deploy-paper")
    assert blocked.status_code == 409

    backtest = client.post(f"/api/agents/{agent_id}/backtests")
    assert backtest.status_code == 201
    assert backtest.json()["status"] == "OK"
    assert backtest.json()["report"]["windows"]["oos"]["status"] == "OK"
    assert client.get(f"/api/agents/{agent_id}").json()["agent"]["status"] == "BACKTESTED"

    paper = client.post(f"/api/agents/{agent_id}/deploy-paper")
    assert paper.status_code == 200
    assert paper.json()["status"] == "PAPER"

    yaml = created.json()["version"]["hypothesis_yaml"].replace("0.0008", "0.0012")
    newer = client.post(f"/api/agents/{agent_id}/versions", json={"yaml": yaml})
    assert newer.status_code == 201
    assert newer.json()["version"] == 2
    listed = client.get(f"/api/agents/{agent_id}/backtests").json()["backtests"]
    assert listed[0]["version_id"] == version_id
    kinds = [item["kind"] for item in client.get(f"/api/agents/{agent_id}/activities").json()["activities"]]
    assert "AGENT_CREATED" in kinds
    assert "BACKTEST_FINISHED" in kinds
    assert "DEPLOYED" in kinds
    stats = client.get(f"/api/agents/{agent_id}/stats").json()
    assert stats["backtest_status"] is None
    assert client.get(f"/api/agents/{agent_id}").json()["agent"]["status"] == "DRAFT"
    store.close()
