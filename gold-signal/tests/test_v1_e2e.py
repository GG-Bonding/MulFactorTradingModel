"""Freeze the V1 path: compile, backtest, paper, feed, settle, then a new draft."""

import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from fastapi.testclient import TestClient

from gold_signal.api.app import create_app
from gold_signal.persistence.store import ProductStore

SHORT = "如果非农高于预期，而且黄金一分钟下跌，我做空黄金。"
MISS = "如果非农低于预期，黄金和白银一分钟上涨，我做多黄金。"


def test_v1_path_freezes_evidence_paper_and_the_next_draft(tmp_path):
    client = _client(tmp_path)
    try:
        assert _ok(client, "/healthz").status_code == 200
        ready = _ok(client, "/readyz")
        assert ready.status_code == 200
        assert ready.json()["database"] == "READY"
        assert "ingest" in ready.json()

        missed = client.post("/api/agents", json={"idea": MISS})
        assert missed.status_code == 201
        missed_id = missed.json()["agent"]["id"]
        flat = client.post(f"/api/agents/{missed_id}/backtests")
        assert flat.status_code == 201
        assert flat.json()["report"]["run_status"] == "COMPLETED"
        assert flat.json()["report"]["evidence_status"] == "NO_TRADES"
        assert flat.json()["report"]["data_quality"]["XAUUSD"] == "RECONSTRUCTED"
        blocked = client.post(f"/api/agents/{missed_id}/deploy-paper")
        assert blocked.status_code == 409

        created = client.post("/api/agents", json={"idea": SHORT})
        assert created.status_code == 201
        agent_id = created.json()["agent"]["id"]
        version_id = created.json()["version"]["id"]
        backtest = client.post(f"/api/agents/{agent_id}/backtests")
        assert backtest.status_code == 201
        body = backtest.json()["report"]
        assert body["run_status"] == "COMPLETED"
        assert body["evidence_status"] == "INSUFFICIENT"
        assert body["events"] == 3
        assert body["trades"] == 1
        assert "Historical Evidence Available" not in str(body["evidence_status"])

        deployed = client.post(f"/api/agents/{agent_id}/deploy-paper")
        assert deployed.status_code == 200
        assert deployed.json()["status"] == "PAPER"

        published = datetime(2026, 3, 24, 14, 30, tzinfo=timezone.utc)
        early = client.post("/api/runtime/ticks", json=_tick(published, 30, include_event=True))
        assert early.status_code == 200
        assert early.json()["signals"] == 0

        filled = client.post("/api/runtime/ticks", json=_tick(published, 60, include_event=True))
        assert filled.status_code == 200
        assert filled.json()["signals"] == 1
        trade = client.get(f"/api/agents/{agent_id}/trades").json()["trades"][0]
        assert trade["side"] == "SHORT"
        assert trade["entry_price"] == 4448.0
        assert trade["version_id"] == version_id
        assert trade["net_return"] is None

        settled = client.post("/api/runtime/ticks", json=_tick(published, 360, include_event=False))
        assert settled.status_code == 200
        assert settled.json()["settled"] == [trade["id"]]
        closed = client.get(f"/api/agents/{agent_id}/trades").json()["trades"][0]
        assert closed["net_return"] is not None

        yaml = created.json()["version"]["hypothesis_yaml"].replace("0.0008", "0.0012")
        newer = client.post(f"/api/agents/{agent_id}/versions", json={"yaml": yaml})
        assert newer.status_code == 201
        assert newer.json()["version"] == 2
        assert client.get(f"/api/agents/{agent_id}").json()["agent"]["status"] == "DRAFT"
        again = client.post(f"/api/agents/{agent_id}/backtests")
        assert again.status_code == 201
        assert again.json()["report"]["run_status"] == "COMPLETED"
        assert again.json()["version_id"] != version_id
        runs = client.get(f"/api/agents/{agent_id}/backtests").json()["backtests"]
        assert runs[0]["version_id"] == version_id
        assert client.get(f"/api/agents/{agent_id}/trades").json()["trades"][0]["version_id"] == version_id
    finally:
        client.close()


def _ok(client, path: str):
    """Docker's port proxy can reset the first request while the process is binding."""
    last = None
    for _ in range(8):
        try:
            last = client.get(path)
        except httpx.HTTPError:
            time.sleep(0.4)
            continue
        if last.status_code < 500:
            return last
        time.sleep(0.4)
    assert last is not None
    return last


def _client(tmp_path):
    url = os.environ.get("GOLD_SIGNAL_URL")
    if url:
        return httpx.Client(base_url=url, timeout=30.0, trust_env=False)
    return TestClient(create_app(ProductStore(tmp_path / "e2e.sqlite")))


def _tick(published: datetime, seconds: int, *, include_event: bool) -> dict:
    points = ((0, 4517.0), (60, 4448.0), (360, 4430.0))
    payload = {
        "now": (published + timedelta(seconds=seconds)).isoformat(),
        "observations": [
            {
                "factor": "market.XAUUSD.close",
                "value": price,
                "observed_at": (published + timedelta(seconds=offset)).isoformat(),
            }
            for offset, price in points
            if offset <= seconds
        ],
    }
    if include_event:
        payload["event"] = {
            "event_id": "e2e-nfp",
            "title": "美国8月非农远高于预期",
            "content": "美国8月非农远高于预期",
            "published_at": published.isoformat(),
        }
    return payload
