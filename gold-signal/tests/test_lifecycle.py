from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from gold_signal.api.app import create_app
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import run_backtest
from gold_signal.persistence.store import ProductStore

IDEA = "如果非农低于预期，黄金和白银一分钟上涨，我做多黄金。"


def _utc(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 24, hour, minute, tzinfo=timezone.utc)


def _tape(published: datetime) -> list[Observation]:
    rows = []
    for minute, gold in ((0, 3800.0), (1, 3810.0), (6, 3830.0)):
        stamp = published + timedelta(minutes=minute)
        silver = 40.0 if minute == 0 else 40.1
        rows.append(Observation("market.XAUUSD.close", gold, stamp, stamp, stamp, "fixture", "XAUUSD"))
        rows.append(Observation("market.XAGUSD.close", silver, stamp, stamp, stamp, "fixture", "XAGUSD"))
    return rows


def _event(published: datetime) -> EventRecord:
    text = "美国8月非农就业人数低于预期"
    return EventRecord("nfp-live", "", published, published, published, text, text, "fixture")


def _obs_payload(rows: list[Observation]) -> list[dict]:
    return [
        {"factor": row.factor, "value": row.value, "observed_at": row.observed_at.isoformat()}
        for row in rows
    ]


def test_paper_agent_waits_one_minute_then_fills_and_settles_five_minutes(tmp_path):
    store = ProductStore(tmp_path / "life.sqlite")
    client = TestClient(create_app(store))
    created = client.post("/api/agents", json={"idea": IDEA})
    assert created.status_code == 201
    agent_id = created.json()["agent"]["id"]
    version_id = created.json()["version"]["id"]
    published = _utc(14, 30)
    news = _event(published)
    tape = _tape(published)
    report = run_backtest(store, agent_id, now=published, events=[news], observations=tape)
    assert report["status"] == "OK"
    assert report["report"]["windows"]["oos"]["samples"] == 1
    assert report["version_id"] == version_id
    assert client.get(f"/api/agents/{agent_id}").json()["agent"]["status"] == "BACKTESTED"

    deployed = client.post(f"/api/agents/{agent_id}/deploy-paper")
    assert deployed.status_code == 200
    assert deployed.json()["status"] == "PAPER"

    early = client.post(
        "/api/runtime/ticks",
        json={
            "now": (published + timedelta(seconds=30)).isoformat(),
            "observations": _obs_payload(tape),
            "event": {
                "event_id": news.event_id,
                "title": news.title,
                "content": news.content,
                "published_at": published.isoformat(),
            },
        },
    )
    assert early.status_code == 200
    assert early.json()["signals"] == 0
    assert client.get(f"/api/agents/{agent_id}/trades").json()["trades"] == []

    filled = client.post(
        "/api/runtime/ticks",
        json={
            "now": (published + timedelta(minutes=1)).isoformat(),
            "observations": _obs_payload(tape),
            "event": {
                "event_id": news.event_id,
                "title": news.title,
                "content": news.content,
                "published_at": published.isoformat(),
            },
        },
    )
    assert filled.status_code == 200
    assert filled.json()["signals"] == 1
    trades = client.get(f"/api/agents/{agent_id}/trades").json()["trades"]
    assert len(trades) == 1
    assert trades[0]["side"] == "LONG"
    assert trades[0]["entry_price"] == 3810.0
    assert trades[0]["version_id"] == version_id
    assert trades[0]["net_return"] is None

    settled = client.post(
        "/api/runtime/ticks",
        json={
            "now": (published + timedelta(minutes=6)).isoformat(),
            "observations": _obs_payload(tape),
        },
    )
    assert settled.status_code == 200
    assert settled.json()["settled"] == [trades[0]["id"]]
    closed = client.get(f"/api/agents/{agent_id}/trades").json()["trades"][0]
    assert closed["net_return"] is not None
    assert closed["net_return"] < (3830.0 / 3810.0 - 1)
    stats = client.get(f"/api/agents/{agent_id}/stats").json()
    assert stats["paper_samples"] == 1
    assert stats["paper_avg_net"] == closed["net_return"]
    assert stats["oos_samples"] == 1

    yaml = created.json()["version"]["hypothesis_yaml"].replace("0.0008", "0.0012")
    newer = client.post(f"/api/agents/{agent_id}/versions", json={"yaml": yaml})
    assert newer.status_code == 201
    assert newer.json()["version"] == 2
    assert client.get(f"/api/agents/{agent_id}/backtests").json()["backtests"][0]["version_id"] == version_id
    assert client.get(f"/api/agents/{agent_id}/signals").json()["signals"][0]["version_id"] == version_id
    assert client.get(f"/api/agents/{agent_id}/trades").json()["trades"][0]["version_id"] == version_id
    kinds = [item["kind"] for item in client.get(f"/api/agents/{agent_id}/activities").json()["activities"]]
    for kind in ("BACKTEST_FINISHED", "DEPLOYED", "EVENT_MATCHED", "SIGNAL_CREATED", "PAPER_ORDER_FILLED", "OUTCOME_UPDATED"):
        assert kind in kinds
    store.close()
