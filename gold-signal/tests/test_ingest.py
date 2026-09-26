import time
from datetime import datetime, timedelta, timezone

import pytest

from gold_signal.domain.models import AgentStatus, TransitionError
from gold_signal.ingest import IngestBatch, IngestLoop, ScriptedFeed
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import add_hypothesis_version, create_from_idea, deploy_paper, run_backtest
from gold_signal.persistence.store import ProductStore

IDEA = "如果非农低于预期，黄金和白银一分钟上涨，我做多黄金。"


def _utc(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 3, 24, hour, minute, second, tzinfo=timezone.utc)


def _prints(published: datetime, through: int) -> tuple[Observation, ...]:
    points = ((0, 3800.0, 40.0), (1, 3810.0, 40.1), (6, 3830.0, 40.1))
    rows = []
    for minute, gold, silver in points:
        if minute > through:
            continue
        stamp = published + timedelta(minutes=minute)
        rows.append(Observation("market.XAUUSD.close", gold, stamp, stamp, stamp, "feed", "XAUUSD"))
        rows.append(Observation("market.XAGUSD.close", silver, stamp, stamp, stamp, "feed", "XAGUSD"))
    return tuple(rows)


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("ingest loop did not finish")


def test_archive_backtest_and_feed_loop_and_new_version_needs_backtest(tmp_path):
    store = ProductStore(tmp_path / "wire.sqlite")
    agent, version = create_from_idea(store, IDEA, now=_utc(9, 0))
    report = run_backtest(store, agent.id, now=_utc(9, 1))
    assert report["status"] == "OK"
    assert report["report"]["counts"]["events"] == 1
    assert report["report"]["counts"]["flat"] == 1
    assert report["report"]["windows"]["oos"]["status"] == "OK"
    assert "news archive" not in " ".join(report["report"]["missing"])
    paper = deploy_paper(store, agent.id, now=_utc(9, 2))
    assert paper.status == AgentStatus.PAPER

    published = _utc(14, 30)
    text = "美国8月非农就业人数低于预期"
    event = EventRecord("feed-nfp", "", published, published, published, text, text, "feed")
    feed = ScriptedFeed()
    loop = IngestLoop(store, feed)
    loop.start()
    try:
        feed.push(IngestBatch(published + timedelta(seconds=30), _prints(published, 0), event))
        _wait(lambda: any(row["kind"] == "EVENT_MATCHED" for row in store.list_activities(agent.id)))
        assert store.list_paper_trades(agent.id) == []

        feed.push(IngestBatch(published + timedelta(minutes=1), _prints(published, 1), event))
        _wait(lambda: len(store.list_signals(agent.id)) == 1)
        assert store.list_signals(agent.id)[0]["side"] == "LONG"
        assert store.list_signals(agent.id)[0]["entry_price"] == 3810.0
        assert store.list_signals(agent.id)[0]["version_id"] == version.id

        feed.push(IngestBatch(published + timedelta(minutes=6), _prints(published, 6), None))
        _wait(lambda: store.list_paper_trades(agent.id)[0]["net_return"] is not None)
    finally:
        loop.stop()

    edited = version.hypothesis_yaml.replace("0.0008", "0.0012")
    second = add_hypothesis_version(store, agent.id, edited, now=_utc(15, 0))
    current = store.get_agent(agent.id)
    assert current is not None
    assert current.status == AgentStatus.DRAFT
    assert current.active_version_id == second.id
    assert store.list_backtests(agent.id)[0]["version_id"] == version.id
    assert store.list_signals(agent.id)[0]["version_id"] == version.id
    assert store.list_paper_trades(agent.id)[0]["version_id"] == version.id
    with pytest.raises(TransitionError):
        deploy_paper(store, agent.id, now=_utc(15, 1))
    store.close()
