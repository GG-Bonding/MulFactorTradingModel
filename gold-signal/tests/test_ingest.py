import time
from datetime import datetime, timedelta, timezone

import pytest

from gold_signal.domain.models import AgentStatus, TransitionError
from gold_signal.ingest import IngestBatch, IngestLoop, ScriptedFeed
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import add_hypothesis_version, create_from_idea, deploy_paper, run_backtest
from gold_signal.persistence.store import ProductStore

IDEA = "如果非农高于预期，而且黄金一分钟下跌，我做空黄金。"


def _utc(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 3, 24, hour, minute, second, tzinfo=timezone.utc)


def _prints(published: datetime, through: int) -> tuple[Observation, ...]:
    points = ((0, 4517.0, 40.0), (1, 4448.0, 39.6), (6, 4430.0, 39.5))
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
    assert report["status"] == "COMPLETED"
    assert report["report"]["run_status"] == "COMPLETED"
    assert report["report"]["evidence_status"] == "INSUFFICIENT"
    assert report["report"]["counts"]["events"] == 6
    assert report["report"]["counts"]["short"] == 1
    assert report["report"]["windows"]["oos"]["evidence_status"] == "INSUFFICIENT"
    assert report["report"]["dataset"]["quality"] == "RECONSTRUCTED"
    assert "news archive" not in " ".join(report["report"]["missing"])
    paper = deploy_paper(store, agent.id, now=_utc(9, 2))
    assert paper.status == AgentStatus.PAPER

    published = _utc(14, 30)
    text = "美国8月非农远高于预期"
    event = EventRecord("feed-nfp", "", published, published, published, text, text, "feed")
    feed = ScriptedFeed()
    loop = IngestLoop(store, feed, interval=0)
    loop.start()
    try:
        feed.push(IngestBatch(published + timedelta(seconds=30), _prints(published, 0), event))
        _wait(lambda: any(row["kind"] == "EVENT_MATCHED" for row in store.list_activities(agent.id)))
        assert store.list_paper_trades(agent.id) == []

        feed.push(IngestBatch(published + timedelta(minutes=1), _prints(published, 1), event))
        _wait(lambda: len(store.list_signals(agent.id)) == 1)
        assert store.list_signals(agent.id)[0]["side"] == "SHORT"
        assert store.list_signals(agent.id)[0]["entry_price"] == 4448.0
        assert store.list_signals(agent.id)[0]["version_id"] == version.id

        feed.push(IngestBatch(published + timedelta(minutes=6), _prints(published, 6), None))
        _wait(
            lambda: bool(store.list_paper_trades(agent.id))
            and store.list_paper_trades(agent.id)[0]["net_return"] is not None
        )
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


def test_ingest_dedups_identical_batches_and_backs_off_after_errors(tmp_path):
    store = ProductStore(tmp_path / "feed.sqlite")
    published = _utc(14, 30)
    batch = IngestBatch(published, _prints(published, 0), None)

    class Counting:
        def __init__(self) -> None:
            self.ticks = 0

        def tick(self, _store, *, now, observations, event):
            self.ticks += 1
            return {"signals": [], "settled": []}

    runtime = Counting()
    feed = ScriptedFeed()
    feed.push(batch)
    feed.push(batch)
    loop = IngestLoop(store, feed, runtime, interval=0)
    loop.start()
    try:
        _wait(lambda: runtime.ticks == 1)
        time.sleep(0.15)
        assert runtime.ticks == 1
        assert loop.health.snapshot()["ingest"] == "READY"
    finally:
        loop.stop()

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def poll(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("jin10 down")
            return None

    flaky = IngestLoop(store, Flaky(), interval=0)
    flaky.start()
    try:
        _wait(lambda: flaky.health.snapshot()["last_error"] is not None)
        assert "jin10 down" in flaky.health.snapshot()["last_error"]
        assert flaky.health.snapshot()["ingest"] == "DEGRADED"
        _wait(lambda: flaky.health.snapshot()["ingest"] == "READY")
        assert flaky.health.consecutive_failures == 0
    finally:
        flaky.stop()
    store.close()
