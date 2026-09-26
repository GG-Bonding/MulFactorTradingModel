from datetime import datetime, timedelta, timezone

import pytest

from gold_signal.compiler import compile_idea
from gold_signal.domain import (
    AgentStatus,
    RunKind,
    TransitionError,
    activate,
    add_version,
    create_agent,
    start_run,
    transition,
)
from gold_signal.observation import EventRecord, Observation
from gold_signal.runtime import HypothesisEngine, LiveEvaluationContext, ReplayEvaluationContext


def _utc(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 3, 24, hour, minute, second, tzinfo=timezone.utc)


def _print(factor: str, value: float, stamp: datetime) -> Observation:
    return Observation(factor, value, stamp, stamp, stamp, "fixture", factor.split(".")[1])


def _tape(published: datetime) -> tuple[Observation, ...]:
    rows = []
    for minute in range(0, 8):
        stamp = published + timedelta(minutes=minute)
        gold = {0: 3800.0, 1: 3810.0}.get(minute, 3810.0)
        silver = 40.0 if minute == 0 else 40.1
        rows.append(_print("market.XAUUSD.close", gold, stamp))
        rows.append(_print("market.XAGUSD.close", silver, stamp))
    return tuple(rows)


def _event(published: datetime) -> EventRecord:
    text = "美国8月非农就业人数低于预期"
    return EventRecord("nfp-1", "", published, published, published, text, text, "fixture")


def test_live_and_replay_contexts_return_the_same_decision():
    published = _utc(14, 30)
    compiled = compile_idea("如果非农低于预期，而且黄金和白银在一分钟内同步上涨，我认为黄金还能继续涨。")
    assert compiled.spec is not None
    event = _event(published)
    observations = _tape(published)
    ready = published + timedelta(minutes=1)
    engine = HypothesisEngine()
    live = engine.evaluate(
        compiled.spec,
        event,
        LiveEvaluationContext(observations, ready),
    )
    replay = engine.evaluate(
        compiled.spec,
        event,
        ReplayEvaluationContext(observations, ready),
    )
    assert live == replay
    assert live.side == "LONG"
    assert live.entry_price == 3810.0

    early = published + timedelta(seconds=59)
    live_early = engine.evaluate(compiled.spec, event, LiveEvaluationContext(observations, early))
    replay_early = engine.evaluate(compiled.spec, event, ReplayEvaluationContext(observations, early))
    assert live_early == replay_early
    assert live_early.side == "WAITING"
    assert live_early.entry_price is None


def test_versions_stay_immutable_and_draft_cannot_skip_to_paper():
    now = _utc(9, 0)
    agent = create_agent("cpi-gold", "CPI Gold", "compiled idea", now)
    first = add_version(agent.id, [], "threshold: 0.0008\n", now)
    stored = [first]
    edited = add_version(agent.id, stored, "threshold: 0.0012\n", now + timedelta(minutes=1))
    stored.append(edited)
    assert first.hypothesis_yaml == "threshold: 0.0008\n"
    assert first.hypothesis_sha256 != edited.hypothesis_sha256
    assert edited.version == 2
    assert first.version == 1

    with pytest.raises(TransitionError, match="cannot transition to PAPER"):
        transition(agent, AgentStatus.PAPER, at=now)
    with pytest.raises(TransitionError, match="INSUFFICIENT"):
        transition(agent, AgentStatus.VALIDATED, at=now, backtest_status="INSUFFICIENT")

    saved = transition(agent, AgentStatus.BACKTESTED, at=now)
    assert saved.status == AgentStatus.BACKTESTED
    active = activate(saved, edited, at=now)
    paper_agent = transition(active, AgentStatus.PAPER, at=now)
    assert paper_agent.status == AgentStatus.PAPER
    paper = start_run(paper_agent, edited, RunKind.PAPER, at=now, run_id="run-1")
    assert paper.version_id == edited.id
    assert paper.kind == RunKind.PAPER
    backtest = start_run(saved, first, RunKind.BACKTEST, at=now, run_id="run-0")
    assert backtest.version_id == first.id
