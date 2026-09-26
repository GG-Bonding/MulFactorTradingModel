from __future__ import annotations

import uuid
from datetime import datetime, timezone

from gold_signal.compiler import compile_idea
from gold_signal.domain.models import (
    Agent,
    AgentStatus,
    AgentVersion,
    activate,
    add_version,
    create_agent,
    transition,
)
from gold_signal.hypothesis import historical_validation, parse_hypothesis
from gold_signal.persistence.store import ProductStore, VersionImmutable


def create_from_idea(store: ProductStore, idea: str, *, now: datetime | None = None) -> tuple[Agent, AgentVersion]:
    result = compile_idea(idea)
    if not result.ok or result.spec is None:
        raise ValueError(result.errors[0] if result.errors else "无法编译")
    stamp = now or _now()
    agent = create_agent(uuid.uuid4().hex, result.spec.name, idea.strip(), stamp)
    version = add_version(agent.id, [], result.yaml, stamp)
    agent = activate(agent, version, at=stamp)
    store.save_agent(agent)
    store.save_version(version)
    store.insert_activity(uuid.uuid4().hex, agent.id, "AGENT_CREATED", {"name": agent.name}, stamp)
    store.insert_activity(
        uuid.uuid4().hex,
        agent.id,
        "VERSION_CREATED",
        {"version": version.version, "sha256": version.hypothesis_sha256},
        stamp,
    )
    return agent, version


def add_hypothesis_version(store: ProductStore, agent_id: str, yaml: str, *, now: datetime | None = None) -> AgentVersion:
    parse_hypothesis(yaml)
    agent = _require_agent(store, agent_id)
    stamp = now or _now()
    version = add_version(agent.id, store.list_versions(agent.id), yaml, stamp)
    store.save_version(version)
    store.insert_activity(
        uuid.uuid4().hex,
        agent.id,
        "VERSION_CREATED",
        {"version": version.version, "sha256": version.hypothesis_sha256},
        stamp,
    )
    return version


def run_backtest(
    store: ProductStore,
    agent_id: str,
    *,
    now: datetime | None = None,
    events: list | None = None,
    observations: list | None = None,
) -> dict:
    agent = _require_agent(store, agent_id)
    if agent.active_version_id is None:
        raise ValueError("agent has no version")
    version = store.get_version(agent.active_version_id)
    if version is None:
        raise ValueError("active version is missing")
    stamp = now or _now()
    store.insert_activity(uuid.uuid4().hex, agent.id, "BACKTEST_STARTED", {"version_id": version.id}, stamp)
    spec = parse_hypothesis(version.hypothesis_yaml)
    if events is not None and observations is not None:
        from gold_signal.research import replay_report

        report = replay_report(spec, "2024-01-01", "2026-09-30", events, observations)
    else:
        report = historical_validation(spec, "2024-01-01", "2026-09-30", strategy_path=None)
    run_id = uuid.uuid4().hex
    store.insert_backtest(run_id, agent.id, version.id, report, stamp)
    if agent.status == AgentStatus.DRAFT and report.get("status") == "OK":
        agent = transition(agent, AgentStatus.VALIDATED, at=stamp, backtest_status="OK")
        agent = transition(agent, AgentStatus.BACKTESTED, at=stamp)
    elif agent.status == AgentStatus.DRAFT:
        agent = transition(agent, AgentStatus.BACKTESTED, at=stamp)
    store.save_agent(agent)
    store.insert_activity(
        uuid.uuid4().hex,
        agent.id,
        "BACKTEST_FINISHED",
        {"run_id": run_id, "version_id": version.id, "status": report.get("status")},
        stamp,
    )
    saved = store.list_backtests(agent.id)
    return next(item for item in saved if item["id"] == run_id)


def deploy_paper(store: ProductStore, agent_id: str, *, now: datetime | None = None) -> Agent:
    agent = _require_agent(store, agent_id)
    stamp = now or _now()
    agent = transition(agent, AgentStatus.PAPER, at=stamp)
    store.save_agent(agent)
    store.insert_activity(uuid.uuid4().hex, agent.id, "DEPLOYED", {"version_id": agent.active_version_id}, stamp)
    return agent


def pause_agent(store: ProductStore, agent_id: str, *, now: datetime | None = None) -> Agent:
    agent = _require_agent(store, agent_id)
    stamp = now or _now()
    agent = transition(agent, AgentStatus.PAUSED, at=stamp)
    store.save_agent(agent)
    store.insert_activity(uuid.uuid4().hex, agent.id, "PAUSED", {}, stamp)
    return agent


def record_signal(store: ProductStore, agent_id: str, version_id: str, signal: dict, *, now: datetime | None = None) -> dict:
    stamp = now or _now()
    row = {
        "id": signal.get("id") or uuid.uuid4().hex,
        "agent_id": agent_id,
        "version_id": version_id,
        "event_id": signal["event_id"],
        "side": signal["side"],
        "entry_price": signal.get("entry_price"),
        "reason": signal.get("reason") or "",
        "evaluated_at": signal.get("evaluated_at") or _iso(stamp),
        "created_at": _iso(stamp),
    }
    store.insert_signal(row)
    store.insert_activity(uuid.uuid4().hex, agent_id, "SIGNAL_CREATED", {"signal_id": row["id"], "version_id": version_id}, stamp)
    return row


def record_paper_trade(store: ProductStore, trade: dict, *, now: datetime | None = None) -> dict:
    stamp = now or _now()
    row = {
        "id": trade.get("id") or uuid.uuid4().hex,
        "agent_id": trade["agent_id"],
        "version_id": trade["version_id"],
        "signal_id": trade.get("signal_id"),
        "side": trade["side"],
        "entry_price": trade.get("entry_price"),
        "spec_sha256": trade["spec_sha256"],
        "status": trade.get("status") or "PAPER",
        "net_return": trade.get("net_return"),
        "created_at": _iso(stamp),
    }
    store.insert_paper_trade(row)
    store.insert_activity(
        uuid.uuid4().hex,
        trade["agent_id"],
        "PAPER_ORDER_FILLED",
        {"trade_id": row["id"], "version_id": trade["version_id"]},
        stamp,
    )
    return row


def _require_agent(store: ProductStore, agent_id: str) -> Agent:
    agent = store.get_agent(agent_id)
    if agent is None:
        raise KeyError(agent_id)
    return agent


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


__all__ = [
    "VersionImmutable",
    "add_hypothesis_version",
    "create_from_idea",
    "deploy_paper",
    "pause_agent",
    "record_paper_trade",
    "record_signal",
    "run_backtest",
]
