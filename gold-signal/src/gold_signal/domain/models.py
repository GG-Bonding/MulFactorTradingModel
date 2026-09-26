from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    BACKTESTED = "BACKTESTED"
    PAPER = "PAPER"
    PAUSED = "PAUSED"


class RunKind(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"


class TransitionError(ValueError):
    """The product state machine rejected this change."""


_ALLOWED: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.DRAFT: frozenset({AgentStatus.VALIDATED, AgentStatus.BACKTESTED}),
    AgentStatus.VALIDATED: frozenset({AgentStatus.BACKTESTED}),
    AgentStatus.BACKTESTED: frozenset({AgentStatus.PAPER}),
    AgentStatus.PAPER: frozenset({AgentStatus.PAUSED}),
    AgentStatus.PAUSED: frozenset({AgentStatus.PAPER}),
}


@dataclass(frozen=True)
class Agent:
    """A long-lived product object. The strategy text lives on a version."""

    id: str
    name: str
    description: str
    status: AgentStatus
    active_version_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class AgentVersion:
    """An immutable hypothesis. Editing the threshold creates the next version."""

    id: str
    agent_id: str
    version: int
    hypothesis_yaml: str
    hypothesis_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class Run:
    """One backtest or paper session, pinned to a single version."""

    id: str
    agent_id: str
    version_id: str
    kind: RunKind
    status: str
    created_at: datetime


def create_agent(agent_id: str, name: str, description: str, now: datetime) -> Agent:
    return Agent(
        id=agent_id,
        name=name,
        description=description,
        status=AgentStatus.DRAFT,
        active_version_id=None,
        created_at=now,
        updated_at=now,
    )


def add_version(agent_id: str, existing: list[AgentVersion], yaml: str, now: datetime) -> AgentVersion:
    """Append a version. Earlier yaml strings stay as they were."""
    number = 1 + sum(1 for item in existing if item.agent_id == agent_id)
    text = yaml if yaml.endswith("\n") else yaml + "\n"
    return AgentVersion(
        id=f"{agent_id}-v{number}",
        agent_id=agent_id,
        version=number,
        hypothesis_yaml=text,
        hypothesis_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        created_at=now,
    )


def activate(agent: Agent, version: AgentVersion, *, at: datetime) -> Agent:
    if version.agent_id != agent.id:
        raise TransitionError("version belongs to another agent")
    return replace(agent, active_version_id=version.id, updated_at=at)


def transition(
    agent: Agent,
    target: AgentStatus,
    *,
    at: datetime,
    backtest_status: str | None = None,
) -> Agent:
    if target == AgentStatus.VALIDATED and backtest_status != "OK":
        raise TransitionError("INSUFFICIENT backtest cannot be marked VALIDATED")
    if target not in _ALLOWED[agent.status]:
        raise TransitionError(f"{agent.status.value} cannot transition to {target.value}")
    return replace(agent, status=target, updated_at=at)


def start_run(agent: Agent, version: AgentVersion, kind: RunKind, *, at: datetime, run_id: str) -> Run:
    if version.agent_id != agent.id:
        raise TransitionError("version belongs to another agent")
    if kind == RunKind.PAPER and agent.active_version_id != version.id:
        raise TransitionError("paper run must use the active version")
    return Run(
        id=run_id,
        agent_id=agent.id,
        version_id=version.id,
        kind=kind,
        status="RUNNING",
        created_at=at,
    )
