from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from gold_signal.domain.models import Agent, AgentStatus, AgentVersion, TransitionError

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    active_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_versions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    version INTEGER NOT NULL,
    hypothesis_yaml TEXT NOT NULL,
    hypothesis_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (agent_id, version)
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    version_id TEXT NOT NULL REFERENCES agent_versions(id),
    status TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    version_id TEXT NOT NULL REFERENCES agent_versions(id),
    event_id TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL,
    reason TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    version_id TEXT NOT NULL REFERENCES agent_versions(id),
    signal_id TEXT,
    side TEXT NOT NULL,
    entry_price REAL,
    spec_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    net_return REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activities (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class VersionImmutable(TransitionError):
    """A saved hypothesis version cannot be rewritten."""


class ProductStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save_agent(self, agent: Agent) -> None:
        self._conn.execute(
            """
            INSERT INTO agents (id, name, description, status, active_version_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                status = excluded.status,
                active_version_id = excluded.active_version_id,
                updated_at = excluded.updated_at
            """,
            (
                agent.id,
                agent.name,
                agent.description,
                agent.status.value,
                agent.active_version_id,
                _stamp(agent.created_at),
                _stamp(agent.updated_at),
            ),
        )
        self._conn.commit()

    def get_agent(self, agent_id: str) -> Agent | None:
        row = self._conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return None if row is None else _agent(row)

    def list_agents(self) -> list[Agent]:
        rows = self._conn.execute("SELECT * FROM agents ORDER BY created_at, id").fetchall()
        return [_agent(row) for row in rows]

    def save_version(self, version: AgentVersion) -> None:
        existing = self.get_version(version.id)
        if existing is not None:
            if existing.hypothesis_yaml != version.hypothesis_yaml or existing.hypothesis_sha256 != version.hypothesis_sha256:
                raise VersionImmutable("agent version is immutable")
            return
        self._conn.execute(
            """
            INSERT INTO agent_versions (id, agent_id, version, hypothesis_yaml, hypothesis_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version.id,
                version.agent_id,
                version.version,
                version.hypothesis_yaml,
                version.hypothesis_sha256,
                _stamp(version.created_at),
            ),
        )
        self._conn.commit()

    def get_version(self, version_id: str) -> AgentVersion | None:
        row = self._conn.execute("SELECT * FROM agent_versions WHERE id = ?", (version_id,)).fetchone()
        return None if row is None else _version(row)

    def list_versions(self, agent_id: str) -> list[AgentVersion]:
        rows = self._conn.execute(
            "SELECT * FROM agent_versions WHERE agent_id = ? ORDER BY version",
            (agent_id,),
        ).fetchall()
        return [_version(row) for row in rows]

    def insert_backtest(self, run_id: str, agent_id: str, version_id: str, report: dict, now: datetime) -> None:
        self._conn.execute(
            """
            INSERT INTO backtest_runs (id, agent_id, version_id, status, report_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, agent_id, version_id, str(report.get("status") or "INSUFFICIENT"), json.dumps(report), _stamp(now)),
        )
        self._conn.commit()

    def list_backtests(self, agent_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM backtest_runs WHERE agent_id = ? ORDER BY created_at, id",
            (agent_id,),
        ).fetchall()
        return [_backtest(row) for row in rows]

    def insert_signal(self, signal: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO signals (
                id, agent_id, version_id, event_id, side, entry_price, reason, evaluated_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal["id"],
                signal["agent_id"],
                signal["version_id"],
                signal["event_id"],
                signal["side"],
                signal.get("entry_price"),
                signal.get("reason") or "",
                signal["evaluated_at"],
                signal["created_at"],
            ),
        )
        self._conn.commit()

    def list_signals(self, agent_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE agent_id = ? ORDER BY created_at, id",
            (agent_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_paper_trade(self, trade: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO paper_trades (
                id, agent_id, version_id, signal_id, side, entry_price, spec_sha256, status, net_return, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade["id"],
                trade["agent_id"],
                trade["version_id"],
                trade.get("signal_id"),
                trade["side"],
                trade.get("entry_price"),
                trade["spec_sha256"],
                trade["status"],
                trade.get("net_return"),
                trade["created_at"],
            ),
        )
        self._conn.commit()

    def list_paper_trades(self, agent_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM paper_trades WHERE agent_id = ? ORDER BY created_at, id",
            (agent_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_activity(self, activity_id: str, agent_id: str, kind: str, detail: dict, now: datetime) -> None:
        self._conn.execute(
            "INSERT INTO activities (id, agent_id, kind, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (activity_id, agent_id, kind, json.dumps(detail, ensure_ascii=False), _stamp(now)),
        )
        self._conn.commit()

    def list_activities(self, agent_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM activities WHERE agent_id = ? ORDER BY created_at, id",
            (agent_id,),
        ).fetchall()
        found = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item["detail"])
            found.append(item)
        return found


def _stamp(value: datetime) -> str:
    return value.isoformat()


def _parse_stamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _agent(row: sqlite3.Row) -> Agent:
    return Agent(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        status=AgentStatus(row["status"]),
        active_version_id=row["active_version_id"],
        created_at=_parse_stamp(row["created_at"]),
        updated_at=_parse_stamp(row["updated_at"]),
    )


def _version(row: sqlite3.Row) -> AgentVersion:
    return AgentVersion(
        id=row["id"],
        agent_id=row["agent_id"],
        version=int(row["version"]),
        hypothesis_yaml=row["hypothesis_yaml"],
        hypothesis_sha256=row["hypothesis_sha256"],
        created_at=_parse_stamp(row["created_at"]),
    )


def _backtest(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "version_id": row["version_id"],
        "status": row["status"],
        "report": json.loads(row["report_json"]),
        "created_at": row["created_at"],
    }
