from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import replace

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gold_signal.compiler import compile_idea
from gold_signal.domain.models import TransitionError
from gold_signal.persistence.service import (
    add_hypothesis_version,
    create_from_idea,
    deploy_paper,
    pause_agent,
    run_backtest,
)
from gold_signal.persistence.store import ProductStore, VersionImmutable
from gold_signal.observation import EventRecord, Observation
from gold_signal.runtime.agent_loop import AgentRuntime


class IdeaIn(BaseModel):
    idea: str


class VersionIn(BaseModel):
    yaml: str


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class ObservationIn(BaseModel):
    factor: str
    value: float
    observed_at: datetime


class EventIn(BaseModel):
    event_id: str
    title: str
    content: str
    published_at: datetime
    available_at: datetime | None = None


class TickIn(BaseModel):
    now: datetime
    observations: list[ObservationIn]
    event: EventIn | None = None


def create_app(store: ProductStore) -> FastAPI:
    app = FastAPI(title="Trading Agent API")
    app.state.store = store
    _install_guards(app)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        store.ping()
        return {"status": "ready"}

    @app.post("/api/agents/compile")
    def compile_agent(body: IdeaIn) -> dict:
        result = compile_idea(body.idea)
        if not result.ok or result.spec is None:
            raise HTTPException(status_code=400, detail={"errors": list(result.errors)})
        spec = result.spec
        return {
            "asset": spec.asset,
            "entry": spec.entry,
            "trigger": spec.trigger,
            "confirmations": [
                {"factor": item.factor, "operator": item.operator, "value": item.value}
                for item in spec.confirmations
            ],
            "yaml": result.yaml,
        }

    @app.post("/api/agents", status_code=201)
    def create_agent_route(body: IdeaIn) -> dict:
        try:
            agent, version = create_from_idea(store, body.idea)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"agent": _agent_json(agent), "version": _version_json(version)}

    @app.put("/api/agents/{agent_id}")
    def update_agent(agent_id: str, body: AgentUpdate) -> dict:
        agent = store.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        updated = replace(
            agent,
            name=agent.name if body.name is None else body.name,
            description=agent.description if body.description is None else body.description,
            updated_at=datetime.now(timezone.utc),
        )
        store.save_agent(updated)
        return _agent_json(updated)

    @app.get("/api/agents")
    def list_agents() -> dict:
        return {"agents": [_agent_json(item) for item in store.list_agents()]}

    @app.get("/api/agents/{agent_id}")
    def get_agent(agent_id: str) -> dict:
        agent = store.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        return {
            "agent": _agent_json(agent),
            "versions": [_version_json(item) for item in store.list_versions(agent_id)],
        }

    @app.post("/api/agents/{agent_id}/versions", status_code=201)
    def create_version(agent_id: str, body: VersionIn) -> dict:
        try:
            version = add_hypothesis_version(store, agent_id, body.yaml)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="agent not found") from exc
        except (ValueError, VersionImmutable) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _version_json(version)

    @app.post("/api/agents/{agent_id}/backtests", status_code=201)
    def create_backtest(agent_id: str) -> dict:
        try:
            return run_backtest(store, agent_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="agent not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/agents/{agent_id}/backtests")
    def list_backtests(agent_id: str) -> dict:
        _missing(store, agent_id)
        return {"backtests": store.list_backtests(agent_id)}

    @app.post("/api/agents/{agent_id}/deploy-paper")
    def deploy(agent_id: str) -> dict:
        try:
            agent = deploy_paper(store, agent_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="agent not found") from exc
        except TransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _agent_json(agent)

    @app.post("/api/agents/{agent_id}/pause")
    def pause(agent_id: str) -> dict:
        try:
            agent = pause_agent(store, agent_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="agent not found") from exc
        except TransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _agent_json(agent)

    @app.get("/api/agents/{agent_id}/signals")
    def signals(agent_id: str) -> dict:
        _missing(store, agent_id)
        return {"signals": store.list_signals(agent_id)}

    @app.get("/api/agents/{agent_id}/trades")
    def trades(agent_id: str) -> dict:
        _missing(store, agent_id)
        return {"trades": store.list_paper_trades(agent_id)}

    @app.get("/api/agents/{agent_id}/activities")
    def activities(agent_id: str) -> dict:
        _missing(store, agent_id)
        return {"activities": store.list_activities(agent_id)}

    @app.get("/api/agents/{agent_id}/stats")
    def stats(agent_id: str) -> dict:
        agent = store.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        runs = store.list_backtests(agent_id)
        if agent.active_version_id is not None:
            scoped = [row for row in runs if row["version_id"] == agent.active_version_id]
        else:
            scoped = runs
        latest = scoped[-1]["report"] if scoped else None
        net = (latest or {}).get("net") or {}
        oos = ((latest or {}).get("windows") or {}).get("oos") or {}
        trades_rows = store.list_paper_trades(agent_id)
        paper_nets = [row["net_return"] for row in trades_rows if row["net_return"] is not None]
        paper_avg = None if not paper_nets else sum(paper_nets) / len(paper_nets)
        return {
            "backtest_status": None if latest is None else latest.get("status"),
            "samples": None if latest is None else latest.get("samples"),
            "avg_net": net.get("avg_return"),
            "median_net": net.get("median_return"),
            "win_rate": net.get("win_rate"),
            "profit_factor": net.get("profit_factor"),
            "avg_mfe": net.get("avg_mfe"),
            "avg_mae": net.get("avg_mae"),
            "oos_samples": oos.get("samples"),
            "oos_avg_net": oos.get("avg_return"),
            "paper_samples": len(paper_nets),
            "paper_avg_net": paper_avg,
            "missing": [] if latest is None else latest.get("missing") or [],
        }

    @app.post("/api/runtime/ticks")
    def tick(body: TickIn) -> dict:
        observations = [
            Observation(
                item.factor,
                item.value,
                item.observed_at,
                item.observed_at,
                item.observed_at,
                "runtime",
                item.factor.split(".")[1] if "." in item.factor else None,
            )
            for item in body.observations
        ]
        event = None
        if body.event is not None:
            event = EventRecord(
                event_id=body.event.event_id,
                event_type="",
                published_at=body.event.published_at,
                available_at=body.event.available_at or body.event.published_at,
                ingested_at=body.now,
                title=body.event.title,
                content=body.event.content,
                source="runtime",
            )
        result = AgentRuntime().tick(store, now=body.now, observations=observations, event=event)
        return {"signals": len(result["signals"]), "settled": result["settled"]}

    from gold_signal.ui.pages import mount_ui

    mount_ui(app)
    return app


def _install_guards(app: FastAPI) -> None:
    expected = os.environ.get("AGENT_BASIC_AUTH", "").strip()
    log = logging.getLogger("gold_signal.access")

    @app.middleware("http")
    async def guard(request, call_next):
        started = time.perf_counter()
        if expected and request.url.path not in ("/healthz", "/readyz"):
            header = request.headers.get("authorization", "")
            if not _basic_matches(header, expected):
                response = JSONResponse(
                    {"detail": "authentication required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                )
                _log_access(log, request, response.status_code, started)
                return response
        response = await call_next(request)
        _log_access(log, request, response.status_code, started)
        return response


def _basic_matches(header: str, expected: str) -> bool:
    if not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    return decoded == expected


def _log_access(log: logging.Logger, request, status: int, started: float) -> None:
    log.info(
        json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    )


def _missing(store: ProductStore, agent_id: str) -> None:
    if store.get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail="agent not found")


def _agent_json(agent: object) -> dict:
    return {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "status": agent.status.value,
        "active_version_id": agent.active_version_id,
    }


def _version_json(version: object) -> dict:
    return {
        "id": version.id,
        "agent_id": version.agent_id,
        "version": version.version,
        "hypothesis_sha256": version.hypothesis_sha256,
        "hypothesis_yaml": version.hypothesis_yaml,
    }
