from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from gold_signal.compiler import compile_idea, render_hypothesis
from gold_signal.domain.models import TransitionError
from gold_signal.hypothesis import parse_hypothesis
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import (
    add_hypothesis_version,
    create_from_idea,
    deploy_paper,
    pause_agent,
    run_backtest,
)
from gold_signal.runtime.agent_loop import AgentRuntime

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def mount_ui(app: FastAPI) -> None:
    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/agents", status_code=303)

    @app.get("/agents")
    def agents_page(request: Request):
        store = request.app.state.store
        rows = []
        for agent in store.list_agents():
            signals = store.list_signals(agent.id)
            trades = store.list_paper_trades(agent.id)
            nets = [row["net_return"] for row in trades if row["net_return"] is not None]
            rows.append(
                {
                    "agent": agent,
                    "side": signals[-1]["side"] if signals else "FLAT",
                    "paper": None if not nets else sum(nets) / len(nets),
                }
            )
        return TEMPLATES.TemplateResponse(request, "agents.html", {"rows": rows, "pct": _pct})

    @app.get("/agents/new")
    def new_page(request: Request):
        return TEMPLATES.TemplateResponse(
            request,
            "new.html",
            {"idea": "", "spec": None, "error": None},
        )

    @app.post("/agents/new")
    def new_submit(request: Request, idea: str = Form(""), action: str = Form("preview")):
        result = compile_idea(idea)
        if action == "create":
            if not result.ok or result.spec is None:
                return TEMPLATES.TemplateResponse(
                    request,
                    "new.html",
                    {"idea": idea, "spec": None, "error": result.errors[0] if result.errors else "无法编译"},
                    status_code=400,
                )
            agent, _version = create_from_idea(request.app.state.store, idea)
            return RedirectResponse(f"/agents/{agent.id}", status_code=303)
        spec = None if not result.ok or result.spec is None else _spec_view(result.spec)
        error = None if spec else (result.errors[0] if result.errors else "无法编译")
        status = 200 if spec else 400
        return TEMPLATES.TemplateResponse(
            request,
            "new.html",
            {"idea": idea, "spec": spec, "error": error},
            status_code=status,
        )

    @app.get("/agents/{agent_id}")
    def detail_page(request: Request, agent_id: str):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        if agent is None:
            return TEMPLATES.TemplateResponse(request, "missing.html", {"message": "Agent not found"}, status_code=404)
        return TEMPLATES.TemplateResponse(request, "detail.html", _detail_context(store, agent, error=None))

    @app.post("/agents/{agent_id}/backtest")
    def backtest_action(request: Request, agent_id: str):
        run = run_backtest(request.app.state.store, agent_id)
        return RedirectResponse(f"/agents/{agent_id}/backtests/{run['id']}", status_code=303)

    @app.post("/agents/{agent_id}/deploy-paper")
    def deploy_action(request: Request, agent_id: str):
        store = request.app.state.store
        try:
            deploy_paper(store, agent_id)
        except TransitionError as exc:
            agent = store.get_agent(agent_id)
            context = _detail_context(store, agent, error=str(exc))
            return TEMPLATES.TemplateResponse(request, "detail.html", context, status_code=409)
        return RedirectResponse(f"/agents/{agent_id}", status_code=303)

    @app.post("/agents/{agent_id}/pause")
    def pause_action(request: Request, agent_id: str):
        store = request.app.state.store
        try:
            pause_agent(store, agent_id)
        except TransitionError as exc:
            agent = store.get_agent(agent_id)
            return TEMPLATES.TemplateResponse(
                request, "detail.html", _detail_context(store, agent, error=str(exc)), status_code=409
            )
        return RedirectResponse(f"/agents/{agent_id}", status_code=303)

    @app.post("/agents/{agent_id}/versions")
    def version_action(request: Request, agent_id: str, threshold_percent: float = Form(...)):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        if agent is None or agent.active_version_id is None:
            return RedirectResponse("/agents", status_code=303)
        current = store.get_version(agent.active_version_id)
        if current is None:
            return RedirectResponse(f"/agents/{agent_id}", status_code=303)
        yaml = _with_threshold(current.hypothesis_yaml, threshold_percent / 100.0)
        add_hypothesis_version(store, agent_id, yaml)
        return RedirectResponse(f"/agents/{agent_id}", status_code=303)

    @app.post("/agents/{agent_id}/ticks")
    def tick_action(
        request: Request,
        agent_id: str,
        stage: str = Form(...),
        headline: str = Form(...),
        published_at: str = Form(...),
    ):
        store = request.app.state.store
        published = datetime.fromisoformat(published_at)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        event = EventRecord(
            event_id=f"ui-{agent_id}-{published.isoformat()}",
            event_type="",
            published_at=published,
            available_at=published,
            ingested_at=published,
            title=headline,
            content=headline,
            source="ui",
        )
        if stage == "arrived":
            now = published + timedelta(seconds=30)
            observations = _prints(published, through=0)
        elif stage == "minute":
            now = published + timedelta(minutes=1)
            observations = _prints(published, through=1)
        else:
            now = published + timedelta(minutes=6)
            observations = _prints(published, through=6)
            event = None
        AgentRuntime().tick(store, now=now, observations=observations, event=event)
        return RedirectResponse(f"/agents/{agent_id}", status_code=303)

    @app.get("/agents/{agent_id}/backtests/{run_id}")
    def backtest_page(request: Request, agent_id: str, run_id: str):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        run = next((item for item in store.list_backtests(agent_id) if item["id"] == run_id), None)
        if agent is None or run is None:
            return TEMPLATES.TemplateResponse(request, "missing.html", {"message": "Backtest not found"}, status_code=404)
        report = run["report"]
        return TEMPLATES.TemplateResponse(
            request,
            "backtest.html",
            {"agent": agent, "run": run, "report": report, "pct": _pct, "windows": _windows(report)},
        )

    @app.get("/agents/{agent_id}/activity")
    def activity_page(request: Request, agent_id: str):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        if agent is None:
            return TEMPLATES.TemplateResponse(request, "missing.html", {"message": "Agent not found"}, status_code=404)
        return TEMPLATES.TemplateResponse(
            request,
            "activity.html",
            {"agent": agent, "activities": list(reversed(store.list_activities(agent_id)))},
        )


def _detail_context(store, agent, error: str | None) -> dict:
    version = store.get_version(agent.active_version_id) if agent.active_version_id else None
    spec = parse_hypothesis(version.hypothesis_yaml) if version else None
    runs = [
        row
        for row in store.list_backtests(agent.id)
        if agent.active_version_id is None or row["version_id"] == agent.active_version_id
    ]
    latest = runs[-1] if runs else None
    report = latest["report"] if latest else None
    signals = store.list_signals(agent.id)
    return {
        "agent": agent,
        "version": version,
        "versions": store.list_versions(agent.id),
        "spec": None if spec is None else _spec_view(spec),
        "latest": latest,
        "report": report,
        "windows": _windows(report),
        "signals": list(reversed(signals))[:8],
        "trades": list(reversed(store.list_paper_trades(agent.id)))[:8],
        "current_side": signals[-1]["side"] if signals else "FLAT",
        "current_reason": signals[-1]["reason"] if signals else "",
        "error": error,
        "pct": _pct,
        "published_at": "2026-03-24T14:30:00+00:00",
        "headline": "美国8月非农就业人数低于预期",
    }


def _spec_view(spec) -> dict:
    return {
        "trigger": spec.trigger or "—",
        "entry": spec.entry,
        "asset": spec.asset,
        "horizons": list(spec.horizons),
        "confirmations": [
            {"factor": item.factor, "operator": item.operator, "value": item.value} for item in spec.confirmations
        ],
    }


def _windows(report: dict | None) -> list[dict]:
    if not report:
        return []
    windows = report.get("windows") or {}
    return [{"name": name, **(windows.get(name) or {})} for name in ("train", "validation", "oos")]


def _with_threshold(yaml: str, value: float) -> str:
    spec = parse_hypothesis(yaml)
    confirmations = tuple(
        replace(item, value=value) if item.factor.endswith("reaction_1m") and item.value is not None else item
        for item in spec.confirmations
    )
    return render_hypothesis(replace(spec, confirmations=confirmations))


def _prints(published: datetime, through: int) -> list[Observation]:
    points = ((0, 3800.0, 40.0), (1, 3810.0, 40.1), (6, 3830.0, 40.1))
    rows: list[Observation] = []
    for minute, gold, silver in points:
        if minute > through:
            continue
        stamp = published + timedelta(minutes=minute)
        rows.append(Observation("market.XAUUSD.close", gold, stamp, stamp, stamp, "ui", "XAUUSD"))
        rows.append(Observation("market.XAGUSD.close", silver, stamp, stamp, stamp, "ui", "XAGUSD"))
    return rows


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2%}"
