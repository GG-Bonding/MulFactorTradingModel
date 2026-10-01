from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from gold_signal.compiler import compile_idea, render_hypothesis
from gold_signal.domain.models import TransitionError
from gold_signal.hypothesis import Condition, parse_hypothesis
from gold_signal.archive import default_archive, factors_for
from gold_signal.research import (
    EVIDENCE_VALID_MIN_TRADES,
    changed_confirmations,
    compare_reports,
    evidence_grade,
    sweep_reaction_threshold,
)
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import (
    add_hypothesis_version,
    create_from_idea,
    create_from_yaml,
    deploy_paper,
    pause_agent,
    run_backtest,
)
from gold_signal.runtime.agent_loop import AgentRuntime

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def mount_ui(app: FastAPI) -> None:
    @app.get("/")
    @app.get("/agents")
    def agents_page(request: Request):
        store = request.app.state.store
        return TEMPLATES.TemplateResponse(
            request,
            "agents.html",
            {"rows": _agent_rows(store), "feed": _feed_view(request)},
        )

    @app.get("/agents/new")
    def new_page(request: Request):
        return TEMPLATES.TemplateResponse(
            request,
            "new.html",
            {"idea": "", "spec": None, "editing": False, "error": None},
        )

    @app.post("/agents/new")
    async def new_submit(request: Request):
        form = await request.form()
        idea = str(form.get("idea") or "")
        action = str(form.get("action") or "compile")
        if action == "preview":
            action = "compile"
        factors = [str(item) for item in form.getlist("factor")]
        operators = [str(item) for item in form.getlist("operator")]
        values = [str(item) for item in form.getlist("value")]
        edited = _edited_spec(idea, factors, operators, values) if factors else None
        if action == "create":
            if edited is not None:
                yaml, error = edited
                if error:
                    return TEMPLATES.TemplateResponse(
                        request, "new.html", {"idea": idea, "spec": None, "editing": False, "error": error}, status_code=400
                    )
                agent, _version = create_from_yaml(request.app.state.store, idea, yaml)
                return RedirectResponse(f"/agents/{agent.id}", status_code=303)
            result = compile_idea(idea)
            if not result.ok or result.spec is None:
                return TEMPLATES.TemplateResponse(
                    request,
                    "new.html",
                    {"idea": idea, "spec": None, "editing": False, "error": result.errors[0] if result.errors else "无法编译"},
                    status_code=400,
                )
            agent, _version = create_from_idea(request.app.state.store, idea)
            return RedirectResponse(f"/agents/{agent.id}", status_code=303)
        result = compile_idea(idea)
        spec = None if not result.ok or result.spec is None else _spec_view(result.spec)
        if edited is not None and action == "edit":
            yaml, error = edited
            spec = None if error else _spec_view(parse_hypothesis(yaml))
            if error:
                return TEMPLATES.TemplateResponse(
                    request, "new.html", {"idea": idea, "spec": spec, "editing": True, "error": error}, status_code=400
                )
        error = None if spec else (result.errors[0] if result.errors else "无法编译")
        status = 200 if spec else 400
        return TEMPLATES.TemplateResponse(
            request,
            "new.html",
            {"idea": idea, "spec": spec, "editing": action == "edit" and spec is not None, "error": error},
            status_code=status,
        )

    @app.get("/agents/{agent_id}")
    def detail_page(request: Request, agent_id: str):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        if agent is None:
            return TEMPLATES.TemplateResponse(request, "missing.html", {"message": "Agent not found"}, status_code=404)
        return TEMPLATES.TemplateResponse(request, "detail.html", _detail_context(store, agent, error=None, request=request))

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
            context = _detail_context(store, agent, error=str(exc), request=request)
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
                request, "detail.html", _detail_context(store, agent, error=str(exc), request=request), status_code=409
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
            observations = _prints(published, through=0, headline=headline)
        elif stage == "minute":
            now = published + timedelta(minutes=1)
            observations = _prints(published, through=1, headline=headline)
        else:
            now = published + timedelta(minutes=6)
            observations = _prints(published, through=6, headline=headline)
            event = None
        AgentRuntime().tick(store, now=now, observations=observations, event=event)
        return RedirectResponse(f"/agents/{agent_id}", status_code=303)

    @app.get("/agents/{agent_id}/backtest")
    def backtest_latest(request: Request, agent_id: str):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        if agent is None:
            return TEMPLATES.TemplateResponse(request, "missing.html", {"message": "Agent not found"}, status_code=404)
        runs = [
            row
            for row in store.list_backtests(agent_id)
            if agent.active_version_id is None or row["version_id"] == agent.active_version_id
        ]
        if not runs:
            return RedirectResponse(f"/agents/{agent_id}", status_code=303)
        return RedirectResponse(f"/agents/{agent_id}/backtests/{runs[-1]['id']}", status_code=303)

    @app.get("/agents/{agent_id}/backtests/{run_id}")
    def backtest_page(request: Request, agent_id: str, run_id: str):
        store = request.app.state.store
        agent = store.get_agent(agent_id)
        run = next((item for item in store.list_backtests(agent_id) if item["id"] == run_id), None)
        if agent is None or run is None:
            return TEMPLATES.TemplateResponse(request, "missing.html", {"message": "Backtest not found"}, status_code=404)
        report = _view_report(run["report"])
        return TEMPLATES.TemplateResponse(
            request,
            "backtest.html",
            {
                "agent": agent,
                "run": run,
                "report": report,
                "pct": _pct,
                "windows": _windows(report),
                "min_trades": EVIDENCE_VALID_MIN_TRADES,
                "feed": _feed_view(request),
                "trace_groups": _trace_groups(report),
                "sweep": _threshold_sweep(store, run),
            },
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


def _detail_context(store, agent, error: str | None, request: Request) -> dict:
    version = store.get_version(agent.active_version_id) if agent.active_version_id else None
    spec = parse_hypothesis(version.hypothesis_yaml) if version else None
    runs = [
        row
        for row in store.list_backtests(agent.id)
        if agent.active_version_id is None or row["version_id"] == agent.active_version_id
    ]
    latest = runs[-1] if runs else None
    report = _view_report(latest["report"] if latest else None)
    signals = store.list_signals(agent.id)
    trades = store.list_paper_trades(agent.id)
    activities = store.list_activities(agent.id)
    windows = _windows(report)
    oos = next((window for window in windows if window.get("name") == "oos"), None)
    paper_nets = [row["net_return"] for row in trades if row["net_return"] is not None]
    return {
        "agent": agent,
        "version": version,
        "versions": store.list_versions(agent.id),
        "spec": None if spec is None else _spec_view(spec),
        "latest": latest,
        "report": report,
        "windows": windows,
        "oos": oos,
        "signals": list(reversed(signals))[:8],
        "trades": list(reversed(trades))[:8],
        "paper_count": len(trades),
        "paper_avg": None if not paper_nets else sum(paper_nets) / len(paper_nets),
        "current_side": signals[-1]["side"] if signals else "—",
        "current_reason": signals[-1]["reason"] if signals else "",
        "current_at": _minute(signals[-1]["evaluated_at"] if signals else None),
        "monitor": _monitor(activities, signals, trades),
        "error": error,
        "pct": _pct,
        "published_at": "2026-03-24T14:30:00+00:00",
        "headline": "美国8月非农就业人数低于预期",
        "can_paper": _can_paper(agent, report),
        "min_trades": EVIDENCE_VALID_MIN_TRADES,
        "feed": _feed_view(request),
        "comparison": _version_comparison(store, agent),
    }


def _agent_rows(store) -> list[dict]:
    rows = []
    for agent in store.list_agents():
        signals = store.list_signals(agent.id)
        trades = store.list_paper_trades(agent.id)
        runs = [
            row
            for row in store.list_backtests(agent.id)
            if agent.active_version_id is None or row["version_id"] == agent.active_version_id
        ]
        report = _view_report(runs[-1]["report"]) if runs else None
        last = signals[-1] if signals else None
        rows.append(
            {
                "agent": agent,
                "side": None if last is None else last["side"],
                "last_signal": _minute(None if last is None else last.get("evaluated_at")),
                "historical_trades": None if report is None else report.get("trades"),
                "paper_trades": len(trades),
            }
        )
    return rows


def _edited_spec(idea: str, factors: list[str], operators: list[str], values: list[str]) -> tuple[str, str | None]:
    result = compile_idea(idea)
    if not result.ok or result.spec is None:
        return "", result.errors[0] if result.errors else "无法编译"
    if not (len(factors) == len(operators) == len(values)) or not factors:
        return "", "每条规则都要有因子、比较和数值"
    confirmations: list[Condition] = []
    for factor, operator, raw in zip(factors, operators, values):
        text = raw.strip()
        try:
            number = None if text == "" else float(text)
        except ValueError:
            return "", f"数值无法读取: {raw}"
        confirmations.append(Condition(factor.strip(), operator.strip(), number))
    spec = replace(result.spec, confirmations=tuple(confirmations))
    try:
        yaml = render_hypothesis(spec)
        parse_hypothesis(yaml)
    except ValueError as exc:
        return "", str(exc)
    return yaml, None


def _monitor(activities: list[dict], signals: list[dict], trades: list[dict]) -> dict | None:
    outcome = next((row for row in reversed(activities) if row["kind"] == "OUTCOME_UPDATED"), None)
    matched = next((row for row in reversed(activities) if row["kind"] == "EVENT_MATCHED"), None)
    signal = signals[-1] if signals else None
    trade = None
    if signal is not None:
        trade = next((row for row in reversed(trades) if row.get("signal_id") == signal["id"]), None)
    if trade is not None and trade.get("net_return") is not None:
        return {
            "status": "CLOSED",
            "position": "CLOSED",
            "signal": signal,
            "trade": trade,
            "exit_reason": None if outcome is None else outcome["detail"].get("exit_reason"),
            "exit_price": None if outcome is None else outcome["detail"].get("exit_price"),
        }
    if trade is not None:
        return {
            "status": "SIGNAL",
            "position": "OPEN",
            "signal": signal,
            "trade": trade,
            "exit_reason": None,
            "exit_price": None,
        }
    if matched is not None:
        return {
            "status": "WAITING_CONFIRMATION",
            "position": None,
            "signal": None,
            "trade": None,
            "exit_reason": None,
            "exit_price": None,
        }
    return None


def _minute(value: str | None) -> str:
    if not value:
        return "—"
    return str(value).replace("T", " ")[:16]


def _trace_groups(report: dict | None) -> list[dict]:
    traces = [] if not report else report.get("traces") or []
    filled, rejected, waiting = [], [], []
    for trace in traces:
        side = trace.get("side")
        if side in ("LONG", "SHORT"):
            filled.append(trace)
        elif side == "WAITING":
            waiting.append(trace)
        else:
            rejected.append(trace)
    return [
        {"name": "Filled trades", "rows": filled},
        {"name": "Rejected events", "rows": rejected},
        {"name": "Waiting", "rows": waiting},
    ]


def _threshold_sweep(store, run: dict) -> list[dict]:
    version = store.get_version(run["version_id"])
    if version is None:
        return []
    spec = parse_hypothesis(version.hypothesis_yaml)
    if spec.drive == "BAR" or not any(
        condition.factor.endswith(("reaction_1bar", "reaction_1m")) and condition.value is not None
        for condition in spec.confirmations
    ):
        return []
    report = run["report"]
    start = str(report.get("start") or "2024-01-01")
    end = str(report.get("end") or "2026-09-30")
    start_at = datetime.fromisoformat(start[:10]).replace(tzinfo=timezone.utc)
    end_at = datetime.fromisoformat(end[:10]).replace(tzinfo=timezone.utc) + timedelta(days=1)
    events, observations, gaps = default_archive().load(start_at, end_at, factors_for(spec))
    if gaps:
        return []
    return sweep_reaction_threshold(spec, events, observations, (0.0004, 0.0008, 0.0012), start, end)


def _version_comparison(store, agent) -> dict | None:
    versions = store.list_versions(agent.id)
    runs = store.list_backtests(agent.id)
    paired = []
    for version in versions:
        own = [row for row in runs if row["version_id"] == version.id]
        if own:
            paired.append((version, _view_report(own[-1]["report"])))
    if len(paired) < 2:
        return None
    (left_version, left_report), (right_version, right_report) = paired[-2], paired[-1]
    compared = compare_reports(left_report, right_report)
    changes = changed_confirmations(
        parse_hypothesis(left_version.hypothesis_yaml),
        parse_hypothesis(right_version.hypothesis_yaml),
    )
    labels = (
        ("Evidence", "evidence_status", False),
        ("Trades", "trades", False),
        ("Win rate", "win_rate", True),
        ("Avg net", "avg_return", True),
        ("Median", "median_return", True),
        ("Profit factor", "profit_factor", False),
        ("End equity", "end_equity", False),
    )
    rows = []
    for label, key, is_percent in labels:
        rows.append(
            {
                "label": label,
                "left": compared["left"].get(key),
                "right": compared["right"].get(key),
                "percent": is_percent,
            }
        )
    return {
        "left_name": f"v{left_version.version}",
        "right_name": f"v{right_version.version}",
        "rows": rows,
        "changed": changes,
    }


def _spec_view(spec) -> dict:
    hold = int(spec.exit.max_hold.total_seconds() // 60)
    return {
        "trigger": spec.trigger or "—",
        "entry": spec.entry,
        "asset": spec.asset,
        "horizons": list(spec.horizons),
        "confirmations": [
            {
                "factor": item.factor,
                "operator": item.operator,
                "value": item.value,
                "shown": _shown(item.value),
            }
            for item in spec.confirmations
        ],
        "exit": {
            "max_hold": f"{hold}m",
            "take_profit": spec.exit.take_profit,
            "stop_loss": spec.exit.stop_loss,
        },
    }


def _shown(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) <= 1:
        return f"{value:.2%}"
    return f"{value:g}"


def _view_report(report: dict | None) -> dict | None:
    """Older saved runs used status OK. Show them with the evidence grade."""
    if not report or report.get("run_status"):
        return report
    viewed = dict(report)
    counts = viewed.get("counts") or {}
    events = int(counts.get("events") or 0)
    trades = int(counts.get("long") or 0) + int(counts.get("short") or 0)
    if viewed.get("status") == "OK":
        viewed["run_status"] = "COMPLETED"
        viewed["evidence_status"] = evidence_grade(events, trades)
    else:
        viewed["run_status"] = "FAILED"
        viewed["evidence_status"] = "NO_DATA"
    viewed["events"] = events
    viewed["trades"] = trades
    viewed["windows"] = {name: _view_window(window) for name, window in (viewed.get("windows") or {}).items()}
    return viewed


def _view_window(window: dict) -> dict:
    if window.get("evidence_status"):
        return window
    viewed = dict(window)
    if window.get("status") == "OK":
        counts = window.get("counts") or {}
        events = int(counts.get("events") or 0)
        trades = int(counts.get("long") or 0) + int(counts.get("short") or 0)
        if not counts:
            trades = int(window.get("samples") or 0)
            events = 1 if trades else 0
        viewed["run_status"] = "COMPLETED"
        viewed["evidence_status"] = evidence_grade(events, trades)
    else:
        viewed["run_status"] = "COMPLETED"
        viewed["evidence_status"] = "NO_DATA"
    return viewed


def _windows(report: dict | None) -> list[dict]:
    if not report:
        return []
    windows = report.get("windows") or {}
    return [{"name": name, **(windows.get(name) or {})} for name in ("train", "validation", "oos")]


def _with_threshold(yaml: str, value: float) -> str:
    spec = parse_hypothesis(yaml)
    confirmations = tuple(
        replace(item, value=value)
        if item.factor.endswith(("reaction_1m", "reaction_1bar")) and item.value is not None
        else item
        for item in spec.confirmations
    )
    return render_hypothesis(replace(spec, confirmations=confirmations))


def _can_paper(agent, report: dict | None) -> bool:
    if agent.status.value != "BACKTESTED" or not report:
        return False
    return report.get("run_status") == "COMPLETED" and report.get("evidence_status") in ("INSUFFICIENT", "VALID")


def _feed_view(request: Request) -> dict:
    health = getattr(request.app.state, "feed", None)
    snap = health.snapshot() if health is not None and hasattr(health, "snapshot") else {
        "ingest": "OFF",
        "last_success_at": None,
        "last_market_at": None,
        "last_error": None,
    }
    label = {"READY": "LIVE", "DEGRADED": "DEGRADED", "STARTING": "STARTING"}.get(snap["ingest"], "OFF")
    return {
        "label": label,
        "updated": _ago(snap.get("last_success_at")),
        "error": snap.get("last_error") if label == "DEGRADED" else None,
    }


def _ago(value: str | None) -> str:
    if not value:
        return "—"
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(timezone.utc) - stamp).total_seconds()))
    if seconds < 60:
        return f"{seconds} sec ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60} hr ago"


def _prints(published: datetime, through: int, headline: str = "") -> list[Observation]:
    falling = "高于" in headline
    points = (
        ((0, 4517.0, 40.0), (1, 4448.0, 39.6), (6, 4430.0, 39.5))
        if falling
        else ((0, 3800.0, 40.0), (1, 3810.0, 40.1), (6, 3830.0, 40.1))
    )
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
