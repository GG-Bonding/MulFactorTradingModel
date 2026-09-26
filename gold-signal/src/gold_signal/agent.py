from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from gold_signal.compiler import CompileResult, compile_idea, render_hypothesis
from gold_signal.hypothesis import HypothesisSpec, _check, load_hypothesis, parse_hypothesis, trigger_matches
from gold_signal.models import FlashNews, MarketSnapshot
from gold_signal.observation import EventRecord, FactorResolver
from gold_signal.replay_clock import DecisionRecord, TradeOutcome
from gold_signal.research import (
    confirmation_value,
    edge_distribution,
)

_EVENT_LABELS = {
    "CPI_BELOW_EXPECTATION": "CPI 低于预期",
    "CPI_ABOVE_EXPECTATION": "CPI 高于预期",
    "NFP_BELOW_EXPECTATION": "非农低于预期",
    "NFP_ABOVE_EXPECTATION": "非农高于预期",
    "GEOPOLITICAL_ESCALATION": "地缘冲突升级",
}


class ExecutionRefused(RuntimeError):
    """An order that did not come from a versioned hypothesis."""


def deploy_agent(spec: HypothesisSpec, root: Path) -> dict:
    """Write the spec the user can open and edit. The hash is the version."""
    yaml = render_hypothesis(spec)
    parsed = parse_hypothesis(yaml)
    if parsed != spec and (parsed.id, parsed.trigger, parsed.confirmations, parsed.entry) != (
        spec.id,
        spec.trigger,
        spec.confirmations,
        spec.entry,
    ):
        raise ValueError("spec does not roundtrip")
    folder = root / "agents"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{spec.id}.yaml"
    path.write_text(yaml, encoding="utf-8")
    digest = hashlib.sha256(yaml.encode("utf-8")).hexdigest()
    meta = {"id": spec.id, "sha256": digest, "path": str(path.name)}
    (folder / f"{spec.id}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def agent_view(
    spec: HypothesisSpec,
    event: EventRecord,
    observations: list,
    historical: dict | None = None,
    paper_returns: list[float] | None = None,
    context: object | None = None,
) -> dict:
    """Explain the engine decision. The numbers come from the replay, not from a model."""
    from gold_signal.runtime.context import ReplayEvaluationContext, evaluation_now
    from gold_signal.runtime.engine import HypothesisEngine

    if context is None:
        context = ReplayEvaluationContext(tuple(observations), evaluation_now(event))
    decision = HypothesisEngine().evaluate(spec, event, context)
    now = context.now
    conditions = explain_conditions(spec, event, observations, now)
    passed = sum(1 for row in conditions if row["passed"])
    yaml = render_hypothesis(spec)
    return {
        "agent": spec.name,
        "asset": spec.asset,
        "side": decision.side,
        "conditions": conditions,
        "passed": f"{passed}/{len(conditions)}",
        "entry": decision.entry_price,
        "signal_time": decision.evaluated_at.isoformat(),
        "reason": decision.reasons[0] if decision.reasons else "",
        "spec_sha256": hashlib.sha256(yaml.encode("utf-8")).hexdigest(),
        "comparison": compare_runs(historical or {}, paper_returns or []),
        "yaml": yaml,
    }


def explain_conditions(
    spec: HypothesisSpec,
    event: EventRecord,
    observations: list,
    now: datetime,
) -> list[dict]:
    resolver = FactorResolver(observations)
    text = f"{event.title} {event.content}".strip()
    direction = 1 if spec.entry == "LONG" else -1 if spec.entry == "SHORT" else 0
    rows: list[dict] = []
    if spec.trigger:
        matched = trigger_matches(spec.trigger, text)
        rows.append(
            {
                "label": _EVENT_LABELS.get(spec.trigger, spec.trigger),
                "passed": matched,
                "detail": "matched" if matched else "not this event",
            }
        )
    published = event.published_at
    for condition in spec.confirmations:
        value = confirmation_value(condition.factor, resolver, published, now)
        label = _condition_label(condition.factor, condition.operator, condition.value)
        if value is None:
            rows.append({"label": label, "passed": False, "detail": "waiting"})
            continue
        values = {condition.factor: value}
        ok, _gap = _check(condition, values, direction)
        rows.append({"label": label, "passed": ok, "detail": f"{value:+.4%}"})
    return rows


def compare_runs(historical: dict, paper_returns: list[float]) -> dict:
    paper = edge_distribution(
        [
            TradeOutcome(value, gross_return=value, net_return=value, mfe=value, mae=value)
            for value in paper_returns
        ]
    )
    windows = historical.get("windows") or {}
    oos = windows.get("oos") or {}
    net = historical.get("net") or {}
    return {
        "historical_status": historical.get("status"),
        "historical_samples": historical.get("samples", 0),
        "historical_avg_net": net.get("avg_return"),
        "historical_median_net": net.get("median_return"),
        "historical_win_rate": net.get("win_rate"),
        "historical_profit_factor": net.get("profit_factor"),
        "historical_avg_mfe": net.get("avg_mfe"),
        "historical_avg_mae": net.get("avg_mae"),
        "oos_samples": oos.get("samples", 0),
        "oos_avg_net": oos.get("avg_return"),
        "paper_samples": paper["samples"],
        "paper_avg_net": paper["avg_return"],
    }


def order_from_decision(decision: DecisionRecord, spec_yaml: str, *, quantity: float = 1.0) -> dict:
    """An order exists only when the versioned engine emitted a priced side."""
    parsed = parse_hypothesis(spec_yaml)
    digest = hashlib.sha256(spec_yaml.encode("utf-8")).hexdigest()
    if decision.hypothesis_id != parsed.id:
        raise ExecutionRefused("decision is not from this hypothesis version")
    if decision.side not in ("LONG", "SHORT") or decision.entry_price is None:
        raise ExecutionRefused("engine did not emit a priced trade")
    return {
        "source": "hypothesis-engine",
        "spec_sha256": digest,
        "hypothesis_id": parsed.id,
        "side": decision.side,
        "entry": decision.entry_price,
        "event_id": decision.event_id,
        "quantity": quantity,
    }


def order_from_text(text: str) -> dict:
    raise ExecutionRefused("LLM text cannot place an order")


class PaperBroker:
    """Records engine orders. It does not read a chat message."""

    def __init__(self) -> None:
        self.orders: list[dict] = []

    def submit(self, order: dict) -> dict:
        if order.get("source") != "hypothesis-engine" or not order.get("spec_sha256"):
            raise ExecutionRefused("broker only accepts a versioned engine order")
        self.orders.append(order)
        return {"status": "PAPER", **order}


def format_agent_view(view: dict) -> str:
    lines = [
        f"Agent: {view['agent']}",
        "",
        "当前判断",
        f"{view['side']} {view['asset']}",
        "",
        "触发条件",
    ]
    for row in view["conditions"]:
        mark = "PASS" if row["passed"] else "WAIT"
        lines.append(f"[{mark}] {row['label']}  {row['detail']}")
    comparison = view["comparison"]
    lines.extend(
        [
            "",
            "历史证据",
            f"状态                {comparison['historical_status']}",
            f"样本数              {comparison['historical_samples']}",
            f"平均净收益          {_fmt(comparison['historical_avg_net'])}",
            f"中位数净收益        {_fmt(comparison['historical_median_net'])}",
            f"胜率                {_fmt(comparison['historical_win_rate'])}",
            f"Profit Factor       {_plain(comparison['historical_profit_factor'])}",
            f"平均 MFE            {_fmt(comparison['historical_avg_mfe'])}",
            f"平均 MAE            {_fmt(comparison['historical_avg_mae'])}",
            "",
            "Out-of-Sample",
            f"样本数              {comparison['oos_samples']}",
            f"平均净收益          {_fmt(comparison['oos_avg_net'])}",
            "",
            "Forward / Paper",
            f"样本数              {comparison['paper_samples']}",
            f"平均净收益          {_fmt(comparison['paper_avg_net'])}",
            "",
            "当前信号",
            f"Entry               {view['entry']}",
            f"Signal Time         {view['signal_time']}",
            f"Reason              {view['reason']}",
            f"Spec                {view['spec_sha256'][:12]}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_compile_cli(text: str) -> int:
    from rich.console import Console

    console = Console()
    result: CompileResult = compile_idea(text)
    if not result.ok or result.spec is None:
        for item in result.errors:
            console.print(item)
        return 1
    console.print(result.yaml)
    return 0


def run_agent_cli(strategy: Path, archive: Path | None = None) -> int:
    from rich.console import Console

    from gold_signal.hypothesis import historical_validation
    from gold_signal.research import load_archive

    console = Console()
    spec = load_hypothesis(strategy)
    yaml = strategy.read_text(encoding="utf-8")
    observations = None
    events = None
    if archive is not None and archive.exists():
        events, observations = load_archive(archive)
    report = historical_validation(
        spec,
        "2024-01-01",
        "2026-09-30",
        strategy_path=strategy,
        observations=observations,
        events=events,
    )
    comparison = compare_runs(report, [])
    console.print(f"Agent: {spec.name}")
    console.print(f"Spec {hashlib.sha256(yaml.encode('utf-8')).hexdigest()}")
    console.print(f"历史 {comparison['historical_status']} samples={comparison['historical_samples']}")
    console.print(f"OOS samples={comparison['oos_samples']} avg={_fmt(comparison['oos_avg_net'])}")
    console.print("Live event: none on this run. The engine does not invent a signal.")
    console.print("LLM text cannot place an order.")
    return 0


def _condition_label(factor: str, operator: str, value: float | None) -> str:
    if value is None:
        return f"{factor} {operator}"
    shown = f"{value:.2%}" if abs(value) < 1 else str(value)
    return f"{factor} {operator} {shown}"


def _plain(value: float | None) -> str:
    if value is None:
        return "INSUFFICIENT"
    return f"{value:.2f}"


def _fmt(value: float | None) -> str:
    if value is None:
        return "INSUFFICIENT"
    return f"{value:+.4%}" if abs(value) < 10 else f"{value:.2f}"


def view_from_market(
    spec: HypothesisSpec,
    news: FlashNews,
    market: MarketSnapshot,
    historical: dict | None = None,
    paper_returns: list[float] | None = None,
) -> dict:
    """Read the live snapshot as knowable closes and explain the spec. No model call."""
    from gold_signal.runtime.context import LiveEvaluationContext

    event = EventRecord(
        event_id=news.event_id,
        event_type="",
        published_at=news.published_at,
        available_at=news.published_at,
        ingested_at=market.as_of,
        title=news.title,
        content=news.content or news.title,
        source="live",
    )
    context = LiveEvaluationContext.from_market(market, event)
    return agent_view(spec, event, list(context.observations), historical, paper_returns, context)
