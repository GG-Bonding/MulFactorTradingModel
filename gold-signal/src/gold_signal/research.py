from __future__ import annotations

import hashlib
import json
import statistics
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gold_signal.execution import ClosedTrade, execution_costs, portfolio_result, simulate_trade, walk_forward
from gold_signal.hypothesis import HypothesisSpec, _check, resolve_direction
from gold_signal.models import Bar
from gold_signal.news import CLASSIFIER_VERSION
from gold_signal.observation import EventRecord, FactorResolver, Observation
from gold_signal.replay_clock import (
    CostModel,
    DecisionRecord,
    TradeOutcome,
    measure_outcome,
    reaction_1bar,
)

HORIZONS: tuple[tuple[str, timedelta], ...] = (
    ("1m", timedelta(minutes=1)),
    ("3m", timedelta(minutes=3)),
    ("5m", timedelta(minutes=5)),
    ("15m", timedelta(minutes=15)),
    ("30m", timedelta(minutes=30)),
)
HEADLINE_HORIZON = "5m"

# Inclusive calendar ranges. Train through June 2025, validation through December 2025, OOS through September 2026.
RESEARCH_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("train", "2024-01-01", "2025-06-30"),
    ("validation", "2025-07-01", "2025-12-31"),
    ("oos", "2026-01-01", "2026-09-30"),
)

# A completed replay is not proof. Only VALID may be labeled historical evidence.
EVIDENCE_VALID_MIN_TRADES = 20


def price_factor(code: str) -> str:
    return f"market.{code}.close"


def evidence_grade(events: int, trades: int) -> str:
    """Separate 'the run finished' from 'the sample can support a claim'."""
    if events <= 0:
        return "NO_DATA"
    if trades <= 0:
        return "NO_TRADES"
    if trades < EVIDENCE_VALID_MIN_TRADES:
        return "INSUFFICIENT"
    return "VALID"


def quality_by_series(events: list[EventRecord], observations: list[Observation]) -> dict[str, str]:
    """One grade per series. Any reconstructed print keeps the series reconstructed."""
    grouped: dict[str, set[str]] = {}
    if events:
        grouped["News"] = {event.quality or "RECONSTRUCTED" for event in events}
    for row in observations:
        grouped.setdefault(_series_label(row), set()).add(row.quality or "RECONSTRUCTED")
    ordered = ["News", *sorted(name for name in grouped if name != "News")]
    return {name: _collapse_quality(grouped[name]) for name in ordered if name in grouped}


def series_code(factor: str) -> str:
    return factor.split(".", 1)[0]


def confirmation_value(
    factor: str,
    resolver: FactorResolver,
    published_at: datetime,
    now: datetime,
) -> float | None:
    """Only a print that is already knowable can confirm. Unsafe factors stay empty."""
    if factor.endswith(".reaction_1bar") or factor.endswith(".reaction_1m"):
        reaction = reaction_1bar(resolver, price_factor(series_code(factor)), published_at, now)
        return None if reaction is None else reaction.value
    return None


def bar_step(resolver: FactorResolver, factor: str, bar_time: datetime, now: datetime):
    """Return of the latest stored step ending at bar_time. The span is whatever the series actually is."""
    from gold_signal.replay_clock import BarReaction

    rows = [row for row in resolver.visible(factor, now) if row.observed_at <= bar_time]
    if len(rows) < 2 or rows[-1].observed_at != bar_time or rows[-1].value == 0 or rows[-2].value == 0:
        return None
    anchor, later = rows[-2], rows[-1]
    return BarReaction(later.value / anchor.value - 1, anchor.observed_at, later.observed_at)


def observations_from_bars(
    factor: str,
    bars: list[Bar],
    *,
    symbol: str,
    source: str,
    timestamp_kind: str,
    quality: str = "RECONSTRUCTED",
) -> list[Observation]:
    """A 1-minute close is knowable when that minute has finished.

    timestamp_kind "open" means bar.ts is the bar start, so the close becomes
    visible one minute later. "close" means bar.ts is already that moment.
    """
    if timestamp_kind not in ("open", "close"):
        raise ValueError("timestamp_kind must be 'open' or 'close'")
    rows: list[Observation] = []
    for bar in bars:
        knowable = bar.ts + timedelta(minutes=1) if timestamp_kind == "open" else bar.ts
        rows.append(
            Observation(
                factor=factor,
                value=bar.close,
                observed_at=knowable,
                available_at=knowable,
                ingested_at=knowable,
                source=source,
                symbol=symbol,
                quality=quality,
            )
        )
    return rows


def decide_at(
    spec: HypothesisSpec,
    event: EventRecord,
    resolver: FactorResolver,
    now: datetime,
) -> DecisionRecord:
    """Replay clock entry. The decision itself is HypothesisEngine.evaluate."""
    from gold_signal.runtime.context import ReplayEvaluationContext
    from gold_signal.runtime.engine import HypothesisEngine

    context = ReplayEvaluationContext(tuple(resolver.observations), now)
    return HypothesisEngine().evaluate(spec, event, context)


def replay_event(
    spec: HypothesisSpec,
    event: EventRecord,
    observations: list[Observation],
    costs: CostModel | None = None,
) -> tuple[DecisionRecord, dict[str, TradeOutcome]]:
    """Decide when the reaction minute exists, then measure forward from that fill."""
    costs = costs or CostModel()
    ready_at = event.published_at + timedelta(minutes=1)
    if event.available_at > ready_at:
        ready_at = event.available_at
    later = [row.observed_at for row in observations if row.observed_at >= ready_at]
    decision_time = min(later) if later else ready_at
    decision = decide_at(spec, event, FactorResolver(observations), decision_time)
    resolver = FactorResolver(observations)
    outcomes: dict[str, TradeOutcome] = {}
    if decision.side in ("LONG", "SHORT") and decision.entry_price is not None and decision.entry_at is not None:
        for name, delta in HORIZONS:
            outcomes[name] = _outcome_at_horizon(resolver, spec.asset, decision, delta, costs, name)
    return decision, outcomes


def replay_report(
    spec: HypothesisSpec,
    start: str,
    end: str,
    events: list[EventRecord],
    observations: list[Observation],
    *,
    strategy_path: Path | None = None,
    costs: CostModel | None = None,
) -> dict:
    costs = costs or CostModel()
    missing = _coverage_gaps(spec, events, observations)
    report = _empty_report(spec, start, end, strategy_path, costs, missing)
    report["data_quality"] = quality_by_series(events, observations)
    if missing:
        _apply_grade(report, 0, 0, failed=True)
        return report
    start_at = _bound(start, end=False)
    end_at = _bound(end, end=True)
    selected = [
        event
        for event in events
        if start_at <= _utc(event.published_at) < end_at
    ]
    selected.sort(key=lambda event: (_utc(event.published_at), event.event_id))
    if spec.drive == "BAR":
        rows = _bar_rows(spec, observations, start_at, end_at, costs)
    else:
        rows = [_played(spec, event, observations, costs) for event in selected]
    _fill_report(report, rows, costs)
    resolver = FactorResolver(observations)
    bars = resolver.series(price_factor(spec.asset))
    closed: list[ClosedTrade] = []
    traces = []
    for event, decision, outcomes in rows:
        trade = simulate_trade(decision, spec.asset, bars, spec.exit, _fill_cost(spec, event, decision, observations, costs))
        if trade is not None:
            closed.append(trade)
        traces.append(_event_trace(spec, event, decision, outcomes, observations, trade))
    report["traces"] = traces
    report["portfolio"] = portfolio_result(closed)
    report["walk_forward"] = walk_forward(closed)
    report["windows"] = {
        name: _window_report(rows, window_start, window_end)
        for name, window_start, window_end in RESEARCH_WINDOWS
    }
    return report


def load_archive(path: Path) -> tuple[list[EventRecord], list[Observation]]:
    events: list[EventRecord] = []
    observations: list[Observation] = []
    if not path.exists():
        return events, observations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "event":
            events.append(
                EventRecord(
                    event_id=str(row["event_id"]),
                    event_type=str(row.get("event_type") or ""),
                    published_at=_parse_stamp(row["published_at"]),
                    available_at=_parse_stamp(row.get("available_at") or row["published_at"]),
                    ingested_at=_parse_optional_stamp(row.get("ingested_at")),
                    title=str(row.get("title") or ""),
                    content=str(row.get("content") or row.get("title") or ""),
                    source=str(row.get("source") or ""),
                    quality=str(row.get("quality") or "RECONSTRUCTED"),
                )
            )
        elif kind == "observation":
            observations.append(
                Observation(
                    factor=str(row["factor"]),
                    value=float(row["value"]),
                    observed_at=_parse_stamp(row["observed_at"]),
                    available_at=_parse_stamp(row["available_at"]),
                    ingested_at=_parse_optional_stamp(row.get("ingested_at")),
                    source=str(row.get("source") or ""),
                    symbol=row.get("symbol"),
                    quality=str(row.get("quality") or "RECONSTRUCTED"),
                )
            )
    return events, observations


def yaml_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_distribution(outcomes: list[TradeOutcome]) -> dict:
    nets = [row.net_return for row in outcomes if row.net_return is not None]
    grosses = [row.gross_return for row in outcomes if row.gross_return is not None]
    mfes = [row.mfe for row in outcomes if row.mfe is not None]
    maes = [row.mae for row in outcomes if row.mae is not None]
    if not nets:
        return {
            "status": "INSUFFICIENT",
            "samples": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "p25": None,
            "p75": None,
            "avg_mfe": None,
            "avg_mae": None,
            "profit_factor": None,
            "expectancy": None,
            "avg_win": None,
            "avg_loss": None,
            "gross_avg_return": None,
        }
    stats = _return_stats(nets)
    stats["status"] = "OK"
    stats["samples"] = len(nets)
    stats["avg_mfe"] = _mean(mfes)
    stats["avg_mae"] = _mean(maes)
    stats["gross_avg_return"] = _mean(grosses) if grosses else None
    return stats


def _event_trace(
    spec: HypothesisSpec,
    event: EventRecord,
    decision: DecisionRecord,
    outcomes: dict[str, TradeOutcome],
    observations: list[Observation],
    trade: ClosedTrade | None = None,
) -> dict:
    """What the clock could see for one event. A reaction names the two closes it used."""
    resolver = FactorResolver(observations)
    reactions = []
    for condition in spec.confirmations:
        if not (condition.factor.endswith(".reaction_1bar") or condition.factor.endswith(".reaction_1m")):
            continue
        if spec.drive == "BAR":
            bar = bar_step(
                resolver,
                price_factor(series_code(condition.factor)),
                decision.evaluated_at,
                decision.evaluated_at,
            )
        else:
            bar = reaction_1bar(
                resolver,
                price_factor(series_code(condition.factor)),
                event.published_at,
                decision.evaluated_at,
            )
        value = None if bar is None else bar.value
        reactions.append(
            {
                "factor": condition.factor,
                "value": value,
                "anchor_at": None if bar is None else bar.anchor_at.isoformat(),
                "later_at": None if bar is None else bar.later_at.isoformat(),
                "span_seconds": None if bar is None else int(bar.span.total_seconds()),
                "passed": _condition_passed(spec, event, condition, value),
            }
        )
    horizon_rows = {}
    for name, outcome in outcomes.items():
        horizon_rows[name] = {
            "net_return": outcome.net_return,
            "mfe": outcome.mfe,
            "mae": outcome.mae,
        }
    return {
        "event_id": event.event_id,
        "title": event.title,
        "published_at": event.published_at.isoformat(),
        "side": decision.side,
        "reasons": list(decision.reasons),
        "entry_at": None if decision.entry_at is None else decision.entry_at.isoformat(),
        "entry_price": decision.entry_price,
        "reactions": reactions,
        "horizons": horizon_rows,
        "exit_reason": None if trade is None else trade.exit_reason,
        "exit_at": None if trade is None else trade.exit_at.isoformat(),
        "exit_price": None if trade is None else trade.exit_price,
        "net_return": None if trade is None else trade.net_return,
    }


def _condition_passed(spec: HypothesisSpec, event: EventRecord, condition, value: float | None) -> bool:
    if value is None:
        return False
    direction, why = resolve_direction(spec, f"{event.title} {event.content}")
    if why or direction == 0:
        return False
    ok, gap = _check(condition, {condition.factor: value}, direction)
    return bool(ok) and gap is None


def _bar_rows(
    spec: HypothesisSpec,
    observations: list[Observation],
    start_at: datetime,
    end_at: datetime,
    costs: CostModel,
) -> list[tuple[EventRecord, DecisionRecord, dict[str, TradeOutcome]]]:
    from gold_signal.runtime.engine import HypothesisEngine

    resolver = FactorResolver(observations)
    series = [
        row
        for row in resolver.series(price_factor(spec.asset))
        if start_at <= row.observed_at < end_at
    ]
    engine = HypothesisEngine()
    rows = []
    for bar in series[1:]:
        decision = engine.evaluate_bar(spec, bar, resolver)
        outcomes: dict[str, TradeOutcome] = {}
        if decision.side in ("LONG", "SHORT") and decision.entry_price is not None and decision.entry_at is not None:
            for name, delta in HORIZONS:
                outcomes[name] = _outcome_at_horizon(resolver, spec.asset, decision, delta, costs, name)
        event = EventRecord(
            event_id=decision.event_id,
            event_type="",
            published_at=bar.observed_at,
            available_at=bar.available_at,
            ingested_at=bar.available_at,
            title=bar.observed_at.isoformat(),
            content="",
            source="bar",
        )
        rows.append((event, decision, outcomes))
    return rows


def _fill_cost(
    spec: HypothesisSpec,
    event: EventRecord,
    decision: DecisionRecord,
    observations: list[Observation],
    costs: CostModel,
) -> CostModel:
    resolver = FactorResolver(observations)
    if spec.drive == "BAR":
        step = bar_step(resolver, price_factor(spec.asset), decision.evaluated_at, decision.evaluated_at)
    else:
        step = reaction_1bar(resolver, price_factor(spec.asset), event.published_at, decision.evaluated_at)
    move = None if step is None else step.value
    return execution_costs(None if spec.drive == "BAR" else event, move, costs)


def sweep_reaction_threshold(
    spec: HypothesisSpec,
    events: list[EventRecord],
    observations: list[Observation],
    thresholds: tuple[float, ...],
    start: str = "2024-01-01",
    end: str = "2026-09-30",
) -> list[dict]:
    """Rerun the same tape at each reaction threshold. The lifecycle, not a rescaled statistic."""
    from dataclasses import replace

    rows = []
    for value in thresholds:
        confirmations = tuple(
            replace(condition, value=value)
            if condition.factor.endswith(("reaction_1bar", "reaction_1m")) and condition.value is not None
            else condition
            for condition in spec.confirmations
        )
        report = replay_report(replace(spec, confirmations=confirmations), start, end, events, observations)
        rows.append(
            {
                "threshold": value,
                "trades": report["trades"],
                "avg_return": report.get("avg_return"),
                "evidence_status": report.get("evidence_status"),
                "end_equity": (report.get("portfolio") or {}).get("end_equity"),
            }
        )
    return rows


def compare_reports(left: dict, right: dict) -> dict:
    def pack(report: dict) -> dict:
        net = report.get("net") or {}
        portfolio = report.get("portfolio") or {}
        return {
            "evidence_status": report.get("evidence_status"),
            "trades": report.get("trades"),
            "avg_return": net.get("avg_return"),
            "end_equity": portfolio.get("end_equity"),
        }

    return {"left": pack(left), "right": pack(right)}


def _played(
    spec: HypothesisSpec,
    event: EventRecord,
    observations: list[Observation],
    costs: CostModel,
) -> tuple[EventRecord, DecisionRecord, dict[str, TradeOutcome]]:
    decision, outcomes = replay_event(spec, event, observations, costs)
    return event, decision, outcomes


def _outcome_at_horizon(
    resolver: FactorResolver,
    asset: str,
    decision: DecisionRecord,
    delta: timedelta,
    costs: CostModel,
    horizon: str,
) -> TradeOutcome:
    assert decision.entry_at is not None and decision.entry_price is not None
    exit_at = decision.entry_at + delta
    series = price_factor(asset)
    bars = resolver.series(series)
    start = bisect_left(bars, exit_at, key=lambda row: row.observed_at)
    if start >= len(bars):
        return measure_outcome(
            side=decision.side,
            entry_price=decision.entry_price,
            entry_at=decision.entry_at,
            path=[],
            exit_price=None,
            costs=costs,
            horizon=horizon,
        )
    last = bars[start]
    path = [
        (row.observed_at, row.value)
        for row in bars
        if decision.entry_at < row.observed_at <= last.observed_at
    ]
    return measure_outcome(
        side=decision.side,
        entry_price=decision.entry_price,
        entry_at=decision.entry_at,
        path=path,
        exit_price=last.value,
        costs=costs,
        horizon=horizon,
    )


def _coverage_gaps(
    spec: HypothesisSpec,
    events: list[EventRecord],
    observations: list[Observation],
) -> list[str]:
    missing: list[str] = []
    if spec.drive != "BAR" and not events:
        missing.append(f"{spec.asset}: no point-in-time news archive")
    present = {row.factor for row in observations}
    if price_factor(spec.asset) not in present:
        missing.append(f"{spec.asset}: no stored prices aligned to past headlines")
    for condition in spec.confirmations:
        code = series_code(condition.factor)
        if code == "DFII10" or condition.factor.startswith("DFII10"):
            missing.append("DFII10 is a daily FRED print and is not safe to use before its observation date")
            continue
        if price_factor(code) not in present:
            missing.append(f"{condition.factor}: no historical series aligned to headlines")
    return missing


def _empty_report(
    spec: HypothesisSpec,
    start: str,
    end: str,
    strategy_path: Path | None,
    costs: CostModel,
    missing: list[str],
) -> dict:
    net = edge_distribution([])
    return {
        "hypothesis": spec.id,
        "yaml_sha256": yaml_sha256(strategy_path),
        "classifier_version": CLASSIFIER_VERSION,
        "start": start,
        "end": end,
        "run_status": "FAILED" if missing else "COMPLETED",
        "evidence_status": "NO_DATA",
        "status": "NO_DATA",
        "events": 0,
        "trades": 0,
        "samples": 0,
        "counts": {"events": 0, "waiting": 0, "flat": 0, "long": 0, "short": 0},
        "horizon": HEADLINE_HORIZON,
        "costs": {
            "spread_cost": costs.spread_cost,
            "slippage_cost": costs.slippage_cost,
            "commission": costs.commission,
        },
        "net": net,
        "by_horizon": {},
        "windows": {
            name: {
                "start": window_start,
                "end": window_end,
                "run_status": "FAILED" if missing else "COMPLETED",
                "evidence_status": "NO_DATA",
                "status": "NO_DATA",
                "samples": 0,
                "expectancy": None,
                "median_return": None,
                "profit_factor": None,
                "avg_return": None,
            }
            for name, window_start, window_end in RESEARCH_WINDOWS
        },
        "missing": missing,
        "traces": [],
        "portfolio": {"start_equity": 1.0, "end_equity": 1.0, "executed": 0, "skipped": 0, "curve": [{"at": None, "equity": 1.0}]},
        "walk_forward": walk_forward([]),
        "win_rate": None,
        "avg_return": None,
    }


def _fill_report(
    report: dict,
    rows: list[tuple[EventRecord, DecisionRecord, dict[str, TradeOutcome]]],
    costs: CostModel,
) -> None:
    counts = {"events": len(rows), "waiting": 0, "flat": 0, "long": 0, "short": 0}
    for _event, decision, _outcomes in rows:
        key = decision.side.lower()
        if key in counts:
            counts[key] += 1
    report["counts"] = counts
    by_horizon = {
        name: edge_distribution([outcomes[name] for _event, decision, outcomes in rows if name in outcomes and decision.side in ("LONG", "SHORT")])
        for name, _delta in HORIZONS
    }
    report["by_horizon"] = by_horizon
    headline = by_horizon[HEADLINE_HORIZON]
    report["net"] = headline
    report["samples"] = headline["samples"]
    report["win_rate"] = headline["win_rate"]
    report["avg_return"] = headline["avg_return"]
    trades = counts["long"] + counts["short"]
    _apply_grade(report, counts["events"], trades, failed=False)
    report["costs"] = {
        "spread_cost": costs.spread_cost,
        "slippage_cost": costs.slippage_cost,
        "commission": costs.commission,
    }


def _window_report(
    rows: list[tuple[EventRecord, DecisionRecord, dict[str, TradeOutcome]]],
    start: str,
    end: str,
) -> dict:
    start_at = _bound(start, end=False)
    end_at = _bound(end, end=True)
    chosen = [row for row in rows if start_at <= _utc(row[0].published_at) < end_at]
    counts = {"events": len(chosen), "waiting": 0, "flat": 0, "long": 0, "short": 0}
    for _event, decision, _outcomes in chosen:
        key = decision.side.lower()
        if key in counts:
            counts[key] += 1
    headline = edge_distribution(
        [
            outcomes[HEADLINE_HORIZON]
            for _event, decision, outcomes in chosen
            if decision.side in ("LONG", "SHORT") and HEADLINE_HORIZON in outcomes
        ]
    )
    trades = counts["long"] + counts["short"]
    grade = evidence_grade(counts["events"], trades)
    return {
        "start": start,
        "end": end,
        "run_status": "COMPLETED",
        "evidence_status": grade,
        "status": grade,
        "samples": 0 if not chosen else headline["samples"],
        "counts": counts,
        "expectancy": None if not chosen else headline["expectancy"],
        "median_return": None if not chosen else headline["median_return"],
        "profit_factor": None if not chosen else headline["profit_factor"],
        "avg_return": None if not chosen else headline["avg_return"],
    }


def _return_stats(values: list[float]) -> dict:
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    count = len(values)
    avg_win = _mean(wins) if wins else 0.0
    avg_loss = abs(_mean(losses)) if losses else 0.0
    profit_factor = None if not losses else (sum(wins) / abs(sum(losses)))
    return {
        "win_rate": len(wins) / count,
        "avg_return": _mean(values),
        "median_return": statistics.median(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "profit_factor": profit_factor,
        "expectancy": (len(wins) / count) * avg_win - (len(losses) / count) * avg_loss,
        "avg_win": None if not wins else avg_win,
        "avg_loss": None if not losses else avg_loss,
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _apply_grade(report: dict, events: int, trades: int, *, failed: bool) -> None:
    report["events"] = events
    report["trades"] = trades
    report["run_status"] = "FAILED" if failed else "COMPLETED"
    report["evidence_status"] = "NO_DATA" if failed else evidence_grade(events, trades)
    report["status"] = report["evidence_status"]


def _series_label(row: Observation) -> str:
    if row.symbol:
        return row.symbol
    parts = row.factor.split(".")
    if len(parts) >= 2 and parts[0] == "market":
        return parts[1]
    return parts[0]


def _collapse_quality(grades: set[str]) -> str:
    if "RECONSTRUCTED" in grades or not grades:
        return "RECONSTRUCTED"
    if grades == {"VERIFIED"}:
        return "VERIFIED"
    return "RECONSTRUCTED"


def _event_text(event: EventRecord) -> str:
    title = event.title.strip()
    content = event.content.strip()
    if title and content and title not in content:
        return f"{title} {content}"
    return content or title


def _bound(value: str, *, end: bool) -> datetime:
    day = datetime.fromisoformat(value[:10]).replace(tzinfo=timezone.utc)
    if end:
        return day + timedelta(days=1)
    return day


def _utc(stamp: datetime) -> datetime:
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _parse_stamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    return _utc(stamp)


def _parse_optional_stamp(value: object) -> datetime | None:
    if not value:
        return None
    return _parse_stamp(str(value))
