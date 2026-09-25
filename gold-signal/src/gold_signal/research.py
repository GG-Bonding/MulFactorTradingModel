from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gold_signal.hypothesis import HypothesisSpec, _check, resolve_direction
from gold_signal.models import Bar
from gold_signal.news import CLASSIFIER_VERSION
from gold_signal.observation import EventRecord, FactorResolver, Observation
from gold_signal.replay_clock import (
    CostModel,
    DecisionRecord,
    TradeOutcome,
    fill_price,
    measure_outcome,
    reaction_1m,
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


def price_factor(code: str) -> str:
    return f"market.{code}.close"


def series_code(factor: str) -> str:
    return factor.split(".", 1)[0]


def confirmation_value(
    factor: str,
    resolver: FactorResolver,
    published_at: datetime,
    now: datetime,
) -> float | None:
    """Only a print that is already knowable can confirm. Unsafe factors stay empty."""
    if factor.endswith(".reaction_1m"):
        return reaction_1m(resolver, price_factor(series_code(factor)), published_at, now)
    return None


def observations_from_bars(
    factor: str,
    bars: list[Bar],
    *,
    symbol: str,
    source: str,
    timestamp_kind: str,
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
            )
        )
    return rows


def decide_at(
    spec: HypothesisSpec,
    event: EventRecord,
    resolver: FactorResolver,
    now: datetime,
) -> DecisionRecord:
    """One decision. reaction_1m is invisible until that minute's print exists."""
    ready_at = event.published_at + timedelta(minutes=1)
    blank = dict(
        event_id=event.event_id,
        hypothesis_id=spec.id,
        evaluated_at=now,
        entry_at=None,
        entry_price=None,
    )
    if now < event.available_at or now < ready_at:
        return DecisionRecord(side="WAITING", reasons=("reaction minute has not arrived",), **blank)
    text = _event_text(event)
    direction, why = resolve_direction(spec, text)
    if why or direction == 0:
        return DecisionRecord(side="FLAT", reasons=(why or "news has no direction for this asset",), **blank)
    values: dict[str, float | None] = {}
    for condition in spec.confirmations:
        change = confirmation_value(condition.factor, resolver, event.published_at, now)
        if change is None:
            return DecisionRecord(side="WAITING", reasons=(f"{condition.factor} is waiting",), **blank)
        values[condition.factor] = change
    for condition in spec.confirmations:
        ok, gap = _check(condition, values, direction)
        if gap:
            return DecisionRecord(side="FLAT", reasons=(gap,), **blank)
        if not ok:
            return DecisionRecord(
                side="FLAT",
                reasons=(f"failed {condition.factor} {condition.operator}",),
                **blank,
            )
    if spec.entry in ("LONG", "SHORT"):
        side = spec.entry
    elif spec.entry == "FOLLOW_NEWS":
        side = "LONG" if direction > 0 else "SHORT"
    else:
        return DecisionRecord(side="FLAT", reasons=(f"unsupported entry {spec.entry}",), **blank)
    entry = resolver.price_at(price_factor(spec.asset), now, now)
    entry_at, entry_price = fill_price(
        side,
        None if entry is None else entry.value,
        None if entry is None else entry.observed_at,
        ready_at,
    )
    if entry_price is None or entry_at is None:
        return DecisionRecord(side="WAITING", reasons=("entry print is not knowable yet",), **blank)
    return DecisionRecord(
        side=side,
        reasons=(f"{len(spec.confirmations)}/{len(spec.confirmations)} confirmations passed",),
        entry_at=entry_at,
        entry_price=entry_price,
        event_id=event.event_id,
        hypothesis_id=spec.id,
        evaluated_at=now,
    )


def replay_event(
    spec: HypothesisSpec,
    event: EventRecord,
    observations: list[Observation],
    costs: CostModel | None = None,
) -> tuple[DecisionRecord, dict[str, TradeOutcome]]:
    """Decide when the reaction minute exists, then measure forward from that fill."""
    costs = costs or CostModel()
    resolver = FactorResolver(observations)
    ready_at = event.published_at + timedelta(minutes=1)
    decision_time = ready_at if ready_at >= event.available_at else event.available_at
    decision = decide_at(spec, event, resolver, decision_time)
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
    if missing:
        return report
    start_at = _bound(start, end=False)
    end_at = _bound(end, end=True)
    selected = [
        event
        for event in events
        if start_at <= _utc(event.published_at) < end_at
    ]
    selected.sort(key=lambda event: (_utc(event.published_at), event.event_id))
    rows = [_played(spec, event, observations, costs) for event in selected]
    _fill_report(report, rows, costs)
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
    last = resolver.price_at(series, exit_at, exit_at)
    if last is None or last.observed_at < exit_at:
        return measure_outcome(
            side=decision.side,
            entry_price=decision.entry_price,
            entry_at=decision.entry_at,
            path=[],
            exit_price=None,
            costs=costs,
            horizon=horizon,
        )
    path = [
        (row.observed_at, row.value)
        for row in resolver.visible(series, exit_at)
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
    if not events:
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
        "status": "INSUFFICIENT" if missing else "OK",
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
                "status": "INSUFFICIENT",
                "samples": 0,
                "expectancy": None,
                "median_return": None,
                "profit_factor": None,
            }
            for name, window_start, window_end in RESEARCH_WINDOWS
        },
        "missing": missing,
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
    report["status"] = "OK"
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
    if not chosen:
        return {
            "start": start,
            "end": end,
            "status": "INSUFFICIENT",
            "samples": 0,
            "counts": counts,
            "expectancy": None,
            "median_return": None,
            "profit_factor": None,
            "avg_return": None,
        }
    return {
        "start": start,
        "end": end,
        "status": "OK",
        "samples": headline["samples"],
        "counts": counts,
        "expectancy": headline["expectancy"],
        "median_return": headline["median_return"],
        "profit_factor": headline["profit_factor"],
        "avg_return": headline["avg_return"],
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
