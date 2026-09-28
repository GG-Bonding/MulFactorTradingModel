"""A decision is not a trade. It becomes an order, a fill, a position, then an exit."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta

from gold_signal.observation import EventRecord, Observation
from gold_signal.replay_clock import CostModel, DecisionRecord, TradeOutcome, measure_outcome, signed_return


@dataclass(frozen=True)
class ExitPolicy:
    """How a filled position leaves. max_hold is the research horizon when nothing else hits."""

    max_hold: timedelta = timedelta(minutes=5)
    take_profit: float | None = None
    stop_loss: float | None = None


@dataclass(frozen=True)
class OrderIntent:
    event_id: str
    asset: str
    side: str
    created_at: datetime
    reason: str


@dataclass(frozen=True)
class Fill:
    event_id: str
    side: str
    price: float
    filled_at: datetime


@dataclass(frozen=True)
class Position:
    event_id: str
    asset: str
    side: str
    quantity: float
    entry_price: float
    entry_at: datetime
    policy: ExitPolicy


@dataclass(frozen=True)
class ClosedTrade:
    event_id: str
    asset: str
    side: str
    entry_price: float
    entry_at: datetime
    exit_price: float
    exit_at: datetime
    exit_reason: str
    net_return: float | None
    outcome: TradeOutcome


def execution_costs(event: EventRecord | None, move: float | None, base: CostModel | None = None) -> CostModel:
    """Macro releases pay a wider spread. A large bar adds a volatility slip, capped at 1%."""
    base = base or CostModel()
    macro = event is not None and (
        event.event_type.startswith(("NFP_", "CPI_"))
        or "非农" in f"{event.title} {event.content}"
        or "CPI" in f"{event.title} {event.content}".upper()
    )
    spread = base.spread_cost * (4 if macro else 1)
    slip = base.slippage_cost * (3 if macro else 1)
    if move is not None:
        slip += min(0.01, abs(move) * 0.25)
    return CostModel(spread_cost=spread, slippage_cost=slip, commission=base.commission)


def walk_forward(trades: list[ClosedTrade], folds: int = 3) -> list[dict]:
    """Contiguous folds in entry order. Later folds are not used to size the earlier ones."""
    ordered = sorted((trade for trade in trades if trade.net_return is not None), key=lambda trade: (trade.entry_at, trade.event_id))
    if not ordered:
        return [{"fold": index, "trades": 0, "skipped": 0, "end_equity": None} for index in range(1, folds + 1)]
    size = max(1, len(ordered) // folds)
    chunks: list[list[ClosedTrade]] = [ordered[index:index + size] for index in range(0, len(ordered), size)]
    while len(chunks) > folds:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    rows = []
    for index, chunk in enumerate(chunks, start=1):
        result = portfolio_result(chunk)
        rows.append(
            {
                "fold": index,
                "trades": result["executed"],
                "skipped": result["skipped"],
                "end_equity": result["end_equity"],
            }
        )
    while len(rows) < folds:
        rows.append({"fold": len(rows) + 1, "trades": 0, "skipped": 0, "end_equity": None})
    return rows


def order_from_decision(decision: DecisionRecord, asset: str) -> OrderIntent | None:
    if decision.side not in ("LONG", "SHORT") or decision.entry_at is None:
        return None
    return OrderIntent(
        event_id=decision.event_id,
        asset=asset,
        side=decision.side,
        created_at=decision.entry_at,
        reason=decision.reasons[0] if decision.reasons else "",
    )


def simulate_trade(
    decision: DecisionRecord,
    asset: str,
    bars: tuple[Observation, ...] | list[Observation],
    policy: ExitPolicy,
    costs: CostModel | None = None,
) -> ClosedTrade | None:
    """Fill at the decision print, then leave on stop, target, or the hold limit."""
    costs = costs or CostModel()
    intent = order_from_decision(decision, asset)
    if intent is None or decision.entry_price is None or decision.entry_at is None:
        return None
    fill = Fill(intent.event_id, intent.side, decision.entry_price, decision.entry_at)
    position = Position(
        event_id=intent.event_id,
        asset=asset,
        side=intent.side,
        quantity=1.0,
        entry_price=fill.price,
        entry_at=fill.filled_at,
        policy=policy,
    )
    ordered = tuple(sorted(bars, key=lambda row: row.observed_at))
    start = bisect_left(ordered, position.entry_at, key=lambda row: row.observed_at)
    deadline = position.entry_at + position.policy.max_hold
    exit_price: float | None = None
    exit_at: datetime | None = None
    reason: str | None = None
    path: list[tuple[datetime, float]] = []
    for row in ordered[start:]:
        if row.observed_at <= position.entry_at:
            continue
        path.append((row.observed_at, row.value))
        signed = signed_return(position.side, position.entry_price, row.value)
        if position.policy.take_profit is not None and signed >= position.policy.take_profit:
            exit_price, exit_at, reason = row.value, row.observed_at, "TAKE_PROFIT"
            break
        if position.policy.stop_loss is not None and signed <= -abs(position.policy.stop_loss):
            exit_price, exit_at, reason = row.value, row.observed_at, "STOP_LOSS"
            break
        if row.observed_at >= deadline:
            exit_price, exit_at, reason = row.value, row.observed_at, "MAX_HOLD"
            break
    if exit_price is None or exit_at is None or reason is None:
        return None
    outcome = measure_outcome(
        side=position.side,
        entry_price=position.entry_price,
        entry_at=position.entry_at,
        path=path,
        exit_price=exit_price,
        costs=costs,
        horizon=reason,
    )
    return ClosedTrade(
        event_id=intent.event_id,
        asset=asset,
        side=position.side,
        entry_price=position.entry_price,
        entry_at=position.entry_at,
        exit_price=exit_price,
        exit_at=exit_at,
        exit_reason=reason,
        net_return=outcome.net_return,
        outcome=outcome,
    )


def portfolio_result(trades: list[ClosedTrade]) -> dict:
    """One unit of equity. A new signal while the last position is open does not stack."""
    equity = 1.0
    open_until: datetime | None = None
    executed = 0
    skipped = 0
    curve = [{"at": None, "equity": 1.0}]
    ordered = sorted(trades, key=lambda trade: (trade.entry_at, trade.event_id))
    for trade in ordered:
        if trade.net_return is None:
            continue
        if open_until is not None and trade.entry_at < open_until:
            skipped += 1
            continue
        equity *= 1.0 + trade.net_return
        open_until = trade.exit_at
        executed += 1
        curve.append({"at": trade.exit_at.isoformat(), "equity": equity})
    return {
        "start_equity": 1.0,
        "end_equity": equity,
        "executed": executed,
        "skipped": skipped,
        "curve": curve,
    }
