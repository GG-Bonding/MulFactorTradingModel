from __future__ import annotations

import uuid
from datetime import datetime

from gold_signal.agent import order_from_decision
from gold_signal.domain.models import AgentStatus
from gold_signal.execution import paper_execution
from gold_signal.hypothesis import parse_hypothesis, trigger_matches
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import record_paper_trade, record_signal
from gold_signal.persistence.store import ProductStore
from gold_signal.replay_clock import CostModel, DecisionRecord
from gold_signal.research import price_factor
from gold_signal.runtime.context import LiveEvaluationContext
from gold_signal.runtime.engine import HypothesisEngine


class AgentRuntime:
    """Paper agents only. The hypothesis engine does not know this is live."""

    def __init__(self, engine: HypothesisEngine | None = None) -> None:
        self.engine = engine or HypothesisEngine()

    def tick(
        self,
        store: ProductStore,
        *,
        now: datetime,
        observations: list[Observation],
        event: EventRecord | None = None,
    ) -> dict:
        settled = self._settle(store, observations, now)
        signals = [] if event is None else self._evaluate(store, event, observations, now)
        return {"signals": signals, "settled": settled}

    def _evaluate(
        self,
        store: ProductStore,
        event: EventRecord,
        observations: list[Observation],
        now: datetime,
    ) -> list[dict]:
        created: list[dict] = []
        text = f"{event.title} {event.content}".strip()
        context = LiveEvaluationContext(tuple(observations), now)
        for agent in store.list_agents():
            if agent.status != AgentStatus.PAPER or agent.active_version_id is None:
                continue
            version = store.get_version(agent.active_version_id)
            if version is None:
                continue
            spec = parse_hypothesis(version.hypothesis_yaml)
            if spec.trigger and not trigger_matches(spec.trigger, text):
                continue
            if not _seen(store, agent.id, "EVENT_MATCHED", event.event_id):
                store.insert_activity(
                    uuid.uuid4().hex,
                    agent.id,
                    "EVENT_MATCHED",
                    {"event_id": event.event_id, "version_id": version.id},
                    now,
                )
            if _signaled(store, agent.id, version.id, event.event_id):
                continue
            decision = self.engine.evaluate(spec, event, context)
            if decision.side not in ("LONG", "SHORT") or decision.entry_price is None:
                continue
            signal = record_signal(
                store,
                agent.id,
                version.id,
                {
                    "event_id": event.event_id,
                    "side": decision.side,
                    "entry_price": decision.entry_price,
                    "reason": decision.reasons[0] if decision.reasons else "",
                    "evaluated_at": decision.evaluated_at.isoformat(),
                },
                now=now,
            )
            order = order_from_decision(decision, version.hypothesis_yaml)
            record_paper_trade(
                store,
                {
                    "agent_id": agent.id,
                    "version_id": version.id,
                    "signal_id": signal["id"],
                    "side": order["side"],
                    "entry_price": order["entry"],
                    "spec_sha256": order["spec_sha256"],
                    "net_return": None,
                },
                now=now,
            )
            created.append(signal)
        return created

    def _settle(self, store: ProductStore, observations: list[Observation], now: datetime) -> list[str]:
        settled: list[str] = []
        visible = [row for row in observations if row.observed_at <= now and row.available_at <= now]
        for trade in store.list_open_paper_trades():
            signal = store.get_signal(trade["signal_id"]) if trade.get("signal_id") else None
            if signal is None or trade.get("entry_price") is None:
                continue
            version = store.get_version(trade["version_id"])
            if version is None:
                continue
            spec = parse_hypothesis(version.hypothesis_yaml)
            opened = datetime.fromisoformat(signal["evaluated_at"])
            decision = DecisionRecord(
                event_id=signal["event_id"],
                hypothesis_id=spec.id,
                evaluated_at=opened,
                side=trade["side"],
                entry_at=opened,
                entry_price=float(trade["entry_price"]),
                reasons=(signal.get("reason") or "",),
            )
            series = [
                row
                for row in visible
                if row.factor == price_factor(spec.asset)
            ]
            closed = paper_execution(decision, spec.asset, series, spec.exit, CostModel())
            if closed is None or closed.net_return is None:
                continue
            store.update_paper_outcome(trade["id"], closed.net_return)
            store.insert_activity(
                uuid.uuid4().hex,
                trade["agent_id"],
                "OUTCOME_UPDATED",
                {
                    "trade_id": trade["id"],
                    "version_id": trade["version_id"],
                    "net_return": closed.net_return,
                    "exit_reason": closed.exit_reason,
                    "exit_price": closed.exit_price,
                },
                now,
            )
            settled.append(trade["id"])
        return settled


def _signaled(store: ProductStore, agent_id: str, version_id: str, event_id: str) -> bool:
    return any(
        row["version_id"] == version_id and row["event_id"] == event_id
        for row in store.list_signals(agent_id)
    )


def _seen(store: ProductStore, agent_id: str, kind: str, event_id: str) -> bool:
    return any(
        row["kind"] == kind and row["detail"].get("event_id") == event_id
        for row in store.list_activities(agent_id)
    )
