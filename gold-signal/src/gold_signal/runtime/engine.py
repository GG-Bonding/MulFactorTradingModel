from __future__ import annotations

from datetime import timedelta

from gold_signal.hypothesis import HypothesisSpec, _check, resolve_direction
from gold_signal.observation import EventRecord
from gold_signal.replay_clock import DecisionRecord, fill_price
from gold_signal.research import confirmation_value, price_factor
from gold_signal.runtime.context import EvaluationContext


class HypothesisEngine:
    """One decision path. It only sees the context's clock and factor resolver."""

    def evaluate(
        self,
        hypothesis: HypothesisSpec,
        event: EventRecord,
        context: EvaluationContext,
    ) -> DecisionRecord:
        now = context.now
        resolver = context.resolver()
        ready_at = event.published_at + timedelta(minutes=1)
        blank = dict(
            event_id=event.event_id,
            hypothesis_id=hypothesis.id,
            evaluated_at=now,
            entry_at=None,
            entry_price=None,
        )
        if now < event.available_at or now < ready_at:
            return DecisionRecord(side="WAITING", reasons=("reaction minute has not arrived",), **blank)
        text = _event_text(event)
        direction, why = resolve_direction(hypothesis, text)
        if why or direction == 0:
            return DecisionRecord(side="FLAT", reasons=(why or "news has no direction for this asset",), **blank)
        values: dict[str, float | None] = {}
        for condition in hypothesis.confirmations:
            change = confirmation_value(condition.factor, resolver, event.published_at, now)
            if change is None:
                return DecisionRecord(side="WAITING", reasons=(f"{condition.factor} is waiting",), **blank)
            values[condition.factor] = change
        for condition in hypothesis.confirmations:
            ok, gap = _check(condition, values, direction)
            if gap:
                return DecisionRecord(side="FLAT", reasons=(gap,), **blank)
            if not ok:
                return DecisionRecord(
                    side="FLAT",
                    reasons=(f"failed {condition.factor} {condition.operator}",),
                    **blank,
                )
        if hypothesis.entry in ("LONG", "SHORT"):
            side = hypothesis.entry
        elif hypothesis.entry == "FOLLOW_NEWS":
            side = "LONG" if direction > 0 else "SHORT"
        else:
            return DecisionRecord(side="FLAT", reasons=(f"unsupported entry {hypothesis.entry}",), **blank)
        entry = resolver.price_at(price_factor(hypothesis.asset), now, now)
        entry_at, entry_price = fill_price(
            side,
            None if entry is None else entry.value,
            None if entry is None else entry.observed_at,
            ready_at,
        )
        if entry_price is None or entry_at is None:
            return DecisionRecord(side="WAITING", reasons=("entry print is not knowable yet",), **blank)
        passed = len(hypothesis.confirmations)
        return DecisionRecord(
            side=side,
            reasons=(f"{passed}/{passed} confirmations passed",),
            entry_at=entry_at,
            entry_price=entry_price,
            event_id=event.event_id,
            hypothesis_id=hypothesis.id,
            evaluated_at=now,
        )


def _event_text(event: EventRecord) -> str:
    title = event.title.strip()
    content = event.content.strip()
    if title and content and title not in content:
        return f"{title} {content}"
    return content or title
