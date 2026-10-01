"""Repeated headlines about one incident stay one sample."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from gold_signal.hypothesis import matching_triggers

COOLDOWN = timedelta(minutes=30)

_FOMC = re.compile(r"(FOMC|美联储议息|联邦基金利率)", re.IGNORECASE)


@dataclass(frozen=True)
class RawEvent:
    event_id: str
    published_at: datetime
    title: str
    content: str
    source: str


@dataclass(frozen=True)
class ClusteredEvent:
    """The earliest headline is the sample. Later wires in the cooldown are members."""

    event: RawEvent
    event_type: str
    event_group_id: str
    member_ids: tuple[str, ...]


def classify_event(event: RawEvent) -> str:
    text = f"{event.title} {event.content}"
    if _FOMC.search(text):
        return "FOMC"
    triggers = matching_triggers(text)
    return triggers[0] if triggers else ""


def cluster_events(events: list[RawEvent]) -> list[ClusteredEvent]:
    ordered = sorted(events, key=lambda item: (item.published_at, item.event_id))
    groups: list[ClusteredEvent] = []
    for event in ordered:
        event_type = classify_event(event)
        if not event_type:
            if groups and event.published_at - groups[-1].event.published_at <= COOLDOWN:
                matched = groups[-1]
                groups[-1] = ClusteredEvent(
                    event=matched.event,
                    event_type=matched.event_type,
                    event_group_id=matched.event_group_id,
                    member_ids=matched.member_ids + (event.event_id,),
                )
            continue
        matched = _open_group(groups, event_type, event.published_at)
        if matched is None:
            groups.append(
                ClusteredEvent(
                    event=event,
                    event_type=event_type,
                    event_group_id=_group_id(event_type, event.published_at),
                    member_ids=(event.event_id,),
                )
            )
            continue
        index = groups.index(matched)
        groups[index] = ClusteredEvent(
            event=matched.event,
            event_type=matched.event_type,
            event_group_id=matched.event_group_id,
            member_ids=matched.member_ids + (event.event_id,),
        )
    return groups


def _open_group(groups: list[ClusteredEvent], event_type: str, published_at: datetime) -> ClusteredEvent | None:
    for group in reversed(groups):
        if group.event_type != event_type:
            continue
        if published_at - group.event.published_at <= COOLDOWN:
            return group
        return None
    return None


def _group_id(event_type: str, published_at: datetime) -> str:
    stamp = published_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{event_type.lower()}_{stamp}"
