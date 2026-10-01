"""Keep a clustered headline only when the next completed bar exists."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from gold_signal.dataset.cluster import ClusteredEvent
from gold_signal.observation import FactorResolver, Observation
from gold_signal.replay_clock import BarReaction, reaction_1bar


@dataclass(frozen=True)
class AlignedSample:
    group: ClusteredEvent
    reaction: BarReaction


@dataclass(frozen=True)
class RejectedSample:
    group: ClusteredEvent
    reason: str


def align_groups(
    groups: list[ClusteredEvent],
    observations: list[Observation],
    factor: str = "market.XAUUSD.close",
) -> tuple[list[AlignedSample], list[RejectedSample]]:
    resolver = FactorResolver(observations)
    kept: list[AlignedSample] = []
    rejected: list[RejectedSample] = []
    for group in groups:
        published = group.event.published_at
        reaction = reaction_1bar(resolver, factor, published, published + timedelta(minutes=3))
        if reaction is None or reaction.span != timedelta(minutes=1):
            rejected.append(RejectedSample(group, "no one-bar price window"))
            continue
        kept.append(AlignedSample(group, reaction))
    return kept, rejected
