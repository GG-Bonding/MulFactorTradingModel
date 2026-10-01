"""A dataset sample is one clustered incident with a one-minute bar reaction."""

from __future__ import annotations

from datetime import timedelta

from gold_signal.dataset.aligner import AlignedSample


def validate_samples(samples: list[AlignedSample]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        group_id = sample.group.event_group_id
        if group_id in seen:
            errors.append(f"duplicate group {group_id}")
        seen.add(group_id)
        if sample.reaction.span != timedelta(minutes=1):
            errors.append(f"{group_id} reaction is not one bar")
        if not sample.group.member_ids:
            errors.append(f"{group_id} has no headlines")
    return errors
