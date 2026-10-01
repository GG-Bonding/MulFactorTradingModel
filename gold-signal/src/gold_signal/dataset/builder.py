"""Turn raw headlines and bars into one dataset. Duplicate wires do not become extra samples."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from gold_signal.dataset.aligner import align_groups
from gold_signal.dataset.cluster import RawEvent, cluster_events
from gold_signal.dataset.manifest import write_manifest
from gold_signal.dataset.validator import validate_samples
from gold_signal.observation import Observation


def build_fetched_dataset(
    destination: Path,
    flash_payload: dict,
    minute_body: str,
    *,
    name: str,
    start: datetime,
    end: datetime,
    factor: str = "market.XAUUSD.close",
    symbol: str = "XAUUSD",
) -> dict:
    """Parse one flash page and one minute payload, then cluster and align."""
    from gold_signal.dataset.event_fetcher import parse_jin10_flash
    from gold_signal.dataset.market_fetcher import parse_sina_minutes

    return build_dataset(
        destination,
        parse_jin10_flash(flash_payload),
        parse_sina_minutes(minute_body, factor, symbol),
        name=name,
        start=start,
        end=end,
        factor=factor,
    )


def build_dataset(
    destination: Path,
    events: list[RawEvent],
    observations: list[Observation],
    *,
    name: str,
    start: datetime,
    end: datetime,
    factor: str = "market.XAUUSD.close",
) -> dict:
    window = [event for event in events if start <= event.published_at < end]
    groups = cluster_events(window)
    aligned, rejected = align_groups(groups, observations, factor)
    errors = validate_samples(aligned)
    if errors:
        raise ValueError("; ".join(errors))
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "events.jsonl", [_event_row(sample) for sample in aligned])
    _write_jsonl(
        destination / "market.jsonl",
        [_observation_row(row) for row in observations if row.factor == factor and start <= row.observed_at < end],
    )
    dropped = sum(len(group.member_ids) - 1 for group in groups)
    write_manifest(
        destination / "manifest.yaml",
        {
            "dataset": name,
            "quality": "RECONSTRUCTED",
            "cooldown": "30m",
            "samples": str(len(aligned)),
            "groups": str(len(groups)),
            "dropped_duplicates": str(dropped),
            "rejected": str(len(rejected)),
            "bar_timestamp": "close",
        },
    )
    return {
        "samples": len(aligned),
        "groups": len(groups),
        "dropped_duplicates": dropped,
        "rejected": len(rejected),
    }


def _event_row(sample) -> dict:
    event = sample.group.event
    return {
        "kind": "event",
        "event_id": event.event_id,
        "event_group_id": sample.group.event_group_id,
        "event_type": sample.group.event_type,
        "member_ids": list(sample.group.member_ids),
        "published_at": event.published_at.isoformat(),
        "available_at": event.published_at.isoformat(),
        "ingested_at": event.published_at.isoformat(),
        "title": event.title,
        "content": event.content,
        "source": event.source,
    }


def _observation_row(row: Observation) -> dict:
    return {
        "kind": "observation",
        "factor": row.factor,
        "value": row.value,
        "observed_at": row.observed_at.isoformat(),
        "available_at": row.available_at.isoformat(),
        "ingested_at": row.ingested_at.isoformat() if row.ingested_at else row.observed_at.isoformat(),
        "source": row.source,
        "symbol": row.symbol,
        "quality": row.quality,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
