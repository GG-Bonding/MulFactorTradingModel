from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    factor: str
    value: float
    observed_at: datetime
    available_at: datetime
    ingested_at: datetime | None
    source: str
    symbol: str | None = None


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    event_type: str
    published_at: datetime
    available_at: datetime
    ingested_at: datetime | None
    title: str
    content: str
    source: str


@dataclass(frozen=True)
class CoverageEntry:
    factor: str
    start: datetime
    end: datetime
    resolution: str
    point_in_time_safe: bool
    research_only: bool = False


@dataclass
class CoverageManifest:
    entries: dict[str, CoverageEntry] = field(default_factory=dict)

    def missing(self, factors: list[str], start: datetime, end: datetime) -> list[str]:
        gaps = []
        for factor in factors:
            entry = self.entries.get(factor)
            if entry is None:
                gaps.append(f"{factor}: not in coverage manifest")
                continue
            if entry.research_only or not entry.point_in_time_safe:
                gaps.append(f"{factor}: research_only or not point-in-time safe")
                continue
            if entry.start > start or entry.end < end:
                gaps.append(f"{factor}: coverage {entry.start.date()}..{entry.end.date()} does not contain the request")
        return gaps


class FactorResolver:
    """At time t, only observations with available_at <= t are visible."""

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = list(observations)

    def visible(self, factor: str, now: datetime) -> list[Observation]:
        rows = [
            row
            for row in self.observations
            if row.factor == factor and row.available_at <= now and row.observed_at <= now
        ]
        return sorted(rows, key=lambda row: row.observed_at)

    def price_at(self, factor: str, when: datetime, now: datetime) -> Observation | None:
        rows = [row for row in self.visible(factor, now) if row.observed_at <= when]
        if not rows:
            return None
        return rows[-1]


class Recorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, payload: dict) -> None:
        record = {"kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def record_live_tick(
    recorder: Recorder,
    market: object,
    news: object | None,
    *,
    event_type: str,
    classifier_version: str,
    ingested_at: datetime,
    source: str = "live",
    seen: set[str] | None = None,
) -> None:
    """Persist every close already on the snapshot, and the event body for later reclassification."""
    seen = seen if seen is not None else set()
    for asset in (market.xau, market.xag, market.eurusd):
        code = getattr(asset, "code", "")
        if not code or code == "FLAT":
            continue
        factor = f"market.{code}.close"
        if factor in seen:
            continue
        seen.add(factor)
        recorder.append(
            "observation",
            {
                "factor": factor,
                "value": asset.price,
                "observed_at": market.as_of,
                "available_at": market.as_of,
                "ingested_at": ingested_at,
                "source": source,
                "symbol": code,
            },
        )
    if news is None:
        return
    event_key = f"event:{news.event_id}"
    if event_key in seen:
        return
    seen.add(event_key)
    recorder.append(
        "event",
        {
            "event_id": news.event_id,
            "published_at": news.published_at,
            "available_at": news.published_at,
            "ingested_at": ingested_at,
            "title": news.title,
            "content": news.content,
            "event_type": event_type,
            "classifier_version": classifier_version,
            "source": "jin10",
        },
    )


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)
