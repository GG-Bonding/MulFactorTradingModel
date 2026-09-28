from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from gold_signal.hypothesis import HypothesisSpec
from gold_signal.observation import EventRecord, Observation
from gold_signal.research import load_archive, price_factor, series_code

PACKAGED_ARCHIVE = Path(__file__).resolve().parents[2] / "examples" / "history" / "archive.jsonl"
PACKAGED_MANIFEST = PACKAGED_ARCHIVE.with_name("manifest.yaml")


class HistoricalArchive:
    """Point-in-time events and closes. Missing series stay missing."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(
        self,
        start: datetime,
        end: datetime,
        factors: list[str],
    ) -> tuple[list[EventRecord], list[Observation], list[str]]:
        events, observations = load_archive(self.path)
        if self.manifest().get("bar_timestamp") == "open":
            observations = [_close_knowable(row) for row in observations]
        events = [event for event in events if start <= event.published_at < end]
        observations = [
            row
            for row in observations
            if row.factor in factors and start <= row.observed_at < end
        ]
        present = {row.factor for row in observations}
        missing = [factor for factor in factors if factor not in present]
        if not events:
            missing.append("news.events: no point-in-time news archive")
        return events, observations, missing

    def manifest(self) -> dict:
        path = self.path.with_name("manifest.yaml")
        if path.exists():
            return _read_manifest(path)
        if self.path == PACKAGED_ARCHIVE and PACKAGED_MANIFEST.exists():
            return _read_manifest(PACKAGED_MANIFEST)
        return {"dataset": self.path.name, "quality": "RECONSTRUCTED"}


def default_archive() -> HistoricalArchive:
    configured = os.environ.get("AGENT_ARCHIVE")
    path = Path(configured) if configured else PACKAGED_ARCHIVE
    return HistoricalArchive(path)


def _close_knowable(row: Observation) -> Observation:
    """Yahoo stamps the minute open. The close can be used one minute later."""
    shift = timedelta(minutes=1)
    return replace(row, observed_at=row.observed_at + shift, available_at=row.available_at + shift)


def _read_manifest(path: Path) -> dict:
    """Read the small archive manifest. Nested maps are one level deep."""
    root: dict = {}
    current: dict | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        value = value.strip()
        if indent == 0:
            if value == "":
                root[key] = {}
                current = root[key]
            else:
                root[key] = value
                current = None
        elif current is not None:
            current[key] = value
    return root


def factors_for(spec: HypothesisSpec) -> list[str]:
    codes = {spec.asset}
    for condition in spec.confirmations:
        code = series_code(condition.factor)
        if code == "DFII10":
            continue
        codes.add(code)
    return [price_factor(code) for code in sorted(codes)]
