from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from gold_signal.hypothesis import HypothesisSpec
from gold_signal.observation import EventRecord, Observation
from gold_signal.research import load_archive, price_factor, series_code

PACKAGED_ARCHIVE = Path(__file__).resolve().parents[2] / "examples" / "history" / "archive.jsonl"


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


def default_archive() -> HistoricalArchive:
    configured = os.environ.get("AGENT_ARCHIVE")
    path = Path(configured) if configured else PACKAGED_ARCHIVE
    return HistoricalArchive(path)


def factors_for(spec: HypothesisSpec) -> list[str]:
    codes = {spec.asset}
    for condition in spec.confirmations:
        code = series_code(condition.factor)
        if code == "DFII10":
            continue
        codes.add(code)
    return [price_factor(code) for code in sorted(codes)]
