"""Market rows are supplied by a fetcher. Tests pass bars in directly."""

from __future__ import annotations

from gold_signal.observation import Observation


def price_rows(observations: list[Observation], factor: str) -> list[Observation]:
    return [row for row in observations if row.factor == factor]
