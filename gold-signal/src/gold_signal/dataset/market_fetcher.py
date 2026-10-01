"""Market rows are supplied by a fetcher. Tests pass bars in directly."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from gold_signal.observation import Observation

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def price_rows(observations: list[Observation], factor: str) -> list[Observation]:
    return [row for row in observations if row.factor == factor]


def parse_sina_minutes(body: str, factor: str, symbol: str) -> list[Observation]:
    """Sina minute JSONP. `d` is the bar open in Shanghai; the close is knowable one minute later."""
    start, end = body.find("(["), body.rfind("])")
    if start < 0 or end < 0:
        return []
    rows = json.loads(body[start + 1 : end + 1])
    observations: list[Observation] = []
    for row in rows:
        close = row.get("c")
        label = row.get("d")
        if close in (None, "") or not label:
            continue
        opened = datetime.strptime(label, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_SHANGHAI).astimezone(timezone.utc)
        knowable = opened.replace(second=0, microsecond=0) + timedelta(minutes=1)
        observations.append(
            Observation(factor, float(close), knowable, knowable, knowable, "sina", symbol, "RECONSTRUCTED")
        )
    return observations
