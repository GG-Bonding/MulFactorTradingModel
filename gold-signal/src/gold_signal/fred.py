from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from gold_signal.models import YieldPoint

DFII10_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
SOURCE = "FRED DFII10"


def fetch_us_real_yield_10y() -> YieldPoint | None:
    """Latest published US 10Y real yield. A missing print stays missing."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx.get(
                DFII10_CSV,
                headers={"User-Agent": "Mozilla/5.0 gold-signal/0.2"},
                timeout=25.0,
                follow_redirects=True,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"FRED HTTP {response.status_code}")
            return parse_fred_csv(response.text)
        except (httpx.HTTPError, RuntimeError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"FRED request failed: {last}") from last


def parse_fred_csv(text: str) -> YieldPoint | None:
    rows = _observations(text)
    if not rows:
        return None
    day, value = rows[-1]
    return YieldPoint(
        code="DFII10",
        name="US 10Y real yield",
        yield_pct=value,
        ts=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
        source=SOURCE,
    )


def _observations(text: str) -> list[tuple[datetime, float]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    out: list[tuple[datetime, float]] = []
    for line in lines[1:]:
        date_text, raw, *_rest = (line.split(",") + [""])[:2]
        if not raw or raw == ".":
            continue
        try:
            day = datetime.strptime(date_text, "%Y-%m-%d")
            value = float(raw)
        except ValueError:
            continue
        out.append((day, value))
    return out


def as_record(point: YieldPoint | None) -> dict[str, Any]:
    if point is None:
        return {"code": "DFII10", "value": None, "source": SOURCE, "missing": "no published observation"}
    return {
        "code": point.code,
        "name": point.name,
        "value": point.yield_pct,
        "timestamp": None if point.ts is None else point.ts.date().isoformat(),
        "source": point.source,
        "point_in_time_safe": False,
        "note": "Daily print. Do not use it before its observation date.",
    }
