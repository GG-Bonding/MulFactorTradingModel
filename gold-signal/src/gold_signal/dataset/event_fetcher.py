"""Normalize a wire item into one raw headline. Network fetchers stay outside tests."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from gold_signal.dataset.cluster import RawEvent

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def raw_event(event_id: str, published_at: datetime, title: str, content: str, source: str) -> RawEvent:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", content or title or "")).strip()
    headline = title.strip() or text[:80]
    return RawEvent(event_id, published_at, headline, text or headline, source)


def parse_jin10_flash(payload: dict) -> list[RawEvent]:
    """Jin10 flash-list JSON. `time` is Shanghai local time."""
    events: list[RawEvent] = []
    for item in payload.get("data") or []:
        raw_time = item.get("time")
        if not raw_time:
            continue
        published = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_SHANGHAI).astimezone(timezone.utc)
        body = (item.get("data") or {}).get("content") or ""
        title = (item.get("data") or {}).get("title") or ""
        events.append(raw_event(str(item.get("id") or raw_time), published, str(title or ""), str(body), "jin10"))
    return events
