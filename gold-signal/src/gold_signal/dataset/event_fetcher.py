"""Normalize a wire item into one raw headline. Network fetchers stay outside tests."""

from __future__ import annotations

import re
from datetime import datetime

from gold_signal.dataset.cluster import RawEvent


def raw_event(event_id: str, published_at: datetime, title: str, content: str, source: str) -> RawEvent:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", content or title or "")).strip()
    headline = title.strip() or text[:80]
    return RawEvent(event_id, published_at, headline, text or headline, source)
