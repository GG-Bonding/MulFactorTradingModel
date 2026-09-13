from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Thresholds:
    """All V0 numeric rules live here. Do not scatter magic numbers."""

    Z_OBVIOUS = 1.5
    RETURN_1M = 0.0008
    RETURN_3M = 0.0012
    BUY_SCORE = 5
    SELL_SCORE = -5
    NEWS_MAX_AGE_SEC = 600
    KLINE_MINUTES = 30
    POLL_SECONDS = 8
    MAX_SCORE = 7


class SignalSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class NewsImpact(BaseModel):
    direction: int
    importance: int
    confidence: float
    reason: str

    @property
    def label(self) -> str:
        if self.direction > 0:
            return "BULLISH GOLD"
        if self.direction < 0:
            return "BEARISH GOLD"
        return "UNCERTAIN"


class FlashNews(BaseModel):
    event_id: str
    title: str
    content: str
    published_at: datetime
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        title = self.title.strip()
        content = self.content.strip()
        if title and content and title not in content:
            return f"{title} {content}"
        return content or title


class Bar(BaseModel):
    ts: datetime
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None


class Quote(BaseModel):
    code: str
    price: float
    ts: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class AssetMove(BaseModel):
    code: str
    price: float
    return_1m: float
    return_3m: float
    return_5m: float
    std_1m: float
    z_1m: float
    obvious_1m: int
    obvious_3m: int


class MarketSnapshot(BaseModel):
    xau: AssetMove
    xag: AssetMove
    eurusd: AssetMove
    as_of: datetime


class ScoreBreakdown(BaseModel):
    news: int = 0
    gold_1m: int = 0
    gold_3m: int = 0
    silver: int = 0
    eurusd: int = 0
    total: int = 0
    notes: list[str] = Field(default_factory=list)


class SignalResult(BaseModel):
    timestamp: datetime
    signal: SignalSide
    score: int
    confidence: float
    news: str
    news_direction: int
    news_impact_label: str
    event_id: str | None = None
    xau_price: float
    xau_1m: float
    xau_3m: float
    xag_1m: float
    eurusd_1m: float
    breakdown: ScoreBreakdown
    reason: str
    is_primary: bool = True
    rejected_by_market: str | None = None
    return_1m: float | None = None
    return_5m: float | None = None
    return_15m: float | None = None
    return_30m: float | None = None
    entry: float | None = None
