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
    # Europe credit/fragmentation — levels in basis points unless noted.
    OAT_BUND_YELLOW_BP = 80.0
    OAT_BUND_ORANGE_BP = 100.0
    OAT_BUND_RED_BP = 150.0
    ITALY_BUND_ORANGE_BP = 150.0
    ITALY_BUND_RED_BP = 250.0
    BANKS_LAGGING_1D = -0.003
    BANKS_STRESS_1D = -0.008
    EURGBP_STRESS_1D = -0.002
    POLICY_BUND_MOVE_BP = 5.0


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
            return "BULLISH"
        if self.direction < 0:
            return "BEARISH"
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


class EventReaction(BaseModel):
    """Price change measured from the news timestamp, not from a trailing window."""

    anchor_price: float | None = None
    anchor_ts: datetime | None = None
    anchor_gap_seconds: float | None = None
    return_15s: float | None = None
    return_30s: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_5m: float | None = None


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
    xau_bars: list[Bar] = Field(default_factory=list)
    xag_bars: list[Bar] = Field(default_factory=list)
    eurusd_bars: list[Bar] = Field(default_factory=list)


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
    strength: int = 0
    news: str
    news_direction: int
    news_impact_label: str
    event_id: str | None = None
    product: str = "XAUUSD"
    xau_price: float
    xau_1m: float
    xau_3m: float
    xag_1m: float
    eurusd_1m: float
    anchor_price: float | None = None
    anchor_ts: datetime | None = None
    reaction_15s: float | None = None
    reaction_30s: float | None = None
    reaction_1m: float | None = None
    reaction_3m: float | None = None
    reaction_5m: float | None = None
    breakdown: ScoreBreakdown
    reason: str
    is_primary: bool = True
    rejected_by_market: str | None = None
    return_1m: float | None = None
    return_5m: float | None = None
    return_15m: float | None = None
    return_30m: float | None = None
    entry: float | None = None


class YieldPoint(BaseModel):
    code: str
    name: str
    yield_pct: float
    change_bp: float | None = None
    ts: datetime | None = None
    source: str


class EuropeSnapshot(BaseModel):
    as_of: datetime
    oat: YieldPoint | None = None
    bund: YieldPoint | None = None
    italy: YieldPoint | None = None
    gilt: YieldPoint | None = None
    oat_bund_bp: float | None = None
    italy_bund_bp: float | None = None
    eurusd: float | None = None
    gbpusd: float | None = None
    eurgbp: float | None = None
    eurusd_1d: float | None = None
    gbpusd_1d: float | None = None
    eurgbp_1d: float | None = None
    cac40: float | None = None
    cac40_1d: float | None = None
    french_banks_1d: float | None = None
    banks_vs_cac_1d: float | None = None
    bank_names: list[str] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


class PricedQuote(BaseModel):
    code: str
    name: str
    group: str
    price: float | None = None
    change_1d: float | None = None
    ts: datetime | None = None
    source: str
    missing: str | None = None


class PolymarketContract(BaseModel):
    topic: str
    event: str
    question: str
    slug: str
    token_id: str | None = None
    yes: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    end_date: str | None = None
    updated_at: str | None = None
    source: str = "Polymarket gamma API"
    missing: str | None = None


class PolymarketBoard(BaseModel):
    as_of: datetime
    contracts: list[PolymarketContract] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class EuropeVerdict(BaseModel):
    stage: str
    driver: str
    eur: str
    gbp_vs_eur: str
    gbp_vs_usd: str
    french_domestic: str
    french_banks: str
    french_exporters: str
    gold: str
    evidence: list[str] = Field(default_factory=list)
    not_proven: list[str] = Field(default_factory=list)
    snapshot: EuropeSnapshot
