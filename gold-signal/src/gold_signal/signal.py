from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from gold_signal.models import (
    FlashNews,
    MarketSnapshot,
    NewsImpact,
    ScoreBreakdown,
    SignalResult,
    SignalSide,
    Thresholds,
)
from gold_signal.news import (
    classify_news,
    ensure_event_id,
    news_age_seconds,
    news_score,
    news_weight,
)


class SignalEngine:
    def __init__(self) -> None:
        self._signaled: set[str] = set()

    def evaluate(
        self,
        news: FlashNews | None,
        market: MarketSnapshot,
        now: datetime | None = None,
    ) -> SignalResult:
        now = now or market.as_of
        impact = classify_news(news.text) if news else NewsImpact(
            direction=0, importance=0, confidence=0.0, reason="无新闻"
        )
        age = news_age_seconds(news, now) if news else 10_000
        weight = news_weight(age) if news else 0.0
        n_score = news_score(impact, age) if news else 0
        event_id = ensure_event_id(news) if news else None

        rejected = None
        if n_score > 0 and market.xau.return_1m <= -Thresholds.RETURN_1M and market.xau.return_3m < 0:
            rejected = "Bullish news rejected by market"
            n_score = 0
        elif n_score < 0 and market.xau.obvious_1m > 0 and market.xau.return_3m > 0:
            rejected = "Bearish news rejected by market"
            n_score = 0

        gold_1m = 2 * market.xau.obvious_1m
        gold_3m = 1 * market.xau.obvious_3m
        silver = 1 * market.xag.obvious_1m
        eurusd = 1 * market.eurusd.obvious_1m
        total = n_score + gold_1m + gold_3m + silver + eurusd

        notes = []
        if news:
            notes.append(f"News {impact.label} {n_score:+d}" if not rejected else rejected)
        notes.append(f"XAUUSD 1m {market.xau.return_1m:+.4%} {gold_1m:+d}")
        notes.append(f"XAUUSD 3m {market.xau.return_3m:+.4%} {gold_3m:+d}")
        notes.append(f"XAGUSD 1m {market.xag.return_1m:+.4%} {silver:+d}")
        notes.append(f"EURUSD 1m {market.eurusd.return_1m:+.4%} {eurusd:+d}")

        breakdown = ScoreBreakdown(
            news=n_score,
            gold_1m=gold_1m,
            gold_3m=gold_3m,
            silver=silver,
            eurusd=eurusd,
            total=total,
            notes=notes,
        )

        signal = self._decide(total, market, impact, weight)
        if rejected and signal == SignalSide.BUY:
            signal = SignalSide.HOLD
        if rejected == "Bullish news rejected by market" and market.xau.return_1m >= 0:
            signal = SignalSide.HOLD
        if rejected == "Bearish news rejected by market" and market.xau.return_1m <= 0:
            signal = SignalSide.HOLD

        is_primary = True
        if event_id and event_id in self._signaled and signal != SignalSide.HOLD:
            signal = SignalSide.HOLD
            is_primary = False
            notes.append("同一新闻已发过主信号，不再重复触发")
        elif signal in (SignalSide.BUY, SignalSide.SELL) and event_id:
            self._signaled.add(event_id)

        reason = build_reason(impact, market, rejected, signal)
        confidence = compute_confidence(total, impact, weight, market)

        return SignalResult(
            timestamp=now,
            signal=signal,
            score=total,
            confidence=confidence,
            news=news.text if news else "",
            news_direction=impact.direction,
            news_impact_label=impact.label,
            event_id=event_id,
            xau_price=market.xau.price,
            xau_1m=market.xau.return_1m,
            xau_3m=market.xau.return_3m,
            xag_1m=market.xag.return_1m,
            eurusd_1m=market.eurusd.return_1m,
            breakdown=breakdown,
            reason=reason,
            is_primary=is_primary,
            rejected_by_market=rejected,
            entry=market.xau.price if signal in (SignalSide.BUY, SignalSide.SELL) else None,
        )

    def _decide(
        self,
        score: int,
        market: MarketSnapshot,
        impact: NewsImpact,
        weight: float,
    ) -> SignalSide:
        if weight <= 0 or impact.importance <= 0:
            return SignalSide.HOLD
        gold_up = market.xau.return_1m > 0
        gold_down = market.xau.return_1m < 0
        if score >= Thresholds.BUY_SCORE and gold_up:
            return SignalSide.BUY
        if score <= Thresholds.SELL_SCORE and gold_down:
            return SignalSide.SELL
        return SignalSide.HOLD


def build_reason(
    impact: NewsImpact,
    market: MarketSnapshot,
    rejected: str | None,
    signal: SignalSide,
) -> str:
    parts: list[str] = []
    if rejected:
        parts.append(rejected)
    elif impact.direction > 0:
        parts.append("利多消息")
    elif impact.direction < 0:
        parts.append("利空消息")
    else:
        parts.append("新闻方向不确定")

    parts.append("黄金上涨" if market.xau.return_1m > 0 else "黄金下跌" if market.xau.return_1m < 0 else "黄金持平")
    parts.append("白银上涨" if market.xag.return_1m > 0 else "白银下跌" if market.xag.return_1m < 0 else "白银持平")
    if market.eurusd.return_1m > 0:
        parts.append("美元走弱确认")
    elif market.eurusd.return_1m < 0:
        parts.append("美元走强确认")
    else:
        parts.append("EURUSD持平")
    parts.append(f"最终 {signal.value}")
    return " + ".join(parts)


def compute_confidence(
    score: int,
    impact: NewsImpact,
    weight: float,
    market: MarketSnapshot,
) -> float:
    magnitude = min(abs(score) / Thresholds.MAX_SCORE, 1.0)
    agree = 0.0
    if impact.direction != 0 and (
        (impact.direction > 0 and market.xau.return_1m > 0)
        or (impact.direction < 0 and market.xau.return_1m < 0)
    ):
        agree = 0.12
    conf = 0.35 + 0.45 * magnitude + 0.2 * impact.confidence * weight + agree
    return round(min(max(conf, 0.05), 0.95), 2)


class SignalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: SignalResult) -> None:
        record = result_to_record(result)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def update_returns(
        self,
        event_id: str,
        timestamp: str,
        *,
        return_1m: float | None = None,
        return_5m: float | None = None,
        return_15m: float | None = None,
        return_30m: float | None = None,
    ) -> None:
        rows = self.load()
        changed = False
        for row in rows:
            if row.get("event_id") == event_id and row.get("timestamp") == timestamp:
                if return_1m is not None:
                    row["return_1m"] = return_1m
                if return_5m is not None:
                    row["return_5m"] = return_5m
                if return_15m is not None:
                    row["return_15m"] = return_15m
                if return_30m is not None:
                    row["return_30m"] = return_30m
                changed = True
        if not changed:
            return
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def result_to_record(result: SignalResult) -> dict:
    return {
        "timestamp": result.timestamp.isoformat(),
        "signal": result.signal.value,
        "score": result.score,
        "confidence": result.confidence,
        "news": result.news,
        "news_direction": result.news_direction,
        "event_id": result.event_id,
        "is_primary": result.is_primary,
        "xau_price": result.xau_price,
        "xau_1m": result.xau_1m,
        "xau_3m": result.xau_3m,
        "xag_1m": result.xag_1m,
        "eurusd_1m": result.eurusd_1m,
        "entry": result.entry,
        "return_1m": result.return_1m,
        "return_5m": result.return_5m,
        "return_15m": result.return_15m,
        "return_30m": result.return_30m,
        "reason": result.reason,
    }
