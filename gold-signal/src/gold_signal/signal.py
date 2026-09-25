from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from gold_signal.market import event_reaction
from gold_signal.replay_clock import fill_price
from gold_signal.models import (
    EventReaction,
    FlashNews,
    MarketSnapshot,
    NewsImpact,
    ScoreBreakdown,
    SignalResult,
    SignalSide,
    Thresholds,
)
from gold_signal.news import (
    classify_gold_news,
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
        impact: NewsImpact | None = None,
    ) -> SignalResult:
        now = now or market.as_of
        impact = impact if impact is not None else (
            classify_gold_news(news.text) if news else NewsImpact(
                direction=0, importance=0, confidence=0.0, reason="无新闻"
            )
        )
        age = news_age_seconds(news, now) if news else 10_000
        weight = news_weight(age) if news else 0.0
        n_score = news_score(impact, age) if news else 0
        event_id = ensure_event_id(news) if news else None
        primary = _reaction(market.xau_bars, news, now)
        confirm = _reaction(market.xag_bars, news, now)
        third = _reaction(market.eurusd_bars, news, now)
        post_1m = None if primary is None else primary.return_1m
        post_3m = None if primary is None else primary.return_3m
        confirm_1m = None if confirm is None else confirm.return_1m
        third_1m = None if third is None else third.return_1m

        rejected = None
        if n_score > 0 and post_1m is not None and post_1m <= -Thresholds.RETURN_1M:
            rejected = "Bullish news rejected by market"
            n_score = 0
        elif n_score < 0 and post_1m is not None and post_1m >= Thresholds.RETURN_1M:
            rejected = "Bearish news rejected by market"
            n_score = 0

        gold_1m = 2 * _obvious(post_1m, Thresholds.RETURN_1M)
        gold_3m = 1 * _obvious(post_3m, Thresholds.RETURN_3M)
        silver = 1 * _obvious(confirm_1m, Thresholds.RETURN_1M)
        eurusd = 1 * _obvious(third_1m, Thresholds.RETURN_1M)
        total = n_score + gold_1m + gold_3m + silver + eurusd

        notes = []
        if news and post_1m is None and n_score != 0:
            notes.append("新闻后未满1分钟，或窗口内还没有新价格")
        if news:
            notes.append(f"News {impact.label} {n_score:+d}" if not rejected else rejected)
        notes.append(f"{market.xau.code} post-1m {_fmt_optional(post_1m)} {gold_1m:+d}")
        notes.append(f"{market.xau.code} post-3m {_fmt_optional(post_3m)} {gold_3m:+d}")
        notes.append(f"{market.xag.code} post-1m {_fmt_optional(confirm_1m)} {silver:+d}")
        notes.append(f"{market.eurusd.code} post-1m {_fmt_optional(third_1m)} {eurusd:+d}")

        breakdown = ScoreBreakdown(
            news=n_score,
            gold_1m=gold_1m,
            gold_3m=gold_3m,
            silver=silver,
            eurusd=eurusd,
            total=total,
            notes=notes,
        )

        signal = self._decide(total, post_1m, impact, weight)
        if rejected:
            signal = SignalSide.HOLD

        is_primary = True
        signal_key = f"{market.xau.code}:{event_id}" if event_id else None
        if signal_key and signal_key in self._signaled and signal != SignalSide.HOLD:
            signal = SignalSide.HOLD
            is_primary = False
            notes.append("同一产品的这条新闻已发过主信号，不再重复触发")
        elif signal in (SignalSide.BUY, SignalSide.SELL) and signal_key:
            self._signaled.add(signal_key)

        reason = build_reason(
            impact,
            market,
            rejected,
            signal,
            post_1m,
            confirm_1m,
            third_1m,
        )
        strength = compute_strength(total, impact, weight, post_1m)
        anchor = None if primary is None else primary.anchor_price
        ready_at = news.published_at + timedelta(minutes=1) if news else now
        trade_side = "LONG" if signal == SignalSide.BUY else "SHORT" if signal == SignalSide.SELL else "FLAT"
        _filled_at, entry_px = fill_price(trade_side, market.xau.price, now, ready_at)

        return SignalResult(
            timestamp=now,
            signal=signal,
            score=total,
            strength=strength,
            news=news.text if news else "",
            news_direction=impact.direction,
            news_impact_label=impact.label,
            event_id=event_id,
            product=market.xau.code,
            xau_price=market.xau.price,
            xau_1m=0.0 if post_1m is None else post_1m,
            xau_3m=0.0 if post_3m is None else post_3m,
            xag_1m=0.0 if confirm_1m is None else confirm_1m,
            eurusd_1m=0.0 if third_1m is None else third_1m,
            anchor_price=anchor,
            anchor_ts=None if primary is None else primary.anchor_ts,
            reaction_15s=None if primary is None else primary.return_15s,
            reaction_30s=None if primary is None else primary.return_30s,
            reaction_1m=post_1m,
            reaction_3m=post_3m,
            reaction_5m=None if primary is None else primary.return_5m,
            breakdown=breakdown,
            reason=reason,
            is_primary=is_primary,
            rejected_by_market=rejected,
            entry=entry_px,
        )

    def _decide(
        self,
        score: int,
        post_1m: float | None,
        impact: NewsImpact,
        weight: float,
    ) -> SignalSide:
        if weight <= 0 or impact.importance <= 0 or post_1m is None:
            return SignalSide.HOLD
        if score >= Thresholds.BUY_SCORE and post_1m > 0:
            return SignalSide.BUY
        if score <= Thresholds.SELL_SCORE and post_1m < 0:
            return SignalSide.SELL
        return SignalSide.HOLD


def build_reason(
    impact: NewsImpact,
    market: MarketSnapshot,
    rejected: str | None,
    signal: SignalSide,
    post_1m: float | None,
    confirm_1m: float | None,
    third_1m: float | None,
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

    parts.append(_move_text(market.xau.code, post_1m))
    parts.append(_move_text(market.xag.code, confirm_1m))
    if market.eurusd.code == "FLAT":
        parts.append("无第二确认")
    elif market.eurusd.code == "EURUSD" and market.xau.code in ("XAUUSD", "XAGUSD"):
        if third_1m is None:
            parts.append("EURUSD新闻后未到")
        elif third_1m > 0:
            parts.append("美元走弱确认")
        elif third_1m < 0:
            parts.append("美元走强确认")
        else:
            parts.append("EURUSD持平")
    else:
        parts.append(_move_text(market.eurusd.code, third_1m))
    parts.append(f"最终 {signal.value}")
    return " + ".join(parts)


def _move_text(code: str, ret: float | None) -> str:
    if ret is None:
        return f"{code}新闻后未到"
    if ret > 0:
        return f"{code}新闻后上涨"
    if ret < 0:
        return f"{code}新闻后下跌"
    return f"{code}新闻后持平"


def _reaction(bars: list, news: FlashNews | None, now: datetime) -> EventReaction | None:
    if news is None:
        return None
    return event_reaction(bars, news.published_at, now)


def _obvious(value: float | None, threshold: float) -> int:
    if value is None:
        return 0
    if value >= threshold:
        return 1
    if value <= -threshold:
        return -1
    return 0


def _fmt_optional(value: float | None) -> str:
    return "waiting" if value is None else f"{value:+.4%}"


def compute_strength(
    score: int,
    impact: NewsImpact,
    weight: float,
    post_1m: float | None,
) -> int:
    """Score-shaped strength from 0 to 100. This is not a probability."""
    magnitude = min(abs(score) / Thresholds.MAX_SCORE, 1.0)
    agree = 0.0
    if post_1m is not None and impact.direction != 0 and (
        (impact.direction > 0 and post_1m > 0) or (impact.direction < 0 and post_1m < 0)
    ):
        agree = 0.12
    raw = 0.35 + 0.45 * magnitude + 0.2 * impact.confidence * weight + agree
    return int(round(100 * min(max(raw, 0.05), 0.95)))


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


class EventLog:
    """One row per product event. Later ticks fill horizons that were still waiting."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, result: SignalResult) -> None:
        if not result.event_id:
            return
        rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        record = {
            "event_id": result.event_id,
            "product": result.product,
            "news": result.news,
            "signal": result.signal.value,
            "news_direction": result.news_direction,
            "strength": result.strength,
            "anchor_price": result.anchor_price,
            "anchor_ts": None if result.anchor_ts is None else result.anchor_ts.isoformat(),
            "reaction_15s": result.reaction_15s,
            "reaction_30s": result.reaction_30s,
            "reaction_1m": result.reaction_1m,
            "reaction_3m": result.reaction_3m,
            "reaction_5m": result.reaction_5m,
            "updated_at": result.timestamp.isoformat(),
        }
        for index, row in enumerate(rows):
            if row.get("event_id") == result.event_id and row.get("product") == result.product:
                rows[index] = _fill_event(row, record)
                break
        else:
            rows.append(record)
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fill_event(previous: dict, current: dict) -> dict:
    merged = dict(previous)
    for key, value in current.items():
        if value is None and key.startswith("reaction_"):
            continue
        merged[key] = value
    return merged


def result_to_record(result: SignalResult) -> dict:
    return {
        "timestamp": result.timestamp.isoformat(),
        "signal": result.signal.value,
        "score": result.score,
        "strength": result.strength,
        "news": result.news,
        "news_direction": result.news_direction,
        "event_id": result.event_id,
        "product": result.product,
        "is_primary": result.is_primary,
        "xau_price": result.xau_price,
        "anchor_price": result.anchor_price,
        "anchor_ts": None if result.anchor_ts is None else result.anchor_ts.isoformat(),
        "reaction_15s": result.reaction_15s,
        "reaction_30s": result.reaction_30s,
        "reaction_1m": result.reaction_1m,
        "reaction_3m": result.reaction_3m,
        "reaction_5m": result.reaction_5m,
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
