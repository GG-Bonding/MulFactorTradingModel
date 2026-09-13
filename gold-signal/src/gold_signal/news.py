from __future__ import annotations

import re
from datetime import datetime

from gold_signal.jin10 import make_event_id
from gold_signal.models import FlashNews, NewsImpact, Thresholds

RELEVANT_KEYWORDS = (
    "CPI",
    "PCE",
    "非农",
    "失业率",
    "初请",
    "美联储",
    "利率决议",
    "FOMC",
    "Powell",
    "鲍威尔",
    "GDP",
    "零售销售",
    "美元",
    "战争",
    "军事",
    "冲突",
    "地缘",
    "空袭",
    "导弹",
    "开战",
    "袭击",
    "遭袭",
    "军方",
    "霍尔木兹",
    "比特币",
    "BTC",
    "以太坊",
    "ETH",
    "加密货币",
    "加密",
    "现货ETF",
    "银行危机",
    "金融危机",
    "降息",
    "加息",
    "鹰派",
    "鸽派",
)

HIGHER_IS_HAWKISH = ("CPI", "PCE", "非农", "GDP", "零售销售", "利率")
HIGHER_IS_DOVISH = ("失业率", "初请")

BULLISH_PATTERNS = (
    r"非农.{0,12}(不及|低于|未及|逊于|不及预期|不及预估)",
    r"(CPI|PCE).{0,12}(不及|低于|回落|降温|放缓|弱于)",
    r"失业率.{0,12}(升|高于|超预期|高于预期)",
    r"初请.{0,12}(升|增加|高于|超预期)",
    r"(零售销售|GDP).{0,12}(不及|低于|萎缩|不及预期)",
    r"(降息|鸽派|暂停加息)",
    r"美元.{0,8}(下跌|走弱|大跌|暴跌|下滑)",
    r"(战争|开战|空袭|导弹|军事冲突|冲突升级|地缘政治|袭击|遭袭|霍尔木兹)",
    r"(银行危机|金融危机|违约潮)",
    r"比特币.{0,12}(上涨|突破|暴涨|新高)",
    r"(现货ETF).{0,8}(批准|通过|获批)",
)

BEARISH_PATTERNS = (
    r"非农.{0,12}(好于|高于|超预期|强于|爆表)",
    r"(CPI|PCE).{0,12}(高于|超预期|反弹|升温|强于|超市场)",
    r"失业率.{0,12}(降|低于|不及预期)",
    r"初请.{0,12}(降|减少|低于)",
    r"(零售销售|GDP).{0,12}(好于|高于|超预期|强劲)",
    r"(加息|鹰派|再次加息)",
    r"美元.{0,8}(上涨|走强|大涨|暴涨|升值)",
    r"比特币.{0,12}(暴跌|大跌|闪崩)",
    r"(SEC|监管).{0,12}(打击|起诉|禁止|严打)",
)


def classify_news(text: str) -> NewsImpact:
    raw = (text or "").strip()
    if not raw:
        return NewsImpact(direction=0, importance=0, confidence=0.0, reason="empty")
    if not is_gold_relevant(raw):
        return NewsImpact(direction=0, importance=0, confidence=0.0, reason="普通新闻忽略")

    importance = importance_of(raw)
    actual_vs_expected = classify_actual_vs_expected(raw)
    if actual_vs_expected is not None:
        direction, reason = actual_vs_expected
        return NewsImpact(
            direction=direction,
            importance=importance,
            confidence=0.85,
            reason=reason,
        )

    for pattern in BULLISH_PATTERNS:
        if re.search(pattern, raw, flags=re.IGNORECASE):
            return NewsImpact(
                direction=1,
                importance=importance,
                confidence=0.8,
                reason=f"规则命中利多: {pattern}",
            )
    for pattern in BEARISH_PATTERNS:
        if re.search(pattern, raw, flags=re.IGNORECASE):
            return NewsImpact(
                direction=-1,
                importance=importance,
                confidence=0.8,
                reason=f"规则命中利空: {pattern}",
            )
    return NewsImpact(
        direction=0,
        importance=importance,
        confidence=0.4,
        reason="相关但方向不确定",
    )


def is_gold_relevant(text: str) -> bool:
    return any(keyword.lower() in text.lower() for keyword in RELEVANT_KEYWORDS)


def importance_of(text: str) -> int:
    high = ("非农", "CPI", "PCE", "利率决议", "FOMC", "战争", "军事冲突", "金融危机", "霍尔木兹", "袭击")
    mid = ("失业率", "初请", "GDP", "零售销售", "Powell", "鲍威尔", "美联储", "美元", "比特币", "BTC", "加密")
    if any(k.lower() in text.lower() for k in high):
        return 3
    if any(k.lower() in text.lower() for k in mid):
        return 2
    return 1


def classify_actual_vs_expected(text: str) -> tuple[int, str] | None:
    actual = _extract_number(text, ("公布值", "公布", "实际", "今值"))
    expected = _extract_number(text, ("预期", "预估"))
    if actual is None or expected is None:
        return None
    if actual == expected:
        return 0, f"公布值={actual} 等于预期={expected}"

    stronger = actual > expected
    if any(k in text for k in HIGHER_IS_DOVISH):
        # higher unemployment / claims = weaker USD = bullish gold
        direction = 1 if stronger else -1
        return direction, f"公布值={actual} 预期={expected} → 弱数据偏多黄金" if direction > 0 else (
            f"公布值={actual} 预期={expected} → 强数据偏空黄金"
        )
    if any(k in text for k in HIGHER_IS_HAWKISH):
        direction = -1 if stronger else 1
        return direction, f"公布值={actual} 预期={expected} → 弱数据偏多黄金" if direction > 0 else (
            f"公布值={actual} 预期={expected} → 强数据偏空黄金"
        )
    return None


def news_weight(age_seconds: float) -> float:
    if age_seconds < 0:
        age_seconds = 0
    if age_seconds <= 120:
        return 1.0
    if age_seconds <= 300:
        return 0.7
    if age_seconds <= Thresholds.NEWS_MAX_AGE_SEC:
        return 0.3
    return 0.0


def news_score(impact: NewsImpact, age_seconds: float) -> int:
    if impact.importance <= 0 or impact.direction == 0:
        return 0
    weight = news_weight(age_seconds)
    if weight <= 0:
        return 0
    base = 2 if impact.importance >= 2 else 1
    return int(round(base * impact.direction * weight))


def news_age_seconds(news: FlashNews, now: datetime) -> float:
    published = news.published_at
    if published.tzinfo is None and now.tzinfo is not None:
        published = published.replace(tzinfo=now.tzinfo)
    if now.tzinfo is None and published.tzinfo is not None:
        now = now.replace(tzinfo=published.tzinfo)
    return (now - published).total_seconds()


def ensure_event_id(news: FlashNews) -> str:
    if news.event_id:
        return news.event_id
    return make_event_id(news.published_at, news.text)


def _extract_number(text: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        match = re.search(
            rf"{label}\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        if match:
            return float(match.group(1))
    return None
