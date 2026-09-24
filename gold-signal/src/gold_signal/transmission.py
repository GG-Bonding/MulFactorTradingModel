from __future__ import annotations

import re
from dataclasses import dataclass

from gold_signal.models import NewsImpact
from gold_signal.news import classify_btc_news, classify_gold_news

_WAR = re.compile(r"(战争|开战|空袭|导弹|军事冲突|冲突升级|地缘政治|袭击|遭袭|霍尔木兹)")
_US_MACRO = ("非农", "CPI", "PCE", "美联储", "FOMC", "Powell", "鲍威尔", "美债", "美国国债", "实际利率", "TIPS", "降息", "加息", "鹰派", "鸽派")
_FRANCE_YIELD_UP = re.compile(r"(法国国债|OAT|欧债).{0,16}(收益率)?.{0,8}(上涨|上升|走高|飙升|上行)")
_FRANCE_YIELD_DOWN = re.compile(r"(法国国债|OAT|欧债).{0,16}(收益率)?.{0,8}(下跌|回落|走低|下行)")
_OIL_UP = re.compile(r"(原油|WTI|布伦特|石油|OPEC).{0,16}(上涨|减产|紧缺|中断|飙升|供应紧张)")
_OIL_DOWN = re.compile(r"(原油|WTI|布伦特|石油|OPEC).{0,16}(下跌|增产|过剩|需求疲软|累库|供应增加)")


@dataclass(frozen=True)
class TransmissionRule:
    family: str
    event: str
    direction: int | str
    reason: str


# First matching rule wins. Direction "gold" copies the gold classifier; "invert_gold" flips it.
TRANSMISSION_RULES: tuple[TransmissionRule, ...] = (
    TransmissionRule("metal", "gold_rules", "gold", "黄金规则"),
    TransmissionRule("crypto", "btc_rules", "btc", "比特币自身规则"),
    TransmissionRule("oil", "oil_down", -1, "原油供应增加或需求走弱"),
    TransmissionRule("oil", "oil_up_or_war", 1, "原油供应收紧或地缘冲击供给"),
    TransmissionRule("eur", "war", -1, "风险事件偏美元，欧元承压"),
    TransmissionRule("eur", "france_yield_up", -1, "法国或欧债溢价上升，欧元承压"),
    TransmissionRule("eur", "france_yield_down", 1, "法国或欧债溢价回落，欧元压力减轻"),
    TransmissionRule("eur", "us_macro", "gold", "美元宏观：欧元与黄金同向于美元强弱"),
    TransmissionRule("dollar", "war", 0, "这条新闻没有美元兑日元方向"),
    TransmissionRule("dollar", "us_macro", "invert_gold", "美元宏观：美日与黄金方向相反"),
    TransmissionRule("nasdaq", "war", -1, "风险事件偏空股指"),
    TransmissionRule("nasdaq", "us_macro", "gold", "美国利率新闻：股指与黄金同向于宽松或收紧"),
)


def apply_transmission(text: str, family: str) -> NewsImpact:
    raw = (text or "").strip()
    if not raw:
        return _none("empty")
    for rule in TRANSMISSION_RULES:
        if rule.family != family or not _matches(raw, rule.event):
            continue
        return _apply(raw, rule)
    return _none(_missing_reason(family))


def _matches(text: str, event: str) -> bool:
    if event in ("gold_rules", "btc_rules"):
        return True
    if event == "war":
        return _WAR.search(text) is not None
    if event == "us_macro":
        return _has_us_macro(text)
    if event == "france_yield_up":
        return _FRANCE_YIELD_UP.search(text) is not None
    if event == "france_yield_down":
        return _FRANCE_YIELD_DOWN.search(text) is not None
    if event == "oil_down":
        return _OIL_DOWN.search(text) is not None
    if event == "oil_up_or_war":
        return _OIL_UP.search(text) is not None or _WAR.search(text) is not None
    return False


def _apply(text: str, rule: TransmissionRule) -> NewsImpact:
    if rule.direction == "gold":
        gold = classify_gold_news(text)
        if gold.direction == 0:
            return _none(_missing_reason(rule.family)) if rule.family != "metal" else gold
        if rule.family == "metal":
            return gold
        return _hit(gold.direction, rule.reason)
    if rule.direction == "invert_gold":
        gold = classify_gold_news(text)
        if gold.direction == 0:
            return _none("美国宏观方向不确定，不映射美日")
        return _hit(-gold.direction, rule.reason)
    if rule.direction == "btc":
        return classify_btc_news(text)
    if rule.direction == 0:
        return _none(rule.reason)
    return _hit(int(rule.direction), rule.reason)


def _has_us_macro(text: str) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in _US_MACRO)


def _missing_reason(family: str) -> str:
    return {
        "oil": "这条新闻没有原油方向",
        "eur": "这条新闻没有欧元方向",
        "dollar": "这条新闻没有美元兑日元方向",
        "nasdaq": "这条新闻没有纳指方向",
    }.get(family, "这条新闻没有方向")


def _hit(direction: int, reason: str) -> NewsImpact:
    return NewsImpact(direction=direction, importance=2, confidence=0.75, reason=reason)


def _none(reason: str) -> NewsImpact:
    return NewsImpact(direction=0, importance=0, confidence=0.0, reason=reason)
