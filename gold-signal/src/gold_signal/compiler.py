from __future__ import annotations

import re
from dataclasses import dataclass

from gold_signal.hypothesis import Condition, HypothesisSpec, matching_triggers, parse_hypothesis
from gold_signal.models import Thresholds

_ASSETS: tuple[tuple[str, str, str], ...] = (
    ("XAUUSD", "XAUUSD", "metal"),
    ("XAGUSD", "XAGUSD", "metal"),
    ("USOIL", "USOIL", "oil"),
    ("EURUSD", "EURUSD", "eur"),
    ("黄金", "XAUUSD", "metal"),
    ("金价", "XAUUSD", "metal"),
    ("白银", "XAGUSD", "metal"),
    ("原油", "USOIL", "oil"),
    ("WTI", "USOIL", "oil"),
    ("欧元", "EURUSD", "eur"),
)

_FACTOR_ORDER = (
    "DFII10.change",
    "XAUUSD.reaction_1m",
    "XAGUSD.reaction_1m",
    "EURUSD.reaction_1m",
    "USOIL.reaction_1m",
)

_OPERATORS = {"same_sign", "same_sign_abs_gte", ">", ">=", "<", "<="}
_HORIZONS = ("1m", "3m", "5m", "15m", "30m")
_DEFAULT_HORIZONS = ("1m", "3m", "5m", "15m")


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    spec: HypothesisSpec | None
    yaml: str
    errors: tuple[str, ...]


def compile_idea(text: str) -> CompileResult:
    """Turn one trading sentence into a spec. Refuse when the sentence is incomplete."""
    raw = (text or "").strip()
    if not raw:
        return _refuse("没有交易想法")
    asset = _trade_target(raw)
    if asset is None:
        return _refuse("没有说交易哪个资产，也没有说做多还是做空")
    code, family, entry = asset
    triggers = matching_triggers(raw)
    if len(triggers) > 1:
        return _refuse("这句话里有多个触发事件: " + ", ".join(triggers))
    confirmations = _confirmations(raw)
    if not triggers and not confirmations:
        return _refuse("没有触发事件，也没有确认条件")
    spec = HypothesisSpec(
        id=_spec_id(triggers[0] if triggers else "price", code, entry),
        name=_spec_name(triggers[0] if triggers else "", code, entry),
        asset=code,
        family=family,
        entry=entry,
        horizons=_DEFAULT_HORIZONS,
        confirmations=confirmations,
        trigger=triggers[0] if triggers else "",
    )
    return _accept(spec)


def compile_model_output(payload: dict) -> CompileResult:
    """Validate a model-written spec. Invalid structure does not become a strategy."""
    if not isinstance(payload, dict):
        return _refuse("模型输出不是规格")
    asset = str(payload.get("asset") or "")
    entry = str(payload.get("entry") or "")
    family = str(payload.get("family") or _family_for(asset))
    trigger = str(payload.get("trigger") or "")
    if isinstance(payload.get("trigger"), dict):
        trigger = str(payload["trigger"].get("event") or "")
    if asset not in {item[1] for item in _ASSETS}:
        return _refuse(f"未知资产 {asset}")
    if entry not in ("LONG", "SHORT", "FOLLOW_NEWS"):
        return _refuse(f"未知入场 {entry}")
    horizons = payload.get("evaluation") or payload.get("horizons") or list(_DEFAULT_HORIZONS)
    if not isinstance(horizons, list) or any(item not in _HORIZONS for item in horizons):
        return _refuse("评估期限不在 1m、3m、5m、15m、30m")
    raw_conditions = payload.get("confirmations")
    if not isinstance(raw_conditions, list):
        return _refuse("没有确认条件")
    confirmations: list[Condition] = []
    for item in raw_conditions:
        parsed = _condition_from_model(item)
        if isinstance(parsed, str):
            return _refuse(parsed)
        confirmations.append(parsed)
    if not trigger and not confirmations:
        return _refuse("没有触发事件，也没有确认条件")
    spec = HypothesisSpec(
        id=str(payload.get("id") or _spec_id(trigger or "model", asset, entry)),
        name=str(payload.get("name") or _spec_name(trigger, asset, entry)),
        asset=asset,
        family=family,
        entry=entry,
        horizons=tuple(str(item) for item in horizons),
        confirmations=tuple(confirmations),
        trigger=trigger,
    )
    return _accept(spec)


def render_hypothesis(spec: HypothesisSpec) -> str:
    lines = [
        f"id: {spec.id}",
        f"name: {spec.name}",
        f"asset: {spec.asset}",
        f"family: {spec.family}",
        f"entry: {spec.entry}",
    ]
    if spec.trigger:
        lines.append(f"trigger: {spec.trigger}")
    lines.append("horizons:")
    for horizon in spec.horizons:
        lines.append(f"  - {horizon}")
    lines.append("confirmations:")
    for condition in spec.confirmations:
        lines.append(f"  - factor: {condition.factor}")
        if condition.operator in {"<", ">", "<=", ">="}:
            lines.append(f'    operator: "{condition.operator}"')
        else:
            lines.append(f"    operator: {condition.operator}")
        if condition.value is not None:
            lines.append(f"    value: {_format_value(condition.value)}")
    return "\n".join(lines) + "\n"


def _accept(spec: HypothesisSpec) -> CompileResult:
    yaml = render_hypothesis(spec)
    parsed = parse_hypothesis(yaml)
    if not _same_spec(spec, parsed):
        return _refuse("编译结果无法按原样读回")
    return CompileResult(True, parsed, yaml, ())


def _refuse(reason: str) -> CompileResult:
    return CompileResult(False, None, "", (reason,))


def _same_spec(left: HypothesisSpec, right: HypothesisSpec) -> bool:
    return (
        left.id == right.id
        and left.asset == right.asset
        and left.family == right.family
        and left.entry == right.entry
        and left.trigger == right.trigger
        and left.horizons == right.horizons
        and left.confirmations == right.confirmations
    )


def _trade_target(text: str) -> tuple[str, str, str] | None:
    patterns = (
        (r"做空\s*(黄金|金价|XAUUSD)", "XAUUSD", "metal", "SHORT"),
        (r"做多\s*(黄金|金价|XAUUSD)", "XAUUSD", "metal", "LONG"),
        (r"(黄金|金价).{0,8}(还能|继续)?(涨|上涨)", "XAUUSD", "metal", "LONG"),
        (r"(黄金|金价).{0,8}(还能|继续)?(跌|下跌)", "XAUUSD", "metal", "SHORT"),
        (r"做多\s*(白银|XAGUSD)", "XAGUSD", "metal", "LONG"),
        (r"做空\s*(白银|XAGUSD)", "XAGUSD", "metal", "SHORT"),
        (r"做多\s*(原油|WTI|USOIL)", "USOIL", "oil", "LONG"),
        (r"做空\s*(原油|WTI|USOIL)", "USOIL", "oil", "SHORT"),
        (r"做多\s*(欧元|EURUSD)", "EURUSD", "eur", "LONG"),
        (r"long\s+gold", "XAUUSD", "metal", "LONG"),
        (r"short\s+gold", "XAUUSD", "metal", "SHORT"),
    )
    for pattern, code, family, entry in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return code, family, entry
    return None


def _confirmations(text: str) -> tuple[Condition, ...]:
    found: dict[str, Condition] = {}
    if re.search(r"实际利率.{0,8}(下降|下跌|回落|走低|下行)", text):
        found["DFII10.change"] = Condition("DFII10.change", "<", 0.0)
    elif re.search(r"实际利率.{0,8}(上升|上涨|走高|上行)", text):
        found["DFII10.change"] = Condition("DFII10.change", ">", 0.0)
    clauses = re.split(r"[，,；;]|而且|并且", text)
    for clause in clauses:
        if not re.search(r"(一分钟|1分钟|1m)", clause, flags=re.IGNORECASE):
            continue
        if not re.search(r"(上涨|下跌|走高|走低|同步)", clause):
            continue
        operator = "<" if re.search(r"(下跌|走低)", clause) else ">"
        threshold = _percent(clause)
        if threshold is None:
            threshold = Thresholds.RETURN_1M
        for label, code, _family in _ASSETS:
            if label in clause or label.lower() in clause.lower():
                found[f"{code}.reaction_1m"] = Condition(f"{code}.reaction_1m", operator, threshold)
    ordered = [found[factor] for factor in _FACTOR_ORDER if factor in found]
    return tuple(ordered)


def _condition_from_model(item: object) -> Condition | str:
    if not isinstance(item, dict):
        return "确认条件不是对象"
    factor = str(item.get("factor") or "")
    operator = str(item.get("operator") or "")
    if not re.fullmatch(r"[A-Z0-9]+\.[A-Za-z0-9_]+", factor):
        return f"未知因子 {factor}"
    if operator not in _OPERATORS:
        return f"未知比较 {operator}"
    value = item.get("value")
    if value is None:
        return Condition(factor, operator, None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{factor} 的阈值不是数字"
    return Condition(factor, operator, number)


def _percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match is None:
        return None
    return float(match.group(1)) / 100.0


def _family_for(asset: str) -> str:
    for _label, code, family in _ASSETS:
        if code == asset:
            return family
    return "metal"


def _spec_id(trigger: str, asset: str, entry: str) -> str:
    slug = (trigger or "price").lower()
    return f"{slug}_{asset.lower()}_{entry.lower()}"


def _spec_name(trigger: str, asset: str, entry: str) -> str:
    labels = {
        "CPI_BELOW_EXPECTATION": "CPI 低于预期",
        "CPI_ABOVE_EXPECTATION": "CPI 高于预期",
        "NFP_BELOW_EXPECTATION": "非农低于预期",
        "NFP_ABOVE_EXPECTATION": "非农高于预期",
        "GEOPOLITICAL_ESCALATION": "地缘冲突升级",
    }
    action = "做多" if entry == "LONG" else "做空" if entry == "SHORT" else "跟随"
    event = labels.get(trigger, "价格确认")
    return f"{event}后{action}{asset}"


def _format_value(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".10g")
