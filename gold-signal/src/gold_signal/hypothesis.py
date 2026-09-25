from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gold_signal.market import event_reaction
from gold_signal.models import FlashNews, MarketSnapshot
from gold_signal.replay_clock import CostModel
from gold_signal.transmission import apply_transmission

REQUIRED_HISTORY = (
    ("news.events", "no point-in-time news archive"),
    ("price.reaction_path", "no stored prices aligned to past headlines"),
)


@dataclass(frozen=True)
class Condition:
    factor: str
    operator: str
    value: float | None = None


@dataclass(frozen=True)
class HypothesisSpec:
    id: str
    name: str
    asset: str
    family: str
    entry: str
    horizons: tuple[str, ...]
    confirmations: tuple[Condition, ...]
    notes: tuple[str, ...] = ()


@dataclass
class HypothesisDecision:
    side: str
    news_direction: int
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def load_hypothesis(path: Path) -> HypothesisSpec:
    return parse_hypothesis(path.read_text(encoding="utf-8"))


def parse_hypothesis(text: str) -> HypothesisSpec:
    data: dict[str, object] = {}
    horizons: list[str] = []
    notes: list[str] = []
    confirmations: list[Condition] = []
    mode: str | None = None
    pending: dict[str, str] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        confirmations.append(
            Condition(
                factor=pending["factor"],
                operator=pending["operator"],
                value=float(pending["value"]) if "value" in pending else None,
            )
        )
        pending = None

    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- ") and indent >= 2 and mode == "horizons":
            horizons.append(line[2:].strip())
            continue
        if line.startswith("- ") and indent >= 2 and mode == "notes":
            notes.append(line[2:].strip())
            continue
        if line.startswith("- ") and mode == "confirmations":
            flush()
            pending = {}
            rest = line[2:].strip()
            if rest:
                key, value = _split_kv(rest)
                pending[key] = value
            continue
        if indent >= 4 and pending is not None and ":" in line:
            key, value = _split_kv(line)
            pending[key] = value
            continue
        flush()
        mode = None
        key, value = _split_kv(line)
        if value == "":
            mode = key
            continue
        data[key] = value
    flush()
    required = ("id", "name", "asset", "family", "entry")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"hypothesis missing {', '.join(missing)}")
    return HypothesisSpec(
        id=str(data["id"]),
        name=str(data["name"]),
        asset=str(data["asset"]),
        family=str(data["family"]),
        entry=str(data["entry"]),
        horizons=tuple(horizons),
        confirmations=tuple(confirmations),
        notes=tuple(notes),
    )


def evaluate_hypothesis(
    spec: HypothesisSpec,
    news: FlashNews | None,
    market: MarketSnapshot,
    now: datetime,
) -> HypothesisDecision:
    if market.xau.code != spec.asset:
        return HypothesisDecision("FLAT", 0, missing=[f"snapshot is {market.xau.code}, hypothesis wants {spec.asset}"])
    impact = apply_transmission(news.text, spec.family) if news else None
    direction = 0 if impact is None else impact.direction
    decision = HypothesisDecision("FLAT", direction)
    if direction == 0:
        decision.reasons.append("news has no direction for this asset")
        return decision
    values = _factor_values(spec, market, news, now)
    for condition in spec.confirmations:
        ok, gap = _check(condition, values, direction)
        if gap:
            decision.missing.append(gap)
            decision.reasons.append(gap)
            return decision
        if not ok:
            decision.reasons.append(f"failed {condition.factor} {condition.operator}")
            return decision
    if spec.entry == "FOLLOW_NEWS":
        decision.side = "LONG" if direction > 0 else "SHORT"
        decision.reasons.append("confirmations passed")
        return decision
    decision.reasons.append(f"unsupported entry {spec.entry}")
    return decision


def historical_validation(
    spec: HypothesisSpec,
    start: str,
    end: str,
    *,
    strategy_path: Path | None = None,
    observations: list | None = None,
    events: list | None = None,
    costs: CostModel | None = None,
) -> dict:
    """Replay a point-in-time archive. With no archive, report the gap and invent nothing."""
    if observations is None or events is None:
        from gold_signal.research import RESEARCH_WINDOWS, yaml_sha256

        missing = [f"{spec.asset}: {reason}" for _key, reason in REQUIRED_HISTORY]
        for condition in spec.confirmations:
            if condition.factor.startswith("DFII10"):
                missing.append("DFII10 is a daily FRED print and is not safe to use before its observation date")
            elif condition.factor not in {item[0] for item in REQUIRED_HISTORY}:
                missing.append(f"{condition.factor}: no historical series aligned to headlines")
        return {
            "hypothesis": spec.id,
            "yaml_sha256": yaml_sha256(strategy_path),
            "start": start,
            "end": end,
            "status": "INSUFFICIENT",
            "samples": 0,
            "win_rate": None,
            "avg_return": None,
            "missing": missing,
            "windows": {
                name: {
                    "start": window_start,
                    "end": window_end,
                    "status": "INSUFFICIENT",
                    "samples": 0,
                    "expectancy": None,
                    "median_return": None,
                    "profit_factor": None,
                    "avg_return": None,
                }
                for name, window_start, window_end in RESEARCH_WINDOWS
            },
        }
    from gold_signal.research import replay_report

    return replay_report(
        spec,
        start,
        end,
        events,
        observations,
        strategy_path=strategy_path,
        costs=costs,
    )


def _factor_values(
    spec: HypothesisSpec,
    market: MarketSnapshot,
    news: FlashNews | None,
    now: datetime,
) -> dict[str, float | None]:
    published = news.published_at if news else now
    values: dict[str, float | None] = {}
    for code, bars in (
        (market.xau.code, market.xau_bars),
        (market.xag.code, market.xag_bars),
        (market.eurusd.code, market.eurusd_bars),
    ):
        reaction = event_reaction(bars, published, now)
        values[f"{code}.reaction_1m"] = None if reaction is None else reaction.return_1m
    return values


def _check(condition: Condition, values: dict[str, float | None], direction: int) -> tuple[bool, str | None]:
    if condition.factor not in values:
        return False, f"missing {condition.factor}"
    value = values[condition.factor]
    if value is None:
        return False, f"{condition.factor} is waiting"
    if condition.operator == "same_sign":
        return (value > 0 and direction > 0) or (value < 0 and direction < 0), None
    if condition.operator == "same_sign_abs_gte":
        threshold = condition.value or 0.0
        return (value > 0 and direction > 0 or value < 0 and direction < 0) and abs(value) >= threshold, None
    if condition.operator == ">=":
        return value >= (condition.value or 0.0), None
    if condition.operator == ">":
        return value > (condition.value or 0.0), None
    if condition.operator == "<=":
        return value <= (condition.value or 0.0), None
    if condition.operator == "<":
        return value < (condition.value or 0.0), None
    return False, f"unknown operator {condition.operator}"


def _split_kv(line: str) -> tuple[str, str]:
    key, _, value = line.partition(":")
    return key.strip(), value.strip().strip('"').strip("'")
