"""Combine separate hypotheses by mechanism. Two headlines in one channel count once."""

from __future__ import annotations

from dataclasses import dataclass

from gold_signal.hypothesis import HypothesisSpec, parse_hypothesis

MECHANISMS = ("Macro", "FX", "Commodity", "Event", "Price")


@dataclass(frozen=True)
class HypothesisReading:
    asset: str
    hypothesis_id: str
    name: str
    mechanism: str
    side: str
    status: str
    as_of: str = ""


def mechanism_of(spec: HypothesisSpec) -> str:
    """One hypothesis votes in one channel. Confirmations do not create extra votes."""
    text = f"{spec.name} {spec.trigger}".lower()
    if spec.trigger.startswith("GEOPOLITICAL") or "geopolitical" in text:
        return "Event"
    if spec.trigger.startswith(("NFP_", "CPI_")):
        return "Macro"
    factors = " ".join(condition.factor for condition in spec.confirmations)
    if "DFII10" in factors or "US10Y" in factors:
        return "Macro"
    if "DXY" in factors:
        return "FX"
    others = [condition.factor for condition in spec.confirmations if not condition.factor.startswith(spec.asset)]
    if any(factor.startswith(("XAGUSD", "USOIL")) for factor in others):
        return "Commodity"
    return "Price"


def build_thesis(readings: list[HypothesisReading]) -> list[dict]:
    assets: list[str] = []
    for reading in readings:
        if reading.asset not in assets:
            assets.append(reading.asset)
    return [_asset_thesis(asset, [row for row in readings if row.asset == asset]) for asset in assets]


def thesis_for_store(store) -> list[dict]:
    pairs: list[tuple[str, list[dict]]] = []
    for agent in store.list_agents():
        if not agent.active_version_id:
            continue
        version = store.get_version(agent.active_version_id)
        if version is None:
            continue
        pairs.append((version.hypothesis_yaml, store.list_signals(agent.id)))
    return build_thesis(readings_from_agents(pairs))


def readings_from_agents(agents: list[tuple[str, list[dict]]]) -> list[HypothesisReading]:
    """Latest LONG/SHORT signal is a pass. No signal is not a pass."""
    readings: list[HypothesisReading] = []
    for yaml, signals in agents:
        spec = parse_hypothesis(yaml)
        latest = signals[-1] if signals else None
        side = str(latest["side"]) if latest else "FLAT"
        passed = side in ("LONG", "SHORT")
        readings.append(
            HypothesisReading(
                asset=spec.asset,
                hypothesis_id=spec.id,
                name=spec.name,
                mechanism=mechanism_of(spec),
                side=side if passed else "FLAT",
                status="PASS" if passed else "FAIL",
                as_of="" if latest is None else str(latest.get("evaluated_at") or ""),
            )
        )
    return readings


def _asset_thesis(asset: str, readings: list[HypothesisReading]) -> dict:
    by_mechanism: dict[str, list[HypothesisReading]] = {name: [] for name in MECHANISMS}
    for reading in readings:
        by_mechanism.setdefault(reading.mechanism, []).append(reading)
    mechanisms = []
    agreeing: list[str] = []
    for name in list(MECHANISMS) + [key for key in by_mechanism if key not in MECHANISMS]:
        rows = by_mechanism.get(name) or []
        if not rows:
            continue
        passed = [row.side for row in rows if row.status == "PASS" and row.side in ("LONG", "SHORT")]
        sides = set(passed)
        if sides == {"LONG"}:
            side = "LONG"
            agreeing.append(side)
        elif sides == {"SHORT"}:
            side = "SHORT"
            agreeing.append(side)
        elif len(sides) > 1:
            side = "CONFLICT"
        else:
            side = "NONE"
        mechanisms.append(
            {
                "mechanism": name,
                "side": side,
                "hypotheses": [
                    {"id": row.hypothesis_id, "name": row.name, "status": row.status, "side": row.side}
                    for row in rows
                ],
            }
        )
    unique = set(agreeing)
    if unique == {"LONG"}:
        bias = "LONG"
    elif unique == {"SHORT"}:
        bias = "SHORT"
    elif len(unique) > 1:
        bias = "CONFLICT"
    else:
        bias = "NO_SETUP"
    aligned = len(agreeing) if bias in ("LONG", "SHORT") else 0
    stamps = [row.as_of for row in readings if row.as_of]
    return {
        "asset": asset,
        "bias": bias,
        "aligned_mechanisms": aligned,
        "as_of": max(stamps) if stamps else "",
        "mechanisms": mechanisms,
    }
