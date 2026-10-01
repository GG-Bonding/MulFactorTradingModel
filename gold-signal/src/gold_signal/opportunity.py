"""Turn a thesis into something worth looking at. Conflict is not a setup."""

from __future__ import annotations

from gold_signal.hypothesis import parse_hypothesis
from gold_signal.thesis import thesis_for_store

_RANK = {"LONG_SETUP": 0, "SHORT_SETUP": 0, "WATCH": 1, "NO_SETUP": 2}


def build_opportunities(theses: list[dict], evidence: dict[str, str] | None = None) -> list[dict]:
    known = evidence or {}
    rows = [_opportunity(thesis, known) for thesis in theses]
    return sorted(rows, key=lambda row: (_RANK[row["status"]], row["asset"]))


def opportunities_for_store(store) -> list[dict]:
    evidence: dict[str, str] = {}
    for agent in store.list_agents():
        if not agent.active_version_id:
            continue
        version = store.get_version(agent.active_version_id)
        if version is None:
            continue
        spec = parse_hypothesis(version.hypothesis_yaml)
        runs = [row for row in store.list_backtests(agent.id) if row["version_id"] == version.id]
        if not runs:
            continue
        status = runs[-1]["report"].get("evidence_status")
        if status:
            evidence[spec.id] = str(status)
    return build_opportunities(thesis_for_store(store), evidence)


def _opportunity(thesis: dict, evidence: dict[str, str]) -> dict:
    mechanisms = []
    price_side = None
    for row in thesis["mechanisms"]:
        hypotheses = []
        for item in row["hypotheses"]:
            hypotheses.append({**item, "evidence": evidence.get(item["id"])})
        mechanisms.append({**row, "hypotheses": hypotheses})
        if row["mechanism"] == "Price":
            price_side = row["side"]
    bias = thesis["bias"]
    if bias == "CONFLICT":
        status, note = "NO_SETUP", "Conflict"
    elif bias not in ("LONG", "SHORT"):
        status, note = "NO_SETUP", ""
    elif price_side == "NONE":
        status, note = "WATCH", "Price not triggered"
    elif bias == "LONG":
        status, note = "LONG_SETUP", ""
    else:
        status, note = "SHORT_SETUP", ""
    return {
        "asset": thesis["asset"],
        "status": status,
        "bias": bias,
        "note": note,
        "aligned_mechanisms": thesis["aligned_mechanisms"],
        "as_of": thesis.get("as_of") or "",
        "mechanisms": mechanisms,
    }
