from __future__ import annotations

import json
from pathlib import Path

HORIZONS = ("reaction_15s", "reaction_30s", "reaction_1m", "reaction_3m", "reaction_5m")


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def win_rate(rows: list[dict], horizon: str = "reaction_5m") -> dict:
    """Win means a BUY rose or a SELL fell after the news. Waiting rows are not a 0% rate."""
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon}")
    ready: list[tuple[bool, float]] = []
    waiting = 0
    for row in rows:
        signal = row.get("signal")
        if signal not in ("BUY", "SELL"):
            continue
        value = row.get(horizon)
        if value is None:
            waiting += 1
            continue
        signed = value if signal == "BUY" else -value
        won = signed > 0
        ready.append((won, signed))
    if not ready:
        return {
            "horizon": horizon,
            "status": "INSUFFICIENT",
            "samples": 0,
            "waiting": waiting,
            "wins": 0,
            "win_rate": None,
            "avg_return": None,
        }
    wins = sum(1 for won, _signed in ready if won)
    average = sum(signed for _won, signed in ready) / len(ready)
    return {
        "horizon": horizon,
        "status": "OK",
        "samples": len(ready),
        "waiting": waiting,
        "wins": wins,
        "win_rate": wins / len(ready),
        "avg_return": average,
    }


def win_rate_table(rows: list[dict]) -> list[dict]:
    return [win_rate(rows, horizon) for horizon in HORIZONS]
