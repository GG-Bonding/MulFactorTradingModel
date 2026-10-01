"""Pick parameters on the train window only, then freeze them."""

from __future__ import annotations

from dataclasses import dataclass

from gold_signal.hypothesis import HypothesisSpec
from gold_signal.observation import EventRecord, Observation
from gold_signal.research import replay_report, sweep_reaction_threshold


@dataclass(frozen=True)
class ResearchWindow:
    start: str
    end: str


@dataclass(frozen=True)
class WalkWindow:
    """Train a threshold here, then score the following period without refitting."""

    train: ResearchWindow
    test: ResearchWindow


def select_threshold(rows: list[dict]) -> float:
    """Highest train average net. A tie keeps the stricter threshold."""
    eligible = [row for row in rows if (row.get("trades") or 0) > 0 and row.get("avg_return") is not None]
    if not eligible:
        return max(row["threshold"] for row in rows)
    return max(eligible, key=lambda row: (row["avg_return"], row["threshold"]))["threshold"]


def run_experiment(
    spec: HypothesisSpec,
    events: list[EventRecord],
    observations: list[Observation],
    *,
    train: ResearchWindow,
    validation: ResearchWindow,
    oos: ResearchWindow,
    thresholds: tuple[float, ...],
) -> dict:
    search = sweep_reaction_threshold(spec, events, observations, thresholds, train.start, train.end)
    locked = select_threshold(search)
    tuned = _locked(spec, locked)
    return {
        "locked_threshold": locked,
        "search": search,
        "train": _score(tuned, events, observations, train),
        "validation": _score(tuned, events, observations, validation),
        "oos": _score(tuned, events, observations, oos),
    }


def run_walk_forward(
    spec: HypothesisSpec,
    events: list[EventRecord],
    observations: list[Observation],
    windows: list[WalkWindow],
    thresholds: tuple[float, ...],
) -> list[dict]:
    folds = []
    for window in windows:
        search = sweep_reaction_threshold(
            spec, events, observations, thresholds, window.train.start, window.train.end
        )
        locked = select_threshold(search)
        test = _score(_locked(spec, locked), events, observations, window.test)
        folds.append(
            {
                "locked_threshold": locked,
                "train_start": window.train.start,
                "train_end": window.train.end,
                "test_start": window.test.start,
                "test_end": window.test.end,
                "test_trades": test["trades"],
                "test_avg_return": test["avg_return"],
            }
        )
    return folds


def _score(
    spec: HypothesisSpec,
    events: list[EventRecord],
    observations: list[Observation],
    window: ResearchWindow,
) -> dict:
    report = replay_report(spec, window.start, window.end, events, observations)
    return {"trades": report["trades"], "avg_return": report.get("avg_return")}


def _locked(spec: HypothesisSpec, threshold: float) -> HypothesisSpec:
    from dataclasses import replace

    confirmations = tuple(
        replace(condition, value=threshold)
        if condition.factor.endswith(("reaction_1bar", "reaction_1m")) and condition.value is not None
        else condition
        for condition in spec.confirmations
    )
    return replace(spec, confirmations=confirmations)
