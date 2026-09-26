"""Runtime. Live and replay only differ by the context that feeds the engine."""

from gold_signal.runtime.context import (
    EvaluationContext,
    LiveEvaluationContext,
    ReplayEvaluationContext,
    evaluation_now,
)
from gold_signal.runtime.engine import HypothesisEngine

__all__ = [
    "EvaluationContext",
    "HypothesisEngine",
    "LiveEvaluationContext",
    "ReplayEvaluationContext",
    "evaluation_now",
]
