from datetime import datetime, timedelta, timezone

from gold_signal.compiler import compile_idea
from gold_signal.experiment import ResearchWindow, WalkWindow, run_experiment, run_walk_forward
from gold_signal.observation import EventRecord, Observation


def _stamp(day: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, day, 14, minute, tzinfo=timezone.utc)


def _event(day: int) -> EventRecord:
    published = _stamp(day)
    text = "美国非农高于预期"
    return EventRecord(f"nfp-{day}", "NFP_ABOVE_EXPECTATION", published, published, published, text, text, "fixture")


def _tape(day: int, reaction_price: float, exit_price: float) -> list[Observation]:
    origin = _stamp(day)
    rows = []
    for minute, price in ((0, 100.0), (1, reaction_price), (6, exit_price)):
        stamp = origin + timedelta(minutes=minute)
        rows.append(Observation("market.XAUUSD.close", price, stamp, stamp, stamp, "fixture", "XAUUSD"))
    return rows


def test_parameters_lock_on_train_and_are_not_refit_on_oos():
    compiled = compile_idea("如果非农高于预期，而且黄金一分钟上涨，我做多黄金。")
    assert compiled.spec is not None
    events = [_event(2), _event(3), _event(20)]
    observations = [
        *_tape(2, 101.0, 103.0),
        *_tape(3, 100.1, 90.0),
        *_tape(20, 100.1, 120.0),
    ]
    result = run_experiment(
        compiled.spec,
        events,
        observations,
        train=ResearchWindow("2026-03-01", "2026-03-10"),
        validation=ResearchWindow("2026-03-11", "2026-03-15"),
        oos=ResearchWindow("2026-03-16", "2026-03-31"),
        thresholds=(0.0008, 0.0012),
    )
    assert result["locked_threshold"] == 0.0012
    assert result["train"]["trades"] == 1
    assert result["oos"]["trades"] == 0
    loose = next(row for row in result["search"] if row["threshold"] == 0.0008)
    assert loose["trades"] == 2


def test_walk_forward_freezes_each_window_before_its_test():
    compiled = compile_idea("如果非农高于预期，而且黄金一分钟上涨，我做多黄金。")
    assert compiled.spec is not None
    events = [_event(2), _event(3), _event(12)]
    observations = [
        *_tape(2, 101.0, 103.0),
        *_tape(3, 100.1, 90.0),
        *_tape(12, 100.1, 120.0),
    ]
    folds = run_walk_forward(
        compiled.spec,
        events,
        observations,
        [
            WalkWindow(
                ResearchWindow("2026-03-01", "2026-03-10"),
                ResearchWindow("2026-03-11", "2026-03-20"),
            )
        ],
        (0.0008, 0.0012),
    )
    assert folds[0]["locked_threshold"] == 0.0012
    assert folds[0]["test_trades"] == 0
