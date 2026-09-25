from datetime import datetime, timedelta, timezone

from gold_signal.observation import CoverageEntry, CoverageManifest, EventRecord, Observation, Recorder
from gold_signal.replay_clock import coverage_status, replay_gold_confirmation


def _utc(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 9, 24, hour, minute, second, tzinfo=timezone.utc)


def _tape() -> tuple[EventRecord, list[Observation]]:
    published = _utc(14, 30)
    event = EventRecord(
        event_id="news-1430",
        event_type="GEOPOLITICAL_ESCALATION",
        published_at=published,
        available_at=published,
        ingested_at=published + timedelta(seconds=2),
        title="headline",
        content="headline",
        source="fixture",
    )
    prices = [
        Observation("market.XAUUSD.close", 3800.0, _utc(14, 30), _utc(14, 30), _utc(14, 30), "fixture", "XAUUSD"),
        Observation("market.XAUUSD.close", 3810.0, _utc(14, 31), _utc(14, 31), _utc(14, 31), "fixture", "XAUUSD"),
        Observation("market.XAUUSD.close", 3820.0, _utc(14, 32), _utc(14, 32), _utc(14, 32), "fixture", "XAUUSD"),
    ]
    return event, prices


def test_reaction_cannot_enter_before_the_minute_arrives():
    event, prices = _tape()
    rows = {stamp: (decision, outcome) for stamp, decision, outcome in replay_gold_confirmation(event, prices)}
    before = rows[_utc(14, 30, 59)]
    at_minute = rows[_utc(14, 31)]
    later = rows[_utc(14, 32)]
    assert before[0] is None
    decision = at_minute[0]
    assert decision is not None
    assert decision.side == "LONG"
    assert decision.entry_at >= _utc(14, 31)
    assert decision.entry_price == 3810.0
    outcome = later[1]
    assert outcome is not None
    trade = 3820.0 / 3810.0 - 1
    event_move = 3820.0 / 3800.0 - 1
    assert abs(outcome.return_from_entry - trade) < 1e-12
    assert abs(outcome.return_from_entry - event_move) > 1e-4


def test_replay_is_deterministic():
    event, prices = _tape()
    first = replay_gold_confirmation(event, prices)
    second = replay_gold_confirmation(event, prices)
    assert first == second


def test_future_print_is_invisible():
    from gold_signal.observation import FactorResolver

    _event, prices = _tape()
    resolver = FactorResolver(prices)
    hidden = resolver.price_at("market.XAUUSD.close", _utc(14, 31), _utc(14, 30, 59))
    assert hidden is not None
    assert hidden.value == 3800.0


def test_coverage_refuses_a_missing_series():
    manifest = CoverageManifest(
        entries={
            "market.XAUUSD.close": CoverageEntry(
                "market.XAUUSD.close",
                _utc(0, 0).replace(month=1, day=1),
                _utc(14, 32),
                "1m",
                True,
            )
        }
    )
    report = coverage_status(
        manifest,
        ["market.XAUUSD.close", "market.USOIL.close"],
        _utc(0, 0).replace(month=1, day=1),
        _utc(14, 0),
    )
    assert report["status"] == "INSUFFICIENT_HISTORY"
    assert report["samples"] == 0
    assert any("USOIL" in item for item in report["missing"])


def test_recorder_keeps_the_three_timestamps(tmp_path):
    path = tmp_path / "live.jsonl"
    recorder = Recorder(path)
    stamp = _utc(14, 30)
    recorder.append(
        "observation",
        {
            "factor": "market.XAUUSD.close",
            "value": 3800.0,
            "observed_at": stamp,
            "available_at": stamp,
            "ingested_at": stamp + timedelta(seconds=2),
            "source": "jin10",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "observed_at" in text
    assert "available_at" in text
    assert "ingested_at" in text
