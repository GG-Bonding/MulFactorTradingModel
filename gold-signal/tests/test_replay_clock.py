from datetime import datetime, timedelta, timezone

from gold_signal.observation import CoverageEntry, CoverageManifest, EventRecord, FactorResolver, Observation, Recorder
from gold_signal.replay_clock import coverage_status, reaction_1bar, replay_gold_confirmation


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


def test_off_grid_news_uses_one_completed_bar_not_the_bar_after_next():
    published = datetime(2026, 9, 4, 12, 30, 17, tzinfo=timezone.utc)
    prices = [
        Observation("market.XAUUSD.close", 4518.9, _stamp(12, 30), _stamp(12, 30), _stamp(12, 30), "yahoo", "XAUUSD"),
        Observation("market.XAUUSD.close", 4517.7, _stamp(12, 31), _stamp(12, 31), _stamp(12, 31), "yahoo", "XAUUSD"),
        Observation("market.XAUUSD.close", 4448.6, _stamp(12, 32), _stamp(12, 32), _stamp(12, 32), "yahoo", "XAUUSD"),
    ]
    resolver = FactorResolver(prices)
    early = reaction_1bar(resolver, "market.XAUUSD.close", published, _stamp(12, 30, 59))
    reaction = reaction_1bar(resolver, "market.XAUUSD.close", published, published + timedelta(minutes=1))
    assert early is None
    assert reaction is not None
    assert reaction.anchor_at == _stamp(12, 30)
    assert reaction.later_at == _stamp(12, 31)
    assert reaction.span == timedelta(minutes=1)
    assert abs(reaction.value - (4517.7 / 4518.9 - 1)) < 1e-12


def test_a_skipped_bar_is_not_called_a_one_bar_reaction():
    published = datetime(2026, 9, 4, 12, 30, 17, tzinfo=timezone.utc)
    prices = [
        Observation("market.XAUUSD.close", 4518.9, _stamp(12, 30), _stamp(12, 30), _stamp(12, 30), "yahoo", "XAUUSD"),
        Observation("market.XAUUSD.close", 4448.6, _stamp(12, 32), _stamp(12, 32), _stamp(12, 32), "yahoo", "XAUUSD"),
    ]
    reaction = reaction_1bar(
        FactorResolver(prices),
        "market.XAUUSD.close",
        published,
        _stamp(12, 33),
    )
    assert reaction is None


def test_packaged_archive_includes_later_refinery_strikes():
    from datetime import timezone as tz

    from gold_signal.archive import default_archive

    start = datetime(2026, 9, 1, tzinfo=tz.utc)
    end = datetime(2026, 9, 29, tzinfo=tz.utc)
    events, observations, _missing = default_archive().load(start, end, ["market.XAUUSD.close"])
    titles = " ".join(event.title for event in events)
    assert "非农" in titles
    assert "炼油厂" in titles
    assert "油库" in titles
    sina = [row for row in observations if row.source == "sina"]
    assert sina
    assert all(row.quality == "RECONSTRUCTED" for row in sina)
    published = next(event.published_at for event in events if "炼油厂" in event.title)
    reaction = reaction_1bar(
        FactorResolver(observations),
        "market.XAUUSD.close",
        published,
        published + timedelta(minutes=3),
    )
    assert reaction is not None
    assert reaction.span == timedelta(minutes=1)


def test_packaged_yahoo_opens_become_close_knowable_times():
    from datetime import timezone as tz

    from gold_signal.archive import default_archive

    start = datetime(2026, 9, 4, tzinfo=tz.utc)
    end = datetime(2026, 9, 5, tzinfo=tz.utc)
    events, observations, _missing = default_archive().load(start, end, ["market.XAUUSD.close"])
    published = events[0].published_at
    reaction = reaction_1bar(
        FactorResolver(observations),
        "market.XAUUSD.close",
        published,
        published + timedelta(minutes=3),
    )
    assert default_archive().manifest()["bar_timestamp"] == "open"
    assert reaction is not None
    assert reaction.span == timedelta(minutes=1)
    assert reaction.anchor_at <= published < reaction.later_at


def _stamp(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 9, 4, hour, minute, second, tzinfo=timezone.utc)


def test_indexed_resolver_matches_the_last_visible_close():
    prices = []
    origin = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    for index in range(180):
        stamp = origin + timedelta(minutes=index)
        prices.append(Observation("market.XAUUSD.close", 1000.0 + index, stamp, stamp, None, "fixture", "XAUUSD"))
    resolver = FactorResolver(prices)
    now = datetime(2026, 9, 4, 13, 0, tzinfo=timezone.utc)
    found = resolver.price_at("market.XAUUSD.close", now, now)
    assert found is not None
    assert found.observed_at == now
    assert found.value == 1060.0
    hidden = resolver.price_at("market.XAUUSD.close", now + timedelta(minutes=30), now)
    assert hidden is not None
    assert hidden.observed_at == now


def test_future_print_is_invisible():
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
