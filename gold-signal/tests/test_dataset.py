import json
from datetime import datetime, timedelta, timezone

from gold_signal.dataset.builder import build_dataset
from gold_signal.dataset.cluster import RawEvent
from gold_signal.dataset.event_fetcher import raw_event
from gold_signal.observation import Observation


def _utc(hour: int, minute: int) -> datetime:
    return datetime(2026, 9, 28, hour, minute, tzinfo=timezone.utc)


def _headline(event_id: str, minute: int, text: str) -> RawEvent:
    return raw_event(event_id, _utc(10, minute), text, text, "jin10")


def _prices() -> list[Observation]:
    rows = []
    for minute, price in ((0, 4100.0), (1, 4102.0), (2, 4090.0), (3, 4088.0), (40, 4070.0), (41, 4060.0)):
        stamp = _utc(10, minute)
        rows.append(Observation("market.XAUUSD.close", price, stamp, stamp, stamp, "sina", "XAUUSD"))
    return rows


def test_repeated_wires_about_one_strike_are_one_sample(tmp_path):
    events = [
        _headline("a", 1, "俄罗斯袭击第聂伯一座办公楼"),
        _headline("b", 4, "第聂伯袭击造成 3 人死亡"),
        _headline("c", 8, "州长证实办公楼遭袭"),
        _headline("d", 12, "办公楼被击中"),
        _headline("e", 50, "黑海货船遭袭"),
    ]
    summary = build_dataset(
        tmp_path,
        events,
        _prices(),
        name="gold_macro_v1",
        start=_utc(0, 0),
        end=_utc(12, 0),
    )
    assert summary["samples"] == 1
    assert summary["dropped_duplicates"] == 3
    assert summary["rejected"] == 1
    saved = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 1
    assert saved[0]["member_ids"] == ["a", "b", "c", "d"]
    assert saved[0]["event_group_id"].startswith("geopolitical_escalation_")
    manifest = (tmp_path / "manifest.yaml").read_text(encoding="utf-8")
    assert "cooldown: 30m" in manifest
    assert "dropped_duplicates: 3" in manifest


def test_a_headline_without_the_next_bar_is_not_a_sample(tmp_path):
    events = [_headline("late", 50, "黑海货船遭袭")]
    summary = build_dataset(
        tmp_path,
        events,
        _prices(),
        name="gold_macro_v1",
        start=_utc(0, 0),
        end=_utc(12, 0),
    )
    assert summary["samples"] == 0
    assert summary["rejected"] == 1
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8") == ""
