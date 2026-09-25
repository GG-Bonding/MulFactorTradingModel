import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gold_signal.hypothesis import historical_validation, load_hypothesis
from gold_signal.market import snapshot
from gold_signal.models import Bar, FlashNews
from gold_signal.news import CLASSIFIER_VERSION
from gold_signal.observation import EventRecord, Observation, Recorder, record_live_tick
from gold_signal.replay_clock import CostModel
from gold_signal.research import (
    decide_at,
    edge_distribution,
    observations_from_bars,
    replay_event,
    replay_report,
)
from gold_signal.observation import FactorResolver


def _utc(hour: int, minute: int, second: int = 0, day: int = 24) -> datetime:
    return datetime(2026, 3, day, hour, minute, second, tzinfo=timezone.utc)


def _print(factor: str, value: float, stamp: datetime) -> Observation:
    symbol = factor.split(".")[1]
    return Observation(factor, value, stamp, stamp, stamp, "fixture", symbol)


def _event(published: datetime, text: str = "美国8月非农就业人数低于预期") -> EventRecord:
    return EventRecord(
        event_id=f"news-{published.isoformat()}",
        event_type="BULLISH",
        published_at=published,
        available_at=published,
        ingested_at=published,
        title=text,
        content=text,
        source="fixture",
    )


def _path(published: datetime) -> list[Observation]:
    """Close-stamped minutes. 14:30 is the anchor. 14:31 is the first knowable reaction."""
    rows: list[Observation] = []
    # Minute 0 is the news print. Entry is minute 1. Horizons are measured from that fill.
    gold = {0: 3800.0, 1: 3810.0, 2: 3795.0, 6: 3830.0, 16: 3840.0, 31: 3820.0}
    silver = {0: 40.0, 1: 40.2}
    euro = {0: 1.10, 1: 1.102}
    for minute in range(0, 32):
        stamp = published + timedelta(minutes=minute)
        rows.append(_print("market.XAUUSD.close", gold.get(minute, 3810.0), stamp))
        rows.append(_print("market.XAGUSD.close", silver.get(minute, 40.2), stamp))
        rows.append(_print("market.EURUSD.close", euro.get(minute, 1.102), stamp))
    return rows


def test_reaction_is_waiting_until_the_minute_print_exists():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    published = _utc(14, 30)
    event = _event(published)
    resolver = FactorResolver(_path(published))
    early = decide_at(spec, event, resolver, published + timedelta(seconds=59))
    assert early.side == "WAITING"
    assert early.entry_price is None
    decision, outcomes = replay_event(spec, event, _path(published))
    assert decision.side == "LONG"
    assert decision.entry_at == published + timedelta(minutes=1)
    assert decision.entry_price == 3810.0
    assert decision.entry_price != 3800.0
    assert "5m" in outcomes
    gross = 3830.0 / 3810.0 - 1
    anchor_move = 3830.0 / 3800.0 - 1
    assert abs(outcomes["5m"].gross_return - gross) < 1e-12
    assert abs(outcomes["5m"].gross_return - anchor_move) > 1e-4
    assert abs(outcomes["15m"].return_from_entry - (3840.0 / 3810.0 - 1)) < 1e-12
    assert abs(outcomes["30m"].return_from_entry - (3820.0 / 3810.0 - 1)) < 1e-12


def test_open_stamped_bar_is_invisible_until_the_close():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    published = _utc(14, 30)
    bars = {
        "XAUUSD": [Bar(ts=published - timedelta(minutes=1), close=3800.0), Bar(ts=published, close=3810.0)],
        "XAGUSD": [Bar(ts=published - timedelta(minutes=1), close=40.0), Bar(ts=published, close=40.2)],
        "EURUSD": [Bar(ts=published - timedelta(minutes=1), close=1.10), Bar(ts=published, close=1.102)],
    }
    observations = []
    for code, series in bars.items():
        observations.extend(
            observations_from_bars(
                f"market.{code}.close",
                series,
                symbol=code,
                source="fixture",
                timestamp_kind="open",
            )
        )
    early = decide_at(spec, _event(published), FactorResolver(observations), published + timedelta(seconds=30))
    assert early.side == "WAITING"
    decision, _outcomes = replay_event(spec, _event(published), observations)
    assert decision.side == "LONG"
    assert decision.entry_price == 3810.0
    assert decision.entry_at == published + timedelta(minutes=1)


def test_net_return_subtracts_the_fixed_cost():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    costs = CostModel(spread_cost=0.0002, slippage_cost=0.0003, commission=0.0001)
    _decision, outcomes = replay_event(spec, _event(_utc(14, 30)), _path(_utc(14, 30)), costs)
    outcome = outcomes["1m"]
    assert outcome.gross_return is not None
    assert abs(outcome.net_return - (outcome.gross_return - costs.total)) < 1e-12
    assert outcome.mfe is not None and outcome.mae is not None
    assert outcome.mae <= outcome.mfe


def test_five_minute_path_keeps_the_adverse_and_favorable_prints():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    _decision, outcomes = replay_event(spec, _event(_utc(14, 30)), _path(_utc(14, 30)))
    outcome = outcomes["5m"]
    adverse = 3795.0 / 3810.0 - 1
    favorable = 3830.0 / 3810.0 - 1
    assert abs(outcome.mae - adverse) < 1e-12
    assert abs(outcome.mfe - favorable) < 1e-12


def test_failed_confirmation_is_flat_with_no_fill():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    published = _utc(14, 30)
    rows = _path(published)
    rows = [
        _print("market.XAGUSD.close", 39.5, published + timedelta(minutes=1)) if row.factor == "market.XAGUSD.close" and row.observed_at == published + timedelta(minutes=1) else row
        for row in rows
    ]
    decision, outcomes = replay_event(spec, _event(published), rows)
    assert decision.side == "FLAT"
    assert decision.entry_price is None
    assert outcomes == {}


def test_short_pnl_is_positive_when_price_falls():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    published = _utc(14, 30)
    text = "美国CPI高于预期"
    rows = []
    for minute, gold in ((0, 3800.0), (1, 3760.0), (6, 3700.0)):
        stamp = published + timedelta(minutes=minute)
        rows.append(_print("market.XAUUSD.close", gold, stamp))
        rows.append(_print("market.XAGUSD.close", 40.0 if minute == 0 else 39.5, stamp))
        rows.append(_print("market.EURUSD.close", 1.10 if minute == 0 else 1.09, stamp))
    decision, outcomes = replay_event(spec, _event(published, text), rows)
    assert decision.side == "SHORT"
    assert decision.entry_price == 3760.0
    assert outcomes["5m"].gross_return is not None
    assert outcomes["5m"].gross_return > 0


def test_missing_confirmation_series_refuses_the_sample():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    published = _utc(14, 30)
    observations = [row for row in _path(published) if not row.factor.startswith("market.EURUSD")]
    report = replay_report(spec, "2026-01-01", "2026-09-01", [_event(published)], observations)
    assert report["status"] == "INSUFFICIENT"
    assert report["samples"] == 0
    assert report["win_rate"] is None
    assert any("EURUSD.reaction_1m" in item for item in report["missing"])


def test_replay_report_is_deterministic_and_splits_windows():
    spec = load_hypothesis(Path(__file__).parents[1] / "strategies" / "gold_v0.yaml")
    strategy = Path(__file__).parents[1] / "strategies" / "gold_v0.yaml"
    published = _utc(14, 30)
    events = [_event(published)]
    observations = _path(published)
    first = replay_report(spec, "2024-01-01", "2026-09-30", events, observations, strategy_path=strategy)
    second = replay_report(spec, "2024-01-01", "2026-09-30", events, observations, strategy_path=strategy)
    assert first == second
    assert first["status"] == "OK"
    assert first["samples"] == 1
    assert first["counts"]["long"] == 1
    assert first["yaml_sha256"] == hashlib.sha256(strategy.read_bytes()).hexdigest()
    assert first["windows"]["oos"]["samples"] == 1
    assert first["windows"]["train"]["samples"] == 0
    assert first["windows"]["validation"]["status"] == "INSUFFICIENT"
    assert first["net"]["expectancy"] == first["net"]["avg_return"]
    assert first["net"]["avg_return"] < first["by_horizon"]["5m"]["gross_avg_return"]


def test_edge_distribution_matches_win_loss_expectancy():
    from gold_signal.replay_clock import TradeOutcome

    rows = [
        TradeOutcome(0.0032, gross_return=0.0032, net_return=0.0032, mfe=0.004, mae=-0.001),
        TradeOutcome(-0.0018, gross_return=-0.0018, net_return=-0.0018, mfe=0.0002, mae=-0.002),
    ]
    stats = edge_distribution(rows)
    assert abs(stats["expectancy"] - (0.5 * 0.0032 - 0.5 * 0.0018)) < 1e-12
    assert abs(stats["profit_factor"] - (0.0032 / 0.0018)) < 1e-12
    assert abs(stats["median_return"] - 0.0007) < 1e-12
    assert stats["p25"] < stats["p75"]


def test_archive_absent_still_refuses_to_invent_a_win_rate():
    strategy = Path(__file__).parents[1] / "strategies" / "iran_gold.yaml"
    spec = load_hypothesis(strategy)
    report = historical_validation(spec, "2025-01-01", "2026-09-01", strategy_path=strategy)
    assert report["status"] == "INSUFFICIENT"
    assert report["samples"] == 0
    joined = " ".join(report["missing"])
    assert "news archive" in joined
    assert "DFII10" in joined
    assert report["yaml_sha256"] == hashlib.sha256(strategy.read_bytes()).hexdigest()
    assert report["windows"]["train"]["status"] == "INSUFFICIENT"
    assert report["windows"]["oos"]["samples"] == 0


def test_recorder_writes_confirmation_legs_and_event_body(tmp_path: Path):
    now = _utc(14, 30)
    market = snapshot(
        [Bar(ts=now, close=3800.0), Bar(ts=now, close=3801.0)],
        [Bar(ts=now, close=40.0), Bar(ts=now, close=40.1)],
        [Bar(ts=now, close=1.1), Bar(ts=now, close=1.101)],
        as_of=now,
    )
    news = FlashNews(event_id="flash-1", title="非农低于预期", content="公布值:11.4 预期:16.5", published_at=now)
    path = tmp_path / "live.jsonl"
    record_live_tick(
        Recorder(path),
        market,
        news,
        event_type="BULLISH",
        classifier_version=CLASSIFIER_VERSION,
        ingested_at=now,
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    factors = {row["factor"] for row in rows if row["kind"] == "observation"}
    assert factors == {"market.XAUUSD.close", "market.XAGUSD.close", "market.EURUSD.close"}
    event = next(row for row in rows if row["kind"] == "event")
    assert event["content"] == news.content
    assert event["event_type"] == "BULLISH"
    assert event["classifier_version"] == CLASSIFIER_VERSION
