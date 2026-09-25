import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gold_signal.agent import (
    ExecutionRefused,
    PaperBroker,
    agent_view,
    compare_runs,
    deploy_agent,
    format_agent_view,
    order_from_decision,
    order_from_text,
)
from gold_signal.compiler import compile_idea, compile_model_output
from gold_signal.hypothesis import historical_validation, parse_hypothesis
from gold_signal.main import main
from gold_signal.observation import EventRecord, Observation
from gold_signal.research import replay_event, replay_report

CPI = "如果 CPI 低于预期，美国实际利率下降，而且黄金和白银在一分钟内同步上涨，我认为黄金还能继续涨。"
NFP = "如果非农低于预期，而且黄金和白银在一分钟内同步上涨，我认为黄金还能继续涨。"


def _utc(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 24, hour, minute, tzinfo=timezone.utc)


def _print(factor: str, value: float, stamp: datetime) -> Observation:
    return Observation(factor, value, stamp, stamp, stamp, "fixture", factor.split(".")[1])


def _tape(published: datetime) -> list[Observation]:
    rows = []
    for minute in range(0, 8):
        stamp = published + timedelta(minutes=minute)
        gold = {0: 3800.0, 1: 3810.0, 6: 3830.0}.get(minute, 3810.0)
        silver = 40.0 if minute == 0 else 40.1
        rows.append(_print("market.XAUUSD.close", gold, stamp))
        rows.append(_print("market.XAGUSD.close", silver, stamp))
    return rows


def _news(published: datetime, text: str) -> EventRecord:
    return EventRecord(
        event_id="cpi-1",
        event_type="",
        published_at=published,
        available_at=published,
        ingested_at=published,
        title=text,
        content=text,
        source="fixture",
    )


def test_cpi_sentence_compiles_to_an_editable_spec():
    first = compile_idea(CPI)
    second = compile_idea(CPI)
    assert first.ok and first.spec is not None
    assert first.yaml == second.yaml
    spec = first.spec
    assert spec.asset == "XAUUSD"
    assert spec.entry == "LONG"
    assert spec.trigger == "CPI_BELOW_EXPECTATION"
    assert spec.horizons == ("1m", "3m", "5m", "15m")
    factors = [(item.factor, item.operator, item.value) for item in spec.confirmations]
    assert factors == [
        ("DFII10.change", "<", 0.0),
        ("XAUUSD.reaction_1m", ">", 0.0008),
        ("XAGUSD.reaction_1m", ">", 0.0008),
    ]
    edited = first.yaml.replace("0.0008", "0.05", 1)
    changed = parse_hypothesis(edited)
    assert changed.confirmations[1].value == 0.05
    assert hashlib.sha256(edited.encode()).hexdigest() != hashlib.sha256(first.yaml.encode()).hexdigest()


def test_incomplete_sentence_does_not_become_a_strategy():
    result = compile_idea("今天天气不错")
    assert result.ok is False
    assert result.spec is None
    assert result.yaml == ""
    assert main(["--mode", "compile", "--text", "今天天气不错"]) == 1


def test_model_output_is_rejected_when_the_structure_is_wrong():
    refused = compile_model_output({"asset": "XAUUSD", "entry": "MAYBE", "confirmations": []})
    assert refused.ok is False
    accepted = compile_model_output(
        {
            "asset": "XAUUSD",
            "entry": "LONG",
            "trigger": "CPI_BELOW_EXPECTATION",
            "confirmations": [{"factor": "XAUUSD.reaction_1m", "operator": ">", "value": 0.0008}],
        }
    )
    assert accepted.ok and accepted.spec is not None
    assert accepted.spec.trigger == "CPI_BELOW_EXPECTATION"


def test_compiled_idea_replays_the_same_way_twice_and_refuses_an_unsafe_factor():
    published = _utc(14, 30)
    nfp = compile_idea(NFP)
    assert nfp.spec is not None
    news = _news(published, "美国8月非农就业人数低于预期")
    tape = _tape(published)
    first = replay_report(nfp.spec, "2024-01-01", "2026-09-30", [news], tape)
    second = replay_report(nfp.spec, "2024-01-01", "2026-09-30", [news], tape)
    assert first == second
    assert first["status"] == "OK"
    assert first["samples"] == 1
    assert first["counts"]["long"] == 1
    net = first["net"]
    assert net["win_rate"] == 1
    assert net["avg_return"] is not None
    assert net["median_return"] == net["avg_return"]
    assert net["avg_mfe"] is not None
    assert net["avg_mae"] is not None
    assert "profit_factor" in net
    assert first["costs"]["spread_cost"] > 0
    assert first["windows"]["oos"]["samples"] == 1
    decision, outcomes = replay_event(nfp.spec, news, tape)
    assert decision.side == "LONG"
    assert decision.entry_price == 3810.0
    assert outcomes["5m"].net_return is not None
    assert outcomes["5m"].net_return < outcomes["5m"].gross_return

    cpi = compile_idea(CPI)
    assert cpi.spec is not None
    blocked = historical_validation(cpi.spec, "2024-01-01", "2026-09-30", events=[news], observations=tape)
    assert blocked["status"] == "INSUFFICIENT"
    assert blocked["samples"] == 0
    assert any("DFII10" in item for item in blocked["missing"])


def test_user_can_raise_the_threshold_and_the_same_tape_goes_flat():
    published = _utc(14, 30)
    compiled = compile_idea(NFP)
    assert compiled.spec is not None
    news = _news(published, "美国8月非农就业人数低于预期")
    loose, _outcomes = replay_event(compiled.spec, news, _tape(published))
    strict = parse_hypothesis(compiled.yaml.replace("0.0008", "0.05"))
    blocked, _none = replay_event(strict, news, _tape(published))
    assert loose.side == "LONG"
    assert blocked.side == "FLAT"


def test_agent_card_names_the_conditions_and_compares_paper():
    published = _utc(14, 30)
    compiled = compile_idea(NFP)
    assert compiled.spec is not None
    news = _news(published, "美国8月非农就业人数低于预期")
    report = replay_report(compiled.spec, "2024-01-01", "2026-09-30", [news], _tape(published))
    view = agent_view(compiled.spec, news, _tape(published), report, [0.001, -0.0004])
    assert view["side"] == "LONG"
    assert view["entry"] == 3810.0
    assert all(row["passed"] for row in view["conditions"])
    labels = " ".join(row["label"] for row in view["conditions"])
    assert "非农低于预期" in labels
    assert "XAUUSD.reaction_1m" in labels
    assert "XAGUSD.reaction_1m" in labels
    assert view["comparison"]["historical_samples"] == 1
    assert view["comparison"]["historical_avg_net"] is not None
    assert view["comparison"]["paper_samples"] == 2
    assert view["comparison"]["paper_avg_net"] is not None
    assert view["comparison"]["oos_samples"] == 1
    assert view["comparison"]["oos_avg_net"] == view["comparison"]["historical_avg_net"]
    card = format_agent_view(view)
    assert "当前判断" in card and "LONG XAUUSD" in card
    assert "触发条件" in card and "PASS" in card
    assert "Reason" in card and view["reason"] in card
    compared = compare_runs(report, [0.001])
    assert compared["paper_avg_net"] == 0.001
    assert compared["historical_avg_net"] == view["comparison"]["historical_avg_net"]


def test_execution_accepts_only_a_versioned_engine_decision():
    published = _utc(14, 30)
    compiled = compile_idea(NFP)
    assert compiled.spec is not None and compiled.yaml
    news = _news(published, "美国8月非农就业人数低于预期")
    decision, _outcomes = replay_event(compiled.spec, news, _tape(published))
    order = order_from_decision(decision, compiled.yaml, quantity=1)
    broker = PaperBroker()
    filled = broker.submit(order)
    assert filled["status"] == "PAPER"
    assert filled["spec_sha256"] == hashlib.sha256(compiled.yaml.encode()).hexdigest()
    assert broker.orders == [order]
    with pytest.raises(ExecutionRefused):
        order_from_text("BUY GOLD confidence 87%")
    with pytest.raises(ExecutionRefused):
        broker.submit({"side": "LONG", "entry": 3800})


def test_live_snapshot_explains_the_deployed_spec_without_asking_a_model():
    from gold_signal.agent import view_from_market
    from gold_signal.market import snapshot
    from gold_signal.models import Bar, FlashNews

    published = _utc(14, 30)
    later = published + timedelta(minutes=1)
    market = snapshot(
        [Bar(ts=published, close=3800.0), Bar(ts=later, close=3810.0)],
        [Bar(ts=published, close=40.0), Bar(ts=later, close=40.1)],
        [Bar(ts=published, close=1.1), Bar(ts=later, close=1.101)],
        as_of=later,
    )
    news = FlashNews(
        event_id="n1",
        title="美国8月非农就业人数低于预期",
        content="美国8月非农就业人数低于预期",
        published_at=published,
    )
    compiled = compile_idea(NFP)
    assert compiled.spec is not None
    view = view_from_market(compiled.spec, news, market)
    assert view["side"] == "LONG"
    assert view["entry"] == 3810.0
    assert "LLM" not in view["reason"]


def test_deploy_writes_a_spec_the_user_can_load(tmp_path: Path):
    compiled = compile_idea(CPI)
    assert compiled.spec is not None
    meta = deploy_agent(compiled.spec, tmp_path)
    loaded = parse_hypothesis((tmp_path / "agents" / meta["path"]).read_text(encoding="utf-8"))
    assert loaded.trigger == "CPI_BELOW_EXPECTATION"
    assert loaded.asset == "XAUUSD"
    assert meta["sha256"] == hashlib.sha256(compiled.yaml.encode()).hexdigest()
