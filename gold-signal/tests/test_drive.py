from datetime import datetime, timedelta, timezone

from gold_signal.compiler import compile_idea
from gold_signal.execution import execution_costs
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.service import create_from_idea, run_backtest
from gold_signal.persistence.store import ProductStore
from gold_signal.research import compare_reports, replay_report, sweep_reaction_threshold


def _utc(minute: int, day: int = 24) -> datetime:
    return datetime(2026, 3, day, 14, minute, tzinfo=timezone.utc)


def _print(price: float, minute: int) -> Observation:
    stamp = _utc(minute)
    return Observation("market.XAUUSD.close", price, stamp, stamp, stamp, "fixture", "XAUUSD")


def test_each_bar_can_decide_without_a_news_event():
    compiled = compile_idea("每天收盘如果黄金一分钟上涨，我做多黄金。")
    assert compiled.ok and compiled.spec is not None
    assert compiled.spec.drive == "BAR"
    prices = [100.0, 101.0, 101.2, 101.2, 101.2, 101.2, 101.3, 101.3]
    observations = [_print(price, 30 + index) for index, price in enumerate(prices)]
    report = replay_report(compiled.spec, "2026-03-01", "2026-03-31", [], observations)
    assert report["run_status"] == "COMPLETED"
    assert report["trades"] >= 1
    assert report["portfolio"]["executed"] == 1
    assert report["portfolio"]["skipped"] >= 1
    filled = next(trace for trace in report["traces"] if trace["side"] == "LONG")
    assert filled["reactions"][0]["passed"] is True
    assert filled["reactions"][0]["span_seconds"] == 60
    assert filled["exit_reason"] == "MAX_HOLD"


def test_a_daily_step_is_not_labeled_as_sixty_seconds():
    compiled = compile_idea("每天收盘如果黄金一分钟上涨，我做多黄金。")
    assert compiled.spec is not None
    first = datetime(2026, 3, 2, tzinfo=timezone.utc)
    second = first + timedelta(days=1)
    observations = [
        Observation("market.XAUUSD.close", 100.0, first, first, first, "fixture", "XAUUSD"),
        Observation("market.XAUUSD.close", 110.0, second, second, second, "fixture", "XAUUSD"),
    ]
    report = replay_report(compiled.spec, "2026-03-01", "2026-03-31", [], observations)
    reaction = report["traces"][0]["reactions"][0]
    assert reaction["span_seconds"] == 86400
    assert reaction["passed"] is True


def test_macro_releases_pay_a_wider_spread_than_a_quiet_bar():
    news = EventRecord("nfp", "NFP_ABOVE_EXPECTATION", _utc(30), _utc(30), _utc(30), "非农高于预期", "非农高于预期", "jin10")
    wide = execution_costs(news, 0.01)
    quiet = execution_costs(None, None)
    assert wide.spread_cost > quiet.spread_cost
    assert wide.slippage_cost > quiet.slippage_cost


def test_window_sweep_and_compare_use_the_same_lifecycle():
    compiled = compile_idea("如果非农高于预期，而且黄金一分钟下跌，我做空黄金。")
    assert compiled.spec is not None
    published = _utc(30)
    later = published + timedelta(days=10)
    news = EventRecord("nfp", "NFP_ABOVE_EXPECTATION", published, published, published, "非农高于预期", "非农高于预期", "fixture")
    other = EventRecord("later", "NFP_ABOVE_EXPECTATION", later, later, later, "非农高于预期", "非农高于预期", "fixture")

    def tape(origin: datetime) -> list[Observation]:
        return [
            Observation("market.XAUUSD.close", price, origin + timedelta(minutes=offset), origin + timedelta(minutes=offset), None, "fixture", "XAUUSD")
            for offset, price in ((0, 100.0), (1, 99.0), (6, 98.0))
        ]

    observations = tape(published) + tape(later)
    early = replay_report(compiled.spec, "2026-03-24", "2026-03-25", [news, other], observations)
    both = replay_report(compiled.spec, "2026-03-01", "2026-04-30", [news, other], observations)
    assert early["trades"] == 1
    assert both["trades"] == 2
    rising = compile_idea("如果非农高于预期，而且黄金一分钟上涨，我做多黄金。")
    assert rising.spec is not None
    up = [
        Observation("market.XAUUSD.close", price, published + timedelta(minutes=offset), published + timedelta(minutes=offset), None, "fixture", "XAUUSD")
        for offset, price in ((0, 100.0), (1, 101.0), (6, 102.0))
    ]
    swept = sweep_reaction_threshold(rising.spec, [news], up, (0.0008, 0.05))
    assert swept[0]["trades"] == 1
    assert swept[1]["trades"] == 0
    compared = compare_reports(early, both)
    assert compared["left"]["trades"] == 1
    assert compared["right"]["trades"] == 2


def test_api_window_can_exclude_the_packaged_event(tmp_path):
    store = ProductStore(tmp_path / "window.sqlite")
    agent, _version = create_from_idea(store, "如果非农低于预期，黄金和白银一分钟上涨，我做多黄金。")
    missed = run_backtest(store, agent.id, start="2026-09-05", end="2026-09-06")
    assert missed["report"]["run_status"] == "FAILED"
    assert missed["report"]["evidence_status"] == "NO_DATA"
    store.close()
