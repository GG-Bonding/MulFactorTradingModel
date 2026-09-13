from datetime import datetime, timedelta, timezone

from gold_signal.jin10 import parse_flash_list, parse_kline, parse_quote
from gold_signal.market import asset_move, compute_returns, normalized_move, rolling_std_1m
from gold_signal.models import Bar, Thresholds


def _bars(closes: list[float], end: datetime | None = None) -> list[Bar]:
    end = end or datetime(2026, 9, 13, 14, 35, tzinfo=timezone.utc)
    start = end - timedelta(minutes=len(closes) - 1)
    return [Bar(ts=start + timedelta(minutes=i), close=c) for i, c in enumerate(closes)]


def test_return_formula():
    closes = [100.0] * 28 + [100.0, 100.10, 100.30]
    bars = _bars(closes)
    rets = compute_returns(bars)
    assert abs(rets["return_1m"] - (100.30 / 100.10 - 1)) < 1e-12
    assert abs(rets["return_3m"] - (100.30 / 100.0 - 1)) < 1e-12
    assert abs(rets["return_5m"] - (100.30 / 100.0 - 1)) < 1e-12


def test_normalized_move_and_obvious():
    quiet = [3600.0 + (i % 2) * 0.02 for i in range(26)]
    closes = quiet + [3600.0, 3603.0, 3606.0, 3612.0]
    move = asset_move("XAUUSD", _bars(closes))
    assert move.return_1m > Thresholds.RETURN_1M
    assert move.obvious_1m == 1
    assert move.obvious_3m == 1
    std = rolling_std_1m(_bars(closes))
    assert std > 0
    assert normalized_move(move.return_1m, std) > Thresholds.Z_OBVIOUS


def test_down_move_is_obvious_negative():
    quiet = [3600.0] * 26
    closes = quiet + [3600.0, 3597.0, 3594.0, 3588.0]
    move = asset_move("XAUUSD", _bars(closes))
    assert move.obvious_1m == -1
    assert move.obvious_3m == -1


def test_parse_quote_and_kline_and_flash():
    quote = parse_quote({"data": {"code": "XAUUSD", "close": 3642.1, "time": "2026-09-13 14:35:12"}}, "XAUUSD")
    assert quote.price == 3642.1
    bars = parse_kline(
        {
            "data": {
                "items": [
                    {"time": "2026-09-13 14:33:00", "close": 3640.0},
                    {"time": "2026-09-13 14:34:00", "close": 3641.0},
                    {"time": "2026-09-13 14:35:00", "close": 3642.1},
                ]
            }
        }
    )
    assert len(bars) == 3
    assert bars[-1].close == 3642.1
    news = parse_flash_list(
        {
            "data": {
                "items": [
                    {
                        "id": "abc",
                        "time": "2026-09-13 14:34:00",
                        "title": "美国8月非农就业人数低于预期",
                        "content": "美国8月非农就业人数低于预期",
                    }
                ]
            }
        }
    )
    assert news[0].event_id == "abc"
    assert "非农" in news[0].text


def test_parse_binance_klines():
    from gold_signal.market import parse_binance_klines

    rows = [
        [1789290780000, "76800", "76850", "76750", "76827.53"],
        [1789290840000, "76827.53", "76840", "76790", "76810.00"],
    ]
    bars = parse_binance_klines(rows, "BTCUSDT")
    assert len(bars) == 2
    assert bars[-1].close == 76810.0
    rets = compute_returns(bars)
    assert abs(rets["return_1m"] - (76810.0 / 76827.53 - 1)) < 1e-12
