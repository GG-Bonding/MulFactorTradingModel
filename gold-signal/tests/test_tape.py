from datetime import datetime, timezone

from gold_signal.jin10 import Jin10Error
from gold_signal.models import Quote
from gold_signal.tape import BOARD, GROUP_ORDER, fetch_tape, ups_percent_to_fraction


def test_board_has_nasdaq_oil_fx_and_us_shares():
    groups = {group for _code, _name, group, _source in BOARD}
    assert groups == set(GROUP_ORDER)
    codes = {code for code, *_rest in BOARD}
    assert {"NQ=F", "USOIL", "UKOIL", "USDJPY", "USDCNH", "NVDA", "AAPL", "TSLA"} <= codes
    assert all(source == "jin10" for code, _n, group, source in BOARD if group in ("oil", "fx"))
    assert all(source == "yahoo" for code, _n, group, source in BOARD if group in ("nasdaq", "us_equity"))


def test_jin10_ups_percent_is_percent_points():
    assert abs(ups_percent_to_fraction("-0.307") + 0.00307) < 1e-9
    assert abs(ups_percent_to_fraction("0.38") - 0.0038) < 1e-9
    assert ups_percent_to_fraction(None) is None


class _FakeJin10:
    def get_quote(self, code: str) -> Quote:
        if code == "USOIL":
            return Quote(
                code=code,
                price=89.207,
                ts=datetime(2026, 9, 23, 7, 34, tzinfo=timezone.utc),
                raw={"name": "WTI原油", "ups_percent": "-0.307"},
            )
        raise Jin10Error(f"no quote {code}")


def test_missing_quote_stays_missing(monkeypatch):
    monkeypatch.setattr("gold_signal.tape._fetch_yahoo", lambda spec, symbol: _missing(spec))
    rows = fetch_tape(_FakeJin10())  # type: ignore[arg-type]
    by_code = {row.code: row for row in rows}
    assert by_code["USOIL"].price == 89.207
    assert by_code["USOIL"].source == "jin10:USOIL"
    assert abs(by_code["USOIL"].change_1d + 0.00307) < 1e-9
    assert by_code["UKOIL"].price is None
    assert by_code["UKOIL"].missing
    assert by_code["NQ=F"].price is None
    assert by_code["NVDA"].price is None


def _missing(spec):
    from gold_signal.tape import missing_quote

    return missing_quote(spec, "yahoo down")
