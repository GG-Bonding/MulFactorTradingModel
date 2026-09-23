from datetime import datetime, timezone

from gold_signal.europe import classify_europe, parse_cnbc_quotes, verdict_to_record
from gold_signal.models import EuropeSnapshot, YieldPoint


def _yp(code: str, last: float, change_bp: float | None = None) -> YieldPoint:
    return YieldPoint(
        code=code,
        name=code,
        yield_pct=last,
        change_bp=change_bp,
        ts=datetime(2026, 9, 19, 15, 10, tzinfo=timezone.utc),
        source="CNBC quote API",
    )


def _snap(**kwargs) -> EuropeSnapshot:
    oat = kwargs.pop("oat", _yp("FR10Y-FR", 4.553, 6.0))
    bund = kwargs.pop("bund", _yp("DE10Y-DE", 3.57, 4.0))
    italy = kwargs.pop("italy", _yp("IT10Y-IT", 4.40, 3.0))
    oat_bund = None if oat is None or bund is None else (oat.yield_pct - bund.yield_pct) * 100
    italy_bund = None if italy is None or bund is None else (italy.yield_pct - bund.yield_pct) * 100
    if oat_bund is not None:
        oat_bund = round(oat_bund, 1)
    if italy_bund is not None:
        italy_bund = round(italy_bund, 1)
    defaults = dict(
        as_of=datetime(2026, 9, 19, 23, 10, tzinfo=timezone.utc),
        oat=oat,
        bund=bund,
        italy=italy,
        gilt=_yp("GB10Y-GB", 5.30, 2.0),
        oat_bund_bp=oat_bund,
        italy_bund_bp=italy_bund,
        eurusd=1.17,
        gbpusd=1.32,
        eurgbp=0.886,
        eurusd_1d=-0.001,
        gbpusd_1d=0.0005,
        eurgbp_1d=0.0004,
        cac40=7800.0,
        cac40_1d=-0.004,
        french_banks_1d=-0.003,
        banks_vs_cac_1d=0.001,
        bank_names=["BNP.PA", "GLE.PA", "ACA.PA"],
        sources={"oat": "CNBC quote API", "bund": "CNBC quote API"},
        missing=[],
    )
    defaults.update(kwargs)
    return EuropeSnapshot(**defaults)


def test_missing_yields_are_insufficient_not_a_gold_sell():
    verdict = classify_europe(_snap(oat=None, bund=None, oat_bund_bp=None, italy_bund_bp=None))
    assert verdict.stage == "INSUFFICIENT"
    assert verdict.gold == "UNKNOWN"
    assert "BUY" not in (verdict.gold, verdict.eur)
    assert "SELL" not in (verdict.gold, verdict.eur)
    assert any("missing" in item.lower() or "OAT" in item for item in verdict.not_proven)


def test_98bp_is_france_repricing_not_fragmentation():
    # 4.553 - 3.57 = 98.3bp: yellow, not 2012 red.
    verdict = classify_europe(_snap())
    assert round(verdict.snapshot.oat_bund_bp, 1) == 98.3
    assert verdict.stage == "FRANCE_REPRICING"
    assert verdict.driver == "FRANCE_CREDIT"
    assert verdict.eur == "BEARISH"
    assert verdict.french_domestic == "BEARISH"
    assert verdict.french_banks == "WATCH"
    assert verdict.french_exporters == "EUR_TRANSLATION_TAILWIND"
    assert verdict.gold == "USD_BID_MIXED"
    assert verdict.gold not in ("BUY", "SELL")
    joined = " ".join(verdict.not_proven)
    assert "financial stress" in joined
    assert "fragmentation" in joined
    assert "EUR/GBP" in joined
    assert any("98.3bp" in item or "98.3" in item for item in verdict.evidence)


def test_orange_spread_and_banks_lagging_is_france_stress():
    oat = _yp("FR10Y-FR", 4.57, 8.0)
    bund = _yp("DE10Y-DE", 3.52, 2.0)
    verdict = classify_europe(
        _snap(
            oat=oat,
            bund=bund,
            banks_vs_cac_1d=-0.004,
            french_banks_1d=-0.009,
            cac40_1d=-0.005,
            eurgbp_1d=0.0001,
        )
    )
    assert verdict.snapshot.oat_bund_bp == 105.0
    assert verdict.stage == "FRANCE_STRESS"
    assert verdict.french_banks == "BEARISH"
    assert verdict.french_domestic == "BEARISH"
    assert verdict.eur == "BEARISH"
    assert verdict.gold == "USD_BID_MIXED"
    assert any("fragmentation is not proven" in item for item in verdict.not_proven)


def test_fragmentation_needs_italy_banks_and_eurgbp_together():
    oat = _yp("FR10Y-FR", 5.20, 20.0)
    bund = _yp("DE10Y-DE", 3.50, 4.0)
    italy = _yp("IT10Y-IT", 6.10, 25.0)
    verdict = classify_europe(
        _snap(
            oat=oat,
            bund=bund,
            italy=italy,
            banks_vs_cac_1d=-0.010,
            french_banks_1d=-0.018,
            cac40_1d=-0.008,
            eurgbp_1d=-0.003,
        )
    )
    assert verdict.snapshot.oat_bund_bp == 170.0
    assert verdict.snapshot.italy_bund_bp == 260.0
    assert verdict.stage == "EZ_FRAGMENTATION"
    assert verdict.eur == "BEARISH"
    assert verdict.gbp_vs_eur == "MIXED"
    assert verdict.gbp_vs_usd == "BEARISH"
    assert verdict.gold == "MIXED_HAVEN_VS_REAL_RATES"


def test_wide_oat_bund_alone_is_not_fragmentation():
    oat = _yp("FR10Y-FR", 5.20, 20.0)
    bund = _yp("DE10Y-DE", 3.50, 4.0)
    italy = _yp("IT10Y-IT", 4.40, 3.0)
    verdict = classify_europe(
        _snap(
            oat=oat,
            bund=bund,
            italy=italy,
            banks_vs_cac_1d=0.001,
            eurgbp_1d=0.0004,
        )
    )
    assert verdict.stage == "FRANCE_REPRICING"
    assert any("fragmentation is not proven" in item for item in verdict.not_proven)


def test_bund_up_spread_contained_is_policy_not_france_credit():
    oat = _yp("FR10Y-FR", 3.90, 8.0)
    bund = _yp("DE10Y-DE", 3.40, 8.0)
    verdict = classify_europe(_snap(oat=oat, bund=bund, italy=_yp("IT10Y-IT", 4.10, 7.0)))
    assert round(verdict.snapshot.oat_bund_bp, 1) == 50.0
    assert verdict.stage == "POLICY_HAWKISH"
    assert verdict.driver == "POLICY_RATE"
    assert verdict.eur == "BULLISH"
    assert verdict.gold == "BEARISH_REAL_RATES"


def test_gbp_is_split_versus_eur_and_usd():
    verdict = classify_europe(_snap())
    assert verdict.gbp_vs_eur == "MIXED"  # gilt already >= 5%
    assert verdict.gbp_vs_usd == "BEARISH"
    cheap_gilt = classify_europe(_snap(gilt=_yp("GB10Y-GB", 4.40, 1.0)))
    assert cheap_gilt.gbp_vs_eur == "BULLISH"
    assert cheap_gilt.gbp_vs_usd == "MIXED"


def test_cac_is_not_treated_as_all_french_stocks():
    verdict = classify_europe(_snap(cac40_1d=-0.02, eurusd_1d=-0.004))
    assert verdict.french_domestic == "BEARISH"
    assert verdict.french_exporters == "EUR_TRANSLATION_TAILWIND"
    assert verdict.french_banks == "WATCH"


def test_cnbc_parser_keeps_source_and_levels():
    points = parse_cnbc_quotes(
        {
            "FormattedQuoteResult": {
                "FormattedQuote": [
                    {"symbol": "FR10Y-FR", "name": "France 10Y", "last": "4.565%", "change": "0.018", "last_time": "2026-09-19T16:15:25.000+0200"},
                    {"symbol": "DE10Y-DE", "name": "Germany 10Y", "last": "3.519", "change": "-0.004", "last_time": "2026-09-19T16:15:33.000+0200"},
                ]
            }
        }
    )
    assert points["FR10Y-FR"].yield_pct == 4.565
    assert abs(points["FR10Y-FR"].change_bp - 1.8) < 1e-9
    assert points["FR10Y-FR"].source == "CNBC quote API"
    assert points["FR10Y-FR"].ts is not None
    spread_bp = (points["FR10Y-FR"].yield_pct - points["DE10Y-DE"].yield_pct) * 100
    assert abs(spread_bp - 104.6) < 0.05


def test_record_keeps_missing_and_does_not_invent_yields():
    verdict = classify_europe(_snap(oat=None, bund=None, oat_bund_bp=None, italy_bund_bp=None, missing=["OAT 10Y"]))
    record = verdict_to_record(verdict)
    assert record["oat_pct"] is None
    assert record["bund_pct"] is None
    assert record["stage"] == "INSUFFICIENT"
    assert record["gold"] == "UNKNOWN"
