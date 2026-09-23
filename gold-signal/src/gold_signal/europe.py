from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from gold_signal.jin10 import SHANGHAI, Jin10Client, Jin10Error
from gold_signal.models import EuropeSnapshot, EuropeVerdict, Thresholds, YieldPoint

CNBC_QUOTE_URL = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRENCH_BANKS = ("BNP.PA", "GLE.PA", "ACA.PA")
_UA = {"User-Agent": "Mozilla/5.0 gold-signal/0.2"}


def classify_europe(snap: EuropeSnapshot) -> EuropeVerdict:
    """Rule-based regime. Missing inputs go into not_proven; they never become facts."""
    evidence: list[str] = []
    not_proven: list[str] = []
    T = Thresholds

    if snap.oat is None or snap.bund is None or snap.oat_bund_bp is None:
        return EuropeVerdict(
            stage="INSUFFICIENT",
            driver="MISSING_YIELDS",
            eur="UNKNOWN",
            gbp_vs_eur="UNKNOWN",
            gbp_vs_usd="UNKNOWN",
            french_domestic="UNKNOWN",
            french_banks="UNKNOWN",
            french_exporters="UNKNOWN",
            gold="UNKNOWN",
            evidence=[],
            not_proven=["OAT 10Y or Bund 10Y missing — cannot split policy rate vs France credit"],
            snapshot=snap,
        )

    spread = snap.oat_bund_bp
    evidence.append(
        f"OAT 10Y {snap.oat.yield_pct:.3f}% vs Bund 10Y {snap.bund.yield_pct:.3f}% "
        f"= {spread:.1f}bp ({snap.oat.source} / {snap.bund.source})"
    )

    bund_move = snap.bund.change_bp
    oat_move = snap.oat.change_bp
    policy_like = (
        bund_move is not None
        and bund_move >= T.POLICY_BUND_MOVE_BP
        and (oat_move is None or abs((oat_move or 0) - bund_move) < 5)
    )
    oat_outpaced = oat_move is not None and bund_move is not None and oat_move - bund_move >= 3
    if spread >= T.OAT_BUND_YELLOW_BP or oat_outpaced:
        driver = "FRANCE_CREDIT"
        evidence.append(
            "OAT-Bund is elevated or OAT outpaced Bund — France-specific risk premium, not an ECB policy-rate move"
        )
    elif policy_like:
        driver = "POLICY_RATE"
        evidence.append("Bund and OAT moved together with a contained spread — closer to ECB/policy rate than France credit")
    else:
        driver = "MIXED"
        not_proven.append("Cannot cleanly split policy vs France credit from today's moves")

    banks_lag = snap.banks_vs_cac_1d is not None and snap.banks_vs_cac_1d <= T.BANKS_LAGGING_1D
    banks_stress = snap.banks_vs_cac_1d is not None and snap.banks_vs_cac_1d <= T.BANKS_STRESS_1D
    eurgbp_break = snap.eurgbp_1d is not None and snap.eurgbp_1d <= T.EURGBP_STRESS_1D
    italy_hot = snap.italy_bund_bp is not None and snap.italy_bund_bp >= T.ITALY_BUND_ORANGE_BP
    italy_red = snap.italy_bund_bp is not None and snap.italy_bund_bp >= T.ITALY_BUND_RED_BP

    if snap.banks_vs_cac_1d is None:
        not_proven.append("French banks vs CAC 1d missing")
    else:
        evidence.append(f"French banks vs CAC 1d {snap.banks_vs_cac_1d:+.2%}")
    if snap.eurgbp_1d is None:
        not_proven.append("EUR/GBP 1d missing")
    else:
        evidence.append(f"EUR/GBP 1d {snap.eurgbp_1d:+.2%}")
    if snap.italy_bund_bp is None:
        not_proven.append("Italy-Bund spread missing")
    else:
        evidence.append(f"Italy-Bund {snap.italy_bund_bp:.1f}bp")

    stage = "CONTAINED"
    if spread >= T.OAT_BUND_RED_BP and banks_stress and eurgbp_break and italy_red:
        stage = "EZ_FRAGMENTATION"
        evidence.append("OAT-Bund, French banks, EUR/GBP and Italy-Bund all deteriorated together")
    elif spread >= T.OAT_BUND_ORANGE_BP and banks_lag:
        stage = "FRANCE_STRESS"
        evidence.append("OAT-Bund orange and French banks lagging CAC — sovereign-bank loop risk, not just budget optics")
        if snap.italy_bund_bp is None:
            not_proven.append("Italy-Bund missing — Eurozone fragmentation is not proven")
        elif not italy_hot:
            not_proven.append("Italy-Bund is not in the orange zone, so Eurozone fragmentation is not proven")
        if snap.eurgbp_1d is None:
            not_proven.append("EUR/GBP 1d missing — EUR-to-GBP capital flight is not proven")
        elif not eurgbp_break:
            not_proven.append("EUR/GBP is not breaking down, so this is France stress, not a proven EUR-to-GBP flight")
    elif spread >= T.OAT_BUND_YELLOW_BP:
        stage = "FRANCE_REPRICING"
        if snap.banks_vs_cac_1d is None:
            not_proven.append("French banks vs CAC missing — cannot upgrade to France financial stress")
        elif not banks_lag:
            not_proven.append("French banks are not clearly lagging CAC, so this is not yet France financial stress")
        if snap.italy_bund_bp is None:
            not_proven.append("Italy-Bund missing — Eurozone fragmentation is not proven")
        elif not italy_hot:
            not_proven.append("Italy-Bund is not in the orange zone, so Eurozone fragmentation is not proven")
        if snap.eurgbp_1d is None:
            not_proven.append("EUR/GBP 1d missing — EUR-to-GBP capital flight is not proven")
        elif not eurgbp_break:
            not_proven.append("EUR/GBP is not breaking down, so capital is not clearly leaving EUR for GBP")
    elif driver == "POLICY_RATE":
        stage = "POLICY_HAWKISH"

    eur = _eur_bias(stage, driver)
    gbp_vs_eur, gbp_vs_usd = _gbp_bias(stage, snap)
    domestic, banks, exporters = _equity_bias(stage, snap)
    gold = _gold_bias(stage, driver)
    stale = _stale_yield_note(snap)
    if stale:
        not_proven.append(stale)

    return EuropeVerdict(
        stage=stage,
        driver=driver,
        eur=eur,
        gbp_vs_eur=gbp_vs_eur,
        gbp_vs_usd=gbp_vs_usd,
        french_domestic=domestic,
        french_banks=banks,
        french_exporters=exporters,
        gold=gold,
        evidence=evidence,
        not_proven=not_proven,
        snapshot=snap,
    )


def _eur_bias(stage: str, driver: str) -> str:
    if stage in ("FRANCE_STRESS", "EZ_FRAGMENTATION"):
        return "BEARISH"
    if stage == "FRANCE_REPRICING":
        return "BEARISH"
    if stage == "POLICY_HAWKISH" or driver == "POLICY_RATE":
        return "BULLISH"
    return "MIXED"


def _gbp_bias(stage: str, snap: EuropeSnapshot) -> tuple[str, str]:
    gilt_hot = snap.gilt is not None and snap.gilt.yield_pct >= 5.0
    if stage in ("FRANCE_REPRICING", "FRANCE_STRESS"):
        vs_eur = "BULLISH"
        vs_usd = "BEARISH" if stage == "FRANCE_STRESS" else "MIXED"
        if gilt_hot:
            vs_eur = "MIXED"
            vs_usd = "BEARISH"
        return vs_eur, vs_usd
    if stage == "EZ_FRAGMENTATION":
        return "MIXED", "BEARISH"
    return "MIXED", "MIXED"


def _equity_bias(stage: str, snap: EuropeSnapshot) -> tuple[str, str, str]:
    if stage == "EZ_FRAGMENTATION":
        return "BEARISH", "BEARISH", "BEARISH"
    if stage == "FRANCE_STRESS":
        exporters = "MIXED"
        if snap.eurusd_1d is not None and snap.eurusd_1d < 0:
            exporters = "EUR_TRANSLATION_TAILWIND"
        return "BEARISH", "BEARISH", exporters
    if stage == "FRANCE_REPRICING":
        exporters = "MIXED"
        if snap.eurusd_1d is not None and snap.eurusd_1d < 0:
            exporters = "EUR_TRANSLATION_TAILWIND"
        return "BEARISH", "WATCH", exporters
    return "MIXED", "MIXED", "MIXED"


def _gold_bias(stage: str, driver: str) -> str:
    if stage == "EZ_FRAGMENTATION":
        return "MIXED_HAVEN_VS_REAL_RATES"
    if driver == "POLICY_RATE":
        return "BEARISH_REAL_RATES"
    if stage in ("FRANCE_REPRICING", "FRANCE_STRESS"):
        return "USD_BID_MIXED"
    return "NO_CALL"


def _stale_yield_note(snap: EuropeSnapshot) -> str | None:
    stamps = [p.ts for p in (snap.oat, snap.bund, snap.italy, snap.gilt) if p is not None and p.ts is not None]
    if not stamps:
        return None
    newest = max(stamps)
    age_hours = (snap.as_of - newest).total_seconds() / 3600
    if age_hours >= 12:
        return (
            f"Yield last print {newest.isoformat()} is {age_hours:.1f}h old — "
            "this is a last exchange print, not a live session move"
        )
    return None


def fetch_europe_snapshot(jin10: Jin10Client | None = None) -> EuropeSnapshot:
    missing: list[str] = []
    sources: dict[str, str] = {}
    as_of = datetime.now(tz=SHANGHAI)

    try:
        yields = _fetch_cnbc_yields()
    except RuntimeError as exc:
        missing.append(f"CNBC yields ({exc})")
        yields = {}
    oat = yields.get("FR10Y-FR")
    bund = yields.get("DE10Y-DE")
    italy = yields.get("IT10Y-IT")
    gilt = yields.get("GB10Y-GB")
    if oat:
        sources["oat"] = oat.source
    else:
        missing.append("OAT 10Y (CNBC FR10Y-FR)")
    if bund:
        sources["bund"] = bund.source
    else:
        missing.append("Bund 10Y (CNBC DE10Y-DE)")
    if italy:
        sources["italy"] = italy.source
    else:
        missing.append("Italy 10Y (CNBC IT10Y-IT)")
    if gilt:
        sources["gilt"] = gilt.source
    else:
        missing.append("Gilt 10Y (CNBC GB10Y-GB)")

    oat_bund = round((oat.yield_pct - bund.yield_pct) * 100, 1) if oat and bund else None
    italy_bund = round((italy.yield_pct - bund.yield_pct) * 100, 1) if italy and bund else None

    fx = _fetch_fx(jin10)
    for key in ("eurusd", "gbpusd", "eurgbp"):
        if fx.get(key) is None:
            missing.append(key.upper())
        elif fx.get(f"{key}_source"):
            sources[key] = fx[f"{key}_source"]

    cac = _fetch_yahoo_move("^FCHI")
    if cac[0] is None:
        if jin10 is not None:
            try:
                q = jin10.get_quote("FCHI")
                cac = (q.price, None, "jin10:FCHI")
            except Jin10Error:
                missing.append("CAC40")
        else:
            missing.append("CAC40")
    else:
        sources["cac40"] = cac[2]

    bank_rets: list[float] = []
    bank_ok: list[str] = []
    for code in FRENCH_BANKS:
        price, ret, src = _fetch_yahoo_move(code)
        if ret is None:
            missing.append(code)
            continue
        bank_rets.append(ret)
        bank_ok.append(code)
        sources[code] = src
    banks_1d = sum(bank_rets) / len(bank_rets) if bank_rets else None
    banks_vs_cac = None
    if banks_1d is not None and cac[1] is not None:
        banks_vs_cac = banks_1d - cac[1]
        sources["banks_vs_cac"] = "yahoo equal-weight BNP/GLE/ACA minus CAC"

    return EuropeSnapshot(
        as_of=as_of,
        oat=oat,
        bund=bund,
        italy=italy,
        gilt=gilt,
        oat_bund_bp=oat_bund,
        italy_bund_bp=italy_bund,
        eurusd=fx.get("eurusd"),
        gbpusd=fx.get("gbpusd"),
        eurgbp=fx.get("eurgbp"),
        eurusd_1d=fx.get("eurusd_1d"),
        gbpusd_1d=fx.get("gbpusd_1d"),
        eurgbp_1d=fx.get("eurgbp_1d"),
        cac40=cac[0],
        cac40_1d=cac[1],
        french_banks_1d=banks_1d,
        banks_vs_cac_1d=banks_vs_cac,
        bank_names=bank_ok,
        sources=sources,
        missing=missing,
    )


def _fetch_cnbc_yields() -> dict[str, YieldPoint]:
    symbols = "FR10Y-FR|DE10Y-DE|IT10Y-IT|GB10Y-GB"
    payload = _http_json(
        CNBC_QUOTE_URL,
        {
            "symbols": symbols,
            "requestMethod": "quick",
            "exthrs": "1",
            "extMode": "ALL",
            "noform": "1",
            "partnerId": "2",
            "output": "json",
            "fields": "symbol,name,last,change,change_pct,last_time",
        },
    )
    return parse_cnbc_quotes(payload)


def parse_cnbc_quotes(payload: dict[str, Any]) -> dict[str, YieldPoint]:
    quotes = (
        payload.get("FormattedQuoteResult", {}).get("FormattedQuote")
        if isinstance(payload, dict)
        else None
    )
    if isinstance(quotes, dict):
        quotes = [quotes]
    if not isinstance(quotes, list):
        raise RuntimeError(f"CNBC yield payload unexpected: {str(payload)[:300]}")
    out: dict[str, YieldPoint] = {}
    for item in quotes:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "")
        if "," in symbol or "|" in symbol:
            cc = str(item.get("countryCode") or "")
            symbol = {"FR": "FR10Y-FR", "DE": "DE10Y-DE", "IT": "IT10Y-IT", "GB": "GB10Y-GB"}.get(cc, "")
        last = _pct_number(item.get("last"))
        if not symbol or last is None:
            continue
        change = _pct_number(item.get("change"))
        change_bp = change * 100 if change is not None else None
        ts = _cnbc_time(item.get("last_time"))
        out[symbol] = YieldPoint(
            code=symbol,
            name=str(item.get("name") or symbol),
            yield_pct=last,
            change_bp=change_bp,
            ts=ts,
            source="CNBC quote API",
        )
    return out


def append_europe(path: Path, verdict: EuropeVerdict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(verdict_to_record(verdict), ensure_ascii=False) + "\n")


def verdict_to_record(verdict: EuropeVerdict) -> dict[str, Any]:
    snap = verdict.snapshot
    return {
        "as_of": snap.as_of.isoformat(),
        "stage": verdict.stage,
        "driver": verdict.driver,
        "eur": verdict.eur,
        "gbp_vs_eur": verdict.gbp_vs_eur,
        "gbp_vs_usd": verdict.gbp_vs_usd,
        "french_domestic": verdict.french_domestic,
        "french_banks": verdict.french_banks,
        "french_exporters": verdict.french_exporters,
        "gold": verdict.gold,
        "oat_pct": None if snap.oat is None else snap.oat.yield_pct,
        "bund_pct": None if snap.bund is None else snap.bund.yield_pct,
        "italy_pct": None if snap.italy is None else snap.italy.yield_pct,
        "gilt_pct": None if snap.gilt is None else snap.gilt.yield_pct,
        "oat_bund_bp": snap.oat_bund_bp,
        "italy_bund_bp": snap.italy_bund_bp,
        "eurusd": snap.eurusd,
        "gbpusd": snap.gbpusd,
        "eurgbp": snap.eurgbp,
        "eurgbp_1d": snap.eurgbp_1d,
        "cac40_1d": snap.cac40_1d,
        "french_banks_1d": snap.french_banks_1d,
        "banks_vs_cac_1d": snap.banks_vs_cac_1d,
        "sources": snap.sources,
        "missing": snap.missing,
        "evidence": verdict.evidence,
        "not_proven": verdict.not_proven,
    }


def _fetch_fx(jin10: Jin10Client | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if jin10 is not None:
        for code, key in (("EURUSD", "eurusd"), ("GBPUSD", "gbpusd")):
            try:
                q = jin10.get_quote(code)
                out[key] = q.price
                out[f"{key}_source"] = f"jin10:{code}"
            except Jin10Error:
                pass
    eurgbp_px, eurgbp_1d, eurgbp_src = _fetch_yahoo_move("EURGBP=X")
    if eurgbp_px is not None:
        out["eurgbp"] = eurgbp_px
        out["eurgbp_1d"] = eurgbp_1d
        out["eurgbp_source"] = eurgbp_src
    elif out.get("eurusd") and out.get("gbpusd"):
        out["eurgbp"] = out["eurusd"] / out["gbpusd"]
        out["eurgbp_source"] = "jin10 EURUSD/GBPUSD cross"
    usd_eur, usd_eur_1d, _ = _fetch_yahoo_move("EURUSD=X")
    if out.get("eurusd") is None and usd_eur is not None:
        out["eurusd"] = usd_eur
        out["eurusd_1d"] = usd_eur_1d
        out["eurusd_source"] = "yahoo:EURUSD=X"
    elif usd_eur_1d is not None:
        out["eurusd_1d"] = usd_eur_1d
    usd_gbp, usd_gbp_1d, _ = _fetch_yahoo_move("GBPUSD=X")
    if out.get("gbpusd") is None and usd_gbp is not None:
        out["gbpusd"] = usd_gbp
        out["gbpusd_1d"] = usd_gbp_1d
        out["gbpusd_source"] = "yahoo:GBPUSD=X"
    elif usd_gbp_1d is not None:
        out["gbpusd_1d"] = usd_gbp_1d
    return out


def _fetch_yahoo_move(symbol: str) -> tuple[float | None, float | None, str]:
    try:
        payload = _http_json(
            YAHOO_CHART_URL.format(symbol=symbol),
            {"range": "5d", "interval": "1d"},
        )
    except RuntimeError:
        return None, None, ""
    result = (payload.get("chart") or {}).get("result") if isinstance(payload, dict) else None
    if not result:
        return None, None, ""
    meta = result[0].get("meta") or {}
    price = _as_float(meta.get("regularMarketPrice"))
    prev = _as_float(meta.get("chartPreviousClose") or meta.get("previousClose"))
    ret = None
    if price is not None and prev not in (None, 0):
        ret = price / prev - 1
    if price is None:
        closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        nums = [c for c in closes if c is not None]
        if nums:
            price = float(nums[-1])
            if len(nums) >= 2 and nums[-2]:
                ret = price / float(nums[-2]) - 1
    if price is None:
        return None, None, ""
    return price, ret, f"yahoo:{symbol}"


def _http_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx.get(url, params=params, headers=_UA, timeout=25.0, follow_redirects=True)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code} {url}: {response.text[:200]}")
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"request failed {url}: {last_exc}") from last_exc


def _pct_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("%", "").replace(",", "").replace("+", "").strip()
    if not text or text in ("-", "UNCH"):
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group()) if match else None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cnbc_time(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value).replace("Z", "+00:00")
    raw = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(SHANGHAI)
