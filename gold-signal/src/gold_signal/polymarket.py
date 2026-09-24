from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from gold_signal.models import PolymarketBoard, PolymarketContract

SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SOURCE = "Polymarket gamma API"
CLOB_SOURCE = "Polymarket CLOB websocket"

# Fixed searches. The event that comes back is whatever Polymarket ranks first.
TOPICS: tuple[tuple[str, str], ...] = (
    ("Fed", "Fed decision"),
    ("recession", "US recession"),
    ("inflation", "CPI"),
    ("gold", "gold"),
    ("oil", "crude oil"),
)


def fetch_polymarket_board(limit_per_topic: int = 4) -> PolymarketBoard:
    contracts: list[PolymarketContract] = []
    missing: list[str] = []
    for topic, query in TOPICS:
        try:
            payload = _get(SEARCH_URL, {"q": query})
        except RuntimeError as exc:
            missing.append(f"{topic}: {exc}")
            continue
        rows = contracts_from_search(payload, topic, limit=limit_per_topic, hint=query)
        if not rows:
            missing.append(f"{topic}: no open Polymarket event")
            continue
        contracts.extend(rows)
    return PolymarketBoard(as_of=datetime.now(tz=timezone.utc), contracts=contracts, missing=missing)


def contracts_from_search(
    payload: dict[str, Any],
    topic: str,
    limit: int = 4,
    hint: str = "",
) -> list[PolymarketContract]:
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return []
    event = _pick_event(events, hint)
    if event is None:
        return []
    title = str(event.get("title") or topic)
    markets = [item for item in event.get("markets") or [] if isinstance(item, dict) and not item.get("closed")]
    markets.sort(key=lambda item: _as_float(item.get("volume")) or 0.0, reverse=True)
    rows: list[PolymarketContract] = []
    for market in markets[:limit]:
        yes = _yes_price(market)
        rows.append(
            PolymarketContract(
                topic=topic,
                event=title,
                question=str(market.get("question") or title),
                slug=str(market.get("slug") or ""),
                token_id=_yes_token_id(market),
                yes=yes,
                bid=_as_float(market.get("bestBid")),
                ask=_as_float(market.get("bestAsk")),
                volume=_as_float(market.get("volume")),
                end_date=_as_text(market.get("endDate")),
                updated_at=_as_text(market.get("updatedAt")),
                source=SOURCE,
                missing=None if yes is not None else "Yes price missing",
            )
        )
    return rows


def _pick_event(events: list[Any], hint: str) -> dict[str, Any] | None:
    open_events = [
        item
        for item in events
        if isinstance(item, dict) and not item.get("closed") and item.get("markets")
    ]
    needle = hint.lower()
    if needle:
        for item in open_events:
            if needle in str(item.get("title") or "").lower():
                return item
    return open_events[0] if open_events else None


def _yes_token_id(market: dict[str, Any]) -> str | None:
    outcomes = _json_list(market.get("outcomes"))
    tokens = _json_list(market.get("clobTokenIds"))
    if not outcomes or not tokens or len(outcomes) != len(tokens):
        return None
    for name, token in zip(outcomes, tokens):
        if str(name).strip().lower() == "yes" and token:
            return str(token)
    return None


def apply_market_message(board: PolymarketBoard, payload: Any) -> bool:
    """Update Yes bid/ask from a public CLOB market frame. Unknown tokens are ignored."""
    if isinstance(payload, list):
        return any(apply_market_message(board, item) for item in payload)
    if not isinstance(payload, dict):
        return False
    event = str(payload.get("event_type") or "")
    changed = False
    if event == "book":
        changed = _touch(
            board,
            payload.get("asset_id"),
            bid=_best_level(payload.get("bids"), highest=True),
            ask=_best_level(payload.get("asks"), highest=False),
            stamp=payload.get("timestamp"),
        )
    elif event == "best_bid_ask":
        changed = _touch(
            board,
            payload.get("asset_id"),
            bid=_as_float(payload.get("best_bid")),
            ask=_as_float(payload.get("best_ask")),
            stamp=payload.get("timestamp"),
        )
    elif event == "price_change":
        for change in payload.get("price_changes") or []:
            if not isinstance(change, dict):
                continue
            changed = _touch(
                board,
                change.get("asset_id"),
                bid=_as_float(change.get("best_bid")),
                ask=_as_float(change.get("best_ask")),
                stamp=payload.get("timestamp"),
            ) or changed
    elif event == "last_trade_price":
        changed = _touch(
            board,
            payload.get("asset_id"),
            last=_as_float(payload.get("price")),
            stamp=payload.get("timestamp"),
        )
    if changed:
        board.as_of = datetime.now(tz=timezone.utc)
    return changed


def _touch(
    board: PolymarketBoard,
    token_id: Any,
    *,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    stamp: Any = None,
) -> bool:
    token = str(token_id or "")
    if not token:
        return False
    contract = next((row for row in board.contracts if row.token_id == token), None)
    if contract is None:
        return False
    if bid is not None:
        contract.bid = bid
    if ask is not None:
        contract.ask = ask
    if contract.bid is not None and contract.ask is not None:
        contract.yes = (contract.bid + contract.ask) / 2
    elif last is not None:
        contract.yes = last
    contract.updated_at = _stamp_text(stamp)
    contract.source = CLOB_SOURCE
    contract.missing = None if contract.yes is not None else "Yes price missing"
    return True


def _best_level(levels: Any, *, highest: bool) -> float | None:
    if not isinstance(levels, list):
        return None
    prices = [_as_float(level.get("price")) for level in levels if isinstance(level, dict)]
    prices = [price for price in prices if price is not None]
    if not prices:
        return None
    return max(prices) if highest else min(prices)


def _stamp_text(stamp: Any) -> str:
    if stamp is None or stamp == "":
        return datetime.now(tz=timezone.utc).isoformat()
    try:
        millis = int(str(stamp))
    except ValueError:
        return str(stamp)
    if millis > 10_000_000_000:
        moment = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
    else:
        moment = datetime.fromtimestamp(millis, tz=timezone.utc)
    return moment.isoformat()


class PolymarketFeed:
    """Gamma finds the contracts once. The public CLOB socket then pushes the book."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._board = PolymarketBoard(as_of=datetime.now(tz=timezone.utc))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.updates = 0

    def start(self) -> PolymarketBoard:
        board = fetch_polymarket_board()
        if not any(row.token_id for row in board.contracts):
            board.missing.append("no CLOB token ids — prices stay on the Gamma snapshot")
        with self._lock:
            self._board = board
        self._thread = threading.Thread(target=self._loop, name="polymarket-clob", daemon=True)
        self._thread.start()
        return self.snapshot()

    def snapshot(self) -> PolymarketBoard:
        with self._lock:
            return self._board.model_copy(deep=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._session()
            except Exception as exc:
                self.error = str(exc)
                if self._stop.wait(2):
                    return

    def _session(self) -> None:
        from websockets.sync.client import connect

        ids = [row.token_id for row in self.snapshot().contracts if row.token_id]
        if not ids or self._stop.is_set():
            self._stop.wait(2)
            return
        with connect(MARKET_WS, open_timeout=15, close_timeout=2, proxy=None) as socket:
            socket.send(json.dumps({
                "assets_ids": ids,
                "type": "market",
                "custom_feature_enabled": True,
            }))
            while not self._stop.is_set():
                try:
                    raw = socket.recv(timeout=10)
                except TimeoutError:
                    socket.send("PING")
                    continue
                if raw == "PONG":
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    if apply_market_message(self._board, payload):
                        self.updates += 1


def _yes_price(market: dict[str, Any]) -> float | None:
    outcomes = _json_list(market.get("outcomes"))
    prices = _json_list(market.get("outcomePrices"))
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    for name, price in zip(outcomes, prices):
        if str(name).strip().lower() == "yes":
            return _as_float(price)
    return None


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _get(url: str, params: dict[str, str]) -> dict[str, Any]:
    last: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0 gold-signal/0.2"}
    for attempt in range(3):
        try:
            response = httpx.get(url, params=params, headers=headers, timeout=25.0, follow_redirects=True)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code} {url}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"Polymarket payload is not an object: {str(payload)[:200]}")
            return payload
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Polymarket request failed: {last}") from last
