from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from gold_signal.models import Bar, FlashNews, Quote

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_MCP_URL = "https://mcp.jin10.com/mcp"
DEFAULT_PROTOCOL_VERSION = "2025-11-25"


class Jin10Error(RuntimeError):
    """Raised when Jin10 MCP request or payload parsing fails."""


class Jin10Client:
    def __init__(
        self,
        token: str,
        url: str = DEFAULT_MCP_URL,
        timeout: float = 30.0,
    ) -> None:
        if not token or not token.strip():
            raise Jin10Error("JIN10_TOKEN is empty. Copy .env.example to .env and fill the token.")
        self.token = token.strip()
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session_id: str | None = None
        self._request_id = 0
        self._http = httpx.Client(timeout=timeout)
        self._connected = False
        self.tools: list[dict[str, Any]] = []

    def close(self) -> None:
        self._http.close()

    def connect(self) -> dict[str, Any]:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "gold-signal", "version": "0.1.0"},
            },
        )
        self._rpc("notifications/initialized", {}, expect_response=False)
        listed = self._rpc("tools/list", {})
        self.tools = listed.get("tools") or []
        self._connected = True
        return result

    def tool_names(self) -> list[str]:
        if not self._connected:
            self.connect()
        return [str(t.get("name")) for t in self.tools if t.get("name")]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self._connected:
            self.connect()
        names = self.tool_names()
        if name not in names:
            raise Jin10Error(
                f"Jin10 MCP has no tool {name!r}. Available tools: {names}"
            )
        result = self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        if result.get("isError"):
            raise Jin10Error(f"Jin10 tool {name} returned an error: {result}")
        return pick_primary_data(result)

    def list_flash(self, cursor: str | None = None) -> list[FlashNews]:
        args: dict[str, Any] = {}
        if cursor:
            args["cursor"] = cursor
        payload = self.call_tool("list_flash", args)
        return parse_flash_list(payload)

    def search_flash(self, keyword: str) -> list[FlashNews]:
        payload = self.call_tool("search_flash", {"keyword": keyword})
        return parse_flash_list(payload)

    def get_quote(self, code: str) -> Quote:
        payload = self.call_tool("get_quote", {"code": code})
        return parse_quote(payload, code)

    def get_kline(self, code: str, count: int = 30, start_ts: int | None = None) -> list[Bar]:
        if not self._connected:
            self.connect()
        if start_ts is None:
            start_ts = int(time.time()) - max(count, 1) * 60
        payload = self.call_tool("get_kline", {"code": code, "time": start_ts, "count": count})
        bars = parse_kline(payload)
        return bars[-count:] if bars else []

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.token}",
            "MCP-Protocol-Version": DEFAULT_PROTOCOL_VERSION,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        expect_response: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if expect_response:
            body["id"] = self._next_id()
        try:
            response = self._http.post(self.url, headers=self._headers(), json=body)
        except httpx.HTTPError as exc:
            raise Jin10Error(f"Jin10 MCP network error calling {method}: {exc}") from exc

        session = response.headers.get("mcp-session-id")
        if session:
            self.session_id = session

        if not expect_response:
            if response.status_code not in (200, 202, 204):
                raise Jin10Error(
                    f"Jin10 MCP {method} failed: HTTP {response.status_code} {response.text[:500]}"
                )
            return {}

        if response.status_code >= 400:
            raise Jin10Error(
                f"Jin10 MCP {method} failed: HTTP {response.status_code} {response.text[:800]}"
            )

        payload = parse_mcp_response(response)
        if payload.get("error"):
            err = payload["error"]
            raise Jin10Error(
                f"Jin10 MCP JSON-RPC error on {method}: {err.get('code')} {err.get('message')}"
            )
        result = payload.get("result")
        if result is None:
            raise Jin10Error(f"Jin10 MCP {method} returned no result: {payload!r}")
        if not isinstance(result, dict):
            return {"value": result}
        return result


def parse_mcp_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
        events = extract_sse_json(text)
        message = next((e["data"] for e in events if e.get("event") in ("message", "")), None)
        if message is None and events:
            message = events[-1]["data"]
        if not isinstance(message, dict):
            raise Jin10Error(f"Jin10 MCP SSE payload is not JSON object: {text[:500]}")
        return message
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise Jin10Error(f"Jin10 MCP returned non-JSON: {text[:500]}") from exc
    if not isinstance(data, dict):
        raise Jin10Error(f"Jin10 MCP JSON is not an object: {data!r}")
    return data


def extract_sse_json(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = ""
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = ""
            return
        raw = "\n".join(data_lines)
        events.append({"event": event_name or "message", "data": json.loads(raw)})
        event_name = ""
        data_lines = []

    for line in text.splitlines():
        if line == "":
            flush()
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    flush()
    if not events:
        raise Jin10Error(f"Jin10 MCP SSE had no data events: {text[:500]}")
    return events


def pick_primary_data(result: dict[str, Any]) -> Any:
    if "structuredContent" in result and result["structuredContent"] is not None:
        return result["structuredContent"]
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
    return result


def parse_flash_list(payload: Any) -> list[FlashNews]:
    items = extract_list(payload, keys=("items", "list", "flashes", "data"))
    news: list[FlashNews] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed = parse_flash_item(item)
        if parsed is not None:
            news.append(parsed)
    return news


def parse_flash_item(item: dict[str, Any]) -> FlashNews | None:
    title = _first_str(item, "title", "t", "name") or ""
    content = _first_str(item, "content", "data", "text", "body", "desc", "description") or ""
    if isinstance(item.get("data"), dict):
        nested = item["data"]
        title = title or _first_str(nested, "title", "content") or ""
        content = content or _first_str(nested, "content", "text", "body") or ""
    text = (content or title).strip()
    if not text:
        return None
    published = parse_time(
        item.get("time")
        or item.get("published_at")
        or item.get("created_at")
        or item.get("timestamp")
        or item.get("date")
    )
    event_id = str(item.get("id") or item.get("flash_id") or make_event_id(published, text))
    return FlashNews(
        event_id=event_id,
        title=title.strip(),
        content=text,
        published_at=published,
        raw=item,
    )


def parse_quote(payload: Any, fallback_code: str) -> Quote:
    data = unwrap_data(payload)
    if not isinstance(data, dict):
        raise Jin10Error(f"get_quote payload is not an object: {payload!r}")
    price = _first_number(
        data,
        "close",
        "price",
        "last",
        "last_price",
        "bid",
        "sell",
        "c",
    )
    if price is None:
        raise Jin10Error(f"get_quote missing price fields: {data!r}")
    code = _first_str(data, "code", "symbol") or fallback_code
    ts = parse_time(data.get("time") or data.get("timestamp") or data.get("updated_at"), allow_now=True)
    return Quote(code=code, price=float(price), ts=ts, raw=data)


def parse_kline(payload: Any) -> list[Bar]:
    items = extract_list(payload, keys=("items", "list", "klines", "bars", "data", "kline"))
    bars: list[Bar] = []
    for item in items:
        bar = parse_bar(item)
        if bar is not None:
            bars.append(bar)
    bars.sort(key=lambda b: b.ts)
    return bars


def parse_bar(item: Any) -> Bar | None:
    if isinstance(item, list) and len(item) >= 2:
        ts = parse_time(item[0], allow_now=False)
        close = _as_float(item[-1] if len(item) == 2 else item[4] if len(item) >= 5 else item[1])
        if close is None:
            return None
        return Bar(ts=ts, close=close)
    if not isinstance(item, dict):
        return None
    close = _first_number(item, "close", "c", "price", "last")
    if close is None:
        return None
    ts = parse_time(item.get("time") or item.get("timestamp") or item.get("t") or item.get("date"))
    return Bar(
        ts=ts,
        close=float(close),
        open=_first_number(item, "open", "o"),
        high=_first_number(item, "high", "h"),
        low=_first_number(item, "low", "l"),
    )


def make_event_id(published_at: datetime, text: str) -> str:
    import hashlib

    stamp = published_at.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha1(f"{stamp}|{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"{stamp}-{digest}"


def parse_time(value: Any, *, allow_now: bool = False) -> datetime:
    if value is None:
        if allow_now:
            return datetime.now(tz=SHANGHAI)
        raise Jin10Error("missing timestamp")
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=SHANGHAI)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=SHANGHAI)
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return parse_time(int(raw))
        iso = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    parsed = None
            if parsed is None:
                raise Jin10Error(f"unrecognized time format: {value!r}")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=SHANGHAI)
        return parsed.astimezone(SHANGHAI)
    raise Jin10Error(f"unrecognized time value: {value!r}")


def unwrap_data(payload: Any) -> Any:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), dict) and _looks_like_quote(payload["data"]):
            return payload["data"]
        if "close" in payload or "price" in payload:
            return payload
        if isinstance(payload.get("data"), dict):
            return payload["data"]
    return payload


def _looks_like_quote(data: dict[str, Any]) -> bool:
    return any(k in data for k in ("close", "price", "last", "code"))


def extract_list(payload: Any, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = extract_list(value, keys)
            if nested:
                return nested
    return []


def _first_str(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _as_float(data.get(key))
        if number is not None:
            return number
    return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None
