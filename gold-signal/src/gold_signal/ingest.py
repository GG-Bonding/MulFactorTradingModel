from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime

from gold_signal.models import FlashNews, MarketSnapshot
from gold_signal.observation import EventRecord, Observation
from gold_signal.persistence.store import ProductStore
from gold_signal.research import observations_from_bars
from gold_signal.runtime.agent_loop import AgentRuntime


@dataclass(frozen=True)
class IngestBatch:
    now: datetime
    observations: tuple[Observation, ...]
    event: EventRecord | None = None


class ScriptedFeed:
    """Test feed. The server loop pulls batches; callers do not post ticks."""

    def __init__(self) -> None:
        self._queue: queue.Queue[IngestBatch | None] = queue.Queue()

    def push(self, batch: IngestBatch) -> None:
        self._queue.put(batch)

    def poll(self) -> IngestBatch | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class Jin10Feed:
    def __init__(self, token: str, url: str | None = None) -> None:
        self.token = token
        self.url = url

    def poll(self) -> IngestBatch | None:
        from gold_signal.jin10 import Jin10Client
        from gold_signal.main import fetch_market, merge_news, pick_news

        client = Jin10Client(token=self.token, url=self.url or "https://mcp.jin10.com/mcp")
        try:
            client.connect()
            news_list = client.list_flash()
            for keyword in ("黄金", "美联储", "非农", "CPI"):
                try:
                    news_list = merge_news(news_list, client.search_flash(keyword))
                except Exception:
                    continue
            market = fetch_market(client)
            now = market.as_of
            news = pick_news(news_list, now)
            return IngestBatch(now, tuple(snapshot_observations(market)), _event_from_news(news, now))
        except Exception:
            return None
        finally:
            client.close()


class IngestLoop:
    def __init__(self, store: ProductStore, feed: ScriptedFeed | Jin10Feed, runtime: AgentRuntime | None = None) -> None:
        self.store = store
        self.feed = feed
        self.runtime = runtime or AgentRuntime()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="agent-ingest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self.feed.poll()
            if batch is None:
                self._stop.wait(0.02)
                continue
            self.runtime.tick(
                self.store,
                now=batch.now,
                observations=list(batch.observations),
                event=batch.event,
            )


def snapshot_observations(market: MarketSnapshot) -> list[Observation]:
    rows: list[Observation] = []
    for asset, bars in (
        (market.xau, market.xau_bars),
        (market.xag, market.xag_bars),
        (market.eurusd, market.eurusd_bars),
    ):
        if not asset.code or asset.code == "FLAT" or not bars:
            continue
        rows.extend(
            observations_from_bars(
                f"market.{asset.code}.close",
                list(bars),
                symbol=asset.code,
                source="live",
                timestamp_kind="close",
            )
        )
    return rows


def feed_agents(store: ProductStore, market: MarketSnapshot, news: FlashNews | None, now: datetime) -> None:
    AgentRuntime().tick(
        store,
        now=now,
        observations=snapshot_observations(market),
        event=_event_from_news(news, now),
    )


def start_live_feed(store: ProductStore) -> IngestLoop | None:
    token = os.getenv("JIN10_TOKEN") or os.getenv("JIN10_BEARER_TOKEN")
    if not token or os.getenv("AGENT_INGEST", "1") == "0":
        return None
    loop = IngestLoop(store, Jin10Feed(token, os.getenv("JIN10_MCP_URL") or os.getenv("JIN10_MCP_SERVER_URL")))
    loop.start()
    return loop


def _event_from_news(news: FlashNews | None, now: datetime) -> EventRecord | None:
    if news is None:
        return None
    return EventRecord(
        event_id=news.event_id,
        event_type="",
        published_at=news.published_at,
        available_at=news.published_at,
        ingested_at=now,
        title=news.title,
        content=news.content or news.title,
        source="jin10",
    )
