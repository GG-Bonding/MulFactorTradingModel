import sqlite3

from fastapi.testclient import TestClient

from gold_signal import __version__
from gold_signal.api.app import create_app
from gold_signal.persistence.store import ProductStore, SCHEMA_VERSION


def test_existing_database_gains_fill_cost_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_migrations (version) VALUES (1)")
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            signal_id TEXT,
            side TEXT NOT NULL,
            entry_price REAL,
            spec_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            net_return REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    store = ProductStore(path)
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(paper_trades)")}
    assert {"spread_cost", "slippage_cost", "commission"} <= columns
    latest = store._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    assert latest[0] == SCHEMA_VERSION


def test_release_is_1_0_0_and_health_stays_public(tmp_path, monkeypatch):
    assert __version__ == "1.0.0"
    store = ProductStore(tmp_path / "release.sqlite")
    row = store._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    assert row[0] == SCHEMA_VERSION
    monkeypatch.delenv("AGENT_BASIC_AUTH", raising=False)
    open_client = TestClient(create_app(store))
    assert open_client.get("/healthz").json() == {"status": "ok"}
    ready = open_client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["database"] == "READY"
    assert ready.json()["ingest"] == "OFF"
    assert ready.json()["last_market_at"] is None
    assert open_client.get("/agents").status_code == 200

    monkeypatch.setenv("AGENT_BASIC_AUTH", "ada:secret")
    protected = TestClient(create_app(ProductStore(tmp_path / "auth.sqlite")))
    assert protected.get("/healthz").status_code == 200
    assert protected.get("/agents").status_code == 401
    assert protected.get("/agents", auth=("ada", "secret")).status_code == 200
    store.close()
