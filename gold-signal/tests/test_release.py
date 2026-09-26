from fastapi.testclient import TestClient

from gold_signal import __version__
from gold_signal.api.app import create_app
from gold_signal.persistence.store import ProductStore, SCHEMA_VERSION


def test_release_is_1_0_0_and_health_stays_public(tmp_path, monkeypatch):
    assert __version__ == "1.0.0"
    store = ProductStore(tmp_path / "release.sqlite")
    row = store._conn.execute("SELECT version FROM schema_migrations").fetchone()
    assert row[0] == SCHEMA_VERSION
    monkeypatch.delenv("AGENT_BASIC_AUTH", raising=False)
    open_client = TestClient(create_app(store))
    assert open_client.get("/healthz").json() == {"status": "ok"}
    assert open_client.get("/readyz").status_code == 200
    assert open_client.get("/agents").status_code == 200

    monkeypatch.setenv("AGENT_BASIC_AUTH", "ada:secret")
    protected = TestClient(create_app(ProductStore(tmp_path / "auth.sqlite")))
    assert protected.get("/healthz").status_code == 200
    assert protected.get("/agents").status_code == 401
    assert protected.get("/agents", auth=("ada", "secret")).status_code == 200
    store.close()
