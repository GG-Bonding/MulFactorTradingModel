from pathlib import Path
import os

import uvicorn

from gold_signal.api.app import create_app
from gold_signal.persistence.store import ProductStore


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    db = Path(os.environ.get("AGENT_DB", root / "data" / "agents.sqlite"))
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    store = ProductStore(db)
    from gold_signal.ingest import start_live_feed

    start_live_feed(store)
    uvicorn.run(create_app(store), host=host, port=port)


if __name__ == "__main__":
    main()
