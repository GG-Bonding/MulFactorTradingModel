from pathlib import Path

import uvicorn

from gold_signal.api.app import create_app
from gold_signal.persistence.store import ProductStore


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    store = ProductStore(root / "data" / "agents.sqlite")
    uvicorn.run(create_app(store), host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
