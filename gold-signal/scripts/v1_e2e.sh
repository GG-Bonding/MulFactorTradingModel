#!/bin/sh
# Build the image, boot it, and run the V1 freeze test against that container.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PORT=${E2E_PORT:-8766}
NAME=gold-signal-v1-e2e

docker build -t gold-signal:v1-e2e .
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8765" -e AGENT_INGEST=0 -e HOST=0.0.0.0 -e PORT=8765 gold-signal:v1-e2e
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
stable=0
i=0
while [ "$i" -lt 30 ]; do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null && curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null; then
    stable=$((stable + 1))
    if [ "$stable" -ge 2 ]; then
      ready=1
      break
    fi
  else
    stable=0
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "container did not become ready" >&2
  docker logs "$NAME" >&2 || true
  exit 1
fi

GOLD_SIGNAL_URL="http://127.0.0.1:$PORT" "$ROOT/.venv/bin/pytest" tests/test_v1_e2e.py
