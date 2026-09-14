#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec python live_server/app.py --host "${AIRVLN_SERVER_HOST:-127.0.0.1}" --port "${AIRVLN_SERVER_PORT:-18080}"
