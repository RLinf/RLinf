#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$(command -v python3 || command -v python)}"

if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "[oft-npu-attn] ERROR: python interpreter not found: $PY" >&2
    echo "               set PYTHON=/path/to/venv/bin/python" >&2
    exit 2
fi

if [ "$#" -eq 0 ]; then
    set -- --install
fi

exec "$PY" "$DIR/patch.py" "$@"