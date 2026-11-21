#!/usr/bin/env bash
set -euo pipefail
local_launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$local_launcher/tianji_local.env"
if [[ ! -x "$TIANJI_CPU_PYTHON" ]] && ! command -v "$TIANJI_CPU_PYTHON" >/dev/null 2>&1; then
  echo "CPU validation environment missing: $TIANJI_CPU_PYTHON" >&2
  exit 2
fi
exec "$TIANJI_CPU_PYTHON" "$local_launcher/preflight_tianji_local.py" "$@"
