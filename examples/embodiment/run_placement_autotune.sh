#! /bin/bash
# Backward-compatible wrapper. The canonical script now lives under toolkits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec "${REPO_PATH}/toolkits/auto_placement/run_placement_autotune.sh" "$@"