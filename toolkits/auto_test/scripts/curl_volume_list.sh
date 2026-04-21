#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-200}"

api_curl "/api/platform/open/v1/volume/list" "{
  \"offset\": ${OFFSET},
  \"limit\": ${LIMIT}
}"
