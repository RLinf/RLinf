#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

REGION_ID="${REGION_ID:-rg-da5azznpr4i4fdrx}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-20}"
TARGET_POOL_TYPE="${TARGET_POOL_TYPE:-1}"
EXCLUDE_POOL_IDS="${EXCLUDE_POOL_IDS:-[]}"

api_curl "/api/platform/open/v1/pool/get_resource_detail" "{
  \"return_count\": true,
  \"offset\": ${OFFSET},
  \"limit\": ${LIMIT},
  \"region_id\": \"${REGION_ID}\",
  \"exclude_pool_ids\": ${EXCLUDE_POOL_IDS},
  \"target_pool_type\": ${TARGET_POOL_TYPE}
}"
