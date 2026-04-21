#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

POOL_ID="${POOL_ID:-po-da73jexmoe4rfgej}"
RESOURCE_TYPE="${RESOURCE_TYPE:-1}"
SHELF_STATUS="${SHELF_STATUS:-1}"

api_curl "/api/platform/open/v1/load_spec/list" "{
  \"shelf_status\": ${SHELF_STATUS},
  \"pool_id\": \"${POOL_ID}\",
  \"resource_type\": ${RESOURCE_TYPE},
  \"virtual_cluster_exclusive\": true,
  \"available_load_type_list\": [\"virtual_cluster\"]
}"
