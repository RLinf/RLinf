#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

REGION_ID="${REGION_ID:-rg-da5azznpr4i4fdrx}"
RESOURCE_SPEC_ID="${RESOURCE_SPEC_ID:-rs-dba3vcq2k4po4o5g}"
WORKER_NUM="${WORKER_NUM:-1}"
EXPECT_TRAIN_COMPLETE_TIME="${EXPECT_TRAIN_COMPLETE_TIME:-3600}"

api_curl "/api/platform/open/v1/train_plan/idle_resources/generate" "{
  \"region_id\": \"${REGION_ID}\",
  \"resource_spec_id\": \"${RESOURCE_SPEC_ID}\",
  \"worker_num\": ${WORKER_NUM},
  \"expect_train_complete_time\": ${EXPECT_TRAIN_COMPLETE_TIME}
}"
