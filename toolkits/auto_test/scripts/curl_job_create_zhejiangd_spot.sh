#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

JOB_NAME="${JOB_NAME:-zhejiangd-spot-smoke-manual}"
JOB_DESCRIPTION="${JOB_DESCRIPTION:-zhejiangD spot smoke test}"
IMAGE_ID="${IMAGE_ID:-im-dcm6egnmqdgurn6e}"
FRAMEWORK_ID="${FRAMEWORK_ID:-fw-c6q6a7sfyhoeb5xi}"
RESOURCE_SPEC_ID="${RESOURCE_SPEC_ID:-rs-dba3vcq2k4po4o5g}"
ENTRY_POINT="${ENTRY_POINT:-sleep 600}"
WORKER_NUM="${WORKER_NUM:-1}"

api_curl "/api/platform/open/v1/job/create" "{
  \"job_type\": \"train\",
  \"job_name\": \"${JOB_NAME}\",
  \"job_description\": \"${JOB_DESCRIPTION}\",
  \"image_id\": \"${IMAGE_ID}\",
  \"framework_id\": \"${FRAMEWORK_ID}\",
  \"worker_num\": ${WORKER_NUM},
  \"entry_point\": \"${ENTRY_POINT}\",
  \"resource_type\": 1,
  \"rdma_enable\": false,
  \"fault_tolerance\": {
    \"auto_restart\": {
      \"enable\": true,
      \"conditions\": [\"job_fail\"],
      \"max_retry\": 1
    },
    \"hang_check\": {
      \"enable\": false,
      \"timeout\": 0
    },
    \"environment_check\": {
      \"enable\": true,
      \"conditions\": [\"job_init\", \"job_fail\"]
    }
  },
  \"check_train_slow\": 1,
  \"resource_spec_id\": \"${RESOURCE_SPEC_ID}\"
}"
