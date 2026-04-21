#!/usr/bin/env bash

set -euo pipefail

API_KEY="${INFINI_API_KEY:-}"
BASE_URL="${INFINI_BASE_URL:-https://cloud.infini-ai.com}"
JOB_NAME="${JOB_NAME:-zhejiangd-spot-smoke-manual}"
JOB_DESCRIPTION="${JOB_DESCRIPTION:-zhejiangD spot smoke test}"
IMAGE_ID="${IMAGE_ID:-im-c7flk4j34cxnbjut}"
FRAMEWORK_ID="${FRAMEWORK_ID:-fw-c6q6a7sfyhoeb5xi}"
RESOURCE_SPEC_ID="${RESOURCE_SPEC_ID:-rs-dba3vcq2k4po4o5g}"
ENTRY_POINT="${ENTRY_POINT:-sleep 600}"

if [[ -z "${API_KEY}" ]]; then
  echo "INFINI_API_KEY is required" >&2
  exit 1
fi

curl --request POST \
  --url "${BASE_URL}/api/platform/open/v1/job/create" \
  --header 'Accept: application/json' \
  --header "Authorization: Bearer ${API_KEY}" \
  --header 'Content-Type: application/json' \
  --data "{
  \"job_type\": \"train\",
  \"job_name\": \"${JOB_NAME}\",
  \"job_description\": \"${JOB_DESCRIPTION}\",
  \"image_id\": \"${IMAGE_ID}\",
  \"framework_id\": \"${FRAMEWORK_ID}\",
  \"worker_num\": 1,
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

