#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

REGION_ID="${REGION_ID:-rg-da5azznpr4i4fdrx}"
RESOURCE_SPEC_ID="${RESOURCE_SPEC_ID:-rs-dba3vcq2k4po4o5g}"
WORKER_NUM="${WORKER_NUM:-1}"
EXPECT_TRAIN_COMPLETE_TIME="${EXPECT_TRAIN_COMPLETE_TIME:-3600}"
JOB_NAME="${JOB_NAME:-training-server-test-manual}"
JOB_DESCRIPTION="${JOB_DESCRIPTION:-idle resources smoke test}"
IMAGE_ID="${IMAGE_ID:-im-dcm6egnmqdgurn6e}"
FRAMEWORK_ID="${FRAMEWORK_ID:-fw-c6q6a7sfyhoeb5xi}"
ENTRY_POINT="${ENTRY_POINT:-cd /mnt/public/xusi/RLinf-fork-v0.2 && bash run_yaml.sh}"
SHARED_MEM="${SHARED_MEM:-1}"
VOLUME_ID="${VOLUME_ID:-vo-dba4je5b5vun473d}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/public/}"

PLAN_RESULT="$(api_curl "/api/platform/open/v1/train_plan/idle_resources/generate" "{
  \"region_id\": \"${REGION_ID}\",
  \"resource_spec_id\": \"${RESOURCE_SPEC_ID}\",
  \"worker_num\": ${WORKER_NUM},
  \"expect_train_complete_time\": ${EXPECT_TRAIN_COMPLETE_TIME}
}")"

echo "${PLAN_RESULT}"

TRAIN_PLAN_PRE_EXECUTION_ID="$(printf '%s' "${PLAN_RESULT}" | python -c 'import json,sys; data=json.load(sys.stdin)["data"]; print(data["train_plan_pre_execution_id"])')"
TRAIN_PLAN_ID="$(printf '%s' "${PLAN_RESULT}" | python -c 'import json,sys; data=json.load(sys.stdin)["data"]; print(data["train_plan_list"][0]["train_plan_id"])')"

api_curl "/api/platform/open/v1/train_service/idle_resources/create" "{
  \"region_id\": \"${REGION_ID}\",
  \"resource_spec_id\": \"${RESOURCE_SPEC_ID}\",
  \"job_name\": \"${JOB_NAME}\",
  \"job_description\": \"${JOB_DESCRIPTION}\",
  \"image_id\": \"${IMAGE_ID}\",
  \"framework_id\": \"${FRAMEWORK_ID}\",
  \"worker_num\": ${WORKER_NUM},
  \"entry_point\": \"${ENTRY_POINT}\",
  \"shared_mem\": ${SHARED_MEM},
  \"expect_train_complete_time\": ${EXPECT_TRAIN_COMPLETE_TIME},
  \"train_plan_pre_execution_id\": \"${TRAIN_PLAN_PRE_EXECUTION_ID}\",
  \"train_plan_id\": \"${TRAIN_PLAN_ID}\",
  \"rdma_enable\": false,
  \"mount\": [
    {
      \"path\": \"${MOUNT_PATH}\",
      \"volume_id\": \"${VOLUME_ID}\",
      \"rw_setting\": \"can_write\"
    }
  ]
}"
