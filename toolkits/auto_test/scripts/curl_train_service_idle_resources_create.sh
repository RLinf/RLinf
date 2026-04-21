#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

REGION_ID="${REGION_ID:-rg-da5azznpr4i4fdrx}"
RESOURCE_SPEC_ID="${RESOURCE_SPEC_ID:-rs-dba3vcq2k4po4o5g}"
JOB_NAME="${JOB_NAME:-training-server-test-manual}"
JOB_DESCRIPTION="${JOB_DESCRIPTION:-idle resources smoke test}"
IMAGE_ID="${IMAGE_ID:-im-dcm6egnmqdgurn6e}"
FRAMEWORK_ID="${FRAMEWORK_ID:-fw-c6q6a7sfyhoeb5xi}"
WORKER_NUM="${WORKER_NUM:-1}"
ENTRY_POINT="${ENTRY_POINT:-cd /mnt/public/xusi/RLinf-fork-v0.2 && bash run_yaml.sh}"
SHARED_MEM="${SHARED_MEM:-1}"
EXPECT_TRAIN_COMPLETE_TIME="${EXPECT_TRAIN_COMPLETE_TIME:-3600}"
TRAIN_PLAN_PRE_EXECUTION_ID="${TRAIN_PLAN_PRE_EXECUTION_ID:-}"
TRAIN_PLAN_ID="${TRAIN_PLAN_ID:-}"
VOLUME_ID="${VOLUME_ID:-vo-dba4je5b5vun473d}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/public/}"
RDMA_ENABLE="${RDMA_ENABLE:-false}"

if [[ -z "${TRAIN_PLAN_PRE_EXECUTION_ID}" ]]; then
  echo "TRAIN_PLAN_PRE_EXECUTION_ID is required" >&2
  exit 1
fi

if [[ -z "${TRAIN_PLAN_ID}" ]]; then
  echo "TRAIN_PLAN_ID is required" >&2
  exit 1
fi

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
  \"rdma_enable\": ${RDMA_ENABLE},
  \"mount\": [
    {
      \"path\": \"${MOUNT_PATH}\",
      \"volume_id\": \"${VOLUME_ID}\",
      \"rw_setting\": \"can_write\"
    }
  ]
}"
