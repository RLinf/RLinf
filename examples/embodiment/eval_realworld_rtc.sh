#! /bin/bash

# Real-world PI0.5 evaluation with optional RTC (Real-Time Correction).
#
# Usage:
#   bash eval_realworld_rtc.sh                 # RTC enabled (default)
#   RTC_ENABLED=False bash eval_realworld_rtc.sh  # baseline (no RTC)

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

CONFIG_NAME="realworld_pnp_eval_pi05_sft_RTC"
RTC_ENABLED="${RTC_ENABLED:-True}"

echo "Using Python at $(which python)"
echo "RTC enabled: ${RTC_ENABLED}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_eval.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.rtc.enabled=${RTC_ENABLED} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
