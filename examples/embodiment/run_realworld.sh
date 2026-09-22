#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# WSL interface names are not stable across installations.  Tell Gloo which
# interface Ray workers should use before the training entry point starts.
if [ -z "${RLINF_COMM_NET_DEVICES:-}" ] && command -v ip >/dev/null 2>&1; then
    RLINF_COMM_NET_DEVICES="$(ip route show default 2>/dev/null | awk 'NR == 1 {print $5}')"
    if [ -n "${RLINF_COMM_NET_DEVICES}" ]; then
        export RLINF_COMM_NET_DEVICES
    fi
fi

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_sac_cnn"
else
    CONFIG_NAME=$1
fi
shift || true

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}" "$@")
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
