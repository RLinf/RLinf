#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
# setup_sim_env is defined in sim_env.sh.
source "${EMBODIED_PATH}/sim_env.sh"
setup_sim_env

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_ppo_openvlaoft"}
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
# Forward optional overrides exported by callers (e.g. tests/parity_tests/run_all.sh).
# Sentinel: "-2" means "do not override, use YAML default". -1 is a legitimate value
# (e.g. runner.max_steps=-1 means unlimited) and is forwarded as-is.
EXTRA_OVERRIDES=""
[ -n "${STEPS:-}" ]      && [ "$STEPS"      != "-2" ] && EXTRA_OVERRIDES+=" runner.max_steps=${STEPS}"
[ -n "${SAVE_INTER:-}" ] && [ "$SAVE_INTER" != "-2" ] && EXTRA_OVERRIDES+=" runner.save_interval=${SAVE_INTER}"
[ -n "${NODES:-}" ]      && [ "$NODES"      != "-2" ] && EXTRA_OVERRIDES+=" cluster.num_nodes=${NODES}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}${EXTRA_OVERRIDES}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
