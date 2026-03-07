#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_d4rl_offline.py"
export VENV_PATH="${REPO_PATH}/.venv"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

if [ -f "${VENV_PATH}/bin/activate" ]; then
    # Use project-local virtual environment by default.
    source "${VENV_PATH}/bin/activate"
else
    echo "Warning: ${VENV_PATH}/bin/activate not found, using system python."
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [config_name|env_name|d4rl_offline_<env_name>] [hydra_overrides...]"
    echo "Available config_name:"
    echo "  d4rl_offline_mujoco         # Locomotion preset"
    echo "  d4rl_offline_antmaze        # AntMaze preset"
    echo "  d4rl_offline_kitchen_adroit # Kitchen/Adroit preset"
    echo "Examples with env override:"
    echo "  $0 halfcheetah-medium-v2"
    echo "  $0 d4rl_offline_antmaze-large-play-v0"
    exit 0
fi

# config / env selector
USER_ARG="$1"
EXTRA_ARGS=()
if [ "$#" -gt 1 ]; then
    EXTRA_ARGS=("${@:2}")
fi
if [ -z "${USER_ARG}" ]; then
    CONFIG_NAME="d4rl_offline_mujoco"
    ENV_NAME_OVERRIDE=""
    EXPERIMENT_NAME="${CONFIG_NAME}"
else
    case "${USER_ARG}" in
        d4rl_offline_mujoco|d4rl_offline_antmaze|d4rl_offline_kitchen_adroit)
            CONFIG_NAME="${USER_ARG}"
            ENV_NAME_OVERRIDE=""
            EXPERIMENT_NAME="${CONFIG_NAME}"
            ;;
        *)
            if [[ "${USER_ARG}" == d4rl_offline_* ]]; then
                ENV_NAME_OVERRIDE="${USER_ARG#d4rl_offline_}"
            else
                ENV_NAME_OVERRIDE="${USER_ARG}"
            fi

            case "${ENV_NAME_OVERRIDE}" in
                antmaze-*)
                    CONFIG_NAME="d4rl_offline_antmaze"
                    ;;
                halfcheetah-*|hopper-*|walker2d-*)
                    CONFIG_NAME="d4rl_offline_mujoco"
                    ;;
                pen-*|door-*|hammer-*|relocate-*|kitchen-*)
                    CONFIG_NAME="d4rl_offline_kitchen_adroit"
                    ;;
                *)
                    echo "Unsupported env/config: ${USER_ARG}"
                    echo "Run '$0 --help' to see supported usage."
                    exit 1
                    ;;
            esac
            EXPERIMENT_NAME="d4rl_offline_${ENV_NAME_OVERRIDE}"
            ;;
    esac
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${EXPERIMENT_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_d4rl_offline.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.logger.experiment_name=${EXPERIMENT_NAME}"
if [ -n "${ENV_NAME_OVERRIDE}" ]; then
    CMD="${CMD} env_name=${ENV_NAME_OVERRIDE}"
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD="${CMD} ${EXTRA_ARGS[*]}"
fi
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}

