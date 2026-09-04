#! /bin/bash
set -x

tabs 4

CONFIG=$1

# $2 is a backend only if it's non-empty and not a hydra override, which is
# either "+key=value" or a plain "key=value".
if [[ -n "${2:-}" && "${2:-}" != +* && "${2:-}" != *=* ]]; then
    BACKEND=$2
    SHIFT_COUNT=2
else
    BACKEND="egl"
    SHIFT_COUNT=1
fi

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# A "mock" config runs a real-world env against faked vendor SDKs, so the
# hardware paths run without a robot. The fakes install themselves through
# sitecustomize in every process, the driver and each scheduler worker alike,
# which is why this is an environment setting rather than a config field.
if [[ "${CONFIG}" == *mock* ]]; then
    export RLINF_ROBOT_MOCKS=1
    export PYTHONPATH=${REPO_PATH}/tests:${REPO_PATH}/tests/robot_mocks:$PYTHONPATH
    echo "[mock] faking the vendor SDKs; no robot is required"
fi

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

shift $SHIFT_COUNT

# A config named for a stage runs that stage's entry point. The Franka guide
# walks through collection, then dataset processing, then training, and each
# is a different script that reaches RealWorldEnv its own way.
case "${CONFIG}" in
    *collect_dataset*)  ENTRY=examples/reward/realworld_collect_process_dataset.py ;;
    *collect_data*)     ENTRY=examples/embodiment/collect_real_data.py ;;
    *)                  ENTRY=examples/embodiment/train_embodied_agent.py ;;
esac
echo "[entry] ${ENTRY}"

python ${REPO_PATH}/${ENTRY} --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ${CONFIG} $@
