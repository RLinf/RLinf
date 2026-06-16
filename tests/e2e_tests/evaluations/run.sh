#! /bin/bash
set -x

tabs 4

CONFIG=$1
BACKEND=${2:-"egl"}

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH=${REPO_PATH}/examples/embodiment
export EVALUATIONS_PATH=${REPO_PATH}/evaluations

export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

python ${EVALUATIONS_PATH}/eval_embodied_agent.py \
    --config-path ${REPO_PATH}/tests/e2e_tests/evaluations \
    --config-name ${CONFIG}
