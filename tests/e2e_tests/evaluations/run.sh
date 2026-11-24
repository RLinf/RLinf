#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_PATH

CONFIG="${1%.yaml}"
if [[ -n "${2:-}" && "${2:-}" != +* && "${2:-}" != *=* && "${2:-}" != -* && "${2:-}" != ~* ]]; then
    BACKEND=$2
    shift 2
else
    BACKEND="egl"
    shift 1
fi

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${CONFIG}" == *mock* ]]; then
    export RLINF_ROBOT_MOCKS=1
    export PYTHONPATH="${REPO_PATH}/tests:${REPO_PATH}/tests/robot_mocks:${PYTHONPATH}"
    echo "[mock] faking the vendor SDKs; no robot is required"
fi

export HYDRA_FULL_ERROR=1

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

rlinf eval "${REPO_PATH}/tests/e2e_tests/evaluations/${CONFIG}.yaml" "$@"
