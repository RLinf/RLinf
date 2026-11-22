#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_PATH

CONFIG="${1%.yaml}"

# $2 is a backend only if it's non-empty and not a hydra override, which is
# either "+key=value" or a plain "key=value".
if [[ -n "${2:-}" && "${2:-}" != +* && "${2:-}" != *=* && "${2:-}" != -* && "${2:-}" != ~* ]]; then
    BACKEND=$2
    SHIFT_COUNT=2
else
    BACKEND="egl"
    SHIFT_COUNT=1
fi

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# A "mock" config runs a real-world env against faked vendor SDKs, so the
# hardware paths run without a robot. The fakes install themselves through
# sitecustomize in every process, the driver and each scheduler worker alike,
# which is why this is an environment setting rather than a config field.
if [[ "${CONFIG}" == *mock* ]]; then
    export RLINF_ROBOT_MOCKS=1
    export PYTHONPATH="${REPO_PATH}/tests:${REPO_PATH}/tests/robot_mocks:${PYTHONPATH}"
    echo "[mock] faking the vendor SDKs; no robot is required"
fi

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

CONFIG_FILE="${REPO_PATH}/tests/e2e_tests/embodied/${CONFIG}.yaml"
case "${CONFIG}" in
    *collect_dataset*|*collect_data*)
        rlinf collect "${CONFIG_FILE}" "$@"
        ;;
    *)
        rlinf train "${CONFIG_FILE}" "$@"
        ;;
esac
