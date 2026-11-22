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

rlinf train --family sft "${REPO_PATH}/tests/e2e_tests/sft/${CONFIG}.yaml" "$@"
