#!/usr/bin/env bash
set -euo pipefail
if [[ "${TIANJI_ACTION_SPACE:-}" == joint ]]; then
  echo "Joint Stage1: use ./deploy/tianji/run_tianji_stage1_local.sh (direct joint queue)." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${RLINF_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export RLINF_TIANJI_STAGE1_ONLY=1
export RLINF_SKIP_REALWORLD_SETUP=1
unset RLT_ACTION_ENV_SCALE RLT_ACTION_ENV_BIAS

PYTHON_BIN="${RLINF_PYTHON:-${TIANJI_GPU_PYTHON:-${PYTHON_BIN:-}}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in "${ROOT_DIR}/.venv-gpu/bin/python" python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 || [[ -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter found; set RLINF_PYTHON in site.env." >&2
  exit 127
fi
export PYTHON_BIN
# shellcheck disable=SC1091
source "$script_dir/tianji_cuda_env.sh"
exec "${PYTHON_BIN}" -u -m rlinf.envs.realworld.tianji.stage1 "$@"
