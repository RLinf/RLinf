#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/.venv/bin/python}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${repo_dir}/toolkits/realworld_check/run_so101_mock.py" "$@"
