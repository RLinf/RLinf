#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/.venv/bin/python}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

command="${1:-mock}"
shift || true
case "${command}" in
    mock)
        exec "${python_bin}" "${repo_dir}/toolkits/realworld_check/run_so101_mock.py" "$@"
        ;;
    grpc-check)
        exec "${python_bin}" "${repo_dir}/toolkits/realworld_check/check_so101_grpc.py" "$@"
        ;;
    *)
        echo "usage: $0 {mock|grpc-check} [options]" >&2
        exit 2
        ;;
esac
