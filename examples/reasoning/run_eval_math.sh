#!/usr/bin/env bash
set -euo pipefail

source /mnt/public2/suheng/venvs/rlinf-agentic/bin/activate
export TMPDIR=/mnt/public2/suheng/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

config_name=eval_qwen3.5-2b
if [[ $# -gt 0 && $1 != *=* && $1 != -* ]]; then
    config_name=$1
    shift
fi

exec python "$script_dir/eval_math.py" --config-path "$script_dir/config/math" \
    --config-name "$config_name" "$@"
