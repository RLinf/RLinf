#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [ $# -eq 0 ] || [[ "$1" == *=* ]] || [[ "$1" == +* ]] || [[ "$1" == ~* ]] || [[ "$1" == -* ]]; then
    CONFIG_NAME="image_alignment"
else
    CONFIG_NAME=$1
    shift
fi

model_raw=""
actor_placement="0-1"
for arg in "$@"; do
    case "$arg" in
        rollout.model.model_path=*)
            model_raw=${arg#rollout.model.model_path=}
            ;;
        cluster.component_placement.actor=*)
            actor_placement=${arg#cluster.component_placement.actor=}
            ;;
    esac
done
if [[ -z "$model_raw" ]]; then
    echo "usage: bash examples/agent/qwen35/run_image_alignment.sh [CONFIG_NAME] rollout.model.model_path=/path/to/Qwen3.5-4B [hydra overrides...]" >&2
    exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpus="$CUDA_VISIBLE_DEVICES"
    IFS=, read -ra gpu_list <<< "$gpus"
    if [[ ${#gpu_list[@]} -ne 2 && ${#gpu_list[@]} -ne 8 ]]; then
        echo "Expected 2 or 8 contiguous GPU indices in CUDA_VISIBLE_DEVICES, got '$gpus'." >&2
        exit 2
    fi
    prev=""
    for idx in "${gpu_list[@]}"; do
        if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
            echo "Expected numeric GPU indices, got '$gpus'." >&2
            exit 2
        fi
        if [[ -n "$prev" && $((idx - prev)) -ne 1 ]]; then
            echo "Expected contiguous ascending GPU indices, got '$gpus'." >&2
            exit 2
        fi
        prev=$idx
    done
    first_gpu=${gpu_list[0]}
else
    first_gpu=${actor_placement%%[-,]*}
    if [[ ! "$first_gpu" =~ ^[0-9]+$ ]]; then
        echo "Expected numeric GPU index in cluster.component_placement.actor, got '$actor_placement'." >&2
        exit 2
    fi
fi

model=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$model_raw")
example_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$example_dir/../../.." && pwd)
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
run_dir=$(mktemp -d)
trap 'rm -rf -- "$run_dir"' EXIT
export RAY_TMPDIR="$run_dir"
# Keep this single-node example off other Ray clusters on the host.
export RAY_ADDRESS=local
CUDA_VISIBLE_DEVICES="$first_gpu" python "$example_dir/check_image_alignment.py" --model "$model" --rows "$run_dir/rows.pt"
for role in actor critic; do
    CUDA_DEVICE_MAX_CONNECTIONS=1 python "$example_dir/update_check.py" --config-name "$CONFIG_NAME" --rows "$run_dir/rows.pt" --role "$role" "$@" "rollout.model.model_path=$model"
done
