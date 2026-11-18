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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

source /mnt/public2/suheng/venvs/rlinf-agentic/bin/activate
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_ADDRESS=local
export TMPDIR=/mnt/public2/suheng/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# Extra arguments are forwarded to Hydra; use --cfg job to inspect each config.
for variant in k1-pg top64-bp student-top64-bp; do
    config_name="qwen3.5-2b-opd-${variant}-fsdp2-sglang"
    experiment_name="qwen3.5-2b-opd-${variant}-8gpu"
    printf 'Starting %s (Teacher: GPUs 0-1, Student: GPUs 2-7)\n' "$experiment_name"
    printf 'Log: ./logs/%s/log/main.log\n' "$experiment_name"

    python -u examples/reasoning/main_grpo.py \
        --config-path config/math \
        --config-name "$config_name" \
        "runner.experiment_name=$experiment_name" \
        runner.execution_mode=sync \
        cluster.num_nodes=1 \
        '~cluster.component_placement' \
        '+cluster.component_placement={actor:2-7,rollout:2-7,reward:2-7,teacher_default:0-1}' \
        rollout.tensor_parallel_size=1 \
        distillation.teacher_models.default.inference.tensor_parallel_size=2 \
        "$@"

    printf 'Completed %s\n' "$experiment_name"
done
