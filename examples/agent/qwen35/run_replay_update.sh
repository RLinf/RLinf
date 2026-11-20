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

model=${1:?"usage: run_replay_update.sh /path/to/Qwen3.5-4B"}
example_dir=$(cd "$(dirname "$0")" && pwd)
run_dir=$(mktemp -d)
trap 'rm -rf -- "$run_dir"' EXIT
export RAY_TMPDIR="$run_dir"
# Keep this single-node example off other Ray clusters on the host.
export RAY_ADDRESS=local
python "$example_dir/replay_images.py" --model "$model" --rows "$run_dir/rows.pt"
for role in dense chunked critic; do
    python "$example_dir/replay_update.py" --model "$model" --rows "$run_dir/rows.pt" --role "$role"
done
