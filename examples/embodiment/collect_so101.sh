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

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
EPISODES="${1:-1}"

export EMBODIED_PATH REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

exec "${PYTHON_BIN}" "${EMBODIED_PATH}/collect_real_data.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name realworld_so101_collect_data \
  "runner.num_data_episodes=${EPISODES}"
