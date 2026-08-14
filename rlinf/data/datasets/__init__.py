# Copyright 2025 The RLinf Authors.
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

"""Dataset construction entry points.

Import factories from the relevant subpackage:

- ``rlinf.data.datasets.reasoning.create_reasoning_datasets`` —
  text reasoning RL datasets, selected by ``config.data.type``
  (``math`` / ``reasoning`` / ``wideseek_r1`` / ``rstar2``).
- ``rlinf.data.datasets.vlm.create_vlm_datasets`` —
  VLM RL / SFT (``config.data.type == "vlm"``; class via ``dataset_name``).
- ``rlinf.data.datasets.d4rl.build_d4rl_dataset_from_cfg`` —
  offline IQL datasets, selected by ``config.data.dataset_type``.
- Steam / DreamZero / DAgger / OpenPI —
  constructed inside the corresponding workers.

Note: reasoning / VLM use ``config.data.type``; offline / Steam-style paths
often use ``config.data.dataset_type``. Call sites dispatch explicitly rather
than through a single catch-all factory.
"""
