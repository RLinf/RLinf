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

"""Opt-in patches for the Wan world-model env, installed via the shared Patcher.

The NPU-capable attention is referenced by the Patcher via its dotted path in
``WanEnv._build_pipeline``; it is also re-exported here for direct import.
"""

from rlinf.envs.world_model.ascend_patch.wan_video_dit import flash_attention, rope_apply, RMSNorm

__all__ = ["flash_attention", "rope_apply", "RMSNorm"]