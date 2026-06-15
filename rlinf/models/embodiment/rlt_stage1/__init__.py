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

"""RLT Stage 1 model package."""

from __future__ import annotations

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    del torch_dtype
    from .rlt_stage1_policy import RLTStage1Policy

    return RLTStage1Policy(cfg)
