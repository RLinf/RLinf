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

from omegaconf import DictConfig


def get_embodied_eval_runner(cfg: DictConfig) -> type:
    """Runner of an embodied evaluation config."""
    if (
        cfg.rollout.get("rollout_backend", "huggingface") == "huggingface"
        and cfg.algorithm.get("loss_type", None) == "supermimic"
    ):
        from .supermimic_eval_runner import SuperMimicEvalRunner

        return SuperMimicEvalRunner
    from .embodied_eval_runner import EmbodiedEvalRunner

    return EmbodiedEvalRunner
