# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from omegaconf import DictConfig, open_dict

from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiInferRolloutWorker(MultiStepRolloutWorker):
    """Use an in-process rollout policy with the shared weight-sync lifecycle."""

    def _build_model(self, model_cfg: DictConfig) -> BasePolicy:
        # Runtime selection belongs to the rollout; the actor config stays unchanged.
        runtime_cfg = copy.deepcopy(model_cfg)
        with open_dict(runtime_cfg):
            runtime_cfg.model_backend = "embodiinfer"
            runtime_cfg.embodiinfer = self.cfg.rollout.get("embodiinfer", {})
        return get_model(runtime_cfg)
