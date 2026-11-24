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

from typing import Any, Literal

import numpy as np
import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class ARTMultiStepRolloutWorker(MultiStepRolloutWorker):
    """Attach OpenPi ART anchors and state features to ordinary rollout results."""

    def _art_anchor_noise(self) -> torch.Tensor:
        """Use one reference across all states and ranks of a policy version."""
        seed = int(self.cfg.actor.seed) * 1_000_003 + int(self.version)
        return torch.randn(
            self.hf_model.action_horizon,
            self.hf_model.model_action_dim,
            generator=torch.Generator().manual_seed(seed),
        )

    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode == "eval":
            return super().predict(env_obs, mode=mode)
        with self.worker_timer("predict"), torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                mode=mode,
                art_anchor_noise=self._art_anchor_noise(),
            )
        result["forward_inputs"].update(env_obs.get("policy_inputs", {}))
        result["expert_label_flag"] = False
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        return actions, result
