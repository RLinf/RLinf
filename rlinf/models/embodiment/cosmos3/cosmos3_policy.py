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

"""Environment input/output adapter for the Cosmos3 action model."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.cosmos3.action_codec import (
    Cosmos3ActionCodec,
    decode_prompts,
    encode_prompts,
)

__all__ = ["Cosmos3Policy", "decode_prompts", "encode_prompts"]


class Cosmos3Policy:
    """Translate RLinf environment observations and native model actions."""

    def _codec(self) -> Cosmos3ActionCodec:
        if not hasattr(self, "_action_codec"):
            self._action_codec = Cosmos3ActionCodec(self.model_cfg)
        return self._action_codec

    def _observations(self, env_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self._codec().encode_observations(env_obs)

    def _env_actions(self, action_list: list[torch.Tensor]) -> torch.Tensor:
        import numpy as np

        codec = self._codec()
        chunks = [
            codec.action_to_env(a.detach().float().cpu().numpy()) for a in action_list
        ]
        return torch.from_numpy(np.stack(chunks)).float().contiguous()

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "eval", **kwargs):
        """Adapt observations, sample actions, and return environment chunks."""
        del kwargs
        images, prompts = self._observations(env_obs)
        n = images.shape[0]
        if mode == "eval":
            action_list = self.generate_eval_actions(images, prompts)
            actions = self._env_actions(action_list)
            flat = actions.reshape(n, -1)
            return actions, {
                "prev_logprobs": torch.zeros_like(flat),
                "prev_values": torch.zeros((n, 1), dtype=torch.float32),
                "forward_inputs": {"action": flat},
            }
        elif mode == "train":
            action_list, replay = self.sample_train_actions(images, prompts)
            return self._env_actions(action_list), replay
        else:
            raise ValueError(f"Unsupported Cosmos3 rollout mode: {mode}")
