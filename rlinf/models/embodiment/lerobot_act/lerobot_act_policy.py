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

"""LeRobot ACT policy wrapped for RLinf rollouts."""

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy


class LeRobotACTPolicy(nn.Module, BasePolicy):
    """Runs a LeRobot ``ACTPolicy`` on RLinf env observations.

    ``predict_action_batch`` maps ``main_images`` (uint8 ``[B, H, W, 3]``) to the
    checkpoint's single camera input and ``states`` to ``observation.state``,
    runs ACT's inference path (zero latent, normalization from the checkpoint)
    and returns the first ``num_action_chunks`` actions of the predicted chunk.

    Only ``mode="eval"`` is supported for now; training needs a stochastic head
    (see RLinf issue #1785).

    Args:
        act_policy: A loaded ``lerobot`` ``ACTPolicy``.
        num_action_chunks: Number of predicted actions executed per call, at
            most the checkpoint's ``chunk_size``.
    """

    def __init__(self, act_policy: nn.Module, num_action_chunks: int):
        super().__init__()
        config = act_policy.config
        image_keys = list(config.image_features)
        if len(image_keys) != 1:
            raise ValueError(
                "LeRobotACTPolicy maps main_images to one camera, but the "
                f"checkpoint expects {len(image_keys)} image inputs: {image_keys}."
            )
        if not 1 <= num_action_chunks <= config.chunk_size:
            raise ValueError(
                f"num_action_chunks must be in [1, {config.chunk_size}] for this "
                f"checkpoint, got {num_action_chunks}."
            )
        if config.temporal_ensemble_coeff is not None:
            raise ValueError(
                "Temporal ensembling replans every step and is not supported; "
                "use a checkpoint with temporal_ensemble_coeff=None."
            )
        self.act_policy = act_policy
        self.image_key = image_keys[0]
        self.num_action_chunks = num_action_chunks
        self.action_dim = config.action_feature.shape[0]

    def _lerobot_batch(self, env_obs) -> dict[str, torch.Tensor]:
        device = next(self.act_policy.parameters()).device
        images = env_obs["main_images"].to(device)
        images = images.permute(0, 3, 1, 2).float() / 255.0
        return {
            "observation.state": env_obs["states"].to(device).float(),
            self.image_key: images,
        }

    def _predict_chunk(self, env_obs) -> torch.Tensor:
        policy = self.act_policy
        batch = policy.normalize_inputs(self._lerobot_batch(env_obs))
        batch["observation.images"] = [batch[self.image_key]]
        actions = policy.model(batch)[0][:, : self.num_action_chunks]
        return policy.unnormalize_outputs({"action": actions})["action"]

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        if mode != "eval":
            raise NotImplementedError(
                "LeRobotACTPolicy supports mode='eval' only; RL training is not "
                "implemented yet."
            )
        self.act_policy.eval()
        chunk_actions = self._predict_chunk(env_obs)
        batch_size = chunk_actions.shape[0]
        zeros = chunk_actions.new_zeros(batch_size, self.num_action_chunks)
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros[:, :1],
            "forward_inputs": {
                "action": chunk_actions.reshape(batch_size, -1),
                "model_action": chunk_actions.reshape(batch_size, -1),
            },
        }
        return chunk_actions, result

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "LeRobotACTPolicy has no training forward yet; it supports eval only."
        )
