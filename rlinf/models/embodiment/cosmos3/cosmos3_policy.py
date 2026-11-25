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


def encode_prompts(prompts: list[str], width: int) -> torch.Tensor:
    """Pack environment instructions into fixed-width trajectory tensors."""
    encoded = torch.zeros((len(prompts), width), dtype=torch.uint8)
    for index, prompt in enumerate(prompts):
        data = prompt.encode("utf-8")
        if len(data) > width:
            raise ValueError(
                f"Cosmos3 prompt needs {len(data)} bytes; prompt_bytes={width}."
            )
        encoded[index, : len(data)] = torch.tensor(list(data), dtype=torch.uint8)
    return encoded


def decode_prompts(encoded: torch.Tensor) -> list[str]:
    """Decode instructions saved in an RLinf trajectory."""
    return [bytes(row[row != 0].cpu().tolist()).decode("utf-8") for row in encoded]


class Cosmos3Policy:
    """Translate RLinf environment observations and native model actions."""

    def _codec(self):
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

        if not hasattr(self, "_action_codec"):
            cfg = OmegaConf.create(
                {
                    "rollout": {
                        "model": OmegaConf.to_container(self.model_cfg, resolve=True),
                        "sglang": {},
                    },
                    "env": {
                        "eval": {"env_type": self._option("domain_name", "libero")}
                    },
                }
            )
            self._action_codec = Cosmos3SGLangAdapter(cfg, rank=0)
        return self._action_codec

    def _observations(self, env_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        codec = self._codec()
        tasks = codec._extract_tasks(env_obs)
        n_cam = 2 if env_obs.get("wrist_images") is not None else 1
        prompts = [codec._augment_prompt(task, n_cam) for task in tasks]
        images = codec.prepare_images(
            env_obs, len(tasks), codec._target_w, codec._target_h
        )
        if images is None:
            raise ValueError("Cosmos3 local inference requires camera images")
        return torch.from_numpy(images.copy()), encode_prompts(
            prompts, int(self._option("prompt_bytes", 512))
        )

    def _env_actions(self, action_list: list[torch.Tensor]) -> torch.Tensor:
        import numpy as np

        codec = self._codec()
        chunks = [
            codec._action_to_env(a.detach().float().cpu().numpy()) for a in action_list
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
