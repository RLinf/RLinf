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

"""BEHAVIOR pi05 eval preprocessing/postprocessing (self-contained).

``BehaviorEvalProcessor`` reproduces the old openpi-based eval pipeline exactly,
using the vendored, parity-verified primitives:

  env_obs
    -> obs_processor (env keys -> observation/* dict)
    -> BehaviorInputs (R1Pro 23-dim state extraction, image key mapping)
    -> resize images to 224x224 (PIL resize_with_pad)
    -> quantile-Normalize the 23-dim state (norm_stats)
    -> tokenize prompt + discretized normalized state (PaligemmaTokenizer)
    -> pad state to the model action dim (32)
    -> model.Observation

and the inverse on actions: quantile-Unnormalize -> slice to the env action dim
(23) -> keep the first ``action_chunk`` steps.
"""

from __future__ import annotations

import numpy as np
import torch

from rlinf.data.datasets.openpi_pytorch.eval_processor import EvalProcessor
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    BehaviorInputs,
    BehaviorOutputs,
)
from rlinf.models.embodiment.openpi_pytorch.utils.image_tools import resize_with_pad
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    NormStats,
    normalize_quantile,
    unnormalize_quantile,
)
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
    PaligemmaTokenizer,
)

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    cur = x.shape[axis]
    if cur >= target_dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, target_dim - cur)
    return np.pad(x, pad_width, constant_values=0.0)


class BehaviorEvalProcessor(EvalProcessor):
    """Stateless (per-call) BEHAVIOR pi05 eval transform, matching the old path."""

    def __init__(
        self,
        norm_stats: dict[str, NormStats],
        tokenizer: PaligemmaTokenizer,
        *,
        action_chunk: int,
        action_env_dim: int = 23,
        model_action_dim: int = 32,
        image_resolution: tuple[int, int] = (224, 224),
    ):
        if "state" not in norm_stats or "actions" not in norm_stats:
            raise ValueError("norm_stats must contain 'state' and 'actions'.")
        self.state_stats = norm_stats["state"]
        self.action_stats = norm_stats["actions"]
        self.tokenizer = tokenizer
        self.action_chunk = action_chunk
        self.action_env_dim = action_env_dim
        self.model_action_dim = model_action_dim
        self.image_resolution = image_resolution
        self._inputs = BehaviorInputs(
            extract_state_from_proprio=True, use_all_wrist_images=True
        )
        self._outputs = BehaviorOutputs(action_dim=action_env_dim)

    def _obs_processor(self, env_obs: dict) -> dict:
        proc = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
            "observation/state": env_obs["states"],
        }
        wrist = env_obs.get("wrist_images")
        if wrist is not None:
            proc["observation/wrist_image"] = wrist
        return proc

    def build_observation(self, env_obs: dict, device) -> Observation:
        proc = self._obs_processor(env_obs)
        prompts = list(proc.pop("prompt"))
        np_proc = {k: _to_numpy(v) for k, v in proc.items()}
        batch_size = next(iter(np_proc.values())).shape[0]

        images = {k: [] for k in _IMAGE_KEYS}
        image_masks = {k: [] for k in _IMAGE_KEYS}
        states, tokens, token_masks = [], [], []

        for i in range(batch_size):
            sample = {k: v[i] for k, v in np_proc.items()}
            sample["prompt"] = prompts[i]
            inputs = self._inputs(sample)

            norm_state = normalize_quantile(
                np.asarray(inputs["state"], dtype=np.float32), self.state_stats
            )
            tok, tmask = self.tokenizer.tokenize(inputs["prompt"], norm_state)
            state_padded = _pad_to_dim(norm_state, self.model_action_dim)

            for key in _IMAGE_KEYS:
                resized = resize_with_pad(inputs["image"][key], *self.image_resolution)
                images[key].append(resized)
                image_masks[key].append(np.asarray(inputs["image_mask"][key]))
            states.append(state_padded)
            tokens.append(tok)
            token_masks.append(tmask)

        data = {
            "image": {
                k: torch.from_numpy(np.stack(images[k])).to(device) for k in _IMAGE_KEYS
            },
            "image_mask": {
                k: torch.from_numpy(np.stack(image_masks[k]).astype(bool)).to(device)
                for k in _IMAGE_KEYS
            },
            "state": torch.from_numpy(np.stack(states)).to(device).float(),
            "tokenized_prompt": torch.from_numpy(np.stack(tokens)).to(device).long(),
            "tokenized_prompt_mask": torch.from_numpy(
                np.stack(token_masks).astype(bool)
            ).to(device),
        }
        return Observation.from_dict(data)

    def postprocess_actions(self, model_actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize model actions and slice to the env action dim + chunk."""
        arr = model_actions.detach().float().cpu().numpy()
        out = []
        for i in range(arr.shape[0]):
            unnorm = unnormalize_quantile(arr[i], self.action_stats)
            out.append(self._outputs({"actions": unnorm})["actions"])
        actions = torch.from_numpy(np.stack(out).astype(np.float32))
        return actions[:, : self.action_chunk]
