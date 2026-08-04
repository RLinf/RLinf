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

"""DreamZero embodied sglang adapter backed by an SGLang action server.

This module is the SGLang counterpart of
``rlinf.models.embodiment.dreamzero.dreamzero_policy.DreamZeroPolicy`` for eval
rollout.  It intentionally does not import or construct the HF policy/model:
the large DreamZero network lives in a separately spawned ``sglang serve``
process.  The adapter keeps only the lightweight parts required on the rollout
worker:

1. convert RLinf env observations to DreamZero modality keys;
2. run DreamZero dataset transforms and metadata-based normalization;
3. build the request for the SGLang ``/v1/actions/generations`` action endpoint
   (the HTTP POST itself is performed by ``SGLangEmbodiedWorker``);
4. invert normalized actions back to environment-scale action chunks.

For a Libero batch of size ``B`` and action horizon ``H``, the main shape flow is
roughly:

``env_obs`` -> converted obs -> normalized input -> server action
``[B, H, max_action_dim]`` -> unnormalized env action ``[B, H, action_dim]``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

from rlinf.data.datasets.dreamzero.data_transforms import (
    build_dreamzero_composed_transform,
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    format_training_prompt,
    load_dreamzero_dataset_metadata,
    normalize_instruction_text,
    rollout_obs_layout_for_embodiment,
)

_RLINF_POLICY_CONTEXT_KEYS = ("_rlinf_stage_id", "_rlinf_reset")


class DreamZeroSGLangAdapter:
    """DreamZero env-obs <-> action-chunks adapter for the sglang serve path.

    Combines the DreamZero dataset transforms (reused without the HF model) with
    the sglang action-request assembly:

    - :meth:`build_request` turns an env observation into the msgpack action
      payload (plus a state handed back to :meth:`parse_response`);
    - :meth:`parse_response` turns the server response into action chunks.

    The HTTP round-trip itself is performed by :class:`SGLangEmbodiedWorker`,
    which posts :attr:`action_path`.
    """

    action_path = "/v1/actions/generations"

    def __init__(self, cfg: Any, rank: int):
        self.cfg_rollout = cfg.rollout
        self.model_cfg = cfg.rollout.model
        self.rank = rank
        sglang_cfg = self.cfg_rollout.get("sglang", {})
        self._model = sglang_cfg.get("model", str(self.model_cfg.model_path))
        self._seed = int(sglang_cfg.get("seed", 1140))

        # DreamZero dataset transforms (obs normalization + action unnormalization).
        model_cfg = self.model_cfg.copy()
        self.embodiment_tag = str(model_cfg.embodiment_tag)
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(
            self.embodiment_tag
        )
        tokenizer_path = os.path.join(str(model_cfg.model_path), "tokenizer")
        self.data_transforms = build_dreamzero_composed_transform(
            model_cfg, tokenizer_path
        )
        self.data_transforms.set_metadata(load_dreamzero_dataset_metadata(model_cfg))
        self.data_transforms.eval()
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            self.data_transforms, self.embodiment_tag
        )
        self._action_keys = tuple(action_keys)
        self._dream_transform = self.data_transforms.transforms[-1]
        self._relative_action = bool(model_cfg.get("relative_action", False))
        self._relative_action_per_horizon = bool(
            model_cfg.get("relative_action_per_horizon", False)
        )
        self._relative_action_keys = list(model_cfg.get("relative_action_keys") or [])

    def build_request(
        self, env_obs: dict, mode: Literal["train", "eval"] = "eval"
    ) -> tuple[dict, dict]:
        """env_obs -> (msgpack action payload, state for :meth:`parse_response`).

        The returned state (converted observation) is handed back to
        :meth:`parse_response` so action unnormalization can use it. The worker
        performs the HTTP POST in between.
        """
        if mode != "eval":
            raise NotImplementedError("DreamZero sglang adapter supports eval only.")
        rollout_obs, context = self._split_context(env_obs)
        converted = self._observation_convert(rollout_obs)
        normalized = self._normalize_obs(converted)
        return self._build_payload(normalized, context), converted

    def parse_response(
        self, resp: dict, converted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict]:
        """Server response -> action chunks ``[B, H, action_dim]`` and info dict."""
        try:
            values = resp["data"][0]["action"]["values"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"DreamZero action response missing data[0].action.values: {resp}"
            ) from exc
        act = self._unapply(torch.as_tensor(values, dtype=torch.float32), converted_obs)
        actions = torch.as_tensor(
            self._actions_from_unapply(act).astype(np.float32, copy=False),
            dtype=torch.float32,
        )
        flat = actions.reshape(actions.shape[0], -1)
        info = {
            "prev_logprobs": torch.zeros_like(flat),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": {"action": flat.cpu()},
        }
        return actions, info

    def _observation_convert(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Map RLinf env observation keys to DreamZero dataset modality keys.

        Example input keys are ``main_images``, ``wrist_images``, ``states`` and
        ``task_descriptions``.  The converted observation is the same structure
        consumed by the HF policy's DreamZero data transforms.
        """
        converted = convert_rollout_env_obs(self.embodiment_tag, env_obs)
        tasks = converted.get("annotation.task")
        if isinstance(tasks, list) and all(isinstance(item, str) for item in tasks):
            converted["annotation.task"] = np.asarray(tasks, dtype=object)
        return converted

    def _normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Apply DreamZero dataset transforms to obtain server model inputs.

        The output contains normalized tensors such as images/video, state,
        ``embodiment_id`` and server-side prompt strings. Text tokenization is
        owned by the SGLang server.
        """
        data = obs
        for transform in self.data_transforms.transforms[:-1]:
            data = transform(data)
        return dict(self._apply_dream_transform_without_tokenizer(data))

    def _apply_dream_transform_without_tokenizer(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        import tree

        data = dict(data)
        if not self._dream_transform.training and data["video"].ndim == 5:
            data["video"] = data["video"][None, ...]
        is_batched, batch_size = self._dream_transform.check_keys_and_batch_size(data)
        if is_batched:
            samples = [
                self._dream_transform.apply_single(
                    tree.map_structure(lambda x: x[i], data)
                )
                for i in range(batch_size)
            ]
        else:
            samples = [self._dream_transform.apply_single(data)]
        return self._collate_server_dream_batch(samples)

    def _collate_server_dream_batch(
        self, features: list[dict[str, Any]]
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for key in features[0]:
            if key == "text":
                batch["_dreamzero_prompt_texts"] = [
                    format_training_prompt(
                        normalize_instruction_text(elem[key]),
                        int(elem["embodiment_id"]),
                        self._dream_transform.embodiment_tag_mapping,
                    )
                    for elem in features
                ]
            elif key == "text_negative":
                batch["_dreamzero_negative_prompt_texts"] = [
                    str(elem[key]) for elem in features
                ]
            else:
                values = [elem[key] for elem in features]
                try:
                    batch[key] = torch.from_numpy(np.stack(values))
                except ValueError as e:
                    shapes = [np.asarray(v).shape for v in values]
                    raise ValueError(
                        f"Shape mismatch in collate for key='{key}': shapes={shapes}"
                    ) from e
        return batch

    def _unapply(
        self, normalized_action: torch.Tensor, obs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Invert normalized actions back to environment-scale action tensors.

        ``normalized_action`` comes from the SGLang server and has the padded
        DreamZero model width, e.g. ``[B, H, max_action_dim]``.  The reverse
        transform slices and unnormalizes it according to metadata, producing
        keys such as ``action.actions`` with env width, e.g. ``[B, H, 7]`` for
        Libero.
        """
        unnormalized_action = self.data_transforms.unapply(
            {"action": normalized_action.cpu()}
        )
        if (
            (self._relative_action or self._relative_action_per_horizon)
            and self._relative_action_keys
            and obs is not None
        ):
            for key in self._relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"
                if action_key not in unnormalized_action:
                    continue
                last_state = obs.get(state_key)
                if last_state is None:
                    for obs_key, obs_value in obs.items():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs_value
                            break
                if last_state is None and "state" in obs:
                    state_data = obs["state"]
                    action_dim = unnormalized_action[action_key].shape[-1]
                    state_dim = (
                        state_data.shape[-1] if hasattr(state_data, "shape") else None
                    )
                    if state_dim == action_dim:
                        last_state = state_data
                if last_state is None:
                    continue
                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(last_state, axis=-2)
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )
        return unnormalized_action

    def _actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        """Concatenate unnormalized action modalities in dataset action order."""
        parts: list[np.ndarray] = []
        for key in self._action_keys:
            if key not in act_dict:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; "
                    f"available keys: {sorted(act_dict)}."
                )
            value = act_dict[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        actions = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )
        return actions

    @staticmethod
    def _split_context(env_obs: Any) -> tuple[Any, dict[str, Any]]:
        """Separate RLinf routing metadata (e.g. stage id) from the observation."""
        if not isinstance(env_obs, Mapping):
            return env_obs, {}
        context = {k: env_obs[k] for k in _RLINF_POLICY_CONTEXT_KEYS if k in env_obs}
        if not context:
            return env_obs, {}
        cleaned = {
            k: v for k, v in env_obs.items() if k not in _RLINF_POLICY_CONTEXT_KEYS
        }
        return cleaned, context

    def _build_payload(
        self, normalized: dict[str, Any], context: dict[str, Any]
    ) -> dict:
        """Assemble the DreamZero action-endpoint request payload."""
        observation = dict(normalized)
        prompts = observation.pop("_dreamzero_prompt_texts", None)
        neg_prompts = observation.pop("_dreamzero_negative_prompt_texts", None)
        for key in (
            "text",
            "text_attention_mask",
            "text_negative",
            "text_attention_mask_negative",
        ):
            observation.pop(key, None)

        batch_size = self._infer_batch_size(observation)
        stage_id = self._stage_id(context)
        session_ids = [
            f"rlinf-eval-r{self.rank}-stage{stage_id}-slot{i}"
            for i in range(batch_size)
        ]
        reset = bool(context.get("_rlinf_reset", False))
        reset_mask = [reset] * batch_size
        return {
            "model": self._model,
            "input": {
                "prompt": self._as_texts(prompts, batch_size, "prompt"),
                "observation": observation,
            },
            "parameters": {
                "session_ids": session_ids,
                "reset_mask": reset_mask,
                "negative_prompts": self._as_texts(
                    neg_prompts, batch_size, "negative_prompt", default=""
                ),
                "seed": self._seed,
            },
            "runtime": {"response_format": "envelope", "output_format": "numpy"},
        }

    @staticmethod
    def _stage_id(context: dict[str, Any]) -> int:
        stage_id = context.get("_rlinf_stage_id", 0)
        if torch.is_tensor(stage_id) or isinstance(stage_id, np.ndarray):
            stage_id = stage_id.item()
        return int(stage_id)

    @staticmethod
    def _infer_batch_size(value: Any) -> int:
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, Mapping):
            for item in value.values():
                try:
                    return DreamZeroSGLangAdapter._infer_batch_size(item)
                except (TypeError, IndexError):
                    continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
        raise TypeError("Unable to infer DreamZero rollout batch size.")

    @staticmethod
    def _as_texts(
        value: Any, batch_size: int, name: str, default: str | None = None
    ) -> list[str]:
        if value is None:
            if default is None:
                raise ValueError(f"DreamZero {name} is required")
            return [default] * batch_size
        if isinstance(value, str):
            return [value] * batch_size
        values = list(value)
        if len(values) != batch_size or not all(isinstance(v, str) for v in values):
            raise ValueError(
                f"DreamZero {name} must be {batch_size} strings, got {values!r}"
            )
        return values
