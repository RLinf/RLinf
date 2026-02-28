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

"""RLinf 'BasePolicy' adapter for starVLA checkpoints."""

from __future__ import annotations

import os
import warnings
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

from .dispatch import get_default_forward_handler, get_rollout_handler
from .utils import action_space as action_space_utils
from .utils import data_pipeline as data_pipeline_utils
from .utils import state as state_utils
from .utils.profile import (
    infer_hidden_size,
    infer_policy_profile,
    iter_gradient_checkpointing_targets,
)

logger = get_logger()


class StarVLAForRLActionPrediction(nn.Module, BasePolicy):
    """RLinf policy wrapper for starVLA checkpoints.

    This wrapper adapts a starVLA model to RLinf's embodied 'BasePolicy'
    interface. Training-time 'default_forward' and rollout-time
    'predict_action_batch' are dispatched based on the action head type inferred
    from the loaded checkpoint.

    For continuous-action heads, the wrapper can optionally unnormalize actions
    into environment space using normalization statistics from the checkpoint
    (or runtime overrides).
    """

    _WARN_KEY_DEFAULT_FORWARD_DATA_FALLBACK = "default_forward_data_fallback"
    _WARN_KEY_MISSING_ACTION_NORM_STATS = "missing_action_norm_stats"

    def __init__(
        self,
        starvla_model: nn.Module,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool = True,
        unnorm_key: Optional[str] = None,
        disable_action_unnormalization: bool = False,
    ):
        super().__init__()
        self.starvla_model = starvla_model
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.unnorm_key = unnorm_key
        self.disable_action_unnormalization = bool(disable_action_unnormalization)

        # Infer policy profile to determine action-head/VLM/state-adapter types and param dtype.
        policy_profile = infer_policy_profile(starvla_model)
        self.vlm_type = policy_profile.vlm_type
        self.action_head_type = policy_profile.action_head_type
        self.state_adapter_type = policy_profile.state_adapter_type
        self.is_continuous_action = bool(policy_profile.is_continuous_action)
        policy_param_dtype = None
        for p in starvla_model.parameters():
            if p.is_floating_point():
                policy_param_dtype = p.dtype
                break
        if policy_param_dtype is None:
            policy_param_dtype = torch.float32

        # Add value head in the wrapper for RL value estimation from hidden states.
        self.value_head: Optional[nn.Module] = None
        if add_value_head:
            hidden_size = infer_hidden_size(starvla_model)
            self.value_head = nn.Linear(hidden_size, 1).to(dtype=policy_param_dtype)

        # Diagonal Gaussian policy log-std for continuous actions.
        self.actor_logstd = nn.Parameter(
            torch.full((self.action_dim,), -2.0, dtype=policy_param_dtype)
        )

        self._warned_once: set[str] = set()
        self._rollout_prompt_seq_len: Optional[int] = None

        self._action_norm_stats: Optional[dict[str, np.ndarray]] = None
        resolved_key = self.unnorm_key
        if self.is_continuous_action and not self.disable_action_unnormalization:
            self._action_norm_stats, resolved_key = (
                action_space_utils.resolve_action_norm_stats(
                    starvla_model=self.starvla_model,
                    unnorm_key=self.unnorm_key,
                    action_dim=self.action_dim,
                )
            )
            if resolved_key is not None:
                self.unnorm_key = resolved_key
        elif self.disable_action_unnormalization and self.unnorm_key is not None:
            warnings.warn(
                "starVLA disable_action_unnormalization=True; "
                f"ignore cfg.unnorm_key={self.unnorm_key!r}.",
                stacklevel=2,
            )
            self.unnorm_key = None
        self._debug_action_stats_calls = 0

    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Dispatch forward passes for RLinf.

        Args:
            forward_type: Which forward path to run.
            **kwargs: Forward inputs forwarded to 'default_forward'.

        Returns:
            A dict containing optional RL terms: 'logprobs', 'entropy', and 'values'.

        Raises:
            NotImplementedError: If 'forward_type' is not supported.
        """
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def default_forward(
        self,
        data: Optional[dict[str, torch.Tensor]] = None,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Run training-time forward for PPO terms (logprob/entropy/value).

        This method delegates to an action-head-specific handler based on the
        inferred 'action_head_type' of the wrapped starVLA checkpoint.

        Args:
            data: Training batch dict. Usually comes from rollout caches stored
                in 'forward_inputs' produced by 'predict_action_batch'.
            forward_inputs: Backward-compatible alias for cached rollout tensors.
            compute_logprobs: Whether to compute action log-probabilities.
            compute_entropy: Whether to compute policy entropy.
            compute_values: Whether to compute value baseline.
            use_cache: Whether to enable backbone kv-cache when supported.
            **kwargs: Extra tensor fields (legacy path).

        Returns:
            Dict with optional RL terms: 'logprobs', 'entropy', and 'values'.

        Raises:
            ValueError: If no usable batch tensors are provided.
            NotImplementedError: If the action head type is not supported.
        """
        # Use provided data if available, otherwise use forward_inputs and kwargs to construct data.
        if data is None:
            used_forward_inputs = isinstance(forward_inputs, dict)
            data = {}
            if used_forward_inputs:
                data.update(
                    {
                        k: v
                        for k, v in forward_inputs.items()
                        if isinstance(v, torch.Tensor)
                    }
                )
            data.update(
                {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            )

            if not data:
                raise ValueError(
                    "starVLA.default_forward requires 'data' (or 'forward_inputs') in RLinf training."
                )
            # temporary fallback warning for cases where using older RLinf training code
            if self._WARN_KEY_DEFAULT_FORWARD_DATA_FALLBACK not in self._warned_once:
                if used_forward_inputs:
                    warnings.warn(
                        "starVLA.default_forward used 'forward_inputs' as training 'data'. "
                        "Prefer calling model(data=...) in training path.",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "starVLA.default_forward inferred 'data' from tensor kwargs because "
                        "'data' was not provided explicitly. "
                        "Prefer calling model(data=...) in training path.",
                        stacklevel=2,
                    )
                self._warned_once.add(self._WARN_KEY_DEFAULT_FORWARD_DATA_FALLBACK)
        # Automatically dispatch to the correct default forward handler based on action head type.
        handler = get_default_forward_handler(self.action_head_type)
        if handler is None:
            raise NotImplementedError(
                "default_forward not implemented for starVLA action head "
                f"{self.action_head_type}."
            )

        return handler(
            self,
            data=data,
            compute_logprobs=compute_logprobs,
            compute_entropy=compute_entropy,
            compute_values=compute_values,
            use_cache=use_cache,
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict a batch of actions and return rollout caches for training replay.

        Args:
            env_obs: Environment observation dict. Must include fields required
                by the configured state adapter and VLM pre-processing.
            calculate_logprobs: Whether to compute rollout-time logprob baseline.
            calculate_values: Whether to compute rollout-time value baseline.
            return_obs: Kept for compatibility with RLinf policy interface.
            mode: Rollout mode, typically "train" or "eval".
            **kwargs: Sampling overrides, such as 'do_sample', 'temperature',
                'top_k', 'top_p', 'max_new_tokens', and 'max_length'.

        Returns:
            Tuple of '(actions, result)' where:
                - 'actions' is a numpy array shaped [B, T, D] in env action space.
                - 'result' contains 'prev_logprobs', 'prev_values', and
                  'forward_inputs' for training replay.
        """
        del return_obs

        # Build examples based on env_obs and state adapter.
        examples = data_pipeline_utils.build_examples_from_env_obs(
            env_obs=env_obs,
            state_adapter_name=self.state_adapter_type,
            prepare_state_tensor=partial(
                state_utils.prepare_state_tensor,
                starvla_model=self.starvla_model,
                default_state_adapter_name=self.state_adapter_type,
                warned_keys=self._warned_once,
            ),
        )
        sampling_kwargs = {
            "do_sample": kwargs.pop("do_sample", False),
            "temperature": kwargs.pop("temperature", 1.0),
            "top_k": kwargs.pop("top_k", 0),
            "top_p": kwargs.pop("top_p", 1.0),
            "max_new_tokens": kwargs.pop("max_new_tokens", None),
            "max_length": kwargs.pop("max_length", None),
        }
        if mode in {"train", "training"}:
            sampling_kwargs["do_sample"] = True

        forward_inputs = data_pipeline_utils.build_sampling_param_tensors(
            do_sample=sampling_kwargs["do_sample"],
            temperature=sampling_kwargs["temperature"],
            top_k=sampling_kwargs["top_k"],
            top_p=sampling_kwargs["top_p"],
            batch_size=len(examples),
        )

        prev_logprobs: Optional[torch.Tensor] = None
        prev_values: Optional[torch.Tensor] = None
        output: dict[str, Any]
        model_inputs: dict[str, Any] = {}
        extra_forward_inputs: dict[str, Any] = {}
        state_for_storage: Optional[torch.Tensor] = None
        # Automatically dispatch to the correct rollout handler based on action head type.
        rollout_handler = get_rollout_handler(self.action_head_type)
        if rollout_handler is not None:
            payload = rollout_handler(
                self,
                examples=examples,
                env_obs=env_obs,
                mode=mode,
                calculate_logprobs=calculate_logprobs,
                calculate_values=calculate_values,
                sampling_kwargs=sampling_kwargs,
            )
            output = payload["output"]
            model_inputs = payload.get("model_inputs", {})
            if not isinstance(model_inputs, dict):
                model_inputs = {}
            prev_logprobs = payload.get("prev_logprobs")
            prev_values = payload.get("prev_values")
            extra_forward_inputs = payload.get("extra_forward_inputs", {})
            state_for_storage = payload.get("state")
        else:
            if hasattr(self.starvla_model, "predict_action"):
                output = self.starvla_model.predict_action(examples=examples)
            else:
                raise NotImplementedError(
                    "Unsupported starVLA model for rollout fallback: "
                    f"action_head={self.action_head_type}, vlm={self.vlm_type}."
                )

        if model_inputs:
            model_inputs, target_len = (
                data_pipeline_utils.normalize_model_inputs_for_storage(
                    model_inputs=model_inputs,
                    starvla_model=self.starvla_model,
                    rollout_prompt_seq_len=self._rollout_prompt_seq_len,
                )
            )
            self._rollout_prompt_seq_len = target_len
            model_inputs = data_pipeline_utils.pack_model_inputs_for_storage(
                model_inputs=model_inputs,
                batch_size=len(examples),
            )

        normalized_actions = np.asarray(output["normalized_actions"])
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[:, None, :]
        chunk_actions = normalized_actions.astype(np.float32)

        bsz, n_chunks, act_dim = normalized_actions.shape
        if act_dim != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: model returns {act_dim}, expected {self.action_dim}"
            )
        if n_chunks != self.num_action_chunks:
            raise ValueError(
                f"num_action_chunks mismatch: model returns {n_chunks}, expected {self.num_action_chunks}"
            )

        env_chunk_actions = chunk_actions
        action_for_storage = chunk_actions
        if self.is_continuous_action:
            missing_norm_stats_warned = (
                self._WARN_KEY_MISSING_ACTION_NORM_STATS in self._warned_once
            )
            warned_missing = (
                True
                if self.disable_action_unnormalization
                else missing_norm_stats_warned
            )
            env_chunk_actions, warned_missing = (
                action_space_utils.unnormalize_actions_for_env(
                    normalized_actions=chunk_actions,
                    action_norm_stats=None
                    if self.disable_action_unnormalization
                    else self._action_norm_stats,
                    warned_missing_action_norm_stats=warned_missing,
                )
            )
            if not self.disable_action_unnormalization and warned_missing:
                self._warned_once.add(self._WARN_KEY_MISSING_ACTION_NORM_STATS)
            action_for_storage = env_chunk_actions

        self._maybe_debug_actions(
            normalized_actions=chunk_actions,
            env_actions=env_chunk_actions,
        )

        if prev_logprobs is None:
            prev_logprobs = torch.zeros(
                (bsz, n_chunks, self.action_dim), dtype=torch.float32
            )
        if prev_values is None:
            prev_values = torch.zeros((bsz, 1), dtype=torch.float32)

        forward_inputs["action"] = torch.from_numpy(action_for_storage.reshape(bsz, -1))

        storage_inputs = dict(model_inputs)
        storage_inputs.update(extra_forward_inputs)
        forward_inputs.update(
            {k: v for k, v in storage_inputs.items() if isinstance(v, torch.Tensor)}
        )
        if state_for_storage is not None:
            forward_inputs["state"] = state_for_storage.detach().cpu()
        for key, value in list(forward_inputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value
            if tensor.ndim == 0:
                tensor = tensor.view(1).repeat(bsz)
            elif tensor.shape[0] == bsz:
                pass
            elif tensor.shape[0] == 1:
                tensor = tensor.expand(bsz, *tensor.shape[1:]).clone()
            else:
                raise RuntimeError(
                    f"forward_inputs['{key}'] has leading dim {tensor.shape[0]}, "
                    f"but rollout batch size is {bsz}. "
                    "Expected scalar/[1,...]/[B,...] tensor for trajectory splitting."
                )
            forward_inputs[key] = tensor

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return env_chunk_actions, result

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enable gradient checkpointing on supported starVLA submodules.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs forwarded to submodules
                that support 'gradient_checkpointing_enable'.
        """
        enabled = False
        warned_types: set[type] = set()
        for module in iter_gradient_checkpointing_targets(self.starvla_model):
            fn = getattr(module, "gradient_checkpointing_enable", None)
            if not callable(fn):
                continue
            try:
                if gradient_checkpointing_kwargs is None:
                    fn()
                else:
                    try:
                        fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                    except TypeError:
                        fn()
            except ValueError as exc:
                if type(module) not in warned_types:
                    warnings.warn(
                        f"gradient_checkpointing_enable skipped for {type(module).__name__}: {exc}",
                        stacklevel=2,
                    )
                    warned_types.add(type(module))
                continue
            enabled = True

        if not enabled:
            warnings.warn(
                "gradient_checkpointing_enable() was requested, but no wrapped starVLA "
                "submodule exposes this API.",
                stacklevel=2,
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on supported starVLA submodules."""
        disabled = False
        warned_types: set[type] = set()
        for module in iter_gradient_checkpointing_targets(self.starvla_model):
            fn = getattr(module, "gradient_checkpointing_disable", None)
            if not callable(fn):
                continue
            try:
                fn()
            except ValueError as exc:
                if type(module) not in warned_types:
                    warnings.warn(
                        f"gradient_checkpointing_disable skipped for {type(module).__name__}: {exc}",
                        stacklevel=2,
                    )
                    warned_types.add(type(module))
                continue
            disabled = True

        if not disabled:
            warnings.warn(
                "gradient_checkpointing_disable() was requested, but no wrapped starVLA "
                "submodule exposes this API.",
                stacklevel=2,
            )

    def _compute_values_from_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.value_head is None:
            return torch.zeros(
                (hidden.shape[0], 1), device=hidden.device, dtype=torch.float32
            )

        if attention_mask is not None:
            idx = attention_mask.long().sum(dim=1) - 1
            feat = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        else:
            feat = hidden[:, -1]

        return self.value_head(feat.float()).to(dtype=torch.float32)

    def _maybe_debug_actions(
        self,
        *,
        normalized_actions: np.ndarray,
        env_actions: np.ndarray,
    ) -> None:
        """Optionally log action statistics for debugging.

        Enable by setting 'STARVLA_DEBUG_ACTIONS=1'. Frequency and thresholds can
        be controlled with:
            - 'STARVLA_DEBUG_ACTIONS_EVERY' (default: 50)
            - 'STARVLA_DEBUG_ACTIONS_ZERO_THRESH' (default: 0.005)
        """
        self._debug_action_stats_calls += 1

        enabled = str(os.environ.get("STARVLA_DEBUG_ACTIONS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return

        try:
            every = int(os.environ.get("STARVLA_DEBUG_ACTIONS_EVERY", "50"))
        except ValueError:
            every = 50
        if every <= 0:
            every = 1
        if self._debug_action_stats_calls % every != 0:
            return

        try:
            zero_thresh = float(
                os.environ.get("STARVLA_DEBUG_ACTIONS_ZERO_THRESH", "0.005")
            )
        except ValueError:
            zero_thresh = 0.005

        bsz, n_chunks, act_dim = normalized_actions.shape
        norm_flat = normalized_actions.reshape(-1, act_dim)
        env_flat = env_actions.reshape(-1, act_dim)

        norm_mean_abs = np.mean(np.abs(norm_flat), axis=0)
        env_mean_abs = np.mean(np.abs(env_flat), axis=0)
        env_min = np.min(env_flat, axis=0)
        env_max = np.max(env_flat, axis=0)

        arm_dims = min(6, act_dim)
        arm_near_zero_ratio = (
            float((np.abs(env_flat[:, :arm_dims]) < zero_thresh).mean())
            if arm_dims > 0
            else 0.0
        )

        prefix = (
            f"[starvla-action-debug] call={self._debug_action_stats_calls} "
            f"action_head={self.action_head_type} "
            f"vlm={self.vlm_type} "
            f"unnorm_key={self.unnorm_key} "
            f"disable_unnorm={self.disable_action_unnormalization} "
            f"has_norm_stats={self._action_norm_stats is not None} "
            f"shape=({bsz},{n_chunks},{act_dim})"
        )
        logger.info(prefix)
        logger.info(
            "[starvla-action-debug] norm_mean_abs=%s",
            np.array2string(norm_mean_abs, precision=4, suppress_small=True),
        )
        logger.info(
            "[starvla-action-debug] env_mean_abs=%s env_min=%s env_max=%s "
            "arm_near_zero_ratio(|a|<%s)=%0.4f",
            np.array2string(env_mean_abs, precision=4, suppress_small=True),
            np.array2string(env_min, precision=4, suppress_small=True),
            np.array2string(env_max, precision=4, suppress_small=True),
            zero_thresh,
            arm_near_zero_ratio,
        )
        logger.info(
            "[starvla-action-debug] sample0_chunk0_norm=%s sample0_chunk0_env=%s",
            np.array2string(normalized_actions[0, 0], precision=4, suppress_small=True),
            np.array2string(env_actions[0, 0], precision=4, suppress_small=True),
        )
