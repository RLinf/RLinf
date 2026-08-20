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

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.pi0_fast.data_pipeline import (
    build_lerobot_batch_from_env_obs,
)
from rlinf.models.embodiment.pi0_fast.fast_replay import (
    compute_token_logprobs,
    generate_action_tokens_with_logprobs,
    replay_action_logits,
)


class PI0FastForRLActionPrediction(nn.Module, BasePolicy):
    """Adapt LeRobot PI0-Fast generation and replay to RLinf's policy API."""

    def __init__(
        self,
        policy: nn.Module,
        *,
        action_dim: int,
        num_action_chunks: int,
        max_action_tokens: int,
        image_size: int | None = None,
        temperature_train: float = 0.3,
        temperature_eval: float = 0.0,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.max_action_tokens = int(max_action_tokens)
        self.image_size = image_size
        self.temperature_train = float(temperature_train)
        self.temperature_eval = float(temperature_eval)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a supported RLinf forward pass."""
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"Unsupported forward_type for pi0_fast: {forward_type}"
        )

    def _model_dtype(self) -> torch.dtype | None:
        for param in self.parameters():
            if param.is_floating_point():
                return param.dtype
        return None

    def _prepare_lerobot_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        first_param = next(self.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        model_dtype = self._model_dtype()
        prepared = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.is_floating_point() and model_dtype is not None:
                    value = value.to(device=device, dtype=model_dtype)
                else:
                    value = value.to(device=device)
                prepared[key] = value.contiguous()
            else:
                prepared[key] = value
        prepared = self._ensure_policy_image_keys(prepared)
        return self._ensure_language_tokens(prepared, device)

    def _ensure_policy_image_keys(self, batch: dict[str, Any]) -> dict[str, Any]:
        image_features = getattr(
            getattr(self.policy, "config", None), "image_features", None
        )
        if not image_features:
            return batch

        expected_keys = list(image_features)
        if any(key in batch for key in expected_keys):
            return batch

        primary = batch.get("observation.images.image")
        wrist = batch.get("observation.images.image2")
        if wrist is None:
            wrist = batch.get("observation.images.wrist_image")

        if primary is not None and expected_keys:
            batch[expected_keys[0]] = primary
        if wrist is not None and len(expected_keys) > 1:
            batch[expected_keys[1]] = wrist
        return batch

    def _ensure_language_tokens(
        self, batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        token_key = "observation.language.tokens"
        mask_key = "observation.language.attention_mask"
        if token_key in batch and mask_key in batch:
            return batch
        if "task" not in batch:
            raise ValueError(
                "pi0_fast batch requires either tokenized language inputs or task text."
            )
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("pi0_fast prompt tokenization requires observation.state.")
        tokenizer = getattr(self.policy, "_paligemma_tokenizer", None)
        if tokenizer is None:
            raise ValueError("pi0_fast policy does not expose a PaliGemma tokenizer.")

        max_state_dim = int(getattr(self.policy.config, "max_state_dim", 32))
        if state.shape[-1] < max_state_dim:
            state = torch.nn.functional.pad(state, (0, max_state_dim - state.shape[-1]))
        state = state[..., :max_state_dim].to(dtype=torch.float32)
        bins = torch.linspace(-1, 1, 257, device=state.device)[:-1]
        discretized = torch.bucketize(state.clamp(-1, 1), bins) - 1
        discretized = discretized.clamp(0, 255).detach().cpu().tolist()

        prompts = []
        for task, state_row in zip(batch["task"], discretized, strict=True):
            cleaned_task = str(task).strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, state_row))
            prompts.append(f"Task: {cleaned_task}, State: {state_str};\n")

        tokenized = tokenizer(
            prompts,
            max_length=int(getattr(self.policy.config, "tokenizer_max_length", 200)),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch[token_key] = tokenized["input_ids"].to(device=device, dtype=torch.long)
        batch[mask_key] = tokenized["attention_mask"].to(
            device=device, dtype=torch.bool
        )
        return batch

    def _storage_tensors_from_batch(
        self, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    @torch.no_grad()
    def _generate_action_tokens_with_logprobs(
        self,
        batch: dict[str, Any],
        *,
        temperature: float,
        do_sample: bool = True,
        max_action_tokens: int | None = None,
        compute_logprobs: bool = True,
    ) -> dict[str, torch.Tensor]:
        if max_action_tokens is None:
            max_action_tokens = self.max_action_tokens
        return generate_action_tokens_with_logprobs(
            self.policy,
            batch,
            max_action_tokens=max_action_tokens,
            num_action_chunks=self.num_action_chunks,
            action_dim=self.action_dim,
            temperature=temperature,
            do_sample=do_sample,
            compute_logprobs=compute_logprobs,
        )

    def _replay_action_logits(
        self,
        forward_inputs: dict[str, torch.Tensor],
        action_tokens: torch.Tensor,
        action_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return replay_action_logits(
            self.policy, forward_inputs, action_tokens, action_token_mask
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = False,
        calculate_logprobs: bool | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate one action chunk batch for rollout or evaluation.

        Args:
            env_obs: Batched RLinf environment observations.
            mode: Rollout mode. Evaluation without replay uses native LeRobot
                inference; training returns tokens and behavior logprobs.
            compute_values: Reserved for policies with a critic head.
            calculate_logprobs: Override behavior-logprob collection.
            **kwargs: RLinf sampling parameters.

        Returns:
            Decoded actions and replay inputs needed by the actor update.
        """
        batch = build_lerobot_batch_from_env_obs(env_obs, image_size=self.image_size)
        batch = self._prepare_lerobot_batch(batch)
        temperature = float(
            kwargs.get(
                "temperature",
                self.temperature_train if mode == "train" else self.temperature_eval,
            )
        )
        do_sample = bool(kwargs.get("do_sample", temperature > 0))
        top_k = int(kwargs.get("top_k", 0))
        top_p = float(kwargs.get("top_p", 1.0))
        if top_k != 0:
            raise ValueError(
                "pi0_fast does not support top-k sampling during RL rollout; "
                f"set top_k=0, got {top_k}."
            )
        if top_p != 1.0:
            raise ValueError(
                "pi0_fast does not support top-p sampling during RL rollout; "
                f"set top_p=1.0, got {top_p}."
            )
        max_action_tokens = int(kwargs.get("max_new_tokens", self.max_action_tokens))
        if not 0 < max_action_tokens <= self.max_action_tokens:
            raise ValueError(
                "pi0_fast max_new_tokens must be in "
                f"[1, {self.max_action_tokens}], got {max_action_tokens}."
            )
        if calculate_logprobs is None:
            calculate_logprobs = mode == "train"
        if mode == "eval" and not calculate_logprobs and not compute_values:
            try:
                with torch.inference_mode():
                    actions = self.policy.predict_action_chunk(batch)
            except (AssertionError, OverflowError, ValueError):
                generated = self._generate_action_tokens_with_logprobs(
                    batch,
                    temperature=temperature,
                    do_sample=do_sample,
                    max_action_tokens=max_action_tokens,
                    compute_logprobs=False,
                )
                actions = generated["actions"]
                decode_valid = generated.get("decode_valid")
                if decode_valid is not None:
                    valid_action_mask = decode_valid.to(
                        device=actions.device, dtype=torch.bool
                    ).view(-1, 1, 1)
                    actions = torch.where(
                        valid_action_mask, actions, torch.zeros_like(actions)
                    )
            actions = actions[:, : self.num_action_chunks, : self.action_dim]
            if self.postprocessor is not None:
                actions = self.postprocessor(actions)
            return actions.detach().cpu(), {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {},
            }
        generated = self._generate_action_tokens_with_logprobs(
            batch,
            temperature=temperature,
            do_sample=do_sample,
            max_action_tokens=max_action_tokens,
            compute_logprobs=bool(calculate_logprobs),
        )
        actions = generated["actions"][:, : self.num_action_chunks, : self.action_dim]
        if self.postprocessor is not None:
            actions = self.postprocessor(actions)
        decode_valid = generated.get("decode_valid")
        if decode_valid is not None:
            valid_action_mask = decode_valid.to(
                device=actions.device, dtype=torch.bool
            ).view(-1, 1, 1)
            actions = torch.where(valid_action_mask, actions, torch.zeros_like(actions))
        prev_logprobs = None
        if calculate_logprobs and "token_logprobs" in generated:
            prev_logprobs = generated["token_logprobs"].float()
        elif calculate_logprobs:
            action_token_mask = generated["action_token_mask"].bool()
            action_logprob_mask = generated.get(
                "action_logprob_mask", action_token_mask
            ).bool()
            logits, _ = self._replay_action_logits(
                batch,
                generated["action_tokens"].long(),
                action_token_mask,
            )
            prev_logprobs, _ = compute_token_logprobs(
                logits,
                generated["action_tokens"],
                action_logprob_mask,
                temperature=temperature,
                compute_entropy=False,
            )

        forward_inputs = {
            **self._storage_tensors_from_batch(batch),
            "action": actions.reshape(actions.shape[0], -1).detach().cpu(),
            "action_tokens": generated["action_tokens"].detach().cpu(),
            "action_token_mask": generated["action_token_mask"].detach().cpu(),
        }
        if "action_logprob_mask" in generated:
            forward_inputs["action_logprob_mask"] = (
                generated["action_logprob_mask"].detach().cpu()
            )
        result = {
            "prev_logprobs": prev_logprobs.detach().cpu()
            if torch.is_tensor(prev_logprobs)
            else None,
            "prev_values": generated.get("prev_values") if compute_values else None,
            "forward_inputs": forward_inputs,
        }
        return actions.detach().cpu(), result

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Replay rollout tokens and return actor-update policy statistics.

        Args:
            forward_inputs: Cached rollout inputs and sampled action tokens.
            compute_logprobs: Return selected-token log probabilities.
            compute_entropy: Return token entropy when enabled.
            compute_values: Reserved for policies with a critic head.
            temperature: Rollout sampling temperature.
            **kwargs: Additional RLinf forward arguments, currently unused.

        Returns:
            Log probabilities, token mask, optional entropy, and no value head.
        """
        del kwargs
        if temperature is None:
            temperature = self.temperature_train
        action_tokens = forward_inputs["action_tokens"].long()
        action_token_mask = forward_inputs["action_token_mask"].bool()
        action_logprob_mask = forward_inputs.get(
            "action_logprob_mask", action_token_mask
        ).bool()
        logits, _ = self._replay_action_logits(
            forward_inputs, action_tokens, action_token_mask
        )
        token_logprobs, token_entropy = compute_token_logprobs(
            logits,
            action_tokens,
            action_logprob_mask,
            temperature=temperature,
            compute_entropy=compute_entropy,
        )
        return {
            "logprobs": token_logprobs if compute_logprobs else None,
            "logprob_mask": action_logprob_mask.to(device=logits.device),
            "entropy": token_entropy if compute_entropy else None,
            "values": None,
        }
