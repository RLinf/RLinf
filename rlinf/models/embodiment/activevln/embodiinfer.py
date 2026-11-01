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

"""EmbodiInfer rollout with the shared ActiveVLN replay format."""

import hashlib
from typing import Any

import torch

from rlinf.models.embodiment.activevln.policy import ActiveVLNPolicy


class EmbodiInferActiveVLNPolicy(ActiveVLNPolicy):
    """Generate responses with EmbodiInfer and export them for the HF actor."""

    def __init__(self, core, processor, *, runtime_options=None, **kwargs):
        super().__init__(
            core,
            processor,
            max_context=core.max_context,
            max_new_tokens=core.max_new_tokens,
            **kwargs,
        )
        self.runtime_options = dict(runtime_options or {})
        self.runtime_options.setdefault("workspace_tokens", min(512, self.max_context))
        self.runtime_options.setdefault(
            "sampling_seed", int(torch.randint(0, 2**31, ()).item())
        )
        self._sampling_states = {}
        self._runtime = None

    @classmethod
    def from_config(cls, cfg: Any, dtype: torch.dtype):
        """Load frozen rollout weights in the actor's checkpoint layout."""
        from embodiinfer.policies import make_policy
        from transformers import AutoProcessor

        options = cfg.activevln
        if cfg.get("is_lora", False):
            raise ValueError("EmbodiInfer rollout requires is_lora=False")
        if cfg.num_action_chunks != 1 or cfg.action_dim != options.max_new_tokens:
            raise ValueError(
                "ActiveVLN requires num_action_chunks=1 and action_dim=max_new_tokens"
            )
        core = make_policy(
            "activevln",
            checkpoint=cfg.model_path,
            attention="eager",
            max_context=options.max_context,
            max_new_tokens=options.max_new_tokens,
            action_space=options.get("action_space", "r2r"),
        ).to(dtype=dtype)
        core.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(
            cfg.model_path,
            trust_remote_code=False,
            use_fast=False,
            min_pixels=options.get("min_pixels", 3136),
            max_pixels=options.get("max_pixels", 76800),
        )
        return cls(
            core,
            processor,
            max_image_tokens=options.max_image_tokens,
            max_images=options.get("max_images", 32),
            runtime_options=cfg.get("embodiinfer"),
        )

    def clear_rollout_runtime(self) -> None:
        """Drop packed weights, graph captures and KV before refit or migration."""
        if self._runtime is not None:
            self._sampling_states[str(self._runtime.device)] = (
                self._runtime.sampling_generator.get_state()
            )
        self._runtime = None
        self.core.clear_cuda_graphs()

    def _apply(self, fn, recurse=True):
        self.clear_rollout_runtime()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_rollout_runtime()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def default_forward(self, **kwargs):
        raise RuntimeError("EmbodiInfer is rollout-only; use a Hugging Face actor")

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict,
        *,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Generate with EmbodiInfer and retain sampled tokens including EOS."""
        from embodiinfer.policies.activevln.batching_activevln import (
            PreparedActiveVLNBatch,
        )
        from embodiinfer.policies.activevln.modeling_activevln import (
            PreparedActiveVLNTurn,
        )
        from embodiinfer.policies.activevln.processor_activevln import ProcessedTurn

        if top_k not in (0, -1):
            raise ValueError("ActiveVLN requires top_k=0")
        if (
            (do_sample and temperature <= 0)
            or not 0 < top_p <= 1
            or repetition_penalty <= 0
        ):
            raise ValueError(
                "Expected positive sampling temperature/penalty and 0 < top_p <= 1"
            )
        if max_new_tokens not in (None, self.max_new_tokens):
            raise ValueError(
                "sampling max_new_tokens must match the model action capacity"
            )
        self.core.eval()
        self.core.do_sample = do_sample
        self.core.temperature = temperature if do_sample else 1.0
        self.core.top_p = top_p if do_sample else 1.0
        self.core.repetition_penalty = repetition_penalty
        prompts, turns = [], []
        parameter = next(self.core.parameters())
        for messages, images in zip(
            env_obs["messages"], env_obs["images"], strict=True
        ):
            ids, positions, encoded = self.prepare_observation(messages, images)
            prompts.append((ids, positions, encoded))
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            turn = ProcessedTurn(
                ids,
                encoded.attention_mask,
                encoded.pixel_values,
                encoded.image_grid_thw,
                text,
                hashlib.sha256(text.encode()).hexdigest(),
            ).to(parameter.device, parameter.dtype)
            turns.append(
                PreparedActiveVLNTurn(
                    turn, positions.to(parameter.device), None, int(positions.max()) + 1
                )
            )
        if not turns:
            raise ValueError("ActiveVLN requires a nonempty observation batch")
        if self._runtime is None or self._runtime.batch_size != len(turns):
            self.clear_rollout_runtime()
            self._runtime = self.core.create_batched_runtime(
                batch_size=len(turns), **self.runtime_options
            )
            state = self._sampling_states.get(str(self._runtime.device))
            if state is not None:
                self._runtime.sampling_generator.set_state(state)
        generations = self._runtime.generate(
            self._runtime.prefill(PreparedActiveVLNBatch(tuple(turns)))
        )
        return self.pack_rollout(
            prompts,
            [item.token_ids[0].cpu() for item in generations],
            [item.token_logprobs[0].cpu() for item in generations],
            sampling=(
                self.core.temperature,
                self.core.top_p,
                repetition_penalty,
                float(do_sample),
            ),
        )
