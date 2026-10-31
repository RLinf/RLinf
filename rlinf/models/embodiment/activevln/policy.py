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

"""RLinf token actions and differentiable replay over official EmbodiInfer."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class ActiveVLNPolicy(nn.Module, BasePolicy):
    """Generate whole navigation turns and replay their categorical logprobs.

    The environment supplies complete messages and images. Each rollout call
    prefills that history with current weights; no KV survives between turns.
    Actor and rollout share parameter names for RLinf's existing weight syncers.
    """

    def __init__(
        self,
        core: nn.Module,
        processor: Any,
        *,
        max_image_tokens: int = 2048,
        max_images: int = 32,
        runtime_options: dict | None = None,
    ) -> None:
        super().__init__()
        if not 0 < core.max_new_tokens < core.max_context:
            raise ValueError("Expected 0 < max_new_tokens < max_context")
        if max_image_tokens < 1 or max_images < 1:
            raise ValueError("max_image_tokens and max_images must be positive")
        self.core = core
        self.processor = processor
        self.max_context = core.max_context
        self.max_new_tokens = core.max_new_tokens
        self.max_images = max_images
        self.max_patches = (
            max_image_tokens * core.qwen.config.vision_config.spatial_merge_size**2
        )
        self.runtime_options = dict(runtime_options or {})
        self.runtime_options.setdefault("workspace_tokens", min(512, self.max_context))
        self.runtime_options.setdefault(
            "sampling_seed", int(torch.randint(0, 2**31, ()).item())
        )
        self._sampling_states = {}
        self._runtime = None
        # EmbodiInfer reads layer parameters directly, requiring root FSDP.
        self._no_split_modules = []

    @classmethod
    def from_config(cls, cfg: Any, dtype: torch.dtype) -> ActiveVLNPolicy:
        """Load an official checkpoint and select trainable language layers."""
        from embodiinfer.policies import make_policy
        from transformers import AutoProcessor

        options = cfg.activevln
        if cfg.get("is_lora", False):
            raise ValueError(
                "EmbodiInfer reads base weights directly; LoRA is unsupported"
            )
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
        # The dependency profile pins Qwen2.5-VL's Transformers 4.51.3 layout.
        text = core.qwen.model
        count = options.get("train_last_n_layers", 2)
        if count < -1 or count > len(text.layers):
            raise ValueError("train_last_n_layers must be -1 or fit the language model")
        layers = text.layers if count == -1 else text.layers[len(text.layers) - count :]
        for layer in layers:
            layer.requires_grad_(True)
        text.norm.requires_grad_(True)
        processor = AutoProcessor.from_pretrained(
            cfg.model_path,
            trust_remote_code=False,
            local_files_only=True,
            use_fast=False,
            min_pixels=options.get("min_pixels", 3136),
            max_pixels=options.get("max_pixels", 76800),
        )
        return cls(
            core,
            processor,
            max_image_tokens=options.max_image_tokens,
            max_images=options.max_images,
            runtime_options=dict(cfg.get("embodiinfer", {})),
        )

    def get_fsdp_ignored_states(self) -> list[nn.Parameter]:
        """Keep frozen parameters replicated outside the trainable flat gradient."""
        return [
            parameter for parameter in self.parameters() if not parameter.requires_grad
        ]

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

    def prepare_observation(self, messages: list, images: list):
        """Render the server's complete history using official image mRoPE."""
        from embodiinfer.policies.activevln.modeling_activevln import (
            PreparedActiveVLNTurn,
            build_mrope_position_ids,
        )
        from embodiinfer.policies.activevln.processor_activevln import ProcessedTurn

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.processor(
            text=[text], images=images, return_tensors="pt", padding=False
        )
        if encoded.input_ids.shape[1] + self.max_new_tokens > self.max_context:
            raise ValueError("ActiveVLN history exceeds max_context")
        if (
            encoded.pixel_values.shape[0] > self.max_patches
            or len(images) > self.max_images
        ):
            raise ValueError("ActiveVLN history exceeds max_image_tokens or max_images")
        cfg = self.core.qwen.config
        positions, next_position = build_mrope_position_ids(
            encoded.input_ids,
            encoded.image_grid_thw,
            vision_start_token_id=cfg.vision_start_token_id,
            image_token_id=cfg.image_token_id,
            spatial_merge_size=cfg.vision_config.spatial_merge_size,
        )
        parameter = next(self.core.parameters())
        turn = ProcessedTurn(
            encoded.input_ids,
            encoded.attention_mask,
            encoded.pixel_values,
            encoded.image_grid_thw,
            text,
            hashlib.sha256(text.encode()).hexdigest(),
        ).to(parameter.device, parameter.dtype)
        return PreparedActiveVLNTurn(
            turn, positions.to(parameter.device), None, next_position
        )

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
        """Use EmbodiInfer's tensor batch runtime and retain exact replay inputs."""
        from embodiinfer.policies.activevln.batching_activevln import (
            PreparedActiveVLNBatch,
        )

        if top_k != 0:
            raise ValueError("EmbodiInfer ActiveVLN supports top_k=0 only")
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
        turns = tuple(
            self.prepare_observation(messages, images)
            for messages, images in zip(
                env_obs["messages"], env_obs["images"], strict=True
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
            self._runtime.prefill(PreparedActiveVLNBatch(turns))
        )
        actions, probabilities, rows = [], [], []
        for prepared, generation in zip(turns, generations, strict=True):
            turn = prepared.turn
            tokens = generation.token_ids[0]
            ids = generation.memory.token_ids[0]
            positions = generation.memory.position_ids[:, 0]
            length, response = ids.numel(), tokens.numel()
            actions.append(
                F.pad(
                    tokens,
                    (0, self.max_new_tokens - response),
                    value=self.processor.tokenizer.pad_token_id,
                )
            )
            probabilities.append(
                F.pad(generation.token_logprobs[0], (0, self.max_new_tokens - response))
            )
            rows.append(
                {
                    "input_ids": F.pad(ids, (0, self.max_context - length)),
                    "position_ids": F.pad(positions, (0, self.max_context - length)),
                    "pixel_values": F.pad(
                        turn.pixel_values,
                        (0, 0, 0, self.max_patches - turn.pixel_values.shape[0]),
                    ),
                    "image_grid_thw": F.pad(
                        turn.image_grid_thw,
                        (0, 0, 0, self.max_images - turn.image_grid_thw.shape[0]),
                    ),
                    "lengths": torch.tensor(
                        [
                            length - response,
                            length,
                            turn.pixel_values.shape[0],
                            turn.image_grid_thw.shape[0],
                        ],
                        device=ids.device,
                    ),
                    "sampling": torch.tensor(
                        [
                            self.core.temperature,
                            self.core.top_p,
                            repetition_penalty,
                            float(do_sample),
                        ],
                        dtype=torch.float64,
                        device=ids.device,
                    ),
                }
            )
        return torch.stack(actions)[:, None].cpu(), {
            "prev_logprobs": torch.stack(probabilities),
            "prev_values": None,
            "forward_inputs": {
                key: torch.stack([row[key] for row in rows]) for key in rows[0]
            },
        }

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """Dispatch the GRPO actor forward contract."""
        if forward_type != ForwardType.DEFAULT:
            raise ValueError("ActiveVLN supports the GRPO forward contract")
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        forward_inputs: dict,
        *,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ) -> dict:
        """Re-score saved response tokens with the actor's current parameters."""
        if compute_values:
            raise ValueError("ActiveVLN uses GRPO without a value head")
        probabilities, entropies, masks = [], [], []
        for index, (start, length, patches, images) in enumerate(
            forward_inputs["lengths"].tolist()
        ):
            ids = forward_inputs["input_ids"][index : index + 1, :length]
            positions = forward_inputs["position_ids"][index, :, :length][:, None]
            temperature, top_p, penalty, do_sample = forward_inputs["sampling"][
                index
            ].tolist()
            embeds = self.core.qwen.get_input_embeddings()(ids)
            with torch.no_grad():
                features = self.core.qwen.visual(
                    forward_inputs["pixel_values"][index, :patches],
                    grid_thw=forward_inputs["image_grid_thw"][index, :images],
                )
            # Sampled image-token IDs are text tokens, not new visual inputs.
            image_mask = ids == self.core.qwen.config.image_token_id
            image_mask[:, start:] = False
            embeds = embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(embeds), features.to(embeds)
            )
            logits = self.core.full_logits(ids, embeds, positions)
            scores, entropy = [], []
            for position in range(start, length):
                effective = logits[:, position - 1]
                if penalty != 1.0:
                    seen = ids[:, :position]
                    selected = effective.gather(1, seen)
                    selected = torch.where(
                        selected < 0, selected * penalty, selected / penalty
                    )
                    effective = effective.scatter(1, seen, selected)
                effective = effective.float() / temperature if do_sample else effective
                if top_p < 1.0:
                    ordered, order = effective.sort(descending=True, dim=-1)
                    remove = ordered.softmax(-1).cumsum(-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    effective = torch.full_like(effective, -torch.inf).scatter(
                        1, order, ordered.masked_fill(remove, -torch.inf)
                    )
                logprobs = effective.log_softmax(-1)
                scores.append(
                    logprobs.gather(1, ids[:, position : position + 1]).squeeze()
                )
                if compute_entropy:
                    finite = logprobs.masked_fill(~torch.isfinite(logprobs), 0)
                    entropy.append(-(logprobs.exp() * finite).sum())
            count = length - start
            probabilities.append(
                F.pad(torch.stack(scores), (0, self.max_new_tokens - count))
            )
            masks.append(torch.arange(self.max_new_tokens, device=ids.device) < count)
            if compute_entropy:
                entropies.append(
                    F.pad(torch.stack(entropy), (0, self.max_new_tokens - count))
                )
        return {
            "logprobs": torch.stack(probabilities),
            "logprob_mask": torch.stack(masks),
            "entropy": torch.stack(entropies) if compute_entropy else None,
            "values": None,
        }
