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
from dataclasses import dataclass
from typing import Any

import torch

from rlinf.models.embodiment.activevln.policy import ActiveVLNPolicy


@dataclass
class _EpisodeState:
    """Committed append-only observation history for one policy version."""

    step: int
    pixels: torch.Tensor
    grids: torch.Tensor
    prefix_ids: torch.Tensor
    prefix_positions: torch.Tensor
    memory: Any


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
        self.reuse_history = self.runtime_options.pop("reuse_history", True)
        self.runtime_options.setdefault("workspace_tokens", min(512, self.max_context))
        self.runtime_options.setdefault(
            "sampling_seed", int(torch.randint(0, 2**31, ()).item())
        )
        self._sampling_states = {}
        self._runtime = None
        self._episodes: dict[str, _EpisodeState] = {}
        self.history_stats = {
            "cache_hits": 0,
            "prefix_misses": 0,
            "processed_images": 0,
            "reused_images": 0,
            "reused_tokens": 0,
            "weight_invalidations": 0,
        }

    def set_global_step(self, step: int) -> None:
        """Invalidate episode state after in-place actor weight synchronization."""
        # Evaluation can synchronize new weights with the same step number.
        self._episodes.clear()
        self.history_stats["weight_invalidations"] += 1

    def _prepare_image_batches(self, image_batches):
        """Process new RGB frames together and retain CPU pixels for actor replay."""
        import numpy as np
        from transformers import BatchFeature

        transform = self.core.gpu_image_transform
        if transform is None:
            return [None] * len(image_batches)
        parameter = next(self.core.parameters())
        device = parameter.device
        rows = [[None] * len(images) for images in image_batches]
        shapes = {}
        for row, images in enumerate(image_batches):
            for index, image in enumerate(images):
                rgb = np.asarray(
                    image.convert("RGB") if hasattr(image, "convert") else image
                )
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                    raise ValueError("GPU preprocessing requires uint8 HWC RGB images")
                shapes.setdefault(rgb.shape, []).append((row, index, rgb))
        for shape, entries in shapes.items():
            frames = torch.from_numpy(np.stack([entry[2] for entry in entries]))
            pixels = transform(frames.to(device)).to(parameter.dtype).cpu()
            height, width = transform.resized_size(shape[0], shape[1])
            grid = torch.tensor([
                [1, height // transform.patch_size, width // transform.patch_size]
            ])
            for (row, index, _), patches in zip(
                entries, pixels.chunk(len(entries)), strict=True
            ):
                rows[row][index] = (patches, grid)
        return [
            BatchFeature({
                "pixel_values": torch.cat([entry[0] for entry in row]),
                "image_grid_thw": torch.cat([entry[1] for entry in row]),
            })
            if row
            else None
            for row in rows
        ]

    def _prepare_history(self, messages, images, state, encoded_images=None):
        """Tokenize the full replay prompt, processing only newly appended images."""
        from transformers import BatchFeature

        previous_images = 0 if state is None else len(state.grids)
        new_images = images[previous_images:]
        if not new_images:
            return self.prepare_observation(messages, images)
        if encoded_images is None:
            encoded_images = self.processor.image_processor(
                images=new_images, return_tensors="pt"
            )
        pixels, grids = encoded_images.pixel_values, encoded_images.image_grid_thw
        self.history_stats["processed_images"] += len(new_images)
        if state is not None:
            pixels = torch.cat((state.pixels, pixels))
            grids = torch.cat((state.grids, grids))
            self.history_stats["reused_images"] += previous_images
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_token = self.processor.image_token
        parts = text.split(image_token)
        if len(parts) != len(grids) + 1:
            raise ValueError("Image placeholders must match the episode image history")
        merge = self.processor.image_processor.merge_size**2
        expanded = parts[0] + "".join(
            image_token * (int(grid.prod()) // merge) + part
            for grid, part in zip(grids, parts[1:], strict=True)
        )
        encoded = BatchFeature(
            dict(
                self.processor.tokenizer([expanded], return_tensors="pt", padding=False)
            )
        )
        encoded["pixel_values"], encoded["image_grid_thw"] = pixels, grids
        ids = encoded.input_ids
        if ids.shape[1] + self.max_new_tokens > self.max_context:
            raise ValueError("ActiveVLN history exceeds max_context")
        if pixels.shape[0] > self.max_patches or len(grids) > self.max_images:
            raise ValueError("ActiveVLN history exceeds max_image_tokens or max_images")
        # Use the policy's CPU mRoPE builder for the complete replay history.
        from embodiinfer.policies.activevln.processor_activevln import ProcessedTurn

        turn = ProcessedTurn(ids, encoded.attention_mask, pixels, grids, text, "")
        positions, _ = self.core._positions(turn)
        return ids, positions, encoded

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
        """Drop captured addresses before replacing or moving model tensors."""
        if self._runtime is not None:
            self._sampling_states[str(self._runtime.device)] = (
                self._runtime.sampling_generator.get_state()
            )
        self._runtime = None
        self._episodes.clear()
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
        prompts, turns, states = [], [], []
        parameter = next(self.core.parameters())
        count = len(env_obs["messages"])
        if self._runtime is None or self._runtime.batch_size != count:
            self.clear_rollout_runtime()
            self._runtime = self.core.create_batched_runtime(
                batch_size=count, **self.runtime_options
            )
            if self._runtime.gpu_preprocess:
                from embodiinfer.policies.activevln.image_gpu import QwenImageTransform

                self.core.gpu_image_transform = QwenImageTransform(
                    self.processor.image_processor
                )
            sampling_state = self._sampling_states.get(str(self._runtime.device))
            if sampling_state is not None:
                self._runtime.sampling_generator.set_state(sampling_state)
        session_ids = env_obs.get("episode_session_ids", [None] * count)
        steps = env_obs.get("observation_steps", [None] * count)
        if len(session_ids) != count or len(steps) != count:
            raise ValueError(
                "Episode identifiers and steps must align with observations"
            )
        active_ids = {key for key in session_ids if key is not None}
        self._episodes = {k: v for k, v in self._episodes.items() if k in active_ids}
        histories, image_batches = [], []
        for images, key, step in zip(
            env_obs["images"], session_ids, steps, strict=True
        ):
            state = self._episodes.get(key) if self.reuse_history else None
            if state is not None and (
                step != state.step + 1 or len(images) <= len(state.grids)
            ):
                state = None
            histories.append(state)
            image_batches.append(images[0 if state is None else len(state.grids) :])
        encoded_batches = self._prepare_image_batches(image_batches)
        for messages, images, key, step, state, encoded_images in zip(
            env_obs["messages"],
            env_obs["images"],
            session_ids,
            steps,
            histories,
            encoded_batches,
            strict=True,
        ):
            if (self.reuse_history and key is not None) or encoded_images is not None:
                ids, positions, encoded = self._prepare_history(
                    messages, images, state, encoded_images
                )
            else:
                ids, positions, encoded = self.prepare_observation(messages, images)
                self.history_stats["processed_images"] += len(images)
            prompts.append((ids, positions, encoded))
            memory, start, image_start, patch_start = None, 0, 0, 0
            if state is not None:
                length = state.prefix_ids.shape[1]
                if (
                    length < ids.shape[1]
                    and torch.equal(ids[:, :length], state.prefix_ids)
                    and torch.equal(positions[:, :, :length], state.prefix_positions)
                ):
                    memory, start = state.memory, length
                    image_start, patch_start = len(state.grids), len(state.pixels)
                    self.history_stats["cache_hits"] += 1
                    self.history_stats["reused_tokens"] += start
                else:
                    self.history_stats["prefix_misses"] += 1
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            turn = ProcessedTurn(
                ids[:, start:],
                encoded.attention_mask[:, start:],
                encoded.pixel_values[patch_start:],
                encoded.image_grid_thw[image_start:],
                text,
                hashlib.sha256(text.encode()).hexdigest(),
            ).to(parameter.device, parameter.dtype)
            turns.append(
                PreparedActiveVLNTurn(
                    turn,
                    positions[:, :, start:].to(parameter.device),
                    memory,
                    int(positions.max()) + 1,
                )
            )
            states.append((key, step, encoded, ids, positions))
        if not turns:
            raise ValueError("ActiveVLN requires a nonempty observation batch")
        prepared = PreparedActiveVLNBatch(tuple(turns))
        try:
            # Capturing graphs overwrites the runtime's cached KV workspace.
            if all(turn.memory is None for turn in turns):
                self._runtime.prewarm_prepared(prepared)
            generations = self._runtime.generate(self._runtime.prefill(prepared))
        except Exception:
            self._episodes.clear()
            raise
        responses = [item.token_ids[0].cpu() for item in generations]
        output = self.pack_rollout(
            prompts,
            responses,
            [item.token_logprobs[0].cpu() for item in generations],
            sampling=(
                self.core.temperature,
                self.core.top_p,
                repetition_penalty,
                float(do_sample),
            ),
        )
        for (key, step, encoded, ids, positions), generation, response in zip(
            states, generations, responses, strict=True
        ):
            if self.reuse_history and key is not None:
                next_position = int(positions.max()) + 1
                response_positions = torch.arange(
                    next_position, next_position + response.numel()
                )[None, None].expand(3, 1, -1)
                self._episodes[key] = _EpisodeState(
                    step,
                    encoded.pixel_values,
                    encoded.image_grid_thw,
                    torch.cat((ids, response[None]), dim=1),
                    torch.cat((positions, response_positions), dim=2),
                    generation.memory,
                )
        return output
