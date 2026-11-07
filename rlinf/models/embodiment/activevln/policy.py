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

"""ActiveVLN generation and replay under RLinf's embodied policy contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalize each token in the current response once, excluding the prompt."""
    if penalty == 1.0 or generated_ids.shape[-1] == 0:
        return logits
    seen = torch.zeros_like(logits, dtype=torch.bool)
    seen.scatter_(1, generated_ids, True)
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen, penalized, logits)


class HuggingFaceActiveVLNModel(nn.Module):
    """Keep the native checkpoint layout shared with the SGLang actor."""

    def __init__(
        self, qwen: nn.Module, *, gradient_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        image_features: torch.Tensor,
        image_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-history hidden states for response scoring."""
        text = getattr(self.qwen.model, "language_model", self.qwen.model)
        hidden = self.qwen.get_input_embeddings()(input_ids)
        hidden = hidden.masked_scatter(
            image_token_mask.unsqueeze(-1).expand_as(hidden), image_features.to(hidden)
        )

        def text_forward(embeddings):
            return text(
                inputs_embeds=embeddings,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(text_forward, hidden, use_reentrant=False)
        return text_forward(hidden)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project only positions used by the response loss."""
        return self.qwen.lm_head(hidden)


class ActiveVLNPolicy(nn.Module, BasePolicy):
    """Generate token actions and replay their saved images and sampling settings."""

    def __init__(
        self,
        core: nn.Module,
        processor: Any,
        *,
        max_context: int = 4096,
        max_new_tokens: int = 64,
        max_image_tokens: int = 2048,
        max_images: int = 32,
        model_backend: str = "huggingface",
    ) -> None:
        super().__init__()
        if not 0 < max_new_tokens < max_context:
            raise ValueError("Expected 0 < max_new_tokens < max_context")
        if max_image_tokens < 1 or max_images < 1:
            raise ValueError("max_image_tokens and max_images must be positive")
        if model_backend not in ("huggingface", "sglang"):
            raise ValueError("ActiveVLN model_backend must be huggingface or sglang")
        self.core = core
        self.processor = processor
        self.model_backend = model_backend
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.max_images = max_images
        self.max_patches = (
            max_image_tokens * core.qwen.config.vision_config.spatial_merge_size**2
        )
        self._no_split_modules = []

    @classmethod
    def from_config(cls, cfg: Any, dtype: torch.dtype) -> ActiveVLNPolicy:
        """Load a checkpoint and select the actor implementation and language layers."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if cfg.get("is_lora", False):
            raise ValueError(
                "ActiveVLN uses direct language-layer training; is_lora must be False"
            )
        options = cfg.activevln
        backend = cfg.get("model_backend", "huggingface")
        if backend not in ("huggingface", "sglang"):
            raise ValueError("ActiveVLN model_backend must be huggingface or sglang")
        if cfg.num_action_chunks != 1 or cfg.action_dim != options.max_new_tokens:
            raise ValueError(
                "ActiveVLN requires num_action_chunks=1 and action_dim=max_new_tokens"
            )
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        if backend == "sglang":
            from sglang.multimodal_gen.runtime.models.vlas.activevln import (
                ActiveVLNModel,
            )

            core = ActiveVLNModel(
                qwen,
                gradient_checkpointing=options.get("gradient_checkpointing", False),
            )
        else:
            core = HuggingFaceActiveVLNModel(
                qwen,
                gradient_checkpointing=options.get("gradient_checkpointing", False),
            )
        qwen.requires_grad_(False)
        text = getattr(qwen.model, "language_model", qwen.model)
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
            use_fast=False,
            min_pixels=options.get("min_pixels", 3136),
            max_pixels=options.get("max_pixels", 76800),
        )
        return cls(
            core,
            processor,
            model_backend=backend,
            max_context=options.max_context,
            max_new_tokens=options.max_new_tokens,
            max_image_tokens=options.max_image_tokens,
            max_images=options.get("max_images", 32),
        )

    def get_fsdp_ignored_states(self) -> list[nn.Parameter]:
        """Keep frozen weights outside the trainable flat gradient."""
        return [p for p in self.parameters() if not p.requires_grad]

    def named_srt_weights(
        self, *, trainable_only: bool = False
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Export either actor's checkpoint names for SRT's fused weight loader."""
        for name, parameter in self.core.qwen.named_parameters(remove_duplicate=False):
            if trainable_only and not parameter.requires_grad:
                continue
            if name.startswith("model.language_model."):
                name = "model." + name.removeprefix("model.language_model.")
            elif name.startswith("model.visual."):
                name = name.removeprefix("model.")
            yield name, parameter.detach()

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs) -> dict:
        if forward_type != ForwardType.DEFAULT:
            raise ValueError("ActiveVLN supports the GRPO forward contract")
        return self.default_forward(**kwargs)

    def prepare_observation(
        self, messages: list, images: list
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Render complete histories with the checkpoint processor and image mRoPE."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.processor(
            text=[text], images=images, return_tensors="pt", padding=False
        )
        ids = encoded.input_ids
        if ids.shape[1] + self.max_new_tokens > self.max_context:
            raise ValueError("ActiveVLN history exceeds max_context")
        if (
            encoded.pixel_values.shape[0] > self.max_patches
            or encoded.image_grid_thw.shape[0] > self.max_images
        ):
            raise ValueError("ActiveVLN history exceeds max_image_tokens or max_images")
        qwen = self.core.qwen
        rope_model = qwen.model if hasattr(qwen.model, "get_rope_index") else qwen
        rope_inputs = {}
        if "mm_token_type_ids" in encoded:
            rope_inputs["mm_token_type_ids"] = encoded.mm_token_type_ids
        positions, _ = rope_model.get_rope_index(
            input_ids=ids,
            image_grid_thw=encoded.image_grid_thw,
            attention_mask=encoded.attention_mask,
            **rope_inputs,
        )
        return ids, positions, encoded

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
        """Generate a batch with Hugging Face and retain sampled tokens including EOS."""
        if top_k not in (0, -1):
            raise ValueError("ActiveVLN requires top_k=0")
        if (
            (do_sample and temperature <= 0)
            or not 0 < top_p <= 1
            or not 0 < repetition_penalty <= 2
        ):
            raise ValueError(
                "Expected positive sampling temperature, 0 < top_p <= 1, "
                "and 0 < repetition_penalty <= 2"
            )
        if max_new_tokens not in (None, self.max_new_tokens):
            raise ValueError(
                "sampling max_new_tokens must match the model action capacity"
            )
        prompts = [
            self.prepare_observation(messages, images)
            for messages, images in zip(
                env_obs["messages"], env_obs["images"], strict=True
            )
        ]
        if not prompts:
            raise ValueError("ActiveVLN requires a nonempty observation batch")
        self.core.eval()
        device = next(self.parameters()).device
        width = max(ids.shape[1] for ids, _, _ in prompts)
        pad = self.processor.tokenizer.pad_token_id
        input_ids = torch.cat(
            [F.pad(ids, (width - ids.shape[1], 0), value=pad) for ids, _, _ in prompts]
        ).to(device)
        attention_mask = torch.cat(
            [
                F.pad(torch.ones_like(ids), (width - ids.shape[1], 0))
                for ids, _, _ in prompts
            ]
        ).to(device)
        # SRT applies repetition penalties to the current response only.
        params = {"do_sample": do_sample, "repetition_penalty": 1.0}
        if repetition_penalty != 1.0:

            def penalize_response(input_ids, scores):
                return _apply_repetition_penalty(
                    scores, input_ids[:, width:], repetition_penalty
                )

            params["logits_processor"] = [penalize_response]
        if do_sample:
            params.update(temperature=temperature, top_k=0, top_p=top_p)
        if "mm_token_type_ids" in prompts[0][2]:
            params["mm_token_type_ids"] = torch.cat(
                [
                    F.pad(encoded.mm_token_type_ids, (width - ids.shape[1], 0))
                    for ids, _, encoded in prompts
                ]
            ).to(device)
        generated = self.core.qwen.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=torch.cat(
                [encoded.pixel_values for _, _, encoded in prompts]
            ).to(device),
            image_grid_thw=torch.cat(
                [encoded.image_grid_thw for _, _, encoded in prompts]
            ).to(device),
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=True,
            use_cache=True,
            **params,
        )
        scores = []
        for offset, logits in enumerate(generated.logits):
            effective = _apply_repetition_penalty(
                logits.float(),
                generated.sequences[:, width : width + offset],
                repetition_penalty,
            )
            if do_sample:
                effective = effective / temperature
            # Like SRT, report logprobs before top-p truncation.
            scores.append(
                effective.log_softmax(-1).gather(
                    1, generated.sequences[:, width + offset : width + offset + 1]
                )
            )
        scores = torch.cat(scores, dim=1).cpu()
        eos = self.core.qwen.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        responses, logprobs = [], []
        for row in range(len(prompts)):
            tokens = generated.sequences[row, width:].cpu()
            count = next(
                (i + 1 for i, token in enumerate(tokens.tolist()) if token in eos),
                len(tokens),
            )
            responses.append(tokens[:count])
            logprobs.append(scores[row, :count])
        return self.pack_rollout(
            prompts,
            responses,
            logprobs,
            sampling=(
                temperature if do_sample else 1.0,
                top_p if do_sample else 1.0,
                repetition_penalty,
                float(do_sample),
            ),
        )

    def pack_rollout(
        self,
        prompts: list,
        responses: list[torch.Tensor],
        logprobs: list[torch.Tensor],
        *,
        sampling: tuple[float, float, float, float],
    ) -> tuple[torch.Tensor, dict]:
        """Pack both rollout backends into the shared ActiveVLN replay schema."""
        rows, actions, old_probs = [], [], []
        for (prompt, positions, encoded), response, probs in zip(
            prompts, responses, logprobs, strict=True
        ):
            count = len(response)
            if not 0 < count <= self.max_new_tokens or probs.numel() != count:
                raise ValueError(
                    f"Expected 1..{self.max_new_tokens} response tokens with logprobs"
                )
            ids = torch.cat((prompt[0], response))
            start, length = prompt.shape[1], ids.numel()
            next_position = int(positions.max()) + 1
            response_pos = torch.arange(next_position, next_position + count)[
                None
            ].expand(3, -1)
            positions = torch.cat((positions[:, 0], response_pos), -1)
            pixels, grids = encoded.pixel_values, encoded.image_grid_thw
            rows.append(
                {
                    "input_ids": F.pad(ids, (0, self.max_context - length)),
                    "position_ids": F.pad(positions, (0, self.max_context - length)),
                    "pixel_values": F.pad(
                        pixels.to(next(self.core.parameters()).dtype),
                        (0, 0, 0, self.max_patches - pixels.shape[0]),
                    ),
                    "image_grid_thw": F.pad(
                        grids, (0, 0, 0, self.max_images - grids.shape[0])
                    ),
                    "lengths": torch.tensor(
                        [start, length, pixels.shape[0], grids.shape[0]]
                    ),
                    "sampling": torch.tensor(sampling, dtype=torch.float64),
                }
            )
            actions.append(
                F.pad(
                    response,
                    (0, self.max_new_tokens - count),
                    value=self.processor.tokenizer.pad_token_id,
                )
            )
            old_probs.append(F.pad(probs, (0, self.max_new_tokens - count)))
        inputs = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        return torch.stack(actions)[:, None], {
            "prev_logprobs": torch.stack(old_probs),
            "prev_values": None,
            "forward_inputs": inputs,
        }

    def default_forward(
        self,
        forward_inputs: dict,
        *,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ) -> dict:
        """Re-score saved response tokens using the selected actor backend."""
        if compute_values:
            raise ValueError("ActiveVLN uses GRPO without a value head")
        probs, entropies, masks = [], [], []
        qwen = self.core.qwen
        visual = qwen.visual if hasattr(qwen, "visual") else qwen.model.visual
        visual.eval()
        for index, (start, length, patches, images) in enumerate(
            forward_inputs["lengths"].tolist()
        ):
            ids = forward_inputs["input_ids"][index : index + 1, :length]
            positions = forward_inputs["position_ids"][index, :, :length][:, None]
            with torch.no_grad():
                features = visual(
                    forward_inputs["pixel_values"][index, :patches].to(
                        next(visual.parameters()).dtype
                    ),
                    grid_thw=forward_inputs["image_grid_thw"][index, :images],
                )
                features = getattr(features, "pooler_output", features)
                if isinstance(features, (tuple, list)):
                    features = features[0]
            image_mask = ids == qwen.config.image_token_id
            # Sampled image-token IDs belong to the response text.
            image_mask[:, start:] = False
            hidden = self.core(
                ids, positions, image_features=features, image_token_mask=image_mask
            )
            logits = self.core.project(hidden[:, start - 1 : -1])
            temperature, _top_p, penalty, do_sample = forward_inputs["sampling"][
                index
            ].tolist()
            scores, entropy = [], []
            for offset, position in enumerate(range(start, length)):
                effective = _apply_repetition_penalty(
                    logits[:, offset].float(), ids[:, start:position], penalty
                )
                if do_sample:
                    effective = effective / temperature
                # Top-p selects rollout tokens; training scores the full distribution.
                logs = effective.log_softmax(-1)
                scores.append(logs.gather(1, ids[:, position : position + 1]).squeeze())
                if compute_entropy:
                    finite = logs.masked_fill(~torch.isfinite(logs), 0)
                    entropy.append(-(logs.exp() * finite).sum())
            count = length - start
            probs.append(F.pad(torch.stack(scores), (0, self.max_new_tokens - count)))
            masks.append(torch.arange(self.max_new_tokens, device=ids.device) < count)
            if compute_entropy:
                entropies.append(
                    F.pad(torch.stack(entropy), (0, self.max_new_tokens - count))
                )
        return {
            "logprobs": torch.stack(probs),
            "logprob_mask": torch.stack(masks),
            "entropy": torch.stack(entropies) if compute_entropy else None,
            "values": None,
        }
