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

"""Qwen3.8-27B as a VLM policy.

The model emits thinking text and one JSON macro action. Training replays the
log probabilities of the generated tokens; the parsed command is only the
environment interface.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.qwen_vlm.conversation import (
    IncrementalEpisodeTokens,
    render_incremental_user_delta,
)
from rlinf.models.embodiment.qwen_vlm.fp32_head import (
    response_logprobs_and_heads_from_hidden,
    response_logprobs_from_hidden,
)
from rlinf.models.embodiment.qwen_vlm.protocol import (
    DEFAULT_REASONING_EFFORT,
    LIBERO_SCHEMA,
    ROBODOJO_SCHEMA,
    InvalidAction,
    action_to_vector,
    build_user_content,
    cameras_for_schema,
    chat_template_kwargs,
    parse_action,
    round_state,
    system_prompt,
)

QWEN3_8_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def response_logprobs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_index: torch.Tensor,
) -> torch.Tensor:
    """Return log probabilities of tokens addressed by ``response_index``.

    ``response_index`` selects positions in ``input_ids``. A nonpositive index
    is padding and its log probability is zero.
    """
    shifted = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    safe_index = response_index.clamp(min=1)
    token_ids = input_ids.gather(1, safe_index)
    logit_pos = safe_index - 1
    batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)[:, None]
    logprobs = shifted[batch_index, logit_pos, token_ids]
    return logprobs.masked_fill(response_index <= 0, 0.0)


def _image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if float(array.max()) <= 1.0:
            array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def extend_mm_token_type_ids(
    inputs: dict[str, Any], sequence_length: int
) -> torch.Tensor | None:
    """Extend the processor's multimodal token types across generated tokens.

    Generated tokens are text, so the tail is filled with zeros. ``None`` when
    the processor did not return the field.
    """
    token_types = inputs.get("mm_token_type_ids")
    if not torch.is_tensor(token_types):
        return None
    token_types = token_types.detach().cpu().long()
    if token_types.ndim == 2:
        token_types = token_types[0]
    if token_types.shape[0] >= sequence_length:
        return token_types[:sequence_length]
    pad = torch.zeros(sequence_length - token_types.shape[0], dtype=torch.long)
    return torch.cat([token_types, pad])


def _state_from_observation(
    obs: dict[str, Any],
    index: int,
    *,
    require_vlm_state: bool = False,
) -> dict[str, dict[str, float]]:
    vlm_state = obs.get("vlm_state")
    if vlm_state is not None:
        row = vlm_state[index]
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().tolist()
        names = ("x", "y", "z", "roll", "pitch", "yaw", "width")
        values = {name: float(row[pos]) for pos, name in enumerate(names)}
        return round_state(
            {
                "arm": {name: values[name] for name in names[:6]},
                "gripper": {"width": values["width"]},
            }
        )
    if require_vlm_state:
        raise KeyError(
            "Observation is missing base-frame 'vlm_state'. Refusing to fall "
            "back to world-frame 'states' (which breaks VLM coordinate frame "
            "alignment). Ensure EnvOutput.prepare_observations preserves "
            "'vlm_state' and env.vlm_action_mode is enabled."
        )
    state = obs["states"][index]
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().tolist()
    return round_state(
        {
            "arm": {
                "x": float(state[0]) * 100.0,
                "y": float(state[1]) * 100.0,
                "z": float(state[2]) * 100.0,
                "roll": float(state[3]),
                "pitch": float(state[4]),
                "yaw": float(state[5]),
            },
            "gripper": {"width": float(state[6]) * 100.0},
        }
    )


def _turn_images_from_observation(
    obs: dict[str, Any],
    index: int,
    cameras: tuple[str, ...],
) -> list[Image.Image]:
    """Extract ordered PIL images for ``cameras`` from an RLinf observation batch."""
    images: list[Image.Image] = [_image_to_pil(obs["main_images"][index])]
    if len(cameras) == 1:
        return images
    if obs.get("wrist_images") is not None:
        images.append(_image_to_pil(obs["wrist_images"][index]))
    extra = obs.get("extra_view_images")
    if len(images) < len(cameras) and extra is not None:
        row = extra[index]
        if torch.is_tensor(row) and row.ndim == 4:
            for view_idx in range(row.shape[0]):
                if len(images) >= len(cameras):
                    break
                images.append(_image_to_pil(row[view_idx]))
        elif isinstance(row, (list, tuple)):
            for frame in row:
                if len(images) >= len(cameras):
                    break
                images.append(_image_to_pil(frame))
        else:
            images.append(_image_to_pil(row))
    if len(images) != len(cameras):
        raise ValueError(
            f"Expected {len(cameras)} camera frames for {cameras}, got {len(images)}."
        )
    return images


class QwenVLMPolicy(nn.Module, BasePolicy):
    """Multi-turn Qwen3.8 VLM policy with response-token log probabilities."""

    def __init__(self, cfg: DictConfig, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.model_path = cfg.model_path
        self.revision = cfg.get("model_revision", QWEN3_8_REVISION)
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.schema = (
            ROBODOJO_SCHEMA if cfg.get("embodiment") == "robodojo" else LIBERO_SCHEMA
        )
        self.cameras = cameras_for_schema(self.schema)
        self.action_width = sum(len(part) for part in self.schema.values()) + 1
        self.reasoning_effort = str(
            cfg.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
        )
        self.response_budget = int(cfg.get("response_budget", self.max_new_tokens))
        self.require_vlm_state = bool(cfg.get("require_vlm_state", False))
        self._histories: list[list[dict[str, Any]]] = []
        self._turn_response_ids: list[list[list[int]]] = []
        self._episodes: list[IncrementalEpisodeTokens | None] = []
        self._sp_group = None
        self.processor = None
        self.model = None
        if cfg.get("load_model", True):
            self._load_model()

    def set_ulysses_sp_group(self, sp_group: Any) -> None:
        """Attach the Ulysses sequence-parallel process group when sp_size > 1."""
        self._sp_group = sp_group

    def _load_model(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Forward FSDP's checkpointing switch to the wrapped Qwen model."""
        if self.model is None:
            raise RuntimeError("Qwen VLM policy is not loaded")
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                "Qwen VLM policy supports the response-token RL forward only."
            )
        return self.default_forward(**kwargs)

    def predict_action_batch(
        self, env_obs: dict[str, Any], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = self._batch_size(env_obs)
        self._reset_histories(env_obs, batch_size)
        temperature = float(
            kwargs.get("temperature_train", kwargs.get("temperature", 1.0))
        )
        do_sample = bool(kwargs.get("do_sample", True))
        max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        actions = []
        packed = []
        for index in range(batch_size):
            action, sample = self._generate_one(
                env_obs,
                index,
                temperature=temperature,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
            )
            actions.append(action)
            packed.append(sample)
        forward_inputs = self._collate(packed)
        return torch.tensor(actions, dtype=torch.float32).reshape(
            batch_size, 1, self.action_width
        ), {
            "prev_logprobs": forward_inputs["response_logprobs"],
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }

    def default_forward(
        self, forward_inputs: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        del kwargs
        input_ids = forward_inputs["input_ids"]
        attention_mask = forward_inputs["attention_mask"]
        response_index = forward_inputs["response_index"]
        response_mask = forward_inputs["response_mask"].to(dtype=torch.bool)
        rollout_topk_ids = forward_inputs.get("rollout_topk_ids")
        if self.model is None:
            raise RuntimeError("Qwen VLM replay requires the loaded Qwen3.8 model.")
        logprobs = []
        entropy = []
        train_head_log_probs = [] if rollout_topk_ids is not None else None
        for index in range(input_ids.shape[0]):
            # Unpad each episode to its true token length so padded micro-batch
            # tails never consume attention or activation memory.
            valid_len = max(int(attention_mask[index].sum().item()), 1)
            valid_pixels = int(forward_inputs["pixel_mask"][index].sum())
            valid_grids = int(forward_inputs["grid_mask"][index].sum())
            sample_input_ids = input_ids[index : index + 1, :valid_len]
            sample_attention_mask = attention_mask[index : index + 1, :valid_len]
            model_inputs = {
                "input_ids": sample_input_ids,
                "attention_mask": sample_attention_mask,
                "pixel_values": forward_inputs["pixel_values"][index, :valid_pixels],
                "image_grid_thw": forward_inputs["image_grid_thw"][index, :valid_grids],
                "use_cache": False,
            }
            token_types = forward_inputs.get("mm_token_type_ids")
            if torch.is_tensor(token_types):
                model_inputs["mm_token_type_ids"] = token_types[
                    index : index + 1, :valid_len
                ]
            # The backbone only. The head is a chunked fp32 GEMM below, so
            # the [sequence, vocab] logits are never allocated.
            hidden = self.model.model(**model_inputs)[0]
            if rollout_topk_ids is not None:
                (
                    sample_logprobs,
                    sample_entropy,
                    sample_head_logprobs,
                ) = response_logprobs_and_heads_from_hidden(
                    hidden,
                    self.model.lm_head.weight,
                    sample_input_ids,
                    response_index[index : index + 1],
                    rollout_topk_ids[index : index + 1],
                )
                train_head_log_probs.append(sample_head_logprobs)
            else:
                sample_logprobs, sample_entropy = response_logprobs_from_hidden(
                    hidden,
                    self.model.lm_head.weight,
                    sample_input_ids,
                    response_index[index : index + 1],
                )
            logprobs.append(sample_logprobs)
            entropy.append(sample_entropy)
        result = {
            "logprobs": torch.cat(logprobs, dim=0),
            "logprob_mask": response_mask,
            "values": None,
            "entropy": torch.cat(entropy, dim=0),
        }
        if train_head_log_probs is not None:
            result["train_head_log_probs"] = torch.cat(train_head_log_probs, dim=0)
        return result

    def _generate_one(
        self,
        env_obs: dict[str, Any],
        index: int,
        *,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int,
    ) -> tuple[list[float], dict[str, Any]]:
        instruction = env_obs["task_descriptions"][index]
        state = _state_from_observation(
            env_obs,
            index,
            require_vlm_state=getattr(self, "require_vlm_state", False),
        )
        cameras = getattr(self, "cameras", cameras_for_schema(self.schema))
        turn_images = _turn_images_from_observation(env_obs, index, cameras)
        user = {
            "role": "user",
            "content": build_user_content(state, turn_images, cameras),
        }
        budget = int(getattr(self, "response_budget", max_new_tokens))
        while len(self._episodes) <= index:
            self._episodes.append(None)
        episode = self._episodes[index]
        template_kwargs = chat_template_kwargs(self.reasoning_effort)
        if episode is None or not episode.initialized:
            episode = IncrementalEpisodeTokens(response_budget=budget)
            messages = [
                {"role": "system", "content": system_prompt(instruction, self.schema)},
                user,
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            inputs = self.processor(
                text=[prompt], images=turn_images, return_tensors="pt", padding=True
            )
            episode.start_episode(prompt_text=prompt, inputs=inputs)
            self._episodes[index] = episode
        else:
            if episode.budget_exhausted or episode.remaining_budget() <= 0:
                episode.budget_exhausted = True
                return action_to_vector(None, self.schema), episode.build_sample()
            delta_full, delta_no_eos = render_incremental_user_delta(
                self.processor, user, template_kwargs
            )
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            use_no_eos = (
                eos_id is not None
                and bool(episode.input_ids)
                and episode.input_ids[-1] == eos_id
            )
            delta_text = delta_no_eos if use_no_eos else delta_full
            delta_inputs = self.processor(
                text=[delta_text],
                images=turn_images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            if not episode.append_user_observation(
                delta_text=delta_text, delta_inputs=delta_inputs
            ):
                return action_to_vector(None, self.schema), episode.build_sample()

        remaining = min(int(max_new_tokens), episode.remaining_budget())
        if remaining <= 0:
            episode.budget_exhausted = True
            return action_to_vector(None, self.schema), episode.build_sample()

        current_sample = episode.build_sample()
        device_inputs: dict[str, Any] = {
            "input_ids": current_sample["input_ids"].unsqueeze(0).to(self.model.device),
            "attention_mask": current_sample["attention_mask"]
            .unsqueeze(0)
            .to(self.model.device),
            "pixel_values": current_sample["pixel_values"].to(self.model.device),
            "image_grid_thw": current_sample["image_grid_thw"].to(self.model.device),
        }
        if "mm_token_type_ids" in current_sample:
            device_inputs["mm_token_type_ids"] = (
                current_sample["mm_token_type_ids"].unsqueeze(0).to(self.model.device)
            )
        prompt_length = int(current_sample["input_ids"].shape[0])
        output = self.model.generate(
            **device_inputs,
            max_new_tokens=remaining,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequence = output.sequences[0].detach().cpu()
        response_tensor = sequence[prompt_length:]
        response_ids = response_tensor.tolist()
        text = self.processor.tokenizer.decode(
            response_tensor, skip_special_tokens=False
        )
        scores = torch.stack([score[0].float().cpu() for score in output.scores], dim=0)
        token_logprobs = (
            torch.log_softmax(scores, dim=-1)
            .gather(-1, response_tensor.unsqueeze(-1))
            .squeeze(-1)
            .tolist()
        )
        episode.append_assistant_response(
            response_ids=response_ids,
            response_text=text,
            token_logprobs=token_logprobs,
        )
        if episode.budget_exhausted:
            action = None
            vector = action_to_vector(None, self.schema)
        else:
            try:
                action = parse_action(text, self.schema)
                vector = action_to_vector(action, self.schema)
            except InvalidAction:
                action = None
                vector = action_to_vector(None, self.schema)
        self._histories[index].extend([user, {"role": "assistant", "content": text}])
        while len(self._turn_response_ids) <= index:
            self._turn_response_ids.append([])
        self._turn_response_ids[index].append(response_ids)
        sample = episode.build_sample()
        sample["text"] = text
        sample["action"] = action
        return vector, sample

    def _collate(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(sample["input_ids"].shape[0] for sample in samples)
        max_response = max(sample["response_logprobs"].shape[0] for sample in samples)
        max_pixels = max(sample["pixel_values"].shape[0] for sample in samples)
        max_grids = max(sample["image_grid_thw"].shape[0] for sample in samples)
        batch = len(samples)
        input_ids = torch.zeros(batch, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch, max_length, dtype=torch.long)
        response_logprobs = torch.zeros(batch, max_response, dtype=torch.float32)
        response_mask = torch.zeros(batch, max_response, dtype=torch.bool)
        response_index = torch.zeros(batch, max_response, dtype=torch.long)
        pixel_shape = samples[0]["pixel_values"].shape[1:]
        pixel_values = torch.zeros(batch, max_pixels, *pixel_shape)
        grid_shape = samples[0]["image_grid_thw"].shape[1:]
        image_grid_thw = torch.zeros(batch, max_grids, *grid_shape, dtype=torch.long)
        pixel_mask = torch.zeros(batch, max_pixels, dtype=torch.bool)
        grid_mask = torch.zeros(batch, max_grids, dtype=torch.bool)
        for index, sample in enumerate(samples):
            length = sample["input_ids"].shape[0]
            response = sample["response_logprobs"].shape[0]
            pixels = sample["pixel_values"].shape[0]
            grids = sample["image_grid_thw"].shape[0]
            input_ids[index, :length] = sample["input_ids"]
            attention_mask[index, :length] = 1
            response_logprobs[index, :response] = sample["response_logprobs"]
            response_mask[index, :response] = True
            response_index[index, :response] = sample["response_index"]
            pixel_values[index, :pixels] = sample["pixel_values"]
            image_grid_thw[index, :grids] = sample["image_grid_thw"]
            pixel_mask[index, :pixels] = True
            grid_mask[index, :grids] = True
        token_types = [sample.get("mm_token_type_ids") for sample in samples]
        collated = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_mask": pixel_mask,
            "grid_mask": grid_mask,
            "response_logprobs": response_logprobs,
            "response_mask": response_mask,
            "response_index": response_index,
        }
        if any(torch.is_tensor(item) for item in token_types):
            typed = torch.zeros(batch, max_length, dtype=torch.long)
            for index, item in enumerate(token_types):
                if torch.is_tensor(item):
                    typed[index, : item.shape[0]] = item
            collated["mm_token_type_ids"] = typed
        topk_ids_list = [sample.get("rollout_topk_ids") for sample in samples]
        topk_lps_list = [sample.get("sampler_head_log_probs") for sample in samples]
        if any(torch.is_tensor(item) for item in topk_ids_list):
            first_topk = next(item for item in topk_ids_list if torch.is_tensor(item))
            k = int(first_topk.shape[-1])
            placeholder_ids = torch.arange(k, dtype=torch.long)
            placeholder_lps = torch.full(
                (k,), -float(np.log(max(k, 1))), dtype=torch.float32
            )
            padded_topk_ids = (
                placeholder_ids.view(1, 1, k).expand(batch, max_response, k).clone()
            )
            padded_topk_lps = (
                placeholder_lps.view(1, 1, k).expand(batch, max_response, k).clone()
            )
            for index, (ids_item, lps_item) in enumerate(
                zip(topk_ids_list, topk_lps_list)
            ):
                if torch.is_tensor(ids_item) and torch.is_tensor(lps_item):
                    n = min(int(ids_item.shape[0]), max_response)
                    padded_topk_ids[index, :n] = ids_item[:n]
                    padded_topk_lps[index, :n] = lps_item[:n]
            collated["rollout_topk_ids"] = padded_topk_ids
            collated["sampler_head_log_probs"] = padded_topk_lps
        return collated

    def _reset_histories(self, env_obs: dict[str, Any], batch_size: int) -> None:
        if len(self._histories) != batch_size:
            self._histories = [[] for _ in range(batch_size)]
        if len(getattr(self, "_turn_response_ids", [])) != batch_size:
            self._turn_response_ids = [[] for _ in range(batch_size)]
        if len(getattr(self, "_episodes", [])) != batch_size:
            self._episodes = [None for _ in range(batch_size)]
        reset_mask = env_obs.get("reset_mask")
        if reset_mask is None:
            return
        if isinstance(reset_mask, torch.Tensor):
            reset_mask = reset_mask.detach().cpu().tolist()
        for index, reset in enumerate(reset_mask):
            if reset:
                self._histories[index] = []
                self._turn_response_ids[index] = []
                self._episodes[index] = None

    @staticmethod
    def _batch_size(env_obs: dict[str, Any]) -> int:
        tasks = env_obs.get("task_descriptions")
        if isinstance(tasks, (list, tuple)):
            return len(tasks)
        state = env_obs["states"]
        return int(state.shape[0])


def serialize_action(action: dict[str, Any] | None) -> str:
    """Serialize a parsed action for diagnostics."""
    return "" if action is None else json.dumps(action)
