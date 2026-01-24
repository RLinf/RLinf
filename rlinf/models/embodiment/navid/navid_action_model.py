# Copyright 2025 The RLinf Authors.
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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import warnings

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.navid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IAMGE_SEPARATOR,
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    NAVIGATION_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
)
from rlinf.models.embodiment.navid.conversation import (
    SeparatorStyle,
    conv_templates,
)
from rlinf.models.embodiment.navid.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_image_token,
)


class NaVidForRLActionPrediction(nn.Module, BasePolicy):
    """
    A thin wrapper around NaVid (LLaVA-style) model to fit RLinf embodied policy interface.
    """

    def __init__(
        self,
        *,
        tokenizer,
        model,
        image_processor,
        action_dim: int,
        num_action_chunks: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self._history_rgb_tensor: dict[str, torch.Tensor | None] = {}
        self._max_history_len: int | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str,
        model_base: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        action_dim: int,
        num_action_chunks: int,
    ) -> "NaVidForRLActionPrediction":
        from rlinf.models.embodiment.navid.mm_utils import get_model_name_from_path
        from rlinf.models.embodiment.navid.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map=None,
        )
        if torch_dtype is not None:
            model.to(torch_dtype)
        return cls(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            action_dim=int(action_dim),
            num_action_chunks=int(num_action_chunks),
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        # This wrapper is intended for rollout-time action prediction.
        raise NotImplementedError(
            "NaVidForRLActionPrediction does not implement training forward in RLinf yet."
        )

    def preprocess_env_obs(self, env_obs):
        out = dict(env_obs)
        images_tensor = out["main_images"]  # [B, C, H, W]
        batch_images_np = []
        for i in range(images_tensor.shape[0]):
            img_np = images_tensor[i].detach().cpu().numpy()
            img_np = img_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            batch_images_np.append(img_np)
        out["main_images"] = batch_images_np

        return out

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "train",
        return_obs: bool = True,
        **kwargs,
    ):
        assert mode in {"train", "eval"}, f"{mode=} is not supported"
        device = self._get_device()
        gen_params = self._get_generation_params(**kwargs)

        images = env_obs["main_images"]
        task_descs = env_obs.get("task_descriptions", [""] * len(images))
        episode_ids = env_obs.get("episode_ids", None)
        bsz = len(images)

        prompts, questions, convs = self._build_prompts_and_convs(
            bsz=bsz,
            task_descs=task_descs,
            conv_templates=conv_templates,
            conversation_template="vicuna_v1",
            default_image_token=DEFAULT_IMAGE_TOKEN,
            default_im_start_token=DEFAULT_IM_START_TOKEN,
            default_im_end_token=DEFAULT_IM_END_TOKEN,
        )

        new_frames_tensor = self._preprocess_new_frames(images=images, device=device)
        images_for_model = self._accumulate_history_frames(
            new_frames_tensor=new_frames_tensor, episode_ids=episode_ids, device=device
        )

        special_tokens = self._build_special_token_tensors(
            image_start_token=IMAGE_START_TOKEN,
            image_end_token=IMAGE_END_TOKEN,
            video_start_token=VIDEO_START_SPECIAL_TOKEN,
            video_end_token=VIDEO_END_SPECIAL_TOKEN,
            navigation_token=NAVIGATION_SPECIAL_TOKEN,
            image_separator_token=IAMGE_SEPARATOR,
            device=device,
        )

        input_ids, attention_mask = self._tokenize_and_pad_prompts(
            prompts=prompts,
            tokenizer_image_token=tokenizer_image_token,
            image_token_index=IMAGE_TOKEN_INDEX,
            special_tokens=special_tokens,
            device=device,
        )

        stop_str, stopping_criteria = self._build_stopping_criteria(
            convs=convs, SeparatorStyle=SeparatorStyle, input_ids=input_ids
        )

        outputs = self._generate(
            questions=questions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images_for_model=images_for_model,
            stopping_criteria=stopping_criteria,
            gen_params=gen_params,
        )

        gen_texts = self._decode_generated_texts(
            outputs=outputs,
            input_token_len=input_ids.shape[1],
            stop_str=stop_str,
        )

        batch_actions = self._parse_actions_from_texts(
            gen_texts=gen_texts, stop_str=stop_str
        )
        chunk_actions = self._pack_actions_to_chunks(
            batch_actions=batch_actions, bsz=bsz
        )

        prev_logprobs = torch.zeros(
            (bsz, self.action_dim), device=device, dtype=torch.float32
        )
        prev_values = torch.zeros((bsz, 1), device=device, dtype=torch.float32)

        forward_inputs: dict[str, Any] = {}
        if return_obs:
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["task_descriptions"] = env_obs["task_descriptions"]
            forward_inputs["generated_text"] = gen_texts

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _get_generation_params(self, **kwargs: Any) -> dict[str, Any]:
        do_sample = kwargs.get("do_sample", "True")
        temperature = float(kwargs.get("temperature", 0.2))
        max_new_tokens = int(kwargs.get("max_new_tokens", 1024))
        return {
            "do_sample": bool(do_sample),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

    def _build_prompts_and_convs(
        self,
        *,
        bsz: int,
        task_descs: list[str],
        conv_templates,
        conversation_template: str,
        default_image_token: str,
        default_im_start_token: str,
        default_im_end_token: str,
    ) -> tuple[list[str], list[str], list]:
        conv_tmpl = conv_templates.get(conversation_template)
        if conv_tmpl is None:
            raise ValueError(
                f"Unknown NaVid conversation_template={conversation_template}. "
                f"Available: {sorted(conv_templates.keys())}"
            )

        navid_prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. You have been given "
            "a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward "
            "a certain distance."
        )

        prompts: list[str] = []
        questions: list[str] = []
        convs: list = []

        for i in range(bsz):
            user_msg = navid_prompt_template.format(task_descs[i])
            question = user_msg.replace(default_image_token, "").replace("\n", "")

            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    default_im_start_token
                    + default_image_token
                    + default_im_end_token
                    + "\n"
                    + user_msg.replace("<image>", "")
                )
            else:
                qs = default_image_token + "\n" + user_msg.replace("<image>", "")

            conv = conv_tmpl.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            questions.append(question)
            prompts.append(conv.get_prompt())
            convs.append(conv)

        return prompts, questions, convs

    def _preprocess_new_frames(
        self, *, images: list[np.ndarray], device: torch.device
    ) -> torch.Tensor:
        batch_image = np.asarray(images)
        new_frames_tensor = self.image_processor.preprocess(
            batch_image, return_tensors="pt"
        )["pixel_values"]  # [B, C, H, W]

        if getattr(self.model, "dtype", None) == torch.float16:
            new_frames_tensor = new_frames_tensor.half()
        else:
            new_frames_tensor = new_frames_tensor.float()
        return new_frames_tensor.to(device=device)

    def _accumulate_history_frames(
        self,
        *,
        new_frames_tensor: torch.Tensor,
        episode_ids: list[Any],
        device: torch.device,
    ) -> list[torch.Tensor]:
        bsz = int(new_frames_tensor.shape[0])
        images_for_model: list[torch.Tensor] = []

        for i in range(bsz):
            ep_key = str(episode_ids[i])
            if ep_key not in self._history_rgb_tensor:
                self._history_rgb_tensor[ep_key] = None

            new_frame = new_frames_tensor[i : i + 1]  # [1, C, H, W]
            if self._history_rgb_tensor[ep_key] is None:
                self._history_rgb_tensor[ep_key] = new_frame
            else:
                self._history_rgb_tensor[ep_key] = torch.cat(
                    (self._history_rgb_tensor[ep_key], new_frame), dim=0
                )

            if self._max_history_len is not None:
                hist = self._history_rgb_tensor[ep_key]
                if hist is not None and hist.shape[0] > self._max_history_len:
                    self._history_rgb_tensor[ep_key] = hist[-self._max_history_len :]

            images_for_model.append(self._history_rgb_tensor[ep_key].to(device=device))

        return images_for_model

    @dataclass(frozen=True)
    class _SpecialTokens:
        image_start: torch.Tensor
        image_end: torch.Tensor
        video_start: torch.Tensor
        video_end: torch.Tensor
        navigation: torch.Tensor
        image_separator: torch.Tensor

    def _build_special_token_tensors(
        self,
        *,
        image_start_token: str,
        image_end_token: str,
        video_start_token: str,
        video_end_token: str,
        navigation_token: str,
        image_separator_token: str,
        device: torch.device,
    ) -> _SpecialTokens:
        def _tok(s: str) -> torch.Tensor:
            # drop leading BOS (match VLN-CE)
            return self.tokenizer(s, return_tensors="pt").input_ids[0][1:].to(device)

        return self._SpecialTokens(
            image_start=_tok(image_start_token),
            image_end=_tok(image_end_token),
            video_start=_tok(video_start_token),
            video_end=_tok(video_end_token),
            navigation=_tok(navigation_token),
            image_separator=_tok(image_separator_token),
        )

    def _tokenize_and_pad_prompts(
        self,
        *,
        prompts: list[str],
        tokenizer_image_token,
        image_token_index: int,
        special_tokens: _SpecialTokens,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_id_list: list[torch.Tensor] = []

        for prompt in prompts:
            token_prompt = tokenizer_image_token(
                prompt, self.tokenizer, image_token_index, return_tensors="pt"
            ).to(device)
            indices_to_replace = torch.where(token_prompt == image_token_index)[0]
            new_list: list[torch.Tensor] = []

            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                new_list.append(token_prompt[:idx])
                new_list.append(special_tokens.video_start)
                new_list.append(special_tokens.image_separator)
                new_list.append(token_prompt[idx : idx + 1])  # keep IMAGE_TOKEN_INDEX
                new_list.append(special_tokens.video_end)
                new_list.append(special_tokens.image_start)
                new_list.append(special_tokens.image_end)
                new_list.append(special_tokens.navigation)
                token_prompt = token_prompt[idx + 1 :]
                indices_to_replace = torch.where(token_prompt == image_token_index)[0]

            if token_prompt.numel() > 0:
                new_list.append(token_prompt)

            input_ids_single = torch.cat(new_list, dim=0).unsqueeze(0)  # [1, seq]
            input_id_list.append(input_ids_single)

        bsz = len(input_id_list)
        max_len = max(int(x.shape[1]) for x in input_id_list)
        pad_id = int(self.tokenizer.pad_token_id or 0)

        input_ids = torch.full(
            (bsz, max_len), fill_value=pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

        for i, ids in enumerate(input_id_list):
            seq_len = ids.shape[1]
            input_ids[i, -seq_len:] = ids[0]
            attention_mask[i, -seq_len:] = 1

        return input_ids, attention_mask

    def _build_stopping_criteria(
        self, *, convs: list, SeparatorStyle, input_ids: torch.Tensor
    ) -> tuple[str, KeywordsStoppingCriteria]:
        stop_str = (
            convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )
        return stop_str, stopping_criteria

    def _generate(
        self,
        *,
        questions: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images_for_model: list[torch.Tensor],
        stopping_criteria,
        gen_params: dict[str, Any],
    ) -> torch.Tensor:
        prompts_for_update = [[q] for q in questions]
        with torch.inference_mode():
            self.model.update_prompt(prompts_for_update)
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images_for_model,
                do_sample=gen_params["do_sample"],
                temperature=gen_params["temperature"],
                max_new_tokens=gen_params["max_new_tokens"],
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

    def _decode_generated_texts(
        self,
        *,
        outputs: torch.Tensor,
        input_token_len: int,
        stop_str: str,
    ) -> list[str]:
        gen_texts = self.tokenizer.batch_decode(
            outputs[:, input_token_len:], skip_special_tokens=True
        )
        return [
            text[: -len(stop_str)].strip() if text.endswith(stop_str) else text.strip()
            for text in gen_texts
        ]

    def _parse_actions_from_texts(
        self, *, gen_texts: list[str], stop_str: str
    ) -> list[str]:
        actions: list[str] = []
        for gen_text in gen_texts:
            text = (
                gen_text[: -len(stop_str)].strip()
                if stop_str and gen_text.endswith(stop_str)
                else gen_text.strip()
            )
            if "forward" in text:
                actions.append("move_forward")
            elif "left" in text:
                actions.append("turn_left")
            elif "right" in text:
                actions.append("turn_right")
            elif "stop" in text:
                actions.append("stop")
            else:
                warnings.warn(
                    f"NaVid: action not found in generated text: {text!r}; padding with 'no_op'.",
                    stacklevel=2,
                )
                actions.append("no_op")
        return actions

    def _pack_actions_to_chunks(
        self, *, batch_actions: list[str], bsz: int
    ) -> np.ndarray:
        if len(batch_actions) != bsz:
            raise ValueError(
                f"Generated actions length mismatch: {len(batch_actions)} vs {bsz=}"
            )
        if self.action_dim != 1:
            raise ValueError(
                f"NaVidForRLActionPrediction currently outputs discrete string actions; "
                f"expected {self.action_dim=} to be 1."
            )

        # NaVid emits one action per env step. If num_action_chunks>1, pad remaining steps with no_op.
        chunk_actions = np.full((bsz, self.num_action_chunks, 1), "no_op", dtype="<U12")
        chunk_actions[:, 0, 0] = np.asarray(batch_actions, dtype="<U12")
        return chunk_actions
