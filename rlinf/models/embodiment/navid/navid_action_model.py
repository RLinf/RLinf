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
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


@dataclass(frozen=True)
class NaVidGenerationConfig:
    max_new_tokens: int = 32
    conversation_template: str = "imgsp_v1"


class NaVidForRLActionPrediction(nn.Module, BasePolicy):
    """
    A thin wrapper around NaVid (LLaVA-style) model to fit RLinf embodied policy interface.

    Notes:
      - NaVid is a text-generation VLM; here we *parse* continuous actions from generated text.
      - If parsing fails, we return zeros to keep rollout pipeline running.
      - We provide dummy `prev_logprobs` / `prev_values` (zeros), since this model is not trained
        with explicit action likelihood/value heads in RLinf.
    """

    def __init__(
        self,
        *,
        tokenizer,
        model,
        image_processor,
        action_dim: int,
        num_action_chunks: int,
        gen_cfg: NaVidGenerationConfig,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.gen_cfg = gen_cfg
        # Per-episode RGB history (aligned with VLN-CE NaVid_Agent incremental processing).
        # Key: episode_id (str), Value: torch.Tensor of shape [T, C, H, W] (accumulated video frames).
        self._history_rgb_tensor: dict[str, torch.Tensor | None] = {}
        # Optional cap on history length to bound memory (None = keep full episode).
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
        max_new_tokens: int = 32,
        conversation_template: str = "imgsp_v1",
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
            # For RLinf rollout (and this smoke test), keep everything on a single device to avoid
            # cross-GPU matmul/device mismatch errors.
            device_map=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if torch_dtype is not None:
            model.to(torch_dtype)
        gen_cfg = NaVidGenerationConfig(
            max_new_tokens=int(max_new_tokens),
            conversation_template=str(conversation_template),
        )
        return cls(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            action_dim=int(action_dim),
            num_action_chunks=int(num_action_chunks),
            gen_cfg=gen_cfg,
        )

    def preprocess_env_obs(self, env_obs):
        # Keep original structure; only ensure tensors are on the same device as the model.
        device = (
            next(self.parameters()).device
            if any(True for _ in self.parameters())
            else (self.model.device if hasattr(self.model, "device") else "cpu")
        )
        out = dict(env_obs)
        # Habitat env may provide `rgb` instead of `main_images`. Mirror to keep downstream stable.
        if "main_images" not in out and "rgb" in out:
            out["main_images"] = out["rgb"]
        if "main_images" in out and torch.is_tensor(out["main_images"]):
            out["main_images"] = out["main_images"].to(
                device="cpu"
            )  # PIL conversion path
        if "states" in out and torch.is_tensor(out["states"]):
            out["states"] = out["states"].to(device=device)
        return out

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        # This wrapper is intended for rollout-time action prediction.
        raise NotImplementedError(
            "NaVidForRLActionPrediction does not implement training forward in RLinf yet."
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "train",
        return_obs: bool = True,
        **kwargs,
    ):
        """
        Returns:
          - chunk_actions: np.ndarray shaped [B, num_action_chunks, action_dim]
          - result: dict with keys `prev_logprobs`, `prev_values`, `forward_inputs`
        """
        assert mode in {"train", "eval"}, f"{mode=} is not supported"
        do_sample = kwargs.get("do_sample", mode == "train")
        temperature = float(kwargs.get("temperature", 1.0))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k_raw = kwargs.get("top_k", 0)
        top_k = int(top_k_raw) if top_k_raw is not None else 0
        if top_k <= 0:
            top_k = None
        max_new_tokens_raw = kwargs.get("max_new_tokens", self.gen_cfg.max_new_tokens)
        if max_new_tokens_raw is None:
            # fall back to generation config or a sensible default
            max_new_tokens_raw = self.gen_cfg.max_new_tokens or 32
        max_new_tokens = int(max_new_tokens_raw)

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

        images = env_obs["main_images"]
        task_descs = env_obs.get("task_descriptions", None)
        if task_descs is None:
            # fallback: empty prompts
            bsz = int(images.shape[0])
            task_descs = [""] * bsz

        bsz = int(images.shape[0])

        conv_tmpl = conv_templates.get(self.gen_cfg.conversation_template)
        if conv_tmpl is None:
            raise ValueError(
                f"Unknown NaVid conversation_template={self.gen_cfg.conversation_template}. "
                f"Available: {sorted(conv_templates.keys())}"
            )

        # Prepare text prompts (one per env in the batch)
        prompts: list[str] = []
        questions: list[str] = []
        convs: list = []

        for i in range(bsz):
            # Build a single-turn prompt with <image>.
            conv = conv_tmpl.copy()
            if int(self.action_dim) == 1:
                # Explicitly instruct discrete VLN action format for Habitat.
                user_msg = (
                    "<image>\n"
                    "You are an embodied navigation agent in Habitat (VLN-CE R2R). "
                    "Given the instruction, output a sequence of actions, one per step, "
                    f"for the next {self.num_action_chunks} steps. "
                    "Use only these tokens: move_forward, turn_left, turn_right, stop, no_op. "
                    "If you decide to stop, include stop and then no_op for the remaining steps.\n\n"
                    f"Instruction: {task_descs[i]}"
                )
            else:
                user_msg = f"<image>\n{task_descs[i]}"

            # Extract question (aligned with VLN-CE agent_navid.py:73)
            question = user_msg.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
            questions.append(question)

            # Build prompt with image tokens (aligned with VLN-CE agent_navid.py:89-92)
            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + user_msg.replace("<image>", "")
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + user_msg.replace("<image>", "")

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            convs.append(conv)

        episode_ids = env_obs.get("episode_ids", None)
        if episode_ids is not None:
            # episode_ids may be tensor or list; normalize to list of strings
            if torch.is_tensor(episode_ids):
                episode_ids_list = episode_ids.detach().cpu().tolist()
            else:
                episode_ids_list = list(episode_ids)
        else:
            # Fallback: use batch indices as pseudo episode ids
            episode_ids_list = list(range(bsz))

        device = self.model.device if hasattr(self.model, "device") else "cpu"

        # Convert all images to numpy arrays (HWC uint8) for batch processing
        batch_images_np = []
        for i in range(bsz):
            if torch.is_tensor(images[i]):
                # Convert tensor to numpy: handle both (H, W, C) and (C, H, W) formats
                img_np = images[i].detach().cpu().numpy()
                if img_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    img_np = img_np.transpose(1, 2, 0)
                if img_np.dtype != np.uint8:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            else:
                # Already numpy or PIL, convert to numpy
                if isinstance(images[i], Image.Image):
                    img_np = np.array(images[i])
                else:
                    img_np = np.asarray(images[i])
                # Ensure HWC format
                if img_np.ndim == 3 and img_np.shape[0] == 3:
                    img_np = img_np.transpose(1, 2, 0)
            batch_images_np.append(img_np)

        # Batch process all new frames at once (shape: [B, H, W, C])
        batch_image = np.asarray(batch_images_np)
        new_frames_tensor = self.image_processor.preprocess(
            batch_image, return_tensors="pt"
        )["pixel_values"]  # shape: [B, C, H, W]

        # Move to device and convert dtype to match model
        if hasattr(self.model, "dtype") and self.model.dtype == torch.float16:
            new_frames_tensor = new_frames_tensor.half()
        else:
            new_frames_tensor = new_frames_tensor.float()
        new_frames_tensor = new_frames_tensor.to(device=device)

        # Accumulate each new frame to its episode's history tensor
        images_for_model: list[torch.Tensor] = []
        for i in range(bsz):
            ep_id = episode_ids_list[i]
            ep_key = str(ep_id)

            if ep_key not in self._history_rgb_tensor:
                self._history_rgb_tensor[ep_key] = None

            # Get the new frame for this environment (unsqueeze to [1, C, H, W])
            new_frame = new_frames_tensor[i : i + 1]  # [1, C, H, W]

            if self._history_rgb_tensor[ep_key] is None:
                self._history_rgb_tensor[ep_key] = new_frame
            else:
                self._history_rgb_tensor[ep_key] = torch.cat(
                    (self._history_rgb_tensor[ep_key], new_frame), dim=0
                )

            # Optionally cap history length to bound memory
            if self._max_history_len is not None:
                if self._history_rgb_tensor[ep_key].shape[0] > self._max_history_len:
                    keep_start = (
                        self._history_rgb_tensor[ep_key].shape[0]
                        - self._max_history_len
                    )
                    self._history_rgb_tensor[ep_key] = self._history_rgb_tensor[ep_key][
                        keep_start:
                    ]

            # Return as list format expected by model.generate (VLN-CE returns [tensor])
            images_for_model.append(self._history_rgb_tensor[ep_key])

        device = self.model.device if hasattr(self.model, "device") else "cpu"

        # Prepare special tokens for token replacement (aligned with VLN-CE agent_navid.py:82-87)
        image_start_special_token = (
            self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt")
            .input_ids[0][1:]
            .to(device)
        )
        image_end_special_token = (
            self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt")
            .input_ids[0][1:]
            .to(device)
        )
        video_start_special_token = (
            self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt")
            .input_ids[0][1:]
            .to(device)
        )
        video_end_special_token = (
            self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt")
            .input_ids[0][1:]
            .to(device)
        )
        navigation_special_token = (
            self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt")
            .input_ids[0][1:]
            .to(device)
        )
        image_seperator = (
            self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt")
            .input_ids[0][1:]
            .to(device)
        )

        # Tokenize and replace IMAGE_TOKEN_INDEX with special tokens (aligned with VLN-CE agent_navid.py:99-116)
        input_id_list = []
        for prompt in prompts:
            token_prompt = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).to(device)
            indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
            new_list = []
            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                new_list.append(token_prompt[:idx])
                new_list.append(video_start_special_token)
                new_list.append(image_seperator)
                new_list.append(token_prompt[idx : idx + 1])  # Keep IMAGE_TOKEN_INDEX
                new_list.append(video_end_special_token)
                new_list.append(image_start_special_token)
                new_list.append(image_end_special_token)
                new_list.append(navigation_special_token)
                token_prompt = token_prompt[idx + 1 :]
                indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
            if token_prompt.numel() > 0:
                new_list.append(token_prompt)
            input_ids_single = torch.cat(new_list, dim=0).unsqueeze(0)  # [1, seq_len]
            input_id_list.append(input_ids_single)

        # Pad to same length
        max_len = max(int(x.shape[1]) for x in input_id_list)
        input_ids = torch.full(
            (bsz, max_len),
            fill_value=int(self.tokenizer.pad_token_id or 0),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(input_id_list):
            seq_len = ids.shape[1]
            input_ids[i, -seq_len:] = ids[0]
            attention_mask[i, -seq_len:] = 1

        # Ensure all image tensors are on the same device as the model.
        if isinstance(images_for_model, list):
            images_for_model = [
                img.to(device=device) if torch.is_tensor(img) else img
                for img in images_for_model
            ]
        elif torch.is_tensor(images_for_model):
            images_for_model = images_for_model.to(device=device)

        # Prepare stopping criteria (aligned with VLN-CE agent_navid.py:118-120)
        # Use the first conv's stop_str for all (they should be the same)
        stop_str = (
            convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Update prompt using model.update_prompt (aligned with VLN-CE agent_navid.py:125-127)
        # Note: update_prompt expects list of lists, one per batch item
        prompts_for_update = [[q] for q in questions]
        with torch.inference_mode():
            self.model.update_prompt(prompts_for_update)
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images_for_model,
                do_sample=bool(do_sample),
                temperature=temperature if do_sample else 0.2,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # Decode only the generated suffix and remove stop_str (aligned with VLN-CE agent_navid.py:137-145)
        input_token_len = input_ids.shape[1]
        # Use batch_decode for efficiency (aligned with VLN-CE agent_navid.py:141)
        gen_texts = self.tokenizer.batch_decode(
            outputs[:, input_token_len:], skip_special_tokens=True
        )
        # Remove stop_str if present and strip
        gen_texts = [
            text[: -len(stop_str)].strip() if text.endswith(stop_str) else text.strip()
            for text in gen_texts
        ]

        chunk_actions = []
        for gen_text in gen_texts:
            gen_text = (
                gen_text[: -len(stop_str)].strip()
                if gen_text.endswith(stop_str)
                else gen_text.strip()
            )
            if "forward" in gen_text:
                chunk_actions.append("move_forward")
            elif "left" in gen_text:
                chunk_actions.append("turn_left")
            elif "right" in gen_text:
                chunk_actions.append("turn_right")
            elif "stop" in gen_text:
                chunk_actions.append("stop")
        chunk_actions = np.array(chunk_actions)
        # Add num_action_chunks dim
        # It should be strictly equal to 1
        # Since navid can only output one action at a time
        chunk_actions = chunk_actions.reshape(
            bsz, self.num_action_chunks, self.action_dim
        )
        chunk_actions = chunk_actions.astype(np.float32)

        prev_logprobs = torch.zeros(
            (bsz, self.action_dim), device=device, dtype=torch.float32
        )
        prev_values = torch.zeros((bsz, 1), device=device, dtype=torch.float32)

        forward_inputs: dict[str, Any] = {}
        if return_obs:
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["task_descriptions"] = env_obs.get("task_descriptions", None)
            forward_inputs["generated_text"] = gen_texts

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
