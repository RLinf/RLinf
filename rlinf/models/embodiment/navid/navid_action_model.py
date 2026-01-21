#
# Copyright 2023 Haotian Liu
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

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_VLN_ACTION_RE = re.compile(
    r"\b(move_forward|turn_left|turn_right|stop|no_op)\b", flags=re.IGNORECASE
)


def _to_pil_rgb(img: torch.Tensor) -> Image.Image:
    """
    Convert a single image tensor to PIL.

    Supports:
      - (H, W, C) uint8/float tensor with C=3
      - (C, H, W) uint8/float tensor with C=3
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {tuple(img.shape)}")
    if img.shape[-1] == 3:  # HWC
        arr = img.detach().cpu().numpy()
    elif img.shape[0] == 3:  # CHW -> HWC
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unsupported image layout for shape {tuple(img.shape)}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _parse_actions_from_text(
    text: str, *, total_dim: int, fallback: float = 0.0
) -> np.ndarray:
    """
    Parse continuous actions from model-generated text.

    Accepted formats (examples):
      - JSON list: [0.1, -0.2, ...]
      - JSON dict: {"action": [ ... ]}
      - Plain numbers: "0.1 -0.2 0.3 ..."
    """
    text = text.strip()
    if not text:
        return np.full((total_dim,), fallback, dtype=np.float32)

    # Try JSON first (robustly extract the first JSON object/array-like substring).
    try:
        # Heuristic: find the first {...} or [...] block.
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            payload = json.loads(m.group(1))
            if isinstance(payload, dict):
                payload = payload.get("action", payload.get("actions", payload))
            if isinstance(payload, list):
                vals = [float(x) for x in payload]
                vals = (vals + [fallback] * total_dim)[:total_dim]
                return np.asarray(vals, dtype=np.float32)
    except Exception:
        pass

    # Fallback: regex extract floats
    vals = [float(x) for x in _FLOAT_RE.findall(text)]
    if not vals:
        return np.full((total_dim,), fallback, dtype=np.float32)
    vals = (vals + [fallback] * total_dim)[:total_dim]
    return np.asarray(vals, dtype=np.float32)


def _parse_vln_actions_from_text(
    text: str, *, chunk_len: int, fallback: str = "no_op"
) -> np.ndarray:
    """
    Parse discrete VLN actions from model-generated text.

    We accept any occurrences of:
      - move_forward / turn_left / turn_right / stop / no_op
    and build a fixed-length action chunk. If "stop" occurs, we pad the rest with "no_op".
    """
    text = (text or "").strip().lower()
    acts = _VLN_ACTION_RE.findall(text)
    if not acts:
        acts = [fallback] * chunk_len
    else:
        acts = [a.lower() for a in acts][:chunk_len]
        if "stop" in acts:
            stop_i = acts.index("stop")
            acts = acts[: stop_i + 1]
    if len(acts) < chunk_len:
        acts = acts + [fallback] * (chunk_len - len(acts))
    return np.asarray(acts, dtype=object)


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

        from rlinf.models.embodiment.navid.conversation import conv_templates
        from rlinf.models.embodiment.navid.mm_utils import (
            process_images,
            tokenizer_image_token,
        )

        images = env_obs["main_images"]
        task_descs = env_obs.get("task_descriptions", None)
        if task_descs is None:
            # fallback: empty prompts
            bsz = int(images.shape[0])
            task_descs = [""] * bsz

        bsz = int(images.shape[0])
        prompts_text: list[str] = []
        prompts_for_model: list[list[str]] = []
        pil_images: list[Image.Image] = []

        conv_tmpl = conv_templates.get(self.gen_cfg.conversation_template)
        if conv_tmpl is None:
            raise ValueError(
                f"Unknown NaVid conversation_template={self.gen_cfg.conversation_template}. "
                f"Available: {sorted(conv_templates.keys())}"
            )

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
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts_text.append(prompt)
            prompts_for_model.append([prompt])

            pil_images.append(_to_pil_rgb(images[i]))

        # Image processing (returns tensor or dict depending on processor path)
        image_tensor = process_images(
            pil_images, self.image_processor, self.model.config
        )
        if isinstance(image_tensor, dict):
            image_tensor = image_tensor["pixel_values"]
        # NaVid expects list or 5D; list-of-tensors is safest.
        if torch.is_tensor(image_tensor):
            # image_tensor is (B, C, H, W) or (B, N, C, H, W) depending on config
            images_for_model = [image_tensor[i] for i in range(bsz)]
        else:
            images_for_model = image_tensor

        # Tokenize with <image> placeholder handling.
        input_id_list = [
            tokenizer_image_token(p, self.tokenizer, return_tensors="pt")
            for p in prompts_text
        ]
        max_len = max(int(x.shape[0]) for x in input_id_list)
        input_ids = torch.full(
            (bsz, max_len),
            fill_value=int(self.tokenizer.pad_token_id or 0),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, ids in enumerate(input_id_list):
            input_ids[i, -ids.shape[0] :] = ids
            attention_mask[i, -ids.shape[0] :] = 1

        device = self.model.device if hasattr(self.model, "device") else "cpu"
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)

        # Ensure all image tensors are on the same device as the model.
        if isinstance(images_for_model, list):
            images_for_model = [
                img.to(device=device) if torch.is_tensor(img) else img
                for img in images_for_model
            ]
        elif torch.is_tensor(images_for_model):
            images_for_model = images_for_model.to(device=device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images_for_model,
            prompts=prompts_for_model,
            do_sample=bool(do_sample),
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        # Decode only the generated suffix.
        gen_texts: list[str] = []
        for i in range(bsz):
            out_ids = outputs[i]
            gen_ids = out_ids[input_ids.shape[1] :]
            gen_texts.append(
                self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            )

        # Two modes:
        #  - Continuous action parsing (default): action_dim * num_action_chunks floats
        #  - VLN discrete action parsing (Habitat): action_dim==1 means an action token per step
        if int(self.action_dim) == 1:
            # shape: [B, num_action_chunks] of strings
            chunk_actions = np.stack(
                [
                    _parse_vln_actions_from_text(t, chunk_len=self.num_action_chunks)
                    for t in gen_texts
                ],
                axis=0,
            )
            flat_actions = None
        else:
            total_dim = self.action_dim * self.num_action_chunks
            flat_actions = np.stack(
                [
                    _parse_actions_from_text(t, total_dim=total_dim, fallback=0.0)
                    for t in gen_texts
                ],
                axis=0,
            ).astype(np.float32)
            chunk_actions = flat_actions.reshape(
                bsz, self.num_action_chunks, self.action_dim
            )

        # Build tensors for buffer. Shape contracts:
        # - prev_logprobs: [B, action_dim] (ChunkStepResult expects this)
        # - prev_values: [B, 1]
        prev_logprobs = torch.zeros(
            (bsz, self.action_dim), device=device, dtype=torch.float32
        )
        prev_values = torch.zeros((bsz, 1), device=device, dtype=torch.float32)

        forward_inputs: dict[str, Any] = {}
        if flat_actions is not None:
            action_tensor = torch.from_numpy(flat_actions).to(device=device)
            forward_inputs["action"] = action_tensor
        if return_obs:
            forward_inputs["main_images"] = env_obs["main_images"]
            if env_obs.get("states", None) is not None:
                forward_inputs["states"] = env_obs["states"]
            forward_inputs["task_descriptions"] = env_obs.get("task_descriptions", None)
            forward_inputs["generated_text"] = gen_texts

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
