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

"""Dexbotic evaluation utilities for RLinf."""

import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization, Pi0Tokenization
from PIL import Image
from transformers import AutoTokenizer


class BaseDexboticPolicy:
    """Base class for Dexbotic-style policies.

    Subclasses must define:
        model_cls:          The model class to load (e.g. Pi0ForCausalLM).
        tokenization_cls:   The tokenization class to use (e.g. Pi0Tokenization).

    Subclasses may override:
        _build_tokenization_input(text): Returns the conversations list passed to
            the tokenization function.
        _call_inference_action(inputs): Calls model.inference_action with the
            appropriate keyword arguments.
    """

    model_cls = None
    tokenization_cls = None

    def __init__(
        self,
        model_path: str,
        action_dim: int = 7,
        num_images: int = 3,
        non_delta_mask: Optional[list[int]] = None,
        device: str = "cuda",
        diffusion_steps: int = 10,
    ):
        self.model_path = model_path
        self.action_dim = action_dim
        self.num_images = num_images
        self.non_delta_mask = non_delta_mask or [6]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.diffusion_steps = diffusion_steps
        norm_stats_file = os.path.join(model_path, "norm_stats.json")
        self.norm_stats = self._read_normalization_stats(norm_stats_file)
        self._load_model()
        self.timestep = 0
        self.episode = 0
        self.prev_text = None

    def _read_normalization_stats(self, norm_stats_file):
        if not os.path.exists(norm_stats_file):
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_stats_file}. "
                "Make sure the checkpoint directory contains norm_stats.json"
            )
        with open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def _load_model(self):
        # Set HF_HUB_OFFLINE to prevent any network access during model loading
        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            config = self.model_cls.config_class.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            config.processor_config = self.model_path
            self.model = self.model_cls.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                local_files_only=True,
            ).to(self.device)
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=False, local_files_only=True
        )
        self.tokenization_func = self.tokenization_cls(self.tokenizer)
        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False),
                AbsoluteAction(),
            ]
        )

    def reset(self):
        self.timestep = 0
        self.episode += 1
        self.prev_text = None

    def _build_tokenization_input(self, text: str) -> list:
        """Return the conversations list passed to the tokenization function."""
        raise NotImplementedError

    def _call_inference_action(self, inputs: dict) -> torch.Tensor:
        """Call model.inference_action with the appropriate arguments."""
        raise NotImplementedError

    def infer(self, observation: dict) -> dict:
        base_image = observation["observation/image"]
        wrist_image = observation["observation/wrist_image"]
        state = observation["observation/state"]
        text = observation["prompt"]
        images = [
            Image.fromarray(base_image.astype(np.uint8)),
            Image.fromarray(wrist_image.astype(np.uint8)),
        ]
        batch_images_tensor = [
            self.model.process_images(images).to(dtype=self.model.dtype)
        ]
        num_input_images = len(images)
        if num_input_images < self.num_images:
            batch_images_tensor = [
                torch.cat(
                    [
                        image_tensor,
                        torch.zeros_like(image_tensor[0:1]).repeat(
                            self.num_images - num_input_images, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
                for image_tensor in batch_images_tensor
            ]
        batch_image_masks = [
            torch.tensor(
                [True for _ in range(num_input_images)]
                + [False for _ in range(self.num_images - num_input_images)],
                device=self.device,
            )
        ]
        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)
        batch_image_masks = torch.stack(batch_image_masks, dim=0)
        batch_input_ids = np.array(
            [self.tokenization_func(self._build_tokenization_input(text))["input_ids"]]
        )
        batch_attention_mask = np.array(
            [np.array(ids != self.tokenizer.pad_token_id) for ids in batch_input_ids]
        )
        batch_states = np.array([state], dtype=np.float32)
        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(self.non_delta_mask),
            },
        }
        inputs = self.input_transform(inference_args)
        inputs["states"] = inputs["state"]
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            actions = self._call_inference_action(inputs)
        outputs = {
            k: v.detach().cpu().float().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = actions.detach().cpu().float().numpy()
        outputs = self.output_transform(outputs)
        action_sequence = outputs["action"][0, :, : self.action_dim]
        self.timestep += 1
        return {"actions": action_sequence}


class DexboticPI0Policy(BaseDexboticPolicy):
    """Policy for Pi0ForCausalLM (Gemma-based VLM + action expert)."""

    model_cls = Pi0ForCausalLM
    tokenization_cls = Pi0Tokenization

    def _build_tokenization_input(self, text: str) -> list:
        return [{"value": text}]

    def _call_inference_action(self, inputs: dict) -> torch.Tensor:
        return self.model.inference_action(**inputs)


class DM0Policy(BaseDexboticPolicy):
    """Policy for DM0ForCausalLM (Qwen3-based VLM + action expert)."""

    model_cls = DM0ForCausalLM
    tokenization_cls = DM0Tokenization

    def _build_tokenization_input(self, text: str) -> list:
        return [{"from": "human", "value": text}]

    def _call_inference_action(self, inputs: dict) -> torch.Tensor:
        return self.model.inference_action(
            **inputs, diffusion_steps=self.diffusion_steps
        )


def setup_logger(exp_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    return logger


# Mapping from config name to Policy class.
# Use config_name to select the model, e.g. "db_pi0_libero" or "dm0_libero".
_POLICY_CLASS_DICT = {
    "db_pi0_libero": DexboticPI0Policy,
    "dm0_libero": DM0Policy,
}


def setup_policy(args):
    config_name = getattr(args, "config_name", "db_pi0_libero")
    if config_name not in _POLICY_CLASS_DICT:
        raise ValueError(
            f"Unknown config_name '{config_name}'. "
            f"Available options: {list(_POLICY_CLASS_DICT.keys())}"
        )
    policy_cls = _POLICY_CLASS_DICT[config_name]
    policy = policy_cls(
        model_path=args.pretrained_path,
        action_dim=getattr(args, "action_dim", 7),
        num_images=getattr(args, "num_images", 3),
        non_delta_mask=getattr(args, "non_delta_mask", [6]),
        device=getattr(args, "device", "cuda"),
        diffusion_steps=getattr(args, "num_steps", 10),
    )
    return policy


__all__ = [
    "setup_policy",
    "setup_logger",
    "BaseDexboticPolicy",
    "DexboticPI0Policy",
    "DM0Policy",
    "_POLICY_CLASS_DICT",
]
