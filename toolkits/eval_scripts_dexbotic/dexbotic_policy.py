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

from typing import Any, Optional

import numpy as np
import torch
from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM
from dexbotic.tokenization.process import Pi0Tokenization
from PIL import Image
from transformers import AutoTokenizer


class DexboticPolicy:
    def __init__(
        self,
        model_path: str,
        norm_stats: dict[str, Any],
        action_dim: int = 7,
        num_images: int = 3,
        non_delta_mask: Optional[list[int]] = None,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.num_images = num_images
        self.non_delta_mask = non_delta_mask or [6]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._load_model()

        self.timestep = 0
        self.episode = 0
        self.prev_text = None

    def _load_model(self):
        self.model = Pi0ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.tokenization_func = Pi0Tokenization(self.tokenizer)
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

    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        base_image = observation["observation/image"]
        wrist_image = observation["observation/wrist_image"]
        state = observation["observation/state"]
        text = observation["prompt"]

        images = [
            Image.fromarray(base_image.astype(np.uint8)),
            Image.fromarray(wrist_image.astype(np.uint8)),
        ]
        batch_images_tensor = [
            self.model.process_images(images).to(dtype=torch.bfloat16)
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
            [self.tokenization_func([{"value": text}])["input_ids"]]
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
            actions = self.model.inference_action(**inputs)
        outputs = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = actions.detach().cpu().numpy()
        outputs = self.output_transform(outputs)
        action_sequence = outputs["action"][0, :, : self.action_dim]

        self.timestep += 1

        return {"actions": action_sequence}
