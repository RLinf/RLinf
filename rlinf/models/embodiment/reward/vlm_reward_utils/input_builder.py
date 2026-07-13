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

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor

from rlinf.data.datasets.vlm import (
    QwenTrendProgressSFTDataset,
    VLMBaseDataset,
)

logger = logging.getLogger(__name__)


def _to_pil_images(
    images: Union[torch.Tensor, list[torch.Tensor]],
) -> list[Image.Image]:
    """Convert EnvOutput image tensors to per-sample PIL image lists.

    Expected EnvOutput image formats: [B, H, W, C]
    """
    if isinstance(images, torch.Tensor):
        arr = images.detach().cpu().numpy()
    elif isinstance(images, list):
        if len(images) == 0:
            return []
        arr = torch.stack(images).cpu().numpy()
    else:
        raise TypeError(f"Unsupported image input type: {type(images)}")

    arr = arr[None, ...] if arr.ndim == 3 else arr
    arr = arr[:, 0] if arr.ndim == 5 and arr.shape[1] == 1 else arr
    if arr.ndim != 4:
        raise ValueError(f"Invalid image batch shape for PIL conversion: {arr.shape}")

    return [Image.fromarray(frame[..., :3]).convert("RGB") for frame in arr]


def extract_images(
    observations: dict[str, Any],
    image_keys: list[str],
) -> list[list[Any]]:
    """
    Args:
        observations: dict[str, Any], shape = [num_envs, ...]
        image_keys: list[str], shape = [num_image_keys]

    Returns:
        list[list[Any]]: images array with shape [num_envs, num_image_keys]
    """
    image_keys = image_keys or ["main_images"]
    batch_size = observations[image_keys[0]].shape[0]
    images: list[list[Any]] = [[] for _ in range(batch_size)]

    for image_key in image_keys:
        images_all_env = observations[image_key]
        if images_all_env is None:
            continue
        images_all_env = _to_pil_images(images_all_env)
        for i in range(batch_size):
            images[i].append(images_all_env[i])

    return images


INPUT_BUILDER_REGISTRY: dict[str, type] = {}


def register_input_builder(name: str):
    def decorator(cls: type):
        INPUT_BUILDER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_input_builder(name: str) -> type:
    name_lower = name.lower()
    if name_lower not in INPUT_BUILDER_REGISTRY:
        raise ValueError(f"InputBuilder '{name}' not registered")
    return INPUT_BUILDER_REGISTRY[name_lower]


@register_input_builder("base_input_builder")
@dataclass
class BaseInputBuilder:
    system_prompt: Optional[str] = None
    use_chat_template: bool = True
    image_keys: list[str] = field(default_factory=lambda: ["main_images"])
    _processor: Optional[AutoProcessor] = field(default=None)

    def get_valid_input_ids(self, observations: dict[str, Any]) -> list[int]:
        return list(range(len(observations[self.image_keys[0]])))

    def prepare_inputs(
        self, observations: dict[str, Any], valid_input_ids: list[int]
    ) -> torch.Tensor:
        return {"images_list": None, "videos_list": None, "prompt_texts_list": None}

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        return prepared_inputs

    def build_inputs(self, observations: dict[str, Any], device: torch.device):
        valid_input_ids = self.get_valid_input_ids(observations)
        prepared_inputs = self.prepare_inputs(observations, valid_input_ids)
        processed_inputs = self.process_inputs(prepared_inputs)
        processed_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in processed_inputs.items()
        }
        return processed_inputs


@register_input_builder("base_vlm_input_builder")
@dataclass
class BaseVLMInputBuilder(BaseInputBuilder):
    def prepare_inputs(self, observations: dict[str, Any], valid_input_ids: list[int]):
        images = extract_images(observations, self.image_keys)
        images_list = [images[env_idx] for env_idx in valid_input_ids]
        task_descriptions = [
            str(observations["task_descriptions"][env_idx] or "")
            for env_idx in valid_input_ids
        ]

        prompt_texts_list: list[str] = []
        for task_description in task_descriptions:
            task_description = task_description.strip()
            prompt_texts = [
                # One prompt text
                f"Task: {task_description}\n\n"
                "Evaluate the task and return a reward score between 0 and 1."
            ]
            prompt_texts_list.append(prompt_texts)
        return {
            "images_list": images_list,
            "videos_list": None,
            "prompt_texts_list": prompt_texts_list,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        images_list = prepared_inputs.get("images_list")

        processed_inputs: dict[str, Any] = {}
        for prompt_texts, images in zip(prompt_texts_list, images_list):
            _, processed_input = VLMBaseDataset.process_inputs(
                self._processor,
                self.system_prompt,
                self.use_chat_template,
                prompt_texts=prompt_texts,
                images=images,
            )
            for key, value in processed_input.items():
                if isinstance(value, torch.Tensor):
                    processed_inputs[key] = (
                        value
                        if key not in processed_inputs
                        else torch.cat([processed_inputs[key], value], dim=0)
                    )
                else:
                    processed_inputs[key] = value
        return processed_inputs


@register_input_builder("history_vlm_input_builder")
@dataclass(kw_only=True)
class HistoryVLMInputBuilder(BaseVLMInputBuilder):
    history_buffer_names: list[str]

    def get_valid_input_ids(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> list[int]:
        histories = tuple(
            history
            for history_buffer in history_input.values()
            for history in history_buffer.values()
        )
        valid_ids = range(len(histories[0]))
        return [
            env_id
            for env_id in valid_ids
            if all(history[env_id] for history in histories)
        ]

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        del history_input
        return {"images_list": None, "videos_list": None, "prompt_texts_list": None}

    def build_inputs(
        self,
        observations: dict[str, Any],
        device: torch.device,
        history_input: dict[str, dict[str, list[list[Any]]]],
    ):
        valid_input_ids = self.get_valid_input_ids(observations, history_input)
        if len(valid_input_ids) == 0:
            return {}, valid_input_ids

        prepared_inputs = self.prepare_inputs(
            observations, history_input, valid_input_ids
        )
        processed_inputs = self.process_inputs(prepared_inputs)
        processed_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in processed_inputs.items()
        }
        return processed_inputs, valid_input_ids


@register_input_builder("video_vlm_input_builder")
@dataclass
class VideoVLMInputBuilder(HistoryVLMInputBuilder):
    video_keys: list[str] = field(default_factory=lambda: ["main_images"])

    def extract_videos(
        self,
        history_buffer: dict[str, list[list[Any]]],
        video_keys: Optional[list[str]] = None,
    ) -> list[list[Any]]:
        """
        Convert one named history buffer payload into processor-ready videos.
        """
        video_keys = video_keys or self.video_keys
        if not video_keys:
            return []

        first_video_key = video_keys[0]
        batch_size = len(history_buffer.get(first_video_key, []))

        if batch_size == 0:
            return []

        videos: list[list[list[Image.Image]]] = [
            [[] for _ in video_keys] for _ in range(batch_size)
        ]

        for batch_idx in range(batch_size):
            for video_idx, video_key in enumerate(video_keys):
                video_frames = _to_pil_images(history_buffer[video_key][batch_idx])
                videos[batch_idx][video_idx].extend(video_frames)

        return videos


@register_input_builder("qwentrend_input_builder")
@dataclass
class QwentrendInputBuilder(VideoVLMInputBuilder):
    video_keys: list[str] = field(
        default_factory=lambda: ["main_images", "extra_view_images"]
    )
    default_task_description: str = ""

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})
        videos_clip = self.extract_videos(history_window, self.video_keys)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]
        task_descriptions = observations.get(
            "task_descriptions",
            [self.default_task_description] * len(videos_clip),
        )

        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            prompt_texts_list.append(
                [
                    f"You are currently performing the task: {task_descriptions[env_id]}. "
                    "You are given two synchronized 5-frame videos from different camera "
                    "views (main view and third-person view) of the same robot action "
                    "window. Judge whether the action trend is positive, negative, or "
                    "unclear. Answer with exactly one word: positive, negative, or unclear."
                ]
            )

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        videos_list = prepared_inputs.get("videos_list")

        _, processed_inputs, _ = QwenTrendProgressSFTDataset.process_inputs(
            processor=self._processor,
            system_prompt=self.system_prompt,
            use_chat_template=self.use_chat_template,
            prompt_texts=prompt_texts_list,
            videos=videos_list,
            answer_text=None,
        )
        return processed_inputs


@register_input_builder("qwentrend_terminal_success_input_builder")
@dataclass
class QwentrendTerminalSuccessInputBuilder(QwentrendInputBuilder):
    """Build the exact binary prompt used by terminal-success SFT."""

    include_task: bool = True

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})
        videos_clip = self.extract_videos(history_window, self.video_keys)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]
        task_descriptions = observations.get(
            "task_descriptions",
            [self.default_task_description] * len(videos_clip),
        )

        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            task = str(
                task_descriptions[env_id] or self.default_task_description
            ).strip()
            task_text = f" Task: {task}." if self.include_task and task else ""
            prompt_texts_list.append(
                [
                    "Estimate task-conditioned success potential for this robot "
                    f"manipulation state.{task_text} The two synchronized videos show "
                    "the same 5-frame history from two camera views."
                ]
            )

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }


@register_input_builder("qwentrend_potential_input_builder")
@dataclass
class QwentrendPotentialInputBuilder(QwentrendInputBuilder):
    """Build the online five-frame input used by potential SFT."""

    include_task: bool = True
    num_bins: int = 10

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})
        videos_clip = self.extract_videos(history_window, self.video_keys)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]
        task_descriptions = observations.get(
            "task_descriptions",
            [self.default_task_description] * len(videos_clip),
        )

        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            task = str(
                task_descriptions[env_id] or self.default_task_description
            ).strip()
            task_text = f" Task: {task}." if self.include_task and task else ""
            prompt_texts_list.append(
                [
                    "You are estimating task-conditioned success potential for a "
                    f"robot manipulation state.{task_text} The two synchronized "
                    "videos show the same 5-frame history from two camera views. "
                    "Predict the final state's potential as exactly one digit from "
                    f"0 to {self.num_bins - 1}, where 0 is furthest from eventual "
                    f"success and {self.num_bins - 1} is closest."
                ]
            )

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }
