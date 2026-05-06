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
    RoboChallengeProgressSFTDataset,
    SimpleRobochallengeSFTDataset,
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

    per_sample: list[Image.Image] = []
    for i in range(arr.shape[0]):
        per_sample.append(Image.fromarray(arr[i][..., :3]).convert("RGB"))
    return per_sample  # [B, H, W, C]


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


def _process_video_prompt_inputs(
    processor: AutoProcessor,
    prompt_texts_list: list[list[str]],
    videos_list: list[list[Any]],
) -> dict[str, Any]:
    """Build batched Qwen3-VL inputs with one video token per input video."""

    video_tok = getattr(processor, "video_token", "<|video_pad|>")
    rendered_prompts: list[str] = []
    videos_kwargs = {"video_metadata": []}

    for prompt_texts, videos in zip(prompt_texts_list, videos_list):
        prompt_text = prompt_texts[0]
        user_content = "\n".join([video_tok] * len(videos)) + f"\n\n{prompt_text}"

        try:
            rendered_prompt = processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered_prompt = (
                f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            )

        rendered_prompts.append(rendered_prompt)
        videos_kwargs["video_metadata"].append(
            [{"total_num_frames": len(video), "fps": 24.0} for video in videos]
        )

    return processor(
        text=rendered_prompts,
        videos=videos_list,
        return_tensors="pt",
        padding=True,
        videos_kwargs=videos_kwargs,
    )


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
        return list(
            range(len(next(iter(history_input[self.history_buffer_names[0]].values()))))
        )

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


@register_input_builder("robochanallenge_input_builder")
@dataclass
class RobochanallengeInputBuilder(VideoVLMInputBuilder):
    def get_valid_input_ids(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> list[int]:
        history_valid_ids = super().get_valid_input_ids(observations, history_input)

        observations_valid_ids = [
            env_id
            for env_id, task_description in enumerate(observations["task_descriptions"])
            if len(task_description or "") > 0
        ]
        history_valid_ids = [
            env_idx
            for env_idx in history_valid_ids
            if env_idx in observations_valid_ids
        ]

        for history_buffer_name in self.history_buffer_names:
            history_buffer = history_input.get(history_buffer_name)
            if not history_buffer:
                return []

            for video_key in self.video_keys:
                video_buffer = history_buffer.get(video_key)
                if not video_buffer:
                    return []

                video_invalid_ids = [
                    env_idx
                    for env_idx, video in enumerate(video_buffer)
                    if len(video) == 0
                ]
                history_valid_ids = [
                    env_idx
                    for env_idx in history_valid_ids
                    if env_idx not in video_invalid_ids
                ]

        return history_valid_ids

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        full_history = history_input.get("full_history", {})
        history_window = history_input.get("history_window", {})

        full_videos = self.extract_videos(full_history)
        videos_clip = self.extract_videos(history_window)

        if len(full_videos) != len(videos_clip):
            raise ValueError(
                "Mismatched history buffer batch sizes: "
                f"full_history={len(full_videos)}, history_window={len(videos_clip)}."
            )

        videos_list: list[list[Any]] = []
        for env_id in valid_input_ids:
            videos_list.append(full_videos[env_id] + videos_clip[env_id])

        task_descriptions = [
            str(task_description or "")
            for task_description in observations["task_descriptions"]
        ]
        prompt_texts_list: list[str] = []
        for env_id in valid_input_ids:
            task_description = task_descriptions[env_id].strip()
            prompt_texts = [
                # One prompt text
                f"Task: {task_description}.\n\n"
                "Evaluate <|video_clip|> using <|full_video|> only as background context.\n\n"
                "Judge the trend inside the clip by comparing its beginning and end.\n\n"
                "Positive: the clip shows visible task-relevant progress, including a necessary intermediate step such as approaching, contacting, grasping, lifting, aligning, or moving toward the target.\n"
                "Negative: the clip shows no visible task-relevant progress, or shows regression or damage.\n\n"
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "clip_summary": "<describe the trend within the clip>",\n'
                '  "cot": [\n'
                "    {\n"
                '      "subtask": "<task-relevant subtask>",\n'
                '      "status": "completed|improved|unchanged|worsened|broken",\n'
                '      "evidence": "<visible evidence about the change within the clip>"\n'
                "    }\n"
                "  ],\n"
                '  "judgement": "positive|negative",\n'
                '  "confidence": 1.0|0.6|0.2\n'
                "}\n\n"
                "Rules:\n"
                "- Focus on trend, not just final completion.\n"
                "- Describe ONLY what happens within <|video_clip|>.\n"
                "- Use <|full_video|> only to understand the task and the pre-clip context, not as direct evidence for what happens inside the clip.\n"
                "- Do NOT describe actions, progress, or object changes that are visible only in <|full_video|> but not within <|video_clip|>.\n"
                '- "clip_summary", "cot", and "judgement" must all be grounded in the clip interval itself.\n'
                "- If an object is already in a helpful state before the clip starts, that alone does not make the clip positive unless the clip itself shows further progress.\n"
                "- For each subtask, describe the change that happens within the clip interval itself.\n"
                "- Do NOT describe only the absolute final state at the end of the clip.\n"
                "- The status must reflect the difference between the beginning of the clip and the end of the clip.\n"
                '- "completed" means the subtask is visibly completed within this clip interval.\n'
                '- "improved" means the subtask becomes more advanced within this clip interval.\n'
                '- "unchanged" means no visible task-relevant change happens within this clip interval.\n'
                '- "worsened" means the subtask becomes less favorable within this clip interval.\n'
                '- "broken" means a previously favorable state is clearly destroyed within this clip interval.\n'
                "- If a grasp, contact, lift, alignment, or similar intermediate step is established during the clip, treat that as a positive change within the clip rather than only describing the end state.\n"
                "- Subtasks must be task-goal-relevant. Do not use generic motion labels unless they clearly improve task progress.\n"
                '- If a trend is hard to map to one subtask, describe it in "clip_summary".\n'
                "- Use confidence 1.0 only for clear, unambiguous evidence.\n"
                "- If a negative clip still shows any plausible slight progress or useful precondition change, do not use confidence 1.0.\n"
                "- Use 0.6 for weak but plausible evidence, and 0.2 for very ambiguous evidence.\n"
                '- "cot" must contain 2-5 items.\n'
                '- Each "subtask" must be short, concrete, visually grounded, and task-goal-relevant.\n'
                '- Each "evidence" must be short and based only on visible changes.\n'
                '- "clip_summary" should be concise and should not duplicate the whole cot.\n'
                '- "judgement" must agree with the summary and cot.\n'
                '- "confidence" must be exactly 1.0, 0.6, or 0.2.\n'
                "- Do not mention invisible intentions, sensors, forces, or hidden states.\n"
            ]

            prompt_texts_list.append(prompt_texts)

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        videos_list = prepared_inputs.get("videos_list")

        _, processed_inputs, _ = RoboChallengeProgressSFTDataset.process_inputs(
            processor=self._processor,
            system_prompt=self.system_prompt,
            use_chat_template=self.use_chat_template,
            prompt_texts=prompt_texts_list,
            videos=videos_list,
        )

        return processed_inputs


@register_input_builder("threeview_robochallenge_input_builder")
@dataclass
class ThreeViewRobochallengeInputBuilder(RobochanallengeInputBuilder):
    """Original RoboChallenge prompt with full video + two clip views."""

    full_video_keys: list[str] = field(default_factory=lambda: ["main_images"])
    clip_video_keys: list[str] = field(
        default_factory=lambda: ["main_images", "extra_view_images"]
    )
    strict_min_frames: int = 1

    def get_valid_input_ids(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> list[int]:
        if not self.history_buffer_names:
            return []
        first_history = history_input.get(self.history_buffer_names[0])
        if not first_history:
            return []
        first_payload = next(iter(first_history.values()), None)
        if first_payload is None:
            return []

        valid_ids = list(range(len(first_payload)))
        valid_ids = [
            env_id
            for env_id in valid_ids
            if len(observations["task_descriptions"][env_id] or "") > 0
        ]

        required_buffers = [
            ("full_history", self.full_video_keys),
            ("history_window", self.clip_video_keys),
        ]
        for history_buffer_name, video_keys in required_buffers:
            history_buffer = history_input.get(history_buffer_name)
            if not history_buffer:
                return []

            for video_key in video_keys:
                video_buffer = history_buffer.get(video_key)
                if not video_buffer:
                    return []

                invalid_ids = [
                    env_idx
                    for env_idx, video in enumerate(video_buffer)
                    if len(video) < self.strict_min_frames
                ]
                valid_ids = [
                    env_idx for env_idx in valid_ids if env_idx not in invalid_ids
                ]

        return valid_ids

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        base_inputs = super().prepare_inputs(
            observations, history_input, valid_input_ids
        )

        full_history = history_input.get("full_history", {})
        history_window = history_input.get("history_window", {})
        full_videos = self.extract_videos(full_history, self.full_video_keys)
        main_clip_videos = self.extract_videos(history_window, ["main_images"])
        extra_clip_videos = self.extract_videos(history_window, ["extra_view_images"])

        if not (len(full_videos) == len(main_clip_videos) == len(extra_clip_videos)):
            raise ValueError(
                "Mismatched three-view history buffer batch sizes: "
                f"full_history={len(full_videos)}, "
                f"main_clip={len(main_clip_videos)}, "
                f"extra_clip={len(extra_clip_videos)}."
            )

        videos_list: list[list[Any]] = []
        for env_id in valid_input_ids:
            videos_list.append(
                full_videos[env_id]
                + main_clip_videos[env_id]
                + extra_clip_videos[env_id]
            )

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": base_inputs["prompt_texts_list"],
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        videos_list = prepared_inputs.get("videos_list")
        return _process_video_prompt_inputs(
            processor=self._processor,
            prompt_texts_list=prompt_texts_list,
            videos_list=videos_list,
        )


@register_input_builder("simple_robochallenge_input_builder")
@dataclass
class SimpleRobochallengeInputBuilder(RobochanallengeInputBuilder):
    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})

        videos_clip = self.extract_videos(history_window)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]

        task_descriptions = [
            str(task_description or "")
            for task_description in observations["task_descriptions"]
        ]
        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            task_description = task_descriptions[env_id].strip()
            prompt_texts = [
                f"You are currently performing the task: {task_description}. "
                "Please judge whether the operation shown in this video segment "
                "makes the task better or worse. Answer with exactly one word: "
                "positive or negative."
            ]
            prompt_texts_list.append(prompt_texts)

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        videos_list = prepared_inputs.get("videos_list")

        _, processed_inputs, _ = SimpleRobochallengeSFTDataset.process_inputs(
            processor=self._processor,
            system_prompt=self.system_prompt,
            use_chat_template=self.use_chat_template,
            prompt_texts=prompt_texts_list,
            videos=videos_list,
        )

        return processed_inputs


@register_input_builder("simple_dualview_ternary_input_builder")
@dataclass
class SimpleDualViewTernaryInputBuilder(VideoVLMInputBuilder):
    video_keys: list[str] = field(
        default_factory=lambda: ["main_images", "extra_view_images"]
    )
    strict_min_frames: int = 2
    _consecutive_missing_extra_view_empties: int = field(
        default=0, init=False, repr=False
    )

    def _maybe_log_missing_extra_view(self, observations: dict[str, Any]) -> None:
        if observations.get("extra_view_images") is not None:
            self._consecutive_missing_extra_view_empties = 0
            return

        self._consecutive_missing_extra_view_empties += 1
        if self._consecutive_missing_extra_view_empties in {1, 10}:
            logger.warning(
                "SimpleDualViewTernaryInputBuilder produced no valid inputs because "
                "observations.extra_view_images is None. Reward outputs will stay at "
                "0.0 until a second view is available. consecutive_empty_calls=%d",
                self._consecutive_missing_extra_view_empties,
            )

    def get_valid_input_ids(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> list[int]:
        history_valid_ids = super().get_valid_input_ids(observations, history_input)
        observations_valid_ids = [
            env_id
            for env_id, task_description in enumerate(observations["task_descriptions"])
            if len(task_description or "") > 0
        ]
        history_valid_ids = [
            env_idx
            for env_idx in history_valid_ids
            if env_idx in observations_valid_ids
        ]

        for history_buffer_name in self.history_buffer_names:
            history_buffer = history_input.get(history_buffer_name)
            if not history_buffer:
                return []
            for video_key in self.video_keys:
                video_buffer = history_buffer.get(video_key)
                if not video_buffer:
                    return []
                invalid_ids = [
                    env_idx
                    for env_idx, video in enumerate(video_buffer)
                    if len(video) < self.strict_min_frames
                ]
                history_valid_ids = [
                    env_idx
                    for env_idx in history_valid_ids
                    if env_idx not in invalid_ids
                ]
        if history_valid_ids:
            self._consecutive_missing_extra_view_empties = 0
        else:
            self._maybe_log_missing_extra_view(observations)
        return history_valid_ids

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})
        videos_clip = self.extract_videos(history_window, self.video_keys)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]

        task_descriptions = [
            str(task_description or "")
            for task_description in observations["task_descriptions"]
        ]
        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            task_description = task_descriptions[env_id].strip()
            prompt_texts_list.append(
                [
                    f"You are currently performing the task: {task_description}. "
                    "Please judge whether the operation shown in these two video views "
                    "makes the task better, worse, or unchanged. "
                    "Answer with exactly one word: positive, negative, or unchanged."
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

        _, processed_inputs, _ = RoboChallengeProgressSFTDataset.process_inputs(
            processor=self._processor,
            system_prompt=self.system_prompt,
            use_chat_template=self.use_chat_template,
            prompt_texts=prompt_texts_list,
            videos=videos_list,
            answer_text=None,
        )
        return processed_inputs


@register_input_builder("dualview_robochallenge_input_builder")
@dataclass
class DualViewInputBuilder(VideoVLMInputBuilder):
    video_keys: list[str] = field(
        default_factory=lambda: ["main_images", "extra_view_images"]
    )

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})
        videos_clip = self.extract_videos(history_window, self.video_keys)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]

        task_descriptions = [
            str(task_description or "")
            for task_description in observations["task_descriptions"]
        ]
        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            task_description = task_descriptions[env_id].strip()
            prompt_texts_list.append(
                [
                    f"You are currently performing the task: {task_description}. "
                    "Please judge whether the operation shown in these two video views "
                    "makes the task better, worse, or unchanged. "
                    "Answer with exactly one word: positive, negative, or unchanged."
                ]
            )

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }
