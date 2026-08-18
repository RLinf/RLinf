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


import hashlib
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from rlinf.data.datasets.common.item import SftDatasetItem
from rlinf.data.datasets.vlm.base import VLMBaseDataset
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry

DEFAULT_TASK_DESCRIPTION = (
    "Pick up the red cube and place it on the green spot on the table."
)


def _video_metadata(video: Any) -> VideoMetadata:
    """Describe an in-memory video without resampling its frames."""
    frame_count = len(video)
    return VideoMetadata(
        total_num_frames=frame_count,
        fps=24.0,
        duration=frame_count / 24.0,
        frames_indices=list(range(frame_count)),
    )


def to_uint8_rgb(image: Any) -> np.ndarray:
    """Convert an image tensor/array to uint8 RGB."""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")
    return image[..., :3]


def to_numpy_float32(value: Any) -> np.ndarray:
    """Convert tensor/array metadata to float32 numpy."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def extract_extra_view_image(extra_view_images: Any) -> Any | None:
    """Return the first extra-view frame from either a frame or frame stack."""
    if extra_view_images is None:
        return None
    ndim = getattr(extra_view_images, "ndim", 0)
    if ndim == 3:
        return extra_view_images
    if ndim == 4 and len(extra_view_images):
        return extra_view_images[0]
    return None


def extract_dual_view_frames(
    observations: list[dict[str, Any]], start_idx: int, end_idx: int
) -> tuple[list[Any], list[Any]] | None:
    """Extract aligned main and extra camera frames for one inclusive window."""
    main, extra = [], []
    for observation in observations[start_idx : end_idx + 1]:
        main_frame = observation.get("main_images")
        extra_frame = observation.get("third_view_images")
        if extra_frame is None:
            extra_frame = extract_extra_view_image(observation.get("extra_view_images"))
        if main_frame is None or extra_frame is None:
            return None
        main.append(main_frame)
        extra.append(extra_frame)
    return (main, extra) if len(main) == end_idx - start_idx + 1 else None


def transition_observations(
    episode: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Return action-aligned observations and their offset in the episode.

    Current collectors store a leading reset observation, while older episode
    pickles may contain only post-action observations. Online history buffers
    see post-action observations only, so skip the reset entry when present.
    """
    observations = episode.get("observations", [])
    actions = episode.get("actions", [])
    offset = int(len(observations) == len(actions) + 1)
    count = min(len(actions), len(observations) - offset)
    return observations[offset : offset + count], offset


def first_success_transition(
    episode: dict[str, Any], transition_count: int
) -> int | None:
    """Return the first action-aligned success index, if the episode succeeded."""
    infos = episode.get("infos", [])
    actions = episode.get("actions", [])
    info_offset = int(len(infos) == len(actions) + 1)
    for index, info in enumerate(infos[info_offset : info_offset + transition_count]):
        if isinstance(info, dict) and _as_bool(info.get("success")):
            return index
    if bool(episode.get("success", False)) and transition_count:
        return transition_count - 1
    return None


def load_episode_pickle(path: str) -> dict[str, Any] | None:
    """Load one rollout pickle, returning None for unreadable files."""
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except (EOFError, pickle.UnpicklingError, OSError):
        return None


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    return bool(value.item() if hasattr(value, "item") else value)


def inspect_episode(
    path: str,
    window_size: int,
    default_task: str = DEFAULT_TASK_DESCRIPTION,
) -> dict[str, Any] | None:
    """Read the fields shared by terminal-success preprocessing."""
    episode = load_episode_pickle(path)
    if episode is None:
        return None
    observations, observation_offset = transition_observations(episode)
    actions = episode.get("actions", [])
    if len(observations) < window_size or len(actions) < window_size:
        return None
    end_step = min(len(observations), len(actions)) - 1
    first_success = first_success_transition(episode, end_step + 1)
    success = bool(episode.get("success", False) or first_success is not None)
    if first_success is not None:
        end_step = min(end_step, first_success)
    success_steps = [end_step] if success else []
    absolute_path = os.path.abspath(path)
    return {
        "path": absolute_path,
        "task": str(
            episode.get("task")
            or episode.get("task_description")
            or episode.get("task_name")
            or default_task
        ),
        "end_step": end_step,
        "observation_offset": observation_offset,
        "success_steps": success_steps,
        "success": success,
        "is_complete": (
            success
            or bool(episode.get("terminated", []) and episode["terminated"][-1])
            or bool(episode.get("truncated", []) and episode["truncated"][-1])
        ),
        "source_run": Path(absolute_path).parent.parent.name,
    }


def split_for(path: str, val_split: float) -> str:
    """Assign a source episode to a stable train/eval split."""
    fraction = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 2**32
    return "eval" if fraction < val_split else "train"


def source_episode_hash(path: str) -> int:
    """Return a stable integer hash used for rank-local feature extraction."""
    return int(hashlib.sha256(path.encode()).hexdigest()[:16], 16)


def potential_prompt(task: str, window_size: int, num_bins: int = 10) -> str:
    """Build the absolute-potential prompt used offline and online."""
    return (
        "You are estimating task-conditioned success potential for a robot "
        f"manipulation state. Task: {task}. The two synchronized videos show "
        f"the same {window_size}-frame history from two camera views. Predict "
        f"the final state's potential as exactly one digit from 0 to {num_bins - 1}, "
        f"where 0 is furthest from eventual success and {num_bins - 1} is closest."
    )


def progress_prompt(task: str, window_size: int, gap_steps: int | None = None) -> str:
    """Build the paired-window progress prompt."""
    gap_steps = window_size if gap_steps is None else gap_steps
    relation = (
        "immediately adjacent"
        if gap_steps == window_size
        else f"separated by {gap_steps} environment steps"
    )
    return (
        "You are judging local task progress in a robot manipulation trajectory. "
        f"Task: {task}. In each synchronized camera video, the first {window_size} "
        f"frames are the earlier clip and the next {window_size} frames are the "
        f"later clip; their final states are {relation}. Compare their final states. "
        "Answer with exactly one word: up, same, or down."
    )


def _terminal_row(
    item: dict[str, Any], window_size: int, end_step: int, success: bool
) -> dict[str, Any]:
    answer = "1" if success else "0"
    observation_offset = item["observation_offset"]
    source_end_step = end_step + observation_offset
    prompt = (
        "Estimate task-conditioned success potential for this robot manipulation "
        f"state. Task: {item['task']}. The two synchronized videos show the same "
        f"{window_size}-frame history from two camera views."
    )
    return {
        "task": item["task"],
        "prompt": prompt,
        "question": prompt,
        "answer": answer,
        "pkl_path": item["path"],
        "source_episode_path": item["path"],
        "source_run": item["source_run"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "segment_metadata": {
            "start_step": source_end_step - window_size + 1,
            "end_step": source_end_step,
            "window_size": window_size,
            "progress_gap_steps": None,
            "success": success,
            "sample_type": "potential",
            "target_name": "terminal_success",
            "is_complete": item["is_complete"],
            "target_type": "success_observed" if success else "online_negative",
            "source_run": item["source_run"],
        },
        "supervision": {
            "score_name": "terminal_success",
            "teacher_value": float(success),
            "teacher_delta": 0.0,
        },
    }


def build_terminal_success_rows(
    raw_data_paths: list[str],
    window_size: int,
    interval: int,
    val_split: float,
    workers: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build unbalanced online-matched 0/1 windows.

    Every fixed-interval window is retained at the online inference cadence.
    No class balancing, positive oversampling, synthetic positives, or
    hard-negative mining is performed.
    """
    paths = sorted(path for root in raw_data_paths for path in Path(root).glob("*.pkl"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = list(
            executor.map(lambda p: inspect_episode(str(p), window_size), paths)
        )
    items = [item for item in inspected if item is not None]
    rows_by_split = {"train": [], "eval": []}
    stats: dict[str, Any] = {"input_episodes": len(paths), "splits": {}}
    for split in rows_by_split:
        rows = rows_by_split[split]
        for item in items:
            if split_for(item["path"], val_split) != split:
                continue
            first = window_size - 1
            end_steps = list(range(first, item["end_step"] + 1, interval))
            success_steps = {step for step in item["success_steps"] if step >= first}
            end_steps.extend(success_steps - set(end_steps))
            rows.extend(
                _terminal_row(item, window_size, end, end in success_steps)
                for end in sorted(end_steps)
            )
        random.Random(seed + (split == "eval")).shuffle(rows)
        positives = sum(row["answer"] == "1" for row in rows)
        stats["splits"][split] = {
            "positive": positives,
            "negative": len(rows) - positives,
            "interval": interval,
        }
    stats["complete_episodes"] = sum(item["is_complete"] for item in items)
    stats["partial_episodes"] = len(items) - stats["complete_episodes"]
    return rows_by_split, stats


def _resolve_video_path(path: str, data_root: Optional[str]) -> str:
    """Resolve a video path, rewriting it with data_root if the original path is missing."""
    if not isinstance(path, str):
        return str(path)
    if os.path.isfile(path):
        return path
    if not data_root:
        return path
    # Extract the data/ suffix from absolute paths and rewrite with data_root
    # to handle host-to-container path differences.
    idx = path.find("data/")
    if idx >= 0:
        resolved = os.path.join(data_root, path[idx:])
        if os.path.isfile(resolved):
            return resolved
    # Relative path: join it directly with data_root.
    if not os.path.isabs(path):
        resolved = os.path.join(data_root, path)
        if os.path.isfile(resolved):
            return resolved
    return path


@VLMDatasetRegistry.register("vlm_trend_reward_sft")
class VLMTrendRewardSFTDataset(VLMBaseDataset):
    """SFT dataset for VLM Trend reward: full_video + video_clip input.

    Each record: full_video (mp4), video_clip (mp4), question, answer.
    Qwen3-VL uses videos input, matching the dataset format.
    If JSONL paths are host absolute paths, set data.data_root to resolve them in the container.
    """

    def __init__(
        self,
        data_paths: Union[list[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
        eval_dataset: bool = False,
    ) -> None:
        super().__init__(data_paths, config, tokenizer)
        self.eval_dataset = eval_dataset
        self._data_root = config.data.get("data_root") or os.environ.get(
            "RLINF_DATA_ROOT"
        )

    @classmethod
    def _build_video_user_content(
        cls, processor: AutoProcessor, prompt_text: str
    ) -> str:
        video_tok = getattr(processor, "video_token", "<|video_pad|>")
        return f"{video_tok}\n{video_tok}\n\n{prompt_text}"

    @classmethod
    def process_inputs(
        cls,
        processor: AutoProcessor,
        system_prompt: Optional[str],
        use_chat_template: bool,
        prompt_texts: list[str] | list[list[str]],
        videos: list[Any] | list[list[Any]],
        answer_text: Optional[str] | list[Optional[str]] = None,
    ) -> tuple[str | list[str], dict[str, Any], dict[str, Any]]:
        """
        Build Qwen3-VL processor inputs for VLM Trend reward SFT.
        """

        def _render_prompt_text(
            prompt_text: str, answer_text_i: Optional[str]
        ) -> tuple[str, str]:
            user_content = cls._build_video_user_content(processor, prompt_text)

            try:
                if answer_text_i is None:
                    rendered_prompt = processor.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    rendered_label = rendered_prompt
                else:
                    rendered_prompt = processor.apply_chat_template(
                        [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": answer_text_i},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    rendered_label = processor.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            except Exception:
                rendered_prompt = (
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                rendered_label = rendered_prompt

            return rendered_prompt, rendered_label

        is_batch_input = bool(prompt_texts) and isinstance(prompt_texts[0], list)
        if is_batch_input:
            prompt_texts_batch: list[list[str]] = prompt_texts
            videos_batch: list[list[Any]] = videos
            batch_size = len(prompt_texts_batch)

            if isinstance(answer_text, list):
                assert len(answer_text) == batch_size, (
                    f"answer_text list size {len(answer_text)} does not match batch size {batch_size}"
                )
                answer_text_batch = answer_text
            else:
                answer_text_batch = [answer_text for _ in range(batch_size)]

            rendered_prompts: list[str] = []
            rendered_labels: list[str] = []
            videos_kwargs = {"video_metadata": []}

            for prompt_texts_i, videos_i, answer_text_i in zip(
                prompt_texts_batch, videos_batch, answer_text_batch
            ):
                rendered_prompt_i, rendered_label_i = _render_prompt_text(
                    prompt_texts_i[0], answer_text_i
                )
                rendered_prompts.append(rendered_prompt_i)
                rendered_labels.append(rendered_label_i)
                videos_kwargs["video_metadata"].extend(
                    _video_metadata(video) for video in videos_i
                )

            full_inputs = processor(
                text=rendered_prompts,
                videos=videos_batch,
                return_tensors="pt",
                padding=True,
                videos_kwargs=videos_kwargs,
            )

            if all(answer is None for answer in answer_text_batch):
                label_inputs = {"attention_mask": full_inputs["attention_mask"]}
            else:
                label_inputs = processor(
                    text=rendered_labels,
                    videos=videos_batch,
                    return_tensors="pt",
                    padding=True,
                    videos_kwargs=videos_kwargs,
                )

            return rendered_prompts, full_inputs, label_inputs

        prompt_text = prompt_texts[0]
        rendered_prompt, rendered_label = _render_prompt_text(prompt_text, answer_text)
        videos_kwargs = {
            "video_metadata": [_video_metadata(video) for video in videos],
        }

        full_inputs = processor(
            text=[rendered_prompt],
            videos=videos,
            return_tensors="pt",
            padding=True,
            videos_kwargs=videos_kwargs,
        )

        if answer_text is None:
            # Inference: avoid an extra processor call for label_text.
            # Downstream expects `label_inputs` to at least provide `attention_mask`.
            label_inputs = {"attention_mask": full_inputs["attention_mask"]}
        else:
            label_inputs = processor(
                text=[rendered_label],
                videos=videos,
                return_tensors="pt",
                padding=True,
                videos_kwargs=videos_kwargs,
            )

        return rendered_prompt, full_inputs, label_inputs

    def encode_prompt(
        self,
        prompt_text: str,
        videos: list[str],
        answer_text: str,
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Encode prompt into input_ids + masks for SFT training.

        Returns:
          - input_ids: token ids for (user + assistant)
          - length: number of tokens in input_ids
          - attention_mask: mask for input_ids
          - label_mask: mask for prompt part only (used to keep answer tokens)
          - multi_modal_inputs: processor outputs excluding text-only tensors
        """
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.actor.model.model_path
            )
            do_sample_frames = self.cfg.data.get("video_do_sample_frames")
            if do_sample_frames is not None:
                self._processor.video_processor.do_sample_frames = bool(
                    do_sample_frames
                )

        _, full_inputs, label_inputs = self.process_inputs(
            processor=self._processor,
            system_prompt=self.system_prompt,
            use_chat_template=self.use_chat_template,
            prompt_texts=[prompt_text],
            videos=videos,
            answer_text=answer_text,
        )

        input_ids = full_inputs.pop("input_ids")
        attention_mask = full_inputs.pop("attention_mask")
        label_mask = label_inputs.pop("attention_mask")

        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2 and input_ids.size(0) == 1:
                input_ids = input_ids.squeeze(0)
            input_ids = input_ids.to(dtype=torch.long)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        plen = int(input_ids.numel())
        multi_modal_inputs = dict(full_inputs)
        return input_ids, plen, attention_mask, label_mask, multi_modal_inputs

    @classmethod
    def _parse_raw_record(
        cls,
        raw: dict[str, Any],
        idx: int,
        data_root: Optional[str],
    ) -> tuple[str, str, list[Any], list[Any]]:
        full_video = raw.get("full_video")
        video_clip = raw.get("video_clip")
        question = str(raw.get("question", ""))
        answer_text = str(raw.get("answer", ""))

        if full_video and video_clip:
            full_video = _resolve_video_path(str(full_video), data_root)
            video_clip = _resolve_video_path(str(video_clip), data_root)
            return (
                question,
                answer_text,
                [full_video, video_clip],
                [full_video, video_clip],
            )

        pkl_path = raw.get("pkl_path")
        if not pkl_path:
            raise ValueError(f"Sample {idx} missing full_video/video_clip or pkl_path")

        resolved_pkl_path = _resolve_video_path(str(pkl_path), data_root)
        with open(resolved_pkl_path, "rb") as f:
            payload = pickle.load(f)
        main_frames = payload.get("main_frames")
        extra_view_frames = payload.get("extra_view_frames")
        if main_frames is None or extra_view_frames is None:
            metadata = raw.get("segment_metadata", {})
            observations = payload.get("observations")
            start, end = metadata.get("start_step"), metadata.get("end_step")
            if observations is None or start is None or end is None:
                raise ValueError(
                    f"Sample {idx} pkl missing dual-view frames or episode metadata"
                )
            frames = extract_dual_view_frames(observations, int(start), int(end))
            if frames is None:
                raise ValueError(f"Sample {idx} has an invalid dual-view window")
            main_frames, extra_view_frames = frames
        return (
            question,
            answer_text,
            [main_frames, extra_view_frames],
            [resolved_pkl_path, resolved_pkl_path],
        )

    def _process_raw_record(self, raw: dict[str, Any], idx: int) -> "SftDatasetItem":
        prompt_text, answer_text, videos, image_data = self._parse_raw_record(
            raw, idx, self._data_root
        )
        input_ids, plen, attention_mask, label_mask, multi_modal_inputs = (
            self.encode_prompt(
                prompt_text=prompt_text,
                videos=videos,
                answer_text=answer_text,
            )
        )
        if plen > self.max_prompt_length:
            input_ids = input_ids[: self.max_prompt_length]
            attention_mask = attention_mask[..., : self.max_prompt_length]
            label_mask = label_mask[..., : self.max_prompt_length]
            plen = self.max_prompt_length
        return SftDatasetItem(
            prompt=input_ids,
            length=plen,
            idx=idx,
            image_data=image_data,
            answer=answer_text,
            prompt_text=prompt_text,
            attention_mask=attention_mask,
            label_mask=label_mask,
            meta=None,
            multi_modal_inputs=multi_modal_inputs,
        )


@VLMDatasetRegistry.register("simple_vlm_trend_reward_sft")
class SimpleVLMTrendRewardSFTDataset(VLMTrendRewardSFTDataset):
    """SFT dataset for a single-video, single-word VLM Trend reward format."""

    @classmethod
    def _build_video_user_content(
        cls, processor: AutoProcessor, prompt_text: str
    ) -> str:
        video_tok = getattr(processor, "video_token", "<|video_pad|>")
        return f"{video_tok}\n\n{prompt_text}"

    @classmethod
    def _parse_raw_record(
        cls,
        raw: dict[str, Any],
        idx: int,
        data_root: Optional[str],
    ) -> tuple[str, str, list[str], list[str]]:
        clip_path = raw.get("clip_path") or raw.get("video_clip")
        if not clip_path:
            raise ValueError(f"Sample {idx} missing clip_path or video_clip")

        clip_path = _resolve_video_path(str(clip_path), data_root)
        prompt_text = str(raw.get("prompt") or raw.get("question") or "").strip()
        if not prompt_text:
            raise ValueError(f"Sample {idx} missing prompt or question")

        supervision = raw.get("supervision")
        supervision_label = (
            supervision.get("label", "") if isinstance(supervision, dict) else ""
        )
        answer_text = (
            str(raw.get("answer") or raw.get("label") or supervision_label)
            .strip()
            .lower()
        )
        return prompt_text, answer_text, [clip_path], [clip_path]
