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


import os
import pickle
from typing import Any, Optional, Union

import torch
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from rlinf.data.datasets.common.item import SftDatasetItem
from rlinf.data.datasets.vlm.base import VLMBaseDataset
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry


def _video_metadata(video: Any, fps: float) -> VideoMetadata:
    """Describe an in-memory video without resampling its frames."""
    frame_count = len(video)
    return VideoMetadata(
        total_num_frames=frame_count,
        fps=fps,
        duration=frame_count / fps,
        frames_indices=list(range(frame_count)),
    )


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
        self._video_fps = float(config.data.get("video_fps") or 24.0)

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
        video_fps: float = 24.0,
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
                    _video_metadata(video, video_fps) for video in videos_i
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
            "video_metadata": [_video_metadata(video, video_fps) for video in videos],
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
            video_fps=self._video_fps,
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
            raise ValueError(f"Sample {idx} pkl missing dual-view frame arrays")
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
