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

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)

logger = logging.getLogger(__name__)

_RAW_FRAME_VIDEO_FORMAT = "rlinf_raw_frames"


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


class _RawFrameBatch:
    def __init__(self, frames: np.ndarray):
        self._frames = frames

    def asnumpy(self) -> np.ndarray:
        return self._frames


class _RawFrameVideoReader:
    """Small in-memory subset of decord.VideoReader used by SGLang Qwen-VL."""

    def __init__(self, frames: np.ndarray, fps: float):
        if frames.ndim != 4:
            raise ValueError(
                "Raw video frames must have shape (num_frames, height, width, channels), "
                f"got {frames.shape}."
            )
        if frames.shape[0] == 0:
            raise ValueError("SGLang video reward input cannot be empty.")
        self._frames = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
        self._fps = max(float(fps), 1.0)

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices: Any) -> _RawFrameBatch:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        selected = np.ascontiguousarray(self._frames[[int(index) for index in indices]])
        return _RawFrameBatch(selected)


class _RawFrameVideoData(dict[str, Any]):
    """Dict-backed video item that bypasses SGLang load_video and acts as a reader."""

    def __init__(self, frames: np.ndarray, fps: float):
        frame_array = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
        frame_fps = max(float(fps), 1.0)
        super().__init__(
            {
                "format": _RAW_FRAME_VIDEO_FORMAT,
                "frames": frame_array,
                "fps": frame_fps,
            }
        )
        self._reader = _RawFrameVideoReader(frame_array, frame_fps)

    def __len__(self) -> int:
        return len(self._reader)

    def get_avg_fps(self) -> float:
        return self._reader.get_avg_fps()

    def get_batch(self, indices: Any) -> _RawFrameBatch:
        return self._reader.get_batch(indices)


def _is_raw_frame_video_data(value: Any) -> bool:
    return isinstance(value, dict) and value.get("format") == _RAW_FRAME_VIDEO_FORMAT


def _raw_frame_video_reader_from_data(value: dict[str, Any]) -> _RawFrameVideoReader:
    return _RawFrameVideoReader(
        frames=np.asarray(value["frames"], dtype=np.uint8),
        fps=float(value.get("fps", 1.0)),
    )


def _install_sglang_raw_frame_video_patch() -> bool:
    """Teach SGLang Qwen-VL preprocessing to accept RLinf raw-frame dicts."""

    try:
        from sglang.srt.multimodal.processors import qwen_vl
    except Exception as exc:
        logger.debug("Skipping SGLang raw-frame video patch: %s", exc)
        return False

    current_preprocess_video = getattr(qwen_vl, "preprocess_video", None)
    if current_preprocess_video is None:
        return False
    if getattr(current_preprocess_video, "_rlinf_raw_frame_patch", False):
        return True

    async def preprocess_video_with_raw_frames(
        video: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if _is_raw_frame_video_data(video):
            video = _raw_frame_video_reader_from_data(video)
        return await current_preprocess_video(video, *args, **kwargs)

    preprocess_video_with_raw_frames._rlinf_raw_frame_patch = True
    preprocess_video_with_raw_frames._rlinf_original = current_preprocess_video
    qwen_vl.preprocess_video = preprocess_video_with_raw_frames
    return True


class HistoryVLMSGLangRewardModel(BaseRewardModel):
    """SGLang-backed embodied history VLM reward model.

    The backend reuses RLinf's history VLM input builders and reward parsers, but
    sends rendered Qwen chat prompts plus in-memory video frame lists directly to
    SGLang native ``Engine.generate(video_data=...)``.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.model_path: str = cfg.get("model_path")
        if not self.model_path:
            raise ValueError(
                "reward.model.model_path must be set for HistoryVLMSGLangRewardModel"
            )

        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(cfg.get("infer_micro_batch_size", 0))
        self.input_mode: str = str(cfg.get("input_mode", "video_data"))
        if self.input_mode != "video_data":
            raise ValueError(
                "HistoryVLMSGLangRewardModel currently supports only "
                f"input_mode='video_data', got {self.input_mode!r}."
            )

        self.video_fps: float = float(cfg.get("video_fps", 24.0))
        self.sglang_engine_kwargs = _to_plain_dict(cfg.get("sglang_engine_kwargs", {}))
        self.sglang_engine_kwargs.setdefault("model_path", self.model_path)
        self.sglang_engine_kwargs.setdefault("trust_remote_code", True)

        self.sampling_params = self._build_sampling_params(cfg)
        self._engine: Any | None = None
        self.prepare_inputs_ms = 0.0
        self.media_convert_ms = 0.0
        self.sglang_generate_ms = 0.0
        self.parse_ms = 0.0
        self.total_ms = 0.0
        self.last_timing_ms: dict[str, float] = {}
        self.last_generation_stats: dict[str, float] = {}
        self.last_outputs: list[str] = []
        self._empty_input_log_count = 0
        self._executor: ThreadPoolExecutor | None = None

        self.setup_processor()
        self.setup_input_builder()
        self.setup_reward_parser()

    def setup_processor(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._setup_subprocessor(self.cfg.get("subprocessor_kwargs", {}))

    def _setup_subprocessor(self, subprocessor_kwargs: dict) -> None:
        for subprocessor_name, kwargs in subprocessor_kwargs.items():
            subprocessor = getattr(self._processor, subprocessor_name, None)
            if subprocessor is None:
                continue
            for key, value in dict(kwargs).items():
                if hasattr(subprocessor, key):
                    setattr(subprocessor, key, value)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "history_vlm_input_builder")
        )(
            **self.cfg.get("input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        assert isinstance(self.input_builder, HistoryVLMInputBuilder), (
            "HistoryVLMSGLangRewardModel only supports HistoryVLMInputBuilder"
        )

    def setup_reward_parser(self) -> None:
        self.reward_parser = get_reward_parser(
            self.cfg.get("reward_parser_name", "base_reward_parser")
        )(**self.cfg.get("reward_parser_params", {}))

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HistoryVLMSGLangRewardModel is an inference-time reward model; "
            "training via forward() is not supported."
        )

    def _build_sampling_params(self, cfg: DictConfig) -> dict[str, Any]:
        sampling_params = _to_plain_dict(cfg.get("sampling_params", {}))
        sampling_params.setdefault("max_new_tokens", int(cfg.get("max_new_tokens", 32)))
        if cfg.get("min_new_tokens", None) is not None:
            sampling_params.setdefault("min_new_tokens", int(cfg.get("min_new_tokens")))
        if cfg.get("ignore_eos", None) is not None:
            sampling_params.setdefault("ignore_eos", bool(cfg.get("ignore_eos")))
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = float(cfg.get("temperature", 0.0))
        if "top_p" not in sampling_params and cfg.get("top_p", None) is not None:
            sampling_params["top_p"] = float(cfg.get("top_p"))
        return sampling_params

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            _install_sglang_raw_frame_video_patch()

            from sglang import Engine

            python_bin_dir = os.path.dirname(sys.executable)
            os.environ["PATH"] = (
                python_bin_dir
                if not os.environ.get("PATH")
                else f"{python_bin_dir}{os.pathsep}{os.environ['PATH']}"
            )
            self._engine = self._create_engine(Engine)
        return self._engine

    def _create_engine(self, engine_cls: Any) -> Any:
        if threading.current_thread() is threading.main_thread():
            return engine_cls(**self.sglang_engine_kwargs)

        original_signal = signal.signal

        def signal_without_launch_sigquit(signum: int, handler: Any) -> Any:
            if signum == signal.SIGQUIT:
                logger.warning(
                    "Skipping SGLang launch SIGQUIT handler registration because "
                    "the reward worker is not running in the Python main thread."
                )
                return signal.SIG_DFL
            return original_signal(signum, handler)

        signal.signal = signal_without_launch_sigquit
        try:
            return engine_cls(**self.sglang_engine_kwargs)
        finally:
            signal.signal = original_signal

    def slice_history_input(
        self,
        history_input: dict[str, dict[str, list[list[Any]]]],
        start: int,
        end: int,
    ) -> dict[str, dict[str, list[list[Any]]]]:
        return {
            buffer_name: {
                history_key: env_sequences[start:end]
                for history_key, env_sequences in history_buffer.items()
            }
            for buffer_name, history_buffer in history_input.items()
        }

    def slice_observations(
        self,
        observations: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        return {
            observation_key: observation_values[start:end]
            for observation_key, observation_values in observations.items()
        }

    def _render_prompt(self, prompt_texts: list[str], videos: list[Any]) -> str:
        video_tok = getattr(self._processor, "video_token", "<|video_pad|>")
        prompt_text = prompt_texts[0]
        user_content = "\n".join([video_tok] * len(videos)) + f"\n\n{prompt_text}"

        try:
            return self._processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return (
                f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            )

    def _frame_to_numpy(self, frame: Any) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        elif isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGB"))
        else:
            frame = np.asarray(frame)

        if (
            frame.ndim == 3
            and frame.shape[0] in (1, 3, 4)
            and frame.shape[-1] not in (1, 3, 4)
        ):
            frame = np.moveaxis(frame, 0, -1)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(frame[..., :3])

    def _convert_video(self, video: list[Any]) -> _RawFrameVideoData:
        frames = [self._frame_to_numpy(frame) for frame in video]
        if not frames:
            raise ValueError("SGLang video reward input cannot be empty.")

        if len(frames) == 1:
            frames = frames * 2

        frame_array = np.stack(frames, axis=0)
        return _RawFrameVideoData(frame_array, fps=self.video_fps)

    def _build_sglang_inputs(
        self,
        prepared_inputs: dict[str, Any],
    ) -> tuple[list[str], list[list[_RawFrameVideoData]]]:
        prompt_texts_list = prepared_inputs.get("prompt_texts_list") or []
        videos_list = prepared_inputs.get("videos_list") or []

        prompts: list[str] = []
        video_data: list[list[_RawFrameVideoData]] = []
        for prompt_texts, videos in zip(prompt_texts_list, videos_list):
            prompts.append(self._render_prompt(prompt_texts, videos))
            video_data.append([self._convert_video(video) for video in videos])

        return prompts, video_data

    def _extract_texts_and_token_counts(
        self, result: Any
    ) -> tuple[list[str], list[int]]:
        completion_tokens: list[int] = []

        def append_completion_tokens(item: Any) -> None:
            if not isinstance(item, dict):
                return
            meta_info = item.get("meta_info") or {}
            if not isinstance(meta_info, dict):
                return
            token_count = meta_info.get("completion_tokens")
            if token_count is None:
                return
            if isinstance(token_count, list):
                completion_tokens.extend(int(value) for value in token_count)
            else:
                completion_tokens.append(int(token_count))

        if isinstance(result, dict):
            append_completion_tokens(result)
            text = result.get("text", "")
            if isinstance(text, list):
                texts = [str(item) for item in text]
            else:
                texts = [str(text)]
            return texts, completion_tokens

        if isinstance(result, list):
            outputs: list[str] = []
            for item in result:
                if isinstance(item, dict):
                    append_completion_tokens(item)
                    outputs.append(str(item.get("text", "")))
                else:
                    outputs.append(str(item))
            return outputs, completion_tokens

        return [str(result)], []

    @staticmethod
    def _summarize_generation_stats(completion_tokens: list[int]) -> dict[str, float]:
        if not completion_tokens:
            return {}
        return {
            "generated_tokens_mean": float(np.mean(completion_tokens)),
            "generated_tokens_min": float(min(completion_tokens)),
            "generated_tokens_max": float(max(completion_tokens)),
        }

    def _log_empty_valid_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> None:
        self._empty_input_log_count += 1
        if self._empty_input_log_count not in {1, 10}:
            return

        history_lengths = {
            buffer_name: {
                history_key: [len(video) for video in videos]
                for history_key, videos in history_buffer.items()
            }
            for buffer_name, history_buffer in history_input.items()
        }
        task_descriptions = observations.get("task_descriptions") or []
        nonempty_task_count = sum(
            1 for task_description in task_descriptions if task_description
        )
        logger.warning(
            "HistoryVLMSGLangRewardModel received no valid inputs; "
            "observation_keys=%s, nonempty_task_descriptions=%d/%d, "
            "history_lengths=%s",
            sorted(observations.keys()),
            nonempty_task_count,
            len(task_descriptions),
            history_lengths,
        )

    def _generate(
        self, prompts: list[str], video_data: list[list[Any]]
    ) -> tuple[list[str], list[int]]:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sglang_reward",
            )
        return self._executor.submit(
            self._generate_sync,
            prompts,
            video_data,
        ).result()

    def _generate_sync(
        self, prompts: list[str], video_data: list[list[Any]]
    ) -> tuple[list[str], list[int]]:
        _install_sglang_raw_frame_video_patch()
        result = self._ensure_engine().generate(
            prompt=prompts,
            sampling_params=self.sampling_params,
            video_data=video_data,
        )
        return self._extract_texts_and_token_counts(result)

    def _set_timing_attrs(
        self,
        timings: dict[str, float],
        generation_stats: dict[str, float],
    ) -> None:
        self.prepare_inputs_ms = timings["prepare_inputs_ms"]
        self.media_convert_ms = timings["media_convert_ms"]
        self.sglang_generate_ms = timings["sglang_generate_ms"]
        self.parse_ms = timings["parse_ms"]
        self.total_ms = timings["total_ms"]
        self.last_timing_ms = {
            **timings,
            "generate_ms": timings["sglang_generate_ms"],
        }
        self.last_generation_stats = dict(generation_stats)
        logger.debug("HistoryVLMSGLangRewardModel timing_ms=%s", self.last_timing_ms)

    @torch.no_grad()
    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        total_start = time.perf_counter()
        timings = {
            "prepare_inputs_ms": 0.0,
            "media_convert_ms": 0.0,
            "sglang_generate_ms": 0.0,
            "parse_ms": 0.0,
            "total_ms": 0.0,
        }

        history_input: dict[str, dict[str, list[list[Any]]]] = reward_input.pop(
            "history_input"
        )
        input_batch_size = len(next(iter(next(iter(history_input.values())).values())))
        observations = reward_input

        infer_micro_batch_size = self.infer_micro_batch_size or input_batch_size

        reward_chunks: list[torch.Tensor] = []
        all_outputs: list[str] = []
        generated_token_counts: list[int] = []
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            micro_observations = self.slice_observations(observations, start, end)
            micro_history_input = self.slice_history_input(history_input, start, end)
            reward_chunk = torch.zeros((end - start,), dtype=torch.float32)

            valid_input_ids = self.input_builder.get_valid_input_ids(
                micro_observations,
                micro_history_input,
            )
            if len(valid_input_ids) == 0:
                self._log_empty_valid_inputs(
                    micro_observations,
                    micro_history_input,
                )
                reward_chunks.append(reward_chunk)
                continue

            prepare_start = time.perf_counter()
            prepared_inputs = self.input_builder.prepare_inputs(
                micro_observations,
                micro_history_input,
                valid_input_ids,
            )
            timings["prepare_inputs_ms"] += (time.perf_counter() - prepare_start) * 1000

            media_start = time.perf_counter()
            prompts, video_data = self._build_sglang_inputs(prepared_inputs)
            timings["media_convert_ms"] += (time.perf_counter() - media_start) * 1000

            generate_start = time.perf_counter()
            outputs, token_counts = self._generate(prompts, video_data)
            generated_token_counts.extend(token_counts)
            all_outputs.extend(outputs)
            timings["sglang_generate_ms"] += (
                time.perf_counter() - generate_start
            ) * 1000

            if len(outputs) != len(valid_input_ids):
                logger.warning(
                    "SGLang reward output count mismatch: outputs=%d valid_inputs=%d",
                    len(outputs),
                    len(valid_input_ids),
                )
                outputs = (outputs + [""] * len(valid_input_ids))[
                    : len(valid_input_ids)
                ]

            parse_start = time.perf_counter()
            parsed_rewards = self.reward_parser.parse_rewards(outputs).to(
                dtype=torch.float32
            )
            timings["parse_ms"] += (time.perf_counter() - parse_start) * 1000

            reward_chunk[valid_input_ids] = parsed_rewards
            reward_chunks.append(reward_chunk)

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_outputs = all_outputs
        self._set_timing_attrs(
            timings,
            self._summarize_generation_stats(generated_token_counts),
        )
        return torch.cat(reward_chunks, dim=0)
