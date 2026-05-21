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

import asyncio
import logging
import os
import signal
import sys
import tempfile
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


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


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
        self._generate_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rlinf_sglang_reward",
        )

        self.prepare_inputs_ms = 0.0
        self.media_convert_ms = 0.0
        self.sglang_generate_ms = 0.0
        self.parse_ms = 0.0
        self.total_ms = 0.0
        self.last_timing_ms: dict[str, float] = {}
        self.last_generation_stats: dict[str, float] = {}
        self.last_outputs: list[str] = []

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
        previous_sigquit_handler = signal.getsignal(signal.SIGQUIT)

        def signal_without_sigquit(signum: int, handler: Any) -> Any:
            if signum == signal.SIGQUIT:
                return previous_sigquit_handler
            return original_signal(signum, handler)

        signal.signal = signal_without_sigquit
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
            observation_key: self._slice_batch_value(observation_values, start, end)
            for observation_key, observation_values in observations.items()
        }

    def _slice_batch_value(self, value: Any, start: int, end: int) -> Any:
        if isinstance(value, dict):
            return {
                key: self._slice_batch_value(nested_value, start, end)
                for key, nested_value in value.items()
            }
        if isinstance(value, torch.Tensor):
            return (
                value[start:end] if value.ndim > 0 and value.shape[0] >= end else value
            )
        if isinstance(value, np.ndarray):
            return (
                value[start:end] if value.ndim > 0 and value.shape[0] >= end else value
            )
        if isinstance(value, list):
            return value[start:end] if len(value) >= end else value
        if isinstance(value, tuple):
            return value[start:end] if len(value) >= end else value
        return value

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

    def _convert_video(self, video: list[Any]) -> bytes:
        frames = [self._frame_to_numpy(frame) for frame in video]
        if len(frames) == 1:
            frames.append(frames[0])
        return self._encode_video_bytes(frames, self.video_fps)

    def _encode_video_bytes(self, frames: list[np.ndarray], fps: float) -> bytes:
        import cv2

        height, width = frames[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(float(fps), 1.0),
            (width, height),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError("Failed to create temporary mp4 video for SGLang")
            for frame in frames:
                writer.write(np.ascontiguousarray(frame[..., ::-1]))
        finally:
            writer.release()

        try:
            with open(tmp_path, "rb") as video_file:
                return video_file.read()
        finally:
            os.unlink(tmp_path)

    def _build_sglang_inputs(
        self,
        prepared_inputs: dict[str, Any],
    ) -> tuple[list[str], list[list[bytes]]]:
        prompt_texts_list = prepared_inputs.get("prompt_texts_list") or []
        videos_list = prepared_inputs.get("videos_list") or []

        prompts: list[str] = []
        video_data: list[list[tuple[list[np.ndarray], dict[str, Any]]]] = []
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

    def _generate(
        self, prompts: list[str], video_data: list[list[Any]]
    ) -> tuple[list[str], list[int]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._generate_sync(prompts, video_data)

        return self._generate_executor.submit(
            self._generate_sync,
            prompts,
            video_data,
        ).result()

    def _generate_sync(
        self, prompts: list[str], video_data: list[list[Any]]
    ) -> tuple[list[str], list[int]]:
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
