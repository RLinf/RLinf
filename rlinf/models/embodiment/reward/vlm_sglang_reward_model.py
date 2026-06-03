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

import base64
import io
import time
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import requests
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
from rlinf.utils.logging import get_logger

logger = get_logger()


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


@contextmanager
def _record_timing_ms(timings: dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] += (time.perf_counter() - start) * 1000


class HistoryVLMSGLangRewardModel(BaseRewardModel):
    """SGLang HTTP-backed embodied history VLM reward model.

    The runner starts an external SGLang OpenAI-compatible server for this
    backend unless ``reward.model.api_base`` points to a user-managed server.
    The reward worker sends one synchronous ``/v1/chat/completions`` request
    per valid history-window input using multiple ``image_url`` data URLs.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.model_path: str = cfg.get("model_path")
        if not self.model_path:
            raise ValueError(
                f"reward.model.model_path must be set for {self.__class__.__name__}"
            )

        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(cfg.get("infer_micro_batch_size", 0))
        self.gt_success_bonus = float(cfg.get("gt_success_bonus", 0.0))
        self.api_base = str(cfg.get("api_base") or "").rstrip("/")
        if not self.api_base:
            host = str(cfg.get("server_host", "127.0.0.1"))
            port = int(cfg.get("server_port", 30000))
            self.api_base = f"http://{host}:{port}/v1"
        self.model_name = str(cfg.get("served_model_name") or self.model_path)
        self.request_timeout = float(cfg.get("request_timeout", 120.0))
        self.image_format = str(cfg.get("image_format", "jpeg")).lower()
        if self.image_format not in {"jpeg", "png"}:
            raise ValueError(
                "HistoryVLMSGLangRewardModel supports image_format='jpeg' or 'png', "
                f"got {self.image_format!r}."
            )
        self.jpeg_quality = int(cfg.get("jpeg_quality", 95))

        self.sampling_params = self._build_sampling_params(cfg)

        self.prepare_inputs_ms = 0.0
        self.image_encode_ms = 0.0
        self.media_convert_ms = 0.0
        self.http_request_ms = 0.0
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
            f"{self.__class__.__name__} only supports HistoryVLMInputBuilder"
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
        sampling_params.setdefault("max_tokens", int(cfg.get("max_new_tokens", 32)))
        if cfg.get("min_new_tokens", None) is not None:
            sampling_params.setdefault("min_tokens", int(cfg.get("min_new_tokens")))
        if cfg.get("ignore_eos", None) is not None:
            sampling_params.setdefault("ignore_eos", bool(cfg.get("ignore_eos")))
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = float(cfg.get("temperature", 0.0))
        if "top_p" not in sampling_params and cfg.get("top_p", None) is not None:
            sampling_params["top_p"] = float(cfg.get("top_p"))
        return sampling_params

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
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return (
                value[start:end] if value.ndim > 0 and value.shape[0] >= end else value
            )
        if isinstance(value, (list, tuple)):
            return value[start:end] if len(value) >= end else value
        return value

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

    def _frame_to_data_url(self, frame: Any) -> str:
        image = Image.fromarray(self._frame_to_numpy(frame), mode="RGB")
        image_buffer = io.BytesIO()
        if self.image_format == "jpeg":
            image.save(image_buffer, format="JPEG", quality=self.jpeg_quality)
            mime_type = "image/jpeg"
        else:
            image.save(image_buffer, format="PNG")
            mime_type = "image/png"
        encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _build_content_items(
        self,
        prompt_texts: list[str],
        videos: list[list[Any]],
    ) -> list[dict[str, Any]]:
        prompt_text = prompt_texts[0] if prompt_texts else ""
        content: list[dict[str, Any]] = []
        for video in videos:
            for frame in video:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._frame_to_data_url(frame)},
                    }
                )
        content.append({"type": "text", "text": prompt_text})
        return content

    def _build_chat_payloads(
        self,
        prepared_inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompt_texts_list = prepared_inputs.get("prompt_texts_list") or []
        videos_list = prepared_inputs.get("videos_list") or []

        payloads: list[dict[str, Any]] = []
        for prompt_texts, videos in zip(prompt_texts_list, videos_list):
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": self._build_content_items(prompt_texts, videos),
                    }
                ],
            }
            payload.update(self.sampling_params)
            payloads.append(payload)
        return payloads

    def _extract_text_and_token_count(
        self, response: dict[str, Any]
    ) -> tuple[str, int]:
        choices = response.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, list):
                text = "".join(str(item.get("text", "")) for item in content)
            else:
                text = str(content)

        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")
        return text, int(completion_tokens or 0)

    @staticmethod
    def _summarize_generation_stats(completion_tokens: list[int]) -> dict[str, float]:
        completion_tokens = [count for count in completion_tokens if count > 0]
        if not completion_tokens:
            return {}
        return {
            "generated_tokens_mean": float(np.mean(completion_tokens)),
            "generated_tokens_min": float(min(completion_tokens)),
            "generated_tokens_max": float(max(completion_tokens)),
        }

    def _chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        response = requests.post(url, json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        return response.json()

    def _generate(self, payloads: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
        outputs: list[str] = []
        completion_tokens: list[int] = []
        for payload in payloads:
            text, token_count = self._extract_text_and_token_count(
                self._chat_completion(payload)
            )
            outputs.append(text)
            completion_tokens.append(token_count)
        return outputs, completion_tokens

    def _set_timing_attrs(
        self,
        timings: dict[str, float],
        generation_stats: dict[str, float],
    ) -> None:
        self.prepare_inputs_ms = timings["prepare_inputs_ms"]
        self.image_encode_ms = timings["image_encode_ms"]
        self.media_convert_ms = timings["image_encode_ms"]
        self.http_request_ms = timings["http_request_ms"]
        self.sglang_generate_ms = timings["http_request_ms"]
        self.parse_ms = timings["parse_ms"]
        self.total_ms = timings["total_ms"]
        self.last_timing_ms = {
            **timings,
            "media_convert_ms": timings["image_encode_ms"],
            "sglang_generate_ms": timings["http_request_ms"],
            "generate_ms": timings["http_request_ms"],
        }
        self.last_generation_stats = dict(generation_stats)
        logger.debug("HistoryVLMSGLangRewardModel timing_ms=%s", self.last_timing_ms)

    def apply_gt_success_bonus(
        self, rewards: torch.Tensor, reward_input: dict[str, Any]
    ) -> torch.Tensor:
        if rewards is None or self.gt_success_bonus == 0.0:
            return rewards
        env_infos = (
            reward_input.get("env_infos") if isinstance(reward_input, dict) else None
        )
        if not isinstance(env_infos, dict):
            return rewards

        success = None
        final_info = env_infos.get("final_info", {})
        for info_dict in (
            env_infos,
            env_infos.get("episode"),
            final_info,
            final_info.get("episode") if isinstance(final_info, dict) else None,
        ):
            if not isinstance(info_dict, dict):
                continue
            for key in ("success", "success_at_end", "success_once"):
                value = info_dict.get(key)
                if value is not None:
                    success = torch.as_tensor(value).reshape(-1).bool()
                    break
            if success is not None:
                break

        if success is None or success.shape[0] != rewards.shape[0]:
            return rewards
        bonus = success.to(device=rewards.device, dtype=rewards.dtype)
        return rewards + (bonus * self.gt_success_bonus).view(
            -1, *([1] * (rewards.dim() - 1))
        )

    @torch.no_grad()
    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        total_start = time.perf_counter()
        timings = {
            "prepare_inputs_ms": 0.0,
            "image_encode_ms": 0.0,
            "http_request_ms": 0.0,
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

            with _record_timing_ms(timings, "prepare_inputs_ms"):
                prepared_inputs = self.input_builder.prepare_inputs(
                    micro_observations,
                    micro_history_input,
                    valid_input_ids,
                )

            with _record_timing_ms(timings, "image_encode_ms"):
                payloads = self._build_chat_payloads(prepared_inputs)

            with _record_timing_ms(timings, "http_request_ms"):
                outputs, token_counts = self._generate(payloads)
            generated_token_counts.extend(token_counts)
            all_outputs.extend(outputs)

            if len(outputs) != len(valid_input_ids):
                logger.warning(
                    "SGLang reward output count mismatch: outputs=%d valid_inputs=%d",
                    len(outputs),
                    len(valid_input_ids),
                )
                outputs += [""] * (len(valid_input_ids) - len(outputs))
                outputs = outputs[: len(valid_input_ids)]

            with _record_timing_ms(timings, "parse_ms"):
                parsed_rewards = self.reward_parser.parse_rewards(outputs).to(
                    dtype=torch.float32
                )

            reward_chunk[valid_input_ids] = parsed_rewards
            reward_chunks.append(reward_chunk)

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_outputs = all_outputs
        self._set_timing_attrs(
            timings,
            self._summarize_generation_stats(generated_token_counts),
        )
        rewards = torch.cat(reward_chunks, dim=0)
        return self.apply_gt_success_bonus(rewards, observations)
