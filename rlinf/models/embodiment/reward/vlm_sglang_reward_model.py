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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
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


class SGLangRewardTimingRecorder:
    """Records timing metrics for one SGLang reward inference call."""

    _BASE_KEYS = (
        "prepare_inputs_ms",
        "image_encode_ms",
        "http_request_ms",
        "parse_ms",
        "total_ms",
    )

    def __init__(self) -> None:
        self.timings = dict.fromkeys(self._BASE_KEYS, 0.0)
        self._total_start = 0.0

    def __enter__(self):
        self._total_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.timings["total_ms"] = (time.perf_counter() - self._total_start) * 1000

    @contextmanager
    def record(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] += (time.perf_counter() - start) * 1000

    def metrics(self) -> dict[str, float]:
        return {
            **self.timings,
            "media_convert_ms": self.timings["image_encode_ms"],
            "sglang_generate_ms": self.timings["http_request_ms"],
            "generate_ms": self.timings["http_request_ms"],
        }


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


def _normalize_sglang_server_args(value: Any) -> dict[str, Any]:
    return {
        str(key).replace("_", "-"): val for key, val in _to_plain_dict(value).items()
    }


class HistoryVLMSGLangRewardModel(BaseRewardModel):
    """SGLang HTTP-backed embodied history VLM reward model.

    The embodied entrypoint starts a Ray-managed SGLang OpenAI-compatible
    server/router stack for this backend unless ``sglang_server_args.api_base``
    points to a user-managed server. The reward worker submits one
    ``/v1/chat/completions`` request per valid history-window input in parallel
    and leaves request scheduling and concurrency limits to the SGLang server.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.model_path: str = cfg.get("model_path")
        if not self.model_path:
            raise ValueError(
                f"reward.model.model_path must be set for {self.__class__.__name__}"
            )

        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.gt_success_bonus = float(cfg.get("gt_success_bonus", 0.0))
        sglang_server_args = _normalize_sglang_server_args(
            cfg.get("sglang_server_args", {})
        )
        self.api_base = str(sglang_server_args.get("api-base") or "").rstrip("/")
        if not self.api_base:
            host = str(sglang_server_args.get("host", "127.0.0.1"))
            port = int(sglang_server_args.get("port", 30000))
            self.api_base = f"http://{host}:{port}/v1"
        self.model_name = str(
            sglang_server_args.get("served-model-name")
            or Path(str(self.model_path)).name
            or self.model_path
        )
        self.request_timeout = float(cfg.get("request_timeout", 120.0))
        self.image_format = str(cfg.get("image_format", "jpeg")).lower()
        if self.image_format not in {"jpeg", "png"}:
            raise ValueError(
                f"{self.__class__.__name__} supports image_format='jpeg' or 'png', "
                f"got {self.image_format!r}."
            )
        self.jpeg_quality = int(cfg.get("jpeg_quality", 95))

        self.sampling_params = self._build_sampling_params(cfg)

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
            f"{self.__class__.__name__} is an inference-time reward model; "
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
        if not payloads:
            return [], []

        def _generate_one(payload: dict[str, Any]) -> tuple[str, int]:
            text, token_count = self._extract_text_and_token_count(
                self._chat_completion(payload)
            )
            return text, token_count

        with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
            results = list(executor.map(_generate_one, payloads))

        outputs = [text for text, _ in results]
        completion_tokens = [token_count for _, token_count in results]
        return outputs, completion_tokens

    def _set_timing_metrics(
        self,
        timing_recorder: SGLangRewardTimingRecorder,
        generation_stats: dict[str, float],
    ) -> None:
        timings = timing_recorder.metrics()
        self.last_timing_ms = timings
        self.last_generation_stats = dict(generation_stats)
        logger.debug("%s timing_ms=%s", self.__class__.__name__, self.last_timing_ms)

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
        with SGLangRewardTimingRecorder() as timing_recorder:
            history_input: dict[str, dict[str, list[list[Any]]]] = reward_input.pop(
                "history_input"
            )
            input_batch_size = len(
                next(iter(next(iter(history_input.values())).values()))
            )
            observations = reward_input

            all_outputs: list[str] = []
            generated_token_counts: list[int] = []
            rewards = torch.zeros((input_batch_size,), dtype=torch.float32)

            valid_input_ids = self.input_builder.get_valid_input_ids(
                observations,
                history_input,
            )
            if len(valid_input_ids) > 0:
                with timing_recorder.record("prepare_inputs_ms"):
                    prepared_inputs = self.input_builder.prepare_inputs(
                        observations,
                        history_input,
                        valid_input_ids,
                    )

                with timing_recorder.record("image_encode_ms"):
                    payloads = self._build_chat_payloads(prepared_inputs)

                with timing_recorder.record("http_request_ms"):
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

                with timing_recorder.record("parse_ms"):
                    parsed_rewards = self.reward_parser.parse_rewards(outputs).to(
                        dtype=torch.float32
                    )

                rewards[valid_input_ids] = parsed_rewards

        self.last_outputs = all_outputs
        self._set_timing_metrics(
            timing_recorder,
            self._summarize_generation_stats(generated_token_counts),
        )
        return self.apply_gt_success_bonus(rewards, observations)
