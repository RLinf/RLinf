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

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models.embodiment.reward.vlm_reward_utils.common import (
    apply_gt_success_bonus,
    load_vlm_processor,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.utils.logging import get_logger
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

logger = get_logger()

_DEFAULT_REQUEST_TIMEOUT = 120.0
_JPEG_QUALITY = 95


class EmbodiedAPIRewardWorker(EmbodiedRewardWorker):
    """Embodied reward worker that calls an OpenAI-compatible reward API."""

    def init_worker(self):
        """Initialize API reward inference without loading a local reward model."""
        if self._standalone_realworld:
            raise ValueError(
                "standalone_realworld reward workers do not support API reward "
                "inference."
            )

        assert self.train_batch_size % self._world_size == 0, (
            f"train_batch_size ({self.train_batch_size}) must be divisible by "
            f"world_size ({self._world_size})."
        )
        self.local_num_train_envs = self.train_batch_size // self._world_size

        self.model_cfg = self.cfg.reward.model
        self.api_cfg = self.cfg.reward.get("api", {})
        if self.model_cfg.get("model_type") != "history_vlm":
            raise ValueError(
                "EmbodiedAPIRewardWorker currently supports only "
                "reward.model.model_type='history_vlm'."
            )

        self.model_path = self.model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError(
                "reward.model.model_path must be set for API reward inference."
            )

        self.api_base = str(self.api_cfg.get("api_base") or "").rstrip("/")
        if not self.api_base:
            raise ValueError(
                "reward.api.api_base must be set for API reward inference. "
                "When using Ray-managed SGLang serving, the entrypoint injects "
                "reward.api.api_base at runtime."
            )
        client_base_url = (
            self.api_base[:-3] if self.api_base.endswith("/v1") else self.api_base
        )
        self.http_client = InferenceHTTPClient(
            client_base_url,
            connect_timeout=_DEFAULT_REQUEST_TIMEOUT,
        )

        self.history_buffer_names = list(self.model_cfg.history_buffers.keys())
        self.interval_reward = float(self.model_cfg.get("interval_reward", 0.0))
        self.gt_success_bonus = float(self.model_cfg.get("gt_success_bonus", 0.0))
        self.model_name = str(
            self.api_cfg.get("model")
            or str(self.model_path).rstrip("/").split("/")[-1]
            or "history_vlm_reward"
        )
        self.sampling_params = dict(
            OmegaConf.to_container(
                self.api_cfg.get("sampling_params", {}) or {}, resolve=True
            )
        )

        self._processor = load_vlm_processor(
            self.model_path, self.model_cfg.get("subprocessor_kwargs", {})
        )
        self.input_builder = get_input_builder(
            self.model_cfg.get("input_builder_name", "history_vlm_input_builder")
        )(
            **self.model_cfg.get("input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        assert isinstance(self.input_builder, HistoryVLMInputBuilder), (
            "EmbodiedAPIRewardWorker only supports HistoryVLMInputBuilder."
        )
        self.reward_parser = get_reward_parser(
            self.model_cfg.get("reward_parser_name", "base_reward_parser")
        )(**self.model_cfg.get("reward_parser_params", {}))

    def _frame_to_data_url(self, frame: Any) -> str:
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
        frame = np.ascontiguousarray(frame[..., :3])

        image_buffer = io.BytesIO()
        Image.fromarray(frame, mode="RGB").save(
            image_buffer, format="JPEG", quality=_JPEG_QUALITY
        )
        encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_chat_payloads(
        self,
        prepared_inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompt_texts_list = prepared_inputs.get("prompt_texts_list") or []
        videos_list = prepared_inputs.get("videos_list") or []

        payloads: list[dict[str, Any]] = []
        for prompt_texts, videos in zip(prompt_texts_list, videos_list):
            content: list[dict[str, Any]] = []
            for video in videos:
                for frame in video:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": self._frame_to_data_url(frame)},
                        }
                    )
            content.append(
                {"type": "text", "text": prompt_texts[0] if prompt_texts else ""}
            )
            payloads.append(
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": content}],
                    **self.sampling_params,
                }
            )
        return payloads

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content", "")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content)
        return str(content)

    def _generate(self, payloads: list[dict[str, Any]]) -> list[str]:
        if not payloads:
            return []

        def _generate_one(payload: dict[str, Any]) -> str:
            payload = dict(payload)
            messages = payload.pop("messages")
            model = payload.pop("model")
            return self._extract_text(
                self.http_client.chat_completion(
                    messages=messages, model=model, **payload
                )
            )

        with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
            return list(executor.map(_generate_one, payloads))

    @torch.no_grad()
    def compute_reward(
        self,
        observations: dict[str, Any],
    ) -> torch.Tensor:
        history_input: dict[str, dict[str, list[list[Any]]]] = observations[
            "history_input"
        ]
        input_batch_size = len(next(iter(next(iter(history_input.values())).values())))
        reward_observations = {
            key: value for key, value in observations.items() if key != "history_input"
        }

        rewards = torch.full(
            (input_batch_size,),
            fill_value=self.interval_reward,
            dtype=torch.float32,
        )

        valid_input_ids = self.input_builder.get_valid_input_ids(
            reward_observations,
            history_input,
        )
        if len(valid_input_ids) > 0:
            prepared_inputs = self.input_builder.prepare_inputs(
                reward_observations,
                history_input,
                valid_input_ids,
            )
            outputs = self._generate(self._build_chat_payloads(prepared_inputs))
            if len(outputs) != len(valid_input_ids):
                logger.warning(
                    "API reward output count mismatch: outputs=%d valid_inputs=%d",
                    len(outputs),
                    len(valid_input_ids),
                )
                outputs = (outputs + [""] * len(valid_input_ids))[
                    : len(valid_input_ids)
                ]

            rewards[valid_input_ids] = self.reward_parser.parse_rewards(outputs).to(
                dtype=torch.float32
            )

        return self._format_reward_output(
            apply_gt_success_bonus(rewards, reward_observations, self.gt_success_bonus)
        )
