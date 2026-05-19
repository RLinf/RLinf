#!/usr/bin/env python3
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
import time
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)

logger = logging.getLogger(__name__)


def _resolve_reward_checkpoint_file(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Reward checkpoint path does not exist: {checkpoint_path}"
        )

    candidates = (
        os.path.join(checkpoint_path, "actor", "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint_path, "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint_path, "full_weights.pt"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Unable to resolve reward checkpoint file from path "
        f"{checkpoint_path}. Expected a file path or one of: {candidates}"
    )


def _load_reward_checkpoint_state_dict(
    checkpoint_path: str,
) -> tuple[str, dict[str, torch.Tensor]]:
    resolved_checkpoint_path = _resolve_reward_checkpoint_file(checkpoint_path)
    checkpoint_state_dict = torch.load(
        resolved_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint_state_dict, dict):
        raise ValueError(
            "Reward checkpoint must contain a state_dict dictionary, got "
            f"{type(checkpoint_state_dict).__name__}"
        )

    normalized_state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint_state_dict.items()
        if isinstance(key, str)
    }
    if not normalized_state_dict:
        raise ValueError(
            f"Reward checkpoint {resolved_checkpoint_path} does not contain any "
            "string-keyed model weights."
        )

    return resolved_checkpoint_path, normalized_state_dict


def _extract_lora_state_dict(
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in checkpoint_state_dict.items() if "lora_" in key
    }


def _build_lora_config(lora_state_dict: dict[str, torch.Tensor]) -> LoraConfig:
    lora_a_entries = [
        (key, value) for key, value in lora_state_dict.items() if "lora_A" in key
    ]
    if not lora_a_entries:
        raise ValueError(
            "Detected LoRA-style reward checkpoint, but no lora_A weights were found."
        )

    lora_rank = int(lora_a_entries[0][1].shape[0])
    target_modules = sorted(
        {
            key.split(".lora_")[0].split(".")[-1]
            for key in lora_state_dict
            if ".lora_" in key
        }
    )
    if not target_modules:
        raise ValueError(
            "Detected LoRA-style reward checkpoint, but could not infer target "
            "modules from lora_* keys."
        )

    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )


def _unpack_load_state_dict_result(
    load_result: Any,
) -> tuple[list[str], list[str]]:
    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        return list(load_result.missing_keys), list(load_result.unexpected_keys)
    if isinstance(load_result, tuple) and len(load_result) == 2:
        missing_keys, unexpected_keys = load_result
        return list(missing_keys), list(unexpected_keys)
    return [], []


def load_reward_checkpoint_into_model(
    model: Any,
    checkpoint_path: str,
) -> tuple[Any, dict[str, str]]:
    resolved_checkpoint_path, checkpoint_state_dict = (
        _load_reward_checkpoint_state_dict(checkpoint_path)
    )
    lora_state_dict = _extract_lora_state_dict(checkpoint_state_dict)
    if lora_state_dict:
        lora_config = _build_lora_config(lora_state_dict)
        model = get_peft_model(model, lora_config)
        load_result = model.load_state_dict(lora_state_dict, strict=False)
        _, unexpected_keys = _unpack_load_state_dict_result(load_result)
        unexpected_lora_keys = [key for key in unexpected_keys if "lora_" in key]
        if unexpected_lora_keys:
            preview = ", ".join(unexpected_lora_keys[:5])
            raise ValueError(
                "Detected LoRA checkpoint, but some LoRA weights were not loaded: "
                f"{preview}"
            )
        return model, {
            "checkpoint_path": resolved_checkpoint_path,
            "checkpoint_format": "lora",
            "loaded_lora_keys": str(len(lora_state_dict)),
        }

    load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
    missing_keys, unexpected_keys = _unpack_load_state_dict_result(load_result)
    if missing_keys or unexpected_keys:
        missing_preview = ", ".join(missing_keys[:5]) or "none"
        unexpected_preview = ", ".join(unexpected_keys[:5]) or "none"
        raise ValueError(
            "Reward checkpoint was recognized as merged full weights, but it is "
            "incompatible with the base model. "
            f"missing_keys={missing_preview}; "
            f"unexpected_keys={unexpected_preview}"
        )

    return model, {
        "checkpoint_path": resolved_checkpoint_path,
        "checkpoint_format": "full_weights",
    }


class VLMRewardModel(BaseRewardModel):
    """A frozen VLM reward model that maps (images, task) -> scalar reward.

    This implementation intentionally avoids hardcoding family-specific HF class
    names. It loads by `model_path` via Auto* APIs (consistent with RLinf SFT).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.model_path: str = cfg.get("model_path")
        if not self.model_path:
            raise ValueError("reward.model.model_path must be set for VLMRewardModel")
        self.lora_path = self.cfg.get("lora_path")
        self.gt_success_bonus = float(cfg.get("gt_success_bonus", 0.0))

        self.dtype = torch_dtype_from_precision(cfg.precision)
        self.checkpoint_metadata: Optional[dict[str, str]] = None
        self.last_timing_ms: dict[str, float] = {}
        self.last_generation_stats: dict[str, float] = {}

        self.setup_processor()
        self.setup_model()

        self.setup_input_builder()
        self.setup_reward_parser()

        self.gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 32)),
            "do_sample": bool(cfg.get("do_sample", True)),
            "temperature": float(cfg.get("temperature", 0.0)),
        }
        if cfg.get("min_new_tokens", None) is not None:
            self.gen_kwargs["min_new_tokens"] = int(cfg.get("min_new_tokens"))

    def setup_processor(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._setup_subprocessor(
            subprocessor_kwargs=self.cfg.get("subprocessor_kwargs", {})
        )

    def _setup_subprocessor(
        self,
        subprocessor_kwargs: dict,
    ) -> None:
        for subprocessor_name, subprocessor_kwargs in subprocessor_kwargs.items():
            subprocessor_kwargs = dict(subprocessor_kwargs)

            subprocessoror = getattr(self._processor, subprocessor_name, None)
            if subprocessoror is None:
                continue
            for key, value in dict(subprocessor_kwargs).items():
                if hasattr(subprocessoror, key):
                    setattr(subprocessoror, key, value)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "base_vlm_input_builder")
        )(**self.cfg.get("input_builder_params", {}), _processor=self._processor)

    def setup_reward_parser(self) -> None:
        self.reward_parser = get_reward_parser(
            self.cfg.get("reward_parser_name", "base_reward_parser")
        )(**self.cfg.get("reward_parser_params", {}))

    def _record_generated_token_stats(
        self, output_ids: torch.Tensor, prompt_length: int
    ) -> None:
        generated_ids = output_ids[..., prompt_length:]
        if generated_ids.numel() == 0:
            self.last_generation_stats = {
                "generated_tokens_mean": 0.0,
                "generated_tokens_min": 0.0,
                "generated_tokens_max": 0.0,
            }
            return

        tokenizer = getattr(self._processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            token_counts = torch.full(
                (generated_ids.shape[0],),
                generated_ids.shape[-1],
                device=generated_ids.device,
                dtype=torch.float32,
            )
        else:
            token_counts = (generated_ids != pad_token_id).sum(dim=-1).float()

        self.last_generation_stats = {
            "generated_tokens_mean": float(token_counts.mean().item()),
            "generated_tokens_min": float(token_counts.min().item()),
            "generated_tokens_max": float(token_counts.max().item()),
        }

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

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "VLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
        )

    def setup_model(self) -> None:
        _ = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

        if self.lora_path:
            self._model, self.checkpoint_metadata = load_reward_checkpoint_into_model(
                self._model, self.lora_path
            )

        self._model.eval()

    @torch.no_grad()
    def compute_reward(
        self,
        observations: Any,
    ) -> torch.Tensor:
        total_start = time.perf_counter()
        timings = {
            "prepare_inputs_ms": 0.0,
            "generate_ms": 0.0,
            "decode_ms": 0.0,
            "parse_ms": 0.0,
            "total_ms": 0.0,
        }
        prepare_start = time.perf_counter()
        batched_inputs = self.input_builder.build_inputs(
            observations, self._model.device
        )
        timings["prepare_inputs_ms"] += (time.perf_counter() - prepare_start) * 1000
        prompt_length = batched_inputs["input_ids"].shape[-1]
        generate_start = time.perf_counter()
        output_ids = self._model.generate(**batched_inputs, **self.gen_kwargs)
        timings["generate_ms"] += (time.perf_counter() - generate_start) * 1000
        del batched_inputs
        self._record_generated_token_stats(output_ids, prompt_length)
        decode_start = time.perf_counter()
        outputs = self._processor.batch_decode(
            output_ids[..., prompt_length:], skip_special_tokens=True
        )
        timings["decode_ms"] += (time.perf_counter() - decode_start) * 1000
        del output_ids
        parse_start = time.perf_counter()
        rewards = self.reward_parser.parse_rewards(outputs)
        timings["parse_ms"] += (time.perf_counter() - parse_start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_timing_ms = timings
        return self.apply_gt_success_bonus(rewards, observations)


class HistoryVLMRewardModel(VLMRewardModel):
    def __init__(self, cfg: DictConfig):
        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(cfg.get("infer_micro_batch_size", 0))
        self.interval_reward: float = float(cfg.get("interval_reward", 0.0))
        self.debug_dump_first_inference: bool = bool(
            cfg.get("debug_dump_first_inference", False)
        )
        self.debug_dump_first_inference_dir: str = str(
            cfg.get("debug_dump_first_inference_dir", "")
        )
        self._debug_dump_first_inference_done = False

        super().__init__(cfg)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "history_vlm_input_builder")
        )(
            **self.cfg.get("input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        assert isinstance(self.input_builder, HistoryVLMInputBuilder), (
            "HistoryVLMRewardModel only supports HistoryVLMInputBuilder"
        )

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HistoryVLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
        )

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
            key: self._slice_batch_value(value, start, end)
            for key, value in observations.items()
        }

    def _slice_batch_value(self, value: Any, start: int, end: int) -> Any:
        if isinstance(value, dict):
            return {
                key: self._slice_batch_value(item, start, end)
                for key, item in value.items()
            }
        if isinstance(value, (torch.Tensor, list, tuple)):
            return value[start:end]
        return value

    def _dump_first_inference_inputs(
        self,
        prepared_inputs: dict[str, Any],
        valid_input_ids: list[int],
        outputs: list[str],
        start: int,
    ) -> None:
        if not self.debug_dump_first_inference or not valid_input_ids:
            return

        dump_dir = self.debug_dump_first_inference_dir or os.path.join(
            os.getcwd(), "reward_first_inference_inputs"
        )
        worker_dir = os.path.join(dump_dir, f"pid_{os.getpid()}")
        os.makedirs(worker_dir, exist_ok=True)

        try:
            import imageio.v2 as imageio
            import numpy as np
        except Exception as exc:
            logger.warning("Failed to import imageio/numpy for reward dump: %s", exc)
            return

        videos_list = prepared_inputs.get("videos_list") or []
        prompt_texts_list = prepared_inputs.get("prompt_texts_list") or []

        def _frame_to_numpy(frame: Any) -> Any:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            elif hasattr(frame, "__array__"):
                frame = np.asarray(frame)
            else:
                frame = np.asarray(frame)

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame[..., :3]

        for local_idx, env_id in enumerate(valid_input_ids):
            global_env_id = start + env_id
            env_dir = os.path.join(worker_dir, f"env_{global_env_id:05d}")
            os.makedirs(env_dir, exist_ok=True)

            prompt = prompt_texts_list[local_idx][0]
            output = outputs[local_idx] if local_idx < len(outputs) else ""
            prompt_path = os.path.join(env_dir, "prompt_and_output.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("PROMPT:\n")
                f.write(prompt)
                f.write("\n\nOUTPUT:\n")
                f.write(output)
                f.write("\n")

            logger.warning(
                "[reward_first_inference_dump] env=%s prompt_path=%s prompt=%s output=%s",
                global_env_id,
                prompt_path,
                prompt,
                output,
            )

            for video_idx, video in enumerate(videos_list[local_idx]):
                video_path = os.path.join(env_dir, f"video_{video_idx:02d}.mp4")
                frames = [_frame_to_numpy(frame) for frame in video]
                if not frames:
                    logger.warning(
                        "[reward_first_inference_dump] skip empty video: env=%s video=%s",
                        global_env_id,
                        video_idx,
                    )
                    continue
                imageio.mimsave(video_path, frames, fps=8)
                logger.warning(
                    "[reward_first_inference_dump] env=%s video=%s frames=%s path=%s",
                    global_env_id,
                    video_idx,
                    len(frames),
                    video_path,
                )

    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        total_start = time.perf_counter()
        timings = {
            "get_valid_inputs_ms": 0.0,
            "prepare_inputs_ms": 0.0,
            "process_inputs_ms": 0.0,
            "h2d_ms": 0.0,
            "generate_ms": 0.0,
            "decode_ms": 0.0,
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
        did_infer = False
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            micro_observations = self.slice_observations(observations, start, end)
            micro_history_input = self.slice_history_input(history_input, start, end)
            reward_chunk = torch.full(
                (end - start,), fill_value=self.interval_reward, dtype=torch.float32
            )

            valid_start = time.perf_counter()
            valid_input_ids = self.input_builder.get_valid_input_ids(
                micro_observations,
                micro_history_input,
            )
            timings["get_valid_inputs_ms"] += (time.perf_counter() - valid_start) * 1000
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
            process_start = time.perf_counter()
            batched_inputs = self.input_builder.process_inputs(prepared_inputs)
            timings["process_inputs_ms"] += (time.perf_counter() - process_start) * 1000
            h2d_start = time.perf_counter()
            batched_inputs = {
                key: value.to(self._model.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batched_inputs.items()
            }
            timings["h2d_ms"] += (time.perf_counter() - h2d_start) * 1000

            prompt_length = batched_inputs["input_ids"].shape[-1]
            generate_start = time.perf_counter()
            output_ids = self._model.generate(**batched_inputs, **self.gen_kwargs)
            timings["generate_ms"] += (time.perf_counter() - generate_start) * 1000
            del batched_inputs
            self._record_generated_token_stats(output_ids, prompt_length)

            decode_start = time.perf_counter()
            outputs = self._processor.batch_decode(
                output_ids[..., prompt_length:], skip_special_tokens=True
            )
            timings["decode_ms"] += (time.perf_counter() - decode_start) * 1000
            del output_ids
            did_infer = True

            if not self._debug_dump_first_inference_done:
                self._dump_first_inference_inputs(
                    prepared_inputs=prepared_inputs,
                    valid_input_ids=valid_input_ids,
                    outputs=outputs,
                    start=start,
                )

            parse_start = time.perf_counter()
            reward_chunk[valid_input_ids] = self.reward_parser.parse_rewards(
                outputs
            ).to(dtype=torch.float32)
            timings["parse_ms"] += (time.perf_counter() - parse_start) * 1000
            reward_chunks.append(reward_chunk)
            del outputs

        if did_infer:
            self._debug_dump_first_inference_done = True
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_timing_ms = timings
        return self.apply_gt_success_bonus(
            torch.cat(reward_chunks, dim=0), observations
        )
