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

import asyncio
import os
import signal
import sys
import threading
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.models.embodiment.reward import vlm_reward_model as vlm_reward_model_module
from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    _to_pil_images,
)
from rlinf.models.embodiment.reward.vlm_sglang_reward_model import (
    _RAW_FRAME_VIDEO_FORMAT,
    HistoryVLMSGLangRewardModel,
    _install_sglang_raw_frame_video_patch,
)
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor, split_dict


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.loaded_state_dict = None
        self.strict = None

    def eval(self):
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.strict = strict
        return [], []

    def generate(
        self, input_ids: torch.Tensor, reward_ids: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del kwargs
        return torch.cat([input_ids, reward_ids.unsqueeze(-1)], dim=-1)


class _FakeProcessor:
    def batch_decode(self, output_ids: torch.Tensor, skip_special_tokens: bool = True):
        del skip_special_tokens
        return [str(int(token.item())) for token in output_ids[:, 0]]


class _FakeSGLangProcessor:
    video_token = "<video>"

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        self.messages.append(messages)
        return "rendered"


class _FakeRewardParser:
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        return torch.tensor([float(output) for output in outputs], dtype=torch.float32)


class _FakeHistoryInputBuilder(HistoryVLMInputBuilder):
    def __init__(self, history_buffer_names: list[str]):
        super().__init__(
            _processor=None,
            history_buffer_names=history_buffer_names,
        )
        self.calls: list[list[int]] = []

    def get_valid_input_ids(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
    ) -> list[int]:
        del observations
        history_window = history_input["history_window"]["main_images"]
        return [
            env_idx for env_idx, frames in enumerate(history_window) if len(frames) > 0
        ]

    def prepare_inputs(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
        valid_input_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        del history_input
        reward_ids = observations["slot_ids"][valid_input_ids].to(dtype=torch.long)
        self.calls.append(reward_ids.tolist())
        return {
            "input_ids": torch.zeros((len(valid_input_ids), 1), dtype=torch.long),
            "reward_ids": reward_ids,
        }

    def process_inputs(
        self, prepared_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return prepared_inputs


class _FakeSGLangHistoryInputBuilder:
    def __init__(self):
        self.calls: list[list[int]] = []

    def get_valid_input_ids(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
    ) -> list[int]:
        del observations
        history_window = history_input["history_window"]["main_images"]
        return [
            env_idx for env_idx, frames in enumerate(history_window) if len(frames) > 0
        ]

    def prepare_inputs(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
        valid_input_ids: list[int],
    ) -> dict[str, object]:
        del history_input
        slot_ids = observations["slot_ids"][valid_input_ids].tolist()
        self.calls.append(slot_ids)
        return {
            "prompt_texts_list": [[f"prompt-{slot_id}"] for slot_id in slot_ids],
            "videos_list": [
                [[torch.zeros((1, 1, 3), dtype=torch.uint8)]] for _ in slot_ids
            ],
        }


class _RecordingRewardParser:
    def __init__(self):
        self.calls: list[list[str]] = []

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        self.calls.append(list(outputs))
        values = [float(output) if output else 0.0 for output in outputs]
        return torch.tensor(values, dtype=torch.float32)


class _TestHistoryVLMRewardModel(HistoryVLMRewardModel):
    def setup_processor(self) -> None:
        self._processor = _FakeProcessor()

    def setup_model(self) -> None:
        self._model = _FakeModel()

    def setup_input_builder(self) -> None:
        self.input_builder = _FakeHistoryInputBuilder(
            history_buffer_names=self.history_buffer_names
        )

    def setup_reward_parser(self) -> None:
        self.reward_parser = _FakeRewardParser()


class _TestHistoryVLMSGLangRewardModel(HistoryVLMSGLangRewardModel):
    def setup_processor(self) -> None:
        self._processor = _FakeSGLangProcessor()

    def setup_input_builder(self) -> None:
        self.input_builder = _FakeSGLangHistoryInputBuilder()

    def setup_reward_parser(self) -> None:
        self.reward_parser = _RecordingRewardParser()

    def _generate(
        self, prompts: list[str], video_data: list[list[object]]
    ) -> tuple[list[str], list[int]]:
        del video_data
        return [prompt.removeprefix("rendered-") for prompt in prompts], [
            len(prompts)
        ] * len(prompts)


def _make_cfg(infer_micro_batch_size: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": "dummy",
            "precision": "bf16",
            "infer_micro_batch_size": infer_micro_batch_size,
            "input_builder_name": "history_vlm_input_builder",
            "reward_parser_name": "base_reward_parser",
            "history_buffers": {
                "history_window": {
                    "history_size": 10,
                    "input_interval": 10,
                    "history_keys": ["main_images"],
                    "input_on_done": False,
                }
            },
        }
    )


def _make_sglang_cfg(infer_micro_batch_size: int) -> OmegaConf:
    cfg = _make_cfg(infer_micro_batch_size)
    cfg.model_path = "dummy"
    return cfg


def _make_reward_input(
    slot_ids: list[int], valid_env_ids: list[int] | None = None
) -> dict[str, object]:
    valid_env_ids = valid_env_ids or list(range(len(slot_ids)))
    valid_env_id_set = set(valid_env_ids)

    return {
        "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "main_images": torch.zeros((len(slot_ids), 1, 1, 1), dtype=torch.uint8),
        "task_descriptions": [f"task-{slot_id}" for slot_id in slot_ids],
        "history_input": {
            "history_window": {
                "main_images": [
                    [f"frame-{slot_id}"] if env_idx in valid_env_id_set else []
                    for env_idx, slot_id in enumerate(slot_ids)
                ]
            }
        },
    }


def _make_sglang_reward_input(
    slot_ids: list[int], valid_env_ids: list[int] | None = None
) -> dict[str, object]:
    valid_env_ids = valid_env_ids or list(range(len(slot_ids)))
    valid_env_id_set = set(valid_env_ids)
    return {
        "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "main_images": [f"obs-{slot_id}" for slot_id in slot_ids],
        "history_input": {
            "history_window": {
                "main_images": [
                    [f"frame-{slot_id}"] if env_idx in valid_env_id_set else []
                    for env_idx, slot_id in enumerate(slot_ids)
                ]
            }
        },
    }


def _make_rank_roundtrip_input(batch_size: int) -> dict[str, object]:
    return {
        "slot_ids": torch.arange(batch_size, dtype=torch.long),
        "main_images": torch.arange(batch_size, dtype=torch.uint8).view(
            batch_size, 1, 1, 1
        ),
        "task_descriptions": [f"task-{idx}" for idx in range(batch_size)],
        "history_input": {
            "history_window": {
                "main_images": [[f"frame-{idx}"] for idx in range(batch_size)]
            }
        },
    }


def _simulate_reward_roundtrip(
    batch_size: int, env_world_size: int, reward_world_size: int
) -> torch.Tensor:
    env_batch_size = batch_size // env_world_size
    reward_input = _make_rank_roundtrip_input(batch_size)
    env_local_batches = split_dict(
        reward_input, [env_batch_size for _ in range(env_world_size)]
    )

    incoming_reward_shards: dict[int, dict[int, dict[str, object]]] = {
        reward_rank: {} for reward_rank in range(reward_world_size)
    }
    for env_rank, env_batch in enumerate(env_local_batches):
        dst_ranks_and_sizes = CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=reward_world_size,
            src_rank=env_rank,
        )
        reward_input_shards = split_dict(
            env_batch, [size for _, size in dst_ranks_and_sizes]
        )
        for (reward_rank, _), shard in zip(dst_ranks_and_sizes, reward_input_shards):
            incoming_reward_shards[reward_rank][env_rank] = shard

    reward_outputs: dict[int, torch.Tensor] = {}
    for reward_rank in range(reward_world_size):
        src_ranks_and_sizes = CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=reward_world_size,
            dst_rank=reward_rank,
        )
        ordered_shards = [
            incoming_reward_shards[reward_rank][env_rank]
            for env_rank, _ in src_ranks_and_sizes
        ]
        merged = cat_list_of_dict_tensor(ordered_shards, dim=0)
        expected_slot_ids = [int(slot_id) for slot_id in merged["slot_ids"].tolist()]

        assert merged["task_descriptions"] == [
            f"task-{slot_id}" for slot_id in expected_slot_ids
        ]
        assert merged["history_input"]["history_window"]["main_images"] == [
            [f"frame-{slot_id}"] for slot_id in expected_slot_ids
        ]

        reward_outputs[reward_rank] = merged["slot_ids"].to(dtype=torch.float32)

    incoming_env_rewards: dict[int, dict[int, torch.Tensor]] = {
        env_rank: {} for env_rank in range(env_world_size)
    }
    for reward_rank, reward_tensor in reward_outputs.items():
        dst_ranks_and_sizes = CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=reward_world_size,
            dst_world_size=env_world_size,
            src_rank=reward_rank,
        )
        reward_shards = torch.split(
            reward_tensor, [size for _, size in dst_ranks_and_sizes], dim=0
        )
        for (env_rank, _), shard in zip(dst_ranks_and_sizes, reward_shards):
            incoming_env_rewards[env_rank][reward_rank] = shard

    env_results: list[torch.Tensor] = []
    for env_rank in range(env_world_size):
        src_ranks_and_sizes = CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=reward_world_size,
            dst_world_size=env_world_size,
            dst_rank=env_rank,
        )
        env_results.append(
            torch.cat(
                [
                    incoming_env_rewards[env_rank][reward_rank]
                    for reward_rank, _ in src_ranks_and_sizes
                ],
                dim=0,
            )
        )

    return torch.cat(env_results, dim=0)


def test_history_vlm_reward_model_keeps_micro_batch_order():
    model = _TestHistoryVLMRewardModel(_make_cfg(infer_micro_batch_size=2))

    rewards = model.compute_reward(_make_reward_input([10, 11, 12, 13]))

    assert torch.equal(rewards, torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert model.input_builder.calls == [[10, 11], [12, 13]]


def test_history_vlm_reward_model_writes_sparse_valid_envs_back_to_slots():
    model = _TestHistoryVLMRewardModel(_make_cfg(infer_micro_batch_size=2))

    rewards = model.compute_reward(
        _make_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert model.input_builder.calls == [[21], [23]]


def test_history_vlm_sglang_reward_model_is_registered():
    assert (
        get_reward_model_class("history_vlm_sglang").__name__
        == "HistoryVLMSGLangRewardModel"
    )


def test_history_vlm_sglang_sampling_params_from_config():
    model = object.__new__(HistoryVLMSGLangRewardModel)
    sampling_params = model._build_sampling_params(
        OmegaConf.create(
            {
                "max_new_tokens": 16,
                "min_new_tokens": 8,
                "ignore_eos": True,
                "temperature": 0.0,
                "top_p": 0.8,
            }
        )
    )

    assert sampling_params == {
        "max_new_tokens": 16,
        "min_new_tokens": 8,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 0.8,
    }


def test_history_vlm_sglang_reward_model_keeps_micro_batch_order():
    model = _TestHistoryVLMSGLangRewardModel(_make_sglang_cfg(2))

    def build_inputs(prepared_inputs):
        prompts = [
            f"rendered-{prompt_texts[0].removeprefix('prompt-')}"
            for prompt_texts in prepared_inputs["prompt_texts_list"]
        ]
        return prompts, [[] for _ in prompts]

    model._build_sglang_inputs = build_inputs

    rewards = model.compute_reward(_make_sglang_reward_input([10, 11, 12, 13]))

    assert torch.equal(rewards, torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert model.input_builder.calls == [[10, 11], [12, 13]]
    assert model.last_outputs == ["10", "11", "12", "13"]


def test_history_vlm_sglang_reward_model_writes_sparse_valid_envs_back_to_slots():
    model = _TestHistoryVLMSGLangRewardModel(_make_sglang_cfg(2))

    def build_inputs(prepared_inputs):
        prompts = [
            f"rendered-{prompt_texts[0].removeprefix('prompt-')}"
            for prompt_texts in prepared_inputs["prompt_texts_list"]
        ]
        return prompts, [[] for _ in prompts]

    model._build_sglang_inputs = build_inputs

    rewards = model.compute_reward(
        _make_sglang_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert model.input_builder.calls == [[21], [23]]


def test_history_vlm_sglang_builds_video_data_as_raw_frames():
    model = object.__new__(HistoryVLMSGLangRewardModel)
    model._processor = _FakeSGLangProcessor()
    model.video_fps = 8.0
    chw_frame = torch.zeros((3, 2, 2), dtype=torch.uint8)
    gray_frame = np.ones((2, 2), dtype=np.float32) * 127.0
    pil_frame = Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8))

    prompts, video_data = model._build_sglang_inputs(
        {
            "prompt_texts_list": [["judge progress"]],
            "videos_list": [[[chw_frame, gray_frame, pil_frame]]],
        }
    )

    assert prompts == ["rendered"]
    assert len(video_data) == 1
    assert len(video_data[0]) == 1
    raw_video = video_data[0][0]
    assert raw_video["format"] == _RAW_FRAME_VIDEO_FORMAT
    assert raw_video.get_avg_fps() == 8.0
    assert len(raw_video) == 3

    batch = raw_video.get_batch([2, 0]).asnumpy()
    assert batch.shape == (2, 2, 2, 3)
    assert batch.dtype == np.uint8
    assert batch.flags.c_contiguous
    assert np.all(batch[0] == 255)
    assert np.all(batch[1] == 0)
    assert model._processor.messages[0][0]["content"].startswith("<video>\n")


def test_history_vlm_sglang_duplicates_single_raw_frame_for_qwen_sampling():
    model = object.__new__(HistoryVLMSGLangRewardModel)
    model.video_fps = 4.0

    raw_video = model._convert_video([np.full((2, 2, 3), 9, dtype=np.uint8)])

    assert len(raw_video) == 2
    assert raw_video.get_batch([0, 1]).asnumpy().shape == (2, 2, 2, 3)
    assert np.all(raw_video.get_batch([0, 1]).asnumpy() == 9)


def test_history_vlm_sglang_raw_frame_patch_wraps_plain_dict(monkeypatch):
    qwen_vl_module = types.ModuleType("qwen_vl")
    seen: dict[str, object] = {}

    async def fake_preprocess_video(video, *args, **kwargs):
        del args, kwargs
        seen["video_type"] = type(video).__name__
        seen["length"] = len(video)
        seen["fps"] = video.get_avg_fps()
        seen["batch"] = video.get_batch([1, 0]).asnumpy()
        return "video", {"fps": video.get_avg_fps()}

    qwen_vl_module.preprocess_video = fake_preprocess_video
    sglang_module = types.ModuleType("sglang")
    srt_module = types.ModuleType("sglang.srt")
    multimodal_module = types.ModuleType("sglang.srt.multimodal")
    processors_module = types.ModuleType("sglang.srt.multimodal.processors")
    sglang_module.__path__ = []
    srt_module.__path__ = []
    multimodal_module.__path__ = []
    processors_module.__path__ = []
    sglang_module.srt = srt_module
    srt_module.multimodal = multimodal_module
    multimodal_module.processors = processors_module
    processors_module.qwen_vl = qwen_vl_module
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.multimodal", multimodal_module)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.multimodal.processors",
        processors_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.multimodal.processors.qwen_vl",
        qwen_vl_module,
    )

    assert _install_sglang_raw_frame_video_patch()
    raw_video = {
        "format": _RAW_FRAME_VIDEO_FORMAT,
        "frames": np.stack(
            [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2, 3), dtype=np.uint8),
            ],
            axis=0,
        ),
        "fps": 7.0,
    }

    result = asyncio.run(qwen_vl_module.preprocess_video(raw_video))

    assert result == ("video", {"fps": 7.0})
    assert seen["video_type"] == "_RawFrameVideoReader"
    assert seen["length"] == 2
    assert seen["fps"] == 7.0
    assert np.all(seen["batch"][0] == 1)
    assert np.all(seen["batch"][1] == 0)


def test_history_vlm_sglang_create_engine_skips_sigquit_outside_main_thread():
    model = object.__new__(HistoryVLMSGLangRewardModel)
    model.sglang_engine_kwargs = {"model_path": "dummy"}
    original_signal = signal.signal
    result: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            signal.signal(signal.SIGQUIT, lambda signum, frame: None)

    def create_engine_in_thread() -> None:
        try:
            result["engine"] = model._create_engine(FakeEngine)
        except Exception as exc:  # pragma: no cover - easier failure reporting
            result["exc"] = exc

    thread = threading.Thread(target=create_engine_in_thread)
    thread.start()
    thread.join()

    assert "exc" not in result
    assert isinstance(result["engine"], FakeEngine)
    assert result["engine"].kwargs == {"model_path": "dummy"}
    assert signal.signal is original_signal


def test_vlm_video_input_flattens_extra_view_frame_dimension():
    frames = [torch.zeros((1, 2, 2, 3), dtype=torch.uint8) for _ in range(5)]

    images = _to_pil_images(frames)

    assert len(images) == 5
    assert all(image.size == (2, 2) for image in images)


def test_history_vlm_sglang_extracts_texts_and_token_counts():
    model = object.__new__(HistoryVLMSGLangRewardModel)

    texts, token_counts = model._extract_texts_and_token_counts(
        [
            {"text": "positive", "meta_info": {"completion_tokens": 3}},
            {"text": "negative", "meta_info": {"completion_tokens": [4]}},
            "neutral",
        ]
    )

    assert texts == ["positive", "negative", "neutral"]
    assert token_counts == [3, 4]


def test_history_vlm_sglang_pads_mismatched_outputs():
    model = _TestHistoryVLMSGLangRewardModel(_make_sglang_cfg(0))
    model._build_sglang_inputs = lambda prepared_inputs: (["prompt"], [[]])
    model._generate = lambda prompts, video_data: (["1.0"], [5])

    rewards = model.compute_reward(_make_sglang_reward_input([30, 31]))

    assert torch.equal(rewards, torch.tensor([1.0, 0.0]))
    assert model.reward_parser.calls == [["1.0", ""]]


def test_history_vlm_sglang_records_timing_and_generation_stats():
    model = _TestHistoryVLMSGLangRewardModel(_make_sglang_cfg(0))
    model._build_sglang_inputs = lambda prepared_inputs: (
        ["rendered-1", "rendered-2"],
        [[], []],
    )
    model._generate = lambda prompts, video_data: (["1", "2"], [3, 5])

    rewards = model.compute_reward(_make_sglang_reward_input([1, 2]))

    assert torch.equal(rewards, torch.tensor([1.0, 2.0]))
    assert set(model.last_timing_ms) == {
        "prepare_inputs_ms",
        "media_convert_ms",
        "sglang_generate_ms",
        "parse_ms",
        "total_ms",
        "generate_ms",
    }
    assert model.last_generation_stats == {
        "generated_tokens_mean": 4.0,
        "generated_tokens_min": 3.0,
        "generated_tokens_max": 5.0,
    }


def test_load_reward_checkpoint_into_model_supports_directory_full_weights(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_file = checkpoint_dir / "actor" / "model_state_dict" / "full_weights.pt"
    checkpoint_file.parent.mkdir(parents=True)
    torch.save(
        {
            "module.linear.weight": torch.ones((2, 2), dtype=torch.float32),
            "linear.bias": torch.zeros((2,), dtype=torch.float32),
        },
        checkpoint_file,
    )

    model = _FakeModel()
    loaded_model, metadata = vlm_reward_model_module.load_reward_checkpoint_into_model(
        model, str(checkpoint_dir)
    )

    assert loaded_model is model
    assert metadata["checkpoint_format"] == "full_weights"
    assert metadata["checkpoint_path"] == str(checkpoint_file)
    assert model.strict is False
    assert sorted(model.loaded_state_dict) == ["linear.bias", "linear.weight"]


def test_load_reward_checkpoint_into_model_supports_lora_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint_file = tmp_path / "full_weights.pt"
    torch.save(
        {
            "module.model.layers.0.q_proj.lora_A.weight": torch.ones(
                (8, 4), dtype=torch.float32
            ),
            "module.model.layers.0.q_proj.lora_B.weight": torch.ones(
                (4, 8), dtype=torch.float32
            ),
        },
        checkpoint_file,
    )

    calls: dict[str, object] = {}

    def fake_get_peft_model(model, config):
        calls["base_model"] = model
        calls["config"] = config
        wrapped_model = _FakeModel()
        calls["wrapped_model"] = wrapped_model
        return wrapped_model

    def fake_set_peft_model_state_dict(model, state_dict):
        calls["state_dict"] = state_dict
        calls["loaded_model"] = model

    monkeypatch.setattr(vlm_reward_model_module, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(
        vlm_reward_model_module,
        "set_peft_model_state_dict",
        fake_set_peft_model_state_dict,
    )

    model = _FakeModel()
    loaded_model, metadata = vlm_reward_model_module.load_reward_checkpoint_into_model(
        model, str(checkpoint_file)
    )

    assert loaded_model is calls["wrapped_model"]
    assert metadata["checkpoint_format"] == "lora"
    assert metadata["checkpoint_path"] == str(checkpoint_file)
    assert calls["config"].r == 8
    assert set(calls["config"].target_modules) == {"q_proj"}
    assert sorted(calls["state_dict"]) == [
        "model.layers.0.q_proj.lora_A.weight",
        "model.layers.0.q_proj.lora_B.weight",
    ]


def test_load_reward_checkpoint_into_model_rejects_incompatible_full_weights(tmp_path):
    checkpoint_file = tmp_path / "bad_full_weights.pt"
    torch.save({"bad.weight": torch.ones((1,), dtype=torch.float32)}, checkpoint_file)

    class _IncompatibleModel(_FakeModel):
        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state_dict = state_dict
            self.strict = strict
            return ["missing.weight"], ["bad.weight"]

    with pytest.raises(ValueError, match="incompatible with the base model"):
        vlm_reward_model_module.load_reward_checkpoint_into_model(
            _IncompatibleModel(), str(checkpoint_file)
        )


@pytest.mark.parametrize(
    ("env_world_size", "reward_world_size"),
    [(1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (4, 2), (2, 4)],
)
def test_reward_roundtrip_keeps_env_slot_alignment(
    env_world_size: int, reward_world_size: int
):
    rewards = _simulate_reward_roundtrip(
        batch_size=12,
        env_world_size=env_world_size,
        reward_world_size=reward_world_size,
    )

    assert torch.equal(rewards, torch.arange(12, dtype=torch.float32))
