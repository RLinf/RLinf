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

# ruff: noqa: E402

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

from rlinf.models.embodiment.reward import (
    get_reward_model_class,
    resolve_reward_model_backend,
    reward_model_registry,
)
from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
)
from rlinf.models.embodiment.reward.vlm_sglang_reward_model import (
    HistoryVLMSGLangRewardModel,
)
from rlinf.runners.sglang_reward_server import (
    _build_server_cfg,
    launch_sglang_reward_server_stack,
    should_launch_sglang_reward_server,
)


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return "rendered"


class _FakeRewardParser:
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        return torch.tensor([float(output) if output else 0.0 for output in outputs])


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
                [
                    [torch.zeros((1, 1, 3), dtype=torch.uint8)],
                    [torch.ones((1, 1, 3), dtype=torch.uint8)],
                ]
                for _ in slot_ids
            ],
        }


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
        self.reward_parser = _FakeRewardParser()

    def _generate(
        self, payloads: list[dict[str, object]]
    ) -> tuple[list[str], list[int]]:
        outputs = []
        for payload in payloads:
            content = payload["messages"][0]["content"]
            outputs.append(content[-1]["text"].removeprefix("prompt-"))
        return outputs, [1] * len(payloads)


class _FakeHandle:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value


class _FakeWorkerGroup:
    def __init__(self):
        self.registered_urls = []
        self.registration_kwargs = []
        self.shutdown_calls = 0

    def init_server(self):
        return _FakeHandle()

    def init_router(self):
        return _FakeHandle()

    def get_server_url(self):
        return _FakeHandle(["http://server-0:31000"])

    def get_router_url(self):
        return _FakeHandle(["http://router:32000"])

    def register_server(self, server_url, **kwargs):
        self.registered_urls.append(server_url)
        self.registration_kwargs.append(kwargs)
        return _FakeHandle()

    def shutdown(self):
        self.shutdown_calls += 1
        return _FakeHandle()


class _FakeWorkerLauncher:
    def __init__(self, group: _FakeWorkerGroup):
        self.group = group
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return self.group


class _FakeServerWorker:
    create_group_kwargs = None
    launcher = _FakeWorkerLauncher(_FakeWorkerGroup())

    @classmethod
    def create_group(cls, **kwargs):
        cls.create_group_kwargs = kwargs
        cls.launcher = _FakeWorkerLauncher(_FakeWorkerGroup())
        return cls.launcher


class _FakeRouterWorker:
    create_group_kwargs = None
    launcher = _FakeWorkerLauncher(_FakeWorkerGroup())

    @classmethod
    def create_group(cls, **kwargs):
        cls.create_group_kwargs = kwargs
        cls.launcher = _FakeWorkerLauncher(_FakeWorkerGroup())
        return cls.launcher


class _FakeComponentPlacement:
    def get_hardware_ranks(self, component_name):
        assert component_name == "reward_server"
        return [0, 1]

    def get_world_size(self, component_name):
        assert component_name == "reward_server"
        return 1

    def get_strategy(self, component_name):
        assert component_name == "reward_server"
        return "reward-server-placement"


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": "dummy",
            "precision": "bf16",
            "infer_micro_batch_size": 2,
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


def _make_sglang_cfg() -> OmegaConf:
    cfg = _make_cfg()
    cfg._runtime_sglang_api_base = "http://router:30000/v1"
    return cfg


def _make_reward_input(
    slot_ids: list[int], valid_env_ids: list[int] | None = None
) -> dict[str, object]:
    valid_env_ids = valid_env_ids or list(range(len(slot_ids)))
    valid_env_id_set = set(valid_env_ids)
    return {
        "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "main_images": torch.zeros((len(slot_ids), 1, 1, 1), dtype=torch.uint8),
        "history_input": {
            "history_window": {
                "main_images": [
                    [f"frame-{slot_id}"] if env_idx in valid_env_id_set else []
                    for env_idx, slot_id in enumerate(slot_ids)
                ]
            }
        },
    }


def test_history_vlm_backend_contracts_and_yaml_defaults():
    hf_cfg = OmegaConf.load(
        REPO_ROOT / "examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml"
    )
    sglang_cfg = OmegaConf.load(
        REPO_ROOT
        / "examples/embodiment/config/maniskill_ppo_mlp_qwentrend_sglang_reward.yaml"
    )

    assert "history_vlm_sglang" not in reward_model_registry
    assert resolve_reward_model_backend("history_vlm") == ("history_vlm", None)
    assert resolve_reward_model_backend("history_vlm", "hf") == ("history_vlm", "hf")
    assert resolve_reward_model_backend("history_vlm", "transformers") == (
        "history_vlm",
        "hf",
    )
    assert resolve_reward_model_backend("history_vlm", "sglang") == (
        "history_vlm",
        "sglang",
    )
    assert get_reward_model_class("history_vlm").__name__ == "HistoryVLMRewardModel"
    assert (
        get_reward_model_class("history_vlm", inference_backend="sglang").__name__
        == "HistoryVLMSGLangRewardModel"
    )
    with pytest.raises(ValueError, match="Unsupported reward.model.inference_backend"):
        resolve_reward_model_backend("history_vlm", "vllm")

    assert hf_cfg.reward.model.inference_backend == "hf"
    assert sglang_cfg.reward.model.inference_backend == "sglang"
    assert sglang_cfg.cluster.component_placement.reward_server == "0-1:0"
    assert "sglang_server_args" not in sglang_cfg.reward.model
    assert "sglang_router_args" not in sglang_cfg.reward.model


def test_history_vlm_transformers_writes_sparse_valid_envs_back_to_slots():
    model = _TestHistoryVLMRewardModel(_make_cfg())

    rewards = model.compute_reward(
        _make_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert model.input_builder.calls == [[21], [23]]


def test_history_vlm_sglang_writes_sparse_valid_envs_back_to_slots():
    model = _TestHistoryVLMSGLangRewardModel(_make_sglang_cfg())

    rewards = model.compute_reward(
        _make_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert model.input_builder.calls == [[21, 23]]


def test_history_vlm_sglang_builds_openai_payload_with_images_and_lora():
    model = object.__new__(HistoryVLMSGLangRewardModel)
    model.model_name = "reward-model"
    model.lora_name = "qwentrend-lora"
    model.sampling_params = {"max_tokens": 16, "temperature": 0.0}
    model.image_format = "jpeg"
    model.jpeg_quality = 95
    chw_frame = torch.zeros((3, 2, 2), dtype=torch.uint8)
    pil_frame = Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8))

    payloads = model._build_chat_payloads(
        {
            "prompt_texts_list": [["judge progress"]],
            "videos_list": [[[chw_frame], [pil_frame]]],
        }
    )

    assert payloads[0]["model"] == "qwentrend-lora"
    assert payloads[0]["lora_path"] == "qwentrend-lora"
    assert payloads[0]["max_tokens"] == 16
    content = payloads[0]["messages"][0]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
    assert content[-1] == {"type": "text", "text": "judge progress"}


def test_history_vlm_sglang_runtime_api_base_and_response_contracts():
    cfg = _make_sglang_cfg()
    del cfg._runtime_sglang_api_base
    with pytest.raises(ValueError, match="_runtime_sglang_api_base must be set"):
        _TestHistoryVLMSGLangRewardModel(cfg)

    cfg = _make_sglang_cfg()
    cfg.sglang_server_args = {"api_base": "http://router:30000/v1"}
    with pytest.raises(ValueError, match="sglang_server_args is no longer supported"):
        _TestHistoryVLMSGLangRewardModel(cfg)

    model = object.__new__(HistoryVLMSGLangRewardModel)
    text, token_count = model._extract_text_and_token_count(
        {
            "choices": [{"message": {"content": "positive"}}],
            "usage": {"completion_tokens": 3},
        }
    )
    assert text == "positive"
    assert token_count == 3


def test_sglang_reward_server_builds_lora_cfg_and_rejects_public_args():
    reward_model_cfg = OmegaConf.create(
        {
            "model_path": "/models/Qwen3-VL",
            "lora_path": "/checkpoints/qwentrend-lora",
        }
    )
    server_cfg = _build_server_cfg(
        reward_model_cfg,
        component_placement=_FakeComponentPlacement(),
    )

    assert server_cfg.enable_lora is True
    assert server_cfg.lora_paths == ["Qwen3-VL=/checkpoints/qwentrend-lora"]

    for removed_key in ("sglang_server_args", "sglang_router_args"):
        cfg = OmegaConf.create(
            {
                "reward": {
                    "use_reward_model": True,
                    "model": {
                        "model_path": "/models/QwenTrend",
                        "model_type": "history_vlm",
                        "inference_backend": "sglang",
                        removed_key: {"policy": "cache_aware"},
                    },
                }
            }
        )
        with pytest.raises(ValueError, match=removed_key):
            should_launch_sglang_reward_server(cfg)


def test_sglang_reward_server_stack_launches_router_and_injects_runtime_api_base(
    monkeypatch,
):
    cfg = OmegaConf.create(
        {
            "reward": {
                "use_reward_model": True,
                "model": {
                    "model_path": "/models/QwenTrend",
                    "model_type": "history_vlm",
                    "inference_backend": "sglang",
                    "sglang_engine_args": {
                        "max_running_requests": 16,
                    },
                },
            }
        }
    )
    monkeypatch.setattr(
        "rlinf.runners.sglang_reward_server._load_sglang_server_worker_classes",
        lambda: (_FakeServerWorker, _FakeRouterWorker),
    )

    stack = launch_sglang_reward_server_stack(
        cfg,
        cluster=object(),
        component_placement=_FakeComponentPlacement(),
    )

    server_cfg = _FakeServerWorker.create_group_kwargs["sglang_cfg"]
    router_cfg = _FakeRouterWorker.create_group_kwargs["router_cfg"]
    assert server_cfg.tp_size == 2
    assert server_cfg.enable_multimodal is True
    assert server_cfg.max_running_requests == 16
    assert router_cfg.policy == "cache_aware"
    assert cfg.reward.model._runtime_sglang_api_base == "http://router:32000/v1"
    assert stack.router_group.registered_urls == ["http://server-0:31000"]
    assert stack.router_group.registration_kwargs == [{"timeout": 600.0}]

    stack.stop()
    assert stack.router_group is None
    assert stack.server_group is None
