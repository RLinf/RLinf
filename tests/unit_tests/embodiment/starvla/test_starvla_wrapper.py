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

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.starvla import starvla_action_model as starvla_model_mod
from rlinf.models.embodiment.starvla.starvla_action_model import (
    StarVLAForRLActionPrediction,
)


class _DummyHFBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size, pad_token_id=0)


class _DummyVLMInterface(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.model = _DummyHFBackbone(hidden_size=hidden_size)
        self._ACTION_TOKEN_MIN = 1
        self._ACTION_TOKEN_MAX = 999


class _DummyStarVLAModel(nn.Module):
    def __init__(self, framework_name: str):
        super().__init__()
        self.framework_name = framework_name
        self.qwen_vl_interface = _DummyVLMInterface(hidden_size=16)
        self.proj = nn.Linear(4, 4)
        self.norm_stats = {}


def _build_wrapper(framework_name: str) -> StarVLAForRLActionPrediction:
    return StarVLAForRLActionPrediction(
        starvla_model=_DummyStarVLAModel(framework_name=framework_name),
        action_dim=7,
        num_action_chunks=2,
        add_value_head=True,
        unnorm_key=None,
        disable_action_unnormalization=True,
    )


@pytest.mark.parametrize(
    ("framework_name", "expected_head"),
    [
        ("QwenFast", "fast"),
        ("QwenAdapter", "adapter"),
        ("QwenPI", "pi"),
    ],
)
def test_default_forward_dispatches_to_handler(
    monkeypatch, framework_name: str, expected_head: str
):
    policy = _build_wrapper(framework_name)
    assert policy.action_head_type == expected_head

    calls: dict[str, object] = {}

    def _handler(
        wrapped_policy: StarVLAForRLActionPrediction,
        *,
        data: dict[str, torch.Tensor],
        compute_logprobs: bool,
        compute_entropy: bool,
        compute_values: bool,
        use_cache: bool,
    ):
        calls["policy"] = wrapped_policy
        calls["data_keys"] = sorted(data.keys())
        calls["compute_logprobs"] = compute_logprobs
        calls["compute_entropy"] = compute_entropy
        calls["compute_values"] = compute_values
        calls["use_cache"] = use_cache
        return {
            "logprobs": torch.zeros((1, 1, 1)),
            "entropy": None,
            "values": torch.zeros((1, 1)),
        }

    monkeypatch.setattr(
        starvla_model_mod,
        "get_default_forward_handler",
        lambda head: _handler if head == expected_head else None,
    )

    out = policy.default_forward(
        data={"mock_tensor": torch.ones((2, 3))},
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=True,
        use_cache=True,
    )

    assert calls["policy"] is policy
    assert calls["data_keys"] == ["mock_tensor"]
    assert calls["compute_logprobs"] is True
    assert calls["compute_entropy"] is False
    assert calls["compute_values"] is True
    assert calls["use_cache"] is True
    assert isinstance(out["logprobs"], torch.Tensor)
    assert isinstance(out["values"], torch.Tensor)


def test_default_forward_requires_data():
    policy = _build_wrapper("QwenFast")
    with pytest.raises(ValueError):
        policy.default_forward()


def test_forward_rejects_unsupported_forward_type():
    policy = _build_wrapper("QwenFast")
    with pytest.raises(NotImplementedError):
        policy.forward(forward_type=ForwardType.SFT)


def test_predict_action_batch_with_mock_rollout_handler(monkeypatch):
    policy = _build_wrapper("QwenFast")

    monkeypatch.setattr(
        starvla_model_mod.data_pipeline_utils,
        "build_examples_from_env_obs",
        lambda **kwargs: [{"id": 0}, {"id": 1}],
    )
    monkeypatch.setattr(
        starvla_model_mod.data_pipeline_utils,
        "build_sampling_param_tensors",
        lambda **kwargs: {
            "do_sample": torch.zeros((2,), dtype=torch.int64),
            "temperature": torch.ones((2,), dtype=torch.float32),
            "top_k": torch.zeros((2,), dtype=torch.int64),
            "top_p": torch.ones((2,), dtype=torch.float32),
        },
    )

    def _rollout_handler(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "output": {
                "normalized_actions": np.zeros((2, 2, 7), dtype=np.float32),
            },
            "prev_logprobs": torch.ones((2, 2, 7), dtype=torch.float32),
            "prev_values": torch.ones((2, 1), dtype=torch.float32),
        }

    monkeypatch.setattr(
        starvla_model_mod,
        "get_rollout_handler",
        lambda head: _rollout_handler,
    )

    actions, result = policy.predict_action_batch(
        env_obs={"main_images": np.zeros((2, 8, 8, 3), dtype=np.uint8)},
        calculate_logprobs=True,
        calculate_values=True,
        mode="train",
    )

    assert actions.shape == (2, 2, 7)
    assert result["prev_logprobs"].shape == (2, 2, 7)
    assert result["prev_values"].shape == (2, 1)
    assert result["forward_inputs"]["action"].shape == (2, 14)
