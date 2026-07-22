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

"""Tests for the RLinf FastWAM adapter without loading model weights."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def fastwam_modules():
    """Import the adapter against a minimal stub of the optional dependency."""
    fastwam = types.ModuleType("fastwam")
    models = types.ModuleType("fastwam.models")
    wan22 = types.ModuleType("fastwam.models.wan22")
    upstream = types.ModuleType("fastwam.models.wan22.fastwam")
    fastwam.__path__ = []
    models.__path__ = []
    wan22.__path__ = []

    class StubFastWAM(torch.nn.Module):
        pass

    upstream.FastWAM = StubFastWAM
    fastwam.models = models
    models.wan22 = wan22
    wan22.fastwam = upstream
    modules = {
        "fastwam": fastwam,
        "fastwam.models": models,
        "fastwam.models.wan22": wan22,
        "fastwam.models.wan22.fastwam": upstream,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    sys.modules.pop("rlinf.models.embodiment.fastwam", None)
    sys.modules.pop("rlinf.models.embodiment.fastwam.fastwam_policy", None)
    try:
        package = importlib.import_module("rlinf.models.embodiment.fastwam")
        policy = importlib.import_module(
            "rlinf.models.embodiment.fastwam.fastwam_policy"
        )
        yield package, policy
    finally:
        sys.modules.pop("rlinf.models.embodiment.fastwam.fastwam_policy", None)
        sys.modules.pop("rlinf.models.embodiment.fastwam", None)
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_compose_fastwam_config_without_hydra_global_state(
    tmp_path: Path,
    fastwam_modules,
) -> None:
    """OmegaConf composition resolves global task defaults and dot overrides."""
    package, _ = fastwam_modules
    (tmp_path / "data").mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "task").mkdir()
    (tmp_path / "root.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - override /task: libero\n"
        "  - _self_\n"
        "name: root\n",
        encoding="utf-8",
    )
    (tmp_path / "task" / "libero.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - override /data: images\n"
        "  - override /model: base\n"
        "  - _self_\n"
        "task_name: libero\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "images.yaml").write_text(
        "height: 224\n",
        encoding="utf-8",
    )
    (tmp_path / "model" / "base.yaml").write_text(
        "hidden_dim: 1024\n",
        encoding="utf-8",
    )

    cfg = package._compose_fastwam_cfg(
        str(tmp_path),
        "root",
        ["model.hidden_dim=2048"],
    )

    assert cfg.name == "root"
    assert cfg.task_name == "libero"
    assert cfg.data.height == 224
    assert cfg.model.hidden_dim == 2048


def test_infer_action_preserves_batch_dimension(fastwam_modules) -> None:
    """The action denoiser handles all observation samples in one model call."""
    _, policy_module = fastwam_modules
    policy = object.__new__(policy_module.FastWAMPolicy)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.torch_dtype = torch.float32
    policy.proprio_dim = 8

    class VideoExpert:
        video_attention_mask_mode = "first_frame_causal"
        fuse_vae_embedding_in_latents = False

        @staticmethod
        def pre_dit(x, timestep, context, context_mask, **kwargs):
            batch_size = x.shape[0]
            return {
                "tokens": torch.zeros(batch_size, 3, 4),
                "freqs": torch.zeros(3, 1, 1),
                "t_mod": torch.zeros(batch_size, 1),
                "context": context,
                "context_mask": context_mask,
                "meta": {"tokens_per_frame": 3},
            }

    class Scheduler:
        @staticmethod
        def build_inference_schedule(**kwargs):
            return torch.tensor([1.0, 0.5]), torch.tensor([-0.5, -0.5])

        @staticmethod
        def step(prediction, delta, latents):
            return latents + prediction * delta

    class Mot:
        @staticmethod
        def prefill_video_cache(**kwargs):
            return [kwargs["video_tokens"]]

    policy.video_expert = VideoExpert()
    policy.action_expert = types.SimpleNamespace(action_dim=7)
    policy.infer_action_scheduler = Scheduler()
    policy.mot = Mot()
    policy._encode_video_latents = types.MethodType(
        lambda self, video, tiled=False: torch.zeros(video.shape[0], 2, 1, 1, 1),
        policy,
    )
    policy.encode_prompt = types.MethodType(
        lambda self, prompt: (
            torch.zeros(len(prompt), 2, 4),
            torch.ones(len(prompt), 2, dtype=torch.bool),
        ),
        policy,
    )
    policy._append_proprio_to_context = types.MethodType(
        lambda self, context, context_mask, proprio: (context, context_mask),
        policy,
    )
    policy._build_mot_attention_mask = types.MethodType(
        lambda self, video_seq_len, action_seq_len, **kwargs: torch.ones(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
            dtype=torch.bool,
        ),
        policy,
    )
    calls = []

    def predict_noise(self, latents_action, **kwargs):
        calls.append(latents_action.shape[0])
        return torch.zeros_like(latents_action)

    policy._predict_action_noise_with_cache = types.MethodType(predict_noise, policy)

    output = policy.infer_action(
        prompt=["task 0", "task 1", "task 2"],
        input_image=torch.zeros(3, 3, 16, 16),
        action_horizon=4,
        proprio=torch.zeros(3, 8),
        num_inference_steps=2,
    )

    assert output["action"].shape == (3, 4, 7)
    assert calls == [3, 3]


def test_gradient_checkpointing_is_controlled_by_policy(fastwam_modules) -> None:
    """RLinf's FSDP lifecycle toggles every upstream checkpointing flag."""
    _, policy_module = fastwam_modules
    policy = object.__new__(policy_module.FastWAMPolicy)
    torch.nn.Module.__init__(policy)
    policy.video_expert = types.SimpleNamespace(use_gradient_checkpointing=False)
    policy.action_expert = types.SimpleNamespace(use_gradient_checkpointing=False)
    policy.mot = types.SimpleNamespace(mot_checkpoint_mixed_attn=False)

    policy.gradient_checkpointing_enable()
    assert policy.video_expert.use_gradient_checkpointing
    assert policy.action_expert.use_gradient_checkpointing
    assert policy.mot.mot_checkpoint_mixed_attn

    policy.gradient_checkpointing_disable()
    assert not policy.video_expert.use_gradient_checkpointing
    assert not policy.action_expert.use_gradient_checkpointing
    assert not policy.mot.mot_checkpoint_mixed_attn
