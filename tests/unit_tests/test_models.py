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

"""Model registration, embeddings, inference adapters, and reward helpers."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import inspect
import logging
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel, validate_embodied_cfg
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.apxinf_adapter import (
    OpenPIApxInfAdapter,
    _active_token_ids,
)
from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.modules.siglip import (
    Encoder,
    Encoder1DBlock,
    SigLIPViT,
)
from rlinf.scheduler import Worker
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)
from rlinf.workers.rollout.phyai import phyai_rollout_worker
from rlinf.workers.rollout.phyai.phyai_rollout_worker import PhyAIRolloutWorker


@pytest.mark.parametrize("layout", ["hwc", "chw"])
def test_evo_device_preprocessing_preserves_per_frame_ranges_and_clipping(layout):
    from rlinf.models.embodiment.evo1.utils.data_pipeline import build_evo1_inputs

    images = np.zeros((2, 8, 10, 3), dtype=np.float32)
    images[0] = np.linspace(-0.5, 1.0, images[0].size).reshape(images[0].shape)
    images[1] = np.linspace(-10, 300, images[1].size).reshape(images[1].shape)
    if layout == "chw":
        images = images.transpose(0, 3, 1, 2)
    observation = {"main_images": images, "task_descriptions": ["pick", "place"]}
    host = build_evo1_inputs(observation, image_size=6)
    device = build_evo1_inputs(observation, image_size=6, device="cpu")
    for host_views, device_views in zip(host[0], device[0]):
        for expected, actual in zip(host_views, device_views):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(device[1], host[1])
    assert device[2] == host[2]


def test_openpi_fused_attention_preserves_fully_masked_query_rows(monkeypatch):
    import rlinf.models.embodiment.openpi.modules.utils as openpi_utils
    from rlinf.models.embodiment.openpi.modules.gemma import Attention

    monkeypatch.setattr(openpi_utils, "_torch_compile_enabled", False)
    monkeypatch.setattr(
        openpi_utils,
        "_sdpa_attention_enabled",
        openpi_utils._sdpa_attention_enabled,
    )
    config = SimpleNamespace(
        width=16, num_heads=4, num_kv_heads=2, head_dim=4, lora_configs={}
    )
    model = Attention([config])
    inputs = [torch.randn(2, 4, 16)]
    positions = torch.arange(4).repeat(2, 1)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool).tril()
    mask[:, :, 0] = False
    for updated in (False, True):
        if updated:
            with torch.no_grad():
                model.v_proj[0].weight.add_(0.1)
        openpi_utils.set_sdpa_attention(False)
        eager_in = inputs[0].detach().clone().requires_grad_(True)
        expected, _ = model([eager_in], positions, mask)
        expected[0].sum().backward()

        openpi_utils.set_sdpa_attention(True)
        fused_in = inputs[0].detach().clone().requires_grad_(True)
        actual, _ = model([fused_in], positions, mask)
        actual[0].sum().backward()

        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fused_in.grad, eager_in.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("aspect_ratio", ["pad", "square"])
@pytest.mark.parametrize("policy_kind", ["dm0", "pi0"])
def test_dm0_tensor_images_match_native_padding_and_view_masks(
    aspect_ratio, policy_kind
):
    pytest.importorskip("dexbotic")
    from PIL import Image
    from transformers import SiglipImageProcessor

    from rlinf.models.embodiment.dexbotic_dm0.dm0_policy import (
        DexboticDM0ForRLActionPrediction,
    )
    from rlinf.models.embodiment.dexbotic_pi.dexbotic_pi_policy import (
        DexboticPi0ForRLActionPrediction,
    )

    cls = (
        DexboticDM0ForRLActionPrediction
        if policy_kind == "dm0"
        else DexboticPi0ForRLActionPrediction
    )
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.anchor = torch.nn.Parameter(torch.zeros(()))
    model.config = SimpleNamespace(image_aspect_ratio=aspect_ratio)
    model.model = SimpleNamespace(
        mm_vision_module=SimpleNamespace(
            image_processor=SiglipImageProcessor(size={"height": 6, "width": 6})
        )
    )
    images = torch.full((2, 4, 6, 3), 31, dtype=torch.uint8)
    expected = model.process_images(
        [Image.fromarray(frame.numpy()) for frame in images]
    )
    for raw in (images, images.numpy()):
        actual, masks = model._process_images_for_training(
            raw, None, torch.device("cpu")
        )
        torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=1e-6)
        assert masks.tolist() == [[True, False, False], [True, False, False]]
        assert not torch.count_nonzero(actual[:, 1:])


@pytest.mark.parametrize("precision", [None, "bf16"])
def test_dexbotic_configured_precision_supports_actions_and_training(
    precision, monkeypatch
):
    source = os.environ.get("RLINF_DEXBOTIC_TEST_OBSERVATIONS")
    if source is None:
        pytest.skip("Dexbotic checkpoint and captured observations required")
    pytest.importorskip("dexbotic")
    if Worker.torch_platform is None or not Worker.torch_platform.is_available():
        pytest.skip("Configured accelerator required")
    from hydra import compose, initialize_config_dir

    from rlinf.models import get_model
    from rlinf.utils.cuda_graph import clone_tensor_tree

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(repo))
    with initialize_config_dir(
        config_dir=str(repo / "tests/e2e_tests/embodied"), version_base=None
    ):
        cfg = compose(config_name="libero_spatial_ppo_dexbotic_pi0")
    model_cfg = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    model_cfg.precision = precision
    torch.manual_seed(42)
    model = get_model(model_cfg)
    expected_dtype = torch.float32 if precision is None else torch.bfloat16
    assert {parameter.dtype for parameter in model.parameters()} == {expected_dtype}
    with Path(source).open("rb") as stream:
        captured = pickle.load(stream)
    observation = (
        captured["args"][0] if captured["args"] else captured["kwargs"]["env_obs"]
    )
    source_batch = len(observation["task_descriptions"])

    def select(value):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return value[:2] if value.ndim and value.shape[0] == source_batch else value
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, list) and len(value) == source_batch:
            return value[:2]
        return value

    model.eval()
    actions, rollout = model.predict_action_batch(select(observation), mode="train")
    assert torch.isfinite(torch.as_tensor(actions)).all()
    model.train()
    model.zero_grad(set_to_none=True)
    output = model.default_forward(
        data=clone_tensor_tree(rollout["forward_inputs"]), compute_values=True
    )
    assert all(torch.isfinite(value).all() for value in output.values())
    weights = torch.linspace(
        -1, 1, output["logprobs"].numel(), device=output["logprobs"].device
    ).reshape_as(output["logprobs"])
    loss = (output["logprobs"].float() * weights).mean()
    loss = loss + output["values"].float().square().mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("ignore_last", [False, True])
@pytest.mark.parametrize("steps", [1, 4])
def test_evo_denoising_preserves_random_draws_masks_and_replay(
    monkeypatch, mode, joint, ignore_last, steps
):
    from rlinf.models.embodiment.evo1 import evo1_rl_head as head_ops

    class ActionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon = 2
            self.per_action_dim = 3
            self.dtype = torch.float32
            self.state_encoder = None
            self.transformer_blocks = torch.nn.ModuleList()
            self.action_projection = torch.nn.Linear(3, 8)
            self.norm_out = torch.nn.LayerNorm(8)
            self.seq_pool_proj = torch.nn.Linear(16, 8)
            self.output_projection = torch.nn.Linear(8, 6)
            self.register_buffer("time_embedding", torch.zeros(1, 1000, 8))

        def time_pos_enc(self, length):
            return self.time_embedding[:, :length]

        def _project_actions(self, actions, embodiment_id):
            return self.action_projection(actions)

        def mlp_head(self, features, embodiment_id):
            return self.output_projection(features)

    head = ActionHead()
    context = torch.zeros(2, 4, 8)
    embodiment = torch.zeros(2, dtype=torch.long)
    mask = torch.ones(2, 2, 3)
    mask[..., -1] = 0
    config = {"joint_logprob": joint, "ignore_last": ignore_last, "noise_level": 0.5}
    selection_rng = random.Random(1234)
    if mode == "eval":
        expected_indices = [-1] * steps
    elif joint:
        expected_indices = list(range(steps))
    else:
        last_step = max(steps - (2 if ignore_last else 1), 0)
        expected_indices = [selection_rng.randint(0, last_step)] * steps
    noise_rng = torch.Generator().manual_seed(1234)
    initial = (torch.rand(2, 2, 3, generator=noise_rng) * 2 - 1) * mask
    for _ in range(steps):
        torch.randn(2, 2, 3, generator=noise_rng)

    def reject_scalar_read(tensor):
        raise AssertionError("Denoising must not read device scalars on the host")

    random.seed(1234)
    torch.manual_seed(1234)
    with monkeypatch.context() as guarded, torch.no_grad():
        guarded.setattr(torch.Tensor, "item", reject_scalar_read)
        actions, chains, indices, logprobs = head_ops.rl_denoise_sample(
            head, context, None, embodiment, mask, steps, config, mode=mode
        )
    assert random.getstate() == selection_rng.getstate()
    assert torch.equal(torch.random.get_rng_state(), noise_rng.get_state())
    torch.testing.assert_close(chains[:, 0], initial, rtol=0, atol=0)
    torch.testing.assert_close(actions, chains[:, -1], rtol=0, atol=0)
    assert torch.count_nonzero(chains[..., -1]) == 0
    assert indices.tolist() == [expected_indices, expected_indices]
    if mode == "eval":
        assert torch.count_nonzero(logprobs[:, int(joint) :]) == 0
    else:
        replay = head_ops.rl_replay_logprob(
            head, context, None, embodiment, mask, chains, indices, steps, config
        )
        if joint:
            expected = logprobs
        else:
            selected = indices[:, :1, None, None].expand(-1, 1, 2, 3)
            expected = logprobs.gather(1, selected)
        torch.testing.assert_close(replay, expected, rtol=0, atol=0)
        replay.mean().backward()
        assert torch.isfinite(head.output_projection.weight.grad).all()


def test_evo_vision_attention_chunking_is_inert_off_npu(monkeypatch):
    from rlinf.models.embodiment.evo1.npu_patches import (
        set_vision_attention_batch_size,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    vision = torch.nn.Linear(4, 8)
    images = torch.randn(2, 4)
    reference = vision(images)
    assert set_vision_attention_batch_size(vision, 8) == 0
    torch.testing.assert_close(vision(images), reference, rtol=0, atol=0)


def test_evo_mask_padded_language_attention_zeros_masked_query_rows():
    from rlinf.models.embodiment.evo1.npu_patches import (
        mask_padded_language_attention,
    )

    class Qwen2Attention(torch.nn.Module):
        def forward(
            self, hidden_states, position_embeddings, attention_mask=None, **kwargs
        ):
            return hidden_states + 1.0, None

    class DummyLM(torch.nn.Module):
        def __init__(self, attn_impl: str):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation=attn_impl)
            self.attn = Qwen2Attention()

    fa2_lm = DummyLM("flash_attention_2")
    assert mask_padded_language_attention(fa2_lm) == 0

    eager_lm = DummyLM("eager")
    assert mask_padded_language_attention(eager_lm) == 1
    hidden = torch.ones(1, 3, 4)
    mask = torch.zeros(1, 1, 3, 3)
    mask[:, :, :, 1] = float("-inf")
    out, _ = eager_lm.attn(hidden, (None, None), attention_mask=mask)
    torch.testing.assert_close(out[:, 0], torch.full((1, 4), 2.0))
    torch.testing.assert_close(out[:, 1], torch.zeros((1, 4)))
    torch.testing.assert_close(out[:, 2], torch.full((1, 4), 2.0))


def test_evo_action_head_graph_capture_failure_falls_back_to_eager(monkeypatch):
    from rlinf.models.embodiment.evo1 import evo1_rl_head as head_ops
    from rlinf.models.embodiment.evo1.evo1_action_model import (
        Evo1ForRLActionPrediction,
    )
    from rlinf.models.embodiment.evo1.utils.normalizer import Normalizer

    class ActionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon = 2
            self.per_action_dim = 3
            self.dtype = torch.float32
            self.state_encoder = None
            self.transformer_blocks = torch.nn.ModuleList()
            self.action_projection = torch.nn.Linear(3, 8)
            self.norm_out = torch.nn.LayerNorm(8)
            self.seq_pool_proj = torch.nn.Linear(16, 8)
            self.output_projection = torch.nn.Linear(8, 6)
            self.register_buffer("time_embedding", torch.zeros(1, 1000, 8))

        def time_pos_enc(self, length):
            return self.time_embedding[:, :length]

        def _project_actions(self, actions, embodiment_id):
            return self.action_projection(actions)

        def mlp_head(self, features, embodiment_id):
            return self.output_projection(features)

    class DummyEvo1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = ActionHead()
            self.horizon = 2
            self.per_action_dim = 3

    class FailingGraphManager:
        def __init__(self, device):
            self.device = device
            self.graphs = {}
            self.capture_calls = 0

        def capture(self, spec):
            self.capture_calls += 1
            raise RuntimeError("capture unsupported")

    monkeypatch.setattr(
        "rlinf.models.embodiment.evo1.evo1_action_model.CUDAGraphManager",
        FailingGraphManager,
    )
    evo1 = DummyEvo1()
    normalizer = Normalizer(
        {
            "arm": {
                "ds": {
                    "state": {"min": [-1.0] * 3, "max": [1.0] * 3},
                    "action": {"min": [-1.0] * 3, "max": [1.0] * 3},
                }
            }
        }
    )
    policy = Evo1ForRLActionPrediction(
        evo1,
        normalizer,
        action_dim=3,
        num_action_chunks=2,
        arm_key="arm",
        dataset_key="ds",
    )
    policy.capture_cuda_graph(train_batch_size=2, eval_batch_size=1)

    context = torch.randn(2, 4, 8)
    action_tokens = torch.randn(2, 2, 8)
    time_emb = torch.randn(2, 8)
    embodiment = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        expected = head_ops.flow_velocity_from_embeddings(
            evo1.action_head, context, action_tokens, time_emb, embodiment
        )
        first = policy._action_head_forward(
            evo1.action_head, context, action_tokens, time_emb, embodiment
        )
        second = policy._action_head_forward(
            evo1.action_head, context, action_tokens, time_emb, embodiment
        )

    assert policy.cuda_graph_manager.capture_calls == 1
    assert len(policy._failed_graphs) == 1
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)


def test_evo_chunked_vision_preserves_actions_replay_gradients_and_rng(monkeypatch):
    """Compare the real NPU checkpoint using captured rollout observations."""
    observation_path = os.environ.get("RLINF_EVO_TEST_OBSERVATIONS")
    if not observation_path:
        pytest.skip("Set RLINF_EVO_TEST_OBSERVATIONS to captured Evo observations")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    from hydra import compose, initialize_config_dir

    from rlinf.models.embodiment.evo1.npu_patches import (
        set_vision_attention_batch_size,
    )
    from rlinf.scheduler import AcceleratorType
    from rlinf.scheduler.hardware.accelerators.ascend_npu import AscendNPUManager
    from rlinf.utils.cuda_graph import clone_tensor_tree

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(repo))
    with initialize_config_dir(
        config_dir=str(repo / "tests/e2e_tests/embodied"), version_base=None
    ):
        cfg = compose(config_name="libero_spatial_grpo_evo1")
    if not Path(cfg.actor.model.model_path).is_dir():
        pytest.skip("Evo checkpoint required")
    torch.npu.set_device(0)
    monkeypatch.setattr(Worker, "torch_device_type", "npu")
    monkeypatch.setattr(Worker, "torch_platform", torch.npu)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    AscendNPUManager.setup_worker_torch()
    cfg.actor.model.evo1.vision_attention_batch_size = 8
    model = get_model(OmegaConf.merge(cfg.actor.model, cfg.rollout.model))
    vision = model.evo1.embedder.model.vision_model
    attention_layers = set_vision_attention_batch_size(vision, 8)
    assert attention_layers > 0
    with Path(observation_path).open("rb") as stream:
        captured = pickle.load(stream)
    observation = (
        captured["args"][0] if captured["args"] else captured["kwargs"]["env_obs"]
    )
    assert observation["main_images"].shape[0] >= 8
    for updated, batch in ((False, 2), (True, 2), (False, 8)):
        if updated:
            with torch.no_grad():
                model.evo1.action_head.mlp_head.fc2.linear.bias[0].add_(0.5)
        predictions, replay, gradients, rng = [], [], [], []
        for size in (0, 8):
            assert set_vision_attention_batch_size(vision, size) == (
                attention_layers if size else 0
            )
            model.eval()
            random.seed(1234)
            torch.manual_seed(1234)
            obs = {
                key: value[:batch].clone()
                if torch.is_tensor(value)
                else value[:batch].copy()
                if isinstance(value, np.ndarray)
                else value[:batch]
                if isinstance(value, list)
                else value
                for key, value in observation.items()
            }
            predictions.append(model.predict_action_batch(obs, mode="train"))
            rng.append((random.getstate(), torch.npu.get_rng_state()))
            model.zero_grad(set_to_none=True)
            output = model.default_forward(
                forward_inputs=clone_tensor_tree(predictions[0][1]["forward_inputs"])
            )["logprobs"]
            replay.append(output.detach().clone())
            output.float().mean().backward()
            gradients.append(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.evo1.action_head.named_parameters()
                    if parameter.grad is not None
                }
            )
        assert gradients[0] and gradients[0].keys() == gradients[1].keys()
        torch.testing.assert_close(
            torch.as_tensor(predictions[1][0]),
            torch.as_tensor(predictions[0][0]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(replay[1], replay[0], rtol=0, atol=0)
        torch.testing.assert_close(
            predictions[1][1]["prev_logprobs"],
            predictions[0][1]["prev_logprobs"],
            rtol=0,
            atol=0,
        )
        for name, reference in gradients[0].items():
            torch.testing.assert_close(gradients[1][name], reference, rtol=0, atol=0)
            assert torch.isfinite(reference).all()
        assert any(torch.count_nonzero(value).item() for value in gradients[0].values())
        assert rng[0][0] == rng[1][0]
        assert torch.equal(rng[0][1], rng[1][1])


def test_evo_rollout_graph_preserves_actions_replay_and_weight_updates(monkeypatch):
    """Run the real checkpoint with observations in RLINF_EVO_TEST_OBSERVATIONS."""
    from hydra import compose, initialize_config_dir

    observation_path = os.environ.get("RLINF_EVO_TEST_OBSERVATIONS")
    if not observation_path:
        pytest.skip("Set RLINF_EVO_TEST_OBSERVATIONS to captured Evo observations")
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(repo))
    with initialize_config_dir(
        config_dir=str(repo / "tests/e2e_tests/embodied"), version_base=None
    ):
        cfg = compose(config_name="libero_spatial_grpo_evo1")
    if not Path(cfg.actor.model.model_path).is_dir():
        pytest.skip("Evo checkpoint required")
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    from rlinf.scheduler import AcceleratorType

    if hasattr(torch, "npu") and torch.npu.is_available():
        from rlinf.scheduler.hardware.accelerators.ascend_npu import AscendNPUManager

        device, platform, accelerator = "npu", torch.npu, AcceleratorType.NPU
        AscendNPUManager.setup_worker_torch()
    elif torch.cuda.is_available():
        device, platform, accelerator = "cuda", torch.cuda, AcceleratorType.NV_GPU
    else:
        pytest.skip("CUDA or Ascend graph support required")
    platform.set_device(0)
    monkeypatch.setattr(Worker, "torch_device_type", device)
    monkeypatch.setattr(Worker, "torch_platform", platform)
    monkeypatch.setattr(Worker, "accelerator_type", accelerator)
    model = get_model(OmegaConf.merge(cfg.actor.model, cfg.rollout.model))
    model.eval()
    with Path(observation_path).open("rb") as stream:
        captured = pickle.load(stream)
    observation = (
        captured["args"][0] if captured["args"] else captured["kwargs"]["env_obs"]
    )
    previous_actions = None
    try:
        for updated, batch in ((False, 2), (True, 2), (False, 1)):
            existing_graph_prediction = None
            if updated:
                with torch.no_grad():
                    model.evo1.action_head.mlp_head.fc2.linear.bias[0].add_(0.5)
                random.seed(1234)
                torch.manual_seed(1234)
                obs = {
                    key: value[:batch].clone()
                    if torch.is_tensor(value)
                    else value[:batch]
                    if isinstance(value, list)
                    else value
                    for key, value in observation.items()
                }
                existing_graph_prediction = model.predict_action_batch(
                    obs, mode="train"
                )
            predictions, rng_states, gradients = [], [], []
            for use_graph in (False, True):
                model.release_cuda_graph()
                if use_graph:
                    model.capture_cuda_graph(train_batch_size=batch, eval_batch_size=1)
                random.seed(1234)
                torch.manual_seed(1234)
                obs = {
                    key: value[:batch].clone()
                    if torch.is_tensor(value)
                    else value[:batch]
                    if isinstance(value, list)
                    else value
                    for key, value in observation.items()
                }
                predictions.append(model.predict_action_batch(obs, mode="train"))
                rng_states.append((random.getstate(), platform.get_rng_state()))
                model.zero_grad(set_to_none=True)
                output = model.default_forward(
                    forward_inputs=predictions[-1][1]["forward_inputs"]
                )["logprobs"]
                output.float().mean().backward()
                gradients.append(
                    model.evo1.action_head.seq_pool_proj.weight.grad.clone()
                )
            assert model.is_cuda_graph_enabled()
            torch.testing.assert_close(
                torch.as_tensor(predictions[1][0]),
                torch.as_tensor(predictions[0][0]),
                rtol=0,
                atol=0,
            )
            for name in ("chains", "denoise_inds"):
                torch.testing.assert_close(
                    predictions[1][1]["forward_inputs"][name],
                    predictions[0][1]["forward_inputs"][name],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                predictions[1][1]["prev_logprobs"],
                predictions[0][1]["prev_logprobs"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)
            assert torch.isfinite(gradients[1]).all()
            assert rng_states[0][0] == rng_states[1][0]
            assert torch.equal(rng_states[0][1], rng_states[1][1])
            if updated:
                torch.testing.assert_close(
                    torch.as_tensor(existing_graph_prediction[0]),
                    torch.as_tensor(predictions[0][0]),
                    rtol=0,
                    atol=0,
                )
                assert not np.array_equal(predictions[1][0], previous_actions)
            previous_actions = predictions[1][0]
    finally:
        model.release_cuda_graph()
        model.release_cuda_graph()
    assert not model.is_cuda_graph_enabled()


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_qwen_npu_attention_preserves_masks_cache_gradients_and_weight_updates(
    monkeypatch, device
):
    import copy

    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU required")
    vendor = pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    if not hasattr(vendor, "Qwen2_5_VLTextModel") or not hasattr(
        vendor.Qwen2_5_VLTextModel, "set_attn_implementation"
    ):
        pytest.skip("Transformers Qwen2.5-VL text attention interface required")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLTextConfig,
    )

    from rlinf.models.embodiment.qwen_vl.npu_attention import (
        enable_qwen_vl_npu_attention,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    torch.manual_seed(1234)
    config = Qwen2_5_VLTextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
    )
    config._attn_implementation = "sdpa"
    dtype = torch.bfloat16 if device == "npu" else torch.float32
    reference = vendor.Qwen2_5_VLTextModel(config).to(device=device, dtype=dtype).eval()
    candidate = copy.deepcopy(reference)
    precise = (
        copy.deepcopy(reference).to(device="cpu", dtype=torch.float32)
        if device == "npu"
        else None
    )
    parameters = list(candidate.parameters())
    assert enable_qwen_vl_npu_attention(candidate)
    assert all(a is b for a, b in zip(parameters, candidate.parameters()))
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    ids = torch.randint(1, 128, (2, 32), device=device)
    upstream = torch.randn(2, 32, 128, device=device) / (2 * 32 * 128)
    rtol, atol = (0.03, 0.01) if device == "npu" else (1e-5, 1e-6)

    for updated in (False, True):
        if updated:
            with torch.no_grad():
                reference.layers[0].self_attn.q_proj.weight.add_(0.001)
            candidate.load_state_dict(reference.state_dict())
            if precise is not None:
                precise.load_state_dict(reference.state_dict())
        for padded in (False, True):
            mask = torch.ones_like(ids)
            if padded:
                mask[1, :4] = 0
            outputs = []
            models = (
                (reference, candidate)
                if precise is None
                else (reference, candidate, precise)
            )
            for model in models:
                model_device = next(model.parameters()).device
                model.zero_grad(set_to_none=True)
                output = model(
                    input_ids=ids.to(model_device),
                    attention_mask=mask.to(model_device),
                    use_cache=False,
                ).last_hidden_state
                # Padding is excluded from the action loss. Fully masked rows
                # have different SDPA behavior across CPU and NPU backends.
                (
                    output.float()
                    * upstream.to(model_device)
                    * mask.to(model_device).unsqueeze(-1)
                ).sum().backward()
                outputs.append(output)
            torch.testing.assert_close(outputs[1], outputs[0], rtol=rtol, atol=atol)
            if precise is None:
                for expected, actual in zip(
                    reference.parameters(), candidate.parameters()
                ):
                    torch.testing.assert_close(
                        actual.grad, expected.grad, rtol=rtol, atol=1e-6
                    )
            else:
                # BF16 reductions can disagree elementwise near zero. Compare
                # both paths with an independent FP32 gradient instead.
                for expected, original, actual in zip(
                    precise.parameters(), reference.parameters(), candidate.parameters()
                ):
                    torch.testing.assert_close(
                        actual.grad, original.grad, rtol=0, atol=0
                    )
                    expected_grad = expected.grad
                    denominator = expected_grad.norm().clamp_min(1e-30)
                    original_error = (
                        original.grad.float().cpu() - expected_grad
                    ).norm() / denominator
                    actual_error = (
                        actual.grad.float().cpu() - expected_grad
                    ).norm() / denominator
                    assert torch.isfinite(actual.grad).all()
                    assert actual_error < 0.03
                    assert actual_error <= original_error * 1.25 + 0.002
        with torch.no_grad():
            decoded = []
            for model in (reference, candidate):
                first = model(input_ids=ids, use_cache=True)
                decoded.append(
                    model(
                        input_ids=ids[:, :2],
                        past_key_values=first.past_key_values,
                        use_cache=True,
                    ).last_hidden_state
                )
            torch.testing.assert_close(decoded[1], decoded[0], rtol=rtol, atol=atol)
            modified = ids.clone()
            modified[:, -1] = (modified[:, -1] + 1) % 128
            original = candidate(input_ids=ids, use_cache=False).last_hidden_state
            changed = candidate(input_ids=modified, use_cache=False).last_hidden_state
            torch.testing.assert_close(
                changed[:, :-1], original[:, :-1], rtol=0, atol=0
            )


def test_qwen_npu_attention_preserves_vision_windows_and_weight_updates(monkeypatch):
    import copy

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    vendor = pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    if not hasattr(vendor, "Qwen2_5_VLTextModel") or not hasattr(
        vendor.Qwen2_5_VLModel, "set_attn_implementation"
    ):
        pytest.skip("Transformers Qwen2.5-VL attention interface required")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

    from rlinf.models.embodiment.qwen_vl.linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.models.embodiment.qwen_vl.npu_attention import (
        enable_qwen_vl_npu_attention,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    torch.manual_seed(1234)
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 6, 6]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "window_size": 8,
            "out_hidden_size": 128,
            "fullatt_block_indexes": [1],
        },
    )
    config._attn_implementation = "sdpa"
    reference = (
        vendor.Qwen2_5_VLModel(config).to(device="npu", dtype=torch.bfloat16).eval()
    )
    patch_vision_patch_embed(reference)
    candidate = copy.deepcopy(reference)
    parameters = list(candidate.parameters())
    assert enable_qwen_vl_npu_attention(candidate)
    assert all(a is b for a, b in zip(parameters, candidate.parameters()))
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    pixels = torch.randn(80, 12, device="npu", dtype=torch.bfloat16)
    grid = torch.tensor([[1, 8, 8], [1, 4, 4]], device="npu")
    upstream = torch.randn(20, 128, device="npu") / (20 * 128)
    for updated in (False, True):
        if updated:
            with torch.no_grad():
                reference.visual.blocks[0].attn.qkv.weight.add_(0.001)
            candidate.load_state_dict(reference.state_dict())
        outputs = []
        for model in (reference, candidate):
            model.zero_grad(set_to_none=True)
            output = model.visual(pixels, grid_thw=grid)
            (output.float() * upstream).sum().backward()
            outputs.append(output)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0.01, atol=0.001)
        for expected, actual in zip(
            reference.visual.parameters(), candidate.visual.parameters()
        ):
            error = (actual.grad.float() - expected.grad.float()).norm()
            assert torch.isfinite(actual.grad).all()
            assert error <= expected.grad.float().norm() * 0.01 + 1e-8


def _small_openvla():
    # Prismatic configures process-wide logging when imported.
    with patch("logging.config.dictConfig"):
        pytest.importorskip("prismatic")
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from transformers import GenerationConfig, LlamaForCausalLM

    from rlinf.models.embodiment.openvla.openvla_action_model import (
        OpenVLAForRLActionPrediction,
    )

    config = OpenVLAConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32064,
        },
        norm_stats={"test": {"action": {"q01": [-1.0] * 7, "q99": [1.0] * 7}}},
    )
    # Keep the real multimodal forward and generation paths, with small modules.
    model = OpenVLAForRLActionPrediction.__new__(OpenVLAForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.config = config
    model.generation_config = GenerationConfig.from_model_config(config)
    model.language_model = LlamaForCausalLM(config.text_config)

    class VisionBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Conv2d(3, 16, 1)

        def forward(self, images):
            images = torch.nn.functional.interpolate(images, size=(16, 16))
            return self.projection(images).flatten(2).transpose(1, 2)

    def input_processor(*, text, images, **kwargs):
        batch = len(text)
        return {
            "input_ids": torch.tensor([[1, 0, 3, 29871]]).repeat(batch, 1),
            "attention_mask": torch.tensor([[1, 0, 1, 1]]).repeat(batch, 1),
            "pixel_values": images.float() / 255,
        }

    model.vision_backbone = VisionBackbone()
    model.projector = torch.nn.Linear(16, 16)
    model.value_head = torch.nn.Linear(16, 1)
    model.action_dim = 7
    model.num_action_chunks = 1
    model.unnorm_key = "test"
    model.max_prompt_length = 4
    model._init_logits_processor()
    model.setup_config_and_processor(config, input_processor)
    return model.eval()


def test_openvla_bootstrap_values_match_generation_before_and_after_weight_update():
    torch.manual_seed(1234)
    model = _small_openvla()
    obs = {
        "main_images": torch.randint(0, 256, (2, 2, 2, 3), dtype=torch.uint8),
        "task_descriptions": ["Move the block", "Move the block"],
    }
    for _ in range(2):
        _, result = model.predict_action_batch(
            env_obs=obs, do_sample=True, max_new_tokens=7
        )
        rng_state = torch.random.get_rng_state()
        values = model.predict_value_batch(obs)
        torch.testing.assert_close(values, result["prev_values"])
        assert values.shape == (2, 1)
        assert not values.requires_grad
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        with torch.no_grad():
            model.value_head.weight.add_(0.1)
            model.language_model.model.layers[0].self_attn.q_proj.weight.add_(0.01)


def test_openvla_bootstrap_values_require_a_value_head():
    model = _small_openvla()
    del model.value_head
    with pytest.raises(ValueError, match="requires a value head"):
        model.predict_value_batch({})


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


@pytest.mark.parametrize(
    "memory_enabled,history_config,use_rlt,error",
    [
        (False, None, False, None),
        (False, None, True, None),
        (False, {"num_frames": 8, "stride": 70}, False, None),
        (True, {"num_frames": 8, "stride": 70}, False, None),
        (True, None, False, "requires data.short_memory"),
        (False, {"num_frames": 8, "stride": 70}, True, "requires the RLT loader"),
        (True, {"num_frames": 8, "stride": 70}, True, "requires the RLT loader"),
    ],
)
def test_openpi_sft_validates_history_config(
    memory_enabled, history_config, use_rlt, error
):
    from rlinf.config import validate_sft_cfg

    cfg = OmegaConf.create(
        {
            "runner": {},
            "data": {"train_data_paths": "dataset", "short_memory": history_config},
            "actor": {
                "global_batch_size": 8,
                "micro_batch_size": 2,
                "model": {
                    "model_type": "openpi",
                    "openpi": {"short_memory": memory_enabled, "use_rlt": use_rlt},
                },
            },
        }
    )
    if error is None:
        validate_sft_cfg(cfg)
        assert cfg.actor.model.openpi.short_memory is memory_enabled
        assert cfg.data.short_memory == history_config
    else:
        with pytest.raises(AssertionError, match=error):
            validate_sft_cfg(cfg)


def test_openpi_eval_requires_history():
    cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied_eval"},
            "env": {"eval": {"short_memory": None}},
            "rollout": {
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 5,
                    "num_steps": 5,
                    "openpi": {
                        "config_name": "pi05_libero",
                        "task": "eval",
                        "short_memory": True,
                    },
                }
            },
        }
    )
    with pytest.raises(AssertionError, match="requires env.eval.short_memory"):
        validate_embodied_cfg(cfg)


@pytest.fixture
def siglip_with_history():
    torch.manual_seed(0)
    vision = SigLIPViT(variant="mu/32", pool_type="none")
    vision.encoder = Encoder(
        vision.width,
        3,
        vision.num_heads,
        vision.mlp_dim,
        temporal_layers=(0, 1),
        drop_history_layer=1,
    )
    return vision


@pytest.mark.parametrize("num_frames", [1, 3, 8])
@pytest.mark.parametrize("precomputed", [False, True])
def test_siglip_current_only_matches_single_frame(
    siglip_with_history, num_frames, precomputed
):
    vision = siglip_with_history.eval()
    images = torch.randn(2, num_frames, 224, 224, 3)
    frame_mask = torch.zeros(2, num_frames, dtype=torch.bool)
    frame_mask[:, -1] = True
    time_offsets = torch.arange(1 - num_frames, 1).float().expand(2, -1)

    with torch.no_grad():
        expected = vision(images[:, -1])[0]
        actual, _ = vision(
            images,
            frame_mask=frame_mask,
            time_offsets=time_offsets,
            current_only=True if precomputed else None,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_siglip_attention_preserves_outputs_and_gradients():
    torch.manual_seed(0)
    block = Encoder1DBlock(dim=32, num_heads=2).double().train()
    reference = copy.deepcopy(block)

    def request_weights(module, args, kwargs):
        return args, {**kwargs, "need_weights": True}

    reference.attn.register_forward_pre_hook(request_weights, with_kwargs=True)
    returned_weights = []
    block.attn.register_forward_hook(
        lambda module, args, output: returned_weights.append(output[1])
    )
    inputs = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    actual = block(inputs)
    expected = reference(reference_inputs)
    assert returned_weights == [None]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        inputs.grad, reference_inputs.grad, rtol=1e-10, atol=1e-10
    )
    for parameter, reference_parameter in zip(
        block.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=1e-10, atol=1e-10
        )


def _reference_space_time_attention(attention, tokens, frame_mask):
    """Compute S(TV) with explicit frame, patch, and head indexing."""
    batch_size, num_frames, num_patches, width = tokens.shape
    num_heads = attention.num_heads
    head_dim = width // num_heads
    projected = tokens @ attention.in_proj_weight.T + attention.in_proj_bias
    queries, keys, values = projected.chunk(3, dim=-1)
    temporal_values = torch.zeros_like(tokens)
    mixed_values = torch.zeros_like(tokens)
    for batch in range(batch_size):
        for frame_index in range(num_frames):
            valid_times = [
                past
                for past in range(frame_index + 1)
                if frame_mask[batch, past] or past == frame_index
            ]
            for patch in range(num_patches):
                for head in range(num_heads):
                    channels = slice(head * head_dim, (head + 1) * head_dim)
                    query = queries[batch, frame_index, patch, channels]
                    scores = torch.stack(
                        [
                            query @ keys[batch, past, patch, channels]
                            for past in valid_times
                        ]
                    ) / math.sqrt(head_dim)
                    weights = scores.softmax(dim=0)
                    temporal_values[batch, frame_index, patch, channels] = sum(
                        weight * values[batch, past, patch, channels]
                        for weight, past in zip(weights, valid_times)
                    )
        for frame_index in range(num_frames):
            for patch in range(num_patches):
                for head in range(num_heads):
                    channels = slice(head * head_dim, (head + 1) * head_dim)
                    query = queries[batch, frame_index, patch, channels]
                    scores = torch.stack(
                        [
                            query @ keys[batch, frame_index, other, channels]
                            for other in range(num_patches)
                        ]
                    ) / math.sqrt(head_dim)
                    weights = scores.softmax(dim=0)
                    mixed_values[batch, frame_index, patch, channels] = sum(
                        weights[other]
                        * temporal_values[batch, frame_index, other, channels]
                        for other in range(num_patches)
                    )
    return mixed_values @ attention.out_proj.weight.T + attention.out_proj.bias


def test_siglip_attention_matches_space_time_reference():
    from rlinf.models.embodiment.openpi.modules.short_memory import (
        factorized_space_time_attention,
    )

    torch.manual_seed(0)
    batch_size, num_frames, num_patches, width, num_heads = 3, 3, 2, 4, 2
    attention = torch.nn.MultiheadAttention(width, num_heads).double().eval()
    torch.nn.init.normal_(attention.in_proj_bias)
    torch.nn.init.normal_(attention.out_proj.bias)
    tokens = torch.randn(
        batch_size, num_frames, num_patches, width, dtype=torch.float64
    )
    frame_mask = torch.tensor([[False, True, True], [True, True, True], [False] * 3])

    with torch.no_grad():
        expected = _reference_space_time_attention(attention, tokens, frame_mask)
        actual = factorized_space_time_attention(attention, tokens, frame_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_siglip_time_encoding_preserves_timestamp_precision():
    from rlinf.models.embodiment.openpi.modules.short_memory import (
        fixed_relative_time_encoding,
    )

    block = Encoder1DBlock(dim=32, num_heads=2).eval()
    for parameter in block.parameters():
        torch.nn.init.zeros_(parameter)
    tokens = torch.zeros(1, 3, 2, 32, dtype=torch.bfloat16)
    time_offsets = torch.tensor([[-24.57, -0.12345, 0.0]])

    actual = block(
        tokens,
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        time_offsets=time_offsets,
    )
    expected = fixed_relative_time_encoding(time_offsets, 32).to(tokens.dtype)
    torch.testing.assert_close(
        actual, expected[:, :, None].expand_as(tokens).float(), rtol=0, atol=0
    )


def test_openpi_prefix_preserves_camera_inputs(siglip_with_history):
    from rlinf.models.embodiment.openpi.pi0 import Pi0

    vision = siglip_with_history.eval()
    masks = {
        "base_0_rgb": torch.tensor([[False, True, True], [False, False, True]]),
        "left_wrist_0_rgb": torch.tensor([[False, False, True]] * 2),
        "right_wrist_0_rgb": torch.zeros(2, 3, dtype=torch.bool),
    }
    observation = Observation(
        images={key: torch.randn(2, 3, 224, 224, 3) for key in masks},
        image_masks=masks,
        state=torch.zeros(2, 8),
        history_time_offsets=torch.tensor([[-0.2, -0.1, 0.0]] * 2),
    )
    policy = SimpleNamespace(
        img=vision, _config=SimpleNamespace(short_memory=True), pcd=False
    )
    with torch.no_grad():
        expected = torch.cat(
            [
                vision(
                    observation.images[name],
                    frame_mask=mask,
                    time_offsets=observation.history_time_offsets,
                )[0]
                for name, mask in masks.items()
            ],
            dim=1,
        )
        decisions = []

        def record_decision(module, args, kwargs):
            decisions.append(kwargs["current_only"])

        handle = vision.register_forward_pre_hook(record_decision, with_kwargs=True)
        actual, input_mask, _ = Pi0.embed_prefix(policy, observation)
        handle.remove()

    assert decisions == [False, True, True]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_mask = torch.cat(
        [mask[:, -1:].expand(-1, 49) for mask in masks.values()], dim=1
    )
    torch.testing.assert_close(input_mask, expected_mask)


@pytest.mark.parametrize("num_frames", [3, 8])
def test_siglip_checkpoint_preserves_outputs_and_gradients(
    siglip_with_history, num_frames
):
    vision = siglip_with_history.train()
    reference = copy.deepcopy(vision)
    vision.encoder.gradient_checkpointing = True
    images = torch.randn(3, num_frames, 224, 224, 3, requires_grad=True)
    frame_mask = torch.ones(3, num_frames, dtype=torch.bool)
    frame_mask[0, :-1] = False
    frame_mask[1, 0] = False
    time_offsets = torch.arange(1 - num_frames, 1).float().expand(3, -1)
    calls = []

    def record_call(module, args):
        calls.append((module, args[0].ndim))

    owners = [vision, vision.stem, vision.encoder, *vision.encoder.layers]
    handles = [owner.register_forward_pre_hook(record_call) for owner in owners]
    tokens, _ = vision(images, frame_mask=frame_mask, time_offsets=time_offsets)
    for handle in handles:
        handle.remove()

    # Mixed histories must preserve the vision encoder's module boundaries.
    assert calls == list(zip(owners, [5, 4, 4, 4, 4, 3]))
    assert tokens.shape == (3, 49, vision.width)
    assert torch.isfinite(tokens).all()
    changed_padding = images.detach().clone()
    changed_padding[~frame_mask] = torch.randn_like(changed_padding[~frame_mask]) * 100
    with torch.no_grad():
        current_only = vision(images[:1, -1])[0]
        torch.testing.assert_close(tokens[:1], current_only, rtol=1e-5, atol=1e-5)
        changed_tokens, _ = vision(
            changed_padding,
            frame_mask=frame_mask,
            time_offsets=time_offsets,
        )
    torch.testing.assert_close(changed_tokens, tokens, rtol=0, atol=0)

    reference_images = images.detach().clone().requires_grad_()
    reference_tokens, _ = reference(
        reference_images, frame_mask=frame_mask, time_offsets=time_offsets
    )
    torch.testing.assert_close(tokens, reference_tokens, rtol=0, atol=0)
    recomputed = []
    handles = [
        layer.register_forward_pre_hook(lambda module, args: recomputed.append(module))
        for layer in vision.encoder.layers
    ]
    upstream_gradient = torch.randn_like(tokens)
    (tokens * upstream_gradient).sum().backward()
    for handle in handles:
        handle.remove()
    (reference_tokens * upstream_gradient).sum().backward()
    assert set(recomputed) == set(vision.encoder.layers)
    torch.testing.assert_close(images.grad, reference_images.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in vision.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)
    assert torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad[~frame_mask]) == 0
    assert images.grad[1, 1:-1].abs().sum() > 0
    assert vision.stem.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("missing", ["frame_mask", "time_offsets"])
def test_siglip_history_requires_mask_and_time(siglip_with_history, missing):
    inputs = {
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "time_offsets": torch.tensor([[-2.0, -1.0, 0.0]]),
    }
    inputs.pop(missing)
    with pytest.raises(ValueError, match="must be provided together"):
        siglip_with_history(torch.randn(1, 3, 224, 224, 3), **inputs)


def test_siglip_temporal_attention_is_causal():
    torch.manual_seed(0)
    block = Encoder1DBlock(dim=32, num_heads=2).eval()
    tokens = torch.randn(1, 4, 3, 32)
    frame_mask = torch.ones(1, 4, dtype=torch.bool)
    time_offsets = torch.tensor([[-3.0, -2.0, -1.0, 0.0]])
    changed_tokens = tokens.clone()
    changed_tokens[:, -1] += torch.randn_like(changed_tokens[:, -1]) * 100

    with torch.no_grad():
        original = block(tokens, frame_mask=frame_mask, time_offsets=time_offsets)
        changed = block(
            changed_tokens, frame_mask=frame_mask, time_offsets=time_offsets
        )

    torch.testing.assert_close(changed[:, :-1], original[:, :-1], rtol=0, atol=0)
    assert not torch.allclose(changed[:, -1], original[:, -1])


def test_openpi_augmentation_aligns_history_frames():
    generator = torch.Generator().manual_seed(0)
    image_keys = ("base_0_rgb", "left_wrist_0_rgb")
    images = {
        key: torch.rand(2, 16, 16, 3, generator=generator) * 2 - 1 for key in image_keys
    }
    time_offsets = torch.tensor([[-2.0, -1.0, 0.0]]).expand(2, -1)
    current = Observation(images=images, image_masks={}, state=torch.zeros(2, 8))
    history = Observation(
        images={
            key: image[:, None].expand(-1, 3, -1, -1, -1)
            for key, image in images.items()
        },
        image_masks={},
        state=current.state,
        history_time_offsets=time_offsets,
    )
    outputs = [
        preprocess_observation(
            observation,
            train=True,
            image_keys=image_keys,
            image_resolution=(16, 16),
            rng=torch.Generator().manual_seed(7),
        )
        for observation in (current, history)
    ]

    for key in image_keys:
        expected = outputs[0].images[key][:, None].expand_as(outputs[1].images[key])
        torch.testing.assert_close(outputs[1].images[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1].history_time_offsets, time_offsets)


def test_hf_rollout_merges_nested_observations():
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    def observation(values):
        values = torch.tensor(values, dtype=torch.float32)
        return {
            "states": values[:, None],
            "task_descriptions": [f"task {value}" for value in values.tolist()],
            "history": {
                "main_images": values[:, None, None, None, None].expand(-1, 3, 2, 2, 3),
                "wrist_images": (values + 10)[:, None, None, None, None].expand(
                    -1, 3, 2, 2, 3
                ),
                "frame_mask": torch.tensor([[False, True, True]] * len(values)),
                "time_offsets": torch.tensor([[0.0, -0.1, 0.0]] * len(values)),
            },
        }

    first, second, terminal = observation([1, 2]), observation([3]), observation([7])
    worker = object.__new__(MultiStepRolloutWorker)
    merged = worker._merge_obs_batches(
        [
            {"obs": first, "final_obs": None},
            {"obs": second, "final_obs": terminal, "eval_epoch_done": True},
        ]
    )

    expected = observation([1, 2, 3])
    expected_terminal = observation([1, 2, 7])
    for key in ("states", "history"):
        torch.testing.assert_close(merged["obs"][key], expected[key])
        torch.testing.assert_close(merged["final_obs"][key], expected_terminal[key])
    assert merged["obs"]["task_descriptions"] == expected["task_descriptions"]
    assert (
        merged["final_obs"]["task_descriptions"]
        == expected_terminal["task_descriptions"]
    )
    assert merged["eval_epoch_done"] is True


def test_openpi_env_io_reads_history():
    from rlinf.models.embodiment.openpi.env_io import EnvIO

    def frame_transform(sample):
        return {
            "image": {
                "base_0_rgb": sample["observation/image"],
                "left_wrist_0_rgb": sample["observation/wrist_image"],
            },
            "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True},
            "state": sample["observation/state"],
            "prompt": sample["prompt"],
        }

    tokenized_prompts = []

    def sample_transform(sample):
        tokenized_prompts.append(sample.pop("prompt"))
        sample["tokenized_prompt"] = np.array([1, 2])
        sample["tokenized_prompt_mask"] = np.array([True, True])
        return sample

    adapter = EnvIO()
    adapter.config_name = "pi05_libero"
    adapter.state_indices = ()
    adapter.device = torch.device("cpu")
    adapter._frame_transform = frame_transform
    adapter._sample_transform = sample_transform
    images = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1, 1)
    history = {
        "main_images": images.expand(-1, -1, 2, 2, 3),
        "wrist_images": (images + 10).expand(-1, -1, 2, 2, 3),
        "frame_mask": torch.tensor([[False, True, True], [False, False, True]]),
        "time_offsets": torch.tensor([[0.0, -0.1, 0.0], [0.0, 0.0, 0.0]]),
    }
    states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    observation = adapter._history_to_observation(
        {
            "history": history,
            "states": states,
            "task_descriptions": ["first task", "second task"],
        }
    )

    assert sorted(tokenized_prompts) == ["first task", "second task"]
    torch.testing.assert_close(observation.images["base_0_rgb"], history["main_images"])
    torch.testing.assert_close(
        observation.images["left_wrist_0_rgb"], history["wrist_images"]
    )
    for mask in observation.image_masks.values():
        torch.testing.assert_close(mask, history["frame_mask"])
    torch.testing.assert_close(observation.state, states)
    torch.testing.assert_close(
        observation.history_time_offsets, history["time_offsets"]
    )


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


_STARVLA_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/starvla/utils"
)
_FRANKA_ACTION_STATS = {
    "q01": [-0.5] * 7,
    "q99": [0.5] * 7,
    "min": [-1.0] * 7,
    "max": [1.0] * 7,
    "mask": [True] * 6 + [False],
}


def _load_starvla_util(name: str) -> ModuleType:
    # The starvla package __init__ imports starVLA, which only its venv has.
    spec = importlib.util.spec_from_file_location(
        f"starvla_{name}_under_test", _STARVLA_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def starvla_preprocessing(monkeypatch):
    from PIL import Image

    image_tools = ModuleType("deployment.model_server.tools.image_tools")
    image_tools.to_pil_preserve = lambda views: [
        Image.fromarray(np.asarray(view.cpu() if torch.is_tensor(view) else view))
        for view in views
    ]
    trainer_tools = ModuleType("starVLA.training.trainer_utils.trainer_tools")
    trainer_tools.resize_images = lambda images, target_size: images
    monkeypatch.setitem(sys.modules, image_tools.__name__, image_tools)
    monkeypatch.setitem(sys.modules, trainer_tools.__name__, trainer_tools)
    root = "starvla_preprocessing_under_test"
    for suffix, directory in (
        ("", _STARVLA_UTILS_DIR.parent),
        (".utils", _STARVLA_UTILS_DIR),
        (".action_heads", _STARVLA_UTILS_DIR.parent / "action_heads"),
    ):
        package = ModuleType(root + suffix)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    try:
        yield SimpleNamespace(
            data=importlib.import_module(root + ".utils.data_pipeline"),
            vlm=importlib.import_module(root + ".utils.vlm_preprocess"),
            oft=importlib.import_module(root + ".action_heads.oft"),
        )
    finally:
        for name in list(sys.modules):
            if name.startswith(root + "."):
                sys.modules.pop(name, None)


def test_starvla_device_views_match_host_clipping(starvla_preprocessing):
    images = torch.linspace(-20, 300, 2 * 4 * 5 * 3).reshape(2, 4, 5, 3)
    obs = {"main_images": images, "task_descriptions": ["pick", "place"]}
    kwargs = {
        "state_adapter_name": "none",
        "prepare_state_tensor": None,
        "include_state": False,
    }
    host = starvla_preprocessing.data.build_examples_from_env_obs(obs, **kwargs)
    device = starvla_preprocessing.data.build_examples_from_env_obs(
        obs, image_device="cpu", **kwargs
    )
    for expected, actual in zip(host, device):
        np.testing.assert_array_equal(
            actual["image"][0].numpy(), np.asarray(expected["image"][0])
        )
        assert actual["lang"] == expected["lang"]


def test_starvla_prompt_cache_uses_each_processors_template(starvla_preprocessing):
    class Processor:
        def __init__(self, prefix):
            self.prefix = prefix

        def apply_chat_template(self, conversation, **kwargs):
            return self.prefix + conversation[0]["content"][-1]["text"]

        def __call__(self, *, text, images, **kwargs):
            return {"input_ids": torch.tensor([[ord(prompt[0])] for prompt in text])}

    examples = [{"image": [torch.zeros(4, 5, 3, dtype=torch.uint8)], "lang": "pick"}]
    first, second = Processor("A"), Processor("B")
    for processor in (first, second, first):
        result = starvla_preprocessing.vlm.build_device_vlm_inputs(
            SimpleNamespace(), examples, SimpleNamespace(processor=processor)
        )
        assert result["input_ids"].tolist() == [[ord(processor.prefix)]]


def test_starvla_oft_keeps_the_host_builder_fallback_for_tensor_views(
    starvla_preprocessing,
):
    class Interface:
        def build_qwenvl_inputs(self, *, images, instructions):
            return {
                "pixel_values": torch.from_numpy(np.asarray(images[0][0]).copy()),
                "prompt": instructions,
            }

    image = torch.full((4, 5, 3), 127, dtype=torch.uint8)
    result = starvla_preprocessing.oft._build_oft_vlm_inputs(
        SimpleNamespace(qwen_vl_interface=Interface(), action_token="x"),
        num_action_chunks=2,
        examples=[{"image": [image], "lang": "pick"}],
    )
    torch.testing.assert_close(result["pixel_values"], image)
    assert "next 2 robot actions" in result["prompt"][0]


@pytest.mark.parametrize(("source", "bound"), [("q01q99", 0.5), ("minmax", 1.0)])
def test_starvla_action_stats_follow_the_configured_source(source, bound):
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    stats = action_space.resolve_action_norm_stats(
        model, "franka", action_dim=7, action_stats_source=source
    )

    np.testing.assert_array_equal(stats["q99"], [bound] * 7)
    np.testing.assert_array_equal(stats["q01"], [-bound] * 7)
    np.testing.assert_array_equal(stats["mask"], [True] * 6 + [False])


def test_starvla_action_stats_name_the_available_keys_for_an_unknown_key():
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    with pytest.raises(RuntimeError, match=r"available keys: \['franka'\]"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)


def test_starvla_action_stats_require_a_norm_stats_mapping():
    action_space = _load_starvla_util("action_space")

    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(
            SimpleNamespace(norm_stats=None), "franka", action_dim=7
        )


def test_starvla_env_actions_keep_their_shape_and_map_the_libero_gripper(monkeypatch):
    action_space = _load_starvla_util("action_space")
    received_shapes = []

    def unnormalize_actions(actions, action_norm_stats):
        received_shapes.append(actions.shape)
        return actions

    tools = ModuleType("starVLA.model.tools")
    tools.FrameworkTools = SimpleNamespace(unnormalize_actions=unnormalize_actions)
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    normalized = np.zeros((2, 3, 7), dtype=np.float32)
    normalized[..., 0] = 0.25
    normalized[0, :, 6] = 1.0
    stats = {"q99": np.ones(7), "q01": -np.ones(7), "mask": np.ones(7, dtype=bool)}

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # starVLA unnormalizes [T, action_dim]; the chunk layout comes back intact.
    assert received_shapes == [(6, 7)]
    assert env_actions.shape == (2, 3, 7)
    np.testing.assert_array_equal(env_actions[..., 0], 0.25)
    # LIBERO wants the 0/1 gripper as -1 (open) / +1 (closed).
    np.testing.assert_array_equal(env_actions[0, :, 6], -1.0)
    np.testing.assert_array_equal(env_actions[1, :, 6], 1.0)


def test_starvla_autocast_targets_the_worker_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    # CPU stands in for a non-CUDA accelerator such as an Ascend NPU.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_starvla_autocast_is_a_noop_without_an_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    monkeypatch.setattr(Worker, "torch_device_type", None)

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


def test_starvla_gaussian_is_float32_and_keeps_the_gradient_path():
    accelerator = _load_starvla_util("accelerator")
    mean = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.nn.Parameter(torch.zeros(3))

    dist = accelerator.build_gaussian(mean, log_std.exp())
    sample = dist.rsample()

    assert dist.loc.dtype == dist.scale.dtype == sample.dtype == torch.float32
    dist.log_prob(sample.detach()).sum().backward()
    assert mean.grad is not None and mean.grad.dtype == torch.bfloat16
    assert log_std.grad is not None


class _QwenVisionPatchEmbed(torch.nn.Module):
    """Shape contract of Qwen2.5-VL PatchEmbed: Conv3d kernel == stride."""

    def __init__(self, in_channels=3, temporal=2, patch=4, embed_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal
        self.patch_size = patch
        kernel = (temporal, patch, patch)
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.view(-1, self.proj.out_channels)


def test_qwen_vl_linear_patch_embed_matches_conv3d_and_backprops():
    from rlinf.models.embodiment.qwen_vl.linear_patch_embed import (
        _linear_patch_embed_forward,
    )

    torch.manual_seed(0)
    module = _QwenVisionPatchEmbed()
    patches = torch.randn(5, 3 * 2 * 4 * 4, requires_grad=True)

    conv_out = module(patches)
    linear_out = _linear_patch_embed_forward(module, patches)
    torch.testing.assert_close(linear_out, conv_out, rtol=1e-5, atol=1e-5)

    linear_out.sum().backward()
    assert module.proj.weight.grad is not None
    assert patches.grad is not None


def test_qwen_vl_linear_patch_embed_is_rebound_on_npu(monkeypatch):
    from rlinf.models.embodiment.qwen_vl.linear_patch_embed import (
        _linear_patch_embed_forward,
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())
    original_forward = model[0].forward

    assert patch_vision_patch_embed(model) == 1
    assert model[0].forward.__func__ is _linear_patch_embed_forward
    assert original_forward.__func__ is not _linear_patch_embed_forward


def test_qwen_vl_linear_patch_embed_is_left_alone_on_nvidia(monkeypatch):
    from rlinf.models.embodiment.qwen_vl.linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())

    assert patch_vision_patch_embed(model) == 0
    assert model[0].forward.__func__ is _QwenVisionPatchEmbed.forward


_WAN_NPU_PATCHES = "rlinf.envs.sim.world_model.backend.npu_patches"


@pytest.fixture
def wan_dit(monkeypatch):
    """diffsynth's Wan DiT module, whose operators the NPU patches rebind."""
    from rlinf.utils.patcher import Patcher

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        return q

    def rope_apply(x, freqs, num_heads):
        return x

    class RMSNorm(torch.nn.Module):
        def forward(self, x):
            return x

    dit = ModuleType("diffsynth.models.wan_video_dit")
    dit.flash_attention, dit.rope_apply, dit.RMSNorm = (
        flash_attention,
        rope_apply,
        RMSNorm,
    )
    for name in ("diffsynth", "diffsynth.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, dit.__name__, dit)
    yield dit
    Patcher.clear()


def _import_wan_npu_patches(monkeypatch, *, mindiesd: bool) -> ModuleType:
    """Import the Wan NPU patches on an Ascend stack with or without MindIE-SD."""
    vendor = {"torch_npu": ModuleType("torch_npu"), "mindiesd": None}
    if mindiesd:
        names = (
            "mindiesd",
            "mindiesd.layers",
            "mindiesd.layers.flash_attn",
            "mindiesd.layers.flash_attn.attention_forward",
        )
        vendor.update({name: ModuleType(name) for name in names})
        vendor["mindiesd"].rotary_position_embedding = lambda *args, **kwargs: None
        vendor[names[-1]].attention_forward = lambda *args, **kwargs: None
    for name, module in vendor.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.find_spec(_WAN_NPU_PATCHES)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _WAN_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _patch_like_wan_backend(npu_patches: ModuleType) -> None:
    """Run the patch sequence ``WanBackend._build_pipeline`` runs."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    npu_patches.apply_npu_patches(Patcher)
    Patcher.apply()


def _wan_operators(dit: ModuleType) -> tuple:
    return dit.flash_attention, dit.rope_apply, dit.RMSNorm.forward


def test_wan_npu_patches_leave_diffsynth_alone_on_nvidia(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators


def test_wan_npu_patches_log_why_mindiesd_is_unavailable(wan_dit, monkeypatch, caplog):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=False)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators
    assert "import of mindiesd halted" in caplog.text


def test_wan_npu_patches_rebind_the_dit_operators_on_every_build(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    kernels = (
        npu_patches.npu_flash_attention,
        npu_patches.npu_rope_apply,
        npu_patches.npu_rmsnorm_forward,
    )

    # Every WanBackend in the process repeats the sequence on the patched module.
    for _ in range(2):
        _patch_like_wan_backend(npu_patches)
        assert _wan_operators(wan_dit) == kernels


_DREAMZERO_NPU_PATCHES = "rlinf.models.embodiment.dreamzero.patch.npu_patches"


@pytest.fixture
def dreamzero_npu_patches(monkeypatch) -> ModuleType:
    """DreamZero's NPU operators, loaded without the DreamZero (groot) package.

    The dreamzero package ``__init__`` builds the model and imports groot; the
    replacement operators only need torch.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/dreamzero/patch/npu_patches.py"
    )
    spec = importlib.util.spec_from_file_location(_DREAMZERO_NPU_PATCHES, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _DREAMZERO_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _complex_rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    """DreamZero's CUDA RoPE table: complex128 rotations per channel pair."""
    angles = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(angles), angles)


def _complex_rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    batch, seq_len, heads, _ = x.shape
    pairs = torch.view_as_complex(x.double().reshape(batch, seq_len, heads, -1, 2))
    return torch.view_as_real(pairs * freqs.unsqueeze(0)).flatten(3).float()


def _video_rope_freqs(tables: list[torch.Tensor], f: int, h: int, w: int):
    """Concatenate the frame/height/width tables per token, as the DiT does."""
    return torch.cat(
        [
            tables[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            tables[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            tables[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)


def test_dreamzero_npu_rope_matches_the_complex_cuda_rope(dreamzero_npu_patches):
    ops = dreamzero_npu_patches
    head_dim, f, h, w = 128, 3, 4, 5
    num_action, num_state, blocks = 16, 1, 2
    splits = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    video = _video_rope_freqs([ops.rope_params(1024, d) for d in splits], f, h, w)
    video_ref = _video_rope_freqs(
        [_complex_rope_params(1024, d) for d in splits], f, h, w
    )
    action, action_ref = (
        ops.rope_params(10240, head_dim),
        _complex_rope_params(10240, head_dim),
    )
    state, state_ref = (
        ops.rope_params(1024, head_dim),
        _complex_rope_params(1024, head_dim),
    )
    torch.manual_seed(0)

    x = torch.randn(2, f * h * w, 4, head_dim)
    torch.testing.assert_close(
        ops.rope_apply(x, None, video), _complex_rope_apply(x, video_ref)
    )

    # Training appends every action block, then every state block.
    register = blocks * (num_action + num_state)
    x = torch.randn(2, f * h * w + register, 4, head_dim)
    expected = torch.cat(
        [
            video_ref,
            action_ref[: blocks * num_action].unsqueeze(1),
            state_ref[: blocks * num_state].unsqueeze(1),
        ]
    )
    torch.testing.assert_close(
        ops.rope_action_apply(x, video, action, state, register, num_action, num_state),
        _complex_rope_apply(x, expected),
    )

    # A cached rollout block uses the offsets of its own action/state index.
    x = torch.randn(2, f * h * w + num_action + num_state, 4, head_dim)
    expected = torch.cat(
        [
            video_ref,
            action_ref[num_action : 2 * num_action].unsqueeze(1),
            state_ref[num_state : 2 * num_state].unsqueeze(1),
        ]
    )
    torch.testing.assert_close(
        ops.causal_rope_action_apply(
            x, video, action, state, num_action + num_state, num_action, num_state, 1
        ),
        _complex_rope_apply(x, expected),
    )

    # FP32 angles drift ~3e-5 from FP64 at t=999; the DiT casts the embedding
    # to bf16 right after, whose rounding is ~100x coarser.
    timesteps = torch.tensor([0.0, 17.0, 999.0])
    angles = torch.outer(
        timesteps.double(),
        torch.pow(10000, -torch.arange(128, dtype=torch.float64).div(128)),
    )
    torch.testing.assert_close(
        ops.sinusoidal_embedding_1d(256, timesteps),
        torch.cat([angles.cos(), angles.sin()], dim=1).float(),
        atol=1e-4,
        rtol=0,
    )


def _masked_attention(q, k, v, mask):
    """Attention over [B, L, H, D] tensors with an explicit [B, Lq, Lk] mask."""
    q, k, v = q.float(), k.float(), v.float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~mask[:, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v).nan_to_num(0.0)


def test_dreamzero_npu_attention_aligns_masks_like_flash_attention(
    dreamzero_npu_patches,
):
    attention = dreamzero_npu_patches.flash_attention
    torch.manual_seed(0)
    q = torch.randn(2, 6, 2, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 10, 2, 8, dtype=torch.bfloat16)
    v = torch.randn(2, 10, 2, 8, dtype=torch.bfloat16)
    rows, cols = torch.arange(6)[:, None], torch.arange(10)[None]
    # FlashAttention aligns causal and window masks to the bottom-right corner,
    # so query i of 6 sees key i + 4 of 10, not key i as in SDPA's is_causal.
    diagonal = rows + 4
    lengths = torch.tensor([10, 7])
    cases = {
        "causal": ({"causal": True}, cols <= diagonal),
        "window": (
            {"window_size": (3, 0)},
            (cols >= diagonal - 3) & (cols <= diagonal),
        ),
        "k_lens": ({"k_lens": lengths}, cols < lengths[:, None, None]),
    }
    for name, (kwargs, mask) in cases.items():
        expected = _masked_attention(q, k, v, mask.expand(2, 6, 10))
        actual = attention(q, k, v, **kwargs)
        assert actual.dtype == torch.bfloat16, name
        torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)

    # Padded query rows have no keys to attend to and come back as zeros.
    q_lens = torch.tensor([6, 4])
    actual = attention(q, k, v, q_lens=q_lens, k_lens=lengths)
    assert actual[1, 4:].abs().max() == 0
    torch.testing.assert_close(
        actual[1, :4].float(),
        _masked_attention(q, k, v, (cols < 7).expand(2, 6, 10))[1, :4],
        atol=2e-2,
        rtol=2e-2,
    )


def test_dreamzero_npu_patches_leave_torch_alone_off_npu(
    dreamzero_npu_patches, monkeypatch
):
    from rlinf.scheduler import AcceleratorType
    from rlinf.utils.patcher import Patcher

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    Patcher.clear()
    dreamzero_npu_patches.apply_npu_patches(Patcher)
    # Nothing is registered, so applying never imports DreamZero.
    Patcher.apply()
    Patcher.clear()

    assert not torch._dynamo.config.disable
    assert "torch_npu.contrib.transfer_to_npu" not in sys.modules


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


def _success_potential_state_machine():
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        ShapedVLMRewardModel,
    )

    model = ShapedVLMRewardModel.__new__(ShapedVLMRewardModel)
    model.potential_gamma = 1.0
    model.potential_scale = 1.0
    model.potential_ema_alpha = 0.5
    model.potential_clip = 0.0
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model.gt_success_bonus = 0.0
    model.infer_micro_batch_size = 0
    model._previous_potentials = None
    model._success_fired = None
    model._success_streak = None
    return model


def test_empty_history_input_still_resets_shaping_state_on_done():
    model = _success_potential_state_machine()
    model._previous_potentials = torch.tensor([0.4, 0.8])
    model._success_fired = torch.tensor([True, True])
    model._success_streak = torch.tensor([3, 1], dtype=torch.int32)

    rewards = model.compute_reward(
        {
            "history_input": {},
            "dones": torch.tensor([True, False]),
        }
    )

    assert rewards.tolist() == pytest.approx([0.0, 0.0])
    assert torch.isnan(model._previous_potentials[0])
    assert float(model._previous_potentials[1]) == pytest.approx(0.8)
    assert model._success_fired.tolist() == [False, True]
    assert model._success_streak.tolist() == [0, 1]


def test_robotwin_t5_reward_scores_env_observations():
    from rlinf.models.embodiment.reward import get_reward_model_class

    model_cls = get_reward_model_class("robotwin_t5")
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.batch_size = 2
    model.register_buffer("_mean", torch.zeros(1, 3, 1, 1), False)
    seen = []

    def fake_logits(images, instructions):
        seen.append((images, instructions))
        return torch.zeros(images.shape[0])

    model._logits = fake_logits
    images = torch.tensor([0, 255, 51], dtype=torch.uint8).view(3, 1, 1, 1)

    rewards = model.compute_reward(
        {
            "main_images": images.expand(3, 4, 4, 3),
            "task_descriptions": ["a", "b", "c"],
        }
    )

    assert rewards.tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert [batch.shape for batch, _ in seen] == [(2, 3, 4, 4), (1, 3, 4, 4)]
    frames = torch.cat([batch for batch, _ in seen])
    assert frames.amax(dim=(1, 2, 3)).tolist() == pytest.approx([0.0, 1.0, 0.2])
    assert [texts for _, texts in seen] == [["a", "b"], ["c"]]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0


class _FakeApxInfModel:
    action_horizon = 10
    action_dim = 32
    num_views = 2
    image_size = 224

    def __init__(self, output_shape=(10, 32)):
        self.output_shape = output_shape
        self.calls = []
        self.closed = False

    def infer_rgb(self, rgb_u8, layout, token_ids, *, noise=None):
        self.calls.append((rgb_u8, layout, token_ids, noise))
        offset = len(self.calls) * 1000
        return (
            np.arange(np.prod(self.output_shape), dtype=np.float32).reshape(
                self.output_shape
            )
            + offset
        )

    def close(self):
        self.closed = True


class _FakeApxInfProcessor:
    def __init__(self):
        self.preprocess_calls = []
        self.postprocess_calls = []

    def preprocess_batch(self, env_obs, *, num_views, image_size):
        self.preprocess_calls.append((env_obs, num_views, image_size))
        prepared = []
        for index in range(len(env_obs["task_descriptions"])):
            prepared.append(
                {
                    "rgb_u8": np.full(
                        (num_views, image_size, image_size, 3),
                        index,
                        dtype=np.uint8,
                    ),
                    "token_ids": np.array([index, index + 1], dtype=np.uint32),
                    "state": np.full(32, index, dtype=np.float32),
                }
            )
        return prepared

    def postprocess_batch(self, normalized_actions, prepared):
        self.postprocess_calls.append((normalized_actions.copy(), prepared))
        return torch.from_numpy(normalized_actions[:, :5, :7].copy())


def _apxinf_model_cfg(**apxinf_overrides):
    apxinf = {
        "action_horizon": 10,
        "num_flow_steps": 5,
        "noise_source": "apxinf",
        "seed": 0,
        **apxinf_overrides,
    }
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": "/not/loaded/in/unit/test",
            "num_action_chunks": 5,
            "action_dim": 7,
            "openpi": {
                "config_name": "pi05_libero",
                "num_steps": 5,
                "noise_method": "flow_sde",
                "noise_level": 0.3,
            },
            "apxinf": apxinf,
        }
    )


def _apxinf_env_obs(batch_size=2):
    return {
        "main_images": torch.zeros(batch_size, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(batch_size, 8, 8, 3, dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.zeros(batch_size, 8),
        "task_descriptions": [f"task {index}" for index in range(batch_size)],
    }


def _apxinf_adapter(*, model=None, processor=None, **apxinf_overrides):
    return OpenPIApxInfAdapter(
        _apxinf_model_cfg(**apxinf_overrides),
        "cpu",
        model=model or _FakeApxInfModel(),
        processor=processor or _FakeApxInfProcessor(),
    )


def test_apxinf_strips_openpi_prompt_padding_before_l1_inference():
    transformed = {
        "tokenized_prompt": np.array([2, 42, 108, 0, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.array([True, True, True, False, False]),
    }

    tokens = _active_token_ids(transformed)

    np.testing.assert_array_equal(tokens, np.array([2, 42, 108], dtype=np.uint32))
    assert tokens.flags.c_contiguous


def test_apxinf_calls_l1_infer_rgb_and_delegates_pre_and_postprocessing():
    model = _FakeApxInfModel()
    processor = _FakeApxInfProcessor()
    adapter = _apxinf_adapter(model=model, processor=processor)
    env_obs = _apxinf_env_obs()

    actions, result = adapter.predict_action_batch(env_obs, mode="eval")

    assert actions.shape == (2, 5, 7)
    assert actions.dtype == torch.float32
    assert processor.preprocess_calls == [(env_obs, 2, 224)]
    assert len(model.calls) == 2
    assert model.calls[0][0].shape == (2, 224, 224, 3)
    assert model.calls[0][0].dtype == np.uint8
    assert model.calls[0][1] == "nhwc"
    assert model.calls[0][2].dtype == np.uint32
    assert model.calls[0][3] is None
    normalized = processor.postprocess_calls[0][0]
    assert normalized.shape == (2, 10, 32)
    assert len(result["apxinf_timing"]) == 2


def test_apxinf_explicit_noise_is_split_and_forwarded_exactly():
    model = _FakeApxInfModel()
    adapter = _apxinf_adapter(model=model, noise_source="observation")
    env_obs = _apxinf_env_obs()
    env_obs["noise"] = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32)

    adapter.predict_action_batch(env_obs)

    np.testing.assert_array_equal(model.calls[0][3], env_obs["noise"][0].numpy())
    np.testing.assert_array_equal(model.calls[1][3], env_obs["noise"][1].numpy())


def test_apxinf_observation_noise_is_required():
    adapter = _apxinf_adapter(noise_source="observation")
    with pytest.raises(ValueError, match="requires env_obs"):
        adapter.predict_action_batch(_apxinf_env_obs())


def test_apxinf_observation_noise_does_not_override_other_noise_sources():
    env_obs = _apxinf_env_obs()
    explicit_noise = torch.full((2, 10, 32), 123.0)
    env_obs["noise"] = explicit_noise

    apxinf_model = _FakeApxInfModel()
    _apxinf_adapter(model=apxinf_model, noise_source="apxinf").predict_action_batch(
        env_obs
    )
    assert all(call[3] is None for call in apxinf_model.calls)

    torch_model = _FakeApxInfModel()
    torch_adapter = _apxinf_adapter(model=torch_model, noise_source="torch")
    torch_adapter.predict_action_batch(env_obs)
    assert all(call[3] is not None for call in torch_model.calls)
    for index, call in enumerate(torch_model.calls):
        assert not np.array_equal(call[3], explicit_noise[index].numpy())


def test_apxinf_torch_noise_is_reproducible_and_has_model_shape():
    model_a = _FakeApxInfModel()
    model_b = _FakeApxInfModel()
    adapter_a = _apxinf_adapter(model=model_a, noise_source="torch", seed=7)
    adapter_b = _apxinf_adapter(model=model_b, noise_source="torch", seed=7)

    adapter_a.predict_action_batch(_apxinf_env_obs())
    adapter_b.predict_action_batch(_apxinf_env_obs())

    assert model_a.calls[0][3].shape == (10, 32)
    np.testing.assert_array_equal(model_a.calls[0][3], model_b.calls[0][3])
    np.testing.assert_array_equal(model_a.calls[1][3], model_b.calls[1][3])


def test_apxinf_rejects_bad_normalized_action_shape():
    adapter = _apxinf_adapter(model=_FakeApxInfModel(output_shape=(10, 7)))
    with pytest.raises(ValueError, match="normalized actions have shape"):
        adapter.predict_action_batch(_apxinf_env_obs(batch_size=1))


def test_apxinf_rejects_mismatched_openpi_and_apxinf_flow_steps():
    with pytest.raises(ValueError, match="must match OpenPI num_steps"):
        _apxinf_adapter(num_flow_steps=10)


def test_apxinf_rejects_training_mode():
    adapter = _apxinf_adapter()
    with pytest.raises(ValueError, match="eval-only"):
        adapter.predict_action_batch(_apxinf_env_obs(), mode="train")


def test_apxinf_close_delegates_to_model():
    model = _FakeApxInfModel()
    adapter = _apxinf_adapter(model=model)
    adapter.close()
    assert model.closed


def test_so101_openpi_transforms_pad_six_joint_contract():
    np = pytest.importorskip("numpy")
    pytest.importorskip("openpi")
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.so101_policy import SO101Inputs

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = SO101Inputs(action_dim=32, model_type=openpi_model.ModelType.PI05)(
        {
            "observation/image": image,
            "observation/state": np.zeros(6, dtype=np.float32),
            "actions": np.zeros((20, 6), dtype=np.float32),
            "prompt": "reach a joint configuration",
        }
    )
    assert result["state"].shape == (32,)
    assert result["actions"].shape == (20, 32)
    assert result["image"]["base_0_rgb"].shape == image.shape
    assert not result["image_mask"]["left_wrist_0_rgb"]


def test_aloha_inputs_mask_wrist_views_missing_from_a_head_only_observation():
    np = pytest.importorskip("numpy")
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.aloha_policy import AlohaInputs

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = AlohaInputs(adapt_to_pi=False)(
        {
            "observation/image": image,
            "observation/wrist_image": None,
            "observation/state": np.zeros(14, dtype=np.float32),
            "prompt": "adjust the bottle",
        }
    )
    assert result["image_mask"]["base_0_rgb"]
    for view in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert not result["image_mask"][view]
        assert result["image"][view].shape == image.shape


def test_so101_dataconfig_repacks_lerobot_task_as_prompt():
    np = pytest.importorskip("numpy")
    pytest.importorskip("openpi")
    from openpi.models.pi0_config import Pi0Config

    from rlinf.models.embodiment.openpi.dataconfig.so101_dataconfig import (
        SO101JointDataConfig,
    )

    data_config = SO101JointDataConfig(repo_id="so101_joint").create(
        assets_dirs=Path("/tmp/nonexistent-openpi-assets"),
        model_config=Pi0Config(pi05=True),
    )
    raw_sample = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "state": np.zeros(6, dtype=np.float32),
        "actions": np.zeros((2, 6), dtype=np.float32),
        "task": "reach a joint configuration",
    }

    repacked = data_config.repack_transforms.inputs[0](raw_sample)

    assert repacked["prompt"] == raw_sample["task"]
    assert "task" not in repacked


def _stub_apxinf_robo(monkeypatch, resolved_tactics=None):
    """Install a fake ``apxinf_robo`` and record what ``_load_model`` asks it for."""
    seen = {}

    def load_bare_model(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return _FakeApxInfModel()

    def resolve_tactics(device, precision, **kwargs):
        seen["resolve"] = {"device": device, "precision": precision, **kwargs}
        return resolved_tactics

    module = ModuleType("apxinf_robo")
    module.load_bare_model = load_bare_model
    engine = ModuleType("apxinf_robo.engine")
    engine.resolve_tactics = resolve_tactics
    module.engine = engine
    monkeypatch.setitem(sys.modules, "apxinf_robo", module)
    monkeypatch.setitem(sys.modules, "apxinf_robo.engine", engine)
    return seen


def test_apxinf_loads_through_the_apxinf_robo_l1_entry_point(monkeypatch):
    seen = _stub_apxinf_robo(monkeypatch)

    OpenPIApxInfAdapter(_apxinf_model_cfg(), "cpu", processor=_FakeApxInfProcessor())

    assert seen["path"] == Path("/not/loaded/in/unit/test")
    kwargs = seen["kwargs"]
    assert kwargs["model"] == "pi05"
    assert kwargs["device"] == "cpu"
    assert kwargs["precision"] == "bf16"
    assert kwargs["action_horizon"] == 10
    assert kwargs["num_flow_steps"] == 5
    assert kwargs["sampling_seed"] == 0
    # Left out so load_bare_model selects the tuned tactics.
    assert "tactics" not in kwargs
    assert "resolve" not in seen


def test_apxinf_a_configured_tactics_file_wins_over_the_default_selection(monkeypatch):
    seen = _stub_apxinf_robo(monkeypatch)

    OpenPIApxInfAdapter(
        _apxinf_model_cfg(tactics="/mine.json"), "cpu", processor=_FakeApxInfProcessor()
    )

    assert seen["kwargs"]["tactics"] == "/mine.json"
    assert "resolve" not in seen


def test_apxinf_an_explicit_weights_file_resolves_tactics_from_the_checkpoint_dir(
    monkeypatch,
):
    seen = _stub_apxinf_robo(monkeypatch, resolved_tactics="/ckpt/tactics.json")

    OpenPIApxInfAdapter(
        _apxinf_model_cfg(checkpoint="/ckpt/model-00001-of-00002.safetensors"),
        "cpu",
        processor=_FakeApxInfProcessor(),
    )

    # The weights file goes to the loader, the directory to the tactics lookup:
    # keying the lookup on the file would miss a checkpoint-local tactics.json.
    assert seen["path"] == Path("/ckpt/model-00001-of-00002.safetensors")
    assert seen["resolve"]["model_dir"] == Path("/not/loaded/in/unit/test")
    assert seen["resolve"]["precision"] == "bf16"
    assert seen["kwargs"]["tactics"] == "/ckpt/tactics.json"


def test_apxinf_a_missing_apxinf_robo_names_what_to_install(monkeypatch):
    # ``None`` in sys.modules is how CPython marks an import as unavailable.
    monkeypatch.setitem(sys.modules, "apxinf_robo", None)

    with pytest.raises(ImportError, match="apxinf_robo"):
        OpenPIApxInfAdapter(
            _apxinf_model_cfg(), "cpu", processor=_FakeApxInfProcessor()
        )


# PhyAI rollout backend. Workers are built through their constructor; only Ray
# placement, the weight-sync transport, and the PhyAI engine are faked.

_PHYAI_INSTALLED = importlib.util.find_spec("phyai") is not None
_requires_phyai = pytest.mark.skipif(
    not _PHYAI_INSTALLED, reason="PhyAI is not installed"
)


def _phyai_cfg(
    *,
    only_eval=False,
    openpi=None,
    phyai=None,
    algorithm=None,
    collect_transitions=False,
    rollout_model=None,
    ckpt_path=None,
):
    model = {
        "model_type": "openpi",
        "model_path": "/checkpoint",
        "precision": "bf16",
        "num_action_chunks": 5,
        "action_dim": 7,
        "num_steps": 3,
        "add_value_head": True,
        "openpi": {
            "config_name": "pi05_libero",
            "action_horizon": 10,
            "model_action_dim": 32,
            "max_token_len": 200,
            "num_images_in_input": 2,
            "value_after_vlm": True,
            **(openpi or {}),
        },
    }
    env = {"rollout_epoch": 1, "total_num_envs": 2, "max_steps_per_rollout_epoch": 10}
    cfg = {
        "runner": {
            "only_eval": only_eval,
            "val_check_interval": -1,
            "ckpt_path": ckpt_path,
        },
        "env": {"group_name": "EnvGroup", "train": env, "eval": dict(env)},
        "actor": {"group_name": "ActorGroup", "model": model},
        "rollout": {
            "group_name": "RolloutGroup",
            "pipeline_stage_num": 1,
            "collect_transitions": collect_transitions,
            "model": {**model, **(rollout_model or {})},
            "phyai": {"plugin": "pi05", **(phyai or {})},
        },
        "weight_syncer": {
            "type": "bucket",
            "bucket": {
                "bucket_size": 1 << 20,
                "bucket_dtype": None,
                "bucket_device": "cpu",
            },
        },
    }
    if algorithm is not None:
        cfg["algorithm"] = algorithm
    cfg = OmegaConf.create(cfg)
    # Hydra-composed configs are in struct mode, so missing keys raise.
    OmegaConf.set_struct(cfg, True)
    return cfg


class _OneGpuPlacement:
    def __init__(self, cfg, cluster):
        del cfg, cluster

    def get_world_size(self, component):
        del component
        return 1


def _build_phyai_worker(monkeypatch, cfg):
    """Build rank 0 of a one-GPU PhyAI rollout group without starting Ray."""

    def init_scheduler_side(worker):
        worker._rank = 0
        worker._world_size = 1
        worker._group_name = "RolloutGroup"
        worker.global_accelerator_ids = [0]
        worker._logger = logging.getLogger("phyai-test")
        worker._stacklevel = 1

    monkeypatch.setattr(Worker, "__init__", init_scheduler_side)
    monkeypatch.setattr(phyai_rollout_worker, "Cluster", lambda: None)
    monkeypatch.setattr(
        phyai_rollout_worker, "HybridComponentPlacement", _OneGpuPlacement
    )
    return PhyAIRolloutWorker(cfg)


class _FakePhyAIEngine:
    """Records what the worker asks of a PhyAI engine and returns ``result``."""

    def __init__(self, args=None):
        self.args = args
        self.kernel_env = os.environ.get("PHYAI_KERNEL_CONFIG")
        self.result = None
        self.requests = []
        self.events = []

    def step(self, request):
        self.requests.append(request)
        return self.result

    def rollout_step(self, request):
        self.requests.append(request)
        return self.result

    def begin_weight_update(self):
        self.events.append("begin")

    def update_weights(self, state_dict):
        self.events.append(("update", state_dict))

    def finish_weight_update(self, version=None):
        self.events.append(("finish", version))
        return SimpleNamespace(loaded=[])

    def abort_weight_update(self):
        self.events.append("abort")


def _phyai_input_transform(sample):
    base = np.asarray(sample["observation/image"], dtype=np.uint8)
    wrist = np.asarray(sample["observation/wrist_image"], dtype=np.uint8)
    state = np.asarray(sample["observation/state"], dtype=np.float64) / 10
    prompt_length = min(len(sample["prompt"]), 8)
    return {
        "image": {
            IMAGE_KEYS[0]: base,
            IMAGE_KEYS[1]: wrist,
            IMAGE_KEYS[2]: np.zeros_like(base),
        },
        "image_mask": {
            IMAGE_KEYS[0]: np.bool_(True),
            IMAGE_KEYS[1]: np.bool_(True),
            IMAGE_KEYS[2]: np.bool_(False),
        },
        "state": state,
        "tokenized_prompt": np.arange(8, dtype=np.int64),
        "tokenized_prompt_mask": np.arange(8) < prompt_length,
    }


def _phyai_output_transform(sample):
    return {"actions": sample["actions"][..., :7] + sample["state"][0]}


def _phyai_env_obs(batch_size=2):
    image_size = batch_size * 224 * 224 * 3
    main_images = torch.arange(image_size, dtype=torch.int64)
    wrist_images = torch.arange(image_size, dtype=torch.int64).flip(0)
    return {
        "main_images": main_images.remainder(251)
        .to(torch.uint8)
        .view(batch_size, 224, 224, 3),
        "wrist_images": wrist_images.remainder(251)
        .to(torch.uint8)
        .view(batch_size, 224, 224, 3),
        "states": torch.arange(batch_size * 8, dtype=torch.float32).view(batch_size, 8),
        "task_descriptions": ["pick", "place"][:batch_size],
    }


def _phyai_rollout_result(batch_size=2):
    return SimpleNamespace(
        actions=torch.arange(batch_size * 10 * 32, dtype=torch.float32).view(
            batch_size, 10, 32
        ),
        chains=torch.arange(batch_size * 4 * 10 * 32, dtype=torch.float32).view(
            batch_size, 4, 10, 32
        ),
        transition_logprobs=torch.arange(
            batch_size * 3 * 10 * 32, dtype=torch.float32
        ).view(batch_size, 3, 10, 32),
        raw_values=torch.tensor([[1.0], [2.0]]),
    )


def _init_phyai_worker(monkeypatch, cfg):
    """Build a PhyAI worker and run ``init_worker`` against a fake engine."""
    import phyai.engine
    import phyai.engine_config
    import phyai.models.pi05.main_pi05

    engines = []

    def make_engine(args):
        engines.append(_FakePhyAIEngine(args))
        return engines[-1]

    # Record the constructor arguments instead of validating them on a CPU host.
    for module, name in [
        (phyai.engine_config, "RuntimeConfig"),
        (phyai.engine_config, "DeviceConfig"),
        (phyai.engine_config, "EngineConfig"),
        (phyai.models.pi05.main_pi05, "PI05Args"),
        (phyai.engine, "EngineArgs"),
    ]:
        monkeypatch.setattr(module, name, SimpleNamespace)
    monkeypatch.setattr(phyai.engine, "Engine", make_engine)

    transform_calls = []

    def build_transforms(model_path, config_name, data_kwargs=None):
        transform_calls.append((model_path, config_name, data_kwargs))
        return [_phyai_input_transform], [_phyai_output_transform]

    # The real builder reads normalization stats from the checkpoint directory.
    monkeypatch.setattr(
        phyai_rollout_worker, "build_openpi_transforms", build_transforms
    )
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    worker = _build_phyai_worker(monkeypatch, cfg)
    worker.init_worker()
    return worker, engines[0], transform_calls


def test_phyai_worker_rejects_an_unknown_engine_plugin(monkeypatch):
    with pytest.raises(NotImplementedError, match="pi05"):
        _build_phyai_worker(monkeypatch, _phyai_cfg(phyai={"plugin": "unknown"}))


def test_phyai_eval_worker_accepts_a_config_without_algorithm(monkeypatch):
    # Standalone evaluation configs have no algorithm section.
    worker = _build_phyai_worker(monkeypatch, _phyai_cfg(only_eval=True))

    assert worker.only_eval
    assert not worker.enable_rlt
    assert worker.weight_syncer is None


@pytest.mark.parametrize(
    ("collect_transitions", "loss_type", "expected"),
    [
        (False, "actor_critic", False),
        (True, "actor_critic", True),
        (True, "rlt_ac", False),
    ],
)
def test_phyai_worker_sends_raw_transition_obs_only_when_the_actor_reads_them(
    monkeypatch, collect_transitions, loss_type, expected
):
    cfg = _phyai_cfg(
        collect_transitions=collect_transitions,
        algorithm={"loss_type": loss_type},
    )
    worker = _build_phyai_worker(monkeypatch, cfg)

    assert worker.collect_raw_transition_obs is expected


def test_phyai_evaluate_routes_each_pipeline_stage_through_the_env_channel(
    monkeypatch,
):
    class _AsyncResult:
        def __init__(self, value):
            self.value = value

        async def async_wait(self):
            return self.value

    cfg = _phyai_cfg(only_eval=True)
    cfg.rollout.pipeline_stage_num = 2
    cfg.env.eval.total_num_envs = 4
    cfg.env.eval.max_steps_per_rollout_epoch = 5
    worker = _build_phyai_worker(monkeypatch, cfg)

    recv_calls = []
    send_calls = []
    predict_calls = []

    def recv_from(**kwargs):
        recv_calls.append(kwargs)
        stage_id = kwargs["route_key"]
        chunks = [
            {
                "obs": {
                    "states": torch.tensor([[stage_id * 2 + offset]]),
                    "task_descriptions": [f"stage-{stage_id}-{offset}"],
                },
                "final_obs": {"states": torch.tensor([[99]])},
                "rlt_switch_flags": torch.ones(1, dtype=torch.bool),
                "intervene_flags": torch.ones(1, dtype=torch.bool),
            }
            for offset in range(2)
        ]
        assert all(kwargs["infer_batch_size_fn"](chunk) == 1 for chunk in chunks)
        merged = kwargs["merge_fn"](chunks)
        assert set(merged) == {"obs"}
        return _AsyncResult(merged)

    def predict(obs, mode):
        predict_calls.append((obs, mode))
        return torch.zeros(2, 1, 1), {}

    # The channel transport and inference are out of scope here.
    monkeypatch.setattr(worker, "recv_from", recv_from)
    monkeypatch.setattr(worker, "send_to", lambda **kwargs: send_calls.append(kwargs))
    monkeypatch.setattr(worker, "predict", predict)

    asyncio.run(inspect.unwrap(PhyAIRolloutWorker.evaluate)(worker, "input", "output"))

    assert [call["route_key"] for call in recv_calls] == [0, 1]
    assert all(call["group_name"] == "EnvGroup" for call in recv_calls)
    assert all(call["channel"] == "input" for call in recv_calls)
    assert all(call["tag"] == "eval_rollout_results" for call in recv_calls)
    assert all(call["batch_size"] == 2 for call in recv_calls)
    assert len(predict_calls) == 2
    for stage_id, (obs, mode) in enumerate(predict_calls):
        assert mode == "eval"
        assert torch.equal(
            obs["states"], torch.tensor([[stage_id * 2], [stage_id * 2 + 1]])
        )
        assert obs["task_descriptions"] == [
            f"stage-{stage_id}-0",
            f"stage-{stage_id}-1",
        ]
    assert [call["route_key"] for call in send_calls] == [0, 1]
    assert all(call["group_name"] == "EnvGroup" for call in send_calls)
    assert all(call["channel"] == "output" for call in send_calls)
    assert all(call["tag"] == "eval_rollout_results" for call in send_calls)
    assert all(call["batch_size"] == 2 for call in send_calls)
    assert all(call["data"].device.type == "cpu" for call in send_calls)
    assert all(call["data"].is_contiguous() for call in send_calls)


def _phyai_worker_with_engine(monkeypatch):
    worker = _build_phyai_worker(monkeypatch, _phyai_cfg())
    engine = _FakePhyAIEngine()
    # Stands in for init_worker, which needs the PhyAI package.
    worker._engine = engine
    worker._weight_target = phyai_rollout_worker._PhyAIWeightTarget(engine)
    return worker, engine


def _broadcast_buckets(monkeypatch, worker, buckets):
    """Deliver ``buckets`` in order through the worker's weight broadcast."""

    class _Pending:
        def __init__(self, value):
            self.value = value

        async def async_wait(self):
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    pending = iter(buckets)
    monkeypatch.setattr(
        worker, "broadcast", lambda *args, **kwargs: _Pending(next(pending))
    )


def _bucket_header(total_buckets, version):
    return {
        "total_buckets": torch.tensor(total_buckets),
        "syncer_version": torch.tensor(version),
    }


@pytest.mark.parametrize("has_accelerator", [False, True])
def test_phyai_weight_sync_loads_every_bucket_in_phyai_layout(
    monkeypatch, has_accelerator
):
    worker, engine = _phyai_worker_with_engine(monkeypatch)
    empty_cache = MagicMock()
    worker.torch_platform = (
        SimpleNamespace(empty_cache=empty_cache) if has_accelerator else None
    )
    qkv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    gating = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    value_bias = torch.ones(4)
    _broadcast_buckets(
        monkeypatch,
        worker,
        [
            {
                **_bucket_header(total_buckets=2, version=7),
                "model.img.encoder.layers.0.attn.in_proj_weight": qkv,
            },
            {
                "model.llm.layers.0.mlps.1.w_gating": gating,
                "value_head.mlp.0.bias": value_bias,
            },
        ],
    )

    asyncio.run(inspect.unwrap(PhyAIRolloutWorker.sync_model_from_actor)(worker))

    assert engine.events[0] == "begin"
    assert [event[0] for event in engine.events[1:3]] == ["update", "update"]
    assert engine.events[3] == ("finish", 7)
    assert worker.version == 7
    weights = {**engine.events[1][1], **engine.events[2][1]}
    vision_prefix = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "encoder.layers.0.self_attn"
    )
    assert torch.equal(weights[f"{vision_prefix}.q_proj.weight"], qkv[:2])
    assert torch.equal(weights[f"{vision_prefix}.k_proj.weight"], qkv[2:4])
    assert torch.equal(weights[f"{vision_prefix}.v_proj.weight"], qkv[4:])
    expert_prefix = "paligemma_with_expert.gemma_expert.model.layers.0.mlp"
    assert torch.equal(weights[f"{expert_prefix}.gate_proj.weight"], gating[0].T)
    assert torch.equal(weights[f"{expert_prefix}.up_proj.weight"], gating[1].T)
    assert weights["value_head.mlp.0.bias"] is value_bias
    if has_accelerator:
        empty_cache.assert_called_once_with()
    else:
        empty_cache.assert_not_called()


def test_phyai_weight_sync_aborts_and_keeps_the_version_when_a_bucket_fails(
    monkeypatch,
):
    worker, engine = _phyai_worker_with_engine(monkeypatch)
    _broadcast_buckets(
        monkeypatch,
        worker,
        [
            {
                **_bucket_header(total_buckets=2, version=7),
                "value_head.mlp.0.bias": torch.ones(4),
            },
            ConnectionError("actor rank 0 went away"),
        ],
    )

    with pytest.raises(ConnectionError, match="went away"):
        asyncio.run(inspect.unwrap(PhyAIRolloutWorker.sync_model_from_actor)(worker))

    assert [
        event if isinstance(event, str) else event[0] for event in engine.events
    ] == [
        "begin",
        "update",
        "abort",
    ]
    assert worker.version == 0


@_requires_phyai
@pytest.mark.parametrize("only_eval", [False, True])
@pytest.mark.parametrize(
    "kernel_config",
    [
        None,
        "~/phyai-kernel.yaml",
        {
            "schema": "phyai.kernel/v1",
            "profile": "static",
            "defaults": {"fallback": "error"},
            "overrides": [],
        },
    ],
)
def test_phyai_init_builds_the_engine_from_the_model_config(
    monkeypatch, tmp_path, only_eval, kernel_config
):
    monkeypatch.setenv("PHYAI_KERNEL_CONFIG", "/environment/kernel.yaml")
    trained_weight = torch.ones(4, 4)
    ckpt_path = tmp_path / "trained.pt"
    torch.save({"model.state_proj.weight": trained_weight}, ckpt_path)
    cfg = _phyai_cfg(
        only_eval=only_eval,
        ckpt_path=str(ckpt_path),
        phyai={"params_dtype": "bf16", "kernel_config": kernel_config},
        rollout_model={"model_path": "/rollout-checkpoint", "num_action_chunks": 10},
    )

    worker, engine, transform_calls = _init_phyai_worker(monkeypatch, cfg)

    # An inline kernel policy hides the environment override from the engine only.
    assert engine.kernel_env == (
        None if isinstance(kernel_config, dict) else "/environment/kernel.yaml"
    )
    assert os.environ["PHYAI_KERNEL_CONFIG"] == "/environment/kernel.yaml"
    engine_args = engine.args
    if isinstance(kernel_config, dict):
        assert engine_args.config.kernel.policy_config == kernel_config
        assert engine_args.config.kernel.config_path is None
    else:
        expected_path = str(Path(kernel_config).expanduser()) if kernel_config else None
        assert engine_args.config.kernel.config_path == expected_path
        assert engine_args.config.kernel.policy_config is None
    plugin_args = engine_args.plugin_args
    plugin_cfg = plugin_args.config
    assert engine_args.config.device.params_dtype is torch.bfloat16
    assert plugin_args.checkpoint_dir == "/rollout-checkpoint"
    assert plugin_cfg.chunk_size == 10
    assert plugin_cfg.max_action_dim == 32
    assert plugin_cfg.num_inference_steps == 3
    assert plugin_cfg.critic_action_chunk is None
    assert plugin_cfg.add_value_head
    assert plugin_cfg.value_after_vlm
    # Training keeps the actor's model and takes only the path from rollout.model.
    assert worker.model_cfg.num_action_chunks == (10 if only_eval else 5)
    assert transform_calls == [("/rollout-checkpoint", "pi05_libero", None)]
    assert engine.events[0] == "begin"
    assert engine.events[1][0] == "update"
    assert torch.equal(engine.events[1][1]["state_proj.weight"], trained_weight)
    assert engine.events[2] == ("finish", None)


@_requires_phyai
def test_phyai_eval_sends_the_preprocessed_observation_to_the_engine(monkeypatch):
    worker, engine, _ = _init_phyai_worker(monkeypatch, _phyai_cfg(only_eval=True))
    env_obs = _phyai_env_obs()
    engine.result = _phyai_rollout_result().actions

    actions, result = inspect.unwrap(PhyAIRolloutWorker.predict)(
        worker, env_obs, mode="eval"
    )

    (request,) = engine.requests
    assert torch.equal(request.input_ids, torch.arange(8).expand(2, -1))
    assert torch.equal(request.lang_lens, torch.tensor([4, 5]))
    assert request.pixel_values.shape[1] == 2
    assert request.pixel_values.dtype is torch.float32
    assert request.pixel_values.is_contiguous()
    for index, images in enumerate((env_obs["main_images"], env_obs["wrist_images"])):
        normalized = images.to(torch.float32) / 255 * 2 - 1
        assert torch.equal(
            request.pixel_values[:, index], normalized.permute(0, 3, 1, 2)
        )
    normalized_state = env_obs["states"] / 10
    assert torch.equal(
        actions, engine.result[:, :5, :7] + normalized_state[:, :1, None]
    )
    assert result["prev_logprobs"] is None
    assert result["prev_values"] is None


@_requires_phyai
def test_phyai_train_predict_returns_the_actor_replay_contract(monkeypatch):
    batch_size = 2
    cfg = _phyai_cfg(
        openpi={"noise_method": "flow_sde", "noise_level": 0.5, "ignore_last": True}
    )
    worker, engine, _ = _init_phyai_worker(monkeypatch, cfg)
    env_obs = _phyai_env_obs(batch_size)
    rollout_result = _phyai_rollout_result(batch_size)
    engine.result = rollout_result

    def select_denoise_step(low, high):
        # ignore_last leaves the final of the three steps out of the draw.
        assert (low, high) == (0, 1)
        return 1

    monkeypatch.setattr(phyai_rollout_worker.random, "randint", select_denoise_step)

    actions, result = inspect.unwrap(PhyAIRolloutWorker.predict)(
        worker, env_obs, mode="train"
    )

    normalized_state = env_obs["states"] / 10
    model_actions = rollout_result.actions
    expected_actions = model_actions[:, :5, :7] + normalized_state[:, :1, None]
    (request,) = engine.requests
    assert request.pixel_values.dtype is torch.float32
    assert request.noise is None
    assert request.step_noise is None
    torch.testing.assert_close(
        request.sigmas, torch.tensor([[0.0, math.sqrt(0.5), 0.0]]).expand(2, -1)
    )
    assert actions.shape == (batch_size, 5, 7)
    assert torch.equal(actions, expected_actions)
    assert torch.equal(
        result["prev_logprobs"], rollout_result.transition_logprobs[:, 1, :5, :7]
    )
    assert torch.equal(result["prev_values"], torch.tensor([[1.0], [2.0]]))
    forward_inputs = result["forward_inputs"]
    assert torch.equal(forward_inputs["chains"], rollout_result.chains)
    assert forward_inputs["denoise_inds"].eq(1).all()
    assert torch.equal(
        forward_inputs["obs_state"],
        torch.nn.functional.pad(normalized_state, (0, 24)),
    )
    assert torch.equal(
        forward_inputs["tokenized_prompt"], torch.arange(8).expand(2, -1)
    )
    assert torch.equal(
        forward_inputs["tokenized_prompt_mask"],
        torch.arange(8)[None] < torch.tensor([[4], [5]]),
    )
    assert torch.equal(forward_inputs["action"], actions.reshape(batch_size, -1))
    assert torch.equal(
        forward_inputs["model_action"], model_actions.reshape(batch_size, -1)
    )
    for key, image in zip(
        IMAGE_KEYS,
        (
            env_obs["main_images"],
            env_obs["wrist_images"],
            torch.zeros_like(env_obs["main_images"]),
        ),
        strict=True,
    ):
        assert torch.equal(
            forward_inputs[f"obs_image__{key}"], image.to(torch.float32) / 255 * 2 - 1
        )
        assert torch.equal(
            forward_inputs[f"obs_image_mask__{key}"],
            torch.full((batch_size,), key != IMAGE_KEYS[2]),
        )


@_requires_phyai
def test_phyai_train_predict_defaults_to_deterministic_denoising(monkeypatch):
    worker, engine, _ = _init_phyai_worker(monkeypatch, _phyai_cfg())
    engine.result = _phyai_rollout_result()

    inspect.unwrap(PhyAIRolloutWorker.predict)(worker, _phyai_env_obs(), mode="train")

    (request,) = engine.requests
    assert request.sigmas.eq(0).all()


@pytest.fixture
def phyai_example_cfg(monkeypatch):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")


@pytest.mark.parametrize(
    ("path", "value"), [("env.train.chunk", 25), ("env.train.action_dim", 7)]
)
def test_world_model_env_must_match_the_policy_action_shape(monkeypatch, path, value):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="bwm_robotwin_adjust_bottle_ppo_openpi_pi05")
    OmegaConf.update(cfg, path, value)

    with pytest.raises(ValueError, match=f"Expected {path} to equal"):
        validate_embodied_cfg(cfg)


def test_phyai_example_config_validates_and_shares_the_actor_checkpoint(
    phyai_example_cfg,
):
    validate_embodied_cfg(phyai_example_cfg)

    assert (
        phyai_example_cfg.rollout.model.model_path
        == phyai_example_cfg.actor.model.model_path
    )


def test_phyai_training_requires_bucket_weight_sync(phyai_example_cfg):
    phyai_example_cfg.weight_syncer.type = "patch"

    with pytest.raises(AssertionError, match="weight_syncer.type='bucket'.*'patch'"):
        validate_embodied_cfg(phyai_example_cfg)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("algorithm.loss_type", "embodied_dagger"),
        ("algorithm.loss_type", "rlt_ac"),
        ("algorithm.loss_type", "rlt_td3"),
        ("algorithm.adv_type", "opd"),
    ],
)
def test_phyai_training_rejects_unsupported_algorithms(phyai_example_cfg, path, value):
    if path == "algorithm.adv_type":
        phyai_example_cfg.algorithm.recompute_logprobs = False
    OmegaConf.update(phyai_example_cfg, path, value)

    with pytest.raises(AssertionError, match=f"rollout_backend='phyai', got '{value}'"):
        validate_embodied_cfg(phyai_example_cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("actor.model.add_value_head", False, "add_value_head=True"),
        ("actor.model.openpi.value_after_vlm", False, "value_after_vlm=True"),
        ("actor.model.openpi.value_vlm_mode", "last_token", "'last_token'"),
        ("actor.model.openpi.joint_logprob", True, "joint_logprob=False"),
        ("actor.model.openpi.noise_method", "flow_noise", "got 'flow_noise'"),
        ("actor.model.openpi.noise_level", -0.1, "noise_level.*got -0.1"),
        ("actor.model.openpi.noise_level", float("nan"), "noise_level.*got nan"),
        ("actor.model.openpi.num_images_in_input", 1, "num_images_in_input=2"),
    ],
)
def test_phyai_training_validates_the_actor_model(
    phyai_example_cfg, path, value, message
):
    OmegaConf.update(phyai_example_cfg, path, value)

    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(phyai_example_cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("pi05", False, "model.pi05=True"),
        ("openpi.paligemma_variant", "gemma_3b", "paligemma_variant='gemma_2b'"),
        (
            "openpi.action_expert_variant",
            "gemma_1b",
            "action_expert_variant='gemma_300m'",
        ),
        ("openpi.num_images_in_input", 4, "num_images_in_input in \\[1, 3\\]"),
    ],
)
def test_phyai_eval_validates_the_engine_plugin_model(
    phyai_example_cfg, path, value, message
):
    phyai_example_cfg.runner.only_eval = True
    OmegaConf.update(phyai_example_cfg.rollout.model, path, value)

    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(phyai_example_cfg)


def test_phyai_eval_requires_a_nonempty_openpi_config_name(phyai_example_cfg):
    phyai_example_cfg.runner.only_eval = True
    phyai_example_cfg.rollout.model.openpi.config_name = ""

    with pytest.raises(ValueError, match="rollout.model.openpi.config_name"):
        validate_embodied_cfg(phyai_example_cfg)


def _cosmos3_sglang_cfg(batching_max_size=None):
    server = (
        {} if batching_max_size is None else {"batching_max_size": batching_max_size}
    )
    return OmegaConf.create(
        {
            "rollout": {
                "model": {"num_action_chunks": 16},
                "sglang": {"server": server},
            },
            "env": {"eval": {"env_type": "libero"}},
        }
    )


def test_cosmos3_sglang_requests_group_envs_by_prompt_up_to_the_server_batch():
    # The Cosmos3 server rejects a batch whose prompts tokenize to different
    # lengths, and one larger than its --batching-max-size.
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(batching_max_size=2), rank=0)
    tasks = ["pick the bowl", "open the drawer", "pick the bowl", "pick the bowl"]

    groups = adapter.request_groups({"task_descriptions": tasks})

    assert groups == [[0, 2], [3], [1]]


def test_cosmos3_sglang_requests_default_to_one_env_like_sglang():
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(), rank=0)

    groups = adapter.request_groups({"task_descriptions": ["a", "a", "a"]})

    assert groups == [[0], [1], [2]]


def test_sglang_request_groups_round_trip_to_env_order():
    from rlinf.models.embodiment.sglang_adapter import (
        gather_env_rows,
        select_env_rows,
    )

    env_obs = {
        "main_images": torch.arange(5).view(5, 1),
        "task_descriptions": ["a", "b", "a", "a", "b"],
        "wrist_images": None,
        "_rlinf_stage_id": 1,
    }
    groups = [[0, 2], [3], [1, 4]]

    parts = [select_env_rows(env_obs, group) for group in groups]
    assert parts[2]["task_descriptions"] == ["b", "b"]
    assert parts[0]["_rlinf_stage_id"] == 1 and parts[0]["wrist_images"] is None

    # Each "request" answers with its envs' rows; gathering restores env order.
    order = [index for group in groups for index in group]
    actions = gather_env_rows([p["main_images"] * 10 for p in parts], order)
    info = gather_env_rows([{"x": {"y": p["main_images"]}} for p in parts], order)
    assert torch.equal(actions, torch.arange(5).view(5, 1) * 10)
    assert torch.equal(info["x"]["y"], torch.arange(5).view(5, 1))


def test_cosmos3_sglang_response_keeps_every_env_in_input_order():
    # The action envelope carries one ``data`` entry per env of the request.
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(), rank=0)

    def entry(input_index, gripper):
        values = np.zeros((16, 12), dtype=np.float32)
        values[:, 3] = 1.0  # rot6d columns (1, 0, 0) and (0, 1, 0): no rotation
        values[:, 7] = 1.0
        values[:, 9] = gripper
        return {"input_index": input_index, "action": {"values": values}}

    actions, _ = adapter.parse_response(
        {"data": [entry(1, 0.5), entry(0, -0.5)]}, state={}
    )

    assert actions.shape == (2, 16, 7)
    assert torch.all(actions[0, :, 6] == -0.5)
    assert torch.all(actions[1, :, 6] == 0.5)


def _gr00t_npu_patch_modules():
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import npu_patches as n1d5
    from rlinf.models.embodiment.gr00t.gr00t_n1d6 import npu_patches as n1d6
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import npu_patches as n1d7

    return {"n1d5": n1d5, "n1d6": n1d6, "n1d7": n1d7}


def test_gr00t_npu_patches_are_inert_off_ascend(monkeypatch):
    # Every GR00T version registers its Ascend patches through the same pair of
    # calls, and off NPU both do nothing, so a CUDA run is untouched.
    from rlinf.scheduler import AcceleratorType
    from rlinf.utils.patcher import Patcher

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NO_ACCEL)
    for version, module in _gr00t_npu_patch_modules().items():
        Patcher.clear()
        state = module.apply_npu_patches(Patcher)
        assert state is None, version
        module.restore_npu_patches(Patcher, state)
        assert Patcher._mappings_dict == {}, version


def test_gr00t_npu_fused_kernel_paths_resolve_for_every_version():
    # The Patcher names replacements by string path, so a moved or renamed
    # kernel wrapper has to fail here rather than at model construction on NPU.
    from rlinf.utils.patcher import Patcher

    for path in (
        "rlinf.models.embodiment.gr00t.npu_common.npu_rmsnorm_forward",
        "rlinf.models.embodiment.gr00t.npu_common.npu_apply_rotary_pos_emb",
        "rlinf.models.embodiment.gr00t.gr00t_n1d5.npu_patches.npu_rmsnorm_forward",
        "rlinf.models.embodiment.gr00t.gr00t_n1d5.npu_patches.npu_apply_rotary_pos_emb",
    ):
        _, obj = Patcher._get_parent_obj_and_obj(path)
        assert callable(obj), path


def test_npu_flash_attention_binding_fills_only_missing_names(monkeypatch):
    # GR00T's vendored SigLIP2 attention leaves its flash-attention names
    # undefined on Ascend; binding supplies them without displacing a module
    # that found real kernels.
    from rlinf.models.embodiment.gr00t import npu_common

    def fake_symbols():
        return {"flash_attn_varlen_func": "npu", "_flash_supports_window_size": False}

    monkeypatch.setattr(npu_common, "_npu_flash_attention_symbols", fake_symbols)

    vendored = ModuleType("modeling_siglip2")
    npu_common.bind_npu_flash_attention(vendored)
    assert vendored.flash_attn_varlen_func == "npu"
    assert vendored._flash_supports_window_size is False

    with_real_kernels = ModuleType("modeling_siglip2")
    with_real_kernels.flash_attn_varlen_func = "cuda"
    npu_common.bind_npu_flash_attention(with_real_kernels)
    assert with_real_kernels.flash_attn_varlen_func == "cuda"


def test_npu_flash_attention_binding_reaches_a_later_import(monkeypatch, tmp_path):
    # The vendored module is loaded by trust_remote_code after the patches are
    # registered, so the binding has to apply as it is imported, and stop once
    # the installer's uninstall callable runs.
    from rlinf.models.embodiment.gr00t import npu_common

    monkeypatch.setattr(
        npu_common, "_npu_flash_attention_symbols", lambda: {"flash_attn_func": "npu"}
    )
    (tmp_path / "modeling_siglip2.py").write_text("flash_attn_func = None\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    stop = npu_common.install_npu_flash_attention("modeling_siglip2")
    try:
        sys.modules.pop("modeling_siglip2", None)
        import modeling_siglip2

        assert modeling_siglip2.flash_attn_func == "npu"
    finally:
        stop()
        sys.modules.pop("modeling_siglip2", None)

    import modeling_siglip2 as unbound

    assert unbound.flash_attn_func is None


def test_npu_flash_attention_cache_is_inert_off_ascend(monkeypatch):
    from rlinf.models.embodiment.gr00t.npu_common import (
        install_npu_flash_attention_cache,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NO_ACCEL)
    assert not install_npu_flash_attention_cache()


@pytest.mark.parametrize("padded", [False, True])
def test_npu_flash_attention_cache_preserves_gradients_and_mask_updates(
    monkeypatch, padded
):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    from transformers import modeling_flash_attention_utils as flash_utils
    from transformers.integrations import npu_flash_attention as native_flash
    from transformers.models.qwen3 import Qwen3Config, Qwen3Model

    if hasattr(native_flash, "get_attn_mask_npu"):
        pytest.skip("Legacy Transformers NPU FlashAttention required")
    from rlinf.models.embodiment.gr00t.npu_common import (
        install_npu_flash_attention_cache,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    monkeypatch.setattr(
        flash_utils, "flash_attn_func", native_flash.npu_flash_attn_func
    )
    monkeypatch.setattr(
        flash_utils, "flash_attn_varlen_func", native_flash.npu_flash_attn_varlen_func
    )
    original = (
        flash_utils._get_unpad_data,
        flash_utils.flash_attn_func,
        flash_utils.flash_attn_varlen_func,
    )
    for name, value in zip(
        ("_get_unpad_data", "flash_attn_func", "flash_attn_varlen_func"), original
    ):
        monkeypatch.setattr(flash_utils, name, value)
    assert install_npu_flash_attention_cache()
    assert install_npu_flash_attention_cache()
    cached = (
        flash_utils._get_unpad_data,
        flash_utils.flash_attn_func,
        flash_utils.flash_attn_varlen_func,
    )
    torch.manual_seed(1234)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen3Model(config).to(device="npu", dtype=torch.bfloat16).eval()
    ids = torch.randint(1, 128, (2, 32), device="npu")
    mask = torch.ones_like(ids)
    if padded:
        mask[1, :4] = 0
        mask[0, -2:] = 0
    upstream = torch.randn(2, 32, 128, device="npu") / (2 * 32 * 128)
    for updated in (False, True):
        if updated:
            mask[1, :4] = 1 - mask[1, :4]
            with torch.no_grad():
                model.layers[0].self_attn.q_proj.weight.add_(0.001)
        outputs, gradients = [], []
        for functions in (original, cached):
            for name, value in zip(
                ("_get_unpad_data", "flash_attn_func", "flash_attn_varlen_func"),
                functions,
            ):
                setattr(flash_utils, name, value)
            model.zero_grad(set_to_none=True)
            output = model(input_ids=ids, attention_mask=mask, use_cache=False)
            (output.last_hidden_state.float() * upstream).sum().backward()
            outputs.append(output.last_hidden_state.detach().clone())
            gradients.append(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            )
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        assert gradients[0].keys() == gradients[1].keys() and gradients[0]
        for name, expected in gradients[0].items():
            torch.testing.assert_close(gradients[1][name], expected, rtol=0, atol=0)


def test_openpi_kernels_compile_only_where_inductor_can_run(monkeypatch):
    # OpenPI's fused kernels go through Inductor, which needs Triton. Ascend and
    # MUSA venvs have none, and compiling there used to fail at the first
    # forward, so absence of Triton has to read as "do not compile".
    from rlinf.models.embodiment.openpi.modules import utils

    real_find_spec = importlib.util.find_spec

    def without_triton(name, *args, **kwargs):
        return None if name == "triton" else real_find_spec(name, *args, **kwargs)

    utils.inductor_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", without_triton)
    try:
        assert utils.inductor_available() is False
    finally:
        utils.inductor_available.cache_clear()

    # An explicit choice still wins over the platform default, in both directions.
    utils.set_torch_compile(True)
    assert utils.torch_compile_enabled() is True
    utils.set_torch_compile(False)
    assert utils.torch_compile_enabled() is False


def test_npu_sdpa_context_materializes_a_broadcast_mask():
    # Ascend's attention kernel rejects a mask whose query dimension is 1, which
    # CUDA broadcasts. GR00T's DiT builds exactly that, so the context has to
    # expand it to the query length and leave every other mask alone.
    from rlinf.models.embodiment.gr00t import npu_common

    seen = {}

    def record(query, key, value, attn_mask=None, *args, **kwargs):
        seen["mask_shape"] = None if attn_mask is None else tuple(attn_mask.shape)
        return torch.zeros_like(query)

    original = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = record
    try:
        q = torch.zeros(2, 32, 51, 64)
        kv = torch.zeros(2, 32, 570, 64)
        with npu_common.npu_sdpa_context():
            torch.nn.functional.scaled_dot_product_attention(
                q, kv, kv, attn_mask=torch.zeros(2, 32, 1, 570)
            )
            assert seen["mask_shape"] == (2, 32, 51, 570)

            # A mask that already spans the queries is passed through untouched.
            torch.nn.functional.scaled_dot_product_attention(
                q, kv, kv, attn_mask=torch.zeros(2, 32, 51, 570)
            )
            assert seen["mask_shape"] == (2, 32, 51, 570)

            torch.nn.functional.scaled_dot_product_attention(q, kv, kv)
            assert seen["mask_shape"] is None
        # The context restores what it replaced.
        assert torch.nn.functional.scaled_dot_product_attention is record
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


@pytest.mark.parametrize(
    ("batch", "frames", "channel", "grid_h", "grid_w", "patch", "merge", "t_p"),
    [
        (2, 2, 3, 4, 4, 14, 2, 2),
        (1, 4, 3, 8, 6, 14, 2, 2),
        (3, 1, 3, 2, 2, 16, 1, 1),
        (2, 2, 3, 6, 4, 7, 2, 1),
    ],
)
def test_qwen_patchify_matches_upstream_10d_layout(
    batch, frames, channel, grid_h, grid_w, patch, merge, t_p
):
    """The staged <=8-dim packing reproduces the upstream 10-dim chain."""
    from rlinf.models.embodiment.qwen_vl.npu_patchify import _patchify_qwen_vl

    torch.manual_seed(0)
    height, width = grid_h * patch, grid_w * patch
    patches = torch.randn(batch, frames, channel, height, width)
    grid_t = frames // t_p

    upstream = patches.view(
        batch,
        grid_t,
        t_p,
        channel,
        grid_h // merge,
        merge,
        patch,
        grid_w // merge,
        merge,
        patch,
    )
    upstream = upstream.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    upstream = upstream.reshape(
        batch, grid_t * grid_h * grid_w, channel * t_p * patch * patch
    )

    staged = _patchify_qwen_vl(patches, patch, t_p, merge)
    assert staged.shape == upstream.shape
    torch.testing.assert_close(staged, upstream)


def test_gr00t_n1d6_batched_eagle_processor_aligns_frames_to_patch_factor(monkeypatch):
    from PIL import Image

    class FakeCollator:
        pass

    class FakeProcessor:
        pass

    processing_mod = ModuleType("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6")
    processing_mod.Gr00tN1d6DataCollator = FakeCollator
    processing_mod.Gr00tN1d6Processor = FakeProcessor
    for name in ("gr00t", "gr00t.model", "gr00t.model.gr00t_n1d6"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, processing_mod.__name__, processing_mod)

    mod_name = "rlinf.models.embodiment.gr00t.gr00t_n1d6.device_transform"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    device_transform = importlib.import_module(mod_name)

    proc = FakeProcessor()
    proc.collator = FakeCollator()
    proc.processor = SimpleNamespace(
        pixels_per_token=784,
        apply_chat_template=lambda conv, **kw: f"prompt:{len(conv[0]['content'])}",
    )
    device_transform.BatchedEagleProcessor.upgrade(proc)
    proc.set_device("cpu")

    # 256x256 is not divisible by sqrt(784)=28 (or SigLIP2 patch_size=14);
    # smart_resize rounds it to 252x252 (9*28 = 18*14) matching PIL resize.
    raw_256 = np.arange(2 * 3 * 256 * 256, dtype=np.uint8).reshape(2, 3, 256, 256)
    out_256 = proc._apply_vlm_processing(raw_256, "pick up bowl")["vlm_content"]
    assert out_256["text"] == "prompt:3"
    assert len(out_256["images"]) == 2
    expected_0 = np.transpose(
        np.asarray(
            Image.fromarray(np.transpose(raw_256[0], (1, 2, 0))).resize((252, 252))
        ),
        (2, 0, 1),
    )
    assert out_256["images"][0].shape == (3, 252, 252)
    torch.testing.assert_close(out_256["images"][0], torch.from_numpy(expected_0))

    # 224x224 is already a multiple of 28 and passes through without resizing.
    raw_224 = np.zeros((1, 3, 224, 224), dtype=np.uint8)
    out_224 = proc._apply_vlm_processing(raw_224, "pick up bowl")["vlm_content"]
    assert out_224["images"][0].shape == (3, 224, 224)


def test_psi0_rtc_prefix_survives_the_last_denoise_step():
    from rlinf.models.embodiment.psi0.psi0_sampler import (
        Psi0StochasticTransitionSampler,
    )

    class Scheduler:
        def set_timesteps(self, num_inference_steps, device=None):
            self.sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
            self.timesteps = self.sigmas[:-1] * 1000

    torch.manual_seed(0)
    batch_size, horizon, action_dim, overlap = 1, 30, 4, 6
    condition_actions = torch.zeros(batch_size, horizon, action_dim)
    condition_actions[:, :overlap] = torch.randn(batch_size, overlap, action_dim)
    condition_mask = torch.zeros(batch_size, horizon, dtype=torch.bool)
    condition_mask[:, :overlap] = True

    plan, transition = Psi0StochasticTransitionSampler().sample(
        scheduler=Scheduler(),
        num_inference_steps=10,
        initial_noise=torch.randn(batch_size, horizon, action_dim),
        # A non-zero velocity on the prefix would move it on every Euler step.
        velocity_fn=lambda samples, timestep: torch.ones_like(samples),
        condition_actions=condition_actions,
        condition_mask=condition_mask,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(plan[:, :overlap], condition_actions[:, :overlap])
    assert not transition.sample_mask[:, :overlap].any()
    assert torch.all(transition.logprobs[:, :overlap] == 0)


@pytest.mark.parametrize("step", [0, 4, 9])
def test_lingbotvlav2_sde_has_finite_first_and_last_step_density(step):
    from rlinf.models.embodiment.lingbotvlav2.utils import (
        flow_sde_transition,
        gaussian_logprob,
    )

    x = torch.zeros(2, 50, 55)
    velocity = torch.ones_like(x, requires_grad=True)
    times = torch.linspace(1, 0, 11)
    time, delta = times[step], times[step] - times[step + 1]
    mean, std = flow_sde_transition(x, velocity, time, delta, 0.3)
    assert torch.isfinite(mean).all() and std > 0
    sample = (mean + std).detach()
    score = gaussian_logprob(sample, mean, std)
    expected = -0.5 - std.log() - 0.5 * np.log(2 * np.pi)
    torch.testing.assert_close(
        score, torch.full_like(score, expected), atol=1e-5, rtol=0
    )
    score.sum().backward()
    assert torch.isfinite(velocity.grad).all()


def test_lingbotvlav2_zero_noise_recovers_flow_ode():
    from rlinf.models.embodiment.lingbotvlav2.utils import (
        flow_sde_transition,
    )

    x, velocity = torch.randn(2, 50, 55), torch.randn(2, 50, 55)
    mean, std = flow_sde_transition(
        x, velocity, torch.tensor(0.5), torch.tensor(0.1), 0
    )
    torch.testing.assert_close(mean, x - 0.1 * velocity)
    assert std == 0


def test_lingbotvlav2_replay_reproduces_the_sampled_sde_step():
    from rlinf.models.embodiment.lingbotvlav2.utils import (
        sample_with_one_sde_step,
        score_sde_step,
    )

    weight = torch.nn.Parameter(torch.tensor(0.5))

    def velocity_fn(x, time):
        return weight * x + time.view(-1, 1, 1)

    torch.manual_seed(0)
    times = torch.linspace(1, 0, 11)
    noise = torch.randn(3, 50, 55)
    sde_steps = torch.tensor([0, 4, 9])
    with torch.no_grad():
        _, states, logprobs = sample_with_one_sde_step(
            noise, velocity_fn, times, 0.3, sde_steps
        )
        # The ODE over the first nine steps reaches the last step's input.
        before_last, _, _ = sample_with_one_sde_step(
            noise, velocity_fn, times[:10], 0.3, None
        )
    torch.testing.assert_close(states[0, 0], noise[0], rtol=0, atol=0)
    torch.testing.assert_close(states[2, 0], before_last[2], rtol=0, atol=0)

    replay = score_sde_step(states, velocity_fn, times, sde_steps, 0.3)
    torch.testing.assert_close(replay, logprobs, rtol=0, atol=0)
    replay.sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad)


def test_lingbotvlav2_rl_trains_the_action_expert_but_not_its_routers():
    from rlinf.models.embodiment.lingbotvlav2.utils import is_rl_trainable

    layer = "model.qwenvl_with_expert.qwen_expert.model.layers.3.mlp"
    assert not is_rl_trainable(f"{layer}.gate.weight")
    assert is_rl_trainable(f"{layer}.experts.gate_proj")
    assert is_rl_trainable(f"{layer}.shared_expert.gate_proj.weight")
    assert is_rl_trainable("model.action_out_proj.weight")
    assert not is_rl_trainable(
        "model.qwenvl_with_expert.qwenvl.model.layers.0.mlp.gate_proj.weight"
    )


def _lingbotvlav2_ppo_cfg(overrides: dict):
    """The PPO settings LingBot-VLA V2 accepts, with dotted-key overrides."""
    cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied", "only_eval": False},
            "actor": {
                "model": {"model_type": "lingbotvlav2", "precision": "fp32"},
                "fsdp_config": {
                    "strategy": "fsdp2",
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "cast_forward_inputs": False,
                    },
                },
            },
            "rollout": {"model": {"precision": "bf16"}},
            "algorithm": {"logprob_type": "token_level", "reward_type": "chunk_level"},
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("algorithm.reward_type", "action_level", "reward_type=chunk_level"),
        ("actor.fsdp_config.strategy", "fsdp", "strategy=fsdp2"),
        ("actor.model.precision", "bf16", "actor.model.precision=fp32"),
        ("actor.fsdp_config.mixed_precision.param_dtype", "fp32", "param_dtype=bf16"),
        (
            "actor.fsdp_config.mixed_precision.cast_forward_inputs",
            True,
            "cast_forward_inputs",
        ),
        ("rollout.model.precision", "fp32", "rollout.model.precision=bf16"),
    ],
)
def test_lingbotvlav2_rejects_unsupported_ppo_settings(key, value, message):
    from rlinf.config import validate_embodied_cfg

    with pytest.raises(ValueError, match=message):
        validate_embodied_cfg(_lingbotvlav2_ppo_cfg({key: value}))


def test_lingbotvlav2_treats_a_missing_cast_forward_inputs_as_the_fsdp_default():
    from rlinf.config import validate_embodied_cfg

    cfg = _lingbotvlav2_ppo_cfg({})
    del cfg.actor.fsdp_config.mixed_precision["cast_forward_inputs"]
    with pytest.raises(ValueError, match="cast_forward_inputs"):
        validate_embodied_cfg(cfg)


def test_lingbotvlav2_eval_rejects_fp32_rollout_weights():
    from rlinf.config import validate_embodied_cfg

    cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied_eval", "only_eval": True},
            "rollout": {
                "model": {
                    "model_type": "lingbotvlav2",
                    "num_action_chunks": 50,
                    "precision": "fp32",
                }
            },
        }
    )
    with pytest.raises(ValueError, match="rollout.model.precision=bf16"):
        validate_embodied_cfg(cfg)


def test_lingbotvlav2_smoke_config_composes(monkeypatch):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    monkeypatch.setenv("ROBOTWIN_ASSETS_PATH", "/explicit/asset")
    monkeypatch.setenv("LINGBOT_VLA_V2_PATH", "/explicit/lingbot-vla-v2")
    with initialize_config_dir(
        config_dir=str(root / "tests/e2e_tests/embodied"), version_base="1.1"
    ):
        cfg = compose(config_name="robotwin_ppo_lingbotvlav2")
    OmegaConf.resolve(cfg)
    assert dict(cfg.cluster.component_placement) == {"actor, env, rollout": "0-1"}
    assert cfg.actor.model.model_type == "lingbotvlav2"
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.actor.model.precision == "fp32" and cfg.rollout.model.precision == "bf16"
    assert not cfg.actor.fsdp_config.mixed_precision.cast_forward_inputs
    assert cfg.actor.model.lingbotvlav2.stats_path.startswith(
        "/explicit/lingbot-vla-v2/"
    )
    assert cfg.algorithm.adv_type == "gae" and cfg.algorithm.loss_type == "actor_critic"
    assert cfg.runner.max_steps == 1 and cfg.env.train.total_num_envs == 4


@pytest.mark.parametrize(
    ("detach_critic_input", "use_vlm_value", "expect_vl_grad", "expect_state_grad"),
    [
        pytest.param(True, False, False, False, id="detached-vl-and-state"),
        pytest.param(True, True, False, False, id="detached-vl-only"),
        pytest.param(False, False, True, True, id="attached-vl-and-state"),
        pytest.param(False, True, True, False, id="attached-vl-only"),
        pytest.param(None, False, True, True, id="default-vl-and-state"),
        pytest.param(None, True, True, False, id="default-vl-only"),
    ],
)
def test_gr00t_n1d7_value_gradient_routing(
    monkeypatch, detach_critic_input, use_vlm_value, expect_vl_grad, expect_state_grad
):
    # Delayed-env tests stub imageio; vendor imports need its real module spec.
    if isinstance(sys.modules.get("imageio"), MagicMock):
        monkeypatch.delitem(sys.modules, "imageio")
    pytest.importorskip("imageio")
    config_module = pytest.importorskip("gr00t.configs.model.gr00t_n1d7")
    pytest.importorskip("gr00t.model.gr00t_n1d7.gr00t_n1d7")
    pytest.importorskip("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        FlowMatchingActionHeadForRLActionPrediction,
    )

    # Construct only the action head; no pretrained backbone or processor loads.
    config = config_module.Gr00tN1d7Config(
        backbone_embedding_dim=16,
        hidden_size=16,
        input_embedding_dim=16,
        max_state_dim=8,
        max_action_dim=7,
        action_dim=7,
        action_horizon=2,
        max_num_embodiments=1,
        state_history_length=1,
        add_pos_embed=False,
        use_alternate_vl_dit=False,
        vl_self_attention_cfg={"num_layers": 0},
        num_inference_timesteps=2,
        num_timestep_buckets=4,
        diffusion_model_cfg={
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "output_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "positional_embeddings": None,
            "interleave_self_attention": False,
        },
    )
    rl_head_config = {
        "add_value_head": True,
        "use_vlm_value": use_vlm_value,
        "value_vlm_mode": "mean_token",
    }
    if detach_critic_input is not None:
        rl_head_config["detach_critic_input"] = detach_critic_input
    head = FlowMatchingActionHeadForRLActionPrediction(
        config, rl_head_config, output_action_chunks=2
    ).float()

    # Positive activations make every value-head layer contribute to backward.
    with torch.no_grad():
        for module in head.value_head.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(0.01)
                if module.bias is not None:
                    module.bias.zero_()
    vl_features = torch.linspace(0.1, 1.0, 96).reshape(2, 3, 16).requires_grad_()
    state_features = torch.linspace(0.2, 1.1, 32).reshape(2, 1, 16).requires_grad_()

    # Detachment must change gradient ownership, never the predicted values.
    head.rl_config["detach_critic_input"] = False
    with torch.no_grad():
        attached_values = head.get_value(vl_features, state_features)
    if detach_critic_input is None:
        head.rl_config.pop("detach_critic_input")
    else:
        head.rl_config["detach_critic_input"] = detach_critic_input
    values = head.get_value(vl_features, state_features)
    assert values.shape == (2,)
    torch.testing.assert_close(values, attached_values, rtol=0, atol=0)
    values.sum().backward()

    for parameter in head.value_head.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() > 0
    for features, expect_grad in (
        (vl_features, expect_vl_grad),
        (state_features, expect_state_grad),
    ):
        if expect_grad:
            assert features.grad is not None
            assert torch.count_nonzero(features.grad).item() > 0
        else:
            assert features.grad is None


def _accelerator_device() -> torch.device:
    """Device the OpenVLA checkpoint smoke test should run on."""
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return torch.device("npu")
    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        return torch.device("musa")
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.fail("OpenVLA accelerator test found no NPU, MUSA, or CUDA device")


def _openvla_checkpoint(env_var: str, default: str) -> str:
    path = os.environ.get(env_var, default)
    if not os.path.isdir(path):
        pytest.fail(f"OpenVLA checkpoint directory does not exist: {path}")
    return path


def _skip_unless_openvla_accelerator_test(package: str) -> None:
    """Load a real checkpoint only when the platform unit-test job asks for it."""
    if os.environ.get("RLINF_RUN_OPENVLA_ACCELERATOR_TEST") != "1":
        pytest.skip("set RLINF_RUN_OPENVLA_ACCELERATOR_TEST=1 to load this checkpoint")
    pytest.importorskip(package)


def _rgb_observation(task: str) -> dict:
    return {
        "task_descriptions": [task],
        "main_images": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8),
        "states": torch.zeros(1, 8),
    }


def test_openvla_predicts_a_finite_action_on_the_accelerator():
    _skip_unless_openvla_accelerator_test("prismatic")
    from rlinf.models.embodiment.openvla import get_model

    model_path = _openvla_checkpoint(
        "RLINF_OPENVLA_MODEL_PATH", "/workspace/dataset/openvla-7b"
    )
    cfg = OmegaConf.create(
        {
            "model_path": model_path,
            "precision": "bf16",
            "action_dim": 7,
            "num_action_chunks": 1,
            "use_proprio": False,
            "unnorm_key": "bridge_orig",
            "center_crop": True,
            "hidden_size": 4096,
            "add_value_head": False,
            "max_prompt_length": 30,
            "is_lora": False,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
    )
    device = _accelerator_device()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()

    actions, result = model.predict_action_batch(
        env_obs=_rgb_observation("pick up the carrot"),
        do_sample=False,
        max_new_tokens=7,
    )

    assert tuple(actions.shape) == (1, 1, 7)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(result["prev_logprobs"]).all()


def test_openvla_oft_predicts_a_finite_action_on_the_accelerator():
    _skip_unless_openvla_accelerator_test("prismatic")
    from rlinf.models.embodiment.openvla_oft import get_model

    model_path = _openvla_checkpoint(
        "RLINF_OPENVLA_OFT_MODEL_PATH",
        "/workspace/dataset/Openvla-oft-SFT-libero-goal-traj1",
    )
    cfg = OmegaConf.create(
        {
            "model_type": "openvla_oft",
            "implement_version": "rlinf",
            "model_path": model_path,
            "precision": "bf16",
            "action_dim": 7,
            "num_action_chunks": 8,
            "use_proprio": False,
            "use_film": False,
            "unnorm_key": "libero_goal_no_noops",
            "center_crop": True,
            "max_prompt_length": 128,
            "hidden_size": 4096,
            "add_value_head": False,
            "num_images_in_input": 1,
            "is_lora": False,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
    )
    device = _accelerator_device()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()

    actions, result = model.predict_action_batch(
        env_obs=_rgb_observation("open the drawer"),
        do_sample=False,
    )

    assert tuple(actions.shape) == (1, 8, 7)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(result["prev_logprobs"]).all()


def test_starvla_exposes_no_split_modules_for_fsdp_auto_wrap(starvla_preprocessing):
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy

    StarVLAForRLActionPrediction = importlib.import_module(
        "starvla_preprocessing_under_test.starvla_action_model"
    ).StarVLAForRLActionPrediction

    class _FakeVisionBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, dtype=torch.bfloat16)

    class _FakeDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, dtype=torch.bfloat16)

    class _FakeVLM(torch.nn.Module):
        _no_split_modules = ["_FakeDecoderLayer", "_FakeVisionBlock"]

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embed = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
            self.vision = _FakeVisionBlock()
            self.layer = _FakeDecoderLayer()

    class _FakeActionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 7, dtype=torch.float32)

    class _FakeStarVLA(torch.nn.Module):
        framework_name = "QwenOFT"

        def __init__(self):
            super().__init__()
            self.qwen_vl_interface = torch.nn.Module()
            self.qwen_vl_interface.model = _FakeVLM()
            self.action_model = _FakeActionHead()
            self.norm_stats = {
                "franka": {"action": {"min": [-1.0] * 7, "max": [1.0] * 7}}
            }

    wrapper = StarVLAForRLActionPrediction(
        starvla_model=_FakeStarVLA(),
        action_dim=7,
        num_action_chunks=2,
        add_value_head=False,
        unnorm_key="franka",
    )
    assert wrapper._no_split_modules == [
        "_FakeDecoderLayer",
        "_FakeVisionBlock",
        "_FakeActionHead",
    ]
    wrap_policy = get_fsdp_wrap_policy(
        wrapper,
        OmegaConf.create({"wrap_policy": {}, "use_orig_params": False}),
        is_lora=False,
        model_type="starvla",
    )
    wrapped = {
        name
        for name, sub in wrapper.named_modules()
        if sub is not wrapper
        and wrap_policy(
            module=sub,
            recurse=False,
            nonwrapped_numel=sum(p.numel() for p in sub.parameters()),
        )
    }
    assert wrapped == {
        "starvla_model.qwen_vl_interface.model.vision",
        "starvla_model.qwen_vl_interface.model.layer",
        "starvla_model.action_model",
    }


def test_abot_m0_override_vggt_model_path_redirects_and_restores(monkeypatch):
    import sys
    import types

    from rlinf.models.embodiment.abot_m0.abot_m0_action_model import (
        _override_vggt_model_path,
    )

    calls: list[str] = []

    class _BaseMixin:
        @classmethod
        def from_pretrained(cls, model_id, *args, **kwargs):
            calls.append(model_id)
            return SimpleNamespace(cls=cls, model_id=model_id)

    class _FakeVGGT(_BaseMixin):
        pass

    vggt_pkg = types.ModuleType("vggt")
    vggt_models = types.ModuleType("vggt.models")
    vggt_mod = types.ModuleType("vggt.models.vggt")
    vggt_mod.VGGT = _FakeVGGT
    monkeypatch.setitem(sys.modules, "vggt", vggt_pkg)
    monkeypatch.setitem(sys.modules, "vggt.models", vggt_models)
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", vggt_mod)

    monkeypatch.delenv("VGGT_MODEL_PATH", raising=False)
    with _override_vggt_model_path():
        _FakeVGGT.from_pretrained("facebook/VGGT-1B")
    assert calls == ["facebook/VGGT-1B"]

    monkeypatch.setenv("VGGT_MODEL_PATH", "/local/VGGT-1B")
    with _override_vggt_model_path():
        _FakeVGGT.from_pretrained("facebook/VGGT-1B")
        _FakeVGGT.from_pretrained("other/model")
    assert "from_pretrained" not in _FakeVGGT.__dict__
    _FakeVGGT.from_pretrained("facebook/VGGT-1B")
    assert calls == [
        "facebook/VGGT-1B",
        "/local/VGGT-1B",
        "other/model",
        "facebook/VGGT-1B",
    ]


def test_openvla_prepare_observation_accepts_5d_multi_image_tensors():
    model = _small_openvla()
    seen_shapes: list[tuple[int, ...]] = []
    orig_processor = model.input_processor

    def recording_processor(*, text, images, padding, max_length):
        seen_shapes.append(tuple(images.shape))
        return orig_processor(
            text=text, images=images, padding=padding, max_length=max_length
        )

    model.input_processor = recording_processor
    obs_4d = {
        "main_images": torch.randint(0, 256, (2, 4, 5, 3), dtype=torch.uint8),
        "task_descriptions": ["pick", "place"],
    }
    obs_5d = {
        "main_images": torch.randint(0, 256, (2, 3, 4, 5, 3), dtype=torch.uint8),
        "task_descriptions": ["pick", "place"],
    }
    model._prepare_observation(obs_4d)
    model._prepare_observation(obs_5d)
    assert seen_shapes == [(2, 1, 3, 4, 5), (2, 3, 3, 4, 5)]


def test_npu_flash_attention_cache_defaults_softmax_scale_and_evicts_dead_keys(
    monkeypatch,
):
    import gc
    import math
    import weakref

    from transformers import modeling_flash_attention_utils as flash_utils
    from transformers.integrations import npu_flash_attention as native_flash

    from rlinf.models.embodiment.gr00t import npu_common
    from rlinf.scheduler import AcceleratorType

    cache = npu_common._TensorMetadataCache(lambda t: t.clone())
    ephemeral = torch.ones(3)
    cached_ref = weakref.ref(cache.get(ephemeral))
    assert cached_ref() is not None
    assert len(cache.entries) == 1
    del ephemeral
    gc.collect()
    assert cached_ref() is None
    assert not cache.entries

    scales: list[float] = []

    def fake_fusion_attention(q, k, v, num_heads, *args, **kwargs):
        scale = kwargs.get("scale")
        assert isinstance(scale, float), f"Expected float scale, got {scale!r}"
        scales.append(scale)
        return (q,)

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(
        npu_common,
        "torch_npu",
        SimpleNamespace(npu_fusion_attention=fake_fusion_attention),
        raising=False,
    )
    monkeypatch.delattr(native_flash, "get_attn_mask_npu", raising=False)
    monkeypatch.setattr(native_flash, "SPARSE_MODE", 2, raising=False)
    monkeypatch.setattr(
        flash_utils,
        "flash_attn_func",
        native_flash.npu_flash_attn_func,
        raising=False,
    )
    monkeypatch.setattr(
        flash_utils,
        "flash_attn_varlen_func",
        native_flash.npu_flash_attn_varlen_func,
        raising=False,
    )
    monkeypatch.setattr(
        flash_utils,
        "_get_unpad_data",
        flash_utils._get_unpad_data,
        raising=False,
    )

    assert npu_common.install_npu_flash_attention_cache()
    q_bsnd = torch.randn(2, 5, 4, 16)
    flash_utils.flash_attn_func(q_bsnd, q_bsnd, q_bsnd)
    flash_utils.flash_attn_func(q_bsnd, q_bsnd, q_bsnd, softmax_scale=0.125)
    q_tnd = torch.randn(7, 4, 16)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    flash_utils.flash_attn_varlen_func(q_tnd, q_tnd, q_tnd, cu, cu)
    assert len(scales) == 3
    assert math.isclose(scales[0], 16**-0.5)
    assert math.isclose(scales[1], 0.125)
    assert math.isclose(scales[2], 16**-0.5)


def test_starvla_predict_action_batch_gates_device_images_by_vlm_and_head(
    starvla_preprocessing, monkeypatch
):
    starvla_mod = importlib.import_module(
        "starvla_preprocessing_under_test.starvla_action_model"
    )
    StarVLAForRLActionPrediction = starvla_mod.StarVLAForRLActionPrediction

    seen_devices: list = []
    orig_build = starvla_mod.data_pipeline_utils.build_examples_from_env_obs

    def record_build(*, env_obs, image_device=None, **kwargs):
        seen_devices.append(image_device)
        return orig_build(env_obs=env_obs, image_device=image_device, **kwargs)

    monkeypatch.setattr(
        starvla_mod.data_pipeline_utils,
        "build_examples_from_env_obs",
        record_build,
    )
    monkeypatch.setattr(
        starvla_mod.vlm_input_utils,
        "resolve_vlm_interface",
        lambda _: SimpleNamespace(processor=object()),
    )
    monkeypatch.setattr(
        starvla_mod,
        "get_rollout_handler",
        lambda _: lambda model, **kw: {
            "output": {"normalized_actions": np.zeros((1, 2, 7), dtype=np.float32)},
            "model_inputs": {},
            "forward_inputs": {},
        },
    )
    monkeypatch.setattr(
        starvla_mod.action_space_utils,
        "unnormalize_actions_for_env",
        lambda normalized_actions, **kw: normalized_actions,
    )

    obs = {
        "main_images": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
        "task_descriptions": ["pick"],
    }
    for vlm_type, head_type in (
        ("qwen", "oft"),
        ("florence", "oft"),
        ("qwen", "dual"),
    ):
        wrapper = torch.nn.Module()
        wrapper.__class__ = StarVLAForRLActionPrediction
        wrapper.p = torch.nn.Parameter(torch.zeros(1))
        wrapper.starvla_model = torch.nn.Module()
        wrapper.vlm_type = vlm_type
        wrapper.action_head_type = head_type
        wrapper.state_adapter_type = "none"
        wrapper.enable_state_input = False
        wrapper.action_dim = 7
        wrapper.num_action_chunks = 2
        wrapper._action_norm_stats = _FRANKA_ACTION_STATS
        wrapper.policy_setup = "franka"
        wrapper.predict_action_batch(
            obs, calculate_logprobs=False, calculate_values=False, mode="eval"
        )

    assert seen_devices == [torch.device("cpu"), None, None]
