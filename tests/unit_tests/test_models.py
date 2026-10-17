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
import importlib.util
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
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.apxinf_adapter import (
    OpenPIApxInfAdapter,
    _active_token_ids,
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


def test_openpi_fused_attention_preserves_fully_masked_query_rows():
    from rlinf.models.embodiment.openpi.modules.gemma import Attention

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
        expected, _ = model(inputs, positions, mask)
        with torch.no_grad():
            actual, _ = model(inputs, positions, mask)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("aspect_ratio", ["pad", "square"])
def test_dm0_tensor_images_match_native_padding_and_view_masks(aspect_ratio):
    pytest.importorskip("dexbotic")
    from PIL import Image
    from transformers import SiglipImageProcessor

    from rlinf.models.embodiment.dexbotic_dm0.dm0_policy import (
        DexboticDM0ForRLActionPrediction,
    )

    model = DexboticDM0ForRLActionPrediction.__new__(DexboticDM0ForRLActionPrediction)
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
    actual, masks = model._process_images_for_training(
        images, None, torch.device("cpu")
    )
    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=1e-6)
    assert masks.tolist() == [[True, False, False], [True, False, False]]
    assert not torch.count_nonzero(actual[:, 1:])


@pytest.mark.parametrize("precision", [None, "bf16"])
def test_dexbotic_configured_precision_supports_actions_and_training(precision):
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
    if not hasattr(vendor, "Qwen2_5_VLTextModel"):
        pytest.skip("Transformers Qwen2.5-VL text attention interface required")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLTextConfig,
    )

    from rlinf.models.embodiment.qwen_vl_npu_attention import (
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
    if not hasattr(vendor, "Qwen2_5_VLTextModel"):
        pytest.skip("Transformers Qwen2.5-VL attention interface required")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.models.embodiment.qwen_vl_npu_attention import (
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
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
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
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
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
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
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


def test_gr00t_npu_patches_are_inert_off_ascend():
    # Every GR00T version registers its Ascend patches through the same pair of
    # calls, and off NPU both do nothing, so a CUDA run is untouched.
    from rlinf.utils.patcher import Patcher

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


def test_npu_flash_attention_cache_is_inert_off_ascend():
    from rlinf.models.embodiment.gr00t.npu_common import (
        install_npu_flash_attention_cache,
    )

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
    from rlinf.models.embodiment.qwen_vl_npu_patchify import _patchify_qwen_vl

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
