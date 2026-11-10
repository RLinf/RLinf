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
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.algorithms.prefix_off_policy import build_prefix_off_policy_algorithm
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
from rlinf.models.embodiment.openpi.modules.model import IMAGE_KEYS
from rlinf.models.embodiment.prefix import (
    PREFIX_OBS_KEYS,
    PrefixFeatureModel,
    PrefixForwardType,
    StateHistoryBuffer,
    extract_prefix_obs,
    get_state_history_config,
    pool_prefix,
)
from rlinf.models.embodiment.prefix.policy import PrefixPolicy
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
    model.extra = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.extra._fsdp_wrap_role = "independent_component"
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.extra, recurse=False, nonwrapped_numel=0)


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
    assert not worker.enable_prefix
    assert worker.weight_syncer is None


@pytest.mark.parametrize(
    ("collect_transitions", "loss_type", "expected"),
    [
        (False, "actor_critic", False),
        (True, "actor_critic", True),
        (True, "prefix_off_policy", False),
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


@pytest.mark.parametrize(
    "config_name,algorithm_name,actor_head,critic_head,num_q_heads,overrides",
    [
        (
            "maniskill_prefix_stage2_td3_mlp_steam",
            "td3",
            "deterministic_mlp",
            "twin_q_mlp",
            2,
            [],
        ),
        (
            "maniskill_rlt_stage2_td3_mlp",
            "td3",
            "deterministic_mlp",
            "twin_q_mlp",
            2,
            [],
        ),
        (
            "maniskill_rlt_stage2_ac_mlp",
            "ac",
            "fixed_std_mlp",
            "multi_q_mlp",
            2,
            [],
        ),
        (
            "maniskill_rlt_stage2_ac_mlp",
            "ac",
            "fixed_std_mlp",
            "multi_q_mlp",
            4,
            ["actor.model.critic_head.num_q_heads=4"],
        ),
    ],
)
def test_maniskill_prefix_configs_build_policy_and_losses(
    monkeypatch,
    config_name,
    algorithm_name,
    actor_head,
    critic_head,
    num_q_heads,
    overrides,
):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name=config_name,
            overrides=[
                *overrides,
                "+actor.model.load_to_device=false",
                "+rollout.model.load_to_device=false",
            ],
        )
    OmegaConf.resolve(cfg)
    assert cfg.algorithm.name == algorithm_name
    assert cfg.algorithm.loss_type == "prefix_off_policy"
    assert cfg.algorithm.entropy_tuning.alpha_type == "fixed_alpha"
    assert cfg.algorithm.entropy_tuning.initial_alpha == 0.0
    assert cfg.actor.model.actor_head.name == actor_head
    assert cfg.actor.model.critic_head.name == critic_head

    algorithm = build_prefix_off_policy_algorithm(cfg)
    model = get_model(cfg.actor.model)
    rollout_model = get_model(cfg.rollout.model)
    rollout_model.load_state_dict(model.state_dict())
    assert isinstance(model, PrefixPolicy)
    assert algorithm.num_q_heads == num_q_heads
    model_cfg = cfg.actor.model
    obs = {
        "z_rl": torch.randn(2, model_cfg.z_dim),
        "proprio": torch.randn(2, model_cfg.proprio_dim),
        "ref_chunk": torch.zeros(
            2, model_cfg.ref_num_action_chunks, model_cfg.action_dim
        ),
    }
    actions, _ = model.predict_action_batch(obs, mode="eval")
    rollout_actions, _ = rollout_model.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, model_cfg.num_action_chunks, model_cfg.action_dim)
    torch.testing.assert_close(rollout_actions, actions)
    assert torch.isfinite(actions).all()
    assert (actions.abs() <= 1.0).all()

    context = SimpleNamespace(
        model=model,
        target_model=copy.deepcopy(model).eval().requires_grad_(False),
        torch_dtype=torch.float32,
        update_step=0,
    )
    batch = {
        "curr_obs": obs,
        "next_obs": obs,
        "actions": actions.flatten(1),
        "rewards": torch.ones(2, model_cfg.num_action_chunks),
        "terminations": torch.zeros(2, model_cfg.num_action_chunks, dtype=torch.bool),
        "dones": torch.zeros(2, model_cfg.num_action_chunks, dtype=torch.bool),
    }
    critic_loss, _ = algorithm.critic_loss(context, batch)
    assert critic_loss.ndim == 0 and torch.isfinite(critic_loss)
    critic_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.q_head.parameters()
    )
    model.zero_grad(set_to_none=True)
    model.set_critic_requires_grad(False)
    actor_loss, entropy, _ = algorithm.actor_loss(context, batch)
    assert actor_loss.ndim == 0 and torch.isfinite(actor_loss)
    assert entropy.ndim == 0 and torch.isfinite(entropy)
    actor_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.actor_head.parameters()
    )


def _make_prefix_training_case(algorithm_name, critic_head):
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "action_dim": 2,
                    "num_action_chunks": 2,
                    "actor_head": {
                        "name": (
                            "fixed_std_mlp"
                            if algorithm_name == "ac"
                            else "deterministic_mlp"
                        ),
                        "hidden_dims": [8, 8],
                        "fixed_std": 0.2,
                        "noise_sigma": 0.1,
                    },
                    "critic_head": {
                        "name": critic_head,
                        "hidden_dims": [8, 8],
                        "num_q_heads": 2,
                    },
                }
            },
            "algorithm": {
                "name": algorithm_name,
                "gamma": 0.9,
                "rlt_transition_replay": True,
                "reference_dropout_prob": 0.3,
                "entropy_tuning": {"alpha_type": "fixed_alpha"},
                "actor_weight_schedule": {
                    "enable": True,
                    "warmup_updates": 2,
                    "ramp_updates": 4,
                    "warmup_bc_weight": 1.0,
                    "online_bc_weight": 0.2,
                    "warmup_q_weight": 0.0,
                    "online_q_weight": 1.0,
                },
            },
        }
    )
    model = PrefixPolicy(
        prefix_dim=4,
        proprio_dim=3,
        action_dim=2,
        num_action_chunks=2,
        ref_num_action_chunks=3,
        actor_head_cfg=cfg.actor.model.actor_head,
        critic_head_cfg=cfg.actor.model.critic_head,
    )

    def observation():
        return {
            "z_rl": torch.randn(3, 4),
            "proprio": torch.randn(3, 3),
            "ref_chunk": torch.randn(3, 3, 2),
        }

    batch = {
        "curr_obs": observation(),
        "next_obs": observation(),
        "actions": torch.randn(3, 4),
        "rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "terminations": torch.zeros(3, 2, dtype=torch.bool),
        "dones": torch.tensor([[False, False], [False, True], [True, False]]),
        "intervene_flags": torch.tensor([[False, True], [False, False], [True, True]]),
    }
    context = SimpleNamespace(
        cfg=cfg,
        model=model,
        target_model=copy.deepcopy(model).eval().requires_grad_(False),
        torch_dtype=torch.float32,
        update_step=3,
    )
    return build_prefix_off_policy_algorithm(cfg), context, batch


def test_prefix_algorithm_extends_only_losses_and_target_actions():
    from rlinf.algorithms.prefix_off_policy import (
        PrefixAlgorithmRequirements,
        PrefixOffPolicyAlgorithm,
    )
    from rlinf.models.embodiment.prefix import PrefixForwardType

    class ScaledCriticAlgorithm(PrefixOffPolicyAlgorithm):
        requirements = PrefixAlgorithmRequirements(actor_distribution="stochastic")

        def critic_loss(self, context, batch):
            loss, metrics = super().critic_loss(context, batch)
            return 2.0 * loss, metrics

        def actor_loss(self, context, batch):
            return self._actor_loss(
                context, batch, actor_agg_q="mean", apply_reference_dropout=False
            )

        def target_actions(self, context, next_obs):
            actions, _, _ = context.model(
                forward_type=PrefixForwardType.ACTOR, obs=next_obs
            )
            return actions

    original, context, batch = _make_prefix_training_case("ac", "multi_q_mlp")
    algorithm = ScaledCriticAlgorithm(context.cfg)
    torch.manual_seed(7)
    original_loss, _ = original.critic_loss(context, batch)
    torch.manual_seed(7)
    critic_loss, _ = algorithm.critic_loss(context, batch)
    torch.testing.assert_close(critic_loss, 2.0 * original_loss)
    critic_loss.backward()
    context.model.zero_grad(set_to_none=True)
    context.model.set_critic_requires_grad(False)
    actor_loss, _, metrics = algorithm.actor_loss(context, batch)
    assert metrics["q_pi"] == pytest.approx(
        (metrics["q_value_0"] + metrics["q_value_1"]) / 2.0
    )
    actor_loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in context.model.actor_head.parameters()
    )


@pytest.mark.parametrize("critic_head", ["twin_q_mlp", "multi_q_mlp", "cross_q_mlp"])
def test_prefix_policy_forward_selects_critic_backend(critic_head):
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.prefix import PrefixForwardType

    _, context, batch = _make_prefix_training_case("td3", critic_head)
    model = context.model
    actions, log_probs, extra = model(
        forward_type=PrefixForwardType.ACTOR,
        obs=batch["curr_obs"],
        deterministic=True,
        apply_action_noise=False,
    )
    assert actions.shape == batch["actions"].shape
    assert log_probs.shape[0] == actions.shape[0] and extra is None
    current_q, next_q = model(
        forward_type=PrefixForwardType.CRITIC,
        obs=batch["curr_obs"],
        actions=batch["actions"],
    )
    assert current_q.shape == (3, 2)
    if critic_head == "cross_q_mlp":
        current_q, next_q = model(
            forward_type=PrefixForwardType.CRITIC,
            obs=batch["curr_obs"],
            actions=batch["actions"],
            next_obs=batch["next_obs"],
            next_actions=actions,
        )
        assert current_q.shape == next_q.shape == (3, 2)
    else:
        assert next_q is None
    for forward_type in (ForwardType.SAC, ForwardType.SAC_Q, ForwardType.CROSSQ_Q):
        with pytest.raises(NotImplementedError, match="Unsupported forward_type"):
            model(
                forward_type=forward_type,
                obs=batch["curr_obs"],
                actions=batch["actions"],
            )


@pytest.mark.parametrize("algorithm_name", ["ac", "td3"])
@pytest.mark.parametrize("critic_head", ["multi_q_mlp", "cross_q_mlp"])
def test_prefix_losses_use_fsdp_forward(tmp_path, algorithm_name, critic_head):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from rlinf.models.embodiment.modules.batch_renorm import BatchRenorm
    from rlinf.models.embodiment.prefix import PrefixForwardType

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU FSDP coverage requires the Gloo distributed backend")
    if dist.is_initialized():
        pytest.skip("CPU FSDP coverage requires an isolated process group")
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
    )
    try:
        algorithm, context, batch = _make_prefix_training_case(
            algorithm_name, critic_head
        )
        raw_model = context.model
        context.model = FSDP(
            raw_model, device_id=torch.device("cpu"), use_orig_params=True
        )
        context.target_model = FSDP(
            context.target_model, device_id=torch.device("cpu"), use_orig_params=True
        )
        online_calls = []
        target_calls = []
        critic_outputs = []

        def record_online(module, args, kwargs, output):
            online_calls.append(kwargs["forward_type"])
            if kwargs["forward_type"] == PrefixForwardType.CRITIC:
                current_q, next_q = output
                current_q.retain_grad()
                if next_q is not None and next_q.requires_grad:
                    next_q.retain_grad()
                critic_outputs.append(output)

        def record_target(module, args, kwargs, output):
            target_calls.append(kwargs["forward_type"])

        context.model.register_forward_hook(record_online, with_kwargs=True)
        context.target_model.register_forward_hook(record_target, with_kwargs=True)
        renorms = [m for m in raw_model.modules() if isinstance(m, BatchRenorm)]
        critic_loss, _ = algorithm.critic_loss(context, batch)
        assert online_calls == (
            [PrefixForwardType.ACTOR] if algorithm_name == "ac" else []
        ) + [PrefixForwardType.CRITIC]
        assert target_calls == (
            [PrefixForwardType.ACTOR] if algorithm_name == "td3" else []
        ) + ([] if renorms else [PrefixForwardType.CRITIC])
        assert all(m.steps.item() == 1 for m in renorms)
        current_q, next_q = critic_outputs[0]
        if renorms:
            assert next_q is not None and next_q.requires_grad
            reward_target = batch["rewards"][:, :1] + 0.9 * batch["rewards"][:, 1:2]
            not_done = ~batch["dones"].any(dim=-1, keepdim=True)
            target = (
                reward_target
                + not_done * 0.9**2 * next_q.detach().min(dim=-1, keepdim=True).values
            )
            torch.testing.assert_close(
                critic_loss, (current_q - target).square().mean()
            )
        critic_loss.backward()
        assert current_q.grad is not None
        assert next_q is None or next_q.grad is None
        assert all(
            p.grad is None or torch.count_nonzero(p.grad) == 0
            for p in raw_model.actor_head.parameters()
        )
        assert all(p.grad is None for p in context.target_model.parameters())
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in raw_model.q_head.parameters()
        )
        context.model.zero_grad(set_to_none=True)
        actor_loss, entropy, metrics = algorithm.actor_loss(context, batch)
        assert torch.isfinite(actor_loss) and torch.isfinite(entropy)
        assert online_calls[-2:] == [PrefixForwardType.ACTOR, PrefixForwardType.CRITIC]
        assert metrics["bc_weight"] == pytest.approx(0.6)
        assert metrics["q_weight"] == pytest.approx(0.5)
        actor_loss.backward()
        # A second microbatch must not let accumulated critic gradients change clipping.
        second_loss, _, _ = algorithm.actor_loss(context, batch)
        second_loss.backward()
        from rlinf.workers.actor.fsdp_prefix_off_policy_worker import (
            PrefixOffPolicyRuntime,
        )

        actor_params = list(raw_model.actor_head.parameters())
        expected_norm = torch.stack([p.grad.norm() for p in actor_params]).norm()
        assert all(p.grad is not None for p in raw_model.q_head.parameters())
        for p in raw_model.q_head.parameters():
            p.grad.mul_(1000)
        optimizer = torch.optim.SGD(actor_params, lr=0.01)
        worker = object.__new__(PrefixOffPolicyRuntime)
        worker.model = context.model
        norm = worker.clip_optimizer_grad_norm(optimizer, max_norm=0.01)
        torch.testing.assert_close(norm, expected_norm)
        assert all(p.grad is None for p in raw_model.q_head.parameters())
        assert torch.stack([p.grad.norm() for p in actor_params]).norm() <= 0.01001
        assert all(m.steps.item() == 1 for m in renorms)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "algorithm_name,actor_head,expected_distribution",
    [
        ("ac", "deterministic_mlp", "stochastic"),
        ("td3", "fixed_std_mlp", "deterministic"),
    ],
)
def test_prefix_algorithms_enforce_declared_actor_capabilities(
    algorithm_name, actor_head, expected_distribution
):
    _, context, _ = _make_prefix_training_case(algorithm_name, "multi_q_mlp")
    context.cfg.actor.model.actor_head.name = actor_head
    with pytest.raises(ValueError, match=f"requires a {expected_distribution} actor"):
        build_prefix_off_policy_algorithm(context.cfg)


@pytest.mark.parametrize(
    "algorithm_name,key,value,match",
    [
        ("td3", "algorithm.entropy_tuning.alpha_type", "softplus", "fixed_alpha"),
        ("ac", "algorithm.entropy_tuning", None, "fixed_alpha"),
        ("td3", "algorithm.actor_agg_q", "max", "actor_agg_q"),
        ("ac", "algorithm.actor_agg_q", "min", "actor_agg_q"),
        ("td3", "algorithm.agg_q", "mean", "agg_q='min'"),
    ],
)
def test_prefix_algorithms_reject_unsupported_config_at_build(
    algorithm_name, key, value, match
):
    _, context, _ = _make_prefix_training_case(algorithm_name, "multi_q_mlp")
    OmegaConf.update(context.cfg, key, value)
    with pytest.raises(ValueError, match=match):
        build_prefix_off_policy_algorithm(context.cfg)


def _compose_prefix_config(monkeypatch, config_dir, config_name):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setenv("REPO_PATH", str(root))
    with initialize_config_dir(config_dir=str(root / config_dir), version_base="1.1"):
        return compose(config_name=config_name)


@pytest.mark.parametrize(
    "config_dir,config_name,feature_path,expected_pool,expected_image_only",
    [
        (
            "examples/embodiment/config",
            "maniskill_prefix_stage2_td3_mlp_steam",
            "rollout.prefix_feature_model",
            "masked_mean",
            True,
        ),
        *[
            (
                "examples/embodiment/config",
                name,
                "rollout.prefix_feature_model",
                "rlt_token",
                False,
            )
            for name in (
                "maniskill_rlt_stage2_ac_mlp",
                "maniskill_rlt_stage2_td3_mlp",
                "realworld_dual_franka_rlt_stage2_ac_mlp",
            )
        ],
        *[
            (config_dir, name, "actor.model", "rlt_token", False)
            for config_dir, name in (
                ("examples/sft/config", "maniskill_rlt_stage1_sft_openpi_pi05"),
                ("examples/sft/config", "realworld_rlt_stage1_sft_openpi_pi05"),
                ("tests/e2e_tests/sft", "maniskill_rlt_stage1_sft_openpi_pi05"),
            )
        ],
        (
            "tests/e2e_tests/embodied",
            "maniskill_rlt_stage2_ac_mlp",
            "rollout.prefix_feature_model",
            "rlt_token",
            False,
        ),
    ],
)
def test_prefix_configs_resolve_pool_from_prefix_block(
    monkeypatch,
    config_dir,
    config_name,
    feature_path,
    expected_pool,
    expected_image_only,
):
    from rlinf.models.embodiment.openpi.rlt_config import build_rlt_config

    cfg = _compose_prefix_config(monkeypatch, config_dir, config_name)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    model_cfg = OmegaConf.select(cfg, feature_path)
    rlt_cfg = build_rlt_config(model_cfg.openpi, parent_cfg=model_cfg)
    assert rlt_cfg.prefix_pool == expected_pool
    assert rlt_cfg.prefix_image_only is expected_image_only


@pytest.mark.parametrize(
    "config_name",
    [
        "maniskill_prefix_stage2_td3_mlp_steam",
        "maniskill_prefix_stage2_td3_mlp_steam_calibration",
    ],
)
def test_steam_presets_use_each_checkpoint_for_normalization(monkeypatch, config_name):
    from rlinf.algorithms.expert import build_expert_model_config

    cfg = _compose_prefix_config(monkeypatch, "examples/embodiment/config", config_name)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    feature_cfg = cfg.rollout.prefix_feature_model
    expert_cfg = build_expert_model_config(
        cfg, cfg.rollout.model, prefix_feature_model_config=feature_cfg
    )

    assert feature_cfg.openpi_data.norm_stats_path is None
    assert expert_cfg.openpi_data.norm_stats_path is None
    assert feature_cfg.openpi_data.repo_id == "maniskill_peginsertionside_joint"
    assert expert_cfg.openpi_data.repo_id == feature_cfg.openpi_data.repo_id
    checkpoint_root = Path("/path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints")
    assert Path(feature_cfg.model_path) == (checkpoint_root / "global_step_1000/actor")
    assert Path(expert_cfg.model_path) == (checkpoint_root / "global_step_3000/actor")
    assert cfg.rollout.routing_gate.model.model_path == (
        "/path/to/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor"
    )
    assert feature_cfg.prefix.pool == "masked_mean"
    assert feature_cfg.prefix.image_only is True


def test_steam_calibration_preset_collects_without_a_phase_head(monkeypatch):
    from rlinf.algorithms.rlt.rlt_steam_critical_phase_gate import SteamRoutingGate

    cfg = _compose_prefix_config(
        monkeypatch,
        "examples/embodiment/config",
        "maniskill_prefix_stage2_td3_mlp_steam_calibration",
    )
    gate = SteamRoutingGate(_FakeSteamCritic(), cfg.rollout.routing_gate)
    assert gate.phase_head is None
    assert not gate.controls_actor_routing
    assert gate.emit_phase_features
    assert gate.expert_mode == "shadow"
    assert cfg.runner.max_steps == 20
    assert cfg.runner.val_check_interval == cfg.runner.save_interval == -1
    assert cfg.algorithm.rlt_gate_calibration.enable
    assert (
        cfg.algorithm.rlt_gate_calibration.save_path
        == f"{cfg.runner.logger.log_path}/{cfg.runner.logger.experiment_name}/gate_calibration"
    )
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 30000
    for mode in ("train", "eval"):
        switch = cfg.env[mode].rlt_policy_switch
        assert switch.routing_source == "environment"
        assert not switch.expert_takeover.enable
        assert switch.expert_takeover.oracle_metrics.enable

    training_cfg = _compose_prefix_config(
        monkeypatch,
        "examples/embodiment/config",
        "maniskill_prefix_stage2_td3_mlp_steam",
    )
    assert training_cfg.rollout.routing_gate.actor_switch.enable
    assert training_cfg.rollout.routing_gate.actor_switch.phase_head_path == (
        "/path/to/steam_phase_head.pt"
    )
    assert training_cfg.rollout.routing_gate.actor_switch.enter_threshold is None
    assert training_cfg.rollout.routing_gate.actor_switch.patience_chunks is None
    expert = training_cfg.rollout.routing_gate.expert_takeover
    assert expert.mode == "shadow"
    assert expert.enter_threshold is None
    assert expert.patience_chunks is None
    assert expert.warmup_chunks is None
    assert training_cfg.env.train.rlt_policy_switch.routing_source == "rollout"
    assert not training_cfg.algorithm.rlt_gate_calibration.enable


def test_prefix_e2e_config_builds_current_policy_and_algorithm(monkeypatch):
    cfg = _compose_prefix_config(
        monkeypatch, "tests/e2e_tests/embodied", "maniskill_rlt_stage2_ac_mlp"
    )
    assert cfg.algorithm.loss_type == "prefix_off_policy"
    assert cfg.algorithm.name == "ac"
    assert cfg.actor.model.model_type == cfg.rollout.model.model_type == "prefix_policy"
    algorithm = build_prefix_off_policy_algorithm(cfg)
    model = get_model(cfg.actor.model)
    rollout_model = get_model(cfg.rollout.model)
    rollout_model.load_state_dict(model.state_dict())
    obs = {
        "z_rl": torch.zeros(2, 2048),
        "proprio": torch.zeros(2, 9),
        "ref_chunk": torch.zeros(2, 10, 8),
    }
    actions = algorithm.target_actions(SimpleNamespace(model=model), obs)
    assert actions.shape == (2, 80)
    q_values, _ = model(forward_type=PrefixForwardType.CRITIC, obs=obs, actions=actions)
    assert q_values.shape == (2, 2)
    assert cfg.runner.max_epochs == 2
    assert cfg.env.train.total_num_envs == 8
    assert cfg.rollout.prefix_feature_model.model_path == (
        "/workspace/dataset/RLinf-Maniskill-RLT-Stage1/"
    )


def test_prefix_realworld_config_preserves_block_peg_task():
    from rlinf.models.embodiment.openpi.rlt_config import build_rlt_config

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml"
    )
    assert cfg.defaults[:2] == [
        "env/realworld_block_peg_insertion@env.train",
        "env/realworld_block_peg_insertion@env.eval",
    ]
    model_cfg = cfg.rollout.prefix_feature_model
    rlt_cfg = build_rlt_config(model_cfg.openpi, parent_cfg=model_cfg)
    assert rlt_cfg.prefix_pool == "rlt_token"
    assert rlt_cfg.prefix_image_only is False


@pytest.mark.parametrize(
    "legacy_key", ["stage2_z_source", "prefix_pool", "rlt_image_only"]
)
def test_build_rlt_config_rejects_openpi_pooling_fields(legacy_key):
    from rlinf.models.embodiment.openpi.rlt_config import build_rlt_config

    model_cfg = OmegaConf.create({"openpi": {"use_rlt": True, legacy_key: False}})
    with pytest.raises(ValueError, match=f"openpi.{legacy_key} is no longer"):
        build_rlt_config(model_cfg.openpi, parent_cfg=model_cfg)


def test_prefix_feature_width_must_match_policy_z_dim():
    from rlinf.models.embodiment.prefix import (
        check_prefix_feature_dim,
        extract_prefix_obs,
    )

    class FeatureModel:
        z_dim = 4

        def __init__(self, width):
            self.width = width

        def extract_prefix_obs(self, env_obs):
            return {
                "z_rl": torch.zeros(2, self.width),
                "proprio": torch.zeros(2, 3),
                "ref_chunk": torch.zeros(2, 2, 2),
            }

    assert extract_prefix_obs(FeatureModel(4), {})["z_rl"].shape == (2, 4)
    with pytest.raises(ValueError, match="last dimension 5, expected z_dim=4"):
        extract_prefix_obs(FeatureModel(5), {})
    check_prefix_feature_dim(FeatureModel(4), 4)
    with pytest.raises(ValueError, match="width 4, but actor.model.z_dim is 6"):
        check_prefix_feature_dim(FeatureModel(4), 6)


class _FakeSteamImageProcessor:
    def __init__(self, image_keys):
        self.image_keys = image_keys

    def __call__(self, images, image_masks, return_tensors, train):
        del return_tensors, train
        return {
            "pixel_values": {
                key: value.permute(0, 3, 1, 2).float() for key, value in images.items()
            },
            "image_masks": image_masks,
        }


class _FakeSteamCritic(torch.nn.Module):
    """Checkpoint stand-in whose score is the mean pixel change of a pair."""

    def __init__(self, image_keys=("image",)):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.prediction_batch_sizes = []
        self.config = SimpleNamespace(
            max_token_len=4,
            num_bins=2,
            fusion_hidden_dim=2,
            num_frames_per_pair=2,
            ensemble_size=1,
        )
        self.processor = SimpleNamespace(
            image_processor=_FakeSteamImageProcessor(list(image_keys)),
            process_text=lambda prompts, max_length, return_tensors: {
                "input_ids": torch.zeros(len(prompts), max_length, dtype=torch.long),
                "attention_mask": torch.ones(len(prompts), max_length),
            },
        )

    def predict(self, observation):
        pair = observation["images"]["image"]
        self.prediction_batch_sizes.append(pair.shape[0])
        score = (pair[:, 1] - pair[:, 0]).mean(dim=(1, 2, 3))
        return SimpleNamespace(
            predicted_values=score,
            hidden_states=score[:, None].repeat(1, 6) + 1.0,
        )


def _steam_gate_cfg(**overrides):
    cfg = {
        "chunk_size": 1,
        "lookback_chunks": 1,
        "actor_switch": {"enable": False, "collect_phase_features": True},
        "expert_takeover": {
            "enable": True,
            "warmup_chunks": 1,
            "patience_chunks": 1,
            "enter_threshold": 0.0,
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_steam_gate_emits_phase_features_only_for_ready_rows():
    from rlinf.algorithms.rlt.rlt_steam_critical_phase_gate import SteamRoutingGate

    gate = SteamRoutingGate(_FakeSteamCritic(), _steam_gate_cfg())

    def step(pixel_values, reset_mask=None):
        images = torch.tensor(pixel_values, dtype=torch.uint8)[:, None, None, None]
        return gate.step(
            {
                "main_images": images.expand(-1, 2, 2, 3),
                "task_descriptions": ["insert", "insert"],
            },
            mode="train",
            stage_id=0,
            reset_mask=reset_mask,
            external_actor_switch=torch.ones(2, dtype=torch.bool),
        )

    first = step([10, 10])
    assert gate.model.prediction_batch_sizes == []
    assert not first.diagnostics["rlt_gate_score_ready"].any()
    assert torch.count_nonzero(first.phase_features) == 0

    second = step([30, 50])
    assert gate.model.prediction_batch_sizes == [2]
    assert second.diagnostics["rlt_gate_score_ready"].all()
    torch.testing.assert_close(
        second.diagnostics["rlt_gate_score_min"][:, 0], torch.tensor([20.0, 40.0])
    )
    assert torch.count_nonzero(second.phase_features) == second.phase_features.numel()

    third = step([60, 10], reset_mask=torch.tensor([False, True]))
    assert gate.model.prediction_batch_sizes == [2, 1]
    assert third.diagnostics["rlt_gate_score_ready"][:, 0].tolist() == [True, False]
    torch.testing.assert_close(
        third.diagnostics["rlt_gate_score_min"][:, 0], torch.tensor([30.0, 0.0])
    )
    torch.testing.assert_close(
        third.phase_features[0], torch.full((1, 6), 31.0, dtype=torch.float16)
    )
    assert torch.count_nonzero(third.phase_features[1]) == 0


def test_steam_gate_maps_processor_cameras_to_env_observations():
    from rlinf.algorithms.rlt.rlt_steam_critical_phase_gate import SteamRoutingGate

    gate = SteamRoutingGate(
        _FakeSteamCritic(), _steam_gate_cfg(cameras={"image": "rgb"})
    )
    env_obs = {
        "rgb": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
        "task_descriptions": ["insert"],
    }
    decision = gate.step(env_obs, mode="eval", stage_id=0)
    assert decision.phase_features.shape == (1, 1, 6)
    with pytest.raises(KeyError, match="'image' to env observation 'rgb'"):
        gate.step(
            {"task_descriptions": ["insert"], "main_images": env_obs["rgb"]},
            mode="eval",
            stage_id=1,
        )
    with pytest.raises(ValueError, match="missing \\['wrist_image'\\]"):
        SteamRoutingGate(
            _FakeSteamCritic(("image", "wrist_image")),
            _steam_gate_cfg(cameras={"image": "rgb"}),
        )
    with pytest.raises(ValueError, match="routing_gate.main_image_key"):
        SteamRoutingGate(_FakeSteamCritic(), _steam_gate_cfg(main_image_key="image"))


def test_dual_franka_rlt_stage2_builds_prefix_policy(monkeypatch):
    from hydra import compose, initialize_config_dir

    from rlinf.models.embodiment.prefix.config import get_prefix_feature_model_config

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="realworld_dual_franka_rlt_stage2_ac_mlp")
    OmegaConf.resolve(cfg)

    assert cfg.algorithm.loss_type == "prefix_off_policy"
    algorithm = build_prefix_off_policy_algorithm(cfg)
    assert algorithm.actor_capabilities.distribution == "stochastic"
    assert algorithm.num_q_heads == 2
    feature_cfg = get_prefix_feature_model_config(cfg)
    assert feature_cfg.openpi.config_name == "pi05_dualfranka_tcp_rot6d"
    assert feature_cfg.openpi.action_env_dim == 20
    assert feature_cfg.openpi.action_chunk == 20

    model = get_model(cfg.actor.model)
    assert isinstance(model, PrefixPolicy)
    obs = {
        "z_rl": torch.zeros(2, 2048),
        "proprio": torch.zeros(2, 20),
        "ref_chunk": torch.zeros(2, 20, 20),
    }
    actions, info = model.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 10, 20)
    assert info["prev_logprobs"].shape == (2, 200)


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
    assert cfg.runner.max_steps == 2 and cfg.env.train.total_num_envs == 4


class _PrefixFeatureModel:
    z_dim = 8

    def extract_prefix_obs(self, env_obs):
        batch = env_obs["states"].shape[0]
        return {
            "z_rl": torch.ones(batch, self.z_dim),
            "proprio": env_obs["states"].to(dtype=torch.float32),
            "ref_chunk": torch.zeros(batch, 4, 7),
        }


def test_pool_prefix_masked_mean():
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0], [8.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    pooled = pool_prefix(hidden, mask, mode="masked_mean")
    torch.testing.assert_close(pooled[0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([2.0, 0.0]))


def test_pool_prefix_mean_and_last():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mean = pool_prefix(hidden, None, mode="mean")
    torch.testing.assert_close(mean, hidden.mean(dim=1))
    last = pool_prefix(hidden, None, mode="last")
    torch.testing.assert_close(last, hidden[:, -1])
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    last_valid = pool_prefix(hidden, mask, mode="last")
    torch.testing.assert_close(last_valid[0], hidden[0, 1])
    torch.testing.assert_close(last_valid[1], hidden[1, 2])


def test_prefix_feature_model_protocol_and_policy():
    feature = _PrefixFeatureModel()
    assert isinstance(feature, PrefixFeatureModel)
    obs = extract_prefix_obs(feature, {"states": torch.zeros(2, 3)})
    assert tuple(obs.keys()) == PREFIX_OBS_KEYS
    policy = PrefixPolicy(
        prefix_dim=feature.z_dim,
        proprio_dim=3,
        action_dim=7,
        num_action_chunks=4,
        ref_num_action_chunks=4,
        actor_head_cfg={"name": "fixed_std_mlp", "hidden_dims": [8, 8]},
        critic_head_cfg={"name": "multi_q_mlp", "hidden_dims": [8, 8]},
    )
    action, _, _ = policy(forward_type=PrefixForwardType.ACTOR, obs=obs)
    assert action.shape == (2, 28)


def test_history_disabled_is_identity():
    obs = _PrefixFeatureModel().extract_prefix_obs({"states": torch.ones(1, 3)})
    hist = StateHistoryBuffer(enable=False, steps=4, proprio_dim=3)
    fused = hist.fuse(obs)
    assert fused["z_rl"].shape[-1] == 8
    assert hist.extra_z_dim == 0
    cfg = OmegaConf.create({"z_dim": 2048, "state_history": {"enable": False}})
    assert get_state_history_config(cfg).extra_dim == 0


def test_history_enabled_concat_reset_and_pad():
    hist = StateHistoryBuffer(enable=True, steps=4, proprio_dim=2, pad="zero")
    prefix = torch.zeros(1, 8)
    obs = {
        "z_rl": prefix,
        "proprio": torch.ones(1, 2),
        "ref_chunk": torch.zeros(1, 1, 1),
    }
    fused = hist.fuse(obs)
    assert fused["z_rl"].shape[-1] == 8 + 8
    torch.testing.assert_close(fused["z_rl"][0, 8:14], torch.zeros(6))
    torch.testing.assert_close(fused["z_rl"][0, 14:], torch.ones(2))

    hist.reset(mask=torch.tensor([True]))
    obs2 = {
        "z_rl": prefix,
        "proprio": torch.full((1, 2), 5.0),
        "ref_chunk": torch.zeros(1, 1, 1),
    }
    fused2 = hist.fuse(obs2)
    torch.testing.assert_close(fused2["z_rl"][0, 8:14], torch.zeros(6))
    torch.testing.assert_close(fused2["z_rl"][0, 14:], torch.full((2,), 5.0))

    repeat = StateHistoryBuffer(enable=True, steps=4, proprio_dim=2, pad="repeat")
    fused_repeat = repeat.fuse(obs)
    torch.testing.assert_close(fused_repeat["z_rl"][0, 8:], torch.ones(8))


def test_history_peek_does_not_commit():
    hist = StateHistoryBuffer(enable=True, steps=2, proprio_dim=1, pad="zero")
    obs_t = {
        "z_rl": torch.zeros(1, 2),
        "proprio": torch.tensor([[1.0]]),
        "ref_chunk": torch.zeros(1, 1, 1),
    }
    hist.fuse(obs_t)
    peek = hist.fuse(
        {
            "z_rl": torch.zeros(1, 2),
            "proprio": torch.tensor([[9.0]]),
            "ref_chunk": torch.zeros(1, 1, 1),
        },
        commit=False,
    )
    torch.testing.assert_close(peek["z_rl"][0, 2:], torch.tensor([1.0, 9.0]))
    committed = hist.fuse(
        {
            "z_rl": torch.zeros(1, 2),
            "proprio": torch.tensor([[9.0]]),
            "ref_chunk": torch.zeros(1, 1, 1),
        }
    )
    torch.testing.assert_close(committed["z_rl"][0, 2:], torch.tensor([1.0, 9.0]))


def test_prefix_policy_consumes_fused_history():
    feature = _PrefixFeatureModel()
    hist = StateHistoryBuffer(enable=True, steps=4, proprio_dim=3)
    obs = hist.fuse(feature.extract_prefix_obs({"states": torch.zeros(2, 3)}))
    policy = PrefixPolicy(
        prefix_dim=feature.z_dim + hist.extra_z_dim,
        proprio_dim=3,
        action_dim=7,
        num_action_chunks=4,
        ref_num_action_chunks=4,
        actor_head_cfg={"name": "fixed_std_mlp", "hidden_dims": [8, 8]},
        critic_head_cfg={"name": "multi_q_mlp", "hidden_dims": [8, 8]},
    )
    action, _, _ = policy(forward_type=PrefixForwardType.ACTOR, obs=obs)
    assert action.shape == (2, 28)


@pytest.mark.parametrize("enabled", [False, True])
def test_prefix_policy_history_dimension_is_derived_without_mutating_config(enabled):
    from rlinf.models.embodiment.prefix import (
        build_state_history_buffer,
        get_prefix_feature_dim,
    )

    cfg = OmegaConf.create(
        {
            "model_type": "prefix_policy",
            "precision": "fp32",
            "is_lora": False,
            "z_dim": 8,
            "proprio_dim": 3,
            "action_dim": 7,
            "num_action_chunks": 4,
            "state_history": {"enable": enabled, "steps": 4},
            "actor_head": {"name": "fixed_std_mlp", "hidden_dims": [8, 8]},
            "critic_head": {"name": "multi_q_mlp", "hidden_dims": [8, 8]},
        }
    )
    before = OmegaConf.to_container(cfg, resolve=True)
    policy = get_model(cfg)
    history = build_state_history_buffer(cfg)
    expected_width = 20 if enabled else 8
    assert policy.prefix_dim == get_prefix_feature_dim(cfg) == expected_width
    obs = history.fuse(
        _PrefixFeatureModel().extract_prefix_obs({"states": torch.zeros(2, 3)})
    )
    assert obs["z_rl"].shape == (2, expected_width)
    actions, _ = policy.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 4, 7)
    assert OmegaConf.to_container(cfg, resolve=True) == before


@pytest.mark.parametrize("algorithm_name", ["ac", "td3"])
@pytest.mark.parametrize("probability", [0.0, 0.3])
def test_prefix_actor_reference_dropout_only_enabled_for_positive_probability(
    algorithm_name, probability
):
    _, context, batch = _make_prefix_training_case(algorithm_name, "multi_q_mlp")
    context.cfg.algorithm.reference_dropout_prob = probability
    algorithm = build_prefix_off_policy_algorithm(context.cfg)
    actor_requests = []

    def record_request(module, args, kwargs):
        if kwargs["forward_type"] == PrefixForwardType.ACTOR:
            actor_requests.append(kwargs)

    handle = context.model.register_forward_pre_hook(record_request, with_kwargs=True)
    try:
        loss, _, _ = algorithm.actor_loss(context, batch)
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert len(actor_requests) == 1
    assert actor_requests[0]["apply_reference_dropout"] is (probability > 0.0)
    assert actor_requests[0]["reference_dropout_prob"] == probability


def test_prefix_runtime_synchronizes_through_worker_platform(monkeypatch):
    from rlinf.workers.actor.fsdp_prefix_off_policy_worker import PrefixOffPolicyRuntime

    platform = MagicMock(spec=["synchronize", "empty_cache"])
    barrier = MagicMock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    runtime = PrefixOffPolicyRuntime()
    runtime.torch_platform = platform
    runtime._synchronize_after_training()
    platform.synchronize.assert_called_once_with()
    barrier.assert_called_once_with()
    platform.empty_cache.assert_called_once_with()


@pytest.mark.parametrize(
    "use_rlt,pool,expected",
    [(False, None, "masked_mean"), (True, None, "rlt_token"), (True, "last", "last")],
)
def test_prefix_pool_selection_and_parent_configuration(use_rlt, pool, expected):
    from rlinf.models.embodiment.openpi.rlt_config import build_rlt_config

    model_cfg = OmegaConf.create(
        {"openpi": {"use_rlt": use_rlt}, "prefix": {"pool": pool, "image_only": False}}
    )
    rlt_cfg = build_rlt_config(model_cfg.openpi, parent_cfg=model_cfg)
    assert rlt_cfg.prefix_pool == expected
    assert rlt_cfg.prefix_image_only is False


@pytest.mark.parametrize("algorithm_name", ["ac", "td3"])
def test_prefix_zero_reference_dropout_preserves_actor_outputs_and_rng(algorithm_name):
    _, context, batch = _make_prefix_training_case(algorithm_name, "multi_q_mlp")
    rng_state = torch.get_rng_state()
    with_dropout = context.model(
        forward_type=PrefixForwardType.ACTOR,
        obs=batch["curr_obs"],
        apply_reference_dropout=True,
        reference_dropout_prob=0.0,
    )
    rng_after_dropout = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    without_dropout = context.model(
        forward_type=PrefixForwardType.ACTOR,
        obs=batch["curr_obs"],
        apply_reference_dropout=False,
        reference_dropout_prob=0.0,
    )
    torch.testing.assert_close(with_dropout, without_dropout, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng_after_dropout)


@pytest.mark.parametrize("raw", [-1.2, 1.2])
def test_prefix_td3_exploration_adds_noise_before_clamping(raw):
    from rlinf.models.embodiment.prefix.heads.mlp import DeterministicMLPActor

    actor = DeterministicMLPActor(1, 1, 1, {"hidden_dims": [], "noise_sigma": 0.5})
    linear = next(m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear))
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(raw)
    state = torch.zeros(32, 1)
    torch.manual_seed(0)
    noise = torch.randn_like(state) * 0.5
    torch.manual_seed(0)
    actual, _ = actor(
        state,
        state,
        deterministic=False,
        apply_reference_dropout=False,
        reference_dropout_prob=0,
        apply_action_noise=True,
    )
    torch.testing.assert_close(actual, (raw + noise).clamp(-1, 1))
    actual.sum().backward()
    expected_grad = ((raw + noise).abs() < 1).sum().float().reshape(1)
    torch.testing.assert_close(linear.bias.grad, expected_grad)


def test_steam_gate_uses_head_metadata_and_warns_on_overrides(tmp_path, caplog):
    from rlinf.algorithms.rlt.rlt_steam_critical_phase_gate import SteamRoutingGate
    from rlinf.algorithms.rlt.rlt_steam_phase_head import SteamPhaseHead

    path = tmp_path / "head.pt"
    head = SteamPhaseHead(input_dim=6, hidden_dim=2)
    for threshold in (0.7, 0.9):
        head.save_checkpoint(
            path,
            metadata={
                "recommended_enter_threshold": threshold,
                "recommended_patience_chunks": 4,
                "recommended_expert_gate": {
                    "enter_threshold": -0.3,
                    "patience_chunks": 6,
                    "warmup_chunks": 2,
                },
            },
        )
        cfg = _steam_gate_cfg(
            actor_switch={
                "enable": True,
                "phase_head_path": str(path),
                "enter_threshold": None,
                "patience_chunks": None,
            },
            expert_takeover={
                "enable": True,
                "mode": "shadow",
                "enter_threshold": None,
                "patience_chunks": None,
                "warmup_chunks": None,
            },
        )
        gate = SteamRoutingGate(_FakeSteamCritic(), cfg)
        assert gate.enter_threshold == threshold
        assert gate.patience_chunks == 4
        assert gate.expert_enter_threshold == -0.3
        assert gate.expert_patience_chunks == 6
        assert gate.expert_warmup_chunks == 2
    cfg.actor_switch.enter_threshold = 0.1
    with caplog.at_level(logging.WARNING):
        gate = SteamRoutingGate(_FakeSteamCritic(), cfg)
    assert gate.enter_threshold == 0.1
    assert "actor_switch.enter_threshold" in caplog.text


def test_prefix_runtime_terminal_history_is_source_scoped_and_nonmutating():
    from rlinf.algorithms.prefix_off_policy.rollout import PrefixRolloutRuntime
    from rlinf.data.schema.embodied_types import (
        EnvPart,
        EnvTransition,
        PolicyInput,
        TrajectoryKey,
        TrajectorySource,
    )

    class Features:
        z_dim = 2

        def extract_prefix_obs(self, obs):
            states = obs["states"]
            return {
                "z_rl": torch.zeros(len(states), 2),
                "proprio": states,
                "ref_chunk": torch.zeros(len(states), 1, 1),
            }

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "prefix_policy",
                    "z_dim": 2,
                    "proprio_dim": 1,
                    "action_dim": 1,
                    "num_action_chunks": 1,
                    "actor_head": {"name": "deterministic_mlp", "hidden_dims": [4]},
                    "critic_head": {"name": "multi_q_mlp", "hidden_dims": [4]},
                    "state_history": {"enable": True, "steps": 3},
                }
            },
            "algorithm": {},
            "env": {"train": {"env_type": "maniskill_rlt"}},
        }
    )
    policy = PrefixPolicy(
        prefix_dim=5,
        proprio_dim=1,
        action_dim=1,
        num_action_chunks=1,
        ref_num_action_chunks=1,
        actor_head_cfg=cfg.actor.model.actor_head,
        critic_head_cfg=cfg.actor.model.critic_head,
    )
    runtime = PrefixRolloutRuntime(cfg, policy, Features())

    def source(offset, size, chunk):
        return TrajectorySource(
            TrajectoryKey(0, 0, 3, 0, chunk), size=size, offset=offset
        )

    def request(states, sources, **kwargs):
        return PolicyInput(
            obs={"states": torch.tensor(states).float().reshape(-1, 1)},
            sources=sources,
            **kwargs,
        )

    runtime.predict(request([1, 10], [source(0, 2, 0)]))
    runtime.predict(request([2, 20], [source(0, 2, 1)]))
    terminal = EnvPart(
        sources=[source(1, 1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[30.0]])},
        requires_inference=True,
    )
    nonterminal = EnvPart(sources=[source(0, 1, 1)], transition=EnvTransition())
    current = request(
        [100, 3],
        [source(1, 1, 2), source(0, 1, 2)],
        dones=torch.tensor([[True], [False]]),
        env_parts=[terminal, nonterminal],
        request_sizes=[1, 1],
    )
    _, result = runtime.predict(current)
    old = runtime.transition_obs(terminal, None, None)
    torch.testing.assert_close(old["z_rl"][0, 2:], torch.tensor([10.0, 20.0, 30.0]))
    new = result["forward_inputs"]["z_rl"][:, 2:]
    torch.testing.assert_close(new, torch.tensor([[0.0, 0.0, 100.0], [1.0, 2.0, 3.0]]))
    final_part = EnvPart(sources=[source(0, 1, 2)], transition=EnvTransition())
    final = request([4], [], env_parts=[final_part], request_sizes=[1], is_last=True)
    runtime.prepare_transitions(final)
    final_features = runtime.transition_obs(final_part, None, None)
    torch.testing.assert_close(
        final_features["z_rl"][0, 2:], torch.tensor([2.0, 3.0, 4.0])
    )
    # A different-sized terminal peek and eval cannot erase the merged train rows.
    runtime.predict(request([99, 999], [source(0, 2, 0)]), mode="eval")
    _, result = runtime.predict(request([101, 5], [source(1, 1, 3), source(0, 1, 3)]))
    torch.testing.assert_close(
        result["forward_inputs"]["z_rl"][:, 2:],
        torch.tensor([[0.0, 100.0, 101.0], [2.0, 3.0, 5.0]]),
    )
