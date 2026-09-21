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
