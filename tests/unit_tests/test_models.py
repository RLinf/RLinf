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
from safetensors.torch import load_file, save_file

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel
from rlinf.envs.action_utils import (
    _openwam_absolute_eef10_to_libero7,
    _openwam_eef10_to_libero7,
)
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.apxinf_adapter import (
    OpenPIApxInfAdapter,
    _active_token_ids,
)
from rlinf.scheduler import Worker
from rlinf.models.embodiment.openwam.openwam_policy import (
    OpenWAMPolicy,
    _batch_value,
    _checkpoint_with_encoder_override,
    _infer_batch_size,
    _libero_state_to_eef10,
    _to_pil,
)
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)
from toolkits.openwam.export_checkpoint import (
    export_checkpoint,
    infer_step,
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


@pytest.fixture
def openwam_eval_recipe(monkeypatch):
    import hydra

    import rlinf.config as config_module

    repo = Path(__file__).resolve().parents[2]
    placement = SimpleNamespace(
        get_world_size=lambda component: {"env": 1, "rollout": 1}.get(component, 1)
    )
    # validate_cfg instantiates the Ray cluster with keyword arguments.
    monkeypatch.setattr(config_module, "Cluster", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        config_module,
        "HybridComponentPlacement",
        lambda *args, **kwargs: placement,
    )
    # The eval recipes resolve ``env/libero_*`` through ${oc.env:EMBODIED_PATH}.
    monkeypatch.setenv("EMBODIED_PATH", str(repo / "examples/embodiment"))
    monkeypatch.setenv("REPO_PATH", str(repo))

    def load(name, overrides=None, subdir="libero"):
        config_dir = repo / "evaluations" / subdir
        with hydra.initialize_config_dir(
            version_base="1.1", config_dir=str(config_dir)
        ):
            return hydra.compose(config_name=name, overrides=overrides or [])

    return load


@pytest.mark.parametrize(
    ("name", "suite"),
    [
        ("libero_spatial_openwam_eval", "libero_spatial"),
        ("libero_object_openwam_eval", "libero_object"),
        ("libero_goal_openwam_eval", "libero_goal"),
        ("libero_10_openwam_eval", "libero_10"),
    ],
)
def test_openwam_libero_eval_recipes_validate(openwam_eval_recipe, name, suite):
    """Every LIBERO suite recipe shares the deploy settings of the spatial one."""
    from rlinf.config import validate_cfg

    cfg = openwam_eval_recipe(name)
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    assert cfg.env.eval.task_suite_name == suite
    assert cfg.env.eval.openwam_action_representation == "native_delta_eef10"
    # OpenWAM's own LIBERO client caps every suite at 600 steps.
    assert cfg.env.eval.max_episode_steps == 600
    assert cfg.rollout.model.openwam.inference_horizon == 10
    assert cfg.rollout.model.num_action_chunks == 10
    assert cfg.rollout.model.load_to_device is True
    assert cfg.runner.logger.experiment_name == f"{suite}_openwam_eval"


def test_openwam_libero_eval_smoke_recipe_validates(openwam_eval_recipe):
    """The e2e smoke recipe keeps its short step budget divisible by the chunk."""
    from rlinf.config import validate_cfg

    repo = Path(__file__).resolve().parents[2]
    cfg = openwam_eval_recipe(
        "libero_spatial_openwam_eval",
        subdir=str(repo / "tests/e2e_tests/evaluations"),
    )
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    env = cfg.env.eval
    assert env.total_num_envs == 1
    assert env.max_episode_steps == env.max_steps_per_rollout_epoch == 30
    assert env.max_steps_per_rollout_epoch % cfg.rollout.model.num_action_chunks == 0
    assert cfg.rollout.model.openwam.inference_horizon == 10


def test_openwam_observation_adapter_smoke():
    observations = {
        "states": np.zeros((2, 7), dtype=np.float32),
        "main_images": np.zeros((2, 16, 16, 3), dtype=np.uint8),
        "task_descriptions": ["pick", "place"],
    }

    assert _infer_batch_size(observations) == 2
    assert _batch_value(observations, ("states",), 1).shape == (7,)
    assert _to_pil(_batch_value(observations, ("main_images",), 0)).size == (16, 16)


def test_openwam_libero_state_adapter_smoke():
    state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, 0.04], dtype=np.float32)
    eef10 = _libero_state_to_eef10(state)
    assert eef10.shape == (10,)
    np.testing.assert_allclose(eef10[:3], state[:3])
    np.testing.assert_allclose(eef10[3:9], [1, 0, 0, 0, 1, 0])
    assert eef10[9] == -1.0


def test_openwam_libero_absolute_action_adapter_smoke():
    pose = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
    target = pose.copy()
    target[0, 0] += 0.025
    converted = _openwam_absolute_eef10_to_libero7(target, pose)
    np.testing.assert_allclose(converted[0, :3], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(converted[0, 3:6], 0.0)
    assert converted[0, 6] == -1.0


def test_openwam_libero_action_adapter_smoke():
    action = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
    converted = _openwam_eef10_to_libero7(action)
    assert converted.shape == (1, 7)
    np.testing.assert_allclose(converted[0, :6], 0.0)
    assert converted[0, 6] == -1.0


def test_openwam_libero_action_adapter_rejects_nonfinite_values():
    pose = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
    invalid = pose.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _openwam_absolute_eef10_to_libero7(invalid, pose)
    with pytest.raises(ValueError, match="non-finite"):
        _openwam_eef10_to_libero7(invalid)


def test_openwam_encoder_path_override_stages_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"weights")
    (checkpoint / "normalization_stats.npy").write_bytes(b"stats")
    (checkpoint / "config.yaml").write_text(
        "model:\n  video_backbone:\n    encoder:\n      name: vjepa2_1\n      model_path: /old/host/path\n",
        encoding="utf-8",
    )
    encoder = tmp_path / "encoder"
    encoder.mkdir()

    with _checkpoint_with_encoder_override(
        str(checkpoint), str(encoder)
    ) as staged_path:
        staged = Path(staged_path)
        cfg = OmegaConf.load(staged / "config.yaml")
        assert cfg.model.video_backbone.encoder.model_path == str(encoder.resolve())
        assert (staged / "checkpoint_step_1.safetensors").is_symlink()
        assert (staged / "normalization_stats.npy").is_symlink()

    assert not Path(staged_path).exists()


def test_openwam_eval_inference_horizon_truncates_chunks():
    """Eval executes the first ``inference_horizon`` actions of a 32-step chunk."""

    class _Engine:
        architecture = SimpleNamespace(action_dim=10)

        def generate(self, condition):
            return {"actions": np.tile(np.arange(32, dtype=np.float32)[:, None], 10)}

    obs = {
        "images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "task_descriptions": ["a", "b"],
    }
    full = OpenWAMPolicy(_Engine(), num_frames=33, height=8, width=8, denoise_steps=2)
    actions, _ = full.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 32, 10)

    policy = OpenWAMPolicy(
        _Engine(),
        num_frames=33,
        height=8,
        width=8,
        denoise_steps=2,
        inference_horizon=10,
    )
    actions, _ = policy.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 10, 10)
    assert torch.equal(actions[0, :, 0], torch.arange(10, dtype=torch.float32))

    with pytest.raises(ValueError, match="inference_horizon"):
        OpenWAMPolicy(
            _Engine(),
            num_frames=33,
            height=8,
            width=8,
            denoise_steps=2,
            inference_horizon=33,
        )


def test_openwam_policy_rejects_rl_paths():
    """This adapter trains with SFT and rolls out for evaluation only."""
    engine = SimpleNamespace(architecture=SimpleNamespace(action_dim=10))
    policy = OpenWAMPolicy(engine, num_frames=33, height=8, width=8, denoise_steps=2)
    with pytest.raises(NotImplementedError, match="mode='eval'"):
        policy.predict_action_batch({"task_descriptions": ["a"]}, mode="train")
    with pytest.raises(NotImplementedError, match="SFT and evaluation only"):
        policy(forward_type=ForwardType.SAC)


def test_openwam_sft_forward_delegates_native_loss():
    class _Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = None

        def prepare_inputs(self, data):
            self.seen = data
            return {"marker": torch.tensor(1.0)}

        def compute_loss(self, **kwargs):
            assert kwargs["lambda_video"] == 0.25
            assert kwargs["lambda_action"] == 0.75
            return {
                "loss": self.weight * 2,
                "loss_video": self.weight.detach(),
                "loss_action": self.weight.detach() * 3,
            }

    architecture = _Architecture()
    engine = type("_Engine", (), {"architecture": architecture})()
    policy = OpenWAMPolicy(
        engine,
        num_frames=33,
        height=384,
        width=320,
        denoise_steps=10,
        lambda_video=0.25,
        lambda_action=0.75,
    )
    sample = {"video": [], "prompt": "pick"}
    output = policy(forward_type=ForwardType.SFT, data=[sample])
    assert output["loss"].requires_grad
    assert architecture.seen == [sample]
    assert output["loss_action"].item() == 3


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
# --- OpenWAM checkpoint export ---------------------------------------------


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "openwam-sft"
    (source / "tokenizer").mkdir(parents=True)
    (source / "tokenizer" / "tokenizer.json").write_text("{}")
    (source / "config.yaml").write_text("model:\n  architecture: dual_system\n")
    np.save(source / "normalization_stats.npy", np.zeros(3, dtype=np.float32))
    save_file(
        {
            "action_backbone.weight": torch.zeros(2, 2),
            "video_backbone.bias": torch.zeros(2),
            "vlm_backbone.ignored": torch.zeros(1),
        },
        str(source / "checkpoint_step_30000.safetensors"),
    )
    return source


def _make_rlinf_checkpoint(tmp_path: Path, extra: dict | None = None) -> Path:
    step_dir = tmp_path / "results" / "checkpoints" / "global_step_7"
    (step_dir / "actor" / "model_state_dict").mkdir(parents=True)
    state = {
        "architecture.action_backbone.weight": torch.full((2, 2), 3.0),
        "architecture.video_backbone.bias": torch.full((2,), 4.0),
        "architecture.vlm_backbone.ignored": torch.ones(1),
    }
    state.update(extra or {})
    torch.save(state, step_dir / "actor" / "model_state_dict" / "full_weights.pt")
    return step_dir


_ROBOTWIN_RECIPES = sorted(
    path.stem
    for path in (Path(__file__).resolve().parents[2] / "evaluations/robotwin").glob(
        "robotwin_*_openwam_eval.yaml"
    )
)


@pytest.mark.parametrize("name", _ROBOTWIN_RECIPES)
def test_openwam_robotwin_eval_recipes_validate(openwam_eval_recipe, name):
    """Every generated RoboTwin recipe switches the env to EEF control."""
    from rlinf.config import validate_cfg
    from toolkits.openwam.gen_robotwin_eval_recipes import STEP_LIMITS, rounded_steps

    task = name[len("robotwin_") : -len("_openwam_eval")]
    assert task in STEP_LIMITS
    cfg = openwam_eval_recipe(name, subdir="robotwin")
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    env = cfg.env.eval
    assert env.env_type == "robotwin"
    assert env.task_config.task_name == task
    assert env.openwam_action_representation == "absolute_eef20"
    assert env.task_config.data_type.endpose is True
    assert env.task_config.camera.collect_wrist_camera is True
    assert list(env.task_config.embodiment) == ["aloha-agilex"]
    assert env.center_crop is False
    steps = rounded_steps(STEP_LIMITS[task])
    assert env.max_episode_steps == env.task_config.step_lim == steps
    assert steps % cfg.rollout.model.num_action_chunks == 0
    assert cfg.rollout.model.action_dim == 20
    assert cfg.rollout.model.openwam.inference_horizon is None


def test_openwam_robotwin_recipe_generator_covers_all_tasks():
    from toolkits.openwam.gen_robotwin_eval_recipes import STEP_LIMITS, rounded_steps

    assert len(STEP_LIMITS) == 50 and len(_ROBOTWIN_RECIPES) == 50
    assert {f"robotwin_{t}_openwam_eval" for t in STEP_LIMITS} == set(_ROBOTWIN_RECIPES)
    assert rounded_steps(400) == 416 and rounded_steps(512) == 512


def test_openwam_eef20_to_robotwin_ee16_layout():
    from rlinf.envs.action_utils import _openwam_eef20_to_robotwin_ee16
    from rlinf.utils.rot6d import quat_xyzw_to_rot6d

    left_q = np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)], dtype=np.float32)
    right_q = np.array([np.sin(0.2), 0.0, 0.0, np.cos(0.2)], dtype=np.float32)
    action = np.concatenate(
        [
            [0.1, 0.2, 0.3],
            quat_xyzw_to_rot6d(left_q),
            [0.9],
            [-0.1, -0.2, -0.3],
            quat_xyzw_to_rot6d(right_q),
            [0.1],
        ]
    ).astype(np.float32)
    chunk = np.stack([action, action])[None]  # [1 env, 2 steps, 20]

    ee = _openwam_eef20_to_robotwin_ee16(chunk)
    assert ee.shape == (1, 2, 16)
    step = ee[0, 0]
    np.testing.assert_allclose(step[0:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(step[8:11], [-0.1, -0.2, -0.3], atol=1e-6)
    assert step[7] == pytest.approx(0.9) and step[15] == pytest.approx(0.1)
    for got, want in ((step[3:7], left_q), (step[11:15], right_q)):
        sign = np.sign(np.dot(got, want)) or 1.0
        np.testing.assert_allclose(sign * got, want, atol=1e-5)

    with pytest.raises(ValueError, match="20-D EEF actions"):
        _openwam_eef20_to_robotwin_ee16(np.zeros((1, 2, 14)))
    bad = chunk.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _openwam_eef20_to_robotwin_ee16(bad)


def test_prepare_actions_robotwin_requires_openwam_representation():
    from rlinf.envs.action_utils import prepare_actions

    joint = np.zeros((1, 3, 14), dtype=np.float32)
    # Joint-space policies pass through untouched, with or without the flag.
    out = prepare_actions(joint, "robotwin", "openpi", 3, 14, env_cfg={})
    assert out is joint

    eef = np.zeros((1, 3, 20), dtype=np.float32)
    eef[..., 3] = eef[..., 7] = eef[..., 13] = eef[..., 17] = 1.0  # identity rot6d
    with pytest.raises(ValueError, match="openwam_action_representation"):
        prepare_actions(eef, "robotwin", "openwam", 32, 20, env_cfg={})
    out = prepare_actions(
        eef,
        "robotwin",
        "openwam",
        32,
        20,
        env_cfg={"openwam_action_representation": "absolute_eef20"},
    )
    assert out.shape == (1, 3, 16)
    np.testing.assert_allclose(out[0, 0, 3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_robotwin_env_eef_proprio_and_action_type_binding():
    from rlinf.envs.sim.robotwin.robotwin_env import (
        bind_robotwin_action_type,
        execute_robotwin_ee_chunk,
        robotwin_task_eef20_proprio,
    )

    class _Robot:
        def get_left_gripper_val(self):
            return 0.25

        def get_right_gripper_val(self):
            return np.array([0.75])

    class _Task:
        def __init__(self, succeed_at=None, step_lim=8):
            self.robot = _Robot()
            self.calls = []
            self.take_action_cnt = 0
            self.step_lim = step_lim
            self.eval_success = False
            self._succeed_at = succeed_at

        def get_arm_pose(self, arm):
            pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
            return pose if arm == "left" else [-p for p in pose[:3]] + pose[3:]

        def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):
            self.calls.append(("chunk", chunk_actions.shape, action_type))
            return None

        def take_action(self, action, action_type="qpos"):
            self.calls.append(("step", tuple(action.shape), action_type))
            self.take_action_cnt += 1
            if (
                self._succeed_at is not None
                and self.take_action_cnt >= self._succeed_at
            ):
                self.eval_success = True

    task = _Task()
    proprio = robotwin_task_eef20_proprio(task)
    assert proprio.shape == (20,) and proprio.dtype == np.float32
    np.testing.assert_allclose(proprio[0:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(proprio[3:9], [1, 0, 0, 0, 1, 0])  # identity rot6d
    assert proprio[9] == pytest.approx(0.25) and proprio[19] == pytest.approx(0.75)
    np.testing.assert_allclose(proprio[10:13], [-0.1, -0.2, -0.3])

    venv = SimpleNamespace(
        envs=[SimpleNamespace(task=task), SimpleNamespace(task=_Task())]
    )
    assert bind_robotwin_action_type(venv, "ee") == 2
    assert bind_robotwin_action_type(venv, "ee") == 0  # idempotent
    # VectorEnv.step calls gen_sparse_reward_data(chunk): the ee binding replays
    # the chunk through take_action(..., action_type="ee") one target at a time.
    reward, term, trunc, infos = venv.envs[0].task.gen_sparse_reward_data(
        np.zeros((4, 16))
    )
    assert task.calls == [("step", (16,), "ee")] * 4
    assert (reward.item(), term.item(), trunc.item(), infos["success"]) == (
        0,
        0,
        0,
        False,
    )
    # Reaching step_lim truncates; further chunks are not executed.
    reward, term, trunc, infos = task.gen_sparse_reward_data(np.zeros((6, 16)))
    assert task.take_action_cnt == 8 and trunc.item() == 1 and term.item() == 0
    assert len(task.calls) == 8
    assert task.gen_sparse_reward_data(np.zeros((2, 16)))[2].item() == 1
    assert len(task.calls) == 8
    # Success mid-chunk stops the chunk and pays the sparse reward.
    winner = _Task(succeed_at=3, step_lim=100)
    reward, term, trunc, infos = execute_robotwin_ee_chunk(winner, np.zeros((10, 16)))
    assert winner.take_action_cnt == 3
    assert (reward.item(), term.item(), trunc.item(), infos["success"]) == (
        1,
        1,
        0,
        True,
    )
    with pytest.raises(ValueError, match="16-D actions"):
        execute_robotwin_ee_chunk(_Task(), np.zeros((2, 14)))
    # Switching back restores the original joint-space entry point.
    assert bind_robotwin_action_type(venv, "qpos") == 2
    venv.envs[0].task.gen_sparse_reward_data(np.zeros((4, 14)))
    assert task.calls[-1] == ("chunk", (4, 14), "qpos")
    with pytest.raises(ValueError, match="Unsupported RoboTwin action_type"):
        bind_robotwin_action_type(venv, "eef")


def test_env_output_keeps_native_proprio():
    """EnvOutput's observation schema carries the checkpoint-native proprio."""
    from rlinf.data.schema.embodied_types import EnvOutput

    obs = {
        "main_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": None,
        "states": torch.zeros(2, 14),
        "native_proprio": torch.arange(40, dtype=torch.float32).view(2, 20),
        "task_descriptions": ["a", "b"],
    }
    packed = EnvOutput(obs=obs, dones=torch.zeros(2, 1, dtype=torch.bool)).to_dict()
    assert torch.equal(packed["obs"]["native_proprio"], obs["native_proprio"])
    assert packed["obs"]["extra_view_images"] is None
    without = EnvOutput(
        obs={k: v for k, v in obs.items() if k != "native_proprio"},
        dones=torch.zeros(2, 1, dtype=torch.bool),
    ).to_dict()
    assert without["obs"]["native_proprio"] is None


def test_step_robotwin_venv_uses_caller_timeout():
    import time
    from concurrent.futures import ThreadPoolExecutor

    from rlinf.envs.sim.robotwin.robotwin_env import step_robotwin_venv

    class _Sub:
        def __init__(self, delay, fail=False):
            self.delay, self.fail = delay, fail

        def step(self, action):
            time.sleep(self.delay)
            if self.fail:
                raise ValueError("boom")
            return {
                "obs": {"a": float(action[0])},
                "reward": 0.0,
                "terminated": 0,
                "truncated": 0,
                "info": {},
            }

    def transform(results):
        return tuple(
            [r[k] for r in results]
            for k in ("obs", "reward", "terminated", "truncated", "info")
        )

    venv = SimpleNamespace(
        envs=[_Sub(0.2), _Sub(0.0)],
        env_thread_pool=ThreadPoolExecutor(2),
        transform=transform,
    )
    obs, *_ = step_robotwin_venv(venv, np.array([[1.0], [2.0]]), timeout_s=None)
    assert [o["a"] for o in obs] == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="SubEnv 0 step error: TimeoutError"):
        step_robotwin_venv(venv, np.array([[1.0], [2.0]]), timeout_s=0.01)
    venv.envs[1] = _Sub(0.0, fail=True)
    with pytest.raises(RuntimeError, match="SubEnv 1 step error: ValueError: boom"):
        step_robotwin_venv(venv, np.array([[1.0], [2.0]]), timeout_s=5)
    # Without the thread-pool attributes the call defers to venv.step.
    plain = SimpleNamespace(step=lambda actions: ("stepped", actions.shape))
    assert step_robotwin_venv(plain, np.zeros((2, 16)), timeout_s=None) == (
        "stepped",
        (2, 16),
    )


def test_openwam_prompt_template_follows_dataset_type():
    from rlinf.models.embodiment.openwam.openwam_policy import (
        ROBOTWIN_PROMPT_PREFIX,
        _prompt_template_for_dataset,
    )

    assert _prompt_template_for_dataset("libero") is None
    assert _prompt_template_for_dataset(None) is None
    template = _prompt_template_for_dataset("robotwin")
    assert template == ROBOTWIN_PROMPT_PREFIX
    assert template.startswith("A video recorded from a robot's point of view")


def test_openwam_compose_fills_every_camera_slot():
    """Head on top, left/right wrists below; single-view crops to the canvas."""
    pytest.importorskip("openwam.dataloader.transforms.multiview")
    from PIL import Image

    from rlinf.models.embodiment.openwam.openwam_policy import (
        _compose_observation_image,
    )

    def solid(color, size=(64, 48)):
        return np.full((size[1], size[0], 3), color, dtype=np.uint8)

    head = Image.fromarray(solid((255, 0, 0)))
    wrists = np.stack([solid((0, 255, 0)), solid((0, 0, 255))])  # [2, H, W, 3]
    layout = ["head_camera", "left_camera", "right_camera"]
    image = _compose_observation_image(
        head, wrists, multiview=True, camera_layout=layout, height=384, width=320
    )
    array = np.asarray(image)
    assert array.shape == (384, 320, 3)
    assert tuple(array[100, 160]) == (255, 0, 0)
    assert tuple(array[320, 80]) == (0, 255, 0)
    assert tuple(array[320, 240]) == (0, 0, 255)
    # One wrist camera only: the right slot stays black, as in the dataset reader.
    array = np.asarray(
        _compose_observation_image(
            head, wrists[0], multiview=True, camera_layout=layout, height=384, width=320
        )
    )
    assert tuple(array[320, 80]) == (0, 255, 0) and tuple(array[320, 240]) == (0, 0, 0)
    single = _compose_observation_image(
        Image.fromarray(solid((9, 9, 9), size=(640, 480))),
        None,
        multiview=False,
        camera_layout=layout,
        height=384,
        width=320,
    )
    assert single.size == (320, 384)


def test_openwam_sft_dataloader_concatenates_multiple_datasets(tmp_path, monkeypatch):
    """data.train_data_paths may list several native dataset roots."""
    import sys

    from omegaconf import OmegaConf

    from rlinf.data.datasets.openwam.dataloader import build_openwam_sft_dataloader

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.yaml").write_text(
        "dataloader:\n  type: libero\n  dataset_dir: /unused\n  num_frames: 33\n"
    )

    class _FakeDataset(torch.utils.data.Dataset):
        def __init__(self, root, length):
            self.root = root
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return {"root": self.root, "index": index}

    calls = []

    def fake_build_dataset(dl_cfg, split):
        calls.append((dl_cfg.dataset_dir, split, dl_cfg.num_frames))
        return _FakeDataset(dl_cfg.dataset_dir, {"/a": 3, "/b": 5}[dl_cfg.dataset_dir])

    registry = ModuleType("openwam.dataloader.registry")
    registry.build_dataset = fake_build_dataset
    monkeypatch.setitem(sys.modules, "openwam", ModuleType("openwam"))
    monkeypatch.setitem(
        sys.modules, "openwam.dataloader", ModuleType("openwam.dataloader")
    )
    monkeypatch.setitem(sys.modules, "openwam.dataloader.registry", registry)

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"model_path": str(ckpt)},
                "micro_batch_size": 2,
                "seed": 3,
            },
            "data": {"openwam": {"num_frames": 9}},
        }
    )
    loader, info = build_openwam_sft_dataloader(cfg, 1, 0, ["/a", "/b"])

    assert calls == [("/a", "train", 9), ("/b", "train", 9)]
    assert info["num_samples"] == 8
    assert info["num_samples_per_dataset"] == {"/a": 3, "/b": 5}
    assert info["dataset_dir"] == ["/a", "/b"]
    roots = [sample["root"] for batch in loader for sample in batch]
    assert sorted(roots) == ["/a"] * 3 + ["/b"] * 5

    _, single = build_openwam_sft_dataloader(cfg, 1, 0, "/a")
    assert single["dataset_dir"] == "/a" and single["num_samples"] == 3
    with pytest.raises(ValueError, match="requires data.train_data_paths"):
        build_openwam_sft_dataloader(cfg, 1, 0, [])


def test_openwam_sft_recipe_builds_on_cpu_and_rejects_rl(monkeypatch):
    """The SFT preset defers device placement to FSDP; RL recipes are refused."""
    import hydra

    from rlinf.config import validate_embodied_cfg, validate_sft_cfg

    repo = Path(__file__).resolve().parents[2]
    with hydra.initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/sft/config")
    ):
        cfg = hydra.compose(config_name="libero_sft_openwam")
    assert cfg.actor.model.load_to_device is False
    assert validate_sft_cfg(cfg) is cfg

    rl_cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied", "only_eval": False},
            "actor": {"model": {"model_type": "openwam"}},
            "rollout": {"model": {"model_type": "openwam"}},
            "algorithm": {},
        }
    )
    with pytest.raises(ValueError, match="RL training with the OpenWAM policy"):
        validate_embodied_cfg(rl_cfg)


def test_openwam_get_model_honors_load_to_device(monkeypatch):
    """SFT builds the policy on the CPU for FSDP; rollout loads straight to the device."""
    from rlinf.models.embodiment import openwam as openwam_module

    seen, retargets = [], []

    def fake_from_checkpoint(cls, **kwargs):
        seen.append(kwargs)
        policy = torch.nn.Module()
        policy.retarget_runtime_device = MagicMock()
        retargets.append(policy.retarget_runtime_device)
        return policy

    monkeypatch.setattr(
        OpenWAMPolicy, "from_checkpoint", classmethod(fake_from_checkpoint)
    )
    cfg = OmegaConf.create(
        {
            "model_path": "/ckpt",
            "device": "cuda:1",
            "load_to_device": False,
            "num_frames": 33,
            "openwam": {"inference_horizon": 10},
        }
    )
    openwam_module.get_model(cfg, torch.bfloat16)
    cfg.load_to_device = True
    openwam_module.get_model(cfg, torch.bfloat16)
    assert [call["device"] for call in seen] == ["cpu", "cuda:1"]
    assert seen[0]["inference_horizon"] == 10
    assert seen[0]["torch_dtype"] == torch.bfloat16
    # The CPU-built policy still prepares inputs on the accelerator FSDP uses.
    retargets[0].assert_called_once_with(torch.device("cuda:1"))
    retargets[1].assert_not_called()


def test_openwam_retarget_runtime_device_updates_cached_devices():
    """Retargeting rewrites OpenWAM's cached devices and leaves weights alone."""
    video = SimpleNamespace(_device=torch.device("cpu"))
    architecture = SimpleNamespace(
        action_dim=10,
        _device=torch.device("cpu"),
        backbones={"video": video, "vlm": SimpleNamespace()},
    )
    policy = OpenWAMPolicy(
        SimpleNamespace(architecture=architecture),
        num_frames=33,
        height=8,
        width=8,
        denoise_steps=2,
    )
    policy.retarget_runtime_device("cuda:3")
    assert architecture._device == torch.device("cuda:3")
    assert video._device == torch.device("cuda:3")
    assert not hasattr(architecture.backbones["vlm"], "_device")


def _openwam_fake_dataset_cfg(tmp_path, monkeypatch, lengths):
    """Register a fake ``openwam.dataloader.registry.build_dataset``."""
    import sys

    class _FakeDataset(torch.utils.data.Dataset):
        def __init__(self, root, length):
            self.root = root
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return {"root": self.root, "index": index}

    calls = []

    def fake_build_dataset(dl_cfg, split):
        calls.append((dl_cfg.dataset_dir, split))
        return _FakeDataset(dl_cfg.dataset_dir, lengths[dl_cfg.dataset_dir])

    registry = ModuleType("openwam.dataloader.registry")
    registry.build_dataset = fake_build_dataset
    monkeypatch.setitem(sys.modules, "openwam", ModuleType("openwam"))
    monkeypatch.setitem(
        sys.modules, "openwam.dataloader", ModuleType("openwam.dataloader")
    )
    monkeypatch.setitem(sys.modules, "openwam.dataloader.registry", registry)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.yaml").write_text(
        "dataloader:\n  type: libero\n  dataset_dir: /unused\n  num_frames: 33\n"
    )
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"model_path": str(ckpt)},
                "micro_batch_size": 2,
                "seed": 3,
            },
            "data": {},
        }
    )
    return cfg, calls


def test_openwam_sft_dataloader_checkpoints_and_resumes_mid_epoch(
    tmp_path, monkeypatch
):
    """A resumed loader continues the same shuffle epoch at the next unseen batch."""
    pytest.importorskip("torchdata")
    from torchdata.stateful_dataloader import StatefulDataLoader

    from rlinf.data.datasets.openwam.dataloader import build_openwam_sft_dataloader

    cfg, _ = _openwam_fake_dataset_cfg(tmp_path, monkeypatch, {"/a": 10})
    loader, _ = build_openwam_sft_dataloader(cfg, 1, 0, "/a")
    assert isinstance(loader, StatefulDataLoader)

    def indices(batch):
        return [sample["index"] for sample in batch]

    reference = []
    for epoch in range(2):
        loader.sampler.set_epoch(epoch)
        reference.append([indices(batch) for batch in loader])
    assert len(reference[1]) == 5 and reference[0] != reference[1]

    # Interrupt epoch 1 after two batches and resume in a fresh process' loader.
    loader.sampler.set_epoch(1)
    iterator = iter(loader)
    consumed = [indices(next(iterator)) for _ in range(2)]
    state = loader.state_dict()

    resumed, _ = build_openwam_sft_dataloader(cfg, 1, 0, "/a")
    resumed.load_state_dict(state)
    remaining = [indices(batch) for batch in resumed]
    assert consumed + remaining == reference[1]
    assert resumed.sampler.epoch == 1


def test_openwam_sft_validation_split_is_configurable_and_never_empty(
    tmp_path, monkeypatch
):
    """Validation reads the val split by default and fails loudly when empty."""
    pytest.importorskip("torchdata")
    from rlinf.data.datasets.openwam.dataloader import build_openwam_sft_dataloader

    cfg, calls = _openwam_fake_dataset_cfg(
        tmp_path, monkeypatch, {"/train": 6, "/held-out": 4, "/no-val": 0}
    )
    build_openwam_sft_dataloader(cfg, 1, 0, "/train")
    build_openwam_sft_dataloader(cfg, 1, 0, "/held-out", eval_dataset=True)
    assert calls == [("/train", "train"), ("/held-out", "val")]

    # A LeRobot root without a val split yields nothing: refuse instead of
    # silently reporting an empty validation pass.
    with pytest.raises(ValueError, match="openwam_val_split=train"):
        build_openwam_sft_dataloader(cfg, 1, 0, "/no-val", eval_dataset=True)
    cfg.data.openwam_val_split = "train"
    _, info = build_openwam_sft_dataloader(cfg, 1, 0, "/held-out", eval_dataset=True)
    assert calls[-1] == ("/held-out", "train") and info["num_samples"] == 4


def test_openwam_sft_eval_reports_mean_native_loss(monkeypatch):
    """Validation averages OpenWAM's SFT losses; other models keep raising."""
    pytest.importorskip("torchdata")
    import contextlib

    from rlinf.workers.sft import fsdp_vla_sft_worker as worker_module
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, forward_type, data):
            assert forward_type == ForwardType.SFT
            self.calls += 1
            value = float(sum(sample["v"] for sample in data))
            return {
                "loss": torch.tensor(value),
                "loss_video": torch.tensor(value / 2),
                "loss_action": torch.tensor([1.0, 2.0]),  # non-scalar: dropped
            }

    stub = SimpleNamespace(
        cfg=OmegaConf.create(
            {"actor": {"model": {"model_type": "openwam"}, "eval_max_batches": 2}}
        ),
        eval_data_loader=[[{"v": 1.0}], [{"v": 3.0}], [{"v": 5.0}]],
        model=_Model(),
        amp_context=contextlib.nullcontext(),
        worker_timer=contextlib.nullcontext,
    )
    stub._is_openwam = lambda: FSDPVlaSftWorker._is_openwam(stub)
    stub.get_eval_model_output = lambda batch: FSDPVlaSftWorker.get_eval_model_output(
        stub, batch
    )
    monkeypatch.setattr(
        worker_module, "all_reduce_dict", lambda metrics, op=None: metrics
    )

    metrics = FSDPVlaSftWorker.run_eval(stub)
    assert metrics == {"loss": 2.0, "loss_video": 1.0, "num_batches": 2.0}
    assert stub.model.calls == 2 and stub.model.training

    stub.cfg.actor.model.model_type = "openpi"
    with pytest.raises(NotImplementedError, match="eval is not supported"):
        FSDPVlaSftWorker.get_eval_model_output(stub, [{"v": 1.0}])


def test_openwam_sft_load_checkpoint_restores_or_tolerates_missing_data_state(
    tmp_path, monkeypatch, caplog
):
    """Resume restores the loader and epoch; old checkpoints without data.pt warn."""
    pytest.importorskip("torchdata")
    from rlinf.data.datasets.openwam.dataloader import build_openwam_sft_dataloader
    from rlinf.workers.sft import fsdp_vla_sft_worker as worker_module
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    monkeypatch.setattr(
        worker_module.FSDPSftWorker, "load_checkpoint", lambda self, path: None
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    cfg, _ = _openwam_fake_dataset_cfg(tmp_path, monkeypatch, {"/a": 10})

    def indices(batch):
        return [sample["index"] for sample in batch]

    # Reference run: two batches into epoch 1, then checkpoint the loader.
    source, _ = build_openwam_sft_dataloader(cfg, 1, 0, "/a")
    source.sampler.set_epoch(1)
    iterator = iter(source)
    consumed = [indices(next(iterator)) for _ in range(2)]
    remaining = [indices(batch) for batch in iterator]
    ckpt = tmp_path / "global_step_2" / "actor"
    ckpt.mkdir(parents=True)
    source_state = source.state_dict()

    def make_stub():
        # super().load_checkpoint needs a real instance; skip the heavy __init__.
        loader, _ = build_openwam_sft_dataloader(cfg, 1, 0, "/a")
        stub = object.__new__(FSDPVlaSftWorker)
        stub.data_loader = loader
        stub.data_iter = iter(loader)
        stub._rank, stub._world_size, stub._data_epoch = 0, 1, 0
        return stub

    # The state was captured after two batches; the resumed worker gets the rest.
    torch.save([source_state], ckpt / "data.pt")
    stub = make_stub()
    FSDPVlaSftWorker.load_checkpoint(stub, str(ckpt))
    assert stub._data_epoch == 1
    assert [indices(batch) for batch in stub.data_iter] == remaining
    assert len(consumed) + len(remaining) == 5

    # An older checkpoint without data.pt is still usable, with a warning.
    old = tmp_path / "global_step_1" / "actor"
    old.mkdir(parents=True)
    stub = make_stub()
    with caplog.at_level("WARNING"):
        FSDPVlaSftWorker.load_checkpoint(stub, str(old))
    assert "has no data.pt" in caplog.text
    assert stub._data_epoch == 0
    assert len([indices(batch) for batch in stub.data_iter]) == 5


def test_openwam_export_rebuilds_native_checkpoint_dir(tmp_path):
    source = _make_source(tmp_path)
    step_dir = _make_rlinf_checkpoint(tmp_path)
    out = tmp_path / "exported"

    written = export_checkpoint(step_dir, source, out)

    assert written == out / "checkpoint_step_7.safetensors"
    exported = load_file(str(written))
    assert set(exported) == {"action_backbone.weight", "video_backbone.bias"}
    assert torch.equal(exported["action_backbone.weight"], torch.full((2, 2), 3.0))
    assert (out / "config.yaml").read_text() == (source / "config.yaml").read_text()
    assert (out / "tokenizer" / "tokenizer.json").is_file()
    assert (out / "normalization_stats.npy").is_file()
    assert not list(out.glob("checkpoint_step_30000*"))
    assert not (out / "rlinf_value_head.pt").exists()
    # Keys outside architecture.* (for example an RL value head) are rejected.
    foreign = _make_rlinf_checkpoint(
        tmp_path / "foreign", {"value_head.0.weight": torch.ones(1, 8)}
    )
    with pytest.raises(ValueError, match="outside architecture"):
        export_checkpoint(foreign, source, tmp_path / "foreign-out")


def test_openwam_export_rejects_foreign_source(tmp_path):
    source = _make_source(tmp_path)
    step_dir = _make_rlinf_checkpoint(
        tmp_path, {"architecture.action_backbone.extra": torch.zeros(1)}
    )
    with pytest.raises(ValueError, match="differ from the source"):
        export_checkpoint(step_dir, source, tmp_path / "out")
    written = export_checkpoint(
        step_dir, source, tmp_path / "out", allow_key_mismatch=True
    )
    assert "action_backbone.extra" in load_file(str(written))


def test_openwam_export_links_assets_and_infers_step(tmp_path):
    source = _make_source(tmp_path)
    step_dir = _make_rlinf_checkpoint(tmp_path)
    assert infer_step(step_dir) == 7
    out = tmp_path / "linked"
    export_checkpoint(step_dir, source, out, step=12, link_assets=True)
    assert (out / "tokenizer").is_symlink()
    assert (out / "checkpoint_step_12.safetensors").is_file()
    # A bare weights file outside a global_step_N directory needs --step.
    bare = tmp_path / "weights.pt"
    bare.write_bytes(
        (step_dir / "actor" / "model_state_dict" / "full_weights.pt").read_bytes()
    )
    assert infer_step(bare) is None
    with pytest.raises(ValueError, match="--step"):
        export_checkpoint(bare, source, tmp_path / "nostep")
    written = export_checkpoint(bare, source, tmp_path / "bare", step=3)
    assert written.name == "checkpoint_step_3.safetensors"
