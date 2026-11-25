# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Sim2Real 16D joint toy-car inference & launchers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_direct(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_rlinf_package_stubs(monkeypatch):
    """Bypass ray-dependent package __init__.py files while loading real submodules."""
    for pkg_name, rel_dir in (
        ("rlinf.models", "rlinf/models"),
        ("rlinf.models.embodiment", "rlinf/models/embodiment"),
        ("rlinf.models.embodiment.openpi", "rlinf/models/embodiment/openpi"),
        (
            "rlinf.models.embodiment.openpi_rlinf",
            "rlinf/models/embodiment/openpi_rlinf",
        ),
        ("rlinf.utils", "rlinf/utils"),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(REPO_ROOT / rel_dir)]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

    if "rlinf.utils.logging" not in sys.modules:
        try:
            importlib.import_module("ray")
        except ImportError:
            log_mod = types.ModuleType("rlinf.utils.logging")
            log_mod.get_logger = lambda name="test": logging.getLogger(name)
            monkeypatch.setitem(sys.modules, "rlinf.utils.logging", log_mod)


def _install_openpi_stubs(monkeypatch):
    """Install lightweight openpi/einops/package stubs when absent."""
    _install_rlinf_package_stubs(monkeypatch)

    if "einops" not in sys.modules:
        try:
            importlib.import_module("einops")
        except ImportError:
            einops = types.ModuleType("einops")
            einops.rearrange = lambda tensor, *args, **kwargs: tensor
            einops.repeat = lambda tensor, pattern, **kwargs: tensor
            monkeypatch.setitem(sys.modules, "einops", einops)

    openpi_pkg = sys.modules.get("openpi")
    if openpi_pkg is None:
        openpi_pkg = types.ModuleType("openpi")
        monkeypatch.setitem(sys.modules, "openpi", openpi_pkg)

    models_pkg = sys.modules.get("openpi.models")
    if models_pkg is None:
        models_pkg = types.ModuleType("openpi.models")
        monkeypatch.setitem(sys.modules, "openpi.models", models_pkg)

    model_mod = types.ModuleType("openpi.models.model")

    class _FakeObservation:
        def __init__(self, state):
            self.state = state

        @classmethod
        def from_dict(cls, data):
            return cls(data["state"])

    model_mod.Observation = _FakeObservation
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_mod)

    shared_pkg = sys.modules.get("openpi.shared")
    if shared_pkg is None:
        shared_pkg = types.ModuleType("openpi.shared")
        monkeypatch.setitem(sys.modules, "openpi.shared", shared_pkg)

    array_mod = types.ModuleType("openpi.shared.array_typing")
    array_mod.Array = object
    monkeypatch.setitem(sys.modules, "openpi.shared.array_typing", array_mod)


def _install_gymnasium_stub(monkeypatch):
    """Provide a minimal gymnasium stub so TianjiEnv(backend='dummy') executes without gymnasium."""
    try:
        importlib.import_module("gymnasium")
        return
    except ImportError:
        pass

    gym_mod = types.ModuleType("gymnasium")

    class _Env:
        def reset(self, *, seed=None, options=None):
            del seed, options
            return None, {}

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Dict:
        def __init__(self, spaces):
            self.spaces = dict(spaces)

    spaces_mod = types.ModuleType("gymnasium.spaces")
    spaces_mod.Box = _Box
    spaces_mod.Dict = _Dict
    gym_mod.Env = _Env
    gym_mod.spaces = spaces_mod
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces_mod)


def _install_shadow_stub(monkeypatch):
    shadow_mod = types.ModuleType("rlinf.tianji_shadow")
    transport_mod = types.ModuleType("rlinf.tianji_shadow.transport")
    transport_mod.ROOT = None
    transport_mod.REVIEW_ROOT = None
    transport_mod.SIZE = 4
    transport_mod.selected = lambda: False
    transport_mod.status = lambda: {}
    transport_mod.event = lambda *args, **kwargs: None

    settings_mod = types.ModuleType("rlinf.tianji_shadow.settings")
    settings_mod.BASE_VERSION = "base"
    settings_mod.FT_VERSION = "ft"
    settings_mod.VERSIONS = ("1116v1",)
    settings_mod.read = lambda *args, **kwargs: (None, None)
    settings_mod.values = lambda *args, **kwargs: {}

    control_mod = types.ModuleType("rlinf.tianji_shadow.control")
    control_mod.enabled = lambda: False

    shadow_mod.transport = transport_mod
    shadow_mod.settings = settings_mod
    shadow_mod.control = control_mod
    monkeypatch.setitem(sys.modules, "rlinf.tianji_shadow", shadow_mod)
    monkeypatch.setitem(sys.modules, "rlinf.tianji_shadow.transport", transport_mod)
    monkeypatch.setitem(sys.modules, "rlinf.tianji_shadow.settings", settings_mod)
    monkeypatch.setitem(sys.modules, "rlinf.tianji_shadow.control", control_mod)


def _make_safetensors_bytes(keys_and_shapes: dict[str, tuple[int, ...]]) -> bytes:
    """Build a valid safetensors file payload without requiring safetensors package."""
    import struct

    header = {}
    offset = 0
    chunks = []
    for key, shape in keys_and_shapes.items():
        numel = int(np.prod(shape))
        arr = np.zeros(numel, dtype=np.float32).tobytes()
        header[key] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(arr)],
        }
        offset += len(arr)
        chunks.append(arr)
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)


def _make_joint_eval_stub(eval_mod, *, config_name="tianji_stage1_joint_h60", use_rlt=False, horizon=60):
    from rlinf.envs.realworld.tianji.tianji_contract import identity_state
    from rlinf.envs.realworld.tianji.tianji_kinematics import (
        JOINT_ACTION_BIAS,
        JOINT_ACTION_SCALE,
    )

    model = object.__new__(eval_mod.OpenPiPytorchEvalActionModel)
    torch.nn.Module.__init__(model)
    model.state_indices = None
    model.num_steps = 2
    model.config_name = config_name
    model.rlt_cfg = types.SimpleNamespace(
        use_rlt=use_rlt,
        rlt_embed_dim=2048,
        action_env_dim=16,
        rlt_image_only=False,
        rlt_use_mask=True,
    )
    model.model = types.SimpleNamespace(
        action_horizon=horizon,
        action_dim=32,
        parameters=lambda: iter([torch.zeros(1)]),
        paligemma_with_expert=types.SimpleNamespace(
            paligemma=types.SimpleNamespace(
                language_model=types.SimpleNamespace(
                    model=types.SimpleNamespace(device=torch.device("cpu"))
                )
            )
        ),
        build_prefix_cache=lambda obs: (
            torch.zeros((obs.state.shape[0], 12, 8), dtype=torch.float32),
            torch.ones((obs.state.shape[0], 12), dtype=torch.bool),
            "kv",
        ),
        run_suffix=lambda obs, x_t, t_tensor, kv_cache, prefix_mask: torch.zeros_like(x_t),
        velocity_from_suffix=lambda suffix_out: torch.zeros_like(suffix_out),
    )

    seen = {}

    def _input_transform(repacked, transpose=False):
        del transpose
        seen["repacked"] = repacked
        out = {
            "state": torch.as_tensor(repacked["observation/state"][None], dtype=torch.float32),
            "image": {
                k: torch.zeros((1, 224, 224, 3), dtype=torch.float32)
                for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "image_mask": {
                k: torch.ones((1,), dtype=torch.bool)
                for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "tokenized_prompt": torch.zeros((1, 4), dtype=torch.long),
            "tokenized_prompt_mask": torch.ones((1, 4), dtype=torch.bool),
        }
        if "actions" in repacked:
            seen["rtc_physical_actions"] = np.asarray(repacked["actions"], dtype=np.float32)
            act = np.zeros((1, horizon, 32), dtype=np.float32)
            act[:, :, :16] = np.asarray(repacked["actions"], dtype=np.float32).reshape(1, horizon, 16)
            out["actions"] = torch.as_tensor(act, dtype=torch.float32)
        return out

    def _output_transform(data):
        physical = torch.zeros((1, horizon, 16), dtype=torch.float32)
        physical[..., :] = torch.as_tensor(JOINT_ACTION_BIAS, dtype=torch.float32)
        # Small valid deviation inside joint limits so _to_rl_frame produces non-zero values.
        physical[..., 0] = float(JOINT_ACTION_BIAS[0] + 0.1 * JOINT_ACTION_SCALE[0])
        physical[..., 7] = 0.75
        return {"actions": physical}

    model.input_transform = _input_transform
    model.output_transform = _output_transform
    model._seen = seen
    model._identity_state = identity_state()
    if use_rlt:
        model.rlt_module = types.SimpleNamespace(
            encode_rlt=lambda prefix_cache, hidden: torch.full((1, 2048), 0.25, dtype=torch.float32)
        )
        model._encode_rlt_flat = lambda prefix_output, prefix_mask: torch.full(
            (prefix_output.shape[0], 2048), 0.25, dtype=torch.float32
        )
    return model


def test_tianji_joint_configs_extract_vla_reference_and_rtc(monkeypatch):
    """Verify tianji_stage1_joint_h60 routes joint_states, _to_rl_frame, RTC, and use_rlt=False."""
    _install_openpi_stubs(monkeypatch)
    eval_mod = importlib.import_module(
        "rlinf.models.embodiment.openpi_rlinf.eval_action_model"
    )
    assert "tianji_stage1_joint" in eval_mod.TIANJI_JOINT_CONFIGS
    assert "tianji_stage1_joint_h60" in eval_mod.TIANJI_JOINT_CONFIGS

    monkeypatch.setenv("TIANJI_FAST_INFER", "0")
    model = _make_joint_eval_stub(
        eval_mod, config_name="tianji_stage1_joint_h60", use_rlt=False, horizon=60
    )

    env_obs = {
        "states": torch.as_tensor(model._identity_state[None]),
        "joint_states": torch.full((1, 16), 0.1, dtype=torch.float32),
        "rlt_qvel": torch.zeros((1, 14), dtype=torch.float32),
        "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        "extra_view_images": torch.zeros((1, 2, 224, 224, 3), dtype=torch.uint8),
        "task_descriptions": ["Install four wheels."],
        "rtc_prev_valid": torch.tensor([True]),
        "rtc_prev_actions": torch.zeros((1, 6, 16), dtype=torch.float32),
    }

    rlt_obs = model.extract_rlt_obs(env_obs)
    # Routed joint_states into observation/state for input_transform.
    assert torch.allclose(
        torch.as_tensor(model._seen["repacked"]["observation/state"]),
        env_obs["joint_states"][0],
    )
    # RTC converted 16D joint RL actions into physical actions for input_transform.
    assert model._seen["rtc_physical_actions"].shape == (1, 60, 16)
    # Pure-SFT returns zero z_rl and RL-frame 16D ref_chunk without raising on _require_rlt.
    assert rlt_obs["z_rl"].shape == (1, 2048)
    assert rlt_obs["z_rl"].dtype == torch.float32
    assert torch.count_nonzero(rlt_obs["z_rl"]) == 0
    assert rlt_obs["ref_chunk"].shape == (1, 60, 16)
    assert torch.isfinite(rlt_obs["ref_chunk"]).all()
    assert torch.isclose(rlt_obs["ref_chunk"][0, 0, 0], torch.tensor(0.1), atol=1e-5)
    assert torch.isclose(rlt_obs["ref_chunk"][0, 0, 7], torch.tensor(0.5), atol=1e-5)
    # _joint_rlt_obs keeps proprio in 16D joint space, critic_proprio in 38D (20D TCP + 18D vel), and ref_chunk_tcp in 20D TCP space.
    assert rlt_obs["proprio"].shape == (1, 16)
    assert rlt_obs["critic_proprio"].shape == (1, 38)
    assert rlt_obs["ref_chunk_tcp"].shape == (1, 60, 20)

    # Confirm use_rlt=True still works normally when rlt_module is present.
    model_rlt = _make_joint_eval_stub(
        eval_mod, config_name="tianji_stage1_joint_h60", use_rlt=True, horizon=60
    )
    rlt_obs_true = model_rlt.extract_rlt_obs(env_obs)
    assert torch.allclose(rlt_obs_true["z_rl"], torch.full((1, 2048), 0.25))


def test_rlt_utils_ready_flag_and_stage1_auto_use_rlt(tmp_path, monkeypatch):
    """Verify _write_ready_flag on both loaders and Stage1 use_rlt auto-detection."""
    _install_rlinf_package_stubs(monkeypatch)
    rlt_utils = importlib.import_module(
        "rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils"
    )

    ready_flag = tmp_path / "nested/ready_flag"
    monkeypatch.setenv("RLT_READY_FLAG", str(ready_flag))

    # 1) Bare model.safetensors (pure-SFT, no rlt_module.*)
    sf_dir = tmp_path / "ckpt_safetensors"
    sf_dir.mkdir()
    sf_path = sf_dir / "model.safetensors"
    sf_path.write_bytes(
        _make_safetensors_bytes({"paligemma_with_expert.dummy.weight": (2, 2)})
    )

    monkeypatch.delenv("TIANJI_USE_RLT", raising=False)
    assert rlt_utils.checkpoint_has_rlt(sf_dir) is False
    assert rlt_utils.resolve_stage1_use_rlt(sf_dir) is False
    monkeypatch.setenv("TIANJI_USE_RLT", "true")
    assert rlt_utils.resolve_stage1_use_rlt(sf_dir) is True
    monkeypatch.delenv("TIANJI_USE_RLT", raising=False)

    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda p, device="cpu": {
        "paligemma_with_expert.dummy.weight": torch.zeros((2, 2))
    }
    fake_safetensors.torch = fake_safetensors_torch
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    dummy_model = torch.nn.Module()
    dummy_model.load_state_dict = lambda sd, strict=True: None
    rlt_utils.load_base_safetensors(dummy_model, sf_path)
    assert ready_flag.read_text(encoding="utf-8") == str(sf_path)

    # 2) full_weights.pt without rlt_module.* vs with rlt_module.*
    ready_flag.unlink()
    pt_sft_dir = tmp_path / "ckpt_pt_sft"
    pt_sft_dir.mkdir()
    pt_sft_path = pt_sft_dir / "full_weights.pt"
    torch.save({"model.paligemma.weight": torch.zeros((2, 2))}, pt_sft_path)
    assert rlt_utils.checkpoint_has_rlt(pt_sft_dir) is False
    assert rlt_utils.resolve_stage1_use_rlt(pt_sft_dir) is False

    pt_rlt_dir = tmp_path / "ckpt_pt_rlt"
    pt_rlt_dir.mkdir()
    pt_rlt_path = pt_rlt_dir / "full_weights.pt"
    torch.save(
        {
            "model.paligemma.weight": torch.zeros((2, 2)),
            "rlt_module.proj.weight": torch.zeros((2, 2)),
        },
        pt_rlt_path,
    )
    assert rlt_utils.checkpoint_has_rlt(pt_rlt_dir) is True
    assert rlt_utils.resolve_stage1_use_rlt(pt_rlt_dir) is True

    # load_full_wrapper_weights also writes RLT_READY_FLAG.
    class _DummyWrapper:
        config_name = "tianji_stage1_joint_h60"

        def load_state_dict(self, state_dict, strict=False):
            return types.SimpleNamespace(unexpected_keys=[], missing_keys=[])

    rlt_utils.load_full_wrapper_weights(_DummyWrapper(), pt_rlt_path, expect_rlt=True)
    assert ready_flag.read_text(encoding="utf-8") == str(pt_rlt_path)


def test_tianji_vla_eval_worker_auto_use_rlt_and_h60(tmp_path, monkeypatch):
    """Verify TianjiVLAEvalWorker.init_worker accepts tianji_stage1_joint_h60 & sets use_rlt."""
    _install_rlinf_package_stubs(monkeypatch)
    captured_features = []

    class _FakeModel(torch.nn.Module):
        def eval(self):
            return self

        def requires_grad_(self, mode=False):
            return self

    monkeypatch.setattr(
        sys.modules["rlinf.models"],
        "get_model",
        lambda feature: (captured_features.append(feature) or _FakeModel()),
        raising=False,
    )

    hf_mod = types.ModuleType("rlinf.workers.rollout.hf.huggingface_worker")

    class _FakeMultiStepRolloutWorker:
        def setup_sample_params(self):
            self.model_cfg = self.cfg.rollout.model

    hf_mod.MultiStepRolloutWorker = _FakeMultiStepRolloutWorker
    monkeypatch.setitem(
        sys.modules, "rlinf.workers.rollout.hf.huggingface_worker", hf_mod
    )

    worker_mod = _load_module_direct(
        "_test_tianji_vla_eval_worker",
        "rlinf/workers/rollout/hf/tianji_vla_eval_worker.py",
    )

    sf_dir = tmp_path / "sft_ckpt"
    sf_dir.mkdir()
    (sf_dir / "model.safetensors").write_bytes(_make_safetensors_bytes({"w": (1,)}))

    monkeypatch.delenv("TIANJI_USE_RLT", raising=False)
    cfg = OmegaConf.create(
        {
            "runner": {"ckpt_path": None},
            "rollout": {
                "model": {"num_action_chunks": 42},
                "rlt_feature_model": {
                    "model_path": str(sf_dir),
                    "action_dim": 16,
                    "openpi": {
                        "config_name": "tianji_stage1_joint_h60",
                        "use_rlt": True,
                    },
                },
            },
        }
    )
    worker = worker_mod.TianjiVLAEvalWorker()
    worker.cfg = cfg
    worker.init_worker()
    assert len(captured_features) == 1
    assert captured_features[0].openpi.config_name == "tianji_stage1_joint_h60"
    assert captured_features[0].openpi.use_rlt is False


def test_tianji_joint_configs_in_client_predictor_and_stage2_check(monkeypatch):
    """Verify pi05_client, GPUStage1Predictor, and check_stage2_infer accept tianji_stage1_joint_h60."""
    _install_openpi_stubs(monkeypatch)
    client_mod = importlib.import_module(
        "rlinf.models.embodiment.openpi_rlinf.pi05_client"
    )
    assert "tianji_stage1_joint_h60" in client_mod.TIANJI_JOINT_CONFIGS

    check_stage2_src = (REPO_ROOT / "deploy/tianji/check_stage2_infer.py").read_text(
        encoding="utf-8"
    )
    assert "tianji_stage1_joint_h60" in check_stage2_src

    captured = {}

    class _FakeOpenPiModel(torch.nn.Module):
        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(
        sys.modules["rlinf.models.embodiment.openpi_rlinf"],
        "get_model",
        lambda cfg: (captured.setdefault("cfg", cfg) and _FakeOpenPiModel()),
        raising=False,
    )

    from rlinf.envs.realworld.tianji import tianji_joint

    monkeypatch.setenv("TIANJI_ACTION_SPACE", "joint")
    monkeypatch.delenv("TIANJI_OPENPI_CONFIG", raising=False)
    monkeypatch.setattr(tianji_joint, "NativeTcpFK", lambda: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: None)

    predictor_mod = importlib.import_module(
        "rlinf.envs.realworld.tianji.stage1.predictor"
    )
    predictor_mod.GPUStage1Predictor(
        contract=types.SimpleNamespace(action_chunk_length=60),
        model_path="/tmp/fake_ckpt",
        norm_stats_path="/tmp/fake_ckpt/norm_stats.json",
        device="cuda:0",
    )
    assert captured["cfg"].openpi.config_name == "tianji_stage1_joint_h60"
    assert captured["cfg"].openpi_data.repo_id == "tianji_stage1_joint_h60"


def test_hydra_tianji2_joint_stage1_env_overrides(monkeypatch):
    """Verify Hydra env overrides on tianji2_joint_stage1.yaml & realworld_tianji.yaml."""
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("TIANJI_OPENPI_CONFIG", "tianji_stage1_joint_h60")
    monkeypatch.setenv("TIANJI_USE_RLT", "false")
    monkeypatch.setenv(
        "TIANJI_TASK_PROMPT",
        "Install four wheels and screws on a wheel-less toy vehicle using the toy drill.",
    )
    config_dir = str(REPO_ROOT / "examples/embodiment/config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="tianji2_joint_stage1")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    rlt_fm = resolved["rollout"]["rlt_feature_model"]
    assert rlt_fm["openpi_data"]["repo_id"] == "tianji_stage1_joint_h60"
    assert rlt_fm["openpi"]["config_name"] == "tianji_stage1_joint_h60"
    assert rlt_fm["openpi"]["use_rlt"] is False
    assert (
        resolved["env"]["eval"]["override_cfg"]["task_description"]
        == "Install four wheels and screws on a wheel-less toy vehicle using the toy drill."
    )


@pytest.mark.parametrize("prompt_count", [1, 4, 12, 24])
def test_schedule_preflight_hil_and_env_multi_prompt_counts(
    tmp_path, monkeypatch, prompt_count
):
    """Verify 1, 4, 12, and 24 subtask prompts across schedule, preflight, HIL, and TianjiEnv."""
    _install_gymnasium_stub(monkeypatch)
    _install_shadow_stub(monkeypatch)
    monkeypatch.delenv("TIANJI_CAR_1118", raising=False)

    prompt_file = tmp_path / f"prompts_{prompt_count}.json"
    prompts = [f"Subtask {i} prompt" for i in range(1, prompt_count + 1)]
    prompt_file.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")

    # 1) tianji_schedule_env.py respects TIANJI_MAX_SUBTASKS=32 and TIANJI_STAGE1_VLA_ONLY=1.
    env = dict(os.environ)
    env.pop("TIANJI_CAR_1118", None)
    env["RLINF_ROOT"] = str(REPO_ROOT)
    env["TIANJI_ACTION_HORIZON"] = "60"
    env["TIANJI_CONTROL_HZ"] = "30"
    env["TIANJI_SUBTASK_PROMPTS"] = str(prompt_file)
    env["TIANJI_MAX_SUBTASKS"] = "32"
    env["TIANJI_STAGE1_VLA_ONLY"] = "1"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "deploy/tianji/tianji_schedule_env.py")],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "TIANJI_STAGE2_HEADS=1;" in proc.stdout
    assert f"subtasks={prompt_count} heads=1" in proc.stderr

    # Without TIANJI_MAX_SUBTASKS, Stage2 multi-head RL still enforces MAX_SUBTASKS=4.
    if prompt_count > 4:
        env_strict = dict(env)
        env_strict.pop("TIANJI_MAX_SUBTASKS")
        env_strict.pop("TIANJI_STAGE1_VLA_ONLY")
        bad = subprocess.run(
            [sys.executable, str(REPO_ROOT / "deploy/tianji/tianji_schedule_env.py")],
            capture_output=True,
            text=True,
            env=env_strict,
            check=False,
        )
        assert bad.returncode != 0

    # 2) preflight_tianji_local.py subtask_status("stage1") accepts 1..24 prompts.
    preflight = _load_module_direct(
        "_test_preflight_tianji", "deploy/tianji/preflight_tianji_local.py"
    )
    monkeypatch.setenv("TIANJI_SUBTASK_PROMPTS", str(prompt_file))
    monkeypatch.setenv("TIANJI_MAX_SUBTASKS", "32")
    monkeypatch.setenv("TIANJI_STAGE1_VLA_ONLY", "1")
    status = preflight.subtask_status("stage1")
    assert status["subtasks"] == prompt_count
    assert status["prompts"] == prompts

    # 3) TianjiEnv(backend="dummy") in inference mode loads all prompt counts cleanly.
    from rlinf.envs.realworld.tianji.tianji_env import TianjiConfig, TianjiEnv

    tianji_env = TianjiEnv(
        TianjiConfig(
            backend="dummy",
            action_space="joint",
            hil_control=True,
            hil_port=0,
            hil_mode="inference",
            subtask_prompts_file=str(prompt_file),
            image_hw=(8, 8),
            step_frequency=30,
            manual_episode_control_only=True,
            stream_chunks=True,
        )
    )
    try:
        assert len(tianji_env.subtask_prompts) == prompt_count
        snap = tianji_env.hil.snapshot()
        assert snap["subtask_count"] == prompt_count
        if prompt_count > 1:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(tianji_env.reset)
                deadline = time.monotonic() + 3.0
                while tianji_env.hil.snapshot()["episode"] != 1:
                    if fut.done():
                        fut.result()
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                tianji_env.hil.command("start")
                fut.result(timeout=3.0)
            tianji_env.hil.command("subtask", prompt_count)
            assert tianji_env.hil.snapshot()["subtask"] == prompt_count
            assert tianji_env.task_description == prompts[-1]
            tianji_env.hil.command("success")
    finally:
        tianji_env.close()


def test_global_prompt_and_training_default_limit(tmp_path, monkeypatch):
    """Verify TIANJI_TASK_PROMPT global override and preserved MAX_SUBTASKS=4 for RL training."""
    _install_gymnasium_stub(monkeypatch)
    _install_shadow_stub(monkeypatch)
    monkeypatch.delenv("TIANJI_CAR_1118", raising=False)
    monkeypatch.delenv("TIANJI_MAX_SUBTASKS", raising=False)
    monkeypatch.delenv("TIANJI_STAGE1_VLA_ONLY", raising=False)
    monkeypatch.delenv("TIANJI_SUBTASK_PROMPTS", raising=False)

    custom_prompt = "Install four wheels and screws on a wheel-less toy vehicle using the toy drill."
    monkeypatch.setenv("TIANJI_TASK_PROMPT", custom_prompt)

    preflight = _load_module_direct(
        "_test_preflight_global", "deploy/tianji/preflight_tianji_local.py"
    )
    assert preflight.subtask_status("stage1") == {
        "subtasks": 0,
        "prompt": custom_prompt,
    }

    from rlinf.envs.realworld.tianji.tianji_env import TianjiConfig, TianjiEnv

    env = TianjiEnv(
        TianjiConfig(
            backend="dummy",
            action_space="joint",
            hil_control=True,
            hil_port=0,
            hil_mode="inference",
            subtask_prompts_file=None,
            image_hw=(8, 8),
            step_frequency=30,
            manual_episode_control_only=True,
            stream_chunks=True,
        )
    )
    try:
        assert env.subtask_prompts == ()
        assert env.task_description == custom_prompt
        assert env.hil.snapshot()["subtask_count"] == 0
    finally:
        env.close()

    # Stage2 multi-head RL training without TIANJI_MAX_SUBTASKS keeps MAX_SUBTASKS=4.
    prompts_12 = tmp_path / "prompts_12.json"
    prompts_12.write_text(
        json.dumps({"prompts": [f"S{i}" for i in range(1, 13)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"1\.\.4"):
        TianjiEnv(
            TianjiConfig(
                backend="dummy",
                action_space="joint",
                hil_control=False,
                hil_mode="training",
                subtask_prompts_file=str(prompts_12),
                image_hw=(8, 8),
                step_frequency=30,
            )
        )


def test_sim2real_car_launcher_help():
    """Verify deploy/tianji/run_tianji_sim2real_car.sh is executable and parses --help."""
    launcher = REPO_ROOT / "deploy/tianji/run_tianji_sim2real_car.sh"
    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)
    proc = subprocess.run(
        ["bash", str(launcher), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--check" in proc.stdout
    assert "--read-only" in proc.stdout
    assert "--run" in proc.stdout
    assert "--dagger" in proc.stdout
    assert "--no-rtc-guidance" in proc.stdout
    assert "--rtc-guidance" in proc.stdout
    assert "--blend-steps" in proc.stdout
    assert "--max-vel-rad-s" in proc.stdout
    assert "--fast" in proc.stdout


def test_fast_infer_use_rlt_false_modes_1_and_2(monkeypatch):
    """Verify fast_infer._forward / rlt_forward return zero z_rl when use_rlt=False in mode 1 & 2."""
    _install_openpi_stubs(monkeypatch)
    pi0_pkg = sys.modules.get("rlinf.models.embodiment.openpi_rlinf.pi0_model")
    if pi0_pkg is None:
        pi0_pkg = types.ModuleType("rlinf.models.embodiment.openpi_rlinf.pi0_model")
        pi0_pkg.__path__ = [str(REPO_ROOT / "rlinf/models/embodiment/openpi_rlinf/pi0_model")]
        monkeypatch.setitem(
            sys.modules, "rlinf.models.embodiment.openpi_rlinf.pi0_model", pi0_pkg
        )
    if "rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0" not in sys.modules:
        fake_pi0 = types.ModuleType("rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0")
        fake_pi0.make_attn_mask = lambda mask, ar: torch.ones(
            (mask.shape[0], mask.shape[1], mask.shape[1]), dtype=torch.bool
        )
        monkeypatch.setitem(
            sys.modules, "rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0", fake_pi0
        )

    fast_mod = importlib.import_module("rlinf.models.embodiment.openpi_rlinf.fast_infer")

    class _MockObservation:
        def __init__(
            self,
            *,
            images=None,
            image_masks=None,
            state=None,
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        ):
            self.images = images or {}
            self.image_masks = image_masks or {}
            self.state = state
            self.tokenized_prompt = tokenized_prompt
            self.tokenized_prompt_mask = tokenized_prompt_mask

    class _StubPi0:
        action_horizon = 4
        action_dim = 16
        max_token_len = 8
        llm = types.SimpleNamespace()

        @staticmethod
        def velocity_from_suffix(hidden):
            return hidden[..., :16]

    class _StubWrapper:
        device = torch.device("cpu")
        num_steps = 2
        rlt_cfg = types.SimpleNamespace(use_rlt=False, rlt_embed_dim=2048)
        model = _StubPi0()

    monkeypatch.setattr(
        fast_mod,
        "_embed_prefix",
        lambda pi0, obs, fused: (
            torch.zeros((obs.state.shape[0], 6, 8), dtype=torch.float32),
            torch.ones((obs.state.shape[0], 6), dtype=torch.bool),
            torch.zeros((6,), dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        fast_mod,
        "_embed_suffix",
        lambda pi0, obs, x_t, t_tensor: (
            torch.ones((obs.state.shape[0], 4, 16), dtype=torch.float32) * 0.1,
            torch.ones((obs.state.shape[0], 4), dtype=torch.bool),
            torch.ones((4,), dtype=torch.bool),
            None,
        ),
    )
    monkeypatch.setattr(
        fast_mod,
        "_llm",
        lambda llm, embs, positions, attn, adarms_cond=None, kv_cache=None, fused=False: (
            [embs[0], embs[1]],
            "kv",
        ),
    )
    monkeypatch.setattr(fast_mod, "make_attn_mask", lambda m, ar: torch.ones((m.shape[0], m.shape[1], m.shape[1]), dtype=torch.bool))
    monkeypatch.setitem(
        sys.modules["einops"].__dict__,
        "repeat",
        lambda tensor, pattern, s=4, **kwargs: tensor.unsqueeze(1).expand(tensor.shape[0], s, tensor.shape[1]),
    )
    pi0_model_mod = importlib.import_module(
        "rlinf.models.embodiment.openpi_rlinf.pi0_model.model"
    )
    monkeypatch.setattr(pi0_model_mod, "Observation", _MockObservation)

    obs = _MockObservation(
        state=torch.zeros((1, 16), dtype=torch.float32),
        tokenized_prompt=torch.zeros((1, 8), dtype=torch.long),
        tokenized_prompt_mask=torch.ones((1, 8), dtype=torch.bool),
    )
    noise = torch.zeros((1, 4, 16), dtype=torch.float32)
    for mode in (1, 2):
        monkeypatch.setenv("TIANJI_FAST_INFER", str(mode))
        runner = fast_mod._runner(_StubWrapper())
        z_rl, x_out = runner.run(obs, noise, None, None, guided=False)
        assert z_rl.shape == (1, 2048)
        assert z_rl.dtype == torch.float32
        assert torch.count_nonzero(z_rl) == 0
        assert x_out.shape == (1, 4, 16)
        assert torch.isfinite(x_out).all()


def test_chunk_alignment_blend_clamp_and_monitor(tmp_path):
    """Verify timestep alignment, Smoothstep blend, step velocity clamp, and all 5 monitor branches."""
    from rlinf.envs.realworld.tianji.tianji_chunk_monitor import (
        ARM_JOINT_IDX,
        GRIPPER_IDX,
        ChunkExecutionMonitor,
        blend_new_chunk,
        clamp_chunk_step_velocity,
        compute_elapsed_skip_steps,
        subsample_chunk_for_rate,
    )
    from rlinf.envs.realworld.tianji.tianji_contract import prefetch_schedule
    from rlinf.envs.realworld.tianji.tianji_kinematics import (
        JOINT_ACTION_BIAS,
        JOINT_ACTION_SCALE,
    )

    # 1) 1:1 timestep alignment & prefetch 24/42/60 invariant.
    chunk_60 = np.arange(60 * 16, dtype=np.float32).reshape(60, 16)
    assert subsample_chunk_for_rate(chunk_60, model_hz=30, control_hz=30).shape == (60, 16)
    sub_10hz = subsample_chunk_for_rate(chunk_60, model_hz=30, control_hz=10)
    assert sub_10hz.shape == (20, 16)
    assert np.array_equal(sub_10hz, chunk_60[::3])

    sched = prefetch_schedule(60, 30.0)
    assert (sched.chunk, sched.prefetch_at, sched.ref_offset) == (42, 23, 18)

    skip_ok, exceeded_ok = compute_elapsed_skip_steps(100.60, 100.00, 1.0 / 30.0)
    assert (skip_ok, exceeded_ok) == (18, False)
    skip_late, exceeded_late = compute_elapsed_skip_steps(100.80, 100.00, 1.0 / 30.0)
    assert (skip_late, exceeded_late) == (24, True)

    # 2) blend_new_chunk: decays boundary jump on arm joints, preserves velocity & grippers, affine-invariant.
    q_prev = np.zeros(16, dtype=np.float32)
    q_prev[7] = 0.2
    q_prev[15] = 0.3
    raw_chunk = np.zeros((30, 16), dtype=np.float32)
    raw_chunk[:, ARM_JOINT_IDX] = 0.6
    raw_chunk[:, GRIPPER_IDX] = 0.9
    blended = blend_new_chunk(q_prev, raw_chunk, skip_m=5, blend_steps=10)
    assert blended.shape == (25, 16)
    assert np.allclose(blended[0, ARM_JOINT_IDX], 0.0168, atol=1e-4)  # 0.6*(3*0.1^2 - 2*0.1^3)
    assert np.allclose(blended[9:, ARM_JOINT_IDX], 0.6, atol=1e-6)
    assert np.allclose(blended[:, GRIPPER_IDX], 0.9)

    q_prev_phys = q_prev * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS
    raw_phys = raw_chunk * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS
    blended_phys = blend_new_chunk(q_prev_phys, raw_phys, skip_m=5, blend_steps=10)
    assert np.allclose(blended * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS, blended_phys, atol=1e-5)

    # 3) Per-step velocity hard clamp keeps smooth motions intact & limits OOD spikes without touching grippers.
    spike_chunk = np.zeros((10, 16), dtype=np.float32)
    spike_chunk[:, 0] = 0.5  # jump of 0.5 rad in one 1/30s step -> 15 rad/s >> 0.7854
    spike_chunk[:, GRIPPER_IDX] = 0.85
    clamped, count, pre_vmax = clamp_chunk_step_velocity(
        np.zeros(16, dtype=np.float32), spike_chunk, dt=1.0 / 30.0, v_max_rad_s=0.7854
    )
    assert count >= 1
    assert pre_vmax > 10.0
    max_step = 0.7854 / 30.0
    assert np.isclose(clamped[0, 0], max_step, atol=1e-5)
    assert np.allclose(clamped[:, GRIPPER_IDX], 0.85)

    # 4) ChunkExecutionMonitor branches + JSONL logging.
    log_file = tmp_path / "logs/chunk_diagnostics.jsonl"
    mon = ChunkExecutionMonitor(dt=1.0 / 30.0, v_max_rad_s=2.0, lag_warn_deg=2.0, log_path=log_file)
    k = 10
    q0 = np.zeros(16, dtype=np.float64)

    # OK (smooth peaked curve, non-flat-top)
    ok_seq = np.zeros((k, 16), dtype=np.float64)
    ok_seq[:, 0] = np.array(
        [0.004, 0.010, 0.020, 0.036, 0.044, 0.050, 0.054, 0.057, 0.059, 0.060]
    )
    r_ok = mon.analyze_chunk(q0, ok_seq, ok_seq)
    assert r_ok["status"] == "OK" and r_ok["flat_top_suspected"] is False

    # WARN_BOUNDARY_JUMP (only k=0 exceeds 0.9*v_max)
    jmp_seq = np.full((k, 16), 0.10, dtype=np.float64)
    r_jmp = mon.analyze_chunk(q0, jmp_seq, jmp_seq)
    assert r_jmp["status"] == "WARN_BOUNDARY_JUMP" and r_jmp["k0_clamp"] == 1 and r_jmp["clamp_count"] == 1

    # WARN_CMD_CLAMP (multiple steps exceed 0.9*v_max)
    cmd_seq = np.zeros((k, 16), dtype=np.float64)
    cmd_seq[:, 0] = np.arange(1, k + 1) * (2.5 / 30.0)
    r_cmd = mon.analyze_chunk(q0, cmd_seq, cmd_seq)
    assert r_cmd["status"] == "WARN_CMD_CLAMP" and r_cmd["clamp_count"] > 1

    # WARN_VEL_SATURATED (clamp_count == 0, end lag > 2 deg, AND worst joint near v_max)
    sat_tgt = np.zeros((k, 16), dtype=np.float64)
    sat_tgt[:, 0] = np.arange(1, k + 1) * (1.79 / 30.0)
    sat_meas = np.zeros((k, 16), dtype=np.float64)
    sat_meas[:, 0] = np.cumsum([0.04, 0.045, 0.05, 0.052, 0.054, 0.056, 0.057, 0.058, 0.058, 0.059])
    r_sat = mon.analyze_chunk(q0, sat_tgt, sat_meas)
    assert r_sat["status"] == "WARN_VEL_SATURATED"

    # INFO_CONTACT_HOLD (end lag > 2 deg AND measured velocity near 0)
    hold_tgt = np.zeros((k, 16), dtype=np.float64)
    hold_tgt[:, 0] = np.linspace(0.01, 0.08, k)
    hold_meas = np.zeros((k, 16), dtype=np.float64)
    r_hold = mon.analyze_chunk(q0, hold_tgt, hold_meas)
    assert r_hold["status"] == "INFO_CONTACT_HOLD"

    # flat_top_suspected (constant plateau meas velocity > 0.3 with clamp_count == 0)
    flat_tgt = np.zeros((20, 16), dtype=np.float64)
    flat_tgt[:, 0] = np.arange(1, 21) * (0.8 / 30.0)
    flat_meas = flat_tgt.copy()
    r_flat = mon.analyze_chunk(q0, flat_tgt, flat_meas)
    assert r_flat["flat_top_suspected"] is True
    assert r_flat["summary"].startswith("[chunk #0006]")

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 6


def test_tianji_env_chunk_blend_clamp_and_diagnostics(tmp_path, monkeypatch):
    """Verify TianjiEnv applies blend_new_chunk, velocity clamp, and writes chunk_diagnostics.jsonl."""
    _install_gymnasium_stub(monkeypatch)
    _install_shadow_stub(monkeypatch)
    monkeypatch.delenv("TIANJI_CAR_1118", raising=False)

    from rlinf.envs.realworld.tianji.tianji_env import TianjiConfig, TianjiEnv

    rec_dir = tmp_path / "rec"
    env = TianjiEnv(
        TianjiConfig(
            backend="dummy",
            action_space="joint",
            hil_control=False,
            hil_mode="inference",
            recorder_dir=str(rec_dir),
            image_hw=(8, 8),
            step_frequency=30.0,
            manual_episode_control_only=True,
            stream_chunks=True,
            first_chunk_ramp_steps=0,
            resume_chunk_ramp_steps=0,
            chunk_blend_steps=6,
            max_step_vel_rad_s=0.7854,
            chunk_diagnostics=True,
        )
    )
    try:
        env.reset()
        chunk_len = 8

        # Chunk 1: small smooth moves starting at current measured label.
        env.set_action_window_steps(chunk_len)
        env.on_action_chunk_begin()
        c1 = np.zeros((chunk_len, 16), dtype=np.float32)
        env.set_chunk_actions(c1)
        for i in range(chunk_len):
            env.step(c1[i])

        # Chunk 2: large discontinuous boundary jump in arm joint 0; verify offer_next_chunk == set_chunk_actions.
        c2 = np.zeros((chunk_len, 16), dtype=np.float32)
        c2[:, 0] = 0.6
        env.offer_next_chunk(c2)
        offered_copy = env._offered_next.copy()

        env.set_action_window_steps(chunk_len)
        env.on_action_chunk_begin()
        env.set_chunk_actions(c2)
        assert np.array_equal(env._chunk_actions, offered_copy)
        # Blended + velocity clamped step 0 must stay close to 0 (not jump immediately to 0.6).
        assert abs(float(env._chunk_actions[0, 0])) < 0.15

        for i in range(chunk_len):
            env.step(c2[i])
    finally:
        env.close()

    diag_file = rec_dir / "chunk_diagnostics.jsonl"
    assert diag_file.is_file()
    records = [json.loads(line) for line in diag_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
