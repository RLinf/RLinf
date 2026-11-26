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

"""Unit tests for Tianji Pi0.5 SFT dataconfigs, 60-step registration, and YAML."""

from __future__ import annotations

import dataclasses
import enum
import importlib
import importlib.util
import pathlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_openpi_and_lightweight_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight ``openpi`` & namespace stubs only when absent.

    Allows running these tests in lightweight CPU environments (such as the
    LoopSim conda env) where ``openpi`` or ``ray`` is not installed, while
    exercising the real ``tianji_dataconfig.py``, ``dataconfig/__init__.py``,
    and ``tianji_contract.py`` source files.
    """
    if importlib.util.find_spec("einops") is None:
        einops_mod = types.ModuleType("einops")

        def _rearrange(tensor, pattern: str):
            arr = np.asarray(tensor)
            lhs, rhs = (part.strip().split() for part in pattern.split("->"))
            perm = [lhs.index(axis) for axis in rhs]
            return np.transpose(arr, perm)

        einops_mod.rearrange = _rearrange
        monkeypatch.setitem(sys.modules, "einops", einops_mod)

    if importlib.util.find_spec("openpi") is None:
        class ModelType(enum.Enum):
            PI0 = "pi0"
            PI05 = "pi05"
            PI0_FAST = "pi0_fast"

        @dataclasses.dataclass(frozen=True)
        class BaseModelConfig:
            action_dim: int = 32
            action_horizon: int = 50
            max_token_len: int = 48
            model_type: ModelType = ModelType.PI0

        @dataclasses.dataclass(frozen=True)
        class Pi0Config(BaseModelConfig):
            pi05: bool = False
            discrete_state_input: bool | None = None
            paligemma_variant: str = "gemma_2b"
            action_expert_variant: str = "gemma_300m"

            def __post_init__(self):
                if self.pi05 and self.max_token_len == 48:
                    object.__setattr__(self, "max_token_len", 200)
                if self.discrete_state_input is None:
                    object.__setattr__(self, "discrete_state_input", self.pi05)
                object.__setattr__(
                    self,
                    "model_type",
                    ModelType.PI05 if self.pi05 else ModelType.PI0,
                )

            def get_freeze_filter(self):
                return None

        class DataTransformFn:
            def __call__(self, data: dict) -> dict:
                raise NotImplementedError

        @dataclasses.dataclass(frozen=True)
        class Group:
            inputs: tuple = ()
            outputs: tuple = ()

            def push(self, *, inputs=(), outputs=()):
                return Group(
                    inputs=tuple(self.inputs) + tuple(inputs),
                    outputs=tuple(outputs) + tuple(self.outputs),
                )

        @dataclasses.dataclass(frozen=True)
        class RepackTransform(DataTransformFn):
            structure: dict

            def __call__(self, data: dict) -> dict:
                return {dst: data[src] for dst, src in self.structure.items()}

        @dataclasses.dataclass(frozen=True)
        class DeltaActions(DataTransformFn):
            mask: np.ndarray

            def __call__(self, data: dict) -> dict:
                return data

        @dataclasses.dataclass(frozen=True)
        class AbsoluteActions(DataTransformFn):
            mask: np.ndarray

            def __call__(self, data: dict) -> dict:
                return data

        def pad_to_dim(x, target_dim: int, axis: int = -1):
            arr = np.asarray(x)
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axis] = (0, max(0, target_dim - arr.shape[axis]))
            return np.pad(arr, pad_width)

        def make_bool_mask(*dims: int):
            parts = [np.full(abs(d), d > 0, dtype=bool) for d in dims]
            return np.concatenate(parts)

        @dataclasses.dataclass(frozen=True)
        class AssetsConfig:
            assets_dir: str | pathlib.Path = "assets"
            asset_id: str | None = None

        @dataclasses.dataclass(frozen=True)
        class DataConfig:
            repo_id: str | None = None
            assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
            base_config: DataConfig | None = None
            repack_transforms: Group = dataclasses.field(default_factory=Group)
            data_transforms: Group = dataclasses.field(default_factory=Group)
            model_transforms: Group = dataclasses.field(default_factory=Group)
            action_sequence_keys: tuple[str, ...] = ("actions",)
            prompt_from_task: bool = False
            use_quantile_norm: bool = False
            rlds_data_dir: str | None = None

        @dataclasses.dataclass(frozen=True)
        class DataConfigFactory:
            repo_id: str | None = None
            assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
            base_config: DataConfig | None = None

            def create_base_config(
                self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
            ) -> DataConfig:
                del assets_dirs, model_config
                base = self.base_config or DataConfig()
                return dataclasses.replace(
                    base,
                    repo_id=self.repo_id,
                    assets=self.assets,
                )

            def create(
                self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
            ) -> DataConfig:
                return self.create_base_config(assets_dirs, model_config)

        @dataclasses.dataclass(frozen=True)
        class ModelTransformFactory:
            default_prompt: str | None = None

            def __call__(self, model_config: BaseModelConfig) -> Group:
                del model_config
                return Group()

        @dataclasses.dataclass(frozen=True)
        class TrainConfig:
            name: str
            model: BaseModelConfig
            data: DataConfigFactory
            assets_dirs: pathlib.Path = pathlib.Path("assets")
            pytorch_weight_path: str | None = None
            weight_loader: object = None
            batch_size: int = 32
            num_workers: int = 2
            seed: int = 42
            num_train_steps: int = 30000
            log_interval: int = 100
            save_interval: int = 1000
            lr_schedule: object = None
            optimizer: object = None
            ema_decay: float | None = None
            freeze_filter: object = None

        openpi_mod = types.ModuleType("openpi")
        transforms_mod = types.ModuleType("openpi.transforms")
        transforms_mod.DataTransformFn = DataTransformFn
        transforms_mod.Group = Group
        transforms_mod.RepackTransform = RepackTransform
        transforms_mod.DeltaActions = DeltaActions
        transforms_mod.AbsoluteActions = AbsoluteActions
        transforms_mod.pad_to_dim = pad_to_dim
        transforms_mod.make_bool_mask = make_bool_mask

        models_mod = types.ModuleType("openpi.models")
        model_mod = types.ModuleType("openpi.models.model")
        model_mod.ModelType = ModelType
        model_mod.BaseModelConfig = BaseModelConfig

        pi0_config_mod = types.ModuleType("openpi.models.pi0_config")
        pi0_config_mod.Pi0Config = Pi0Config

        training_mod = types.ModuleType("openpi.training")
        config_mod = types.ModuleType("openpi.training.config")
        config_mod.AssetsConfig = AssetsConfig
        config_mod.DataConfig = DataConfig
        config_mod.DataConfigFactory = DataConfigFactory
        config_mod.ModelTransformFactory = ModelTransformFactory
        config_mod.TrainConfig = TrainConfig

        optimizer_mod = types.ModuleType("openpi.training.optimizer")
        optimizer_mod.CosineDecaySchedule = lambda **kw: ("cosine", kw)
        optimizer_mod.AdamW = lambda **kw: ("adamw", kw)

        weight_loaders_mod = types.ModuleType("openpi.training.weight_loaders")
        weight_loaders_mod.CheckpointWeightLoader = lambda path: ("ckpt", path)

        openpi_mod.transforms = transforms_mod
        openpi_mod.models = models_mod
        openpi_mod.training = training_mod
        models_mod.model = model_mod
        models_mod.pi0_config = pi0_config_mod
        training_mod.config = config_mod
        training_mod.optimizer = optimizer_mod
        training_mod.weight_loaders = weight_loaders_mod

        for name, mod in {
            "openpi": openpi_mod,
            "openpi.transforms": transforms_mod,
            "openpi.models": models_mod,
            "openpi.models.model": model_mod,
            "openpi.models.pi0_config": pi0_config_mod,
            "openpi.training": training_mod,
            "openpi.training.config": config_mod,
            "openpi.training.optimizer": optimizer_mod,
            "openpi.training.weight_loaders": weight_loaders_mod,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

    # Bypass heavy rlinf.models / rlinf.envs.realworld __init__ imports when ray/gym are absent.
    if importlib.util.find_spec("ray") is None or importlib.util.find_spec("gymnasium") is None:
        def _pkg(name: str, path: Path) -> types.ModuleType:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            return mod

        pkg_specs = {
            "rlinf.models": REPO_ROOT / "rlinf" / "models",
            "rlinf.models.embodiment": REPO_ROOT / "rlinf" / "models" / "embodiment",
            "rlinf.models.embodiment.openpi": (
                REPO_ROOT / "rlinf" / "models" / "embodiment" / "openpi"
            ),
            "rlinf.envs.realworld": REPO_ROOT / "rlinf" / "envs" / "realworld",
            "rlinf.envs.realworld.tianji": (
                REPO_ROOT / "rlinf" / "envs" / "realworld" / "tianji"
            ),
        }
        for name, path in pkg_specs.items():
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, _pkg(name, path))


@pytest.fixture
def tianji_dataconfig_modules(monkeypatch: pytest.MonkeyPatch):
    _install_openpi_and_lightweight_stubs(monkeypatch)
    # Force a clean import when stubs are installed so they bind consistently.
    for mod_name in list(sys.modules):
        if mod_name.startswith("rlinf.models.embodiment.openpi.dataconfig"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    tianji_dc = importlib.import_module(
        "rlinf.models.embodiment.openpi.dataconfig.tianji_dataconfig"
    )
    dc_pkg = importlib.import_module("rlinf.models.embodiment.openpi.dataconfig")
    return tianji_dc, dc_pkg


def test_tianji_repack_transform_handles_both_formats(tianji_dataconfig_modules):
    tianji_dc, _ = tianji_dataconfig_modules
    repack = tianji_dc.TianjiRepackTransform()

    head = np.zeros((16, 16, 3), dtype=np.uint8)
    left = np.full((16, 16, 3), 1, dtype=np.uint8)
    right = np.full((16, 16, 3), 2, dtype=np.uint8)
    state = np.zeros(16, dtype=np.float32)
    actions = np.ones((60, 16), dtype=np.float32)

    # 1. LeRobot split columns + extra metadata columns.
    lerobot_sample = {
        "image": head,
        "extra_view_image-0": left,
        "extra_view_image-1": right,
        "state": state,
        "actions": actions,
        "prompt": "pick up the toy car chassis",
        "episode_index": 0,
        "frame_index": 3,
        "timestamp": 0.1,
    }
    out_split = repack(lerobot_sample)
    assert set(out_split) == {
        "observation/image",
        "observation/extra_view_image-0",
        "observation/extra_view_image-1",
        "observation/state",
        "actions",
        "prompt",
    }
    assert out_split["observation/extra_view_image-0"] is left
    assert out_split["observation/extra_view_image-1"] is right

    # 2. Legacy stacked 4D column.
    stacked = np.stack([left, right], axis=0)
    legacy_sample = {
        "image": head,
        "extra_view_image": stacked,
        "state": state,
        "actions": actions,
        "task": "install wheel",
    }
    out_stacked = repack(legacy_sample)
    assert set(out_stacked) == {
        "observation/image",
        "observation/extra_view_image",
        "observation/state",
        "actions",
        "prompt",
    }
    assert out_stacked["prompt"] == "install wheel"

    # 3. Optional actions/prompt omitted cleanly.
    minimal = repack({"image": head, "extra_view_image": stacked, "state": state})
    assert "actions" not in minimal and "prompt" not in minimal

    # 4. Partial split keys or missing wrist keys fail loudly.
    with pytest.raises(KeyError):
        repack({"image": head, "extra_view_image-0": left, "state": state})
    with pytest.raises(KeyError):
        repack({"image": head, "state": state})


@pytest.mark.parametrize("layout", ["hwc", "chw"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_tianji_joint_inputs_stacked_vs_split_and_outputs_roundtrip(
    tianji_dataconfig_modules, layout, dtype
):
    tianji_dc, dc_pkg = tianji_dataconfig_modules
    cfg = dc_pkg.get_openpi_config("tianji_stage1_joint_h60")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    rng = np.random.default_rng(7)
    base_hwc = rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
    left_hwc = rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
    right_hwc = rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)

    def _encode_img(img_hwc: np.ndarray) -> np.ndarray:
        arr = img_hwc.astype(np.float32) / 255.0 if dtype is np.float32 else img_hwc.copy()
        if layout == "chw":
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    base_in = _encode_img(base_hwc)
    left_in = _encode_img(left_hwc)
    right_in = _encode_img(right_hwc)
    stacked_in = np.stack([left_in, right_in], axis=0)

    state = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    state[[7, 15]] = np.array([0.2, 0.8], dtype=np.float32)
    abs_actions = state[None, :] + rng.normal(0.0, 0.05, size=(60, 16)).astype(np.float32)
    abs_actions[:, 7] = np.linspace(0.0, 1.0, 60, dtype=np.float32)
    abs_actions[:, 15] = np.linspace(1.0, 0.0, 60, dtype=np.float32)

    def _run_inputs(sample: dict) -> dict:
        out = sample
        for fn in (
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ):
            out = fn(out)
        return out

    stacked_out = _run_inputs(
        {
            "image": base_in,
            "extra_view_image": stacked_in,
            "state": state,
            "actions": abs_actions,
            "prompt": b"place the toy car",
        }
    )
    split_out = _run_inputs(
        {
            "image": base_in,
            "extra_view_image-0": left_in,
            "extra_view_image-1": right_in,
            "state": state,
            "actions": abs_actions,
            "prompt": "place the toy car",
        }
    )

    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        np.testing.assert_array_equal(stacked_out["image"][key], split_out["image"][key])
        assert stacked_out["image"][key].dtype == np.uint8
        assert stacked_out["image"][key].shape == (12, 16, 3)
        assert stacked_out["image_mask"][key] is np.True_
        assert split_out["image_mask"][key] is np.True_

    # Verify 14 joint deltas and 2 absolute gripper closures.
    joint_dims = tianji_dc.JOINT_DIMS
    gripper_dims = tianji_dc.JOINT_GRIPPER_DIMS
    np.testing.assert_allclose(
        split_out["actions"][:, joint_dims],
        abs_actions[:, joint_dims] - state[None, joint_dims],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        split_out["actions"][:, gripper_dims],
        abs_actions[:, gripper_dims],
        atol=1e-6,
    )
    np.testing.assert_allclose(stacked_out["actions"], split_out["actions"], atol=1e-6)
    assert stacked_out["prompt"] == "place the toy car"
    assert split_out["prompt"] == "place the toy car"

    # Verify TianjiJointOutputs round-trip reconstruction from 32D padded model tensors.
    padded_actions = np.zeros((60, 32), dtype=np.float32)
    padded_actions[:, :16] = split_out["actions"]
    padded_state = np.zeros((32,), dtype=np.float32)
    padded_state[:16] = split_out["state"]

    # Also test closure clipping outside [0, 1].
    padded_actions[0, 7] = -0.2
    padded_actions[1, 15] = 1.4
    expected = abs_actions.copy()
    expected[0, 7] = 0.0
    expected[1, 15] = 1.0

    reconstructed = data_config.data_transforms.outputs[0](
        {"actions": padded_actions, "state": padded_state}
    )["actions"]
    assert reconstructed.shape == (60, 16)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-6)

    # Online/eval inference path: direct TianjiJointInputs with stacked extra_view_image and no actions.
    eval_in = data_config.data_transforms.inputs[0](
        {
            "observation/image": base_in,
            "observation/extra_view_image": stacked_in,
            "observation/state": state,
            "prompt": "place the toy car",
        }
    )
    assert "actions" not in eval_in
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        np.testing.assert_array_equal(eval_in["image"][key], stacked_out["image"][key])


def test_tianji_eef_inputs_stacked_vs_split_and_outputs_roundtrip(
    tianji_dataconfig_modules,
):
    _, dc_pkg = tianji_dataconfig_modules
    cfg = dc_pkg.get_openpi_config("tianji_stage1_h60")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    base = np.full((8, 8, 3), 10, dtype=np.uint8)
    left = np.full((8, 8, 3), 20, dtype=np.uint8)
    right = np.full((8, 8, 3), 30, dtype=np.uint8)

    state = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.25] * 2, dtype=np.float32)
    abs_actions = np.repeat(state[None, :], 60, axis=0).astype(np.float32)
    abs_actions[:, 0] += 0.05
    abs_actions[:, 10] -= 0.03
    abs_actions[:, 9] = 0.8
    abs_actions[:, 19] = 0.4

    def _run_inputs(sample: dict) -> dict:
        out = sample
        for fn in (
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ):
            out = fn(out)
        return out

    stacked_out = _run_inputs(
        {
            "image": base,
            "extra_view_image": np.stack([left, right], axis=0),
            "state": state,
            "actions": abs_actions,
            "prompt": "assemble car",
        }
    )
    split_out = _run_inputs(
        {
            "image": base,
            "extra_view_image-0": left,
            "extra_view_image-1": right,
            "state": state,
            "actions": abs_actions,
            "prompt": "assemble car",
        }
    )

    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        np.testing.assert_array_equal(stacked_out["image"][key], split_out["image"][key])
    np.testing.assert_allclose(stacked_out["actions"], split_out["actions"], atol=1e-6)
    np.testing.assert_allclose(split_out["actions"][:, 9], abs_actions[:, 9], atol=1e-6)
    np.testing.assert_allclose(split_out["actions"][:, 19], abs_actions[:, 19], atol=1e-6)

    padded_actions = np.zeros((60, 32), dtype=np.float32)
    padded_actions[:, :20] = split_out["actions"]
    reconstructed = data_config.data_transforms.outputs[0](
        {"actions": padded_actions, "state": state}
    )["actions"]
    np.testing.assert_allclose(reconstructed, abs_actions, atol=1e-5)


def test_tianji_train_configs_registration(tianji_dataconfig_modules, tmp_path):
    tianji_dc, dc_pkg = tianji_dataconfig_modules

    # Existing 30-step configs remain untouched.
    for name in ("tianji_stage1", "tianji_stage1_joint"):
        cfg = dc_pkg.get_openpi_config(name)
        assert cfg.model.pi05 is True
        assert cfg.model.action_horizon == 30
        assert cfg.model.discrete_state_input is True

    # New 60-step configs.
    h60_joint = dc_pkg.get_openpi_config("tianji_stage1_joint_h60")
    assert isinstance(h60_joint.data, tianji_dc.TianjiJointDataConfig)
    assert h60_joint.model.pi05 is True
    assert h60_joint.model.action_horizon == 60
    assert h60_joint.model.discrete_state_input is True
    assert h60_joint.model.action_dim == 32
    assert h60_joint.model.max_token_len == 200

    data_cfg = h60_joint.data.create(h60_joint.assets_dirs, h60_joint.model)
    assert data_cfg.use_quantile_norm is True
    assert data_cfg.action_sequence_keys == ("actions",)
    assert data_cfg.prompt_from_task is True

    h60_eef = dc_pkg.get_openpi_config("tianji_stage1_h60")
    assert isinstance(h60_eef.data, tianji_dc.TianjiDataConfig)
    assert h60_eef.model.pi05 is True
    assert h60_eef.model.action_horizon == 60
    assert h60_eef.model.discrete_state_input is True

    # Verify get_openpi_config overrides (used by official_sft_data_loader & calculate_norm_stats).
    ds_dir = tmp_path / "lerobot_ds"
    ds_dir.mkdir()
    norm_file = ds_dir / "norm_stats.json"
    norm_file.write_text("{}", encoding="utf-8")

    overridden = dc_pkg.get_openpi_config(
        "tianji_stage1_joint_h60",
        model_path="/tmp/pi05_base_openpi",
        batch_size=32,
        repo_id=str(ds_dir),
        data_kwargs={"norm_stats_path": str(norm_file)},
    )
    assert overridden.pytorch_weight_path == "/tmp/pi05_base_openpi"
    assert overridden.batch_size == 32
    assert overridden.data.repo_id == str(ds_dir)
    assert str(overridden.data.assets.assets_dir) == str(tmp_path)
    assert overridden.data.assets.asset_id == "lerobot_ds"


def test_tianji_joint_sft_yaml_config(tianji_dataconfig_modules, monkeypatch):
    _, dc_pkg = tianji_dataconfig_modules
    yaml_path = REPO_ROOT / "examples/sft/config/tianji_joint_sft_openpi_pi05.yaml"
    assert yaml_path.is_file()

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_raw = raw["actor"]["model"]
    assert model_raw["model_type"] == "openpi"
    assert model_raw["pi05"] is True
    assert model_raw["num_action_chunks"] == 60
    assert model_raw["action_dim"] == 16
    assert model_raw["openpi"]["config_name"] == "tianji_stage1_joint_h60"
    assert model_raw["openpi"]["action_horizon"] == 60
    assert model_raw["openpi"]["model_action_dim"] == 32
    assert model_raw["openpi"]["discrete_state_input"] is True
    assert model_raw["openpi"]["use_rlt"] is False

    # Verify action_horizon matches TrainConfig exactly (official_sft_data_loader check).
    train_cfg = dc_pkg.get_openpi_config(model_raw["openpi"]["config_name"])
    assert model_raw["openpi"]["action_horizon"] == int(train_cfg.model.action_horizon)
    assert model_raw["openpi"]["model_action_dim"] == int(train_cfg.model.action_dim)

    # Verify Hydra composition + env-var overrides.
    hydra = pytest.importorskip("hydra")
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/sft"))
    monkeypatch.setenv("TIANJI_SFT_DATA_PATH", "/tmp/lerobot_car")
    monkeypatch.setenv("TIANJI_PI05_BASE_PATH", "/tmp/pi05_base_openpi")
    monkeypatch.setenv("TIANJI_SFT_NORM_STATS_PATH", "/tmp/lerobot_car/norm_stats.json")
    with hydra.initialize_config_dir(
        version_base="1.1",
        config_dir=str(REPO_ROOT / "examples/sft/config"),
    ):
        composed = hydra.compose(config_name="tianji_joint_sft_openpi_pi05")

    assert composed.data.train_data_paths == "/tmp/lerobot_car"
    assert composed.actor.model.model_path == "/tmp/pi05_base_openpi"
    assert (
        composed.actor.model.openpi_data.norm_stats_path
        == "/tmp/lerobot_car/norm_stats.json"
    )
    assert composed.actor.model.model_type == "openpi"
    assert composed.actor.model.pi05 is True
    assert composed.actor.model.num_action_chunks == 60
    assert composed.actor.model.action_dim == 16
    assert composed.actor.model.openpi.config_name == "tianji_stage1_joint_h60"
    assert composed.actor.model.openpi.action_horizon == 60
    assert composed.actor.model.openpi.action_chunk == 60
    assert composed.actor.model.openpi.action_env_dim == 16
    assert composed.actor.model.openpi.use_rlt is False


def test_openpi_rlinf_preprocess_observation_train_rng(monkeypatch):
    """Verify openpi_rlinf preprocess_observation draws random augmentations when rng=None."""
    torch = pytest.importorskip("torch")
    TF = pytest.importorskip("torchvision.transforms.functional")

    for pkg_name, rel_dir in (
        ("rlinf.models", "rlinf/models"),
        ("rlinf.models.embodiment", "rlinf/models/embodiment"),
        ("rlinf.models.embodiment.openpi_rlinf", "rlinf/models/embodiment/openpi_rlinf"),
        (
            "rlinf.models.embodiment.openpi_rlinf.pi0_model",
            "rlinf/models/embodiment/openpi_rlinf/pi0_model",
        ),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(REPO_ROOT / rel_dir)]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

    model_mod = importlib.import_module(
        "rlinf.models.embodiment.openpi_rlinf.pi0_model.model"
    )
    Observation = model_mod.Observation
    preprocess_observation = model_mod.preprocess_observation

    gen_init = torch.Generator().manual_seed(0)
    h, w = 224, 224
    obs = Observation(
        images={
            k: torch.rand(2, h, w, 3, generator=gen_init) * 2.0 - 1.0
            for k in model_mod.IMAGE_KEYS
        },
        image_masks={},
        state=torch.zeros(2, 32),
    )

    # 1) train=False: deterministic identity for 224x224 inputs, zero crop/rotate/jitter.
    eval_a = preprocess_observation(obs, train=False)
    eval_b = preprocess_observation(
        obs, train=False, rng=torch.Generator().manual_seed(99)
    )
    for k in model_mod.IMAGE_KEYS:
        torch.testing.assert_close(eval_a.images[k], obs.images[k], rtol=0, atol=0)
        torch.testing.assert_close(eval_b.images[k], obs.images[k], rtol=0, atol=0)
        assert eval_a.image_masks[k].shape == (2,)
        assert eval_a.image_masks[k].dtype == torch.bool

    # 2) train=True, rng=Generator: deterministic for a fixed seed.
    seeded_a = preprocess_observation(
        obs, train=True, rng=torch.Generator().manual_seed(42)
    )
    seeded_b = preprocess_observation(
        obs, train=True, rng=torch.Generator().manual_seed(42)
    )
    seeded_c = preprocess_observation(
        obs, train=True, rng=torch.Generator().manual_seed(43)
    )
    for k in model_mod.IMAGE_KEYS:
        torch.testing.assert_close(
            seeded_a.images[k], seeded_b.images[k], rtol=0, atol=0
        )
    assert not torch.allclose(
        seeded_a.images["base_0_rgb"], seeded_c.images["base_0_rgb"]
    )

    # 3) train=True, rng=None: samples non-constant (top, left) crops & rotations on
    #    base_0_rgb while never cropping or rotating wrist views.
    crop_calls: list[tuple[int, int, int, int]] = []
    rotate_calls: list[float] = []
    orig_crop = TF.crop
    orig_rotate = TF.rotate

    def _spy_crop(img, top, left, height, width):
        crop_calls.append((top, left, height, width))
        return orig_crop(img, top, left, height, width)

    def _spy_rotate(img, angle, *args, **kwargs):
        rotate_calls.append(float(angle))
        return orig_rotate(img, angle, *args, **kwargs)

    monkeypatch.setattr(TF, "crop", _spy_crop)
    monkeypatch.setattr(TF, "rotate", _spy_rotate)

    torch.manual_seed(2026)
    outs = [preprocess_observation(obs, train=True, rng=None) for _ in range(8)]

    # 8 calls * batch_size 2 * 1 non-wrist camera ("base_0_rgb") == 16 crop/rotate calls
    # (confirming left_wrist_0_rgb and right_wrist_0_rgb are never cropped or rotated).
    assert len(crop_calls) == 16
    assert len(rotate_calls) == 16
    unique_tops = {c[0] for c in crop_calls}
    unique_lefts = {c[1] for c in crop_calls}
    assert len(unique_tops) > 1 and unique_tops != {0}
    assert len(unique_lefts) > 1 and unique_lefts != {0}
    assert len({round(a, 4) for a in rotate_calls}) > 1

    # Verify stochastic variation across calls on both base and wrist (color jitter).
    assert not torch.allclose(outs[0].images["base_0_rgb"], outs[1].images["base_0_rgb"])
    assert not torch.allclose(
        outs[0].images["left_wrist_0_rgb"], outs[1].images["left_wrist_0_rgb"]
    )
