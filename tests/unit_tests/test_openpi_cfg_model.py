from types import SimpleNamespace

from omegaconf import OmegaConf


class _FakeTransformGroup:
    inputs = []
    outputs = []


class _FakeDataConfig:
    data_transforms = _FakeTransformGroup()
    model_transforms = _FakeTransformGroup()
    use_quantile_norm = True

    def __init__(self, asset_id: str):
        self.asset_id = asset_id


class _FakeDataFactory:
    def __init__(self, asset_id: str):
        self.asset_id = asset_id

    def create(self, assets_dirs, model_config):
        return _FakeDataConfig(self.asset_id)


class _FakeOpenPi0Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePaligemma:
    def to_bfloat16_for_selected_params(self, dtype):
        self.dtype = dtype


class _FakeCfgModel:
    def __init__(self, config):
        self.config = config
        self.paligemma_with_expert = _FakePaligemma()
        self.loaded = False
        self.wrappers = None

    def freeze_vlm(self):
        self.frozen = True

    def load_state_dict(self, state_dict, strict=False):
        self.loaded = True

    def setup_wrappers(self, transforms, output_transforms):
        self.wrappers = (transforms, output_transforms)


def test_openpi_cfg_model_forwards_openpi_data_to_train_config(monkeypatch, tmp_path):
    """CFG OpenPI loader must use the checkpoint's dataset asset id."""
    import sys
    import types

    captured = {}

    def fake_get_openpi_config(
        config_name,
        model_path=None,
        data_kwargs=None,
    ):
        captured["config_name"] = config_name
        captured["model_path"] = model_path
        captured["data_kwargs"] = data_kwargs
        asset_id = data_kwargs["repo_id"]
        return SimpleNamespace(
            model=SimpleNamespace(train_expert_only=False),
            data=_FakeDataFactory(asset_id),
            assets_dirs=model_path,
        )

    dataconfig_module = types.ModuleType("dataconfig")
    dataconfig_module.get_openpi_config = fake_get_openpi_config
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi.dataconfig",
        dataconfig_module,
    )

    action_model_module = types.ModuleType("openpi_cfg_action_model")
    action_model_module.OpenPi0Config = _FakeOpenPi0Config
    action_model_module.OpenPi0ForCFGActionPrediction = _FakeCfgModel
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model",
        action_model_module,
    )

    download_module = types.ModuleType("download")
    download_module.maybe_download = lambda path: path
    shared_module = types.ModuleType("shared")
    shared_module.download = download_module
    openpi_module = types.ModuleType("openpi")
    openpi_module.shared = shared_module

    transforms_module = types.ModuleType("transforms")
    transforms_module.Group = _FakeTransformGroup
    transforms_module.InjectDefaultPrompt = lambda prompt: ("prompt", prompt)
    transforms_module.Normalize = lambda stats, use_quantiles: (
        "normalize",
        stats,
        use_quantiles,
    )
    transforms_module.Unnormalize = lambda stats, use_quantiles: (
        "unnormalize",
        stats,
        use_quantiles,
    )
    openpi_module.transforms = transforms_module

    checkpoints_module = types.ModuleType("checkpoints")
    checkpoints_module.load_norm_stats = lambda checkpoint_dir, asset_id: {
        "checkpoint_dir": checkpoint_dir,
        "asset_id": asset_id,
    }
    training_module = types.ModuleType("training")
    training_module.checkpoints = checkpoints_module
    openpi_module.training = training_module

    safetensors_module = types.ModuleType("safetensors")
    safetensors_torch_module = types.SimpleNamespace(
        load_model=lambda model, weight_path, strict=False: None
    )
    safetensors_module.torch = safetensors_torch_module

    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.shared", shared_module)
    monkeypatch.setitem(sys.modules, "openpi.shared.download", download_module)
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms_module)
    monkeypatch.setitem(sys.modules, "openpi.training", training_module)
    monkeypatch.setitem(
        sys.modules, "openpi.training.checkpoints", checkpoints_module
    )
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch_module)

    from rlinf.models.embodiment.openpi_cfg import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "openpi": {"config_name": "pi05_aloha_robotwin"},
            "openpi_data": {"repo_id": "pi05_sandwich_new_all"},
        }
    )

    get_model(cfg)

    assert captured == {
        "config_name": "pi05_aloha_robotwin",
        "model_path": str(tmp_path),
        "data_kwargs": {"repo_id": "pi05_sandwich_new_all"},
    }
