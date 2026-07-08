import sys
import types
from types import SimpleNamespace

from omegaconf import OmegaConf


class _FakeTransformGroup:
    inputs = []
    outputs = []


class _FakeDataConfig:
    action_sequence_keys = ["action"]
    data_transforms = _FakeTransformGroup()
    model_transforms = _FakeTransformGroup()
    prompt_from_task = False
    repack_transforms = _FakeTransformGroup()
    use_quantile_norm = True
    norm_stats = {"state": {"mean": 0.0, "std": 1.0}}


class _FakeDataFactory:
    def create(self, assets_dirs, model_config):
        return _FakeDataConfig()


class _FakeDatasetMeta:
    fps = 10
    tasks = {}

    def __init__(self, repo_id, root=None):
        self.repo_id = repo_id
        self.root = root


class _FakeLeRobotDataset:
    hf_dataset = object()
    episode_data_index = {}

    def __init__(self, repo_id, root=None, episodes=None, delta_timestamps=None):
        self.repo_id = repo_id
        self.root = root
        self.episodes = episodes
        self.delta_timestamps = delta_timestamps

    def __len__(self):
        return 1


class _FakeWrappedDataset:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 1


class _FakeCfgDataLoader:
    def __init__(self, data_config, torch_data_loader):
        self._data_config = data_config
        self.torch_data_loader = torch_data_loader

    def data_config(self):
        return self._data_config


def test_cfg_dataloader_forwards_actor_openpi_data(monkeypatch, tmp_path):
    """CFG dataloader must use the checkpoint dataset asset id."""
    captured = {}

    def fake_get_openpi_config(
        config_name,
        model_path=None,
        batch_size=None,
        repo_id=None,
        data_kwargs=None,
    ):
        captured.update(
            {
                "config_name": config_name,
                "model_path": model_path,
                "batch_size": batch_size,
                "repo_id": repo_id,
                "data_kwargs": data_kwargs,
            }
        )
        return SimpleNamespace(
            batch_size=batch_size,
            model=SimpleNamespace(action_horizon=10),
            data=_FakeDataFactory(),
            assets_dirs=model_path,
        )

    lerobot_dataset_module = types.ModuleType("lerobot_dataset")
    lerobot_dataset_module.LeRobotDatasetMetadata = _FakeDatasetMeta
    lerobot_dataset_module.LeRobotDataset = _FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.common", types.ModuleType("common"))
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets", types.ModuleType("datasets"))
    monkeypatch.setitem(
        sys.modules,
        "lerobot.common.datasets.lerobot_dataset",
        lerobot_dataset_module,
    )
    sys.modules["lerobot"].__path__ = []
    sys.modules["lerobot.common"].__path__ = []
    sys.modules["lerobot.common.datasets"].__path__ = []
    sys.modules["lerobot"].common = sys.modules["lerobot.common"]
    sys.modules["lerobot.common"].datasets = sys.modules["lerobot.common.datasets"]
    sys.modules["lerobot.common.datasets"].lerobot_dataset = lerobot_dataset_module

    transforms_module = types.ModuleType("transforms")
    transforms_module.Normalize = lambda stats, use_quantiles: (
        "normalize",
        stats,
        use_quantiles,
    )
    openpi_data_loader_module = types.ModuleType("data_loader")
    openpi_data_loader_module.TransformedDataset = _FakeWrappedDataset
    monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
    monkeypatch.setitem(sys.modules, "openpi.training", types.ModuleType("training"))
    monkeypatch.setitem(
        sys.modules, "openpi.training.data_loader", openpi_data_loader_module
    )
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms_module)
    sys.modules["openpi"].__path__ = []
    sys.modules["openpi.training"].__path__ = []
    sys.modules["openpi"].training = sys.modules["openpi.training"]
    sys.modules["openpi"].transforms = transforms_module
    sys.modules["openpi.training"].data_loader = openpi_data_loader_module

    dataconfig_module = types.ModuleType("dataconfig")
    dataconfig_module.get_openpi_config = fake_get_openpi_config
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi.dataconfig",
        dataconfig_module,
    )

    utils_module = types.ModuleType("utils")
    utils_module.cast_image_features = lambda dataset: dataset
    monkeypatch.setitem(
        sys.modules,
        "rlinf.data.datasets.recap.utils",
        utils_module,
    )

    cfg_model_module = types.ModuleType("cfg_model")
    cfg_model_module.AdvantagePreservingDataset = _FakeWrappedDataset
    cfg_model_module.CFGDataLoaderImpl = _FakeCfgDataLoader
    cfg_model_module.CfgMixtureDataset = _FakeWrappedDataset
    cfg_model_module.TokenizePromptWithGuidance = object
    monkeypatch.setitem(
        sys.modules,
        "rlinf.data.datasets.recap.cfg_model",
        cfg_model_module,
    )

    from rlinf.workers.sft import fsdp_cfg_worker
    from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker

    monkeypatch.setattr(fsdp_cfg_worker, "cast_image_features", lambda dataset: dataset)
    monkeypatch.setattr(
        FSDPCfgWorker,
        "_load_advantages_lookup",
        staticmethod(lambda data_path, advantage_tag=None: {(0, 0): True}),
    )

    worker = object.__new__(FSDPCfgWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 2,
                "model": {
                    "model_path": str(tmp_path / "checkpoint"),
                    "openpi": {"config_name": "pi05_aloha_robotwin"},
                },
                "openpi_data": {"repo_id": "pi05_sandwich_new_all"},
            },
            "data": {
                "advantage_tag": "sandwich",
                "train_data_paths": [
                    {
                        "dataset_path": str(tmp_path / "sandwich_lerobot"),
                        "weight": 1.0,
                    }
                ],
            },
        }
    )
    worker._world_size = 1
    worker._rank = 0
    worker._build_model_transforms = lambda data_config: []
    worker._create_torch_dataloader = lambda dataset, config, loader: object()
    worker.log_info = lambda message: None

    worker.build_dataloader()

    assert captured == {
        "config_name": "pi05_aloha_robotwin",
        "model_path": str(tmp_path / "checkpoint"),
        "batch_size": 2,
        "repo_id": str(tmp_path / "sandwich_lerobot"),
        "data_kwargs": {"repo_id": "pi05_sandwich_new_all"},
    }
