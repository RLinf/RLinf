# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Independent legal action windows using the native GR00T processor.

Shuffle all legal (episode, start) pairs across the dataset. Each sample contains
the observation at start and the following H real actions. The native processor
pads targets to [40, 132]; only real action timesteps and dimensions enter loss.
Batch sizes count independent windows, which may come from different episodes.
"""

import hashlib
import inspect
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from .window_sampling import WindowIndex, WindowSampler

CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
STATE_KEYS = [
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
]
ACTION_KEYS = [
    "base_motion",
    "control_mode",
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
]
LANGUAGE_KEY = "annotation.human.task_description"


def modality_dict(horizon: int = 40) -> dict:
    """Return the RoboCasa365 modality layout for a real action horizon."""
    if type(horizon) is not int or not 1 <= horizon <= 40:
        raise ValueError(f"Action horizon must be an integer in [1, 40], got {horizon}")
    return {
        "video": {"delta_indices": [0], "modality_keys": CAMERAS},
        "state": {"delta_indices": [0], "modality_keys": STATE_KEYS},
        "action": {"delta_indices": list(range(horizon)), "modality_keys": ACTION_KEYS},
        "language": {"delta_indices": [0], "modality_keys": [LANGUAGE_KEY]},
    }


def sample_seed(seed: int, epoch: int, episode: int, purpose: int = 0) -> int:
    """Derive reproducible sample and augmentation streams."""
    return int(
        np.random.SeedSequence([seed, epoch, episode, purpose]).generate_state(1)[0]
    )


@contextmanager
def deterministic_augmentation(seed: int, processor):
    """Make sample augmentation independent of worker count and prefetching."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if hasattr(processor.train_image_transform, "set_random_seed"):
            processor.train_image_transform.set_random_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def build_processor(config: dict, bundle_path: str | None = None):
    """Build the native processor or restore its checkpointed parameters."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.types import ModalityConfig
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    loader = None
    if bundle_path:
        kwargs = json.loads(Path(bundle_path).read_text())
    else:
        base = Path(config["model"]["base_path"])
        if (base / "processor").is_dir():
            base /= "processor"
        kwargs = json.loads((base / "processor_config.json").read_text())[
            "processor_kwargs"
        ]
        modality = modality_dict(config["data"]["action_horizon"])
        loader = LeRobotEpisodeLoader(
            config["data"]["path"],
            {k: ModalityConfig(**v) for k, v in modality.items()},
            video_backend=config["data"]["video_backend"],
        )
        kwargs["modality_configs"] = {"new_embodiment": modality}
        kwargs["statistics"] = {"new_embodiment": loader.get_dataset_statistics()}
        kwargs["embodiment_id_mapping"] = {"new_embodiment": 10}
    kwargs["model_name"] = config["model"]["backbone_path"]
    kwargs["transformers_loading_kwargs"] = {"local_files_only": True}
    for name, parameter in inspect.signature(
        Gr00tN1d7Processor.__init__
    ).parameters.items():
        if name != "self" and parameter.default is not inspect.Parameter.empty:
            kwargs.setdefault(name, parameter.default)
    return Gr00tN1d7Processor(**kwargs), loader, kwargs


def validate_action_supervision(step: dict, horizon: int) -> None:
    """Require supervision only on real RoboCasa action timesteps and dimensions."""
    action, mask = step["action"], step["action_mask"]
    modality_dict(horizon)
    if tuple(action.shape) != (40, 132) or tuple(mask.shape) != (40, 132):
        raise ValueError("Expected actions and action masks padded to [40, 132]")
    expected = torch.zeros_like(mask)
    expected[:horizon, :12] = 1
    if not torch.equal(mask, expected):
        raise ValueError(
            "Action mask must supervise exactly the real action horizon and 12 dimensions"
        )


class WindowDataset(Dataset):
    """Index every legal start once and decode only its current observation."""

    def __init__(self, config: dict, processor, loader):
        self.config, self.processor, self.loader = config, processor, loader
        self.horizon = config["data"]["action_horizon"]
        modality_dict(self.horizon)
        self.index = WindowIndex(loader.episode_lengths, self.horizon)
        self._parquet_cache = {}

    def __len__(self) -> int:
        return len(self.index)

    def raw_content(self, index: tuple[int, int]):
        """Read one observation and its following real action targets."""
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag

        epoch, flat = index
        row, anchor = self.index.locate(flat)
        episode_id = self.loader.episodes_metadata[row]["episode_index"]
        if episode_id not in self._parquet_cache:
            episode = self.loader._load_parquet_data(episode_id)
            if len(episode) != self.loader.episode_lengths[row]:
                raise ValueError("Metadata and parquet episode length differ")
            self._parquet_cache[episode_id] = episode
        window = (
            self._parquet_cache[episode_id]
            .iloc[anchor : anchor + self.horizon]
            .reset_index(drop=True)
            .copy()
        )
        videos = self.loader._load_video_data(episode_id, np.asarray([anchor]))
        for camera, frames in videos.items():
            # Video deltas are zero; future observations are not model inputs.
            window[f"video.{camera}"] = [frames[0]] + [None] * (self.horizon - 1)
        content = extract_step_data(
            window,
            0,
            self.loader.modality_configs,
            EmbodimentTag.NEW_EMBODIMENT,
            allow_padding=False,
        )
        description = {
            "epoch": epoch,
            "row": row,
            "episode": episode_id,
            "window_index": flat,
            "start": anchor,
            "augmentation_seed": sample_seed(self.config["seed"], epoch, flat, 2),
        }
        return content, description

    def __getitem__(self, index: tuple[int, int] | int) -> dict:
        if isinstance(index, int):
            index = (0, index)
        content, description = self.raw_content(index)
        with deterministic_augmentation(
            description["augmentation_seed"], self.processor
        ):
            step = self.processor([{"content": content}])
        validate_action_supervision(step, self.horizon)
        return step


class WindowCollator:
    """Keep the native processor's independent-window tensor layout."""

    def __init__(self, processor):
        self.collator = processor.collator

    def __call__(self, samples: list[dict]) -> dict:
        return dict(self.collator(samples)["inputs"])


class Gr00tSftDataLoader(StatefulDataLoader):
    """Expose a planning length without truncating the infinite sample stream.

    The shared worker divides this nominal microbatch count by gradient
    accumulation. Iteration and consumed-cursor checkpointing remain owned by
    StatefulDataLoader; a nominal epoch is not an iterator boundary.
    """

    def __init__(self, dataset, *, global_batch_size: int, world_size: int, **kwargs):
        micro_batch_size = kwargs["batch_size"]
        if min(
            global_batch_size, micro_batch_size, world_size
        ) <= 0 or global_batch_size % (micro_batch_size * world_size):
            raise ValueError(
                "Global batch must be divisible by microbatch * world size"
            )
        accumulation = global_batch_size // (micro_batch_size * world_size)
        self._num_batches = accumulation * max(1, len(dataset) // global_batch_size)
        super().__init__(dataset, **kwargs)

    def __len__(self) -> int:
        return self._num_batches


def data_metadata(config: dict) -> dict:
    """Record dataset metadata hashes and the actual supervised modality."""
    root = Path(config["data"]["path"]) / "meta"
    files = [
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "stats.json",
        "modality.json",
        "embodiment.json",
    ]
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }
    rows = [json.loads(x) for x in (root / "episodes.jsonl").read_text().splitlines()]
    return {
        "sha256": hashes,
        "episodes": rows,
        "camera_order": CAMERAS,
        "state_order": STATE_KEYS,
        "action_order": ACTION_KEYS,
        "modality": modality_dict(config["data"]["action_horizon"]),
    }


def dataset_settings(cfg, data_paths=None) -> dict:
    """Resolve GR00T SFT data fields without loading the model."""
    from omegaconf import OmegaConf

    return {
        "seed": cfg.actor.seed,
        "model": {
            "base_path": cfg.actor.model.model_path,
            "backbone_path": cfg.actor.model.backbone_model_path,
        },
        "data": {
            **OmegaConf.to_container(cfg.data.gr00t_sft, resolve=True),
            "path": cfg.data.train_data_paths if data_paths is None else data_paths,
            "video_backend": cfg.data.video_backend,
        },
    }


def build_gr00t_sft_dataloader(
    cfg, world_size: int, rank: int, data_paths, eval_dataset: bool = False
):
    """Build rank-sharded windows with a restorable consumed-sample cursor."""
    if eval_dataset:
        raise NotImplementedError("Evaluation is not supported for embodied SFT")
    if not cfg.actor.model.get("native_sft", False):
        raise ValueError("GR00T SFT requires actor.model.native_sft=true")
    if not isinstance(data_paths, str) or not data_paths:
        raise ValueError("GR00T SFT requires one dataset directory")
    if cfg.runner.get("max_steps", -1) <= 0 or cfg.runner.get("max_epochs", -1) > 0:
        raise ValueError(
            "Infinite GR00T samplers require max_steps>0 and max_epochs<=0"
        )
    if cfg.data.gr00t_sft.get("sampling_mode", "random_window") != "random_window":
        raise ValueError("Only random_window sampling is supported")

    settings = dataset_settings(cfg, data_paths)
    processor, episode_loader, bundle = build_processor(settings)
    processor.train()
    expected = cfg.data.get("expected_episodes")
    if expected is not None and len(episode_loader.episodes_metadata) != expected:
        raise ValueError(
            f"Expected {expected} episodes, found {len(episode_loader.episodes_metadata)}"
        )
    dataset = WindowDataset(settings, processor, episode_loader)
    workers = cfg.data.num_workers
    loader = Gr00tSftDataLoader(
        dataset,
        global_batch_size=cfg.actor.global_batch_size,
        world_size=world_size,
        batch_size=cfg.actor.micro_batch_size,
        sampler=WindowSampler(len(dataset), cfg.actor.seed, rank, world_size),
        drop_last=False,
        collate_fn=WindowCollator(processor),
        num_workers=workers,
        generator=torch.Generator().manual_seed(cfg.actor.seed + rank),
        **({"prefetch_factor": cfg.data.prefetch_factor} if workers else {}),
    )
    metadata = data_metadata(settings)
    metadata.update(sample_count=len(dataset), sample_unit="independent_window")
    data_config = {"data": metadata, "processor": bundle}

    # This shared sidecar belongs to the run, not to individual checkpoints.
    # Construction precedes model initialization; no distributed barrier is needed.
    if rank == 0:
        logger = cfg.runner.logger
        path = Path(logger.log_path) / logger.experiment_name / "processor_bundle.json"
        content = json.dumps(data_config["processor"], indent=2, sort_keys=True)
        if path.exists():
            if json.loads(path.read_text()) != json.loads(content):
                raise ValueError(
                    f"Existing processor bundle differs from this run: {path}"
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(content + "\n")
    return loader, data_config
