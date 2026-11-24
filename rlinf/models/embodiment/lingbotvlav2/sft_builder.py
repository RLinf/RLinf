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

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rlinf.models.embodiment.lingbotvlav2.utils import (
    read_checkpoint_values,
    resolve_image_size,
)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


@contextmanager
def _working_directory(path: str) -> Iterator[None]:
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_data_path(data_paths: Any) -> str:
    if isinstance(data_paths, str):
        return data_paths
    if len(data_paths) != 1:
        raise ValueError(
            "LingBot-VLA V2 SFT expects one LeRobot dataset path or one "
            "upstream multi-dataset manifest."
        )
    return str(data_paths[0])


def _resolve_path(path: str | Path, base_dir: str | Path) -> str:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def build_lingbot_v2_sft_dataloader(
    cfg: Any,
    world_size: int,
    global_rank: int,
    data_paths: Any,
) -> tuple[DataLoader, dict[str, int]]:
    """Build the upstream LingBot-VLA V2 dataset for RLinf SFT."""

    from lingbotvla.checkpoint_metadata import build_checkpoint_config
    from lingbotvla.data import build_vla_dataset
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )

    apply_lingbot_qwen3_vl_patch()

    model_cfg = cfg.actor.model
    lingbotvla_cfg = model_cfg.lingbotvlav2
    config_path = str(Path(lingbotvla_cfg.config_path).expanduser().resolve())
    training_config_path = (
        Path(lingbotvla_cfg.training_config_path).expanduser().resolve()
    )
    with training_config_path.open() as source:
        training_config = yaml.safe_load(source)

    training_model_config = dict(training_config["model"])
    training_model_config.update(training_config["train"])
    for key in (
        "bias_update_speed",
        "sequence_wise_loss_coeff",
        "router_z_loss_coeff",
        "routed_scaling_factor",
    ):
        if key in training_model_config:
            training_model_config[key] = float(training_model_config[key])
    training_model_config["align_params"] = {}
    training_model_config["attention_implementation"] = (
        lingbotvla_cfg.attention_implementation
    )
    training_model_config["tokenizer_path"] = str(model_cfg.tokenizer_path)
    training_model_config["return_image_grid_thw"] = True

    policy_config = build_checkpoint_config(
        read_checkpoint_values(
            model_cfg.model_path,
            training_config_path,
            int(model_cfg.num_action_chunks),
        ),
        model_cfg.tokenizer_path,
    )
    policy_config.n_action_steps = int(model_cfg.num_action_chunks)
    policy_config.num_steps = int(
        getattr(model_cfg, "num_steps", policy_config.num_steps)
    )
    policy_config.attention_implementation = lingbotvla_cfg.attention_implementation

    processor = build_processor(training_model_config["tokenizer_path"])
    data_config = dict(training_config["data"])
    data_config["train_path"] = _resolve_path(
        _resolve_data_path(data_paths), os.getcwd()
    )
    data_config["robot_config_root"] = _resolve_path(
        getattr(
            lingbotvla_cfg,
            "robot_config_root",
            os.path.join(config_path, "configs/robot_configs"),
        ),
        config_path,
    )
    data_config["data_name"] = getattr(
        lingbotvla_cfg, "data_name", data_config.get("data_name", "robotwin")
    )
    data_config["chunk_size"] = int(model_cfg.num_action_chunks)
    data_config["prompt_type"] = data_config.get("prompt_type", "global")
    data_config["img_size"] = resolve_image_size(data_config)
    for key in ("joints", "norm_type"):
        data_config[key] = [
            str(item) if isinstance(item, dict) else item for item in data_config[key]
        ]
    data_config["use_future_image"] = False
    dataset_config = _namespace(data_config)

    # Upstream robot configs resolve norm_stats relative to the project root.
    with _working_directory(config_path):
        dataset = build_vla_dataset(
            dataset_config=dataset_config,
            model_config=_namespace(training_model_config),
            config=policy_config,
            processor=processor,
            use_depth_align=False,
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
    )
    num_workers = int(cfg.data.get("num_workers", data_config.get("num_workers", 4)))
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return data_loader, {"num_samples": len(dataset)}
