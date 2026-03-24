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
import pathlib
import sys
import numpy as np


def _extract_cli_flag_value(args: list[str], flag: str) -> str | None:
    """Extract a CLI flag value from raw argv before importing openpi modules."""
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _set_hf_lerobot_home(dataset_dir: str) -> pathlib.Path:
    dataset_path = pathlib.Path(dataset_dir).expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(dataset_path.parent)
    return dataset_path


# NOTE: openpi may read HF_LEROBOT_HOME during import/module initialization.
# Parse --dataset-dir from raw argv so local datasets work without exporting the
# env var manually in the shell.
_raw_dataset_dir = _extract_cli_flag_value(sys.argv[1:], "--dataset-dir")
if _raw_dataset_dir is None:
    _raw_dataset_dir = _extract_cli_flag_value(sys.argv[1:], "--dataset_dir")
if _raw_dataset_dir:
    _set_hf_lerobot_home(_raw_dataset_dir)

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    model_path: str | None = None,
    dataset_dir: str | None = None,
    repo_id: str | None = None,
    batch_size: int | None = None,
    max_frames: int | None = None,
):
    data_kwargs = {}
    if dataset_dir is not None:
        dataset_path = _set_hf_lerobot_home(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
        data_kwargs["repo_id"] = repo_id or dataset_path.name
    elif repo_id is not None:
        data_kwargs["repo_id"] = repo_id

    config = get_openpi_config(
        config_name,
        model_path=model_path,
        data_kwargs=data_kwargs or None,
        batch_size=batch_size,
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
