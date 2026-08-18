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

"""Compute OpenPI normalization statistics without decoding visual columns.

The regular OpenPI data loader materializes every feature in a LeRobot sample.
For datasets with images embedded in Parquet this makes normalization-statistics
calculation unnecessarily I/O bound. This module projects only non-visual
columns, reconstructs episode-aware action chunks, and then applies the existing
OpenPI repack and data transforms.

Example:
    python toolkits/lerobot/calculate_norm_stats_fast.py \\
        --config-name pi05_aloha_robotwin \\
        --repo-id /path/to/lerobot_dataset
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

_INDEX_KEY = "index"
_EPISODE_KEY = "episode_index"
_TASK_INDEX_KEY = "task_index"
_STATS_KEYS = ("state", "actions")
_VISUAL_DTYPES = frozenset({"image", "video"})


@dataclasses.dataclass(frozen=True)
class ProjectedData:
    """Non-visual LeRobot columns and metadata needed by the fast path."""

    columns: dict[str, np.ndarray]
    visual_features: dict[str, dict[str, Any]]
    info: dict[str, Any]

    @property
    def num_frames(self) -> int:
        return len(self.columns[_INDEX_KEY])


def _context(config_name: str, repo_id: str, dataset_root: Path) -> str:
    return (
        f"config_name={config_name!r}, repo_id={repo_id!r}, "
        f"dataset_root={str(dataset_root)!r}"
    )


def _load_info(dataset_root: Path, *, context: str) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"LeRobot metadata not found at {info_path} ({context}). "
            "Pass a local dataset path, a Hugging Face repo id cached under "
            "HF_LEROBOT_HOME, or download the dataset first."
        )
    with info_path.open() as file:
        info = json.load(file)
    if not isinstance(info.get("features"), dict):
        raise ValueError(f"LeRobot info.json has no feature mapping ({context})")
    return info


def _is_visual_arrow_field(field: pa.Field) -> bool:
    """Return whether an Arrow field has LeRobot's embedded-image structure."""

    if not pa.types.is_struct(field.type):
        return False
    child_names = {child.name for child in field.type}
    return {"bytes", "path"}.issubset(child_names)


def projection_columns(
    info: Mapping[str, Any],
    schema: pa.Schema,
    action_sequence_keys: Sequence[str],
    *,
    context: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Select every non-visual Parquet column required by the transforms."""

    features = info.get("features", {})
    visual_features = {
        name: feature
        for name, feature in features.items()
        if str(feature.get("dtype", "")).lower() in _VISUAL_DTYPES
    }
    for field in schema:
        if _is_visual_arrow_field(field):
            visual_features.setdefault(field.name, features.get(field.name, {}))

    schema_names = set(schema.names)
    required = {_INDEX_KEY, _EPISODE_KEY, *action_sequence_keys}
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(
            f"Missing required Parquet columns {missing} ({context}); "
            f"action_sequence_keys={tuple(action_sequence_keys)!r}"
        )

    projected = [name for name in schema.names if name not in visual_features]
    return projected, visual_features


def _arrow_array_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """Convert an Arrow column to NumPy without creating Python float lists."""

    array = column.combine_chunks()
    if array.null_count:
        raise ValueError(f"Column {column.type} contains null values")

    if pa.types.is_fixed_size_list(array.type):
        values = _arrow_array_to_numpy(pa.chunked_array([array.values]))
        return values.reshape(len(array), array.type.list_size, *values.shape[1:])

    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        if len(lengths) == 0:
            return np.empty((0, 0), dtype=np.float32)
        if not np.all(lengths == lengths[0]):
            raise ValueError(
                f"Variable-length list column is unsupported: lengths={np.unique(lengths)}"
            )
        values = _arrow_array_to_numpy(pa.chunked_array([array.values]))
        return values.reshape(len(array), int(lengths[0]), *values.shape[1:])

    if (
        pa.types.is_string(array.type)
        or pa.types.is_large_string(array.type)
        or pa.types.is_binary(array.type)
        or pa.types.is_large_binary(array.type)
        or pa.types.is_struct(array.type)
    ):
        return np.asarray(array.to_pylist(), dtype=object)

    return np.asarray(array.to_numpy(zero_copy_only=False))


def _read_projected_fragments(
    dataset: pads.Dataset,
    projected_columns: Sequence[str],
    *,
    expected_frames: int | None,
    context: str,
) -> dict[str, np.ndarray]:
    fragments = sorted(dataset.get_fragments(), key=lambda fragment: fragment.path)
    if not fragments:
        raise FileNotFoundError(f"No Parquet data files found ({context})")

    pieces: dict[str, list[np.ndarray]] = {name: [] for name in projected_columns}
    with tqdm.tqdm(
        total=expected_frames,
        desc="Scanning Parquet",
        unit="frames",
    ) as progress:
        for fragment in fragments:
            try:
                table = fragment.to_table(
                    columns=list(projected_columns), use_threads=True
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to project Parquet fragment {fragment.path!r} ({context})"
                ) from exc
            for name in projected_columns:
                try:
                    pieces[name].append(_arrow_array_to_numpy(table.column(name)))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to convert projected column {name!r} from "
                        f"{fragment.path!r} ({context})"
                    ) from exc
            progress.update(table.num_rows)

    columns: dict[str, np.ndarray] = {}
    for name, arrays in pieces.items():
        try:
            columns[name] = np.concatenate(arrays, axis=0)
        except ValueError as exc:
            shapes = [array.shape[1:] for array in arrays]
            raise ValueError(
                f"Inconsistent shapes for column {name!r}: {shapes} ({context})"
            ) from exc
    return columns


def _scalar_column(column: np.ndarray, name: str, *, context: str) -> np.ndarray:
    if column.ndim == 1:
        return column
    if column.ndim == 2 and column.shape[1] == 1:
        return column[:, 0]
    raise ValueError(
        f"Expected scalar column {name!r}, got shape {column.shape} ({context})"
    )


def _sort_and_validate_columns(
    columns: dict[str, np.ndarray], *, context: str
) -> dict[str, np.ndarray]:
    lengths = {name: len(column) for name, column in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Projected columns have different lengths: {lengths} ({context})"
        )
    if not lengths or next(iter(lengths.values())) < 2:
        raise ValueError(f"Dataset must contain at least two frames ({context})")

    indices = _scalar_column(columns[_INDEX_KEY], _INDEX_KEY, context=context).astype(
        np.int64, copy=False
    )
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    if np.any(np.diff(sorted_indices) != 1):
        raise ValueError(
            f"Global {_INDEX_KEY!r} values must be unique and contiguous ({context})"
        )

    sorted_columns = {name: column[order] for name, column in columns.items()}
    sorted_columns[_INDEX_KEY] = sorted_indices
    episodes = _scalar_column(
        sorted_columns[_EPISODE_KEY], _EPISODE_KEY, context=context
    ).astype(np.int64, copy=False)
    sorted_columns[_EPISODE_KEY] = episodes

    boundaries = np.flatnonzero(episodes[1:] != episodes[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    segment_episode_ids = episodes[starts]
    if len(np.unique(segment_episode_ids)) != len(segment_episode_ids):
        raise ValueError(f"An episode is split into non-adjacent segments ({context})")

    if "frame_index" in sorted_columns:
        frame_indices = _scalar_column(
            sorted_columns["frame_index"], "frame_index", context=context
        ).astype(np.int64, copy=False)
        ends = np.concatenate((boundaries, [len(episodes)]))
        for start, end in zip(starts, ends, strict=True):
            if np.any(np.diff(frame_indices[start:end]) != 1):
                episode_id = int(episodes[start])
                raise ValueError(
                    f"Non-contiguous frame_index in episode {episode_id} ({context})"
                )
        sorted_columns["frame_index"] = frame_indices

    return sorted_columns


def load_projected_data(
    dataset_root: Path,
    info: dict[str, Any],
    action_sequence_keys: Sequence[str],
    *,
    context: str,
) -> ProjectedData:
    """Load and validate all non-visual columns from a LeRobot dataset."""

    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"LeRobot data directory not found at {data_dir} ({context})"
        )
    try:
        dataset = pads.dataset(data_dir, format="parquet", exclude_invalid_files=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open Parquet dataset at {data_dir} ({context})"
        ) from exc

    projected_columns, visual_features = projection_columns(
        info,
        dataset.schema,
        action_sequence_keys,
        context=context,
    )
    columns = _read_projected_fragments(
        dataset,
        projected_columns,
        expected_frames=info.get("total_frames"),
        context=context,
    )
    columns = _sort_and_validate_columns(columns, context=context)
    return ProjectedData(columns=columns, visual_features=visual_features, info=info)


def _load_task_mapping(dataset_root: Path, *, context: str) -> dict[int, str]:
    """Load LeRobot v2 JSONL or v3 Parquet task metadata."""

    parquet_path = dataset_root / "meta" / "tasks.parquet"
    if parquet_path.is_file():
        table = pq.read_table(parquet_path)
        mapping: dict[int, str] = {}
        for row in table.to_pylist():
            task = row.get("task") or row.get("__index_level_0__")
            if task is None:
                raise ValueError(
                    f"Could not find task text in {parquet_path} ({context})"
                )
            mapping[int(row[_TASK_INDEX_KEY])] = str(task)
        return mapping

    jsonl_path = dataset_root / "meta" / "tasks.jsonl"
    if jsonl_path.is_file():
        mapping = {}
        with jsonl_path.open() as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    mapping[int(row[_TASK_INDEX_KEY])] = str(row["task"])
        return mapping

    raise FileNotFoundError(
        f"prompt_from_task=True but no tasks.parquet or tasks.jsonl exists ({context})"
    )


def _minimal_visual_shape(feature: Mapping[str, Any]) -> tuple[int, ...]:
    raw_shape = feature.get("shape")
    if not isinstance(raw_shape, list) or not raw_shape:
        return (1, 1, 3)
    shape = [max(1, int(size)) for size in raw_shape]
    names = feature.get("names")
    if (
        isinstance(names, list)
        and len(names) == len(shape)
        and all(isinstance(name, str) for name in names)
    ):
        for index, name in enumerate(names):
            if name.lower() in {"height", "width"}:
                shape[index] = 1
        return tuple(shape)
    if len(shape) >= 3 and shape[0] == 3:
        shape[1] = shape[2] = 1
    elif len(shape) >= 3 and shape[-1] == 3:
        shape[-3] = shape[-2] = 1
    elif len(shape) >= 2:
        shape[-2] = shape[-1] = 1
    return tuple(shape)


def make_visual_placeholders(
    visual_features: Mapping[str, Mapping[str, Any]], *, value: int
) -> dict[str, np.ndarray]:
    """Create shared, read-only minimal images for transform compatibility."""

    placeholders = {}
    for name, feature in visual_features.items():
        placeholder = np.full(_minimal_visual_shape(feature), value, dtype=np.uint8)
        placeholder.setflags(write=False)
        placeholders[name] = placeholder
    return placeholders


def episode_end_indices(episode_ids: np.ndarray) -> np.ndarray:
    """Return the exclusive episode end for every frame."""

    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_ids)]))
    return np.repeat(ends, ends - starts)


def action_query_indices(
    frame_indices: np.ndarray,
    episode_ends: np.ndarray,
    action_horizon: int,
) -> np.ndarray:
    """Build vectorized, episode-clamped action indices for anchor frames."""

    offsets = np.arange(action_horizon, dtype=np.int64)
    queries = frame_indices[:, None] + offsets[None, :]
    return np.minimum(queries, episode_ends[frame_indices, None] - 1)


class ProjectedLeRobotSamples:
    """Random-access transformed samples backed only by projected columns."""

    def __init__(
        self,
        projected: ProjectedData,
        data_config: DataConfig,
        action_horizon: int,
        task_mapping: Mapping[int, str] | None,
        *,
        context: str,
    ) -> None:
        self._columns = projected.columns
        self._data_config = data_config
        self._action_keys = tuple(data_config.action_sequence_keys)
        self._action_horizon = action_horizon
        self._task_mapping = task_mapping
        self._context = context
        self._episode_ends = episode_end_indices(self._columns[_EPISODE_KEY])
        self._placeholders = make_visual_placeholders(
            projected.visual_features, value=0
        )
        self._alternate_placeholders = make_visual_placeholders(
            projected.visual_features, value=1
        )
        self._transforms = (
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        )

    def __len__(self) -> int:
        return len(self._columns[_INDEX_KEY])

    def _raw_item(
        self, index: int, placeholders: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for name, column in self._columns.items():
            if name in self._action_keys:
                continue
            value = column[index]
            item[name] = value.copy() if isinstance(value, np.ndarray) else value

        query = action_query_indices(
            np.asarray([index]), self._episode_ends, self._action_horizon
        )[0]
        for name in self._action_keys:
            item[name] = self._columns[name][query].copy()
        item.update(placeholders)

        if self._data_config.prompt_from_task:
            if self._task_mapping is None:
                raise ValueError(f"Task mapping is unavailable ({self._context})")
            if _TASK_INDEX_KEY not in item:
                raise ValueError(
                    f"Missing {_TASK_INDEX_KEY!r} for prompt_from_task ({self._context})"
                )
            task_index = int(np.asarray(item[_TASK_INDEX_KEY]).item())
            try:
                item["prompt"] = self._task_mapping[task_index]
            except KeyError as exc:
                raise ValueError(
                    f"task_index={task_index} is absent from task metadata "
                    f"({self._context})"
                ) from exc
        return item

    def transform(
        self,
        index: int,
        placeholders: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        data = self._raw_item(index, placeholders or self._placeholders)
        for transform in self._transforms:
            try:
                data = transform(data)
            except Exception as exc:
                name = type(transform).__name__
                raise RuntimeError(
                    f"Transform {name} failed at frame {index} ({self._context}). "
                    "Use calculate_norm_stats.py if this transform requires real pixels."
                ) from exc

        missing = [key for key in _STATS_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"Transforms did not produce keys {missing} at frame {index} "
                f"({self._context})"
            )
        result = {key: np.asarray(data[key]) for key in _STATS_KEYS}
        for key, value in result.items():
            if value.ndim == 0 or not np.issubdtype(value.dtype, np.number):
                raise ValueError(
                    f"Transformed {key!r} must be a numeric array, got "
                    f"shape={value.shape}, dtype={value.dtype} ({self._context})"
                )
        return result

    def validate_pixel_independence(self) -> None:
        """Reject transforms whose numeric outputs change with placeholder pixels."""

        representative = sorted({0, len(self) // 2, len(self) - 1})
        for index in representative:
            zeros = self.transform(index, self._placeholders)
            ones = self.transform(index, self._alternate_placeholders)
            for key in _STATS_KEYS:
                if zeros[key].shape != ones[key].shape or not np.array_equal(
                    zeros[key], ones[key], equal_nan=True
                ):
                    raise ValueError(
                        f"Transformed {key!r} depends on visual pixel values at frame "
                        f"{index} ({self._context}). Use calculate_norm_stats.py."
                    )


def compute_norm_stats(
    dataset: ProjectedLeRobotSamples,
    batch_size: int,
    num_workers: int,
) -> dict[str, normalize.NormStats]:
    """Compute full-dataset stats, including a final partial batch."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    dataset.validate_pixel_independence()
    stats = {key: normalize.RunningStats() for key in _STATS_KEYS}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    try:
        ranges = range(0, len(dataset), batch_size)
        for start in tqdm.tqdm(
            ranges,
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Transforming and computing stats",
            unit="batches",
        ):
            indices = range(start, min(start + batch_size, len(dataset)))
            if executor is None:
                samples = [dataset.transform(index) for index in indices]
            else:
                samples = list(executor.map(dataset.transform, indices))

            for key in _STATS_KEYS:
                for sample in samples:
                    shape = sample[key].shape
                    expected = expected_shapes.setdefault(key, shape)
                    if shape != expected:
                        raise ValueError(
                            f"Inconsistent transformed {key!r} shape: expected "
                            f"{expected}, got {shape} at batch starting {start}"
                        )
                batch = np.stack([sample[key] for sample in samples]).astype(
                    np.float32, copy=False
                )
                stats[key].update(batch)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return {key: running.get_statistics() for key, running in stats.items()}


def _check_output_available(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists at {output_path}. Pass --overwrite to replace it "
            "or choose a different --output-path."
        )


def write_norm_stats_atomic(
    output_path: Path,
    norm_stats: dict[str, normalize.NormStats],
    *,
    overwrite: bool,
) -> None:
    """Serialize stats atomically, with race-safe no-overwrite behavior."""

    _check_output_available(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = normalize.serialize_json(norm_stats)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Output appeared while computing stats at {output_path}; "
                    "it was not overwritten."
                ) from exc
            temporary_path.unlink()
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(
    config_name: str,
    repo_id: str,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Compute and save non-visual OpenPI normalization statistics."""

    dataset_root = resolve_lerobot_dataset_root(repo_id)
    context = _context(config_name, repo_id, dataset_root)
    info = _load_info(dataset_root, context=context)
    target = (
        output_path.expanduser().resolve()
        if output_path is not None
        else dataset_root / "norm_stats_fast.json"
    )
    _check_output_available(target, overwrite)

    config = get_openpi_config(config_name, repo_id=repo_id)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError(f"Data config must have a repo_id ({context})")
    if data_config.rlds_data_dir is not None:
        raise ValueError(
            f"RLDS datasets are unsupported by calculate_norm_stats_fast.py ({context})"
        )
    action_keys = tuple(data_config.action_sequence_keys)
    if not action_keys:
        raise ValueError(f"Data config has no action_sequence_keys ({context})")

    projected = load_projected_data(
        dataset_root,
        info,
        action_keys,
        context=context,
    )
    task_mapping = (
        _load_task_mapping(dataset_root, context=context)
        if data_config.prompt_from_task
        else None
    )
    dataset = ProjectedLeRobotSamples(
        projected,
        data_config,
        config.model.action_horizon,
        task_mapping,
        context=context,
    )
    workers = max(0, int(config.num_workers))
    print(
        f"Computing stats for {projected.num_frames} frames with "
        f"batch_size={config.batch_size}, num_workers={workers}"
    )
    norm_stats = compute_norm_stats(dataset, config.batch_size, workers)
    print(f"Writing stats atomically to: {target}")
    write_norm_stats_atomic(target, norm_stats, overwrite=overwrite)


if __name__ == "__main__":
    tyro.cli(main)
