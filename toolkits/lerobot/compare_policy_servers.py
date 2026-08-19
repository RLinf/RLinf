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
"""Compare OpenPI-compatible policy servers on real LeRobot observations.

This is a black-box deployment diagnostic. It deliberately compares the common
action prefix when the servers return different action horizons; it is not a
deterministic implementation-parity test.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import importlib
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

IMAGE_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
REQUEST_IMAGE_KEYS = tuple(
    key.removeprefix("observation.images.") for key in IMAGE_COLUMNS
)
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"


def _optional_import(module: str, install: str, purpose: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"{purpose} requires optional dependency {module!r}. "
            f"Install it with: {install}"
        ) from exc


def _require_pyarrow() -> tuple[Any, Any]:
    pa = _optional_import(
        "pyarrow", "python -m pip install pyarrow", "LeRobot Parquet loading"
    )
    pq = _optional_import(
        "pyarrow.parquet", "python -m pip install pyarrow", "LeRobot Parquet loading"
    )
    return pa, pq


def _require_pillow() -> Any:
    return _optional_import(
        "PIL.Image", "python -m pip install Pillow", "Embedded image decoding"
    )


def _require_plotting() -> Any:
    return _optional_import(
        "matplotlib.pyplot",
        "python -m pip install matplotlib",
        "Diagnostic plot generation",
    )


def _require_openpi_protocol() -> tuple[Any, Any]:
    msgpack_numpy = _optional_import(
        "openpi_client.msgpack_numpy",
        "python -m pip install openpi-client",
        "OpenPI WebSocket serialization",
    )
    websocket_client = _optional_import(
        "websockets.sync.client",
        "python -m pip install websockets openpi-client",
        "OpenPI WebSocket communication",
    )
    return msgpack_numpy, websocket_client


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    """Location and prompt metadata for one contiguous LeRobot episode."""

    episode_id: int
    task_index: int
    prompt: str
    path: Path
    row_start: int
    row_stop: int

    @property
    def num_frames(self) -> int:
        return self.row_stop - self.row_start


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    """Small non-visual columns for one episode."""

    ref: EpisodeRef
    state: np.ndarray
    action: np.ndarray


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    """Validated server response plus client and server timing."""

    actions: np.ndarray
    rtt_ms: float
    server_infer_ms: float | None
    response: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PairedResult:
    """One repeated, same-observation response pair."""

    prompt: str
    episode_id: int
    frame_index: int
    repeat: int
    state: np.ndarray
    truth: np.ndarray
    openpi: InferenceResult
    rlinf: InferenceResult


class PolicyServerClient:
    """Persistent client for the OpenPI msgpack-over-WebSocket protocol."""

    def __init__(self, host: str, port: int, timeout: float, name: str) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.name = name
        self.uri = (
            host
            if host.startswith("ws://") or host.startswith("wss://")
            else f"ws://{host}:{port}"
        )
        self.metadata: dict[str, Any] = {}
        self._connection: Any | None = None
        self._msgpack: Any | None = None

    def __enter__(self) -> PolicyServerClient:
        msgpack_numpy, websocket_client = _require_openpi_protocol()
        self._msgpack = msgpack_numpy
        try:
            self._connection = websocket_client.connect(
                self.uri,
                compression=None,
                max_size=None,
                open_timeout=self.timeout,
                close_timeout=min(self.timeout, 2.0),
            )
            payload = self._connection.recv(timeout=self.timeout)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Could not connect to {self.name} policy server at {self.uri}: {exc}"
            ) from exc
        if isinstance(payload, str):
            self.close()
            raise RuntimeError(
                f"{self.name} policy server sent text instead of handshake metadata: "
                f"{payload}"
            )
        metadata = msgpack_numpy.unpackb(payload)
        if not isinstance(metadata, dict):
            self.close()
            raise RuntimeError(
                f"{self.name} policy server metadata must be a mapping, got "
                f"{type(metadata).__name__}"
            )
        self.metadata = metadata
        return self

    def infer(self, observation: dict[str, Any]) -> InferenceResult:
        if self._connection is None or self._msgpack is None:
            raise RuntimeError(f"{self.name} policy server client is not connected")
        started = time.perf_counter()
        try:
            self._connection.send(self._msgpack.packb(observation))
            payload = self._connection.recv(timeout=self.timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Timed out after {self.timeout:g}s waiting for {self.name} at {self.uri}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Connection to {self.name} policy server at {self.uri} failed: {exc}"
            ) from exc
        rtt_ms = (time.perf_counter() - started) * 1000.0
        if isinstance(payload, str):
            raise RuntimeError(
                f"{self.name} policy server returned a text error:\n{payload}"
            )
        response = self._msgpack.unpackb(payload)
        if not isinstance(response, dict):
            raise RuntimeError(
                f"{self.name} response must be a mapping, got {type(response).__name__}"
            )
        if "actions" not in response:
            raise RuntimeError(
                f"{self.name} response is missing required key 'actions'"
            )
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or not all(size > 0 for size in actions.shape):
            raise RuntimeError(
                f"{self.name} actions must have shape [horizon, action_dim], got "
                f"{actions.shape}"
            )
        if not np.issubdtype(actions.dtype, np.number):
            raise RuntimeError(
                f"{self.name} actions must be numeric, got {actions.dtype}"
            )
        actions = actions.astype(np.float64, copy=False)
        if not np.all(np.isfinite(actions)):
            bad = np.argwhere(~np.isfinite(actions))[0].tolist()
            raise RuntimeError(
                f"{self.name} actions contain a non-finite value at index {bad}"
            )
        timing = response.get("server_timing", {})
        server_ms = timing.get("infer_ms") if isinstance(timing, dict) else None
        if server_ms is not None:
            server_ms = float(server_ms)
        return InferenceResult(actions, rtt_ms, server_ms, response)

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(Exception):
                self._connection.close()
            self._connection = None

    def __exit__(self, *_: object) -> None:
        self.close()


def load_task_prompts(dataset_root: Path) -> dict[int, str]:
    """Read LeRobot v3 task prompts, including pandas index-formatted tables."""
    _, pq = _require_pyarrow()
    path = dataset_root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot task metadata does not exist: {path}")
    table = pq.read_table(path)
    columns = table.column_names
    if "task_index" not in columns:
        raise ValueError(f"Task metadata is missing 'task_index': {path}")
    text_columns = [
        name for name in ("task", "prompt", "__index_level_0__") if name in columns
    ]
    if not text_columns:
        raise ValueError(
            f"Task metadata must contain task, prompt, or __index_level_0__: {path}"
        )
    indices = table["task_index"].to_pylist()
    prompts = table[text_columns[0]].to_pylist()
    mapping = {
        int(index): str(prompt) for index, prompt in zip(indices, prompts, strict=True)
    }
    if len(mapping) != len(indices):
        raise ValueError(f"Task metadata contains duplicate task_index values: {path}")
    return mapping


def _scalar_array(column: Any) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_pylist())


def scan_episodes(dataset_root: Path, prompts: dict[int, str]) -> list[EpisodeRef]:
    """Scan only scalar locator columns and return deterministic episode refs."""
    _, pq = _require_pyarrow()
    data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(
            f"No Parquet data files found under {dataset_root / 'data'}"
        )
    refs: list[EpisodeRef] = []
    seen: set[int] = set()
    required = ("episode_index", "frame_index", "task_index")
    for path in data_paths:
        parquet = pq.ParquetFile(path)
        missing = sorted(set(required) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path} is missing locator columns: {missing}")
        table = parquet.read(columns=list(required))
        episode_ids = _scalar_array(table["episode_index"]).astype(np.int64)
        frame_ids = _scalar_array(table["frame_index"]).astype(np.int64)
        task_ids = _scalar_array(table["task_index"]).astype(np.int64)
        for episode_id in np.unique(episode_ids):
            rows = np.flatnonzero(episode_ids == episode_id)
            start, stop = int(rows[0]), int(rows[-1]) + 1
            if not np.array_equal(rows, np.arange(start, stop)):
                raise ValueError(f"Episode {episode_id} is not contiguous in {path}")
            if int(episode_id) in seen:
                raise ValueError(
                    f"Episode {episode_id} spans multiple Parquet files; consolidate it "
                    "before running this diagnostic"
                )
            seen.add(int(episode_id))
            episode_frames = frame_ids[rows]
            if not np.array_equal(episode_frames, np.arange(len(rows))):
                raise ValueError(
                    f"Episode {episode_id} frame_index must be contiguous from zero in {path}"
                )
            unique_tasks = np.unique(task_ids[rows])
            if len(unique_tasks) != 1:
                raise ValueError(
                    f"Episode {episode_id} has multiple task_index values in {path}"
                )
            task_index = int(unique_tasks[0])
            if task_index not in prompts:
                raise ValueError(
                    f"Episode {episode_id} references unknown task_index {task_index}"
                )
            refs.append(
                EpisodeRef(
                    episode_id=int(episode_id),
                    task_index=task_index,
                    prompt=prompts[task_index],
                    path=path,
                    row_start=start,
                    row_stop=stop,
                )
            )
    return sorted(refs, key=lambda ref: ref.episode_id)


def select_episodes(
    refs: Sequence[EpisodeRef],
    *,
    episodes_per_prompt: int,
    prompt_regex: str | None,
    episode_ids: set[int] | None,
    seed: int,
) -> list[EpisodeRef]:
    """Select a seeded, stable set of episodes grouped by prompt."""
    if episodes_per_prompt < 1:
        raise ValueError("episodes_per_prompt must be at least 1")
    pattern = re.compile(prompt_regex) if prompt_regex else None
    grouped: dict[str, list[EpisodeRef]] = defaultdict(list)
    for ref in refs:
        if episode_ids is not None and ref.episode_id not in episode_ids:
            continue
        if pattern is not None and pattern.search(ref.prompt) is None:
            continue
        grouped[ref.prompt].append(ref)
    if not grouped:
        raise ValueError("No episodes match the prompt and episode filters")
    rng = np.random.default_rng(seed)
    selected: list[EpisodeRef] = []
    for prompt in sorted(grouped):
        candidates = sorted(grouped[prompt], key=lambda ref: ref.episode_id)
        count = min(episodes_per_prompt, len(candidates))
        positions = np.sort(rng.choice(len(candidates), size=count, replace=False))
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def uniform_frame_indices(num_frames: int, count: int) -> np.ndarray:
    """Return unique, evenly spaced frame indices, including both endpoints."""
    if num_frames < 1 or count < 1:
        raise ValueError("num_frames and count must be positive")
    count = min(num_frames, count)
    return np.unique(np.rint(np.linspace(0, num_frames - 1, count)).astype(np.int64))


def replay_frame_indices(
    num_frames: int, horizon: int, chunks: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose consecutive full-horizon replanning anchors within an episode."""
    if horizon < 1 or chunks < 1:
        raise ValueError("horizon and chunks must be positive")
    maximum_start = max(0, num_frames - 1 - horizon * (chunks - 1))
    start = int(rng.integers(maximum_start + 1))
    return np.minimum(start + np.arange(chunks) * horizon, num_frames - 1)


def _vector_column(table: Any, name: str, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def load_episode_numeric(ref: EpisodeRef) -> EpisodeData:
    """Load state and actions without touching image columns."""
    _, pq = _require_pyarrow()
    parquet = pq.ParquetFile(ref.path)
    missing = sorted({STATE_COLUMN, ACTION_COLUMN} - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{ref.path} is missing required columns: {missing}")
    table = parquet.read(columns=[STATE_COLUMN, ACTION_COLUMN]).slice(
        ref.row_start, ref.num_frames
    )
    state = _vector_column(table, STATE_COLUMN)
    action = _vector_column(table, ACTION_COLUMN)
    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(
            f"Episode {ref.episode_id} state must have shape [N, 14], got {state.shape}"
        )
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(
            f"Episode {ref.episode_id} action must have shape [N, 14], got {action.shape}"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError(
            f"Episode {ref.episode_id} contains non-finite state or action"
        )
    return EpisodeData(ref=ref, state=state, action=action)


def _read_selected_cells(
    path: Path, column: str, row_indices: Sequence[int]
) -> list[Any]:
    """Read one column and only row groups containing requested rows."""
    pa, pq = _require_pyarrow()
    parquet = pq.ParquetFile(path)
    requested = np.asarray(row_indices, dtype=np.int64)
    if len(requested) == 0:
        return []
    output: dict[int, Any] = {}
    offset = 0
    for row_group in range(parquet.num_row_groups):
        row_count = parquet.metadata.row_group(row_group).num_rows
        mask = (requested >= offset) & (requested < offset + row_count)
        if np.any(mask):
            locals_ = requested[mask] - offset
            values = parquet.read_row_group(row_group, columns=[column])[column]
            selected = values.take(pa.array(locals_)).to_pylist()
            output.update(zip(requested[mask].tolist(), selected, strict=True))
        offset += row_count
    missing = [int(index) for index in requested if int(index) not in output]
    if missing:
        raise IndexError(f"Rows {missing} are outside {path}")
    return [output[int(index)] for index in requested]


def _decode_image(value: Any, dataset_root: Path) -> np.ndarray:
    image_module = _require_pillow()
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        image_bytes, image_path = bytes(value), None
    else:
        raise ValueError(f"Unsupported embedded image value: {type(value).__name__}")
    if image_bytes is not None:
        source: Any = io.BytesIO(image_bytes)
    elif image_path:
        source = dataset_root / image_path
    else:
        raise ValueError("Image value contains neither bytes nor path")
    with image_module.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_observations(
    dataset_root: Path, episode: EpisodeData, frame_indices: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Load selected frames one image column at a time and build server requests."""
    local_indices = np.unique(np.asarray(frame_indices, dtype=np.int64))
    if np.any(local_indices < 0) or np.any(local_indices >= episode.ref.num_frames):
        raise IndexError(f"Frame indices are outside episode {episode.ref.episode_id}")
    file_indices = episode.ref.row_start + local_indices
    images_by_key: dict[str, list[np.ndarray]] = {}
    for column, request_key in zip(IMAGE_COLUMNS, REQUEST_IMAGE_KEYS, strict=True):
        values = _read_selected_cells(episode.ref.path, column, file_indices)
        images_by_key[request_key] = [
            _decode_image(value, dataset_root) for value in values
        ]
    observations: dict[int, dict[str, Any]] = {}
    for position, frame_index in enumerate(local_indices):
        observations[int(frame_index)] = {
            "images": {key: images_by_key[key][position] for key in REQUEST_IMAGE_KEYS}
        }
        observations[int(frame_index)]["state"] = episode.state[frame_index].copy()
        observations[int(frame_index)]["prompt"] = episode.ref.prompt
    return observations


def format_observation_layout(
    observation: dict[str, Any], layout: str
) -> dict[str, Any]:
    """Format canonical HWC images for one server without changing content."""
    if layout not in {"chw", "hwc"}:
        raise ValueError(f"Unsupported image layout: {layout}")
    images = observation["images"]
    return {
        "images": {
            key: np.moveaxis(value, -1, 0) if layout == "chw" else value
            for key, value in images.items()
        },
        "state": observation["state"],
        "prompt": observation["prompt"],
    }


def padded_action_chunk(
    actions: np.ndarray, frame_index: int, horizon: int
) -> np.ndarray:
    """Build an episode-local ground-truth chunk with tail-repeat padding."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    indices = np.minimum(frame_index + np.arange(horizon), len(actions) - 1)
    return np.asarray(actions[indices], dtype=np.float64)


def common_prefix_metrics(
    left: np.ndarray, right: np.ndarray, joint_std: np.ndarray
) -> dict[str, Any]:
    """Compute errors over the shared action horizon and dimensions."""
    horizon = min(left.shape[0], right.shape[0])
    dimensions = min(left.shape[1], right.shape[1], len(joint_std))
    if horizon == 0 or dimensions == 0:
        raise ValueError("Action chunks have no common prefix")
    difference = np.asarray(left[:horizon, :dimensions] - right[:horizon, :dimensions])
    absolute = np.abs(difference)
    safe_std = np.maximum(np.asarray(joint_std[:dimensions], dtype=np.float64), 1e-12)
    return {
        "common_horizon": horizon,
        "common_action_dim": dimensions,
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(absolute.max()),
        "normalized_mae": float((absolute / safe_std[None, :]).mean()),
        "per_joint_mae": absolute.mean(axis=0),
        "per_horizon_mae": absolute.mean(axis=1),
        "absolute_difference": absolute,
    }


def randomness_metrics(chunks: Sequence[np.ndarray]) -> dict[str, float]:
    """Summarize repeated stochastic outputs over their common prefix."""
    if not chunks:
        raise ValueError("At least one action chunk is required")
    horizon = min(chunk.shape[0] for chunk in chunks)
    dimensions = min(chunk.shape[1] for chunk in chunks)
    stacked = np.stack([chunk[:horizon, :dimensions] for chunk in chunks])
    output_std = float(stacked.std(axis=0).mean())
    pairwise = [
        np.abs(stacked[left] - stacked[right]).mean()
        for left in range(len(stacked))
        for right in range(left + 1, len(stacked))
    ]
    return {
        "output_std": output_std,
        "repeat_to_repeat_mae": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def action_1_to_10_jump_metrics(chunks: Sequence[np.ndarray]) -> dict[str, Any]:
    """Summarize net action change from one-based steps 1 to 10."""
    if not chunks:
        raise ValueError("At least one action chunk is required")
    if any(chunk.ndim != 2 or chunk.shape[0] < 10 for chunk in chunks):
        raise ValueError(
            "Every action chunk must have shape [horizon >= 10, action_dim]"
        )
    dimensions = min(chunk.shape[1] for chunk in chunks)
    deltas = np.stack(
        [chunk[9, :dimensions] - chunk[0, :dimensions] for chunk in chunks]
    )
    absolute = np.abs(deltas)
    per_joint = absolute.mean(axis=0)
    return {
        "num_chunks": len(chunks),
        "action_dim": dimensions,
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(deltas).mean())),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(absolute.max()),
        "mean_l2": float(np.linalg.norm(deltas, axis=1).mean()),
        "worst_joint": int(np.argmax(per_joint)),
        "worst_joint_mae": float(per_joint.max()),
        "per_joint_mae": per_joint,
    }


def chunk_jitter_metrics(
    chunk: np.ndarray, state: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    """Compute state jump, temporal differences, and data error for one chunk."""
    dimensions = min(chunk.shape[1], len(state), truth.shape[1])
    horizon = min(chunk.shape[0], truth.shape[0])
    first_difference = np.diff(chunk[:, :dimensions], axis=0)
    second_difference = np.diff(chunk[:, :dimensions], n=2, axis=0)
    return {
        "first_action_state_jump": float(
            np.abs(chunk[0, :dimensions] - state[:dimensions]).mean()
        ),
        "first_difference_mae": float(np.abs(first_difference).mean())
        if len(first_difference)
        else 0.0,
        "second_difference_mae": float(np.abs(second_difference).mean())
        if len(second_difference)
        else 0.0,
        "ground_truth_mae": float(
            np.abs(chunk[:horizon, :dimensions] - truth[:horizon, :dimensions]).mean()
        ),
    }


def replanning_boundary_metrics(chunks: Sequence[np.ndarray]) -> dict[str, float]:
    """Measure jumps between the last and first actions of adjacent chunks."""
    jumps = []
    for previous, current in zip(chunks, chunks[1:]):
        dimensions = min(previous.shape[1], current.shape[1])
        jumps.append(np.abs(previous[-1, :dimensions] - current[0, :dimensions]))
    if not jumps:
        return {"boundary_jump_mae": 0.0, "boundary_jump_max": 0.0}
    absolute = np.concatenate(jumps)
    return {
        "boundary_jump_mae": float(absolute.mean()),
        "boundary_jump_max": float(absolute.max()),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if np.isscalar(value)}


def _aggregate_pair_metrics(
    paired: Sequence[PairedResult], joint_std: np.ndarray
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    sample_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    by_prompt: dict[str, list[float]] = defaultdict(list)
    absolute_differences: list[np.ndarray] = []
    for result in paired:
        metrics = common_prefix_metrics(
            result.openpi.actions, result.rlinf.actions, joint_std
        )
        identity = {
            "prompt": result.prompt,
            "episode_id": result.episode_id,
            "frame_index": result.frame_index,
            "repeat": result.repeat,
        }
        sample_rows.append(
            {
                **identity,
                **_metric_scalars(metrics),
                "openpi_horizon": result.openpi.actions.shape[0],
                "rlinf_horizon": result.rlinf.actions.shape[0],
                "openpi_action_dim": result.openpi.actions.shape[1],
                "rlinf_action_dim": result.rlinf.actions.shape[1],
                "openpi_rtt_ms": result.openpi.rtt_ms,
                "rlinf_rtt_ms": result.rlinf.rtt_ms,
                "openpi_server_infer_ms": result.openpi.server_infer_ms,
                "rlinf_server_infer_ms": result.rlinf.server_infer_ms,
            }
        )
        by_prompt[result.prompt].append(metrics["mae"])
        absolute_differences.append(metrics["absolute_difference"])
        for joint, mae in enumerate(metrics["per_joint_mae"]):
            joint_rows.append({**identity, "joint": joint, "mae": float(mae)})
        for step, mae in enumerate(metrics["per_horizon_mae"]):
            horizon_rows.append({**identity, "horizon_step": step, "mae": float(mae)})
    common_horizon = min(array.shape[0] for array in absolute_differences)
    common_dim = min(array.shape[1] for array in absolute_differences)
    absolute = np.stack(
        [array[:common_horizon, :common_dim] for array in absolute_differences]
    )
    per_joint = absolute.mean(axis=(0, 1))
    per_horizon = absolute.mean(axis=(0, 2))
    prompt_mae = {
        prompt: float(np.mean(values)) for prompt, values in by_prompt.items()
    }
    worst_prompt = max(prompt_mae, key=prompt_mae.get)
    return (
        {
            "num_pairs": len(paired),
            "common_horizon": common_horizon,
            "common_action_dim": common_dim,
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
            "p95_abs": float(np.percentile(absolute, 95)),
            "max_abs": float(absolute.max()),
            "normalized_mae": float(
                (
                    absolute / np.maximum(joint_std[:common_dim], 1e-12)[None, None, :]
                ).mean()
            ),
            "prompt_mae": prompt_mae,
            "worst_prompt": {"prompt": worst_prompt, "mae": prompt_mae[worst_prompt]},
            "worst_joint": {
                "joint": int(np.argmax(per_joint)),
                "mae": float(per_joint.max()),
            },
            "worst_horizon_step": {
                "horizon_step": int(np.argmax(per_horizon)),
                "mae": float(per_horizon.max()),
            },
            "mean_absolute_difference": absolute.mean(axis=0),
        },
        sample_rows,
        joint_rows,
        horizon_rows,
    )


def _randomness_summary(paired: Sequence[PairedResult]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[PairedResult]] = defaultdict(list)
    for result in paired:
        grouped[(result.prompt, result.episode_id, result.frame_index)].append(result)
    summary: dict[str, Any] = {}
    for server in ("openpi", "rlinf"):
        metrics = [
            randomness_metrics([getattr(item, server).actions for item in items])
            for items in grouped.values()
        ]
        summary[server] = {
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in ("output_std", "repeat_to_repeat_mae")
        }
    return summary


def _action_1_to_10_summary(paired: Sequence[PairedResult]) -> dict[str, Any]:
    """Compare steps 1 and 10 on identical paired observations."""
    output = {
        server: action_1_to_10_jump_metrics(
            [getattr(item, server).actions for item in paired]
        )
        for server in ("openpi", "rlinf")
    }
    dimensions = min(item.openpi.actions.shape[1] for item in paired)
    dimensions = min(dimensions, min(item.rlinf.actions.shape[1] for item in paired))
    displacement_difference = np.stack(
        [
            (item.openpi.actions[9, :dimensions] - item.openpi.actions[0, :dimensions])
            - (item.rlinf.actions[9, :dimensions] - item.rlinf.actions[0, :dimensions])
            for item in paired
        ]
    )
    absolute = np.abs(displacement_difference)
    output["paired_displacement_difference"] = {
        "num_pairs": len(paired),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(displacement_difference).mean())),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(absolute.max()),
    }
    output["rlinf_to_openpi_mae_ratio"] = (
        output["rlinf"]["mae"] / output["openpi"]["mae"]
    )
    return output


def _latency_summary(paired: Sequence[PairedResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for server in ("openpi", "rlinf"):
        values = [getattr(item, server) for item in paired]
        rtt = np.asarray([item.rtt_ms for item in values])
        infer = np.asarray(
            [
                item.server_infer_ms
                for item in values
                if item.server_infer_ms is not None
            ]
        )
        output[server] = {
            "rtt_ms_mean": float(rtt.mean()),
            "rtt_ms_p95": float(np.percentile(rtt, 95)),
            "server_infer_ms_mean": float(infer.mean()) if len(infer) else None,
            "server_infer_ms_p95": float(np.percentile(infer, 95))
            if len(infer)
            else None,
        }
    return output


def _plot_outputs(
    output_dir: Path,
    pair_summary: dict[str, Any],
    paired: Sequence[PairedResult],
    jitter_rows: Sequence[dict[str, Any]],
) -> None:
    plt = _require_plotting()
    heatmap = np.asarray(pair_summary["mean_absolute_difference"])
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(heatmap.T, aspect="auto", origin="lower")
    axis.set(
        xlabel="Horizon step", ylabel="Joint", title="Mean absolute server difference"
    )
    figure.colorbar(image, ax=axis, label="Absolute difference")
    figure.tight_layout()
    figure.savefig(output_dir / "difference_heatmap.png", dpi=160)
    plt.close(figure)

    worst = max(
        paired,
        key=lambda item: common_prefix_metrics(
            item.openpi.actions,
            item.rlinf.actions,
            np.ones(min(item.openpi.actions.shape[1], item.rlinf.actions.shape[1])),
        )["mae"],
    )
    dimensions = min(14, worst.openpi.actions.shape[1], worst.rlinf.actions.shape[1])
    figure, axes = plt.subplots(7, 2, figsize=(12, 18), sharex=True)
    truth_openpi = padded_action_chunk(worst.truth, 0, worst.openpi.actions.shape[0])
    truth_rlinf = padded_action_chunk(worst.truth, 0, worst.rlinf.actions.shape[0])
    for joint, axis in enumerate(axes.flat[:dimensions]):
        axis.plot(worst.openpi.actions[:, joint], label="OpenPI")
        axis.plot(worst.rlinf.actions[:, joint], label="RLinf")
        axis.plot(truth_openpi[:, joint], linestyle="--", label="Dataset/OpenPI H")
        if len(truth_rlinf) != len(truth_openpi):
            axis.plot(truth_rlinf[:, joint], linestyle=":", label="Dataset/RLinf H")
        axis.set_title(f"Joint {joint}")
    axes.flat[0].legend(fontsize="small")
    figure.suptitle(
        f"Worst paired trace: episode {worst.episode_id}, frame {worst.frame_index}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "action_trace.png", dpi=160)
    plt.close(figure)

    metric_names = (
        "first_action_state_jump",
        "first_difference_mae",
        "second_difference_mae",
        "ground_truth_mae",
    )
    servers = ("openpi", "rlinf")
    means = [
        [
            np.mean(
                [float(row[name]) for row in jitter_rows if row["server"] == server]
            )
            for name in metric_names
        ]
        for server in servers
    ]
    x = np.arange(len(metric_names))
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.36
    axis.bar(x - width / 2, means[0], width, label="OpenPI")
    axis.bar(x + width / 2, means[1], width, label="RLinf")
    axis.set_xticks(x, [name.replace("_", "\n") for name in metric_names])
    axis.set(ylabel="Mean absolute magnitude", title="Action jitter proxies")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "jitter_metrics.png", dpi=160)
    plt.close(figure)


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    pair = summary["paired_comparison"]
    endpoint_jump = summary["action_1_to_10_jump"]
    contract = summary["contract"]
    randomness = summary["randomness"]
    latency = summary["latency"]
    lines = [
        "# Policy Server Black-Box Diagnostic",
        "",
        "> This run describes the currently deployed behavior. The servers returned "
        f"horizons {contract['openpi_horizons']} and {contract['rlinf_horizons']}; "
        "when contracts differ, these results are not strict numerical parity evidence.",
        "",
        "## Common-Prefix Comparison",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {pair['num_pairs']} |",
        f"| Common horizon | {pair['common_horizon']} |",
        f"| Common action dimensions | {pair['common_action_dim']} |",
        f"| MAE | {pair['mae']:.6g} |",
        f"| RMSE | {pair['rmse']:.6g} |",
        f"| P95 absolute difference | {pair['p95_abs']:.6g} |",
        f"| Maximum absolute difference | {pair['max_abs']:.6g} |",
        f"| Joint-std normalized MAE | {pair['normalized_mae']:.6g} |",
        "",
        f"Worst prompt: `{pair['worst_prompt']['prompt']}` "
        f"(MAE {pair['worst_prompt']['mae']:.6g}).",
        "",
        f"Worst joint: {pair['worst_joint']['joint']} "
        f"(MAE {pair['worst_joint']['mae']:.6g}).",
        "",
        f"Worst horizon step: {pair['worst_horizon_step']['horizon_step']} "
        f"(MAE {pair['worst_horizon_step']['mae']:.6g}).",
        "",
        "## Per-Prompt MAE",
        "",
        "| Prompt | MAE |",
        "| --- | ---: |",
    ]
    for prompt, mae in sorted(pair["prompt_mae"].items(), key=lambda item: -item[1]):
        escaped_prompt = prompt.replace("|", "\\|")
        lines.append(f"| {escaped_prompt} | {mae:.6g} |")
    lines.extend(
        [
            "",
            "## Action 1 to Action 10 Jump",
            "",
            "For each response chunk, this compares one-based action steps 1 and 10 "
            "(`actions[0]` and `actions[9]`). Both servers provide these steps, so "
            "the metric is unaffected by their different full horizons.",
            "",
            "| Server | Chunks | MAE | RMSE | P95 | Maximum | Mean chunk L2 | Worst joint |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for server in ("openpi", "rlinf"):
        values = endpoint_jump[server]
        lines.append(
            f"| {server} | {values['num_chunks']} | {values['mae']:.6g} | "
            f"{values['rmse']:.6g} | {values['p95_abs']:.6g} | "
            f"{values['max_abs']:.6g} | {values['mean_l2']:.6g} | "
            f"{values['worst_joint']} ({values['worst_joint_mae']:.6g}) |"
        )
    displacement = endpoint_jump["paired_displacement_difference"]
    relative_change = (endpoint_jump["rlinf_to_openpi_mae_ratio"] - 1.0) * 100.0
    lines.extend(
        [
            "",
            f"RLinf's mean absolute jump is {abs(relative_change):.2f}% "
            f"{'higher' if relative_change >= 0 else 'lower'} than OpenPI's. "
            "The paired action-displacement vectors differ by "
            f"MAE {displacement['mae']:.6g}, RMSE {displacement['rmse']:.6g}, "
            f"P95 {displacement['p95_abs']:.6g}, and maximum "
            f"{displacement['max_abs']:.6g}; similar aggregate jump magnitudes do "
            "not imply matching per-joint directions.",
        ]
    )
    lines.extend(
        [
            "",
            "## Randomness and Latency",
            "",
            "| Server | Output std | Repeat MAE | Mean RTT (ms) | Mean inference (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for server in ("openpi", "rlinf"):
        inference = latency[server]["server_infer_ms_mean"]
        inference_text = f"{inference:.3f}" if inference is not None else "n/a"
        lines.append(
            f"| {server} | {randomness[server]['output_std']:.6g} | "
            f"{randomness[server]['repeat_to_repeat_mae']:.6g} | "
            f"{latency[server]['rtt_ms_mean']:.3f} | {inference_text} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `summary.json`: machine-readable aggregate metrics.",
            "- `sample_metrics.csv`: paired sample and timing metrics.",
            "- `per_joint_metrics.csv` and `per_horizon_metrics.csv`: detailed errors.",
            "- `jitter_metrics.csv`: chunk smoothness and replay boundary proxies.",
            "- `raw_outputs.npz`: every raw action chunk and its index.",
            "- `difference_heatmap.png`, `action_trace.png`, and `jitter_metrics.png`: plots.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def _load_joint_std(dataset_root: Path) -> np.ndarray:
    path = dataset_root / "meta" / "stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset statistics do not exist: {path}")
    stats = json.loads(path.read_text())
    if ACTION_COLUMN not in stats or "std" not in stats[ACTION_COLUMN]:
        raise ValueError(f"Dataset statistics are missing action.std: {path}")
    values = np.asarray(stats[ACTION_COLUMN]["std"], dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Dataset action.std must be a finite vector: {path}")
    return values


def run(args: argparse.Namespace) -> Path:
    """Run the comparison and return the timestamped artifact directory."""
    dataset_root = Path(args.dataset_root).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot v3 metadata does not exist: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot codebase_version 'v3.0', got {info.get('codebase_version')!r}"
        )
    prompts = load_task_prompts(dataset_root)
    refs = scan_episodes(dataset_root, prompts)
    selected = select_episodes(
        refs,
        episodes_per_prompt=args.episodes_per_prompt,
        prompt_regex=args.prompt_regex,
        episode_ids=set(args.episode_ids) if args.episode_ids else None,
        seed=args.seed,
    )
    joint_std = _load_joint_std(dataset_root)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    paired: list[PairedResult] = []
    jitter_rows: list[dict[str, Any]] = []
    replay_summary: dict[str, list[dict[str, float]]] = defaultdict(list)
    raw: dict[str, np.ndarray] = {}
    raw_index: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    server_layouts = {
        "openpi": args.openpi_image_layout,
        "rlinf": args.rlinf_image_layout,
    }
    with (
        PolicyServerClient(
            args.openpi_host, args.openpi_port, args.request_timeout, "OpenPI"
        ) as openpi,
        PolicyServerClient(
            args.rlinf_host, args.rlinf_port, args.request_timeout, "RLinf"
        ) as rlinf,
    ):
        for ref in selected:
            print(f"Comparing episode {ref.episode_id}: {ref.prompt}", flush=True)
            episode = load_episode_numeric(ref)
            paired_frames = uniform_frame_indices(ref.num_frames, args.paired_frames)
            observations = load_observations(dataset_root, episode, paired_frames)
            for frame_index in paired_frames:
                per_server_repeats: dict[str, list[np.ndarray]] = defaultdict(list)
                for repeat in range(args.repeats):
                    openpi_result = openpi.infer(
                        format_observation_layout(
                            observations[int(frame_index)], server_layouts["openpi"]
                        )
                    )
                    rlinf_result = rlinf.infer(
                        format_observation_layout(
                            observations[int(frame_index)], server_layouts["rlinf"]
                        )
                    )
                    truth_horizon = max(
                        openpi_result.actions.shape[0], rlinf_result.actions.shape[0]
                    )
                    truth = padded_action_chunk(
                        episode.action, int(frame_index), truth_horizon
                    )
                    result = PairedResult(
                        prompt=ref.prompt,
                        episode_id=ref.episode_id,
                        frame_index=int(frame_index),
                        repeat=repeat,
                        state=episode.state[frame_index],
                        truth=truth,
                        openpi=openpi_result,
                        rlinf=rlinf_result,
                    )
                    paired.append(result)
                    for server, response in (
                        ("openpi", openpi_result),
                        ("rlinf", rlinf_result),
                    ):
                        per_server_repeats[server].append(response.actions)
                        key = f"paired_{len(raw_index):06d}_{server}"
                        raw[key] = response.actions
                        raw_index.append(
                            {
                                "key": key,
                                "kind": "paired",
                                "server": server,
                                "episode_id": ref.episode_id,
                                "frame_index": int(frame_index),
                                "repeat": repeat,
                                "prompt": ref.prompt,
                            }
                        )
                        truth_for_server = padded_action_chunk(
                            episode.action, int(frame_index), response.actions.shape[0]
                        )
                        jitter_rows.append(
                            {
                                "kind": "paired",
                                "server": server,
                                "prompt": ref.prompt,
                                "episode_id": ref.episode_id,
                                "frame_index": int(frame_index),
                                "chunk_index": repeat,
                                **chunk_jitter_metrics(
                                    response.actions,
                                    episode.state[frame_index],
                                    truth_for_server,
                                ),
                                "boundary_jump_mae": "",
                                "boundary_jump_max": "",
                            }
                        )

            horizons = {
                "openpi": paired[-1].openpi.actions.shape[0],
                "rlinf": paired[-1].rlinf.actions.shape[0],
            }
            replay_frames = {
                server: replay_frame_indices(
                    ref.num_frames, horizon, args.replay_chunks, rng
                )
                for server, horizon in horizons.items()
            }
            replay_observations = load_observations(
                dataset_root,
                episode,
                np.unique(np.concatenate(list(replay_frames.values()))),
            )
            for server, client in (("openpi", openpi), ("rlinf", rlinf)):
                chunks: list[np.ndarray] = []
                rows_for_server: list[dict[str, Any]] = []
                for chunk_index, frame_index in enumerate(replay_frames[server]):
                    response = client.infer(
                        format_observation_layout(
                            replay_observations[int(frame_index)],
                            server_layouts[server],
                        )
                    )
                    chunks.append(response.actions)
                    truth = padded_action_chunk(
                        episode.action, int(frame_index), response.actions.shape[0]
                    )
                    row = {
                        "kind": "replay",
                        "server": server,
                        "prompt": ref.prompt,
                        "episode_id": ref.episode_id,
                        "frame_index": int(frame_index),
                        "chunk_index": chunk_index,
                        **chunk_jitter_metrics(
                            response.actions, episode.state[frame_index], truth
                        ),
                        "boundary_jump_mae": "",
                        "boundary_jump_max": "",
                    }
                    rows_for_server.append(row)
                    key = f"replay_{len(raw_index):06d}_{server}"
                    raw[key] = response.actions
                    raw_index.append(
                        {
                            "key": key,
                            "kind": "replay",
                            "server": server,
                            "episode_id": ref.episode_id,
                            "frame_index": int(frame_index),
                            "chunk_index": chunk_index,
                            "prompt": ref.prompt,
                        }
                    )
                boundary = replanning_boundary_metrics(chunks)
                for row in rows_for_server[1:]:
                    row.update(boundary)
                jitter_rows.extend(rows_for_server)
                replay_summary[server].append(boundary)

        pair_summary, sample_rows, joint_rows, horizon_rows = _aggregate_pair_metrics(
            paired, joint_std
        )
        contract = {
            "openpi_horizons": sorted(
                {item.openpi.actions.shape[0] for item in paired}
            ),
            "rlinf_horizons": sorted({item.rlinf.actions.shape[0] for item in paired}),
            "openpi_action_dims": sorted(
                {item.openpi.actions.shape[1] for item in paired}
            ),
            "rlinf_action_dims": sorted(
                {item.rlinf.actions.shape[1] for item in paired}
            ),
            "contracts_match": all(
                item.openpi.actions.shape == item.rlinf.actions.shape for item in paired
            ),
        }
        summary = {
            "diagnostic_scope": (
                "currently deployed behavior; not strict numerical parity when "
                "server contracts differ"
            ),
            "contract": contract,
            "paired_comparison": pair_summary,
            "action_1_to_10_jump": _action_1_to_10_summary(paired),
            "randomness": _randomness_summary(paired),
            "latency": _latency_summary(paired),
            "replanning_boundaries": {
                server: {
                    key: float(np.mean([item[key] for item in items]))
                    for key in ("boundary_jump_mae", "boundary_jump_max")
                }
                for server, items in replay_summary.items()
            },
        }
        run_metadata = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "argv": sys.argv,
            "arguments": vars(args),
            "dataset": {
                "root": dataset_root,
                "codebase_version": info.get("codebase_version"),
                "total_tasks": info.get("total_tasks"),
                "total_episodes": info.get("total_episodes"),
                "selected_episodes": selected,
            },
            "request_contract": {
                "images": {
                    "keys": list(REQUEST_IMAGE_KEYS),
                    "dtype": "uint8",
                    "openpi_layout": args.openpi_image_layout,
                    "rlinf_layout": args.rlinf_image_layout,
                },
                "state": "float32 [14]",
                "prompt": "dataset task string",
                "batched": False,
            },
            "servers": {
                "openpi": {
                    "uri": openpi.uri,
                    "metadata": openpi.metadata,
                },
                "rlinf": {
                    "uri": rlinf.uri,
                    "metadata": rlinf.metadata,
                },
            },
            "contract": contract,
        }

    raw["index_json"] = np.frombuffer(
        json.dumps(_jsonable(raw_index)).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(output_dir / "raw_outputs.npz", **raw)
    _write_json(output_dir / "run_metadata.json", run_metadata)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "sample_metrics.csv", sample_rows)
    _write_csv(output_dir / "per_joint_metrics.csv", joint_rows)
    _write_csv(output_dir / "per_horizon_metrics.csv", horizon_rows)
    _write_csv(output_dir / "jitter_metrics.csv", jitter_rows)
    _plot_outputs(output_dir, pair_summary, paired, jitter_rows)
    _write_report(output_dir, summary)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two OpenPI-compatible policy servers using real ALOHA "
            "observations from a LeRobot v3 dataset."
        )
    )
    parser.add_argument(
        "--dataset-root", default="data/lerobot-data_mixed_8_v30", type=Path
    )
    parser.add_argument("--openpi-host", default="127.0.0.1")
    parser.add_argument("--openpi-port", default=8000, type=int)
    parser.add_argument("--rlinf-host", default="127.0.0.1")
    parser.add_argument("--rlinf-port", default=8001, type=int)
    parser.add_argument("--openpi-image-layout", choices=("chw", "hwc"), default="chw")
    parser.add_argument("--rlinf-image-layout", choices=("chw", "hwc"), default="hwc")
    parser.add_argument("--episodes-per-prompt", default=1, type=int)
    parser.add_argument("--paired-frames", default=3, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--replay-chunks", default=3, type=int)
    parser.add_argument("--prompt-regex")
    parser.add_argument("--episode-ids", nargs="+", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--request-timeout", default=120.0, type=float)
    parser.add_argument(
        "--output-dir", default=Path("results/policy_server_compare"), type=Path
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.paired_frames, args.repeats, args.replay_chunks) < 1:
        raise SystemExit(
            "--paired-frames, --repeats, and --replay-chunks must be positive"
        )
    output_dir = run(args)
    print(f"Wrote policy server diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
