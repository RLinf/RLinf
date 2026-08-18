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

from __future__ import annotations

import dataclasses

import numpy as np
import openpi.shared.normalize as normalize
import openpi.transforms as transforms
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from toolkits.lerobot.calculate_norm_stats_fast import (
    ProjectedData,
    ProjectedLeRobotSamples,
    _load_task_mapping,
    _sort_and_validate_columns,
    action_query_indices,
    compute_norm_stats,
    episode_end_indices,
    load_projected_data,
    projection_columns,
    write_norm_stats_atomic,
)


@dataclasses.dataclass(frozen=True)
class CombineActions:
    pixel_dependent: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if self.pixel_dependent:
            state = state + np.asarray(data["image"]).mean()
        return {
            "state": state,
            "actions": np.concatenate(
                [np.asarray(data["left"]), np.asarray(data["right"])], axis=-1
            ),
        }


def _projected_data() -> ProjectedData:
    return ProjectedData(
        columns={
            "index": np.arange(5),
            "episode_index": np.asarray([0, 0, 0, 1, 1]),
            "frame_index": np.asarray([0, 1, 2, 0, 1]),
            "state_raw": np.arange(10, dtype=np.float32).reshape(5, 2),
            "action_left": np.arange(5, dtype=np.float32)[:, None],
            "action_right": (10 + np.arange(5, dtype=np.float32))[:, None],
        },
        visual_features={
            "camera": {
                "dtype": "image",
                "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            }
        },
        info={"total_frames": 5},
    )


def _data_config(*, pixel_dependent: bool = False) -> DataConfig:
    return DataConfig(
        repo_id="test/repo",
        action_sequence_keys=("action_left", "action_right"),
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "state": "state_raw",
                        "left": "action_left",
                        "right": "action_right",
                        "image": "camera",
                    }
                )
            ]
        ),
        data_transforms=transforms.Group(
            inputs=[CombineActions(pixel_dependent=pixel_dependent)]
        ),
    )


def _samples(*, pixel_dependent: bool = False) -> ProjectedLeRobotSamples:
    return ProjectedLeRobotSamples(
        _projected_data(),
        _data_config(pixel_dependent=pixel_dependent),
        action_horizon=3,
        task_mapping=None,
        context="test-context",
    )


def test_action_query_indices_repeat_episode_tail() -> None:
    episode_ids = np.asarray([0, 0, 0, 1, 1])
    ends = episode_end_indices(episode_ids)

    queries = action_query_indices(np.asarray([1, 2, 3, 4]), ends, 3)

    np.testing.assert_array_equal(
        queries,
        np.asarray(
            [
                [1, 2, 2],
                [2, 2, 2],
                [3, 4, 4],
                [4, 4, 4],
            ]
        ),
    )


def test_projected_samples_support_multiple_action_keys() -> None:
    sample = _samples().transform(1)

    np.testing.assert_array_equal(sample["state"], np.asarray([2.0, 3.0]))
    np.testing.assert_array_equal(
        sample["actions"],
        np.asarray([[1.0, 11.0], [2.0, 12.0], [2.0, 12.0]]),
    )


def test_compute_norm_stats_includes_partial_batch() -> None:
    stats = compute_norm_stats(_samples(), batch_size=4, num_workers=0)

    expected_states = np.arange(10, dtype=np.float32).reshape(5, 2)
    np.testing.assert_allclose(stats["state"].mean, expected_states.mean(axis=0))
    expected_left = np.asarray(
        [
            [0, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
            [3, 4, 4],
            [4, 4, 4],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(stats["actions"].mean[0], expected_left.mean())


def test_compute_norm_stats_matches_reference_float32_dtype() -> None:
    samples = _samples()
    for key in ("state_raw", "action_left", "action_right"):
        samples._columns[key] = samples._columns[key].astype(np.float64)

    stats = compute_norm_stats(samples, batch_size=4, num_workers=0)

    assert stats["state"].mean.dtype == np.dtype(np.float32)
    assert stats["actions"].mean.dtype == np.dtype(np.float32)


def test_pixel_dependent_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="depends on visual pixel values"):
        _samples(pixel_dependent=True).validate_pixel_independence()


def test_projection_columns_exclude_embedded_images() -> None:
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            ("state", pa.list_(pa.float32())),
            ("actions", pa.list_(pa.float32())),
            ("camera", image_type),
            ("index", pa.int64()),
            ("episode_index", pa.int64()),
        ]
    )
    info = {
        "features": {
            "state": {"dtype": "float32", "shape": [2]},
            "actions": {"dtype": "float32", "shape": [1]},
            "camera": {"dtype": "image", "shape": [3, 8, 8]},
        }
    }

    projected, visual = projection_columns(
        info, schema, ("actions",), context="test-context"
    )

    assert "camera" not in projected
    assert visual == {"camera": info["features"]["camera"]}


def test_projection_columns_report_missing_action_key() -> None:
    schema = pa.schema(
        [
            ("state", pa.list_(pa.float32())),
            ("index", pa.int64()),
            ("episode_index", pa.int64()),
        ]
    )

    with pytest.raises(ValueError, match="missing_action.*test-context"):
        projection_columns(
            {"features": {}},
            schema,
            ("missing_action",),
            context="test-context",
        )


def test_load_v2_jsonl_task_mapping(tmp_path) -> None:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "tasks.jsonl").write_text(
        "\n".join(
            [
                '{"task_index": 0, "task": "pick"}',
                '{"task_index": 1, "task": "place"}',
            ]
        )
        + "\n"
    )

    assert _load_task_mapping(tmp_path, context="test-context") == {
        0: "pick",
        1: "place",
    }


def test_load_projected_data_sorts_files_and_skips_images(tmp_path) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

    def write_file(path, indices, episode_index) -> None:
        count = len(indices)
        table = pa.table(
            {
                "state": pa.array(
                    [[float(index), float(index + 1)] for index in indices],
                    type=pa.list_(pa.float32()),
                ),
                "actions": pa.array(
                    [[float(index)] for index in indices],
                    type=pa.list_(pa.float32()),
                ),
                "camera": pa.array(
                    [{"bytes": b"not-an-image", "path": ""}] * count,
                    type=image_type,
                ),
                "index": pa.array(indices, type=pa.int64()),
                "episode_index": pa.array([episode_index] * count, type=pa.int64()),
                "frame_index": pa.array(range(count), type=pa.int64()),
            }
        )
        pq.write_table(table, path)

    write_file(data_dir / "file-000.parquet", [2, 3], 1)
    write_file(data_dir / "file-001.parquet", [0, 1], 0)
    info = {
        "total_frames": 4,
        "features": {
            "state": {"dtype": "float32", "shape": [2]},
            "actions": {"dtype": "float32", "shape": [1]},
            "camera": {"dtype": "image", "shape": [3, 8, 8]},
        },
    }

    projected = load_projected_data(
        tmp_path, info, ("actions",), context="test-context"
    )

    np.testing.assert_array_equal(projected.columns["index"], np.arange(4))
    assert "camera" not in projected.columns
    assert "camera" in projected.visual_features


def test_noncontiguous_global_index_is_rejected() -> None:
    columns = {
        "index": np.asarray([0, 2]),
        "episode_index": np.asarray([0, 0]),
        "actions": np.asarray([[0.0], [1.0]]),
    }

    with pytest.raises(ValueError, match="unique and contiguous"):
        _sort_and_validate_columns(columns, context="test-context")


def test_aloha_fast_sample_matches_reference_transform_pipeline() -> None:
    config = get_openpi_config("pi05_aloha_robotwin", repo_id="test/repo")
    data_config = config.data.create(config.assets_dirs, config.model)
    num_frames = 4
    states = np.linspace(-0.5, 0.5, num_frames * 14, dtype=np.float32).reshape(
        num_frames, 14
    )
    actions = states + 0.25
    visual_features = {
        name: {
            "dtype": "image",
            "shape": [3, 2, 2],
            "names": ["channels", "height", "width"],
        }
        for name in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    }
    projected = ProjectedData(
        columns={
            "index": np.arange(num_frames),
            "episode_index": np.zeros(num_frames, dtype=np.int64),
            "frame_index": np.arange(num_frames),
            "task_index": np.zeros(num_frames, dtype=np.int64),
            "observation.state": states,
            "action": actions,
        },
        visual_features=visual_features,
        info={"total_frames": num_frames},
    )
    fast_dataset = ProjectedLeRobotSamples(
        projected,
        data_config,
        config.model.action_horizon,
        {0: "test task"},
        context="test-context",
    )

    fast = fast_dataset.transform(1)
    query = np.minimum(1 + np.arange(config.model.action_horizon), num_frames - 1)
    reference = {
        "observation.state": states[1].copy(),
        "action": actions[query].copy(),
        "task_index": np.asarray(0),
        "prompt": "test task",
        **{name: np.zeros((3, 2, 2), dtype=np.uint8) for name in visual_features},
    }
    for transform in (
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ):
        reference = transform(reference)

    np.testing.assert_array_equal(fast["state"], reference["state"])
    np.testing.assert_array_equal(fast["actions"], reference["actions"])


def test_atomic_output_does_not_overwrite_by_default(tmp_path) -> None:
    official_path = tmp_path / "norm_stats.json"
    official_path.write_text("official")
    fast_path = tmp_path / "norm_stats_fast.json"
    first = {
        "state": normalize.NormStats(
            mean=np.asarray([1.0]),
            std=np.asarray([2.0]),
            q01=np.asarray([0.0]),
            q99=np.asarray([3.0]),
        )
    }
    write_norm_stats_atomic(fast_path, first, overwrite=False)

    assert official_path.read_text() == "official"
    assert normalize.deserialize_json(fast_path.read_text())["state"].mean == [1.0]
    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        write_norm_stats_atomic(fast_path, first, overwrite=False)

    second = {
        "state": normalize.NormStats(
            mean=np.asarray([5.0]),
            std=np.asarray([2.0]),
            q01=np.asarray([0.0]),
            q99=np.asarray([6.0]),
        )
    }
    write_norm_stats_atomic(fast_path, second, overwrite=True)
    assert normalize.deserialize_json(fast_path.read_text())["state"].mean == [5.0]
    assert not list(tmp_path.glob("*.tmp"))
