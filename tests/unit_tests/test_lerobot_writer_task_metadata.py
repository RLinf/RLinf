# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rlinf.data.lerobot_writer import LeRobotDatasetWriter, merge_distributed_datasets


class _DummyLogger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


@contextmanager
def _workspace_tmpdir():
    root = Path(__file__).resolve().parents[2] / ".tmp_unit_tests"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="lerobot_", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_episode(root_dir, task: str, task_metadata: dict):
    with patch("rlinf.data.lerobot_writer.get_logger", return_value=_DummyLogger()):
        writer = LeRobotDatasetWriter(
            root_dir=str(root_dir),
            image_shape=(4, 4, 3),
            state_dim=3,
            action_dim=2,
            fps=5,
        )
        writer.add_episode(
            images=np.zeros((2, 4, 4, 3), dtype=np.uint8),
            wrist_images=None,
            states=np.ones((2, 3), dtype=np.float32),
            actions=np.ones((2, 2), dtype=np.float32),
            task=task,
            is_success=True,
            task_metadata=task_metadata,
            dones=np.array([False, True], dtype=bool),
        )
        writer.finalize()


def test_lerobot_writer_persists_task_metadata():
    task_metadata = {
        "benchmark_selection": "human/pretrain/atomic_seen",
        "split": "pretrain",
        "task_mode": "atomic",
    }
    with _workspace_tmpdir() as tmp_path:
        _write_episode(tmp_path, "open drawer from registry", task_metadata)

        tasks_path = tmp_path / "meta" / "tasks.jsonl"
        episodes_path = tmp_path / "meta" / "episodes.jsonl"
        info_path = tmp_path / "meta" / "info.json"

        task_record = json.loads(tasks_path.read_text().strip())
        episode_record = json.loads(episodes_path.read_text().strip())
        info = json.loads(info_path.read_text())

        assert task_record["benchmark_selection"] == "human/pretrain/atomic_seen"
        assert task_record["split"] == "pretrain"
        assert episode_record["task_metadata"]["task_mode"] == "atomic"
        assert info["task_metadata_keys"] == [
            "benchmark_selection",
            "split",
            "task_mode",
        ]


def test_merge_distributed_datasets_keeps_task_metadata():
    with _workspace_tmpdir() as tmp_path:
        worker_dir = tmp_path / "collected_data_stage0_rank0"
        output_dir = tmp_path / "merged"
        task_metadata = {
            "benchmark_selection": "human/target/atomic_seen",
            "split": "target",
            "task_mode": "atomic",
        }
        _write_episode(worker_dir, "close drawer from registry", task_metadata)

        with patch(
            "rlinf.data.lerobot_writer.get_logger", return_value=_DummyLogger()
        ):
            merged_episodes = merge_distributed_datasets(
                base_dir=str(tmp_path),
                output_dir=str(output_dir),
                pattern="collected_data_stage*_rank*",
                robot_type="panda",
                fps=5,
            )

        merged_episode = json.loads(
            (output_dir / "meta" / "episodes.jsonl").read_text().strip()
        )
        merged_task = json.loads(
            (output_dir / "meta" / "tasks.jsonl").read_text().strip()
        )
        merged_info = json.loads((output_dir / "meta" / "info.json").read_text())

        assert merged_episodes == 1
        assert (
            merged_episode["task_metadata"]["benchmark_selection"]
            == "human/target/atomic_seen"
        )
        assert merged_task["split"] == "target"
        assert merged_info["task_metadata_keys"] == [
            "benchmark_selection",
            "split",
            "task_mode",
        ]
