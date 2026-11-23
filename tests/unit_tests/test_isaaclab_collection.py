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

"""Isaac collection process boundaries and the raw-to-LeRobot data contract."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from toolkits.isaaclab.collect_graspgen_curobo import (
    CollectionWorkflow,
    verify_runtime,
)
from toolkits.isaaclab.export_graspgen_curobo import (
    RawEpisode,
    export_selection,
    file_sha256,
    verify_selection,
)


@pytest.fixture
def episode_record(tmp_path):
    source = tmp_path / "raw_episode"
    (source / "cameras").mkdir(parents=True)
    qpos = np.arange(36, dtype=np.float64).reshape(4, 9) / 1000
    targets = qpos + 0.01
    np.savez(
        source / "trajectory.npz",
        qpos=qpos,
        action_targets=targets[1:],
        position_targets=targets,
        sim_time=10 + np.arange(4) / 60,
    )
    for index, camera in enumerate(("base_camera", "wrist_camera")):
        pixels = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        pixels[:] = np.arange(4, dtype=np.uint8)[:, None, None, None] + index * 10
        np.save(source / "cameras" / f"{camera}.npy", pixels)
    return {
        "source": str(source),
        "length": 3,
        "task": "Pick up and hold the block.",
        "case_id": "block_0",
        "object_id": "block55",
        "source_sha256": {
            str(p.relative_to(source)): file_sha256(p)
            for p in source.rglob("*")
            if p.is_file()
        },
    }


def test_raw_episode_pairs_pre_action_state_with_executed_target(
    episode_record, tmp_path
):
    episode = RawEpisode(episode_record)
    frames = list(episode.frames())
    assert len(frames) == 3
    np.testing.assert_array_equal(
        frames[0]["state"][:7], episode.qpos[0, :7].astype(np.float32)
    )
    np.testing.assert_array_equal(
        frames[0]["actions"][:7], episode.actions[0, :7].astype(np.float32)
    )
    assert frames[0]["state"][-1] == np.float32(0.007 + 0.008)
    assert [bool(row["done"][0]) for row in frames] == [False, False, True]
    assert frames[-1]["image"][0, 0, 0] == 2
    evidence = tmp_path / "evidence"
    episode.save_evidence(evidence)
    assert np.load(evidence / "terminal_image.npy")[0, 0, 0] == 3
    assert (
        file_sha256(evidence / "trajectory.npz")
        == episode_record["source_sha256"]["trajectory.npz"]
    )


def test_raw_episode_rejects_modified_camera(episode_record):
    path = Path(episode_record["source"]) / "cameras/base_camera.npy"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source evidence changed"):
        RawEpisode(episode_record)


def test_raw_episode_rejects_unbound_images(episode_record):
    del episode_record["source_sha256"]["cameras/base_camera.npy"]
    with pytest.raises(ValueError, match="both camera"):
        RawEpisode(episode_record)


@pytest.mark.parametrize("field", ["action_targets", "sim_time"])
def test_raw_episode_rejects_wrong_executed_target_or_time(episode_record, field):
    path = Path(episode_record["source"]) / "trajectory.npz"
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values[field][-1] += 0.1
    np.savez(path, **values)
    episode_record["source_sha256"]["trajectory.npz"] = file_sha256(path)
    with pytest.raises(AssertionError):
        RawEpisode(episode_record)


def test_selection_rejects_unverified_or_changed_inputs(episode_record, tmp_path):
    evidence = tmp_path / "RUN.json"
    evidence.write_text('{"status": "completed"}')
    selection = {
        "status": "selected_before_export",
        "static_and_motion_verified": True,
        "episodes": [episode_record],
        "original_inputs_sha256": {str(evidence): file_sha256(evidence)},
    }
    verify_selection(selection)
    selection["static_and_motion_verified"] = False
    with pytest.raises(ValueError, match="strictly verified"):
        verify_selection(selection)
    selection["static_and_motion_verified"] = True
    evidence.write_text('{"status": "changed"}')
    with pytest.raises(ValueError, match="Selection input changed"):
        verify_selection(selection)


def test_runtime_revision_rejects_tracked_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    script = tmp_path / "cli.py"
    script.write_text("pass\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "cli.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    verify_runtime(tmp_path, revision)
    script.write_text("raise RuntimeError('changed')\n")
    with pytest.raises(ValueError, match="modified tracked files"):
        verify_runtime(tmp_path, revision)
    with pytest.raises(ValueError, match="Collector must be at"):
        verify_runtime(tmp_path, "0" * 40)


def test_collection_preserves_fixed_attempts_and_resume_config(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "cli.py").write_text(
        "import json, os, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.mkdir(exist_ok=True)\n"
        "(out / 'request.json').write_text(json.dumps({'args': sys.argv[1:], 'sim': os.environ['SIM_PYTHON']}))\n"
    )
    config_path = (
        Path(__file__).parents[2]
        / "examples/embodiment/config/isaaclab_graspgen_curobo_collect.yaml"
    )
    cfg = OmegaConf.load(config_path)
    cfg.runtime.root = str(runtime)
    cfg.runtime.sim_python = sys.executable
    cfg.output = str(tmp_path / "collection with spaces")
    cfg.candidates.root = str(tmp_path / "candidate pools")
    workflow = CollectionWorkflow(cfg)
    workflow.collect()
    request = json.loads((Path(cfg.output) / "raw/request.json").read_text())
    assert "--skip-validation" not in request["args"]
    assert request["sim"] == sys.executable
    workflow.collect(resume=True)
    assert (
        "--resume"
        in json.loads((Path(cfg.output) / "raw/request.json").read_text())["args"]
    )
    cfg.collection.seed += 1
    with pytest.raises(ValueError, match="original collection configuration"):
        CollectionWorkflow(cfg).collect(resume=True)
    (runtime / "cli.py").write_text("raise SystemExit(7)\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        workflow.run_runtime(["collect"])
    assert error.value.returncode == 7


def test_lerobot_export_reads_all_rows_and_keeps_terminal_frame(
    episode_record, tmp_path
):
    pytest.importorskip("lerobot")
    selection = tmp_path / "SELECTION.json"
    selection.write_text(
        json.dumps(
            {
                "status": "selected_before_export",
                "static_and_motion_verified": True,
                "episodes": [episode_record, episode_record],
                "original_inputs_sha256": {},
            }
        )
    )
    output = tmp_path / "export"
    export_selection(selection, output, "local/isaac_collection_test")
    audit = json.loads((output / "AUDIT.json").read_text())
    assert audit["status"] == "passed"
    assert (audit["episodes"], audit["frames"]) == (2, 6)
    assert (output / "sidecars/episode_000001/terminal_wrist_image.npy").is_file()
    assert not output.with_name("export.partial").exists()
    with pytest.raises(FileExistsError):
        export_selection(selection, output, "local/isaac_collection_test")
