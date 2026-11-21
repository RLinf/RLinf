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


"""OGBench contracts and offline QAM resume/evaluation through real workers."""

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rlinf.envs import get_env_cls


def test_ogbench_evaluation_contract():
    cfg = OmegaConf.create(
        {
            "task_name": "antmaze-large-navigate-singletask-task1-v0",
            "is_eval": True,
            "auto_reset": False,
            "seed": 10001,
            "evaluation_seed_offset": 1000000,
            "max_steps_per_rollout_epoch": 2,
            "video_cfg": {"save_video": False},
        }
    )
    env_cls = get_env_cls("ogbench", cfg)
    with pytest.raises(ValueError, match="num_envs >= 1"):
        env_cls(cfg, num_envs=0)
    early_stop = OmegaConf.merge(cfg, {"early_stop_on_done": True})
    with pytest.raises(ValueError, match="early_stop_on_done=false"):
        env_cls(early_stop, num_envs=2)
    invalid = OmegaConf.merge(cfg, {"auto_reset": True})
    with pytest.raises(ValueError, match="auto_reset=false"):
        env_cls(invalid)
    env = env_cls(cfg, seed_offset=1, total_num_processes=2, worker_info=None)
    try:
        assert env.is_start
        global_rng = np.random.get_state()
        first, _ = env.reset()
        _assert_equal(np.random.get_state(), global_rng)
        assert first["states"].shape == (1, 29) and not env.is_start
        repeated, _ = env.reset(seed=1010002)
        torch.testing.assert_close(first["states"], repeated["states"], rtol=0, atol=0)
        with pytest.raises(ValueError, match="action shape"):
            env.step(np.zeros((2, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="chunk_length"):
            env.chunk_step(np.zeros((1, 8), dtype=np.float32))
        observations, rewards, terminated, truncated, infos = env.chunk_step(
            torch.zeros(1, 3, 8)
        )
        assert rewards.shape == terminated.shape == truncated.shape == (1, 3)
        assert rewards.dtype == torch.float32 and terminated.dtype == torch.bool
        assert truncated.tolist() == [[False, True, True]]
        assert rewards[0, 2] == 0
        torch.testing.assert_close(
            observations[1]["states"], observations[2]["states"], rtol=0, atol=0
        )
        assert infos[-1]["episode"]["episode_len"].item() == 2
        assert infos[-1]["episode"]["return"].item() == rewards.sum().item()
        assert "success" in infos[-1]["episode"]
        _assert_equal(np.random.get_state(), global_rng)
        env.update_reset_state_ids()
        env.reset()
        assert env.elapsed_steps.tolist() == [0]
    finally:
        env.close()
        env.close()


def _scalars(path):
    events = EventAccumulator(
        str(path / "tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    return {
        tag: [(event.step, event.value) for event in events.Scalars(tag)]
        for tag in events.Tags()["scalars"]
    }


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _run_offline_resume_and_evaluation(tmp_path, wrap):
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, REPO_PATH=str(repo), RAY_ADDRESS="local")

    def run(name, *overrides):
        run_dir = tmp_path / name
        run_dir.mkdir()
        cmd = [
            "bash",
            str(repo / "tests/e2e_tests/embodied/run_offline.sh"),
            "ogbench_qam",
            "egl",
            f"runner.logger.log_path={run_dir}",
            "runner.logger.experiment_name=qam",
            f"actor.use_fsdp_wrap={str(wrap).lower()}",
            *overrides,
        ]
        if os.environ.get("OGBENCH_DATASET_DIR"):
            cmd.append(f"data.dataset_dir={os.environ['OGBENCH_DATASET_DIR']}")
        with (run_dir / "run.log").open("w") as log:
            result = subprocess.run(
                cmd,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        assert result.returncode == 0, (run_dir / "run.log").read_text()[-12000:]
        return run_dir

    full_dir = run("full")
    step4 = full_dir / "qam/checkpoints/global_step_4"
    step8 = full_dir / "qam/checkpoints/global_step_8"

    def read_checkpoint(step):
        folder = step / "actor"
        if wrap:
            state = torch.load(
                folder / "state_rank_0.pt", map_location="cpu", weights_only=False
            )
            shard = torch.load(
                folder / "qam_model/local_shard_checkpoint/checkpoint_rank_0.pt",
                map_location="cpu",
                weights_only=False,
            )
            state.update(model=shard["model"], optimizer=shard["optimizers"][0])
            return state
        return torch.load(
            folder / "seed_10001.pt", map_location="cpu", weights_only=False
        )

    first = read_checkpoint(step4)
    full = read_checkpoint(step8)
    assert first["step"] == 4 and full["step"] == 8
    assert any(
        not torch.equal(v, first["model"][k])
        for k, v in full["model"].items()
        if k.startswith("actor_fast.")
    )
    full_metrics = _scalars(full_dir)
    assert full_metrics["eval/num_trajectories"] == [(4, 1.0), (8, 1.0)]
    assert full_metrics["eval/episode_len"][0][1] > 0
    for tag in ("train/critic_loss", "train/flow_loss", "train/adj_loss"):
        assert [step for step, _ in full_metrics[tag]] == [2, 4, 6, 8]
        assert np.isfinite([value for _, value in full_metrics[tag]]).all()

    resumed_dir = run("resumed", f"runner.resume_dir={step4}")
    resumed = read_checkpoint(resumed_dir / "qam/checkpoints/global_step_8")
    for key in ("model", "optimizer", "generator", "replay", "step", "seed"):
        _assert_equal(resumed[key], full[key])
    resumed_metrics = _scalars(resumed_dir)
    for tag, values in full_metrics.items():
        if tag.startswith("train/"):
            assert resumed_metrics[tag] == [
                (step, value) for step, value in values if step > 4
            ]

    def hashes():
        return {
            f.relative_to(step4): hashlib.sha256(f.read_bytes()).digest()
            for f in step4.rglob("*.pt")
        }

    before = hashes()
    parallel_dir = run(
        "parallel_eval",
        f"runner.resume_dir={step4}",
        "runner.only_eval=true",
        "env.eval.total_num_envs=2",
        "env.eval.early_stop_on_done=false",
    )
    parallel_metrics = _scalars(parallel_dir)
    assert parallel_metrics["eval/num_trajectories"] == [(4, 2.0)]
    assert parallel_metrics["eval/episode_len"][0][1] > 0
    assert np.isfinite(
        [parallel_metrics[tag][0][1] for tag in ("eval/success", "eval/return")]
    ).all()
    assert not any(tag.startswith("train/") for tag in parallel_metrics)
    assert not (parallel_dir / "qam/checkpoints").exists()
    assert hashes() == before
    return full["model"], full_metrics


def test_offline_resume_and_evaluation(tmp_path):
    native, fsdp = tmp_path / "native", tmp_path / "fsdp"
    native.mkdir()
    fsdp.mkdir()
    expected, native_metrics = _run_offline_resume_and_evaluation(native, False)
    actual, fsdp_metrics = _run_offline_resume_and_evaluation(fsdp, True)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
    for tag in (
        "train/grad_norm",
        "train/critic_loss",
        "train/flow_loss",
        "train/adj_loss",
    ):
        np.testing.assert_allclose(
            fsdp_metrics[tag], native_metrics[tag], atol=2e-5, rtol=2e-5
        )
