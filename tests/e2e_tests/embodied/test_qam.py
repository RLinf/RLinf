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


def test_ogbench_parallel_episodes_finish_independently(monkeypatch):
    import gymnasium as gym
    import ogbench

    class StaggeredEnv(gym.Env):
        def __init__(self, finish_after, success):
            self.finish_after = finish_after
            self.success = success
            self.steps = 0
            self.closed = False
            self.observation_space = gym.spaces.Box(-1, 1, shape=(29,))
            self.action_space = gym.spaces.Box(-1, 1, shape=(8,))

        def reset(self, *, seed=None, options=None):
            self.steps = 0
            self.seed = seed
            return np.full(29, seed, dtype=np.float32), {}

        def step(self, action):
            assert not self.closed and self.steps < self.finish_after
            assert np.asarray(action).shape == (8,)
            self.steps += 1
            reward = float(self.finish_after if self.steps == 1 else 1)
            return (
                np.full(29, self.steps, dtype=np.float32),
                reward,
                self.steps == self.finish_after,
                False,
                {"success": self.success if self.steps == self.finish_after else 0},
            )

        def close(self):
            self.closed = True

    environments = [StaggeredEnv(1, 1), StaggeredEnv(3, 0)]
    monkeypatch.setattr(
        ogbench, "make_env_and_datasets", lambda *args, **kwargs: environments.pop(0)
    )
    cfg = OmegaConf.create(
        {
            "task_name": "antmaze-large-navigate-singletask-task1-v0",
            "is_eval": True,
            "auto_reset": False,
            "early_stop_on_done": False,
            "seed": 10001,
            "evaluation_seed_offset": 1000000,
            "max_steps_per_rollout_epoch": 4,
            "video_cfg": {"save_video": False},
        }
    )
    env = get_env_cls("ogbench", cfg)(cfg, num_envs=2)
    first, _ = env.reset()
    assert first["states"].shape == (2, 29)
    assert [item.seed for item in env.envs] == [1010001, 1010002]
    states, rewards, terminated, truncated, infos = env.chunk_step(torch.zeros(2, 4, 8))
    assert rewards.tolist() == [[1.0, 0.0, 0.0, 0.0], [3.0, 1.0, 1.0, 0.0]]
    assert terminated.tolist() == [
        [True, True, True, True],
        [False, False, True, True],
    ]
    assert not truncated.any()
    assert [item.steps for item in env.envs] == [1, 3]
    torch.testing.assert_close(states[0]["states"][0], states[3]["states"][0])
    episode = infos[-1]["episode"]
    assert episode["success"].tolist() == [1.0, 0.0]
    assert episode["return"].tolist() == [1.0, 5.0]
    assert episode["episode_len"].tolist() == [1.0, 3.0]
    env.close()
    env.close()
    assert all(item.closed for item in env.envs)


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


def test_offline_resume_and_evaluation(tmp_path):
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
    filename = "actor/seed_10001.pt"
    assert (step4 / filename).is_file()
    full = torch.load(step8 / filename, map_location="cpu", weights_only=False)
    assert full["step"] == full["updates"] == 8
    full_metrics = _scalars(full_dir)
    assert full_metrics["eval/num_trajectories"] == [(4, 1.0), (8, 1.0)]
    assert full_metrics["eval/episode_len"][0][1] > 0
    for tag in ("train/critic_loss", "train/flow_loss", "train/adj_loss"):
        assert [step for step, _ in full_metrics[tag]] == [2, 4, 6, 8]
        assert np.isfinite([value for _, value in full_metrics[tag]]).all()

    resumed_dir = run("resumed", f"runner.resume_dir={step4}")
    resumed = torch.load(
        resumed_dir / "qam/checkpoints/global_step_8" / filename,
        map_location="cpu",
        weights_only=False,
    )
    for key in ("model", "optimizer", "generator", "replay", "step", "updates", "seed"):
        _assert_equal(resumed[key], full[key])
    resumed_metrics = _scalars(resumed_dir)
    for tag, values in full_metrics.items():
        if tag.startswith("train/"):
            assert resumed_metrics[tag] == [
                (step, value) for step, value in values if step > 4
            ]

    before = hashlib.sha256((step4 / filename).read_bytes()).digest()
    evaluated_dir = run(
        "evaluated", f"runner.resume_dir={step4}", "runner.only_eval=true"
    )
    metrics = _scalars(evaluated_dir)
    assert metrics["eval/num_trajectories"] == [(4, 1.0)]
    assert metrics["eval/episode_len"][0][1] > 0
    assert "eval/success" in metrics and "eval/return" in metrics
    assert not any(tag.startswith("train/") for tag in metrics)
    assert not (evaluated_dir / "qam/checkpoints").exists()
    assert hashlib.sha256((step4 / filename).read_bytes()).digest() == before

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
    assert hashlib.sha256((step4 / filename).read_bytes()).digest() == before
