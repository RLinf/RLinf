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

import sys
from pathlib import Path
from unittest.mock import Mock

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.behavior import behavior_env


@pytest.fixture
def actors(monkeypatch):
    processes = []

    def launch(cfg, num_envs):
        process = Mock()
        process.get_activity_name.remote.return_value = "turning_on_radio"
        # A reset identifies each row by its actor and local index.
        index = len(processes)
        process.reset.remote.side_effect = lambda rows: (
            [(index, row) for row in rows],
            [{} for row in rows],
        )

        def step(actions, rows):
            assert actions.shape == (num_envs, 1, 1)
            rewards = actions[rows, 0, 0]
            dones = [False] * len(rows)
            return [rows], [rewards], [dones], [dones], [[{} for row in rows]]

        process.chunk_step.remote.side_effect = step
        processes.append(process)
        return process

    monkeypatch.setattr(behavior_env, "BehaviorProcess", Mock(remote=launch))
    monkeypatch.setattr(behavior_env.ray, "get", lambda results: results)
    return processes


@pytest.mark.parametrize(
    "num_envs,num_processes,num_stages",
    [(4, 2, 2), (4, 2, 1), (6, 3, 2)],
    ids=["isolated-stages", "single-stage-round-robin", "uneven-stage-process-count"],
)
def test_stage_dispatch_and_result_order(actors, num_envs, num_processes, num_stages):
    cfg = OmegaConf.create({})
    pool = behavior_env.BehaviorProcessPool(cfg, num_envs, num_processes, num_stages)
    owners = set()
    count = num_envs // num_stages
    for stage in range(num_stages):
        start = stage * count
        observations, _ = pool.env_reset_slice(start, count)
        assert len(set(observations)) == count
        if num_stages == 1:
            assert observations == [
                (i % num_processes, i // num_processes) for i in range(count)
            ]
        active = {actor for actor, row in observations}
        assert owners.isdisjoint(active)
        owners.update(active)

        actions = torch.arange(start + 1, start + count + 1).reshape(count, 1, 1)
        _, rewards, _, _, _ = pool.env_chunk_step_slice(start, count, actions)
        assert rewards[0].tolist() == actions[:, 0, 0].tolist()
        for index, process in enumerate(actors):
            assert process.reset.remote.call_count == int(index in active)
            assert process.chunk_step.remote.call_count == int(index in active)
            process.reset_mock()
    assert owners == set(range(num_processes))


def test_too_few_processes_fails_before_launch(actors):
    with pytest.raises(ValueError, match="one simulator process per pipeline stage"):
        behavior_env.BehaviorProcessPool(OmegaConf.create({}), 4, 1, 2)
    assert not actors


@pytest.mark.parametrize(
    "episode_override,expected",
    [(None, (151, 160)), (49, (49, 64))],
    ids=["official-task-timeout", "manual-episode-limit"],
)
def test_eval_config_resolves_step_limits(
    monkeypatch, tmp_path, episode_override, expected
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", str(tmp_path))
    args = [sys.argv[0], f"hydra.run.dir={tmp_path}", "hydra.job.chdir=false"]
    if episode_override is None:
        stats = tmp_path / "2026-challenge-task-instances/metadata/task.jsonl"
        stats.parent.mkdir(parents=True)
        stats.write_text('{"task_name": "turning_on_radio", "length": 100}\n')
    else:
        args.append(f"env.eval.max_episode_steps={episode_override}")
    monkeypatch.setattr(sys, "argv", args)

    @hydra.main(
        version_base="1.1",
        config_path=str(root / "evaluations/behavior"),
        config_name="behavior_openpi_pi05_eval",
    )
    def check(cfg):
        assert (
            cfg.env.eval.max_episode_steps,
            cfg.env.eval.max_steps_per_rollout_epoch,
        ) == expected

    check()
