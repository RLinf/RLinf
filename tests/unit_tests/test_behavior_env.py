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
from types import SimpleNamespace
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
    [(4, 2, 2), (4, 2, 1), (6, 3, 2), (4, 1, 2), (6, 2, 3)],
    ids=[
        "two-stages-two-processes",
        "single-stage-round-robin",
        "two-stages-three-processes",
        "two-stages-one-process",
        "three-stages-two-processes",
    ],
)
def test_stage_dispatch_and_result_order(actors, num_envs, num_processes, num_stages):
    cfg = OmegaConf.create({})
    pool = behavior_env.BehaviorProcessPool(cfg, num_envs, num_processes, num_stages)
    assert len(actors) == num_processes
    count = num_envs // num_stages
    for stage in range(num_stages):
        start = stage * count
        observations, _ = pool.env_reset_slice(start, count)
        assert observations == [
            (i % num_processes, i // num_processes) for i in range(start, start + count)
        ]
        active = {actor for actor, row in observations}

        actions = torch.arange(start + 1, start + count + 1).reshape(count, 1, 1)
        _, rewards, _, _, _ = pool.env_chunk_step_slice(start, count, actions)
        assert rewards[0].tolist() == actions[:, 0, 0].tolist()
        for index, process in enumerate(actors):
            assert process.reset.remote.call_count == int(index in active)
            assert process.chunk_step.remote.call_count == int(index in active)
            process.reset_mock()


@pytest.mark.parametrize("automatic_reset", [False, True])
def test_shared_stages_step_only_their_own_scenes(monkeypatch, automatic_reset):
    og = pytest.importorskip("omnigibson")
    from omnigibson.envs import Environment
    from omnigibson.tasks.behavior_task import BehaviorTask
    from omnigibson.tasks.task_base import BaseTask

    # Use the SDK's real step, reward, timeout and automatic reset. Only the
    # physical scene, observations and goal predicate need a simulator.
    sim = Mock()
    monkeypatch.setattr(og, "sim", sim)
    env = Environment.__new__(Environment)
    env.num_envs = 4
    env._scenes = [Mock(robots=[Mock(action_dim=1)]) for _ in range(env.num_envs)]
    env._scene_graph_builders = []
    env._current_steps = torch.zeros(env.num_envs, dtype=torch.int32)
    env._current_episodes = torch.zeros_like(env._current_steps)
    env._automatic_reset = automatic_reset
    env._num_rerenders_on_reset = 0
    env._loaded = False

    def get_obs(env_indices=None):
        rows = range(env.num_envs) if env_indices is None else env_indices
        return list(rows), [{} for _ in rows]

    env.get_obs = get_obs
    task = env._task = BehaviorTask.__new__(BehaviorTask)
    BaseTask.__init__(
        task, termination_config={"max_steps": 2}, reward_config={"r_potential": 1.0}
    )
    task._loaded = True
    task._num_envs = env.num_envs
    task.use_presampled_robot_pose = False
    task.object_scopes = [{"goal": False} for _ in env.scenes]

    def check_goal(evaluate):
        achieved = evaluate("goal")
        return achieved, {
            "satisfied": [0] if achieved else [],
            "unsatisfied": [] if achieved else [0],
        }

    monkeypatch.setattr(
        BehaviorTask,
        "_evaluate_predicate",
        lambda self, row, key: self.object_scopes[row][key],
    )
    task.compiled_task = SimpleNamespace(check_goal=check_goal)
    env.reset(get_obs=False)
    env._current_steps[0] = 2
    env._current_steps[3] = 1
    # Scene 0 reaches its goal while waiting for stage 1; its reward must survive.
    task.object_scopes[0]["goal"] = True

    cls = behavior_env.BehaviorProcess.__ray_metadata__.modified_class
    worker = cls.__new__(cls)
    worker.env = env
    worker.step_supports_env_indices = False
    worker.skip_intermediate_renders = False
    worker.use_legacy_proprio = False
    actions = torch.arange(1.0, 9.0).reshape(4, 2, 1)
    obs, _, terms, truncs, _ = worker.chunk_step(actions, [2, 3])
    assert obs == ([2, 3], [2, 3])
    assert not torch.stack(terms).any()
    assert torch.stack(truncs).tolist() == [[False, False], [False, True]]
    assert env.episode_steps.tolist() == [2, 0, 2, 1 if automatic_reset else 3]
    assert env._current_episodes.tolist() == [1, 1, 1, 1 + automatic_reset]
    env.scenes[0].robots[0].apply_action.assert_not_called()
    env.scenes[1].robots[0].apply_action.assert_not_called()
    assert [
        call.args[0].item()
        for call in env.scenes[3].robots[0].apply_action.call_args_list
    ] == [7, 8]
    assert task._termination_conditions["predicate"].goal_status[0] == {
        "satisfied": [],
        "unsatisfied": [],
    }
    assert sim.step.call_count == 2 + automatic_reset

    obs, rewards, terms, truncs, infos = worker.chunk_step(actions[:, :1], [0, 1])
    assert obs == ([0, 1],)
    assert terms[0].tolist() == truncs[0].tolist() == [True, False]
    assert infos[0][0]["reward"]["reward_breakdown"]["potential"] == 1.0
    assert env._current_episodes.tolist() == [
        1 + automatic_reset,
        1,
        1,
        1 + automatic_reset,
    ]
    assert env.scenes[2].reset.call_count == 1

    # Replay scene 0's own step through the unmodified SDK under identical conditions.
    task.object_scopes[0]["goal"] = False
    env.reset(get_obs=False)
    env._current_steps[0] = 2
    task.object_scopes[0]["goal"] = True
    _, expected_rewards, _, _, expected_infos = env.step(actions[:, 0])
    assert torch.equal(rewards[0], expected_rewards[:2])
    assert infos[0][0] == expected_infos[0]


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
