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

from rlinf.envs.behavior.env_access import (
    get_behavior_robot,
    get_task_reward,
    stage_idx_from_info,
    stage_idx_from_reward,
    to_int_or_none,
    unwrap_behavior_env,
)


class FakeTask:
    def __init__(self, robot=None):
        self.task_reward = FakeTaskReward()
        self.robot = robot

    def get_agent(self, env):
        return self.robot


class FakeTaskReward:
    def __init__(self):
        self.current_stage_idx = "2"


class FakeEnv:
    def __init__(self, robot=None, robots=None):
        self.task = FakeTask(robot=robot)
        self.robots = robots or []


class FakeWrapper:
    def __init__(self, env):
        self.env = env


def test_unwrap_behavior_env_returns_base_env():
    base = FakeEnv()
    wrapped_once = FakeWrapper(base)
    wrapped_twice = FakeWrapper(wrapped_once)

    assert unwrap_behavior_env(base) is base
    assert unwrap_behavior_env(wrapped_once) is base
    assert unwrap_behavior_env(wrapped_twice) is base


def test_unwrap_behavior_env_rejects_cycles():
    wrapper_a = FakeWrapper(None)
    wrapper_b = FakeWrapper(wrapper_a)
    wrapper_a.env = wrapper_b

    try:
        unwrap_behavior_env(wrapper_a)
    except RuntimeError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("Expected wrapper cycle to raise RuntimeError.")


def test_get_behavior_robot_prefers_task_agent_then_first_robot():
    task_robot = object()
    fallback_robot = object()

    assert get_behavior_robot(FakeEnv(robot=task_robot)) is task_robot
    assert get_behavior_robot(FakeEnv(robot=None, robots=[fallback_robot])) is fallback_robot
    assert get_behavior_robot(FakeWrapper(FakeEnv(robot=task_robot))) is task_robot


def test_task_reward_and_stage_helpers():
    env = FakeEnv()
    wrapped_env = FakeWrapper(env)

    assert get_task_reward(env) is env.task.task_reward
    assert get_task_reward(wrapped_env) is env.task.task_reward
    assert stage_idx_from_reward(env) == 2
    assert stage_idx_from_reward(wrapped_env) == 2
    assert stage_idx_from_info({"task_reward": {"current_stage_idx": "3"}}) == 3
    assert stage_idx_from_info({"current_stage_idx": "4"}) is None
    assert stage_idx_from_info({"current_stage_idx": "bad"}) is None
    assert to_int_or_none(None) is None
    assert to_int_or_none("5") == 5
