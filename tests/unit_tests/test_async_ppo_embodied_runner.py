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

import asyncio
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import rlinf.runners.async_ppo_embodied_runner as async_ppo_runner_module
import rlinf.runners.embodied_runner as embodied_runner_module
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner


class _FakeHandle:
    def __init__(self, result=None, durations=None, per_rank_durations=None):
        self._result = result
        self._durations = durations or {}
        self._per_rank_durations = per_rank_durations or []
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        return self._result

    def consume_durations(self, return_per_rank=False):
        if return_per_rank:
            return self._durations, self._per_rank_durations
        return self._durations

    def done(self):
        return True


class _FakeChannel:
    def __init__(self, name: str):
        self.name = name

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def qsize(self):
        return 0


class _QueueChannel:
    def __init__(self, items):
        self.items = list(items)

    def get_nowait(self):
        if not self.items:
            raise asyncio.QueueEmpty
        return self.items.pop(0)


class _FakeMetricLogger:
    def __init__(self, cfg):
        del cfg
        self.logged = []
        self.finished = False

    def log(self, data, step, **kwargs):
        self.logged.append((data, step, kwargs))

    def finish(self):
        self.finished = True


class _FakeLogger:
    def warning(self, *args, **kwargs):
        del args, kwargs

    def info(self, *args, **kwargs):
        del args, kwargs


class _FakeActor:
    worker_group_name = "ActorGroup"

    def __init__(self):
        self.set_global_step_calls = []
        self.recv_rollout_trajectories_calls = []
        self.sync_model_to_rollout_calls = 0

    def set_global_step(self, step):
        self.set_global_step_calls.append(step)
        return _FakeHandle()

    def sync_model_to_rollout(self):
        self.sync_model_to_rollout_calls += 1
        return _FakeHandle()

    def recv_rollout_trajectories(self, input_channel):
        self.recv_rollout_trajectories_calls.append(input_channel)
        return _FakeHandle()

    def compute_advantages_and_returns(self):
        return _FakeHandle(result=[{"rewards": 0.25, "advantages_mean": 0.0}])

    def run_training(self):
        return _FakeHandle(
            result=[{"actor/policy_loss": 0.1}],
            durations={"run_training": 0.0},
            per_rank_durations=[],
        )


class _FakeRollout:
    worker_group_name = "RolloutGroup"

    def __init__(self):
        self.set_global_step_calls = []
        self.generate_calls = []
        self.sync_model_from_actor_calls = 0

    def set_global_step(self, step):
        self.set_global_step_calls.append(step)
        return _FakeHandle()

    def sync_model_from_actor(self):
        self.sync_model_from_actor_calls += 1
        return _FakeHandle()

    def generate(self, input_channel, output_channel, metric_channel):
        self.generate_calls.append(
            {
                "input_channel": input_channel,
                "output_channel": output_channel,
                "metric_channel": metric_channel,
            }
        )
        return _FakeHandle()

    def stop(self):
        return _FakeHandle()


class _FakeEnv:
    worker_group_name = "EnvGroup"

    def __init__(self):
        self.interact_calls = []

    def interact(
        self,
        input_channel,
        rollout_channel,
        reward_channel,
        actor_channel,
        metric_channel,
    ):
        self.interact_calls.append(
            {
                "input_channel": input_channel,
                "rollout_channel": rollout_channel,
                "reward_channel": reward_channel,
                "actor_channel": actor_channel,
                "metric_channel": metric_channel,
            }
        )
        return _FakeHandle()

    def stop(self):
        return _FakeHandle()


class _FakeReward:
    worker_group_name = "RewardGroup"

    def __init__(self):
        self.compute_calls = []
        self.stop_calls = 0
        self.handle = _FakeHandle()

    def compute_rewards_async(self, input_channel, output_channel, metric_channel=None):
        self.compute_calls.append(
            {
                "input_channel": input_channel,
                "output_channel": output_channel,
                "metric_channel": metric_channel,
            }
        )
        return self.handle

    def stop(self):
        self.stop_calls += 1
        return _FakeHandle()


def _build_cfg(use_output_step: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "runner": {
                "task_type": "embodied",
                "max_epochs": 1,
                "max_steps": 1,
                "val_check_interval": -1,
                "save_interval": 100,
                "weight_sync_interval": 1,
                "per_worker_log": False,
                "logger": {
                    "log_path": "/tmp/rlinf-test",
                    "project_name": "rlinf",
                    "experiment_name": "async-ppo-test",
                    "logger_backends": [],
                },
            },
            "rollout": {
                "recompute_logprobs": False,
            },
            "env": {
                "train": {
                    "max_episode_steps": 50,
                }
            },
            "reward": {
                "use_reward_model": True,
                "use_output_step": use_output_step,
            },
        }
    )


@pytest.fixture
def _patch_runner_dependencies(monkeypatch):
    created_channels = []

    def fake_channel_create(name: str):
        channel = _FakeChannel(name)
        created_channels.append(channel)
        return channel

    monkeypatch.setattr(
        embodied_runner_module.Channel, "create", staticmethod(fake_channel_create)
    )
    monkeypatch.setattr(embodied_runner_module, "MetricLogger", _FakeMetricLogger)
    monkeypatch.setattr(embodied_runner_module, "get_logger", lambda: _FakeLogger())
    monkeypatch.setattr(
        async_ppo_runner_module,
        "check_progress",
        lambda *args, **kwargs: (False, False, False),
    )
    return created_channels


def test_async_ppo_runner_starts_reward_worker_and_wires_reward_channel(
    _patch_runner_dependencies,
):
    cfg = _build_cfg(use_output_step=0)
    actor = _FakeActor()
    rollout = _FakeRollout()
    env = _FakeEnv()
    reward = _FakeReward()

    runner = AsyncPPOEmbodiedRunner(
        cfg=cfg,
        actor=actor,
        rollout=rollout,
        env=env,
        reward=reward,
    )
    runner.print_metrics_table_async = lambda *args, **kwargs: None

    runner.run()

    assert runner.reward is reward
    assert runner.reward_initialized is True
    assert len(reward.compute_calls) == 1
    assert len(env.interact_calls) == 1

    reward_channel = reward.compute_calls[0]["input_channel"]
    assert reward_channel.name == "Reward"
    assert reward_channel is runner.reward_channel
    assert reward_channel is env.interact_calls[0]["reward_channel"]
    assert reward.compute_calls[0]["output_channel"] is runner.env_channel
    assert reward.compute_calls[0]["metric_channel"] is runner.reward_metric_channel

    assert reward.stop_calls == 1
    assert reward.handle.wait_calls == 1


def test_async_ppo_runner_rejects_delayed_reward_activation(
    _patch_runner_dependencies,
):
    cfg = _build_cfg(use_output_step=1)

    with pytest.raises(ValueError, match="use_output_step=0"):
        AsyncPPOEmbodiedRunner(
            cfg=cfg,
            actor=_FakeActor(),
            rollout=_FakeRollout(),
            env=_FakeEnv(),
            reward=_FakeReward(),
        )


def test_async_ppo_runner_prefixes_channels(_patch_runner_dependencies):
    cfg = _build_cfg(use_output_step=0)
    cfg.runner.channel_prefix = "latency_run"

    runner = AsyncPPOEmbodiedRunner(
        cfg=cfg,
        actor=_FakeActor(),
        rollout=_FakeRollout(),
        env=_FakeEnv(),
        reward=_FakeReward(),
    )

    assert runner.env_channel.name == "latency_run_Env"
    assert runner.rollout_channel.name == "latency_run_Rollout"
    assert runner.actor_channel.name == "latency_run_Actor"
    assert runner.env_metric_channel.name == "latency_run_EnvMetric"
    assert runner.rollout_metric_channel.name == "latency_run_RolloutMetric"
    assert runner.reward_metric_channel.name == "latency_run_RewardMetric"


def test_async_ppo_runner_does_not_prefix_channels_with_run_id(
    _patch_runner_dependencies,
):
    cfg = _build_cfg(use_output_step=0)
    cfg.runner.run_id = "latency_run"

    runner = AsyncPPOEmbodiedRunner(
        cfg=cfg,
        actor=_FakeActor(),
        rollout=_FakeRollout(),
        env=_FakeEnv(),
        reward=_FakeReward(),
    )

    assert runner.env_channel.name == "Env"
    assert runner.rollout_channel.name == "Rollout"
    assert runner.actor_channel.name == "Actor"
    assert runner.env_metric_channel.name == "EnvMetric"
    assert runner.rollout_metric_channel.name == "RolloutMetric"
    assert runner.reward_metric_channel.name == "RewardMetric"


def test_async_ppo_runner_sums_time_per_rank_then_takes_slowest_rank(
    _patch_runner_dependencies,
):
    cfg = _build_cfg(use_output_step=0)
    runner = AsyncPPOEmbodiedRunner(
        cfg=cfg,
        actor=_FakeActor(),
        rollout=_FakeRollout(),
        env=_FakeEnv(),
        reward=_FakeReward(),
    )
    runner.reward_metric_channel = _QueueChannel(
        [
            {"rank": 0, "time": {"time/reward/recv_input_wait": 1.0}},
            {"rank": 0, "time": {"time/reward/recv_input_wait": 2.0}},
            {"rank": 1, "time": {"time/reward/recv_input_wait": 4.0}},
        ]
    )

    time_metrics, ranked_time_metrics, _ = runner.get_reward_metrics()

    assert time_metrics == {"time/reward/recv_input_wait": 4.0}
    assert ranked_time_metrics == [
        {"time/reward/recv_input_wait": 3.0},
        {"time/reward/recv_input_wait": 4.0},
    ]


def test_async_ppo_runner_keeps_reward_queue_depth_max_metric(
    _patch_runner_dependencies,
):
    cfg = _build_cfg(use_output_step=0)
    runner = AsyncPPOEmbodiedRunner(
        cfg=cfg,
        actor=_FakeActor(),
        rollout=_FakeRollout(),
        env=_FakeEnv(),
        reward=_FakeReward(),
    )
    runner.reward_metric_channel = _QueueChannel(
        [
            {
                "rank": 0,
                "reward": {
                    "input_queue_depth": 1.0,
                    "input_queue_depth_max": 1.0,
                },
            },
            {
                "rank": 0,
                "reward": {
                    "input_queue_depth": 3.0,
                    "input_queue_depth_max": 3.0,
                },
            },
            {
                "rank": 1,
                "reward": {
                    "input_queue_depth": 2.0,
                    "input_queue_depth_max": 2.0,
                },
            },
        ]
    )

    _, _, reward_metrics = runner.get_reward_metrics()

    assert reward_metrics["reward/input_queue_depth"] == 2.0
    assert reward_metrics["reward/input_queue_depth_max"] == 3.0
