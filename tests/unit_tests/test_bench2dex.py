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

"""CPU contracts for the Bench2Dex evaluation protocol."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluations.bench2dex.run_all_channels import channel_command
from rlinf.envs.sim.bench2dex.evaluation import configure_profile, normalize_profile
from rlinf.envs.sim.bench2dex.rewards import StageCompletionReward


@pytest.mark.parametrize(
    ("channel", "invariant", "covariant"),
    [
        ("none", False, False),
        ("cov", False, True),
        ("inv", True, False),
        ("inv_cov", True, True),
    ],
)
def test_evaluation_channels_keep_distinct_variation_axes(
    channel, invariant, covariant
):
    original = {"appearance": {"background": {"sentinel": "kept"}}}
    config = configure_profile(original, channel)
    for category, axis in (
        ("appearance", "background"),
        ("appearance", "table_surface"),
        ("appearance", "light"),
        ("clutter", "tabletop"),
        ("spatial", "camera"),
    ):
        assert config[category][axis]["enabled"] is invariant
    for axis in ("object_pose", "table_height"):
        assert config["spatial"][axis]["enabled"] is covariant
    if covariant:
        assert config["spatial"]["object_pose"]["position_jitter_cm"] == 2.0
        assert config["spatial"]["object_pose"]["yaw_jitter_deg"] == 10.0
        assert config["spatial"]["table_height"]["offset_m_range"] == [-0.05, 0.05]
    assert config["appearance"]["background"]["sentinel"] == "kept"
    assert original == {"appearance": {"background": {"sentinel": "kept"}}}


def test_evaluation_rejects_unknown_channel():
    with pytest.raises(ValueError):
        normalize_profile("unknown")


def test_parallel_evaluation_command_keeps_episode_and_seed_contract():
    command = channel_command(
        Path("/repo"),
        "example",
        "inv_cov",
        12,
        7,
        Path("/output"),
        Path("/anchors"),
        123,
        456,
        "unseen",
        [],
        num_envs=3,
    )
    assert "env.eval.rollout_epoch=4" in command
    assert "env.eval.total_num_envs=3" in command
    assert "env.eval.auto_reset=false" in command
    assert "env.eval.seed=123" in command
    assert "env.eval.init_params.inv_cov_seed=456" in command
    assert "env.eval.init_params.start_episode=7" in command
    assert "env.eval.init_params.anchor_dir=/anchors" in command
    assert "env.eval.init_params.generalization_profile=inv_cov" in command


def test_parallel_evaluation_rejects_incomplete_batch():
    with pytest.raises(ValueError):
        channel_command(
            Path("/repo"),
            "example",
            "none",
            5,
            1,
            Path("/out"),
            Path("/anchors"),
            1,
            2,
            "unseen",
            [],
            num_envs=2,
        )


@dataclass
class _MetricsResult:
    stable_success: bool = True


class _CPURuntime:
    """Replace only the Isaac boundary while using real spawned workers."""

    def __init__(self, config, seed, shard, shards):
        self.episode_index = config.get("start_episode", 1) - 1 + shard
        self.episode_stride = shards
        self.base_seed = seed

    def reset(self, seed=None, reset_state_id=None):
        index = self.episode_index if reset_state_id is None else reset_state_id
        if reset_state_id is None:
            self.episode_index += self.episode_stride
        self.info = {
            "seed": self.base_seed + index if seed is None else seed,
            "episode_index": index + 1,
            "control_steps": 0,
            "episode_return": 0.0,
            "success_once": False,
        }
        self.obs = {
            "images": {"main": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": np.zeros(1, dtype=np.float32),
            "full_joint_positions": np.zeros(1, dtype=np.float32),
            "active_joint_names": ["joint"],
            "full_joint_names": ["joint"],
            "instruction": "test",
            "robot_key": "test",
            "tactile": {"site_names": []},
        }
        return self.obs, self.info

    def chunk_step(self, actions, *, capture_intermediate_observations=True):
        return [(self.obs, 0.0, True, False, dict(self.info, action_steps=1))]

    def close(self):
        pass


class _UnresponsiveCloseRuntime(_CPURuntime):
    """One simulator process ignores termination while closing its runtime."""

    def __init__(self, config, seed, shard, shards):
        super().__init__(config, seed, shard, shards)
        self.block_close = shard == 0

    def close(self):
        if self.block_close:
            import signal
            import time

            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True:
                time.sleep(1)


@pytest.fixture
def cpu_env_factory(monkeypatch):
    from omegaconf import OmegaConf

    from rlinf.envs.sim.bench2dex import bench2dex_env

    spec = SimpleNamespace(
        action_dim=1,
        full_dof=1,
        main_camera="main",
        extra_cameras=(),
        wrist_cameras=(),
    )
    monkeypatch.setattr(bench2dex_env.Bench2DexSpec, "from_config", lambda config: spec)
    monkeypatch.setattr(bench2dex_env, "make_bench2dex_runtime", _CPURuntime)

    def create(*, fixed_ids=False, lazy=True):
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "auto_reset": False,
                "group_size": 2 if fixed_ids else 1,
                "use_fixed_reset_state_ids": fixed_ids,
                "enable_offload": lazy,
                "skip_intermediate_renders": True,
                "init_params": {"start_episode": 1},
            }
        )
        return bench2dex_env.Bench2DexEnv(cfg, 2, 0, 1, None)

    return create


@pytest.mark.parametrize("fixed_ids", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_offload_rebuild_preserves_episode_sequence(cpu_env_factory, fixed_ids, lazy):
    import torch

    env = cpu_env_factory(fixed_ids=fixed_ids, lazy=lazy)
    try:
        if lazy:
            assert env.env is None
            env.offload()
        _, info = env.reset()
        assert info["episode_index"] == ([1, 1] if fixed_ids else [1, 2])
        old_workers = list(env.env.workers)
        old_pids = {worker.process.pid for worker in old_workers}
        env.chunk_step(torch.zeros((2, 1, 1)))
        env.update_reset_state_ids()
        env.offload()
        assert env.env is None
        assert all(not worker.process.is_alive() for worker in old_workers)
        env.offload()
        _, info = env.reset()
        assert info["episode_index"] == ([2, 2] if fixed_ids else [3, 4])
        assert env.episode_seed.tolist() == ([124, 124] if fixed_ids else [125, 126])
        assert old_pids.isdisjoint(worker.process.pid for worker in env.env.workers)
    finally:
        env.close()
    env.close()
    env.offload()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_offload_rejects_live_episode(cpu_env_factory):
    env = cpu_env_factory()
    try:
        env.reset()
        with pytest.raises(RuntimeError, match="completed episodes"):
            env.offload()
        assert all(worker.process.is_alive() for worker in env.env.workers)
    finally:
        env.close()


def test_offload_reaps_child_ignoring_termination(cpu_env_factory, monkeypatch):
    import signal

    import torch

    from rlinf.envs.sim.bench2dex import bench2dex_env

    monkeypatch.setattr(
        bench2dex_env, "make_bench2dex_runtime", _UnresponsiveCloseRuntime
    )
    env = cpu_env_factory()
    workers = []
    try:
        env.reset()
        workers = list(env.env.workers)
        env.chunk_step(torch.zeros((2, 1, 1)))
        env.offload()
        assert env.env is None
        assert all(not worker.process.is_alive() for worker in workers)
        assert workers[0].process.exitcode == -signal.SIGKILL
        assert workers[1].process.exitcode == 0
        env.offload()
    finally:
        for worker in workers:
            if worker.process.is_alive():
                worker.process.kill()
            worker.process.join(timeout=5)
        env.close()


def test_runner_waits_for_environment_release():
    from contextlib import nullcontext

    from omegaconf import OmegaConf

    from rlinf.runners.embodied_runner import EmbodiedRunner

    events = []

    class Handle:
        def __init__(self, event):
            self.event = event

        def wait(self):
            events.append(self.event)
            return {}

    class WorkerGroup:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, method):
            def dispatch(*args, **kwargs):
                if method == "run_training":
                    assert "env.interact" in events
                return Handle(f"{self.name}.{method}")

            return dispatch

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"runner": {}, "env": {"train": {"enable_offload": True}}}
    )
    runner.actor = WorkerGroup("actor")
    runner.rollout = WorkerGroup("rollout")
    runner.env = WorkerGroup("env")
    runner.reward = None
    runner.global_step = 0
    runner.max_steps = runner.weight_sync_interval = 1
    runner.overlap_env_bootstrap = False
    runner.env_channel = runner.rollout_channel = runner.actor_channel = None
    runner.reward_channel = None
    runner.timer = lambda *args, **kwargs: nullcontext()
    runner._should_profile_step = lambda step: False
    runner.update_rollout_weights = lambda: None
    runner._maybe_eval_and_checkpoint = lambda step: None
    runner._log_step_metrics = lambda **kwargs: None
    runner._finish_run = lambda: None
    runner.run()
    assert events.index("env.interact") < events.index("actor.run_training")


@pytest.mark.parametrize(
    "key,value",
    [
        ("env.train.auto_reset", True),
        ("env.train.max_steps_per_rollout_epoch", 1),
        ("runner.use_training_pipeline", True),
        ("runner.enable_decoupled_mode", True),
        ("algorithm.loss_type", "decoupled_actor_critic"),
    ],
)
def test_environment_offload_config_contract(monkeypatch, key, value):
    from omegaconf import OmegaConf

    from rlinf.config import validate_bench2dex_cfg
    from rlinf.envs.sim.bench2dex.spec import Bench2DexSpec

    spec = SimpleNamespace(action_dim=1, max_control_steps=2)
    monkeypatch.setattr(Bench2DexSpec, "from_config", lambda config: spec)
    cfg = OmegaConf.create(
        {
            "runner": {},
            "rollout": {"pipeline_stage_num": 1},
            "algorithm": {
                "group_size": 1,
                "adv_type": "gae",
                "loss_type": "actor_critic",
            },
            "env": {
                "train": {
                    "env_type": "bench2dex",
                    "total_num_envs": 2,
                    "enable_offload": True,
                    "auto_reset": False,
                    "max_episode_steps": 2,
                    "max_steps_per_rollout_epoch": 2,
                    "init_params": {},
                }
            },
        }
    )
    model = OmegaConf.create(
        {"action_dim": 1, "num_action_chunks": 1, "model_type": "mlp"}
    )
    validate_bench2dex_cfg(cfg, model, 1, False)
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError, match="offload"):
        validate_bench2dex_cfg(cfg, model, 1, False)


@pytest.mark.parametrize("capture_intermediate", [False, True])
@pytest.mark.parametrize("stage_bonus", [0.0, 0.1])
@pytest.mark.parametrize(
    "success_bonus,complete_on_success", [(1.0, False), (0.2, True)]
)
def test_runtime_stage_rewards_preserve_chunk_and_camera_semantics(
    monkeypatch, capture_intermediate, stage_bonus, success_bonus, complete_on_success
):
    import sys
    from types import ModuleType

    from rlinf.envs.sim.bench2dex.runtime import Bench2DexRuntime

    class ExternalStageTracker:
        def __init__(self, stages, *, dt):
            pass

        def update(self, states, *, sim_step):
            return SimpleNamespace(current_completed={"place": states["placed"]})

    benchmark = ModuleType("benchmark")
    benchmark.__path__ = []
    stage_module = ModuleType("benchmark.stage_tracker")
    stage_module.StageTracker = ExternalStageTracker
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "benchmark.stage_tracker", stage_module)

    runtime = Bench2DexRuntime.__new__(Bench2DexRuntime)
    runtime.config = {
        "stage_reward_coef": stage_bonus,
        "success_reward_coef": success_bonus,
        "complete_stage_rewards_on_success": complete_on_success,
    }
    runtime.task = {"metrics": {"stages": [{"id": "place"}]}}
    runtime.spec = SimpleNamespace(
        physics_dt=0.1, control_decimation=1, max_control_steps=30
    )
    runtime.dof = SimpleNamespace(
        active_joint_names=("joint",), full_joint_names=("joint",)
    )
    runtime.contact_reader = None
    runtime.joint_limits = None
    runtime.early_stop = True
    state = {"placed": False}
    observations = []

    def advance():
        state["placed"] = True

    def observe():
        observations.append(runtime.control_steps)
        return {"frame": len(observations)}

    runtime._advance_physics = advance
    runtime._set_action = lambda action, layout: {}
    runtime._read_object_states = lambda: state
    runtime._read_robot_state = lambda: {}
    runtime._observe = observe
    runtime._info = lambda: {"episode_return": runtime.episode_return}
    runtime.metrics = SimpleNamespace(
        success=False,
        update=lambda *args, **kwargs: None,
        finalize=lambda **kwargs: _MetricsResult(),
    )
    runtime._reset_episode_counters()

    transitions = runtime.chunk_step(
        np.zeros((20, 1)), capture_intermediate_observations=capture_intermediate
    )
    assert len(transitions) == 20
    assert [item[1] for item in transitions] == pytest.approx(
        [stage_bonus] + [0.0] * 19
    )
    assert len(observations) == (20 if capture_intermediate else 1)
    assert runtime.episode_stage_reward == pytest.approx(stage_bonus)
    assert runtime.episode_success_reward == 0.0

    runtime.metrics.success = True
    final = runtime.step(np.zeros(1))
    assert final[1] == success_bonus
    assert final[2] is True
    assert final[4]["metrics"]["success_reward_return"] == success_bonus
    assert final[4]["metrics"]["stage_reward_return"] == pytest.approx(stage_bonus)
    assert final[4]["metrics"]["stage_reward_count"] == int(stage_bonus > 0)
    assert runtime.episode_return == pytest.approx(success_bonus + stage_bonus)


@pytest.mark.parametrize("initial_count", range(5))
def test_runtime_success_settles_four_stage_reward_budget(initial_count):

    from rlinf.envs.sim.bench2dex.runtime import Bench2DexRuntime

    runtime = Bench2DexRuntime.__new__(Bench2DexRuntime)
    stages = ("phillips", "flat", "wrench", "drill")
    runtime._stage_reward = StageCompletionReward(stages, bonus=0.2)
    runtime._stage_reward.reset(
        {sid: index < initial_count for index, sid in enumerate(stages)}
    )
    runtime._pending_stage_reward = runtime._stage_reward.update(
        dict.fromkeys(stages, True)
    )
    runtime.control_steps = runtime.physics_steps = 0
    runtime.episode_return = 0.0
    runtime.episode_success_reward = runtime.episode_stage_reward = 0.0
    runtime.success_once = False
    runtime.early_stop = True
    runtime.success_reward_coef = 0.2
    runtime.complete_stage_rewards_on_success = True
    runtime.spec = SimpleNamespace(max_control_steps=681, control_decimation=3)
    runtime.metrics = SimpleNamespace(
        success=True, finalize=lambda **kwargs: _MetricsResult()
    )
    runtime._info = lambda: {"success_once": runtime.success_once}

    first = runtime._finish_action({"terminal_frame": True})
    assert first[1] == pytest.approx(1.0)
    assert first[2] is True
    assert runtime.success_once is True
    assert runtime.episode_success_reward == pytest.approx(0.2)
    assert runtime.episode_stage_reward == pytest.approx(0.8)
    assert runtime._finish_action({})[1] == 0.0


def test_stage_reward_is_immediate_and_cannot_be_farmed():
    reward = StageCompletionReward(["place"], bonus=0.1)
    assert reward.update({"place": False}) == 0.0
    assert reward.update({"place": True}) == pytest.approx(0.1)
    assert reward.update({"place": False}) == 0.0
    assert reward.update({"place": True}) == 0.0
    assert reward.total_reward == pytest.approx(0.1)
    assert reward.awarded_count == 1


def test_stage_reward_reset_excludes_initial_completion():
    reward = StageCompletionReward(["a", "b"], bonus=0.1)
    reward.reset({"a": True})
    assert reward.update({"a": True, "b": True}) == pytest.approx(0.1)
    assert reward.awarded_count == 1
    reward.reset()
    assert reward.total_reward == 0.0
    assert reward.awarded_count == 0
    assert reward.update({"a": True, "b": True}) == pytest.approx(0.2)


@pytest.mark.parametrize("bonus", [-1.0, float("nan"), float("inf")])
def test_stage_reward_rejects_invalid_configuration(bonus):
    with pytest.raises(ValueError):
        StageCompletionReward(["place"], bonus=bonus)


@pytest.mark.parametrize("stage_ids", [[], ["place", "place"]])
def test_stage_reward_rejects_invalid_stage_ids(stage_ids):
    with pytest.raises(ValueError):
        StageCompletionReward(stage_ids, bonus=0.1)


@pytest.mark.parametrize("initial_count", range(5))
def test_success_completes_stage_budget_once(initial_count):
    stages = ("phillips", "flat", "wrench", "drill")
    reward = StageCompletionReward(stages, bonus=0.2)
    initial = {sid: index < initial_count for index, sid in enumerate(stages)}
    reward.reset(initial)
    assert reward.total_reward == 0.0
    earned = reward.update(dict.fromkeys(stages, True))
    assert earned == pytest.approx((4 - initial_count) * 0.2)
    assert earned + reward.complete() + 0.2 == pytest.approx(1.0)
    assert reward.awarded_count == 4
    assert reward.total_reward == pytest.approx(0.8)
    assert reward.complete() == 0.0
    assert reward.update(dict.fromkeys(stages, True)) == 0.0
