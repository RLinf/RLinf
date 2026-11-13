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

"""Bench2Dex evaluation, scene lifecycle, rewards and recipe contracts."""

import json
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
def bench2dex_params(tmp_path):
    source = tmp_path / "bench2dex"
    metadata = {
        "scenes/test.yaml": {"metrics": {"expert_time_step": 4}},
        "robots/active_dof_maps.yml": {
            "robots": {
                "test": {
                    "full_dof": 1,
                    "active_dof": 1,
                    "full_joint_names": ["joint"],
                    "active_indices": [0],
                }
            }
        },
        "configs/collect/default.yaml": {},
    }
    for name, value in metadata.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return {
        "source_path": str(source),
        "assets_path": str(tmp_path / "assets"),
        "task": "test",
        "embodiment": "test",
        "collect_config": "configs/collect/default.yaml",
        "physics_dt": 0.1,
        "control_decimation": 3,
        "cameras": {"main": "main"},
        "start_episode": 1,
    }


@pytest.fixture
def cpu_env_factory(monkeypatch, bench2dex_params):
    from omegaconf import OmegaConf

    from rlinf.envs.sim.bench2dex import bench2dex_env

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
                "init_params": bench2dex_params,
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


def test_runner_waits_for_environment_release(cpu_env_factory):
    from contextlib import nullcontext

    import torch
    from omegaconf import OmegaConf

    from rlinf.runners.embodied_runner import EmbodiedRunner

    env = cpu_env_factory()
    pools, episodes = [], []

    class Handle:
        def __init__(self, finish=lambda: {}):
            self.finish = finish
            self.completed = False

        def wait(self):
            if not self.completed:
                self.result = self.finish()
                self.completed = True
            return self.result

    def ready(*args, **kwargs):
        return Handle()

    def interact(**kwargs):
        _, info = env.reset()
        episodes.append(info["episode_index"])
        pools.append(list(env.env.workers))

        def finish():
            env.chunk_step(torch.zeros((2, 1, 1)))
            env.update_reset_state_ids()
            env.offload()
            return {}

        return Handle(finish)

    def train():
        assert env.env is None
        assert all(not worker.process.is_alive() for worker in pools[-1])
        return Handle()

    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"runner": {}, "env": {"train": {"enable_offload": True}}}
    )
    runner.actor = SimpleNamespace(
        set_global_step=ready,
        recv_rollout_trajectories=ready,
        compute_advantages_and_returns=ready,
        run_training=train,
    )
    runner.rollout = SimpleNamespace(set_global_step=ready, generate=ready)
    runner.env = SimpleNamespace(set_global_step=ready, interact=interact)
    runner.reward = None
    runner.global_step = 0
    runner.max_steps = 2
    runner.weight_sync_interval = 1
    runner.overlap_env_bootstrap = False
    runner.env_channel = runner.rollout_channel = runner.actor_channel = None
    runner.reward_channel = None
    runner.timer = lambda *args, **kwargs: nullcontext()
    runner._should_profile_step = lambda step: False
    runner.update_rollout_weights = lambda: None
    runner._maybe_eval_and_checkpoint = lambda step: {}
    runner._log_step_metrics = lambda **kwargs: None
    runner._finish_run = lambda: None
    try:
        runner.run()
        assert runner.global_step == 2
        assert episodes == [[1, 2], [3, 4]]
        assert {w.process.pid for w in pools[0]}.isdisjoint(
            w.process.pid for w in pools[1]
        )
    finally:
        env.close()


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
def test_environment_offload_config_contract(bench2dex_params, key, value):
    from omegaconf import OmegaConf

    from rlinf.config import validate_bench2dex_cfg

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
                    "init_params": bench2dex_params,
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


@pytest.fixture
def runtime_factory(monkeypatch):
    import sys
    from types import ModuleType

    from rlinf.envs.sim.bench2dex.runtime import Bench2DexRuntime

    class ExternalStageTracker:
        def __init__(self, stages, *, dt):
            self.stage_ids = [stage["id"] for stage in stages]

        def update(self, states, *, sim_step):
            return SimpleNamespace(
                current_completed={sid: states[sid] for sid in self.stage_ids}
            )

    benchmark = ModuleType("benchmark")
    benchmark.__path__ = []
    stage_module = ModuleType("benchmark.stage_tracker")
    stage_module.StageTracker = ExternalStageTracker
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "benchmark.stage_tracker", stage_module)

    def create(*, stages=("place",), initial=(), bonus=0.1, complete=False):
        states = dict.fromkeys(stages, False)
        states.update(dict.fromkeys(initial, True))
        observations = []
        runtime = object.__new__(Bench2DexRuntime)
        runtime.config = {
            "stage_reward_coef": bonus,
            "success_reward_coef": 0.2,
            "complete_stage_rewards_on_success": complete,
        }
        runtime.task = {"metrics": {"stages": [{"id": sid} for sid in stages]}}
        runtime.spec = SimpleNamespace(
            physics_dt=0.1, control_decimation=1, max_control_steps=30
        )
        runtime.dof = SimpleNamespace(
            active_joint_names=("joint",), full_joint_names=("joint",)
        )
        runtime.contact_reader = runtime.joint_limits = None
        runtime.early_stop = True

        def observe():
            observations.append(runtime.control_steps)
            return {"frame": len(observations)}

        runtime._advance_physics = lambda: None
        runtime._set_action = lambda action, layout: {}
        runtime._read_object_states = lambda: states
        runtime._read_robot_state = lambda: {}
        runtime._observe = observe
        runtime._info = lambda: {"episode_return": runtime.episode_return}
        runtime.metrics = SimpleNamespace(
            success=False,
            update=lambda *args, **kwargs: None,
            finalize=lambda **kwargs: _MetricsResult(),
        )
        runtime._reset_episode_counters()
        return runtime, states, observations

    return create


@pytest.mark.parametrize("capture_intermediate", [False, True])
@pytest.mark.parametrize("stage_bonus", [0.0, 0.1])
def test_runtime_stage_rewards_preserve_chunk_and_camera_semantics(
    runtime_factory, capture_intermediate, stage_bonus
):
    runtime, states, observations = runtime_factory(bonus=stage_bonus)
    states["place"] = True
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
    assert final[1] == pytest.approx(0.2)
    assert final[2] is True
    assert final[4]["metrics"]["success_reward_return"] == pytest.approx(0.2)
    assert final[4]["metrics"]["stage_reward_return"] == pytest.approx(stage_bonus)
    assert final[4]["metrics"]["stage_reward_count"] == int(stage_bonus > 0)
    assert runtime.episode_return == pytest.approx(0.2 + stage_bonus)
    assert runtime.step(np.zeros(1))[1] == 0.0


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("initial_count", [0, 2, 4])
def test_runtime_success_pays_unfinished_stages_only_when_enabled(
    runtime_factory, initial_count, complete
):
    stages = ("phillips", "flat", "wrench", "drill")
    runtime, states, _ = runtime_factory(
        stages=stages, initial=stages[:initial_count], bonus=0.2, complete=complete
    )
    if initial_count < len(stages):
        states[stages[initial_count]] = True
    runtime.metrics.success = True

    first = runtime.step(np.zeros(1))
    expected_stage = 0.8 if complete else 0.2 * (initial_count < len(stages))
    assert first[1] == pytest.approx(0.2 + expected_stage)
    assert first[2] is True
    assert runtime.success_once is True
    assert runtime.episode_success_reward == pytest.approx(0.2)
    assert runtime.episode_stage_reward == pytest.approx(expected_stage)
    assert runtime.step(np.zeros(1))[1] == 0.0


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


@pytest.fixture
def compose_bench2dex_recipe(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[2]
    for key, value in {
        "REPO_PATH": str(repo),
        "EMBODIED_PATH": str(repo / "examples/embodiment"),
        "BENCH2DEX_PATH": "/bench",
        "BENCH2DEX_ASSETS_PATH": "/assets",
        "BENCH2DEX_PI05_PATH": "/model",
        "BENCH2DEX_ANCHOR_DIR": "/anchors",
    }.items():
        monkeypatch.setenv(key, value)

    def load(config_dir, config_name):
        with initialize_config_dir(str(repo / config_dir), version_base="1.1"):
            cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        return cfg

    return load


@pytest.mark.parametrize(
    "config_name,task,embodiment,action_dim,budget,slots",
    [
        (
            "bench2dex_task22_ppo_openpi_pi05",
            "22_tool_box_loading",
            "multi_panda_with_orca",
            40,
            681,
            700,
        ),
        (
            "bench2dex_task22_async_ppo_openpi_pi05",
            "22_tool_box_loading",
            "multi_panda_with_orca",
            40,
            681,
            700,
        ),
        (
            "bench2dex_task22_openpi_pi05_eval",
            "22_tool_box_loading",
            "multi_panda_with_orca",
            40,
            681,
            700,
        ),
        (
            "bench2dex_task64_ppo_openpi_pi05",
            "64_sports_ball_cup_sort",
            "multi_panda_with_allegro",
            32,
            665,
            680,
        ),
        (
            "bench2dex_task64_async_ppo_openpi_pi05",
            "64_sports_ball_cup_sort",
            "multi_panda_with_allegro",
            32,
            665,
            680,
        ),
        (
            "bench2dex_task64_grpo_openpi_pi05",
            "64_sports_ball_cup_sort",
            "multi_panda_with_allegro",
            32,
            665,
            680,
        ),
        (
            "bench2dex_task64_openpi_pi05_eval",
            "64_sports_ball_cup_sort",
            "multi_panda_with_allegro",
            32,
            665,
            680,
        ),
    ],
)
def test_bench2dex_recipes(
    compose_bench2dex_recipe, config_name, task, embodiment, action_dim, budget, slots
):
    only_eval = config_name.endswith("_eval")
    config_dir = "evaluations/bench2dex" if only_eval else "examples/embodiment/config"
    cfg = compose_bench2dex_recipe(config_dir, config_name)
    model = cfg.rollout.model if only_eval else cfg.actor.model
    assert model.action_dim == model.openpi.model_action_dim == action_dim
    assert model.num_action_chunks == model.openpi.action_horizon == 20
    assert model.model_path == "/model"
    assert cfg.rollout.rollout_backend == "huggingface"
    for split in ("eval",) if only_eval else ("train", "eval"):
        env = cfg.env[split]
        assert env.init_params.task == task
        assert env.init_params.embodiment == embodiment
        assert env.auto_reset is False
        assert env.max_episode_steps == budget
        assert env.max_steps_per_rollout_epoch == slots
        assert env.init_params.stage_reward_coef == 0
        assert env.init_params.success_reward_coef == 1
    if only_eval:
        assert cfg.env.eval.skip_intermediate_renders is False
        return
    is_async = "_async_" in config_name
    is_grpo = "_grpo_" in config_name
    assert cfg.actor.enable_offload and cfg.rollout.enable_offload
    assert cfg.env.train.enable_offload is (not is_async)
    assert cfg.env.eval.enable_offload is True
    assert cfg.env.train.skip_intermediate_renders is True
    assert cfg.actor.model.add_value_head is (not is_grpo)
    assert cfg.algorithm.loss_type == (
        "decoupled_actor_critic" if is_async else "actor" if is_grpo else "actor_critic"
    )
    samples = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch * (slots // 20)
    assert samples % cfg.actor.global_batch_size == 0
    assert cfg.actor.global_batch_size % (4 * cfg.actor.micro_batch_size) == 0
    if action_dim == 40:
        assert samples == 4480
        assert cfg.runner.val_check_interval == 10
        assert cfg.runner.save_interval == 2
        assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 32
        assert cfg.env.train.init_params.generalization_profile == "none"
        assert cfg.actor.model.openpi.noise_level == 0.3
        assert cfg.algorithm.use_dual_clip is False


@pytest.mark.parametrize(
    "config_name", ["bench2dex_ppo_openpi_pi05", "bench2dex_async_ppo_openpi_pi05"]
)
def test_bench2dex_e2e_recipes(compose_bench2dex_recipe, config_name):
    cfg = compose_bench2dex_recipe("tests/e2e_tests/embodied", config_name)
    assert cfg.runner.max_steps == cfg.runner.save_interval == 2
    assert cfg.runner.val_check_interval == 10
    assert (
        cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch * 35
        == cfg.actor.global_batch_size
    )
    assert cfg.env.train.enable_offload is ("_async_" not in config_name)


def test_bench2dex_e2e_evaluation_recipe(compose_bench2dex_recipe):
    cfg = compose_bench2dex_recipe(
        "tests/e2e_tests/evaluations", "bench2dex_openpi_pi05_eval"
    )
    assert cfg.runner.only_eval is True
    assert cfg.runner.task_type == "embodied_eval"
    assert cfg.runner.save_interval == -1
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 4
    assert cfg.env.eval.max_episode_steps == 681
    assert cfg.env.eval.max_steps_per_rollout_epoch == 700
    assert cfg.env.eval.skip_intermediate_renders is False
    assert cfg.env.eval.video_cfg.save_video is False
    assert cfg.env.eval.init_params.task == "22_tool_box_loading"
    assert cfg.rollout.model.openpi.task == "eval"
    assert cfg.rollout.model.openpi.noise_rng == "jax"
    assert cfg.rollout.rollout_backend == "huggingface"
