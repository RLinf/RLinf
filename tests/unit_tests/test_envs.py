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

"""Heterogeneous env modes, and the observations env workers send the policy."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.env.env_worker as env_worker_module
from rlinf.config import (
    resolve_env_mode_cfg,
    validate_embodied_cfg,
    validate_env_policy_fields,
    validate_env_training_cfg,
    validate_eval_env_cfg,
)
from rlinf.data.schema.embodied_types import PolicyInput
from rlinf.envs.layout import EnvLayout
from rlinf.envs.policy_obs import PolicyObsSpec
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.workers.env.env_worker import EnvWorker


def _env(env_type: str, total_num_envs: int, **fields) -> dict:
    return {
        "env_type": env_type,
        "total_num_envs": total_num_envs,
        "rollout_epoch": 1,
        "max_steps_per_rollout_epoch": 80,
        "max_episode_steps": 80,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
        "video_cfg": {"save_video": False},
        **fields,
    }


@pytest.mark.parametrize("failure", ["timeout", "eof"])
def test_isaaclab_startup_failure_reaps_subprocess(failure):
    import multiprocessing as mp
    import time

    from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv

    def make_env():
        if failure == "timeout":
            time.sleep(60)
        raise RuntimeError("simulator initialization failed")

    env = SubProcIsaacLabEnv(make_env, timeout=0.1 if failure == "timeout" else 30)
    pid = env.isaac_lab_process.pid
    try:
        with pytest.raises(
            TimeoutError if failure == "timeout" else (EOFError, ConnectionResetError)
        ):
            env.device()
        assert all(child.pid != pid for child in mp.active_children())
    finally:
        env.close()


def test_isaaclab_subprocess_transports_steps_and_closes_once(tmp_path):
    from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv

    class Simulator:
        device = "cpu"

        def reset(self, seed, env_ids=None):
            return seed, env_ids

        def step(self, action):
            return action

        def close(self):
            (tmp_path / "task_closed").touch(exist_ok=False)

    class App:
        def close(self):
            (tmp_path / "app_closed").touch(exist_ok=False)

    env = SubProcIsaacLabEnv(lambda: (Simulator(), App()))
    try:
        assert env.device() == "cpu"
        seed, ids = env.reset(seed=3, env_ids=torch.tensor([1, 2]))
        assert seed == 3 and ids.tolist() == [1, 2]
        action = torch.randn(2, 12)
        assert torch.equal(env.step(action), action)
    finally:
        env.close()
    env.close()
    assert (tmp_path / "task_closed").exists() and (tmp_path / "app_closed").exists()


@pytest.mark.parametrize(
    "auto_reset,ignore_terminations", [(False, False), (True, True)]
)
def test_go2_rejects_episode_modes_its_simulator_cannot_honor(
    auto_reset, ignore_terminations
):
    from rlinf.envs.sim.isaaclab.tasks.go2 import Go2IsaacLabEnv

    cfg = OmegaConf.create(
        {"auto_reset": auto_reset, "ignore_terminations": ignore_terminations}
    )
    with pytest.raises(
        ValueError, match="auto_reset=true and ignore_terminations=false"
    ):
        Go2IsaacLabEnv(cfg, 2)


@pytest.mark.parametrize(
    "task,task_id,family_id,label,seed,legacy",
    [
        ("A", 1, 0, "A2_omni", None, False),
        ("B", 2, 1, "B1_rough", 2, False),
        ("C", 3, 1, "B2_stairs", None, False),
        ("C", 3, 1, "B2_stairs", None, True),
        ("C", 3, 1, "B2_stairs", 2, True),
    ],
)
def test_go2_evaluator_pools_same_task_episodes(
    tmp_path, monkeypatch, task, task_id, family_id, label, seed, legacy
):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from examples.embodiment.go2_abc import evaluate, evaluate_stairs
    from rlinf.envs.sim.isaaclab.tasks.go2 import FLAT, ROUGH, STAIRS
    from rlinf.models.embodiment.go2_policy import Go2Policy

    gym_id = {"A": FLAT, "B": ROUGH, "C": STAIRS}[task]
    report_seed = 0 if legacy and seed is None else seed
    train_name = label if report_seed is None else f"{label}_s{report_seed}"
    checkpoint, output = tmp_path / "full_weights.pt", tmp_path / "output"
    torch.save(Go2Policy().state_dict(), checkpoint)
    seeds = []

    def run_benchmark(command, env, check):
        assert check
        assert Path(command[1]).name == "evaluate_continuation.py"
        options = command[2:]
        assert options[options.index("--task") + 1] == gym_id
        assert options[options.index("--eval_task") + 1] == label
        assert options[options.index("--report_train_task") + 1] == train_name
        assert options[options.index("--num_envs") + 1] == "256"
        assert options[options.index("--num_episodes") + 1] == "1024"
        assert (
            "--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry "
            f"--/persistent/isaac/asset_root/cloud={tmp_path / 'assets'}"
        ) in options
        assert env.get("GO2_B1_NUMERIC_GUARD") == ("1" if task == "B" else None)
        eval_seed = int(env["GO2_EVAL_SEED"])
        seeds.append(eval_seed)
        directory = Path(options[options.index("--output_dir") + 1]) / train_name
        assert directory == output / str(eval_seed) / train_name
        directory.mkdir(parents=True)
        size = 1024 if eval_seed == 23456 else 1026
        success = eval_seed == 23456
        row = {"task_id": task_id, "family_id": family_id, "episode_success": success}
        summary = {
            "train_task": train_name,
            "eval_task": gym_id,
            "num_episodes": size,
            "success_rate": float(success),
        }
        if task == "C":
            row["physical_obstacle_passage"] = not success
            summary.update(
                required_obstacle_metrics_complete=True,
                physical_measured_episodes=size,
                physical_passage_successes=0 if success else size,
                physical_obstacle_passage_rate=float(not success),
            )
        (directory / "summary.json").write_text(json.dumps({label: summary}))
        (directory / f"{label}.json").write_text(
            json.dumps({"summary": summary, "episodes": [row] * size})
        )

    monkeypatch.setenv("GO2_ABC_DATA", str(tmp_path))
    arguments = [] if legacy else ["--task", task]
    if legacy:
        initial = tmp_path / "models/C/baseline.pt"
        if seed is not None:
            initial = tmp_path / "custom.pt"
            arguments += ["--initial-checkpoint", str(initial)]
        initial.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": Go2Policy().state_dict()}, initial)
    if seed is not None:
        arguments += ["--train-seed", str(seed)]
    arguments.extend(
        [str(checkpoint), str(output), "--asset-root", str(tmp_path / "assets")]
    )
    monkeypatch.setattr(sys, "argv", ["evaluate", *arguments])
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("GO2_B1_NUMERIC_GUARD", "0")
    monkeypatch.setattr(evaluate, "subprocess", SimpleNamespace(run=run_benchmark))
    (evaluate_stairs.main if legacy else evaluate.main)()

    assert seeds == [23456, 34567]
    expected = {
        "num_episodes": 2050,
        "tracking_successes": 1024,
        "tracking_rate": 1024 / 2050,
    }
    if task == "C":
        expected.update(passage_successes=1026, passage_rate=1026 / 2050)
    assert json.loads((output / "summary.json").read_text()) == expected


def test_go2_evaluator_rejects_negative_training_seed(monkeypatch, capsys):
    import sys

    from examples.embodiment.go2_abc.evaluate import main

    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--task", "B", "--train-seed", "-1", "ckpt", "out"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--train-seed must be nonnegative" in capsys.readouterr().err


def test_go2_stairs_checks_its_frozen_baseline_by_default(tmp_path, monkeypatch):
    import sys

    from examples.embodiment.go2_abc import evaluate_stairs
    from rlinf.models.embodiment.go2_policy import Go2Policy

    baseline = tmp_path / "models/C/baseline.pt"
    baseline.parent.mkdir(parents=True)
    weights = Go2Policy().state_dict()
    torch.save({"model_state_dict": weights}, baseline)
    checkpoint = tmp_path / "full_weights.pt"
    torch.save(weights, checkpoint)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("GO2_ABC_DATA", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["evaluate_stairs", str(checkpoint), str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="No update to actor"):
        evaluate_stairs.main()


@pytest.mark.parametrize(
    "entry",
    [
        ["-m", "examples.embodiment.go2_abc.evaluate_stairs"],
        ["examples/embodiment/go2_abc/evaluate_stairs.py"],
    ],
)
def test_go2_stairs_command_line_help(entry, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "checkpoint output" in result.stdout


@pytest.mark.parametrize("wrapped,unchanged", [(False, "actor"), (True, "critic")])
def test_go2_evaluator_checks_supplied_starting_weights(tmp_path, wrapped, unchanged):
    from examples.embodiment.go2_abc.evaluate import export_policy
    from rlinf.models.embodiment.go2_policy import Go2Policy

    initial, checkpoint, policy = (
        tmp_path / name for name in ("start", "final", "policy")
    )
    weights = Go2Policy().state_dict()
    torch.save({"model_state_dict": weights} if wrapped else weights, initial)
    updated = "actor" if unchanged == "critic" else "critic"
    weights[f"{updated}.0.bias"].add_(1)
    torch.save(weights, checkpoint)
    with pytest.raises(ValueError, match=f"No update to {unchanged}"):
        export_policy(checkpoint, initial, policy)
    weights[f"{unchanged}.0.bias"].add_(1)
    torch.save(weights, checkpoint)
    export_policy(checkpoint, initial, policy)
    exported = torch.load(policy, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    assert all(torch.equal(value, exported[key]) for key, value in weights.items())


@pytest.fixture
def go2_manager_cfg(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Manager:
        def __init__(self, cfg):
            self.device = "cpu"
            self.episode_length_buf = torch.zeros(cfg.scene.num_envs, dtype=torch.long)
            self.action_manager = SimpleNamespace(total_action_dim=12)
            self.observation_manager = SimpleNamespace(
                group_obs_dim={"policy": (235,)}, compute=self.compute
            )
            self.cfg = cfg

        def compute(self, update_history):
            assert not update_history
            torch.rand(1)
            return {"policy": torch.full((2, 235), 7.0)}

        def _reset_idx(self, ids):
            self.episode_length_buf[ids] = 0

        def reset(self, env_ids):
            return {"policy": torch.zeros(2, 235)}, {}

        def step(self, actions):
            self._reset_idx(torch.tensor([0]))
            obs = {"policy": torch.zeros(2, 235)}
            return (
                obs,
                actions[:, 0],
                torch.tensor([True, False]),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

    task_cfg = SimpleNamespace(
        scene=SimpleNamespace(),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        rewards=SimpleNamespace(
            base_height=SimpleNamespace(
                func=torch.square,
                weight=-10.0,
                params={
                    "target_height": 0.38,
                    "asset_cfg": SimpleNamespace(name="robot"),
                    "sensor_cfg": SimpleNamespace(name="height_scanner"),
                },
            )
        ),
    )
    modules = {
        "isaaclab.app": SimpleNamespace(
            AppLauncher=lambda **kw: SimpleNamespace(app=None)
        ),
        "envs.registry": SimpleNamespace(register_tasks=lambda: None),
        "isaaclab.envs": SimpleNamespace(ManagerBasedRLEnv=Manager),
        "isaaclab_tasks.utils": SimpleNamespace(
            load_cfg_from_registry=lambda *args: task_cfg
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return task_cfg


def test_go2_rough_action_boundary_and_terminal_observation(go2_manager_cfg):
    from rlinf.envs import get_env_cls
    from rlinf.envs.sim.isaaclab.tasks.go2 import ROUGH, Go2IsaacLabEnv, _make_go2_env

    cfg = OmegaConf.create({"seed": 0, "init_params": {"id": ROUGH}})
    assert get_env_cls("isaaclab", cfg) is Go2IsaacLabEnv
    task, _ = _make_go2_env(cfg.init_params, cfg.seed, 2, max_episode_steps=7)
    assert task.task.max_episode_length == 7
    assert task.task.cfg.episode_length_s == pytest.approx(0.14)
    task.reset()
    assert (task.task.episode_length_buf < 7).all()
    state = torch.get_rng_state()
    actions = torch.tensor([[-150.0] * 12, [145.0] * 12])
    obs, reward, _, _, info = task.step(actions)
    assert torch.equal(torch.get_rng_state(), state)
    assert reward.tolist() == [-100.0, 100.0]
    assert obs["states"].eq(0).all()
    assert info["final_observation"]["states"][0].eq(7).all()
    assert info["_final_observation"].tolist() == [True, False]
    actions[1, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        task.step(actions)


@pytest.mark.parametrize("task_name", ["FLAT", "ROUGH", "STAIRS"])
def test_go2_base_height_uses_ground_returns(go2_manager_cfg, task_name):
    from types import SimpleNamespace

    from rlinf.envs.sim.isaaclab.tasks import go2

    params = OmegaConf.create({"id": getattr(go2, task_name)})
    task, _ = go2._make_go2_env(params, 0, 3, 1000)
    term = task.task.cfg.rewards.base_height
    assert term.func is go2.terrain_base_height_l2
    generator = torch.Generator().manual_seed(41)
    hits = torch.randn(3, 187, 3, generator=generator)
    root = torch.randn(3, 3, generator=generator)
    task.task.scene = {
        "robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=root)),
        "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits)),
    }
    original = torch.square(root[:, 2] - (0.38 + torch.mean(hits[..., 2], dim=1)))
    assert torch.equal(term.func(task.task, **term.params), original)
    assert term.weight == -10.0

    hits[1, :-2] = torch.inf
    hits[1, -2:, 2] = torch.tensor([1.0, 3.0])
    hits[2] = torch.inf
    penalty = term.func(task.task, **term.params)
    assert torch.equal(penalty[0], original[0])
    assert torch.equal(penalty[1], (root[1, 2] - 2.38).square())
    assert penalty[2].item() == 0.0

    for invalid in (torch.nan, -torch.inf, 0.0):
        hits[2, 0, 0] = invalid
        with pytest.raises(FloatingPointError, match="terrain hit"):
            term.func(task.task, **term.params)
    hits[2] = torch.inf
    root[2, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="root position"):
        term.func(task.task, **term.params)
    root[2, 0] = 0
    root[0, 2] = 1e30
    with pytest.raises(FloatingPointError, match="height penalty"):
        term.func(task.task, **term.params)


def test_go2_bootstrap_observation_mixes_terminal_and_continuing_rows():
    from types import SimpleNamespace

    from rlinf.envs.sim.isaaclab.tasks.go2 import STAIRS, _Go2Task

    terminal = torch.arange(3 * 235, dtype=torch.float32).reshape(3, 235)
    next_states = terminal + 1000

    class Simulator:
        device = "cpu"
        observation_manager = SimpleNamespace(group_obs_dim={"policy": (235,)})
        action_manager = SimpleNamespace(total_action_dim=12)

        def step(self, actions):
            self.terminal_observation = terminal
            terminated = torch.tensor([False, False, True])
            truncated = torch.tensor([True, False, False])
            return {"policy": next_states}, torch.zeros(3), terminated, truncated, {}

    task = _Go2Task(Simulator(), SimpleNamespace(id=STAIRS), 3)
    obs, _, _, _, info = task.step(torch.zeros(3, 12))

    assert info["_final_observation"].tolist() == [True, False, True]
    expected = torch.stack([terminal[0], next_states[1], terminal[2]])
    for key in ("states", "critic_states"):
        torch.testing.assert_close(obs[key], next_states)
        torch.testing.assert_close(info["final_observation"][key], expected)


def _hetero_mode(**envs: dict):
    return OmegaConf.create({"envs": envs})


def test_homogeneous_mode_is_hosted_unchanged_by_every_worker():
    mode_cfg = OmegaConf.create(_env("libero", 8))
    layout = EnvLayout(mode_cfg, world_size=2, path="env.train")

    assignment = layout.assign(1)

    assert assignment.cfg is mode_cfg
    assert (assignment.name, assignment.rank, assignment.world_size) == (None, 1, 2)
    assert layout.mode_fields() == {}
    metrics = {"success_once": torch.ones(2)}
    assert assignment.scoped(metrics) == metrics


def test_listed_envs_get_contiguous_worker_blocks_sized_by_their_share():
    mode_cfg = _hetero_mode(libero=_env("libero", 48), maniskill=_env("maniskill", 16))
    layout = EnvLayout(mode_cfg, world_size=4, path="env.train")

    hosted = [(a.name, a.rank, a.world_size) for a in map(layout.assign, range(4))]

    assert hosted == [
        ("libero", 0, 3),
        ("libero", 1, 3),
        ("libero", 2, 3),
        ("maniskill", 0, 1),
    ]
    assert layout.assign(3).cfg is mode_cfg.envs.maniskill


@pytest.mark.parametrize(
    ("libero_fields", "agreed"),
    [
        ({}, {"max_steps_per_rollout_epoch": 80, "max_episode_steps": 80}),
        ({"max_episode_steps": 300, "max_steps_per_rollout_epoch": 320}, {}),
    ],
)
def test_heterogeneous_mode_carries_only_the_loop_fields_its_envs_agree_on(
    libero_fields, agreed
):
    mode_cfg = _hetero_mode(
        libero=_env("libero", 8, **libero_fields),
        maniskill=_env("maniskill", 8),
    )

    fields = EnvLayout(mode_cfg, world_size=2, path="env.train").mode_fields()

    assert fields == {
        "total_num_envs": 16,
        "rollout_epoch": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        **agreed,
    }


@pytest.mark.parametrize(
    ("mode", "world_size", "match"),
    [
        # 64 envs on 2 workers is 32 per worker, which 48 LIBERO envs cannot fill.
        (
            {
                "envs": {
                    "libero": _env("libero", 48),
                    "maniskill": _env("maniskill", 16),
                }
            },
            2,
            r"libero.total_num_envs must be a multiple of 32",
        ),
        (
            {
                "max_steps_per_rollout_epoch": 80,
                "envs": {
                    "libero": _env("libero", 8),
                    "maniskill": _env("maniskill", 8, max_steps_per_rollout_epoch=40),
                },
            },
            2,
            r"env.train.max_steps_per_rollout_epoch is 80, but the envs under "
            r"env.train.envs set max_steps_per_rollout_epoch to",
        ),
        (
            {"video_cfg": {"save_video": False}, "envs": {"libero": _env("libero", 8)}},
            1,
            r"env.train.video_cfg would apply to none of them",
        ),
        (
            {
                "total_num_envs": 32,
                "envs": {
                    "libero": _env("libero", 8),
                    "maniskill": _env("maniskill", 8),
                },
            },
            2,
            r"env.train.total_num_envs is 32, but the envs under env.train.envs give 16",
        ),
        (
            {"envs": {"libero": {"total_num_envs": 8}}},
            1,
            r"env.train.envs.libero must be an env config with env_type",
        ),
        (
            {"envs": {"libero": _env("libero", 0)}},
            1,
            r"total_num_envs must be a positive int",
        ),
    ],
)
def test_layout_rejects_modes_its_workers_cannot_run(mode, world_size, match):
    with pytest.raises(ValueError, match=match):
        EnvLayout(OmegaConf.create(mode), world_size=world_size, path="env.train")


def test_each_env_reports_its_own_metrics_beside_the_overall_ones():
    layout = EnvLayout(
        _hetero_mode(libero=_env("libero", 4), maniskill=_env("maniskill", 2)),
        world_size=3,
        path="env.eval",
    )
    success_per_rank = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ]

    metrics = compute_evaluate_metrics(
        [
            {
                "success_once": success,
                **layout.assign(rank).scoped({"success_once": success}),
            }
            for rank, success in enumerate(success_per_rank)
        ]
    )

    assert float(metrics["success_once"]) == pytest.approx(0.5)
    assert float(metrics["libero/success_once"]) == pytest.approx(0.75)
    assert float(metrics["maniskill/success_once"]) == pytest.approx(0.0)
    assert metrics["num_trajectories"] == 6


def test_env_worker_runs_its_env_on_the_workers_hosting_that_env(monkeypatch):
    created = []

    class Simulator:
        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, **_):
            created.append((cfg.env_type, num_envs, seed_offset, total_num_processes))

    requested = []

    def get_env_cls(env_type, env_cfg):
        requested.append(env_type)
        return Simulator

    monkeypatch.setattr(env_worker_module, "get_env_cls", get_env_cls)
    mode_cfg = _hetero_mode(
        libero=_env("libero", 8),
        maniskill=_env(
            "maniskill", 8, override_cfgs=[{"scene": "first"}, {"scene": "second"}]
        ),
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"env": {}})
    worker.stage_num = 2
    worker._worker_info = None
    worker.use_reward_model = False
    worker.use_realworld_reward = False
    # The last of 4 workers is the second of the 2 hosting ManiSkill.
    worker.train_assignment = EnvLayout(mode_cfg, 4, "env.train").assign(3)
    worker.eval_assignment = None

    worker.update_env_cfg()
    worker._setup_env_and_wrappers(
        worker.train_assignment, num_envs_per_stage=2, insert_delay=False
    )

    assert requested == ["maniskill"]
    assert created == [("maniskill", 2, 2, 4), ("maniskill", 2, 3, 4)]
    assert mode_cfg.envs.maniskill.override_cfg.scene == "second"


def test_config_gives_a_heterogeneous_mode_the_loop_fields_rollout_reads():
    mode_cfg = _hetero_mode(libero=_env("libero", 8), maniskill=_env("maniskill", 8))
    OmegaConf.set_struct(mode_cfg, True)

    resolve_env_mode_cfg(mode_cfg, "env.train", env_world_size=2)

    assert mode_cfg.total_num_envs == 16
    assert mode_cfg.max_steps_per_rollout_epoch == 80
    assert mode_cfg.max_episode_steps == 80
    assert mode_cfg.auto_reset is False


def test_spec_normalizes_masked_and_unmasked_trajectories_to_their_own_length():
    layout = EnvLayout(
        _hetero_mode(
            libero=_env(
                "libero", 4, max_episode_steps=300, max_steps_per_rollout_epoch=320
            ),
            maniskill=_env("maniskill", 4, ignore_terminations=True),
        ),
        world_size=2,
        path="env.train",
    )
    libero, maniskill = layout.spec_of_rank(0), layout.spec_of_rank(1)

    assert (libero.name, libero.ranks) == ("libero", range(0, 1))
    assert (libero.has_loss_mask, libero.reference_length) == (True, 300)
    assert (maniskill.has_loss_mask, maniskill.reference_length) == (False, 80)
    assert libero.chunk_steps(num_action_chunks=8) == 40
    assert layout.spec("maniskill") is maniskill


@pytest.mark.parametrize(
    ("env_weighting", "expected"),
    [
        # 4 LIBERO trajectories of 320 steps and 8 ManiSkill ones of 80 steps.
        ("trajectory", {"libero": 160.0, "maniskill": 160.0}),
        ("sample", {"libero": 320.0, "maniskill": 80.0}),
    ],
)
def test_env_weighting_picks_one_shared_or_one_own_length_per_env(
    env_weighting, expected
):
    layout = EnvLayout(
        _hetero_mode(
            libero=_env(
                "libero", 4, max_episode_steps=320, max_steps_per_rollout_epoch=320
            ),
            maniskill=_env("maniskill", 4, rollout_epoch=2),
        ),
        world_size=2,
        path="env.train",
    )

    assert layout.reference_lengths(env_weighting) == expected


def test_one_env_keeps_its_own_length_under_either_weighting():
    layout = EnvLayout(OmegaConf.create(_env("libero", 8)), world_size=2, path="p")

    assert layout.reference_lengths("trajectory") == {None: 80.0}
    assert layout.reference_lengths("sample") == {None: 80.0}
    with pytest.raises(ValueError, match="'trajectory' or 'sample', got 'episode'"):
        layout.reference_lengths("episode")


def test_each_env_sees_the_algorithm_with_its_own_group_size_and_overrides():
    layout = EnvLayout(
        _hetero_mode(
            libero=_env("libero", 8, group_size=8),
            maniskill=_env("maniskill", 8, group_size=4),
        ),
        world_size=2,
        path="env.train",
    )
    algorithm_cfg = OmegaConf.create(
        {
            "group_size": 2,
            "gamma": 1.0,
            "envs": {"maniskill": {"gamma": 0.99, "loss_weight": 2.0}},
        }
    )

    libero = layout.spec("libero").algorithm_cfg(algorithm_cfg)
    maniskill = layout.spec("maniskill").algorithm_cfg(algorithm_cfg)

    assert (libero.group_size, libero.gamma, libero.get("loss_weight")) == (
        8,
        1.0,
        None,
    )
    assert (maniskill.group_size, maniskill.gamma, maniskill.loss_weight) == (
        4,
        0.99,
        2.0,
    )
    assert "envs" not in maniskill


def _training_cfg(algorithm=None, runner=None, **envs):
    return OmegaConf.create(
        {
            "env": {"train": {"envs": envs}},
            "algorithm": {"loss_type": "actor_critic", **(algorithm or {})},
            "runner": runner or {},
        }
    )


@pytest.mark.parametrize(
    ("algorithm", "runner", "match"),
    [
        ({"envs": {"metaworld": {}}}, None, r"must name an env under env.train.envs"),
        (
            {"envs": {"libero": {"group_size": 4}}},
            None,
            r"may set \['loss_weight'.*got \['group_size'\]",
        ),
        ({"adv_norm_scope": "rank"}, None, r"must be 'env' or 'all', got 'rank'"),
        (
            {"adv_type": "grpo", "adv_norm_scope": "all"},
            None,
            r"adv_norm_scope='all' sets the scope .* got adv_type='grpo'",
        ),
        (
            {"adv_type": "gae", "adv_norm_scope": "all", "normalize_advantages": False},
            None,
            r"normalize_advantages=False",
        ),
        (
            {"adv_type": "raw", "adv_norm_scope": "all"},
            None,
            r"normalize_advantages=False",
        ),
        (
            {"adv_type": "gae_dynamic", "adv_norm_scope": "all"},
            None,
            r"got adv_type='gae_dynamic'",
        ),
        (
            {"adv_type": "grpo", "group_size": 8},
            None,
            r"so env.train.envs.libero needs group_size > 1, got 1",
        ),
        (
            {"loss_type": "embodied_nft"},
            None,
            r"heterogeneous NFT needs algorithm.env_weighting='sample'",
        ),
        (
            {"envs": {"libero": {"loss_weight": 0.0}}},
            None,
            r"Expected a positive loss_weight for env.train.envs.libero, got 0.0",
        ),
        (
            {"envs": {"maniskill": {"loss_weight": -0.5}}},
            None,
            r"Expected a positive loss_weight for env.train.envs.maniskill, got -0.5",
        ),
        ({}, {"use_training_pipeline": True}, r"runner.use_training_pipeline"),
    ],
)
def test_config_rejects_per_env_settings_the_run_cannot_honor(algorithm, runner, match):
    cfg = _training_cfg(
        algorithm, runner, libero=_env("libero", 8), maniskill=_env("maniskill", 8)
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=match):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)


@pytest.mark.parametrize(("maniskill_steps", "accepted"), [(80, True), (84, False)])
def test_config_checks_the_horizon_of_each_eval_env(maniskill_steps, accepted):
    # The envs' horizons differ, so env.eval itself carries none.
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "envs": {
                        "libero": _env("libero", 8, max_steps_per_rollout_epoch=320),
                        "maniskill": _env(
                            "maniskill", 8, max_steps_per_rollout_epoch=maniskill_steps
                        ),
                    }
                }
            }
        }
    )

    if accepted:
        assert validate_eval_env_cfg(cfg, 2, 1, num_action_chunks=8) == 8
        return
    with pytest.raises(
        AssertionError, match=r"env.eval.envs.maniskill.max_steps_per_rollout_epoch"
    ):
        validate_eval_env_cfg(cfg, 2, 1, num_action_chunks=8)


def test_config_keeps_reward_models_to_one_env_configuration():
    cfg = _training_cfg(libero=_env("libero", 8), maniskill=_env("maniskill", 8))
    cfg.reward = {"use_reward_model": True}
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=r"reward.use_reward_model"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)


def test_config_runs_listed_envs_only_on_the_hf_rollout_backend():
    # The SGLang eval worker reads one loop per mode, as on a homogeneous run.
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False, "task_type": "embodied"},
            "actor": {"model": {"model_type": "openvla_oft"}},
            "rollout": {"rollout_backend": "sglang"},
            "algorithm": {},
            "env": {
                "eval": {
                    "envs": {
                        "libero": _env("libero", 8),
                        "maniskill": _env("maniskill", 8),
                    }
                }
            },
        }
    )

    with pytest.raises(
        AssertionError, match=r"only rollout_backend='huggingface' runs listed envs"
    ):
        validate_embodied_cfg(cfg)


def test_config_needs_each_env_split_evenly_across_actors_in_whole_groups():
    # One worker of 8 ManiSkill envs feeds 4 actors 2 envs each, which splits
    # its groups of 4.
    cfg = _training_cfg(
        libero=_env("libero", 8), maniskill=_env("maniskill", 8, group_size=4)
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)
    with pytest.raises(AssertionError, match=r"4 shards of whole groups of 4"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=4)


@pytest.mark.parametrize(
    ("model_type", "accepted"), [("openvla_oft", True), ("openpi", False)]
)
def test_config_gives_per_env_action_statistics_only_to_a_policy_that_reads_them(
    model_type, accepted
):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "envs": {
                        "libero": _env("libero", 8),
                        "maniskill": _env("maniskill", 8, unnorm_key="bridge_orig"),
                    }
                }
            }
        }
    )
    model_cfg = OmegaConf.create({"model_type": model_type})

    if accepted:
        validate_env_policy_fields(cfg, model_cfg)
        return
    with pytest.raises(
        AssertionError,
        match=r"maniskill.unnorm_key is read by model_type \['openvla_oft'\], "
        r"got 'openpi'",
    ):
        validate_env_policy_fields(cfg, model_cfg)


def test_config_keeps_smooth_intervention_to_a_homogeneous_mode():
    mode_cfg = _hetero_mode(
        real=_env("real", 1, smooth_intervene=True), maniskill=_env("maniskill", 1)
    )

    with pytest.raises(
        AssertionError, match="real.smooth_intervene requires a homogeneous"
    ):
        resolve_env_mode_cfg(mode_cfg, "env.train", env_world_size=2)


def test_policy_obs_spec_leaves_observations_alone_when_unset():
    obs = {"main_images": torch.zeros(1, 4, 4, 3, dtype=torch.uint8)}

    spec = PolicyObsSpec.from_cfg(OmegaConf.create(_env("libero", 1)))

    assert spec.apply(obs) is obs


def test_observations_of_different_simulators_conform_to_one_policy_batch():
    # LIBERO sends a 256x256 main and wrist view; this ManiSkill scene sends
    # one 480x640 view and a differently sized state.
    libero_obs = {
        "main_images": torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.zeros(2, 8),
        "task_descriptions": ["open the drawer", "open the drawer"],
    }
    maniskill_obs = {
        "main_images": torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8),
        "wrist_images": None,
        "extra_view_images": None,
        "states": torch.zeros(1, 7),
        "task_descriptions": ["put the carrot on the plate"],
    }
    spec = PolicyObsSpec.from_cfg(
        OmegaConf.create(
            {
                "policy_obs": {
                    "keys": ["main_images", "task_descriptions"],
                    "image_size": [224, 224],
                }
            }
        )
    )

    batch = PolicyInput.merge(
        [
            PolicyInput(obs=spec.apply(libero_obs)),
            PolicyInput(obs=spec.apply(maniskill_obs)),
        ]
    )

    assert batch.obs["main_images"].shape == (3, 224, 224, 3)
    assert batch.obs["main_images"].dtype == torch.uint8
    assert batch.obs["wrist_images"] is None
    assert batch.obs["states"] is None
    assert batch.obs["task_descriptions"] == [
        "open the drawer",
        "open the drawer",
        "put the carrot on the plate",
    ]
    assert libero_obs["wrist_images"] is not None


def test_policy_obs_spec_resizes_stacked_views_and_keeps_pixel_values():
    spec = PolicyObsSpec(image_size=[2, 2])
    obs = {
        "wrist_images": np.full((1, 3, 4, 4, 3), 200, dtype=np.uint8),
        "states": torch.ones(1, 7),
    }

    conformed = spec.apply(obs)

    assert conformed["wrist_images"].shape == (1, 3, 2, 2, 3)
    assert torch.equal(
        conformed["wrist_images"], torch.full((1, 3, 2, 2, 3), 200, dtype=torch.uint8)
    )
    assert conformed["states"] is obs["states"]


def test_policy_obs_spec_passes_control_entries_through_and_conforms_history():
    active_mask = torch.tensor([True, False])
    frame_mask = torch.ones(2, 3, dtype=torch.bool)
    history = {
        "main_images": torch.zeros(2, 3, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 3, 4, 4, 3, dtype=torch.uint8),
        "frame_mask": frame_mask,
    }
    obs = {
        "main_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 8),
        "active_mask": active_mask,
        "history": history,
    }

    conformed = PolicyObsSpec(keys=["main_images"], image_size=[2, 2]).apply(obs)

    assert conformed["states"] is None
    assert conformed["active_mask"] is active_mask
    assert conformed["history"]["main_images"].shape == (2, 3, 2, 2, 3)
    assert conformed["history"]["wrist_images"] is None
    assert conformed["history"]["frame_mask"] is frame_mask
    assert history["main_images"].shape == (2, 3, 4, 4, 3)


@pytest.mark.parametrize("only_eval", [True, False])
def test_psi0_validation_checks_each_listed_env(only_eval):
    from rlinf.config import _validate_psi0_eval_cfg, _validate_psi0_train_cfg

    mode = "eval" if only_eval else "train"
    cfg = OmegaConf.create(
        {
            "runner": {
                "only_eval": only_eval,
                "val_check_interval": -1,
                "rtc": {"enabled": False},
                "enable_decoupled_mode": False,
            },
            "rollout": {
                "rollout_backend": "huggingface",
                "collect_prev_infos": not only_eval,
                "recompute_logprobs": False,
            },
            "actor": {
                "model": {
                    "num_action_chunks": 24,
                    "action_dim": 36,
                    "add_value_head": True,
                    "psi0": {
                        "plan_horizon": 30,
                        "overlap_horizon": 6,
                        "sampler": (
                            "deterministic_euler"
                            if only_eval
                            else "stochastic_transition"
                        ),
                        "rtc": {"enabled": True},
                    },
                },
                "fsdp_config": {"use_orig_params": True},
            },
            "algorithm": {"adv_type": "gae", "loss_type": "actor_critic"},
            "critic": {"use_critic_model": False},
            "env": {
                mode: {
                    "envs": {
                        "a": {"env_type": "simple", "total_num_envs": 1},
                        "b": {"env_type": "simple", "total_num_envs": 1},
                    }
                }
            },
        }
    )
    OmegaConf.set_struct(cfg, True)
    validator = _validate_psi0_eval_cfg if only_eval else _validate_psi0_train_cfg

    validator(cfg, cfg.actor.model, only_eval)

    cfg.env[mode].envs.b.env_type = "libero"
    with pytest.raises(AssertionError, match=r"env_type='simple'"):
        validator(cfg, cfg.actor.model, only_eval)


def test_policy_obs_spec_rejects_keys_the_policy_batch_does_not_carry():
    with pytest.raises(ValueError, match=r"unknown keys \['main_image'\]"):
        PolicyObsSpec(keys=["main_image"])


def test_policy_obs_spec_aligns_missing_camera_keys_across_eval_obs():
    from rlinf.data.schema.embodied_types import batch_format

    spec = PolicyObsSpec(keys=["main_images", "task_descriptions"], image_size=[8, 8])
    libero_obs = {
        "main_images": torch.zeros(2, 16, 16, 3),
        "wrist_images": torch.zeros(2, 16, 16, 3),
        "states": torch.zeros(2, 7),
        "task_descriptions": ["pick", "place"],
    }
    maniskill_obs = {
        "main_images": torch.zeros(2, 12, 20, 3),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["put", "stack"],
    }
    assert batch_format(spec.apply(libero_obs)) == batch_format(
        spec.apply(maniskill_obs)
    )


def test_validate_eval_env_cfg_defaults_missing_group_size():
    eval_cfg = _env("d4rl", 4)
    eval_cfg.pop("group_size")
    cfg = OmegaConf.create({"env": {"eval": eval_cfg}})
    OmegaConf.set_struct(cfg, True)
    assert (
        validate_eval_env_cfg(cfg, env_world_size=2, stage_num=1, num_action_chunks=8)
        == 2
    )


def test_config_needs_grpo_envs_to_stop_trajectories_at_their_first_done():
    cfg = _training_cfg(
        {"adv_type": "grpo"},
        libero=_env("libero", 8, group_size=2),
        maniskill=_env("maniskill", 8, group_size=2, ignore_terminations=True),
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=r"env.train.envs.maniskill needs"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)
