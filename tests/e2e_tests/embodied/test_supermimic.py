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

"""SuperMimic smoke runs through the RLinf entries, plus small numerical checks.

Run with PYTHONPATH=. MUJOCO_GL=disable python -m pytest <this file> -q -rs.
Select -k 'not end_to_end' for the numerical checks, which need neither Ray
nor a GPU. The smoke trains the goal730 recipe with small overrides on a
generated standing clip, resumes it and evaluates the checkpoint through
``evaluations/eval_embodied_agent.py``. Two GPUs add a velocity-tracking run
over NCCL. None of the runs measures quality.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from rlinf.algorithms.supermimic.objectives import actor_objective, gae_targets
from rlinf.algorithms.supermimic.reset_sampler import RETURN_EMA, BinResetSampler
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    JOINT_POS,
)
from rlinf.envs.sim.g1_mimic.features import default_state
from rlinf.envs.sim.g1_mimic.g1_mimic_eval_env import (
    checked_eval_states,
    checked_policy_action,
)
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import differentiable_fk, failure_signals

REPO = Path(__file__).resolve().parents[3]
CPU = torch.device("cpu")
needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None, reason="mjlab is not installed"
)


def test_actor_objective_value_and_gradient():
    rewards = torch.tensor([[2.0, 3.0]], requires_grad=True)
    failure = torch.tensor([[0.25, 0.5]], requires_grad=True)
    bootstrap = torch.tensor([4.0], requires_grad=True)
    result = actor_objective(rewards, failure, bootstrap, gamma=0.5)
    # 2 + .5*.75*3 + .5**2*.75*.5*4.
    torch.testing.assert_close(result, torch.tensor(3.5))
    gradients = torch.autograd.grad(result, (rewards, failure, bootstrap))
    for actual, expected in zip(
        gradients,
        (
            torch.tensor([[1.0, 0.375]]),
            torch.tensor([[-2.0, -0.75]]),
            torch.tensor([0.09375]),
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_gae_matches_discounted_returns_at_lambda_one():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    failure = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[99.0, -4.0, 17.0, 4.0]])
    actual = gae_targets(rewards, failure, values, gamma=0.5, lam=1.0)
    torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0, 5.0]]))


@pytest.mark.parametrize("gamma", [0.01, 0.8, 0.99])
def test_reset_returns_match_backward_recurrence(gamma):
    horizon = 500
    lengths = [horizon, 7, 1]
    sampler = BinResetSampler(torch.tensor([horizon]), 3, 10.0, gamma, horizon, CPU)
    rewards = torch.rand(3, horizon, generator=torch.Generator().manual_seed(17))
    for step in range(horizon):
        active = torch.tensor([step < length for length in lengths])
        sampler.log_step(
            rewards[:, step],
            torch.zeros(3, dtype=torch.long),
            torch.full((3,), step),
            active,
        )
    kinds = torch.tensor([EPISODE_TIMEOUT, EPISODE_FAILURE, EPISODE_CLIP_END])
    # Failure and clip-end do not bootstrap, including non-finite predictions.
    bootstrap = torch.tensor([3.0, float("nan"), float("inf")])
    sums = torch.zeros(sampler.num_bins, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for row, length in enumerate(lengths):
        value = 3.0 if row == 0 else 0.0
        for step in range(length - 1, -1, -1):
            value = float(rewards[row, step]) + gamma * value
            bin_id = step // sampler.bin_frames
            sums[bin_id] += value
            counts[bin_id] += 1
    sampler.finish(torch.arange(3), kinds, bootstrap)
    expected = torch.where(
        counts > 0,
        (1 - RETURN_EMA) * 10 + RETURN_EMA * sums / counts.clamp_min(1),
        10.0,
    )
    torch.testing.assert_close(
        sampler.bin_return.double(), expected, atol=2e-5, rtol=2e-6
    )
    assert torch.isfinite(sampler.mixture(update_memory=False)).all()
    assert sampler.sample(4)[0].numel() == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_timeout_bootstrap_is_rejected(value):
    sampler = BinResetSampler(torch.tensor([50]), 1, 10.0, 0.99, 2, CPU)
    sampler.log_step(torch.tensor([1.0]), torch.tensor([0]), torch.tensor([0]))
    before = sampler.bin_return.clone()
    with pytest.raises(RuntimeError, match="non-finite timeout bootstrap"):
        sampler.finish(
            torch.tensor([0]), torch.tensor([EPISODE_TIMEOUT]), torch.tensor([value])
        )
    torch.testing.assert_close(sampler.bin_return, before)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_evaluation_rejects_active_nonfinite_rows_and_masks_finished_rows(value):
    active = torch.tensor([True, False])
    action = torch.ones(2, ACTION_DIM)
    action[1] = value
    torch.testing.assert_close(
        checked_policy_action(action, active),
        torch.cat((torch.ones(1, ACTION_DIM), torch.zeros(1, ACTION_DIM))),
    )
    action[0, 0] = value
    with pytest.raises(RuntimeError, match=r"active evaluation rows \[0\]"):
        checked_policy_action(action, active)
    for field in ("next_state", "ref_state"):
        state = default_state(2, "cpu", torch.float32)
        reference = state.clone()
        state[1] = reference[1] = value
        assert all(
            torch.isfinite(part).all()
            for part in checked_eval_states(state, reference, active)
        )
        (state if field == "next_state" else reference)[0, 0] = value
        with pytest.raises(RuntimeError, match=r"active rows \[0\]"):
            checked_eval_states(state, reference, active)


@needs_mjlab
@pytest.mark.parametrize("joint", ["waist_roll_joint", "waist_pitch_joint"])
def test_torso_tilt_and_gradient_match_joint_rotation(joint):
    state = default_state(2, "cpu", torch.float32)
    reference = state.clone()
    index = JOINT_POS.start + G1_JOINT_NAMES.index(joint)
    angles, ref_angles = torch.tensor([1.4, 0.4]), torch.tensor([0.0, 0.2])
    state[:, index], reference[:, index] = angles, ref_angles
    state.requires_grad_()
    _, torso_tilt, torso_failed = failure_signals(state, reference, "torso")
    _, pelvis_tilt, pelvis_failed = failure_signals(state, reference, "pelvis")
    torch.testing.assert_close(torso_tilt, ref_angles.cos() - angles.cos())
    torch.testing.assert_close(pelvis_tilt, torch.zeros(2))
    assert bool(torso_failed[0]) and not bool(pelvis_failed.any())
    gradient = torch.autograd.grad(torso_tilt.sum(), state)[0]
    torch.testing.assert_close(gradient[:, index], angles.sin())


@needs_mjlab
@pytest.mark.parametrize(
    "storage,query",
    [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:0", "cpu")],
)
def test_reference_fk_matches_online_fk(storage, query):
    if "cuda" in storage + query and not torch.cuda.is_available():
        pytest.skip("a CUDA/HIP device is required for cross-device queries")
    clips = [default_state(n, storage, torch.float32) for n in (3, 4)]
    for index, clip in enumerate(clips):
        clip[:, 0] = index + 0.1 * torch.arange(len(clip), device=storage)
        clip[:, JOINT_POS.start] = 0.05 * torch.arange(len(clip), device=storage)
    motion = MotionLibrary(clips)
    ids, times = (
        torch.tensor([1, 0, 1], device=query),
        torch.tensor([100, -1, 2], device=query),
    )
    result = motion.kinematics_at(ids, times)
    expected = differentiable_fk(motion.state_at(ids, times))
    assert set(motion._kin_cache) == {motion.frames.device}
    assert all(
        part.device == motion.frames.device
        for part in motion._kin_cache[motion.frames.device]
    )
    for actual, wanted in zip(result, expected, strict=True):
        assert actual.device == ids.device
        torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-5)


# ------------------------------------------------------------------ smoke runs

_CLIP_FRAMES = 64


def _entry_env(world):
    env = dict(os.environ)
    devices = env.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = (
        [device.strip() for device in devices.split(",")]
        if devices
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(gpu_ids[:world]),
        PYTHONPATH=f"{REPO}{os.pathsep}{env.get('PYTHONPATH', '')}",
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        MUJOCO_GL="disable",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RAY_ADDRESS="local",
        RLINF_NODE_RANK="0",
        RAY_prestart_worker_first_driver="0",
        RAY_USAGE_STATS_ENABLED="0",
        RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=str(128 * 1024**2),
        RAY_OVERRIDE_RESOURCES=json.dumps({"CPU": 4}),
    )
    env.setdefault("SUPERMIMIC_SCAN_EAGER", "1")
    env.setdefault("SUPERMIMIC_FK_EAGER", "1")
    return env


def _run_entry(work, world, script, arguments):
    work.mkdir(parents=True)
    log = work / "entry.log"
    with log.open("w") as output:
        proc = subprocess.Popen(
            [sys.executable, str(script), *arguments],
            cwd=work,
            env=_entry_env(world),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=900)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    text = log.read_text()
    assert code == 0, f"{script.name} exited with {code}; {log}\n{text[-8000:]}"
    return text


def _standing_clips(directory: Path) -> Path:
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

    pose = asset_default_state()
    directory.mkdir()
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": pose[JOINT_POS],
            "state": pose.repeat(_CLIP_FRAMES, 1),
        },
        directory / "clip_000.pt",
    )
    return directory


def smoke_overrides(clips, world: int, velocity: bool) -> tuple[list, list]:
    """Overrides shared by training and evaluation, and the training-only ones."""
    shared = [
        f"env.train.motion_path={clips}",
        f"env.eval.motion_path={clips}",
        "env.train.num_clips=1",
        "env.eval.num_clips=1",
        "env.train.motion_on_cpu=true",
        "env.eval.motion_on_cpu=true",
        f"env.eval.max_steps_per_rollout_epoch={_CLIP_FRAMES}",
        "runner.logger.logger_backends=null",
        "++cluster.object_store_memory_mb=128",
    ]
    if velocity:
        shared += [
            "actor.model.goal_encoding=paper",
            "actor.model.obs_dim=93",
            "actor.model.critic_features=root4",
            "actor.model.input_norm=none",
            "actor.model.wm_state_loss=paper",
            "actor.model.velocity_tracking=true",
            "algorithm.termination_head=learned",
        ]
    train = [
        f"env.train.total_num_envs={4 * world}",
        f"env.eval.total_num_envs={world}",
        "algorithm.replay_length=32",
        "algorithm.warmup_sim_steps=16",
        "algorithm.updates_per_env_step=2.0",
        "algorithm.wm_only_updates=1",
        "algorithm.wm_batches=1",
        f"algorithm.wm_window_batch={8 * world}",
        f"algorithm.imagine_batch={8 * world}",
        "algorithm.critic_epochs=1",
        "algorithm.critic_minibatches=1",
        "runner.save_interval=3",
        "runner.val_check_interval=3",
        "runner.eval_export.enabled=true",
        "runner.logger.experiment_name=e2e",
    ]
    return shared, train


def _counters(step_dir: Path) -> dict:
    return json.loads((step_dir / "actor" / "supermimic_counters.json").read_text())


def _weights(step_dir: Path) -> dict:
    path = step_dir / "actor" / "model_state_dict" / "full_weights.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


@needs_mjlab
@pytest.mark.placement
@pytest.mark.parametrize("world", [1, 2], ids=["single-gpu", "two-gpu-nccl-velocity"])
def test_end_to_end_train_resume_and_evaluate(tmp_path, world):
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("ray")
    if torch.cuda.device_count() < world:
        pytest.skip(f"this case needs {world} visible CUDA/HIP device(s)")
    if world > 1 and not dist.is_nccl_available():
        pytest.skip("NCCL/RCCL is unavailable")
    velocity = world > 1
    clips = _standing_clips(tmp_path / "clips")
    shared, train = smoke_overrides(clips, world, velocity)
    runs = tmp_path / "runs"
    train_script = REPO / "examples/embodiment/train_embodied_agent.py"
    train_args = [
        "--config-path",
        str(REPO / "examples/embodiment/config"),
        "--config-name",
        "supermimic_lafan1_goal730",
        *shared,
        *train,
        f"runner.logger.log_path={runs}",
    ]
    critic_steps = 2 if velocity else 1

    _run_entry(
        tmp_path / "train", world, train_script, [*train_args, "runner.max_epochs=3"]
    )
    root = runs / "e2e"
    first = root / "checkpoints" / "global_step_3"
    for name in ("runner_state.json", "actor/supermimic_interface.json"):
        assert (first / name).is_file(), name
    assert sorted(path.name for path in (first / "env").iterdir()) == [
        f"env_rank_{rank}.pt" for rank in range(world)
    ]
    counters = _counters(first)
    # Update 0 fits only the world model (wm_only_updates=1).
    assert counters["wm_optimizer_steps"] == 3
    assert counters["actor_optimizer_steps"] == 2
    assert counters["critic_optimizer_steps"] == 2 * critic_steps
    before = _weights(first)
    summary = json.loads(
        (root / "eval_results" / "global_step_3" / "summary.json").read_text()
    )
    assert math.isfinite(summary["metrics"]["normalized_return"])

    output = _run_entry(
        tmp_path / "resume",
        world,
        train_script,
        [*train_args, "runner.max_epochs=5", "runner.resume_dir=auto"],
    )
    assert "Resuming from" in output and "global_step_3" in output
    final = root / "checkpoints" / "global_step_5"
    counters = _counters(final)
    assert counters["wm_optimizer_steps"] == 5
    assert counters["actor_optimizer_steps"] == 4
    assert counters["critic_optimizer_steps"] == 4 * critic_steps
    after = _weights(final)
    assert not torch.equal(
        before["dynamics.net.6.weight"], after["dynamics.net.6.weight"]
    )
    assert all(torch.isfinite(value).all() for value in after.values())

    out = tmp_path / "evaluation"
    tasks = "[tracking,velocity_smoke]" if velocity else "[tracking]"
    _run_entry(
        tmp_path / "eval",
        world,
        REPO / "evaluations/eval_embodied_agent.py",
        [
            "--config-path",
            str(REPO / "evaluations/supermimic"),
            "--config-name",
            "supermimic_lafan1_goal730_eval",
            *shared,
            f"runner.ckpt_path={final / 'actor/model_state_dict/full_weights.pt'}",
            f"runner.logger.log_path={tmp_path / 'eval_runs'}",
            f"runner.eval_export.output_dir={out}",
            f"env.eval.total_num_envs={8 if velocity else 1}",
            f"env.eval.eval_tasks={tasks}",
            "env.eval.velocity_smoke.seconds=0.08",
        ],
    )
    result = json.loads((out / "summary.json").read_text())
    assert result["meta"]["checkpoint"].endswith("full_weights.pt")
    for key in ("normalized_return", "fail_rate", "clip_end_rate"):
        assert math.isfinite(result["metrics"][key])
    with (out / "per_clip.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
    if velocity:
        with (out / "velocity_smoke.csv").open() as handle:
            assert len(list(csv.DictReader(handle))) == 7
