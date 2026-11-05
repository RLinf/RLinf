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

"""Real-entry SuperMimic e2e tests and small numerical regressions.

Run with PYTHONPATH=. MUJOCO_GL=disable python -m pytest <this file> -q -rs.
Select -k 'not end_to_end' for numerical checks without Ray or simulation.
Training uses the production recipe with small overrides and a generated
standing clip. Two GPUs enable the NCCL case; neither case measures quality.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
from datetime import timedelta
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
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import differentiable_fk, failure_signals
from rlinf.models.embodiment.supermimic import RunningNorm
from rlinf.workers.supermimic.checkpoint import verify_checkpoint
from rlinf.workers.supermimic.distributed import DataParallel
from rlinf.workers.supermimic.eval_rollout import (
    checked_eval_states,
    checked_policy_action,
)

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


def _distributed_oracle(rank, out, sizes):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{out}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        dp = DataParallel(2, rank, CPU)
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(sum(sizes), 3, generator=generator)
        y = torch.randn(sum(sizes), 2, generator=generator)
        torch.manual_seed(100 + rank)
        model = torch.nn.Linear(3, 2)
        dp.broadcast_module(model)
        start, count = sum(sizes[:rank]), sizes[rank]
        if count:
            (
                model(x[start : start + count]) - y[start : start + count]
            ).square().mean().backward()
        total, finite = dp.reduce_grads(model.parameters(), count)
        norm = RunningNorm(3)
        dp.update_norm(norm, x[start : start + count])
        torch.save(
            {
                "initial": model.state_dict(),
                "grads": [p.grad for p in model.parameters()],
                "total": total,
                "finite": finite,
                "norm": norm.state_dict(),
            },
            Path(out) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
@pytest.mark.parametrize("sizes", [(3, 5), (4, 0)])
def test_two_rank_gradients_and_norm_equal_one_merged_batch(tmp_path, sizes):
    torch.multiprocessing.spawn(
        _distributed_oracle, args=(str(tmp_path), sizes), nprocs=2, join=True
    )
    ranks = [
        torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)
    ]
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(sum(sizes), 3, generator=generator)
    y = torch.randn(sum(sizes), 2, generator=generator)
    oracle = torch.nn.Linear(3, 2)
    oracle.load_state_dict(ranks[0]["initial"])
    (oracle(x) - y).square().mean().backward()
    for result in ranks:
        assert result["total"] == sum(sizes) and result["finite"]
        for actual, param in zip(result["grads"], oracle.parameters(), strict=True):
            torch.testing.assert_close(actual, param.grad)
        torch.testing.assert_close(result["norm"]["mean"], x.mean(0))
        torch.testing.assert_close(result["norm"]["var"], x.var(0, unbiased=False))
        torch.testing.assert_close(
            result["norm"]["count"], torch.tensor(float(sum(sizes)))
        )


def _entry_env():
    env = dict(os.environ)
    env.update(
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


def _run_entry(work, script, arguments):
    env = _entry_env()
    work.mkdir(parents=True)
    log = work / "entry.log"
    with log.open("w") as output:
        proc = subprocess.Popen(
            [sys.executable, str(script), *arguments],
            cwd=work,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=600)
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
            "state": pose.repeat(64, 1),
        },
        directory / "clip_000.pt",
    )
    return directory


@needs_mjlab
@pytest.mark.placement
@pytest.mark.parametrize(
    "world,wm_only",
    [(1, 0), (1, 1), (2, 0)],
    ids=["single-gpu", "single-gpu-wm-warmup", "two-gpu-nccl"],
)
def test_end_to_end_train_resume_and_evaluate(tmp_path, world, wm_only):
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("ray")
    if torch.cuda.device_count() < world:
        pytest.skip(f"this case needs {world} visible CUDA/HIP device(s)")
    if world > 1 and not dist.is_nccl_available():
        pytest.skip("NCCL/RCCL is unavailable")
    clips = _standing_clips(tmp_path / "clips")
    velocity = world > 1
    common = [
        f"env.motion_path={clips}",
        "env.num_clips=1",
        "env.motion_on_cpu=true",
        f"env.num_envs={4 * world}",
        "env.replay_length=32",
        "algorithm.warmup_sim_steps=16",
        f"algorithm.wm_only_updates={wm_only}",
        "algorithm.updates_per_env_step=2.0",
        "algorithm.wm_batches=1",
        f"algorithm.wm_window_batch={8 * world}",
        f"algorithm.imagine_batch={8 * world}",
        "algorithm.critic_epochs=1",
        "algorithm.critic_minibatches=1",
        f"algorithm.velocity_tracking={str(velocity).lower()}",
        f"algorithm.termination_head={'learned' if velocity else 'analytic'}",
        "algorithm.param_check_interval=1",
        "algorithm.matmul=tf32",
    ]
    if velocity:
        common += [
            "algorithm.obs_dim=93",
            "algorithm.goal_frame_dim=68",
            "algorithm.goal_encoding=paper",
        ]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = (
        [device.strip() for device in devices.split(",")]
        if devices
        else [str(i) for i in range(world)]
    )
    placement = ",".join(gpu_ids[:world])
    train = REPO / "examples/embodiment/train_supermimic.py"
    train_args = [
        "--config-name",
        "supermimic_lafan1_zita",
        *common,
        f"cluster.component_placement.supermimic='{placement}'",
        "cluster.object_store_memory_mb=128",
        f"runner.logger.log_path={tmp_path / 'runs'}",
        "runner.logger.experiment_name=e2e",
        "runner.save_interval=-1",
        "runner.val_check_interval=10",
    ]
    trained = _run_entry(
        tmp_path / "train", train, [*train_args, "algorithm.total_updates=3"]
    )
    assert trained.count("supermimic eval ") == 1
    root = tmp_path / "runs/e2e"
    first = root / "global_step_3"
    meta = verify_checkpoint(first)
    assert meta["world_size"] == world and meta["global_update"] == 3
    saved = torch.load(first / "supermimic.pt", map_location="cpu", weights_only=False)
    assert saved["wm_optimizer_steps"] == 3
    assert saved["actor_optimizer_steps"] == 3 - wm_only
    assert saved["critic_optimizer_steps"] == (3 - wm_only) * (2 if velocity else 1)
    before_steps = saved["sim_vector_steps"]
    before_weight = saved["models"]["dynamics.net.6.weight"].clone()
    del saved
    output = _run_entry(
        tmp_path / "resume",
        train,
        [*train_args, "algorithm.total_updates=5", "runner.resume_dir=auto"],
    )
    assert "Resuming SuperMimic" in output
    assert output.count("supermimic eval ") == 1
    if world > 1:
        divergence = re.findall(
            r"'train/param_divergence': ([0-9.e+-]+)", trained + output
        )
        assert divergence and all(float(value) == 0 for value in divergence)
    final = root / "global_step_5"
    assert verify_checkpoint(final)["global_update"] == 5
    assert sorted(path.name for path in root.glob("global_step_*")) == [
        "global_step_3",
        "global_step_5",
    ]
    saved = torch.load(final / "supermimic.pt", map_location="cpu", weights_only=False)
    assert saved["global_update"] == saved["wm_optimizer_steps"] == 5
    assert saved["actor_optimizer_steps"] == 5 - wm_only
    assert saved["critic_optimizer_steps"] == (5 - wm_only) * (2 if velocity else 1)
    assert saved["sim_vector_steps"] >= before_steps + 17
    assert not torch.equal(before_weight, saved["models"]["dynamics.net.6.weight"])
    assert all(torch.isfinite(value).all() for value in saved["models"].values())
    del saved
    for rank in range(world):
        assert (final / f"rank_{rank}.pt").is_file()
    out = tmp_path / "evaluation"
    _run_entry(
        tmp_path / "eval",
        REPO / "evaluations/eval_supermimic.py",
        [
            "--config-name",
            "supermimic_lafan1_zita_eval",
            *common,
            f"env.num_envs={2 if velocity else 1}",
            f"eval.checkpoint={final}",
            f"eval.output_dir={out}",
            f"eval.velocity={'smoke' if velocity else 'none'}",
            "eval.velocity_smoke_seconds=0.08",
        ],
    )
    result = json.loads((out / "summary.json").read_text())
    assert result["meta"]["global_step"] == 5
    assert result["meta"]["checkpoint_world_size"] == world
    for key in ("eval/normalized_return", "eval/fail_rate", "eval/clip_end_rate"):
        assert math.isfinite(result["metrics"][key])
    with (out / "per_clip.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
    if velocity:
        assert (out / "velocity_smoke.csv").is_file()


@needs_mjlab
def test_end_to_end_velocity_g2_repeats_for_seed(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("a CUDA/HIP device is required for MuJoCo Warp")
    pytest.importorskip("mujoco_warp")
    from omegaconf import OmegaConf

    from rlinf.algorithms.supermimic.validate import validate_supermimic_cfg
    from rlinf.workers.supermimic.evaluator import SuperMimicEvaluator

    cfg = OmegaConf.load(
        REPO / "examples/embodiment/config/supermimic_lafan1_zita.yaml"
    )
    cfg.env.update(
        {
            "num_envs": 2,
            "num_clips": 1,
            "motion_on_cpu": True,
            "motion_path": str(_standing_clips(tmp_path / "clips")),
        }
    )
    cfg.algorithm.update(
        {
            "velocity_tracking": True,
            "obs_dim": 93,
            "goal_frame_dim": 68,
            "goal_encoding": "paper",
            "matmul": "tf32",
        }
    )
    cfg = validate_supermimic_cfg(cfg)
    evaluator = SuperMimicEvaluator(cfg, torch.device("cuda:0"))
    commands = (torch.tensor([[0.2, 0.0, 0.0]]), ["forward"], ["vx_vy"])
    rows = []
    for previous_seed in (111, 999):
        torch.manual_seed(previous_seed)
        _, table, names = evaluator.evaluate_velocity_g2(
            seed=17,
            trials=2,
            seconds=0.04,
            steady_seconds=0.02,
            batch=1,
            commands=commands,
        )
        assert names == ["forward"]
        rows.append(table)
    for key in rows[0]:
        torch.testing.assert_close(
            rows[0][key], rows[1][key], atol=2e-5, rtol=2e-5, equal_nan=True
        )


_RANK_FAILURE_DRIVER = '''
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.runners.supermimic_runner import SuperMimicRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.workers.supermimic.distributed import DataParallel


class FailingRank(Worker):
    """Rank 1 fails in the first iteration while rank 0 waits in an all-reduce."""

    def __init__(self, mode, marker_path):
        super().__init__()
        self.mode = mode
        self.marker_path = marker_path

    def init_worker(self):
        self.dp = DataParallel.initialize(
            self._world_size, self._rank, torch.device("cpu"),
            timeout=Cluster.get_collective_timeout(),
        )

    def warmup(self):
        pass

    def collect(self):
        if self._rank == 1:
            Path(self.marker_path).write_text(self.mode)
            if self.mode == "raise":
                raise RuntimeError("rank 1 failed on purpose")
            os._exit(3)
        self.dp.all_reduce([1.0])
        return {"train/global_update": 1.0}, 0

    def shutdown(self):
        self.dp.destroy()

    def save_checkpoint(self, directory):
        return None

    def evaluate(self):
        return {}


if __name__ == "__main__":
    mode, log_path, marker_path = sys.argv[1:]
    cluster = Cluster(cluster_cfg=OmegaConf.create({"num_nodes": 1, "object_store_memory_mb": 128, "component_placement": {}}))
    group = FailingRank.create_group(mode, marker_path).launch(
        cluster, name="supermimic_rank_failure", placement_strategy=NodePlacementStrategy([0, 0])
    )
    cfg = OmegaConf.create({
        "runner": {
            "logger": {"log_path": log_path, "project_name": "rlinf",
                       "experiment_name": "rank_failure", "logger_backends": []},
            "save_interval": -1, "val_check_interval": -1, "only_eval": False,
        },
        "algorithm": {"total_updates": 3},
    })
    runner = SuperMimicRunner(cfg, group)
    runner.init_workers()
    runner.run()
    print("RUN RETURNED")
'''


@pytest.mark.placement
@pytest.mark.parametrize("mode", ["raise", "exit"])
def test_end_to_end_one_failed_rank_ends_the_run(tmp_path, mode):
    """The other rank is blocked in a collective with the default 180 min timeout.

    RLinf's worker group signals the driver as soon as one rank's call fails,
    and the driver kills the group, so the run ends in seconds.
    """
    import time

    pytest.importorskip("ray")
    script = tmp_path / "driver.py"
    script.write_text(_RANK_FAILURE_DRIVER)
    env = _entry_env()
    env.pop("RLINF_TIMEOUT", None)
    marker = tmp_path / "failed_rank_started"
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script), mode, str(tmp_path / "logs"), str(marker)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
    )
    elapsed = time.monotonic() - start
    output = proc.stdout + proc.stderr
    assert marker.is_file(), output[-5000:]
    assert marker.read_text() == mode
    assert proc.returncode != 0, output[-3000:]
    assert "RUN RETURNED" not in output
    assert elapsed < 180, f"the failed run took {elapsed:.0f} s to end"
    if mode == "raise":
        assert "rank 1 failed on purpose" in output
