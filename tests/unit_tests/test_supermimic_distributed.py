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

"""SuperMimic data parallelism with two CPU processes on Gloo.

These check the reduction arithmetic and that ranks never desynchronize.
They do not exercise NCCL, GPU placement or multi-node networking.
"""

from __future__ import annotations

import importlib.util
import math
import os
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _supermimic_support import synthetic_clip_dir
from omegaconf import OmegaConf

from rlinf.workers.supermimic.distributed import DataParallel

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="the trainer reads G1 joint limits from mjlab",
)
needs_gloo = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with Gloo is not available",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _entry(rank: int, world: int, port: int, name: str, args: tuple) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dp = DataParallel.initialize(
        world, rank, torch.device("cpu"), timeout=timedelta(seconds=120)
    )
    try:
        globals()[name](dp, *args)
    finally:
        dp.destroy()


def _spawn(case, *args, world: int = 2) -> None:
    torch.multiprocessing.spawn(
        _entry,
        args=(world, _free_port(), case.__name__, args),
        nprocs=world,
        join=True,
    )


def _cfg(world: int, velocity: bool = False):
    return OmegaConf.create(
        {
            "algorithm": {
                "goal_anchor": False,
                "k": 8,
                "H": 4,
                "gamma": 0.99,
                "lam": 0.95,
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "velocity_tracking": velocity,
                "warmup_sim_steps": 12,
                "updates_per_env_step": 0.4,
                "wm_batches": 2,
                "wm_window_batch": 4,
                "imagine_batch": 4,
                "critic_epochs": 1,
                "critic_minibatches": 1,
                "bce_weight": 1.0,
                "action_noise_std": 0.1,
                "G_max": 10.0,
                "lr": 2e-4,
                "lr_decay_updates": 10,
                "action_dim": 29,
                "max_episode_seconds": 10.0,
                "termination_body": "pelvis",
                "seed": 0,
                "max_nonfinite_skips": 1,
                "param_check_interval": 1,
                "world_size": world,
            },
            "env": {
                "motion_format": "lafan1",
                "motion_path": synthetic_clip_dir(),
                "num_clips": 2,
                "backend": "analytic",
                "num_envs": 4,
                "replay_length": 32,
            },
        }
    )


def _rows(rank_rows: list[int], width: int, seed: int = 7) -> list[tuple]:
    """Fixed per-rank regression data; rank r gets ``rank_rows[r]`` rows."""
    gen = torch.Generator().manual_seed(seed)
    total = sum(rank_rows)
    x = torch.randn(total, width, generator=gen)
    y = torch.randn(total, 2, generator=gen)
    out, start = [], 0
    for n in rank_rows:
        out.append((x[start : start + n], y[start : start + n]))
        start += n
    return out


# ---- reduction arithmetic -------------------------------------------------


def _reduce_case(dp: DataParallel, out: str, rank_rows: list[int]) -> None:
    torch.manual_seed(100 + dp.rank)
    model = torch.nn.Linear(3, 2)
    dp.broadcast_module(model)
    x, y = _rows(rank_rows, 3)[dp.rank]
    if x.shape[0]:
        (model(x) - y).pow(2).mean().backward()
    total, finite = dp.reduce_grads(model.parameters(), x.shape[0])
    torch.save(
        {
            "init": {k: v.clone() for k, v in model.state_dict().items()},
            "grads": [p.grad.clone() for p in model.parameters()],
            "total": total,
            "finite": finite,
        },
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
@pytest.mark.parametrize("rank_rows", [[3, 5], [4, 0]])
def test_weighted_reduce_matches_one_merged_batch(tmp_path, rank_rows):
    _spawn(_reduce_case, str(tmp_path), rank_rows)
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")
    for a, b in zip(r0["grads"], r1["grads"]):
        assert torch.equal(a, b)
    for key in r0["init"]:
        assert torch.equal(r0["init"][key], r1["init"][key])
    assert r0["total"] == float(sum(rank_rows)) and r0["finite"]

    model = torch.nn.Linear(3, 2)
    model.load_state_dict(r0["init"])
    x = torch.cat([part[0] for part in _rows(rank_rows, 3)])
    y = torch.cat([part[1] for part in _rows(rank_rows, 3)])
    (model(x) - y).pow(2).mean().backward()
    for ours, ref in zip(r0["grads"], model.parameters()):
        assert torch.allclose(ours, ref.grad, atol=1e-6, rtol=1e-5)


def _all_empty_case(dp: DataParallel, out: str) -> None:
    model = torch.nn.Linear(3, 2)
    total, finite = dp.reduce_grads(model.parameters(), 0)
    torch.save(
        {
            "total": total,
            "finite": finite,
            "none": all(p.grad is None for p in model.parameters()),
        },
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
def test_no_rank_with_data_clears_gradients(tmp_path):
    _spawn(_all_empty_case, str(tmp_path))
    for rank in range(2):
        result = torch.load(tmp_path / f"rank{rank}.pt")
        assert result == {"total": 0.0, "finite": True, "none": True}


# ---- trainer: critic fit equals the merged batch ---------------------------


def _synthetic_rollout(trainer, rows: int, seed: int = 11):
    gen = torch.Generator().manual_seed(seed)
    H = trainer.H
    width = trainer.models.critic.net[0].in_features
    return (
        torch.randn(rows, H + 1, width, generator=gen),
        torch.randn(rows, H, generator=gen),
        torch.rand(rows, H, generator=gen),
    )


def _split(rollout, start: int, stop: int):
    return tuple(t[start:stop] for t in rollout)


def _plain_sgd(trainer):
    """Adam's first step is lr * g / (|g| + eps), which turns summation-order
    rounding on near-zero gradients into lr-sized differences. SGD with
    lr=1 leaves the reduced, clipped gradient itself in the weights."""
    return torch.optim.SGD(trainer.models.critic_params(), lr=1.0)


def _critic_case(dp: DataParallel, out: str, rank_rows: list[int]) -> None:
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(dp.world_size), device=torch.device("cpu"), dp=dp)
    trainer.opt_critic = _plain_sgd(trainer)
    init = {k: v.clone() for k, v in trainer.models.critic.state_dict().items()}
    full = _synthetic_rollout(trainer, sum(rank_rows))
    start = sum(rank_rows[: dp.rank])
    trainer.fit_critic(*_split(full, start, start + rank_rows[dp.rank]))
    torch.save(
        {
            "init": init,
            "after": trainer.models.critic.state_dict(),
            "steps": trainer.critic_optimizer_steps,
            "divergence": dp.param_divergence(trainer.models),
        },
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
@needs_mjlab
@pytest.mark.parametrize("rank_rows", [[4, 2], [3, 0]])
def test_critic_fit_on_two_ranks_matches_one_process(tmp_path, rank_rows):
    """Uneven and empty local batches give the merged-batch update."""
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    _spawn(_critic_case, str(tmp_path), rank_rows)
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")
    assert r0["divergence"] == 0.0 and r1["divergence"] == 0.0
    assert r0["steps"] == r1["steps"] == 1
    for key in r0["after"]:
        assert torch.equal(r0["after"][key], r1["after"][key])

    reference = SuperMimicTrainer(_cfg(1), device=torch.device("cpu"))
    reference.opt_critic = _plain_sgd(reference)
    reference.models.critic.load_state_dict(r0["init"])
    reference.fit_critic(*_synthetic_rollout(reference, sum(rank_rows)))
    for key, value in reference.models.critic.state_dict().items():
        assert torch.allclose(r0["after"][key], value, atol=1e-6, rtol=1e-5), key


# ---- trainer: velocity RL and alignment weighting ---------------------------


def _velocity_starts(rows: int, from_ref: list[bool], seed: int = 3) -> dict:
    from rlinf.envs.sim.g1_mimic.constants import HISTORY_LEN
    from rlinf.envs.sim.g1_mimic.env import sample_xi
    from rlinf.envs.sim.g1_mimic.features import default_state

    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    state = default_state(rows, "cpu", torch.float32)
    state[:, 7:] += 0.05 * torch.randn(rows, state.shape[1] - 7, generator=gen)
    return {
        "state": state,
        "prev_action": torch.zeros(rows, 29),
        "xi": sample_xi(rows, "cpu", torch.float32),
        "motion_id": torch.randint(0, 2, (rows,), generator=gen),
        "motion_time": torch.randint(0, 20, (rows,), generator=gen),
        "history": 0.1 * torch.randn(rows, HISTORY_LEN * 93, generator=gen),
        "command": 0.5 * torch.randn(rows, 3, generator=gen),
        "command_from_ref": torch.tensor(from_ref, dtype=torch.bool),
        "task_id": torch.ones(rows, dtype=torch.long),
    }


def _no_noise(obs, noisy):
    # Sensor noise would draw different numbers in split and merged runs.
    return obs


def _velocity_setup(trainer, starts):
    trainer.buffer.sample_starts = lambda n, q0, task=None: starts
    trainer.opt_velocity = torch.optim.SGD(
        trainer.models.velocity_actor_params(), lr=1.0
    )
    trainer.opt_velocity_critic = torch.optim.SGD(
        trainer.models.velocity_critic_params(), lr=1.0
    )


def _velocity_case(dp: DataParallel, out: str, rank_rows: list[int], flags: list[bool]):
    import rlinf.workers.supermimic.trainer as trainer_module
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer_module.add_sensor_noise = _no_noise
    trainer = SuperMimicTrainer(
        _cfg(dp.world_size, velocity=True), device=torch.device("cpu"), dp=dp
    )
    init = {k: v.clone() for k, v in trainer.models.state_dict().items()}
    full = _velocity_starts(sum(rank_rows), flags)
    start = sum(rank_rows[: dp.rank])
    stop = start + rank_rows[dp.rank]
    local = {k: v[start:stop] for k, v in full.items()} if stop > start else None
    _velocity_setup(trainer, local)
    trainer._velocity_imagine()
    torch.save(
        {"init": init, "after": trainer.models.state_dict()},
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
@needs_mjlab
@pytest.mark.parametrize(
    "rank_rows,flags",
    [
        ([3, 1], [True, False, False, True]),
        ([2, 2], [False, False, True, True]),
        ([4, 0], [True, False, True, False]),
    ],
)
def test_velocity_step_on_two_ranks_matches_one_process(
    tmp_path, monkeypatch, rank_rows, flags
):
    """RL term averaged over all starts, alignment over from_ref starts only."""
    import rlinf.workers.supermimic.trainer as trainer_module
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    monkeypatch.setattr(trainer_module, "add_sensor_noise", _no_noise)
    _spawn(_velocity_case, str(tmp_path), rank_rows, flags)
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")
    for key in r0["after"]:
        assert torch.equal(r0["after"][key], r1["after"][key]), key

    reference = SuperMimicTrainer(_cfg(1, velocity=True), device=torch.device("cpu"))
    reference.models.load_state_dict(r0["init"])
    _velocity_setup(reference, _velocity_starts(sum(rank_rows), flags))
    reference._velocity_imagine()
    moved = False
    for key, value in reference.models.state_dict().items():
        assert torch.allclose(r0["after"][key], value, atol=1e-6, rtol=1e-5), key
        if key.startswith("velocity_encoder"):
            moved |= not torch.equal(value, r0["init"][key])
        elif not key.startswith("velocity_critic"):
            assert torch.equal(value, r0["init"][key]), key
    assert moved


# ---- trainer: a rank with an empty replay does not hang --------------------


def _empty_rank_case(dp: DataParallel, out: str) -> None:
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(dp.world_size), device=torch.device("cpu"), dp=dp)
    if dp.rank == 0:
        # Only rank 0 fills its replay, without touching the sampler
        # collective that collect_steps runs once per vector step.
        trainer.reset_sampler.deferred = False
        trainer.collect_steps(trainer.warmup, accumulate=False)
        trainer.reset_sampler.deferred = True
    local_windows = trainer.buffer.num_valid_windows(trainer.k)
    trainer.step_acc = 1.0
    metrics = trainer.maybe_update()
    torch.save(
        {
            "local_windows": local_windows,
            "global_update": trainer.global_update,
            "wm_steps": trainer.wm_optimizer_steps,
            "actor_steps": trainer.actor_optimizer_steps,
            "critic_steps": trainer.critic_optimizer_steps,
            "divergence": metrics.get("train/param_divergence"),
            "state": trainer.models.state_dict(),
        },
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
@needs_mjlab
def test_rank_with_empty_replay_joins_every_step(tmp_path):
    _spawn(_empty_rank_case, str(tmp_path))
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")
    assert r0["local_windows"] > 0 and r1["local_windows"] == 0
    for key in ("global_update", "wm_steps", "actor_steps", "critic_steps"):
        assert r0[key] == r1[key], key
    assert r0["global_update"] == 1
    assert r0["wm_steps"] == 2 and r0["actor_steps"] == 1 and r0["critic_steps"] == 1
    assert r0["divergence"] == 0.0 and r1["divergence"] == 0.0
    for key in r0["state"]:
        assert torch.equal(r0["state"][key], r1["state"][key]), key


# ---- trainer: non-finite gradients are skipped on every rank ---------------


def _nonfinite_case(dp: DataParallel, out: str) -> None:
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(dp.world_size), device=torch.device("cpu"), dp=dp)
    params = trainer.models.critic_params()
    before = [p.detach().clone() for p in params]

    def poison():
        for p in params:
            p.grad = torch.ones_like(p)
        if dp.rank == 1:
            params[0].grad[0] = float("nan")

    def clip(ps):
        return float(torch.nn.utils.clip_grad_norm_(ps, 1.0))

    poison()
    stepped, norm = trainer._dp_step(trainer.opt_critic, params, 1, clip, "critic")
    unchanged = all(torch.equal(a, p.detach()) for a, p in zip(before, params))
    raised = False
    poison()
    try:
        trainer._dp_step(trainer.opt_critic, params, 1, clip, "critic")
    except RuntimeError:
        raised = True
    torch.save(
        {
            "stepped": stepped,
            "norm_is_nan": math.isnan(norm),
            "unchanged": unchanged,
            "skips": trainer.nonfinite_skips,
            "raised": raised,
        },
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
@needs_mjlab
def test_nonfinite_gradient_on_one_rank_is_skipped_everywhere(tmp_path):
    _spawn(_nonfinite_case, str(tmp_path))
    expected = {
        "stepped": False,
        "norm_is_nan": True,
        "unchanged": True,
        "skips": 2,
        "raised": True,
    }
    for rank in range(2):
        assert torch.load(tmp_path / f"rank{rank}.pt") == expected


# ---- running norm: merged batch, identical buffers -------------------------


def _norm_rows():
    prior = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
    local = [
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[2.0, 4.0], [2.0, 4.0]]),
    ]
    extra = [torch.zeros(0, 2), torch.tensor([[4.0, 0.0]])]
    return prior, local, extra


def _norm_case(dp: DataParallel, out: str) -> None:
    from rlinf.models.embodiment.supermimic import RunningNorm

    prior, local, extra = _norm_rows()
    split = RunningNorm(2)
    split.update(prior)
    split.update(local[dp.rank])
    diverged = dp.param_divergence(split)
    synced = RunningNorm(2)
    synced.update(prior)
    dp.update_norm(synced, local[dp.rank])
    dp.update_norm(synced, extra[dp.rank])
    torch.save(
        {
            "diverged": diverged,
            "synced_divergence": dp.param_divergence(synced),
            "mean": synced.mean,
            "var": synced.var,
            "count": synced.count,
        },
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
def test_running_norm_merge_matches_one_batch_on_two_ranks(tmp_path):
    """Different rows, including an empty rank, become one shared Welford state."""
    from rlinf.models.embodiment.supermimic import RunningNorm

    _spawn(_norm_case, str(tmp_path))
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")
    assert r0["diverged"] > 0.0 and r1["diverged"] > 0.0
    assert r0["synced_divergence"] == 0.0 and r1["synced_divergence"] == 0.0
    assert torch.equal(r0["mean"], r1["mean"])
    assert torch.equal(r0["var"], r1["var"])
    assert torch.equal(r0["count"], r1["count"])
    prior, local, extra = _norm_rows()
    ref = RunningNorm(2)
    ref.update(prior)
    ref.update(torch.cat(local))
    ref.update(extra[1])
    assert torch.allclose(r0["mean"], ref.mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(r0["var"], ref.var, atol=1e-5, rtol=1e-5)
    assert torch.allclose(r0["count"], ref.count)


# ---- reset sampler: pooled bin returns -------------------------------------


def _sampler(num_envs: int):
    from rlinf.algorithms.supermimic.reset_sampler import BinResetSampler

    return BinResetSampler(
        torch.tensor([120, 80]), num_envs, 10.0, 0.99, 20, torch.device("cpu")
    )


def _episodes(num_envs: int, seed: int = 5):
    gen = torch.Generator().manual_seed(seed)
    steps = 6
    rewards = torch.rand(steps, num_envs, generator=gen)
    mids = torch.randint(0, 2, (num_envs,), generator=gen)
    times = torch.randint(0, 70, (num_envs,), generator=gen)
    return rewards, mids, times


def _replay_episodes(sampler, rewards, mids, times):
    from rlinf.envs.sim.g1_mimic.constants import EPISODE_FAILURE

    for t in range(rewards.shape[0]):
        sampler.log_step(rewards[t], mids, times + t)
    ids = torch.arange(rewards.shape[1])
    sampler.finish(ids, torch.full_like(ids, EPISODE_FAILURE), None)


def _sampler_case(dp: DataParallel, out: str) -> None:
    rewards, mids, times = _episodes(4)
    half = slice(2 * dp.rank, 2 * dp.rank + 2)
    sampler = _sampler(2)
    sampler.deferred = True
    _replay_episodes(sampler, rewards[:, half], mids[half], times[half])
    sampler.sync(dp)
    torch.save(sampler.state_dict(), Path(out) / f"rank{dp.rank}.pt")


@needs_gloo
def test_reset_bins_pool_finished_episodes_over_ranks(tmp_path):
    _spawn(_sampler_case, str(tmp_path))
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")
    assert torch.equal(r0["bin_return"], r1["bin_return"])
    assert torch.equal(r0["bin_memory"], r1["bin_memory"])
    single = _sampler(4)
    _replay_episodes(single, *_episodes(4))
    assert torch.allclose(r0["bin_return"], single.bin_return, atol=1e-6)
    assert not torch.equal(single.bin_return, torch.full_like(single.bin_return, 10.0))


# ---- checkpoint written by two ranks ---------------------------------------


def _checkpoint_case(dp: DataParallel, out: str) -> None:
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(dp.world_size), device=torch.device("cpu"), dp=dp)
    trainer.global_update = 7
    target = Path(out) / "global_step_7"
    trainer.save_checkpoint(str(target))
    draw_after_save = torch.rand(3)
    torch.rand(5)
    trainer.load_checkpoint(str(target))
    draw_after_load = torch.rand(3)
    torch.save(
        {"same_draw": torch.equal(draw_after_save, draw_after_load)},
        Path(out) / f"rank{dp.rank}.pt",
    )


@needs_gloo
@needs_mjlab
def test_two_rank_checkpoint_is_complete_and_restores_each_rng(tmp_path):
    from rlinf.workers.supermimic import checkpoint as ckpt

    _spawn(_checkpoint_case, str(tmp_path))
    target = tmp_path / "global_step_7"
    meta = ckpt.verify_checkpoint(target)
    assert meta["world_size"] == 2 and meta["global_update"] == 7
    assert set(meta["files"]) == {"supermimic.pt", "rank_0.pt", "rank_1.pt"}
    assert not (tmp_path / "global_step_7.tmp").exists()
    assert ckpt.find_latest_checkpoint(tmp_path) == str(target)
    for rank in range(2):
        assert torch.load(tmp_path / f"rank{rank}.pt") == {"same_draw": True}


_RANK_FAILURE_DRIVER = '''
import os
import sys

import torch
from omegaconf import OmegaConf

from rlinf.runners.supermimic_runner import SuperMimicRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.workers.supermimic.distributed import DataParallel


class FailingRank(Worker):
    """Rank 1 fails in the first iteration while rank 0 waits in an all-reduce."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def init_worker(self):
        self.dp = DataParallel.initialize(
            self._world_size, self._rank, torch.device("cpu"),
            timeout=Cluster.get_collective_timeout(),
        )

    def run_iteration(self):
        if self._rank == 1:
            if self.mode == "raise":
                raise RuntimeError("rank 1 failed on purpose")
            os._exit(3)
        self.dp.all_reduce([1.0])
        return {"train/global_update": 1.0}

    def shutdown(self):
        self.dp.destroy()

    def save_checkpoint(self, directory):
        return None

    def evaluate(self):
        return {}


if __name__ == "__main__":
    mode, log_path = sys.argv[1], sys.argv[2]
    cluster = Cluster(cluster_cfg=OmegaConf.create({"num_nodes": 1, "component_placement": {}}))
    group = FailingRank.create_group(mode).launch(
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
def test_one_failed_rank_ends_the_run_instead_of_hanging(tmp_path, mode):
    """The other rank is blocked in a collective with the default 180 min timeout.

    RLinf's worker group signals the driver as soon as one rank's call fails,
    and the driver kills the group, so the run ends in seconds.
    """
    import subprocess
    import sys
    import time

    script = tmp_path / "driver.py"
    script.write_text(_RANK_FAILURE_DRIVER)
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.pop("RLINF_TIMEOUT", None)
    # Ray otherwise prestarts one idle worker per CPU; on a many-core host with
    # little RAM those alone trip Ray's memory monitor, which stays enabled.
    env.setdefault("RAY_prestart_worker_first_driver", "0")
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script), mode, str(tmp_path / "logs")],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
    )
    elapsed = time.monotonic() - start
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output[-3000:]
    assert "RUN RETURNED" not in output
    assert elapsed < 180, f"the failed run took {elapsed:.0f} s to end"
    if mode == "raise":
        assert "rank 1 failed on purpose" in output
