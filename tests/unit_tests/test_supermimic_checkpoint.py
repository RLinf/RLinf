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

"""SuperMimic checkpoint integrity, atomic publish, RNG, resume, metrics and OOM report."""

from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest
import torch
from _supermimic_support import synthetic_clip_dir
from omegaconf import OmegaConf

from rlinf.workers.supermimic import checkpoint as ckpt

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="the trainer reads G1 joint limits from mjlab",
)


def _fake_checkpoint(directory, world_size=1, step=0):
    directory.mkdir(parents=True)
    torch.save({"step": step}, directory / ckpt.MODEL_FILE)
    for rank in range(world_size):
        torch.save({"rank": rank}, directory / ckpt.rank_file(rank))
    return ckpt.write_meta(directory, {"world_size": world_size, "global_update": step})


def test_latest_checkpoint_skips_unfinished_and_corrupt(tmp_path):
    _fake_checkpoint(tmp_path / "global_step_5", step=5)
    _fake_checkpoint(tmp_path / "global_step_9", step=9)
    with open(tmp_path / "global_step_9" / ckpt.MODEL_FILE, "r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 0xFF]))
    _fake_checkpoint(tmp_path / "global_step_12.tmp", step=12)
    (tmp_path / "global_step_20").mkdir()
    torch.save({}, tmp_path / "global_step_20" / ckpt.MODEL_FILE)

    assert not ckpt.is_complete_checkpoint(tmp_path / "global_step_9")
    assert not ckpt.is_complete_checkpoint(tmp_path / "global_step_12.tmp")
    assert not ckpt.is_complete_checkpoint(tmp_path / "global_step_20")
    assert ckpt.find_latest_checkpoint(tmp_path) == str(tmp_path / "global_step_5")
    assert ckpt.find_latest_checkpoint(tmp_path / "missing") is None


def test_meta_requires_every_rank_file(tmp_path):
    directory = tmp_path / "global_step_1.tmp"
    directory.mkdir()
    torch.save({}, directory / ckpt.MODEL_FILE)
    torch.save({}, directory / ckpt.rank_file(0))
    with pytest.raises(RuntimeError, match="shared by all nodes"):
        ckpt.write_meta(directory, {"world_size": 2, "global_update": 1})


def test_publish_replaces_an_existing_step(tmp_path):
    final = tmp_path / "global_step_3"
    _fake_checkpoint(final, step=3)
    tmp = ckpt.temp_dir(final)
    _fake_checkpoint(tmp, step=33)
    ckpt.publish(tmp, final)
    assert not tmp.exists()
    assert not (tmp_path / "global_step_3.old").exists()
    assert ckpt.verify_checkpoint(final)["global_update"] == 33


def test_rng_capture_round_trips_through_weights_only_load(tmp_path):
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)
    torch.save(ckpt.capture_rng(torch.device("cpu")), tmp_path / "rng.pt")
    expected = (torch.rand(4), random.random(), np.random.rand(3))
    torch.rand(9)
    random.random()
    np.random.rand(5)
    ckpt.restore_rng(
        torch.load(tmp_path / "rng.pt", weights_only=True), torch.device("cpu")
    )
    assert torch.equal(torch.rand(4), expected[0])
    assert random.random() == expected[1]
    assert np.array_equal(np.random.rand(3), expected[2])


def _cfg(velocity: bool = False):
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
                "updates_per_env_step": 1.0,
                "wm_batches": 1,
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


@needs_mjlab
def test_trainer_checkpoint_restores_state_and_rng(tmp_path):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(), device=torch.device("cpu"))
    trainer.run_until_updates(2)
    trainer.reset_sampler.bin_return[0] = 3.25
    target = tmp_path / "global_step_2"
    trainer.save_checkpoint(str(target))
    next_draw = torch.rand(3)

    meta = ckpt.verify_checkpoint(target)
    assert meta["world_size"] == 1 and meta["global_update"] == 2
    assert meta["goal_dim"] == 680 and meta["obs_dim"] == 93
    assert not ckpt.temp_dir(target).exists()

    resumed = SuperMimicTrainer(_cfg(), device=torch.device("cpu"))
    resumed.load_checkpoint(str(target))
    assert torch.equal(torch.rand(3), next_draw)
    for key, value in trainer.models.state_dict().items():
        assert torch.equal(resumed.models.state_dict()[key], value), key
    assert (
        resumed.opt_actor.state_dict()["state"].keys()
        == trainer.opt_actor.state_dict()["state"].keys()
    )
    assert resumed.sched_wm.last_epoch == trainer.sched_wm.last_epoch
    for name in (
        "global_update",
        "sim_vector_steps",
        "env_transitions",
        "wm_optimizer_steps",
        "actor_optimizer_steps",
        "critic_optimizer_steps",
    ):
        assert getattr(resumed, name) == getattr(trainer, name), name
    assert float(resumed.reset_sampler.bin_return[0]) == 3.25
    assert torch.equal(
        resumed.reset_sampler.bin_memory, trainer.reset_sampler.bin_memory
    )
    # Replay is not saved, so the resumed trainer warms a new one first.
    assert resumed.replay_warm is False


@needs_mjlab
def test_interrupted_save_leaves_no_complete_checkpoint(tmp_path, monkeypatch):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(), device=torch.device("cpu"))
    trainer.save_checkpoint(str(tmp_path / "global_step_1"))

    def interrupted(directory, meta):
        raise OSError("disk full")

    monkeypatch.setattr(ckpt, "write_meta", interrupted)
    trainer.global_update = 2
    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint(str(tmp_path / "global_step_2"))
    assert not (tmp_path / "global_step_2").exists()
    assert (tmp_path / "global_step_2.tmp").exists()
    assert ckpt.find_latest_checkpoint(tmp_path) == str(tmp_path / "global_step_1")

    monkeypatch.undo()
    trainer.save_checkpoint(str(tmp_path / "global_step_2"))
    assert not (tmp_path / "global_step_2.tmp").exists()
    assert ckpt.find_latest_checkpoint(tmp_path) == str(tmp_path / "global_step_2")


@needs_mjlab
def test_checkpoint_with_other_dimensions_is_rejected(tmp_path):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    SuperMimicTrainer(_cfg(), device=torch.device("cpu")).save_checkpoint(
        str(tmp_path / "global_step_0")
    )
    other = SuperMimicTrainer(_cfg(velocity=True), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="velocity_tracking"):
        other.load_checkpoint(str(tmp_path / "global_step_0"))


@needs_mjlab
@needs_mjlab
def test_checkpoints_of_another_format_or_goal_window_are_refused(tmp_path):
    import json

    from rlinf.envs.sim.g1_mimic.motion import GOAL_WINDOW
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(), device=torch.device("cpu"))
    target = tmp_path / "global_step_0"
    trainer.save_checkpoint(str(target))
    meta_path = target / ckpt.META_FILE
    meta = json.loads(meta_path.read_text())
    assert meta["format"] == ckpt.FORMAT_VERSION
    assert meta["goal_window"] == list(GOAL_WINDOW["paper"]) == [0, 9]

    def load_with(**changes):
        meta_path.write_text(json.dumps({**meta, **changes}))
        SuperMimicTrainer(_cfg(), device=torch.device("cpu")).load_checkpoint(
            str(target)
        )

    with pytest.raises(ValueError, match="goal_window"):
        load_with(goal_window=[1, 10])
    with pytest.raises(ValueError, match="format 2"):
        load_with(format=2)
    meta_path.unlink()
    with pytest.raises(ValueError, match="has no"):
        SuperMimicTrainer(_cfg(), device=torch.device("cpu")).load_checkpoint(
            str(target)
        )


def test_runner_auto_resume_picks_latest_complete_checkpoint(tmp_path):
    from rlinf.runners.supermimic_runner import SuperMimicRunner

    root = tmp_path / "supermimic_smoke"
    _fake_checkpoint(root / "global_step_3", step=3)
    _fake_checkpoint(root / "global_step_6.tmp", step=6)

    class _Result:
        def __init__(self, payload=None):
            self.payload = payload

        def wait(self):
            return [self.payload]

    class _Worker:
        def __init__(self):
            self.loaded = []

        def init_worker(self):
            return _Result()

        def load_checkpoint(self, directory):
            self.loaded.append(directory)
            return _Result()

    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "rlinf",
                    "experiment_name": "supermimic_smoke",
                    "logger_backends": [],
                },
                "resume_dir": "auto",
            },
            "algorithm": {"total_updates": 10},
        }
    )
    worker = _Worker()
    runner = SuperMimicRunner(cfg, worker)
    runner.init_workers()
    assert worker.loaded == [str(root / "global_step_3")]
    assert runner.global_step == 3

    empty = _Worker()
    cfg.runner.logger.experiment_name = "nothing_here"
    fresh = SuperMimicRunner(cfg, empty)
    fresh.init_workers()
    assert empty.loaded == [] and fresh.global_step == 0


def test_rank_metrics_merge_counts_times_and_losses():
    from rlinf.runners.supermimic_runner import aggregate_rank_metrics

    merged = aggregate_rank_metrics(
        [
            {
                "train/global_update": 5.0,
                "env_transitions": 100.0,
                "time/sim_s": 1.0,
                "critic/loss": 2.0,
                "wm/state_loss": float("nan"),
                "memory/host_rss": 10.0,
            },
            {
                "train/global_update": 5.0,
                "env_transitions": 100.0,
                "time/sim_s": 3.0,
                "critic/loss": 4.0,
                "wm/state_loss": 1.5,
                "memory/host_rss": 20.0,
            },
        ],
        per_rank_detail=True,
    )
    assert merged["train/global_update"] == 5.0
    assert merged["env_transitions"] == 200.0
    assert merged["time/sim_s"] == 3.0
    assert merged["critic/loss"] == 3.0
    assert merged["wm/state_loss"] == 1.5
    assert merged["memory/host_rss"] == 20.0
    assert merged["memory/host_rss_total"] == 30.0
    assert merged["train/world_size"] == 2.0
    assert merged["rank_1/critic/loss"] == 4.0
    single = {"critic/loss": 1.0}
    assert aggregate_rank_metrics([single]) == single


@needs_mjlab
def test_update_iteration_reports_monitoring_metrics():
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_cfg(), device=torch.device("cpu"))
    metrics = trainer.run_until_updates(1)
    for key in (
        "critic/value_mean",
        "critic/target_mean",
        "sim/reward_mean",
        "reset/bin_return_mean",
        "reset/bin_return_min",
        "wm/one_step_loss",
        "wm/k_step_loss",
        "time/comm_s",
    ):
        assert math.isfinite(metrics[key]), key
    assert metrics["reset/bin_return_min"] <= metrics["reset/bin_return_mean"]
    assert metrics["sim/episodes_finished"] >= 0.0
    assert metrics["train/nonfinite_skips"] == 0.0


def test_cuda_oom_names_the_rank_and_per_rank_sizes():
    from rlinf.workers.supermimic.worker import explain_cuda_oom

    cfg = OmegaConf.create(
        {
            "env": {"num_envs": 4096, "replay_length": 1000},
            "algorithm": {"wm_window_batch": 2048, "imagine_batch": 4096},
        }
    )
    original = torch.cuda.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 2 GiB"
    )
    with pytest.raises(torch.cuda.OutOfMemoryError) as info:
        with explain_cuda_oom(cfg, 1, 4, "training"):
            raise original
    message = str(info.value)
    assert "rank 1 during training" in message
    assert "num_envs=1024" in message and "imagine_batch=1024" in message
    assert "wm_window_batch=512" in message and "replay_length=1000" in message
    assert info.value.__cause__ is original
    with pytest.raises(ValueError, match="passes through"):
        with explain_cuda_oom(cfg, 0, 1, "training"):
            raise ValueError("passes through")
