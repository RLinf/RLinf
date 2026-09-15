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

"""Shared helpers: metrics, checkpoint paths, and resume."""

from __future__ import annotations

import importlib.util
import io
import logging
import math
import os
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.utils.metric_utils import compute_evaluate_metrics, compute_rollout_metrics


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


@pytest.fixture
def single_rank_reduction(monkeypatch):
    from rlinf.scheduler.worker.worker import Worker

    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


def test_compute_rollout_metrics_reports_loss_mask_fraction(single_rank_reduction):
    metrics = compute_rollout_metrics(
        {
            "loss_mask": torch.tensor([[[True], [False]], [[True], [True]]]),
            "rewards": torch.tensor([[[1.0], [8.0]], [[2.0], [3.0]]]),
        }
    )

    assert metrics["loss_mask_fraction"] == pytest.approx(0.75)
    assert metrics["rewards"] == pytest.approx(2.0)


def test_compute_rollout_metrics_omits_loss_mask_fraction_without_mask(
    single_rank_reduction,
):
    metrics = compute_rollout_metrics({"rewards": torch.tensor([[[1.0], [3.0]]])})

    assert "loss_mask_fraction" not in metrics
    assert metrics["rewards"] == pytest.approx(2.0)


def _load_checkpoint_utils():
    module_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "checkpoint.py"
    )
    assert module_path.exists(), "checkpoint path utilities are not implemented"
    spec = importlib.util.spec_from_file_location(
        "_rlinf_utils_checkpoint_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/global_step_30",
        "/tmp/checkpoints/global_step_30/",
        "/tmp/checkpoints/global_step_30///",
    ],
)
def test_parse_global_step_accepts_trailing_slashes(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    assert (
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path) == 30
    )


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/step_30",
        "/tmp/checkpoints/global_step_latest/",
        "/tmp/checkpoints/global_step_30/actor",
    ],
)
def test_parse_global_step_rejects_invalid_checkpoint_directories(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    with pytest.raises(ValueError, match="global_step_<step>"):
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path)


class _StubRunner:
    """Expose only the checkpoint helpers and state used by these tests."""

    def __init__(self, critic=None):
        self.critic = critic

    _is_complete_checkpoint = ReasoningRunner._is_complete_checkpoint


class _ImmediateHandle:
    def wait(self):
        return None


class _Actor:
    def save_checkpoint(self, path: str, _step: int):
        os.makedirs(path, exist_ok=True)
        return _ImmediateHandle()


class _Dataloader:
    def state_dict(self):
        return {"offset": 3}


def _write_checkpoint(
    root: Path, step: int, *, complete: bool, with_critic: bool = False
) -> Path:
    checkpoint_dir = root / f"global_step_{step}"
    (checkpoint_dir / "actor").mkdir(parents=True)
    if with_critic:
        (checkpoint_dir / "critic").mkdir()
    if complete:
        data_dir = checkpoint_dir / "data"
        data_dir.mkdir()
        (data_dir / "data.pt").write_bytes(b"dataloader-state")
    return checkpoint_dir


def _resolve_auto_resume(log_path: Path, *, critic=None) -> str | None:
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": "auto", "logger": {"log_path": str(log_path)}}}
    )
    runner = _StubRunner(critic=critic)
    runner.cfg = cfg
    runner.init_rollout_workers = lambda: None
    runner.init_actor_critic_workers = lambda: None

    ReasoningRunner.init_workers(runner)
    return cfg.runner.resume_dir


def _saving_runner(tmp_path: Path) -> _StubRunner:
    runner = _StubRunner()
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "output_dir": str(tmp_path),
                "experiment_name": "experiment",
            }
        }
    )
    runner.global_steps = 8
    runner.actor = _Actor()
    runner.train_dataloader = _Dataloader()
    return runner


@pytest.mark.parametrize(
    "completeness,expected_step",
    [
        pytest.param({40: True, 80: False}, 40, id="skips-the-incomplete-newest"),
        pytest.param({40: True, 80: True}, 80, id="takes-the-newest-complete"),
        pytest.param({40: False}, None, id="starts-fresh-when-none-is-complete"),
    ],
)
def test_auto_resume_selects_the_newest_complete_checkpoint(
    tmp_path, completeness, expected_step
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for step, complete in completeness.items():
        _write_checkpoint(checkpoints_dir, step, complete=complete)

    expected = (
        None
        if expected_step is None
        else str(checkpoints_dir / f"global_step_{expected_step}")
    )
    assert _resolve_auto_resume(tmp_path) == expected


def test_checkpoint_requires_the_critic_only_when_configured(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint = _write_checkpoint(checkpoints_dir, 40, complete=True)

    assert _StubRunner()._is_complete_checkpoint(str(checkpoint))
    assert not _StubRunner(critic=object())._is_complete_checkpoint(str(checkpoint))


def test_dataloader_state_is_published_atomically(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)
    written_paths = []

    def save(_state, path):
        written_paths.append(path)
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", save)

    ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert written_paths == [f"{final_path}.tmp"]
    assert final_path.read_bytes() == b"complete"
    assert not Path(f"{final_path}.tmp").exists()
    assert runner._is_complete_checkpoint(str(checkpoint))


def test_interrupted_dataloader_save_does_not_publish_completion(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)

    def interrupted_save(_state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", interrupted_save)

    with pytest.raises(RuntimeError, match="interrupted"):
        ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert not final_path.exists()
    assert not Path(f"{final_path}.tmp").exists()
    assert not runner._is_complete_checkpoint(str(checkpoint))


def _draw_random_samples() -> tuple[torch.Tensor, np.ndarray, list[float]]:
    return torch.rand(8), np.random.random(8), [random.random() for _ in range(8)]


def _checkpoint_rng_worker(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    rendezvous: str,
    checkpoint_format: str,
    load_world_size: int | None = None,
) -> None:
    from torch.distributed import checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
    from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy
    from rlinf.hybrid_engines.fsdp.utils import FSDPVersion
    from rlinf.utils.utils import get_rng_state, seed_everything

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        seed_everything(0)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        checkpoint = Checkpoint(
            model,
            optimizer,
            scheduler,
            StateDictOptions(),
            FSDPVersion.FSDP2,
            checkpoint_format=checkpoint_format,
        )

        if load_world_size is None:
            # Legacy checkpoints cannot retain different RNG states for each rank.
            seed_everything(42 if checkpoint_format == "legacy" else 42 + rank)
            _draw_random_samples()
            if checkpoint_format == "local_shard":
                shard_dir = Path(checkpoint_dir) / "local_shard_checkpoint"
                shard_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint.state_dict(), shard_dir / f"checkpoint_rank_{rank}.pt"
                )
            else:
                state = checkpoint
                dcp_dir = Path(checkpoint_dir) / "dcp_checkpoint"
                if checkpoint_format == "legacy":
                    state = checkpoint.state_dict()
                    state["rng"] = get_rng_state()
                    dcp_dir = checkpoint_dir
                dcp.save({"fsdp_checkpoint": state}, checkpoint_id=dcp_dir)
            torch.save(
                {
                    "samples": _draw_random_samples(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                Path(checkpoint_dir) / f"expected_rank_{rank}.pt",
            )

        expected_rank = rank if load_world_size is None or rank < 2 else 0
        expected = torch.load(
            Path(checkpoint_dir) / f"expected_rank_{expected_rank}.pt",
            weights_only=False,
        )
        if load_world_size is not None and rank >= 2:
            seed_everything(1234 + rank)
            expected["samples"] = _draw_random_samples()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(999)
        for opt_state in optimizer.state.values():
            opt_state["momentum_buffer"].fill_(999)
        scheduler.step()
        seed_everything(1234 + rank)

        from rlinf.utils.logging import get_logger

        messages = io.StringIO()
        handler = logging.StreamHandler(messages)
        logger = get_logger()
        logger.addHandler(handler)
        try:
            FSDP2Strategy.load_checkpoint(
                model,
                optimizer,
                scheduler,
                checkpoint_dir,
                checkpoint_format="local_shard"
                if checkpoint_format == "local_shard"
                else "dcp",
            )
        finally:
            logger.removeHandler(handler)
        actual = _draw_random_samples()
        torch.testing.assert_close(actual[0], expected["samples"][0], rtol=0, atol=0)
        np.testing.assert_array_equal(actual[1], expected["samples"][1])
        assert actual[2] == expected["samples"][2]
        torch.testing.assert_close(
            model.state_dict(), expected["model"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimizer.state_dict(), expected["optimizer"], rtol=0, atol=0
        )
        assert scheduler.state_dict() == expected["scheduler"]
        assert messages.getvalue().count("RNG world size mismatch") == (
            1 if load_world_size is not None and rank == 0 else 0
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("checkpoint_format", ["dcp", "local_shard", "legacy"])
def test_fsdp_checkpoint_restores_rng(tmp_path: Path, checkpoint_format: str) -> None:
    mp.spawn(
        _checkpoint_rng_worker,
        args=(
            2,
            str(tmp_path / "checkpoint"),
            str(tmp_path / "rendezvous"),
            checkpoint_format,
        ),
        nprocs=2,
        join=True,
    )

    if checkpoint_format == "dcp":
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(
            tmp_path / "checkpoint" / "dcp_checkpoint"
        ).read_metadata()
        assert "fsdp_checkpoint.rng" in metadata.state_dict_metadata
        assert not any(
            key.startswith("fsdp_checkpoint.rng.")
            for key in metadata.state_dict_metadata
        )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("load_world_size", [1, 3])
def test_fsdp_checkpoint_restores_rng_when_world_size_changes(
    tmp_path: Path, load_world_size: int
) -> None:
    checkpoint_dir = str(tmp_path / "checkpoint")
    mp.spawn(
        _checkpoint_rng_worker,
        args=(2, checkpoint_dir, str(tmp_path / "save_rendezvous"), "dcp"),
        nprocs=2,
        join=True,
    )
    mp.spawn(
        _checkpoint_rng_worker,
        args=(
            load_world_size,
            checkpoint_dir,
            str(tmp_path / "load_rendezvous"),
            "dcp",
            load_world_size,
        ),
        nprocs=load_world_size,
        join=True,
    )
