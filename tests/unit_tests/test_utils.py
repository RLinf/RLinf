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
import math
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

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


class _CheckpointTensorModel(torch.nn.Module):
    def __init__(self, mesh, shard_dim: int | None):
        super().__init__()
        from torch.distributed.tensor import Replicate, Shard, distribute_tensor

        # Include uneven and empty local shards, plus a noncontiguous parameter.
        for name, tensor in {
            "even": torch.randn(4, 4),
            "uneven": torch.randn(5, 3),
            "small": torch.randn(1, 1),
            "transposed": torch.randn(3, 5).t(),
        }.items():
            if shard_dim is not None:
                tensor = distribute_tensor(tensor, mesh, [Shard(shard_dim)])
            self.register_parameter(name, torch.nn.Parameter(tensor))
        self.register_buffer("updates", torch.tensor(0))
        constant = torch.tensor([2.0, 3.0])
        if shard_dim is not None:
            constant = distribute_tensor(constant, mesh, [Replicate()])
        self.register_buffer("constant", constant)
        self.label = "initial"

    def get_extra_state(self):
        return self.label

    def set_extra_state(self, state):
        self.label = state


def _checkpoint_tensor_snapshot(value):
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        return {
            "local": value.to_local().detach().clone(),
            "shape": tuple(value.shape),
            "stride": value.stride(),
            "placements": str(value.placements),
        }
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _checkpoint_tensor_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_checkpoint_tensor_snapshot(item) for item in value)
    return value


def _assert_checkpoint_tensor_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_checkpoint_tensor_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_checkpoint_tensor_state(actual_item, expected_item)
    else:
        assert actual == expected


def _local_shard_checkpoint_worker(
    rank: int,
    checkpoint_dir: str,
    rendezvous: str,
    shard_dim: int | None,
    phase: str,
) -> None:
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh

    from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
    from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy
    from rlinf.hybrid_engines.fsdp.utils import FSDPVersion
    from rlinf.scheduler import Worker

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # These processes exercise CPU tensors without starting a Ray Worker.
        Worker.torch_platform = torch.cuda
        Worker.torch_device_type = "cuda"
        torch.manual_seed(42 if phase == "save" else 123)
        mesh = init_device_mesh("cpu", (2,))
        model = _CheckpointTensorModel(mesh, shard_dim)
        parameters = list(model.parameters())
        parameter_groups = (
            [parameters[:2], parameters[2:]] if shard_dim == 1 else [parameters]
        )
        optimizers = [
            torch.optim.AdamW(group, lr=0.01 * (index + 1))
            for index, group in enumerate(parameter_groups)
        ]
        schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
            for optimizer in optimizers
        ]
        optimizer_arg = optimizers[0] if len(optimizers) == 1 else optimizers
        scheduler_arg = schedulers[0] if len(schedulers) == 1 else schedulers
        checkpoint = Checkpoint(
            model,
            optimizer_arg,
            scheduler_arg,
            StateDictOptions(),
            FSDPVersion.FSDP2,
            checkpoint_format="local_shard",
        )

        def train_step():
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss = sum(parameter.square().sum() for parameter in model.parameters())
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            for scheduler in schedulers:
                scheduler.step()
            model.updates.add_(1)
            return float(loss.item())

        def snapshot():
            return _checkpoint_tensor_snapshot(
                {
                    "model": model.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                }
            )

        shard_dir = Path(checkpoint_dir) / "local_shard_checkpoint"
        reference_file = Path(checkpoint_dir) / f"reference_rank_{rank}.pt"
        if phase == "save":
            for _ in range(3):
                train_step()
            model.label = "saved"
            shard_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint.state_dict(), shard_dir / f"checkpoint_rank_{rank}.pt"
            )
            reference = {"saved": snapshot()}
            reference["losses"] = [train_step() for _ in range(4)]
            reference["continued"] = snapshot()
            torch.save(reference, reference_file)
        else:
            reference = torch.load(reference_file, weights_only=False)
            FSDP2Strategy.load_checkpoint(
                model,
                optimizer_arg,
                scheduler_arg,
                checkpoint_dir,
                checkpoint_format="local_shard",
            )
            _assert_checkpoint_tensor_state(snapshot(), reference["saved"])
            assert [train_step() for _ in range(4)] == reference["losses"]
            _assert_checkpoint_tensor_state(snapshot(), reference["continued"])

            if shard_dim is not None:
                state = torch.load(
                    shard_dir / f"checkpoint_rank_{rank}.pt", weights_only=False
                )
                state["model"]["even"] = state["model"]["even"][:1, :1]
                with pytest.raises(
                    ValueError, match="Local shard shape mismatch for even"
                ):
                    checkpoint.load_state_dict(state)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("shard_dim", [0, 1, None], ids=["shard-0", "shard-1", "plain"])
def test_local_shard_checkpoint_restores_training_in_new_processes(
    tmp_path: Path, shard_dim: int | None
) -> None:
    for phase in ("save", "load"):
        mp.spawn(
            _local_shard_checkpoint_worker,
            args=(
                str(tmp_path / "checkpoint"),
                str(tmp_path / f"{phase}_rendezvous"),
                shard_dim,
                phase,
            ),
            nprocs=2,
            join=True,
        )
