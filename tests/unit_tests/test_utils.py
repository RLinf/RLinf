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

"""Shared helpers: metrics, entropy aggregation, checkpoint paths, and resume."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.utils import compute_entropy_loss
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


# The embodied actor's entropy bonus aggregation. The shapes below are the ones
# the shipped models actually emit: openpi, openpi_rlinf, lingbotvla,
# dexbotic_pi, dexbotic_dm0 and flow_policy all reduce entropy to [bsz, 1],
# cnn_policy returns [bsz, action_dim], openvla_oft returns [bsz, seq_len], and
# the StarVLA action heads return [bsz, num_action_chunks, action_dim].
# loss_mask is [bsz, 1] under reward_type: chunk_level and
# [bsz, num_action_chunks] otherwise.


def _entropy(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=generator) + 0.5


@pytest.mark.parametrize("batch_size", [4, 16, 64, 500])
def test_chunk_level_entropy_does_not_scale_with_batch_size(batch_size):
    """The bug signature: entropy_loss came out multiplied by the micro-batch size."""
    entropy = _entropy(batch_size, 1)
    loss_mask = torch.ones(batch_size, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_chunk_level_entropy_averages_only_the_valid_rows():
    entropy = _entropy(16, 1)
    loss_mask = torch.zeros(16, 1, dtype=torch.bool)
    loss_mask[:6] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:6].mean()), rel=1e-6)


def test_chunk_level_sums_a_wide_entropy_before_averaging():
    # cnn_policy shape: one entropy per action dimension.
    entropy = _entropy(12, 4)
    loss_mask = torch.zeros(12, 1, dtype=torch.bool)
    loss_mask[:5] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:5].sum(dim=-1).mean()), rel=1e-6)


def test_token_level_averages_over_every_valid_element():
    # openvla_oft shape: entropy per token, mask per chunk step.
    entropy = _entropy(10, 7)
    loss_mask = torch.zeros(10, 1, dtype=torch.bool)
    loss_mask[:4] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:4].mean()), rel=1e-6)


def test_action_level_sums_action_dim_then_averages():
    entropy = _entropy(6, 3 * 7)
    loss_mask = torch.zeros(6, 1, dtype=torch.bool)
    loss_mask[:2] = True

    got = compute_entropy_loss(
        entropy, "action_level", loss_mask, action_dim=7, batch_size=6
    )

    per_chunk = entropy.reshape(6, 3, 7).sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:2].mean()), rel=1e-6)


def test_a_wider_mask_than_entropy_still_weights_by_valid_steps():
    # lingbotvla: entropy is [bsz, 1] while reward_type != chunk_level keeps the
    # mask at [bsz, num_action_chunks]. Each sample is weighted by its valid steps.
    entropy = _entropy(5, 1)
    loss_mask = torch.zeros(5, 4, dtype=torch.bool)
    loss_mask[0, :4] = True
    loss_mask[1, :1] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    expected = (entropy[0, 0] * 4 + entropy[1, 0] * 1) / 5
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_three_dim_entropy_reduces_to_the_mask_rank():
    # StarVLA action heads return [bsz, num_action_chunks, action_dim]; the
    # chunk_level sum already lands on the mask's rank, so nothing is unsqueezed.
    entropy = _entropy(4, 8, 7)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:, :3].mean()), rel=1e-6)


def test_three_dim_entropy_is_right_when_batch_equals_num_chunks():
    # Same shape family with bsz == num_action_chunks, where a rank mismatch
    # broadcasts successfully instead of raising and would go unnoticed.
    entropy = _entropy(8, 8, 7)
    loss_mask = torch.zeros(8, 8, dtype=torch.bool)
    loss_mask[:3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:3].mean()), rel=1e-6)


def test_no_mask_averages_everything():
    entropy = _entropy(9, 1)

    got = compute_entropy_loss(entropy, "chunk_level", None)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_a_fully_masked_batch_contributes_zero():
    entropy = _entropy(8, 1)
    loss_mask = torch.zeros(8, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(0.0)


def test_entropy_loss_keeps_the_gradient_path():
    entropy = _entropy(8, 1).requires_grad_(True)
    loss_mask = torch.ones(8, 1, dtype=torch.bool)

    compute_entropy_loss(entropy, "chunk_level", loss_mask).backward()

    # A correct mean spreads 1/8 of the gradient onto each row; the outer-product
    # bug put 1.0 on each instead.
    assert torch.allclose(entropy.grad, torch.full((8, 1), 1 / 8))


def test_entropy_loss_rejects_a_model_that_computes_no_entropy():
    # gr00t, abot_m0 and evo1 return entropy=None; pairing one with a non-zero
    # entropy_bonus is a config error, not a zero bonus.
    with pytest.raises(ValueError, match="algorithm.entropy_bonus"):
        compute_entropy_loss(None, "chunk_level", torch.ones(4, 1, dtype=torch.bool))


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


_REDIRECTED_ENTRYPOINT = textwrap.dedent(
    """
    import hydra

    from rlinf.scheduler import Cluster
    from rlinf.utils.utils import output_redirector


    @hydra.main(version_base="1.1", config_path=None)
    @output_redirector
    def main(cfg):
        Cluster(num_nodes=1)
        print("entrypoint ran")
        if cfg.outcome == "raise":
            raise RuntimeError("entrypoint failed")


    main()
    """
)


def _run_redirected_entrypoint(tmp_path, outcome):
    script = tmp_path / "entrypoint.py"
    script.write_text(_REDIRECTED_ENTRYPOINT)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            f"+outcome={outcome}",
            f"+runner.output_dir={tmp_path}",
            "+runner.experiment_name=exp",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_redirected_entrypoint_exits_zero_and_keeps_its_log(tmp_path):
    result = _run_redirected_entrypoint(tmp_path, "return")

    assert result.returncode == 0, result.stderr
    assert "entrypoint ran" in (tmp_path / "exp" / "log" / "main.log").read_text()


def test_redirected_entrypoint_failure_is_not_reported_as_success(tmp_path):
    # Hydra catches the exception and calls sys.exit(1) itself, so the failure
    # never reaches sys.excepthook.
    result = _run_redirected_entrypoint(tmp_path, "raise")

    assert result.returncode == 1, result.stderr
    assert "entrypoint failed" in (tmp_path / "exp" / "log" / "main.log").read_text()
