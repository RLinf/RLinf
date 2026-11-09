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

"""Shared helpers: metrics, padding, rollout redistribution, and checkpointing."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import math
import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    apply_importance_sampling_correction,
    compute_entropy_loss,
)
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.utils.distributed import balance_rollout_groups
from rlinf.utils.metric_utils import compute_evaluate_metrics, compute_rollout_metrics
from rlinf.utils.torch_functionals import batch_pad_to_fixed_len
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


@pytest.mark.parametrize("left_pad", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int64, torch.float32, torch.float64]
)
def test_batch_padding_preserves_values_and_dtype(left_pad, dtype):
    rows = [torch.tensor(values, dtype=dtype) for values in ([], [1], [2, 3, 4])]
    padded = batch_pad_to_fixed_len(rows, 3, -1, left_pad=left_pad)
    expected = torch.tensor(
        [[-1, -1, -1], [-1, -1, 1] if left_pad else [1, -1, -1], [2, 3, 4]],
        dtype=dtype,
    )
    torch.testing.assert_close(padded, expected)
    assert [len(row) for row in rows] == [0, 1, 3]


@pytest.mark.parametrize("left_pad", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_batch_padding_preserves_input_device(device, left_pad):
    rows = [torch.ones(n, device=device) for n in (1, 3)]
    padded = batch_pad_to_fixed_len(rows, 5, 0, left_pad=left_pad)
    assert padded.device == rows[0].device
    assert padded.dtype == rows[0].dtype and padded.shape == (2, 5)


@pytest.mark.parametrize("left_pad", [False, True])
def test_batch_padding_preserves_gradients(left_pad):
    rows = [
        torch.arange(1, n + 1, dtype=torch.float32).requires_grad_() for n in (1, 3, 5)
    ]
    padded = batch_pad_to_fixed_len(rows, 8, 0, left_pad=left_pad)
    padded.square().sum().backward()
    for row in rows:
        torch.testing.assert_close(row.grad, 2 * row.detach())


@pytest.mark.parametrize("left_pad", [False, True])
def test_batch_padding_does_not_truncate_overlength_rows(left_pad):
    with pytest.raises(RuntimeError, match="negative"):
        batch_pad_to_fixed_len([torch.arange(5)], 4, 0, left_pad=left_pad)


@pytest.mark.parametrize("valid_tokens", [0, 5])
def test_seq_mean_token_mean_handles_empty_rows(valid_tokens):
    from rlinf.utils.utils import seq_mean_token_mean

    values = torch.ones(2, 8, requires_grad=True)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[1, :valid_tokens] = True
    loss = seq_mean_token_mean(values, mask)
    assert loss.item() == (0.5 if valid_tokens else 0.0)
    loss.backward()
    expected = torch.zeros_like(values)
    if valid_tokens:
        expected[1, :valid_tokens] = 1 / (2 * valid_tokens)
    torch.testing.assert_close(values.grad, expected)


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
# the shipped models actually emit: openpi, lingbotvla,
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


# ReinForce++ advantages on the reasoning path: rewards are one scalar per
# response and response_mask is [bsz, seq_len], right-padded.


def _reinpp_advantages(rewards, use_reinpp_baseline):
    response_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    )
    advantages, _ = calculate_adv_and_returns(
        task_type="reasoning",
        adv_type="reinpp",
        rewards=rewards,
        loss_mask=response_mask,
        group_size=2,
        use_reinpp_baseline=use_reinpp_baseline,
    )
    return advantages


def test_reinpp_baseline_subtracts_the_group_mean_reward():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])

    with_baseline = _reinpp_advantages(rewards, use_reinpp_baseline=True)
    centered = torch.tensor([0.5, -0.5, 0.0, 0.0])
    expected = _reinpp_advantages(centered, use_reinpp_baseline=False)

    assert torch.allclose(with_baseline, expected)


def test_reinpp_baseline_leaves_the_caller_rewards_unchanged():
    # The workers pass batch["rewards"] straight through and log it afterwards.
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])

    _reinpp_advantages(rewards, use_reinpp_baseline=True)

    assert rewards.tolist() == [1.0, 0.0, 1.0, 1.0]


# Importance-sampling correction: the actor rescores rollout actions, and
# each advantage is scaled by the clipped actor/rollout probability ratio.


class _ReplayModel:
    def __init__(self, offset: float = 0.25) -> None:
        self.batch_sizes = []
        self.kwargs = []
        self.offset = offset

    def __call__(self, *, forward_inputs, **kwargs):
        expected = forward_inputs["expected_logprobs"]
        self.batch_sizes.append(expected.shape[0])
        self.kwargs.append(kwargs)
        return {"logprobs": expected + self.offset}


def test_importance_sampling_correction_scales_advantages_by_clipped_weight():
    advantages = torch.tensor([1.0, 2.0], dtype=torch.float32)
    rollout_logprobs = torch.zeros(2, dtype=torch.float32)
    recomputed_logprobs = torch.tensor(
        [math.log(2.0), math.log(4.0)], dtype=torch.float32
    )

    corrected, weights, log_ratios = apply_importance_sampling_correction(
        advantages,
        recomputed_logprobs,
        rollout_logprobs,
        clip=3.0,
    )

    torch.testing.assert_close(corrected, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(weights, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(log_ratios, recomputed_logprobs)


@pytest.mark.parametrize("importance_sampling_fix", [False, True])
def test_embodied_actor_recompute_preserves_behavior_logprobs_when_enabled(
    monkeypatch, importance_sampling_fix
):
    monkeypatch.setattr(
        "rlinf.workers.actor.embodied_fsdp_actor_worker.clear_memory", lambda: None
    )
    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 3,
                "model": {"model_type": "openpi"},
            },
            "algorithm": {"importance_sampling_fix": importance_sampling_fix},
            "rollout": {},
        }
    )
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.is_weight_offloaded = False
    worker.model = _ReplayModel()
    expected = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    behavior = torch.zeros_like(expected)
    worker.rollout_batch = {
        "prev_logprobs": behavior,
        "forward_inputs": {"expected_logprobs": expected},
    }

    metrics = inspect.unwrap(EmbodiedFSDPActor.recompute_prev_logprobs)(
        worker, batch_size_per_rank=3
    )

    assert worker.model.batch_sizes == [3, 3]
    assert all("temperature" not in kwargs for kwargs in worker.model.kwargs)
    assert all("top_k" not in kwargs for kwargs in worker.model.kwargs)
    if importance_sampling_fix:
        assert worker.rollout_batch["rollout_logprobs"] is behavior
    else:
        assert "rollout_logprobs" not in worker.rollout_batch
    assert torch.equal(worker.rollout_batch["prev_logprobs"], expected + 0.25)
    assert metrics["actor/rollout_train_logprob_gap"] == pytest.approx(11.75)
    assert metrics["actor/rollout_train_joint_logprob_gap_max"] == pytest.approx(87.0)


def test_lingbot_recompute_restores_action_layout_before_masking(monkeypatch):
    monkeypatch.setattr(
        "rlinf.workers.actor.embodied_fsdp_actor_worker.clear_memory", lambda: None
    )
    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 3,
                "model": {"model_type": "lingbotvlav2", "action_dim": 2},
            },
            "algorithm": {"max_recompute_logprob_gap": 1.0e-3},
            "rollout": {},
        }
    )
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.is_weight_offloaded = False
    worker.model = _ReplayModel(offset=0.0)

    recomputed = torch.zeros(6, 4, dtype=torch.float32)
    per_action_gap = torch.linspace(5.0e-5, 2.0e-4, 6).unsqueeze(-1)
    rollout = recomputed - per_action_gap
    worker.rollout_batch = {
        "prev_logprobs": rollout,
        "forward_inputs": {"expected_logprobs": recomputed},
        "loss_mask": torch.ones(6, 1, dtype=torch.bool),
    }

    metrics = inspect.unwrap(EmbodiedFSDPActor.recompute_prev_logprobs)(
        worker, batch_size_per_rank=3
    )

    assert metrics["actor/rollout_train_joint_logprob_gap_max"] == pytest.approx(8.0e-4)


@pytest.mark.parametrize(
    (
        "logprob_type",
        "expected_loss",
        "mean_weight",
        "max_weight",
        "clip_fraction",
        "log_gap",
    ),
    [
        ("token_level", -2.5, 2.5, 3.0, 0.0, math.log(6.0) / 2),
        ("action_level", -4.5, 6.5, 9.0, 0.5, math.log(6.0)),
        ("chunk_level", -5.0, 36.0, 36.0, 1.0, math.log(36.0)),
    ],
)
def test_embodied_importance_sampling_uses_loss_logprob_granularity(
    logprob_type, expected_loss, mean_weight, max_weight, clip_fraction, log_gap
):
    rollout_logprobs = torch.zeros(2, 2, 2, dtype=torch.float32)
    recomputed_logprobs = torch.tensor(
        [[[math.log(2.0)] * 2, [math.log(3.0)] * 2]] * 2,
        dtype=torch.float32,
    )

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type=logprob_type,
        reward_type="chunk_level",
        single_action_dim=2,
        logprobs=recomputed_logprobs.clone(),
        old_logprobs=recomputed_logprobs,
        rollout_logprobs=rollout_logprobs,
        advantages=torch.ones(2, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        importance_sampling_clip=5.0,
    )

    assert loss.item() == pytest.approx(expected_loss)
    assert metrics["actor/ratio"] == pytest.approx(1.0)
    assert metrics["actor/importance_sampling_weight"] == pytest.approx(mean_weight)
    assert metrics["actor/importance_sampling_weight_max"] == pytest.approx(max_weight)
    assert metrics["actor/importance_sampling_clip_fraction"] == pytest.approx(
        clip_fraction
    )
    assert metrics["actor/recomputed_logprob_abs_diff"] == pytest.approx(log_gap)


def test_embodied_importance_sampling_metrics_broadcast_loss_mask():
    rollout_logprobs = torch.zeros(1, 2, 2, dtype=torch.float32)
    recomputed_logprobs = torch.tensor(
        [[[math.log(2.0), math.log(4.0)], [math.log(128.0), math.log(256.0)]]],
        dtype=torch.float32,
    )

    _, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=2,
        logprobs=recomputed_logprobs.clone(),
        old_logprobs=recomputed_logprobs,
        rollout_logprobs=rollout_logprobs,
        advantages=torch.ones(1, 2, dtype=torch.float32),
        loss_mask=torch.tensor([[True, False]]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        importance_sampling_clip=3.0,
    )

    assert metrics["actor/importance_sampling_weight"] == pytest.approx(3.0)
    assert metrics["actor/importance_sampling_weight_max"] == pytest.approx(4.0)
    assert metrics["actor/importance_sampling_clip_fraction"] == pytest.approx(0.5)
    assert metrics["actor/recomputed_logprob_abs_diff"] == pytest.approx(
        math.log(8.0) / 2
    )


def test_embodied_loss_is_unchanged_without_importance_sampling():
    logprobs = torch.tensor([[[0.2, -0.1]]], dtype=torch.float32)
    old_logprobs = torch.tensor([[[0.1, -0.2]]], dtype=torch.float32)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="token_level",
        reward_type="chunk_level",
        single_action_dim=2,
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=torch.ones(1, 1, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    expected_ratio = torch.exp(logprobs - old_logprobs)
    assert loss.item() == pytest.approx(-expected_ratio.mean().item())
    assert metrics["actor/ratio"] == pytest.approx(expected_ratio.mean().item())
    assert "actor/importance_sampling_weight" not in metrics


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


def _flash_availability(monkeypatch, **installed):
    """Pretend transformers reports exactly the given flash variants."""
    import transformers.utils as transformers_utils

    for version in (2, 3, 4):
        name = f"is_flash_attn_{version}_available"
        monkeypatch.setattr(
            transformers_utils,
            name,
            lambda version=version: installed.get(f"fa{version}", False),
            raising=False,
        )


def test_attn_implementation_keeps_the_preferred_variant(monkeypatch):
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch, fa2=True)
    assert resolve_attn_implementation() == "flash_attention_2"


def test_attn_implementation_falls_back_to_the_installed_variant(monkeypatch):
    """An sm90+ image ships FA4 instead of FA2, so FA2 must not be requested."""
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch, fa4=True)
    assert resolve_attn_implementation("flash_attention_2") == "flash_attention_4"


def test_attn_implementation_falls_back_to_sdpa_without_flash_attn(monkeypatch):
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch)
    assert resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_attn_implementation_survives_a_probe_that_raises(monkeypatch):
    """A probe can import kernels and fail on platforms without flash-attn."""
    import transformers.utils as transformers_utils

    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch)

    def explode():
        raise RuntimeError("no flash-attn kernels for this platform")

    monkeypatch.setattr(
        transformers_utils, "is_flash_attn_2_available", explode, raising=False
    )
    assert resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_attn_implementation_passes_through_non_flash_choices(monkeypatch):
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch)
    assert resolve_attn_implementation("eager") == "eager"


def test_pinned_offload_context_restores_after_overlapping_exits():
    from rlinf.utils.pinned_offload import pinned_offload_context

    original_to = torch.Tensor.to
    first = pinned_offload_context()
    second = pinned_offload_context()
    first_active = True
    first.__enter__()
    second.__enter__()
    try:
        patched_to = torch.Tensor.to
        assert patched_to is not original_to

        tensor = torch.ones(2)
        assert tensor.to(torch.float64).dtype == torch.float64
        torch.testing.assert_close(tensor.to(torch.zeros(2)), tensor)

        first.__exit__(None, None, None)
        first_active = False
        assert torch.Tensor.to is patched_to
    finally:
        if first_active:
            first.__exit__(None, None, None)
        second.__exit__(None, None, None)

    assert torch.Tensor.to is original_to


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "source_format", [torch.contiguous_format, torch.channels_last]
)
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format, torch.channels_last],
)
def test_pinned_offload_preserves_cuda_tensor_values_and_layout(
    dtype, source_format, memory_format
):
    from rlinf.utils.pinned_offload import pinned_offload_context

    source = torch.randn(2, 3, 8, 8, device="cuda", dtype=dtype).contiguous(
        memory_format=source_format
    )
    expected = source.to("cpu", memory_format=memory_format)
    original_to = torch.Tensor.to
    with pinned_offload_context():
        result = source.to("cpu", None, False, memory_format=memory_format)
        assert result.is_pinned()
        assert result.device.type == "cpu"
        assert result.dtype == source.dtype
        assert result.stride() == expected.stride()
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.to(source.device), source)
    assert torch.Tensor.to is original_to


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_pinned_offload_restores_after_cuda_copy_and_exception():
    from rlinf.utils.pinned_offload import pinned_offload_context

    source = torch.ones(2, device="cuda")
    original_to = torch.Tensor.to
    with pytest.raises(RuntimeError, match="abort offload"):
        with pinned_offload_context():
            assert source.to(device=torch.device("cpu")).is_pinned()
            raise RuntimeError("abort offload")
    assert torch.Tensor.to is original_to


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_pinned_offload_reports_allocation_failure_and_restores_patch(monkeypatch):
    from rlinf.utils.pinned_offload import pinned_offload_context

    source = torch.ones(2, device="cuda", dtype=torch.float32)
    original_to = torch.Tensor.to
    allocator_error = RuntimeError("pinned allocator exhausted")

    def fail_allocation(*args, **kwargs):
        assert kwargs["pin_memory"] is True
        raise allocator_error

    monkeypatch.setattr(torch, "empty_like", fail_allocation)
    with pytest.raises(
        RuntimeError, match="Failed to allocate pinned CPU memory"
    ) as info:
        with pinned_offload_context():
            source.to("cpu")
    assert info.value.__cause__ is allocator_error
    assert "shape=(2,)" in str(info.value)
    assert "dtype=torch.float32" in str(info.value)
    assert f"source_device={source.device}" in str(info.value)
    assert "requested_bytes=8" in str(info.value)
    assert torch.Tensor.to is original_to


def _balance_groups_process(rank, counts, num_groups, init_file):
    world_size = len(counts) * num_groups
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        dp_group = dist.group.WORLD
        if num_groups > 1:
            groups = [
                dist.new_group(list(range(i, world_size, num_groups)))
                for i in range(num_groups)
            ]
            dp_group = groups[rank % num_groups]
        dp_rank = dist.get_rank(dp_group)
        total = sum(counts)
        first_identity = (rank % num_groups) * total
        start = first_identity + sum(counts[:dp_rank])
        results = []
        for i in range(counts[dp_rank]):
            identity = start + i
            length = 11 + identity * 7
            results.append(
                DynamicRolloutResult(
                    num_sequence=2,
                    group_size=2,
                    idx_to_traj=[0, 1],
                    input_ids=[[identity] * length, [identity] * (length + 1)],
                    prompt_lengths=[3, 3],
                    response_lengths=[length - 3, length - 2],
                    is_end=[True, True],
                    rewards=[float(identity), -float(identity)],
                    rollout_logprobs=[[-0.5] * (length - 3), [-0.25] * (length - 2)],
                    extra_fields_group={"identity": identity},
                    extra_fields_train={"idx_to_sub_traj": [0, 0]},
                )
            )
        snapshots = copy.deepcopy(results)
        output = balance_rollout_groups(results, dp_group)
        assert len(output) == total // len(counts)
        local = {res.extra_fields_group["identity"]: res for res in results}
        for result in output:
            identity = result.extra_fields_group["identity"]
            if identity in local:
                assert result is local[identity]
            length = 11 + identity * 7
            assert result.idx_to_traj == [0, 1]
            assert result.input_ids == [[identity] * length, [identity] * (length + 1)]
            assert result.rewards == [float(identity), -float(identity)]
            assert result.rollout_logprobs == [
                [-0.5] * (length - 3),
                [-0.25] * (length - 2),
            ]
            assert result.extra_fields_train == {"idx_to_sub_traj": [0, 0]}
        assert results == snapshots
        identities = [None] * len(counts)
        dist.all_gather_object(
            identities,
            [res.extra_fields_group["identity"] for res in output],
            group=dp_group,
        )
        assert sorted(sum(identities, [])) == list(
            range(first_identity, first_identity + total)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("counts", "num_groups"),
    [
        pytest.param((2,), 1, id="dp1"),
        pytest.param((2, 2), 1, id="dp2"),
        pytest.param((2, 2, 2, 2), 1, id="dp4"),
        pytest.param((1, 3), 1, id="dp2-uneven-increasing"),
        pytest.param((3, 1), 1, id="dp2-uneven-decreasing"),
        pytest.param((0, 4), 1, id="dp2-empty-first"),
        pytest.param((4, 0), 1, id="dp2-empty-last"),
        pytest.param((1, 2, 0, 5), 1, id="dp4-uneven-empty-middle"),
        pytest.param((0, 0, 0, 8), 1, id="dp4-single-source"),
        pytest.param((5, 0, 2, 1), 1, id="dp4-uneven-decreasing"),
        pytest.param((1, 3), 2, id="dp2-interleaved-subgroups"),
    ],
)
def test_balance_complete_rollout_groups(tmp_path, counts, num_groups):
    mp.spawn(
        _balance_groups_process,
        args=(counts, num_groups, str(tmp_path / "rendezvous")),
        nprocs=len(counts) * num_groups,
        join=True,
    )


def _source_weights_process(rank, redistribute, init_file):
    import torch

    from rlinf.algorithms.loss_scales import restore_source_dp_normalization

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        sources = (
            ([0, 0, 1] if rank == 0 else [0]) if redistribute else [rank] * (3, 1)[rank]
        )
        batch = {
            "extra:source_dp": torch.tensor(sources),
            "advantages": [torch.ones(i + 1) for i in range(len(sources))],
        }
        restore_source_dp_normalization(
            batch, {"folding_scale": ["group_level"]}, dist.group.WORLD, "cpu"
        )
        expected = (
            ([1.0, 1.0, 1.0 / 3] if rank == 0 else [3.0])
            if redistribute
            else [1.0] * len(sources)
        )
        for row, value in zip(batch["advantages"], expected, strict=True):
            torch.testing.assert_close(row, torch.full_like(row, value))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("redistribute", [False, True])
def test_source_dp_loss_weights_survive_uneven_redistribution(tmp_path, redistribute):
    mp.spawn(
        _source_weights_process,
        args=(redistribute, str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )
