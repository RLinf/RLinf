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

"""CPU coverage for opt-in ART diagnostics, bounded capture, and probe sampling."""

import asyncio
import inspect
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.run_art_ablation import experiment_matrix, summarize_probes
from rlinf.envs.sim.libero.art_states import ARTStateReservoir, select_art_states
from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.utils import warmup_optimizer_state
from rlinf.workers.actor.art_diagnostics import ARTParameterProbe
from rlinf.workers.actor.fsdp_art_policy_worker import (
    EmbodiedARTFSDPPolicy,
    _art_mask_count_metrics,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_reservoir_limits_capture_and_keeps_late_states():
    bank = ARTStateReservoir(2, 4, np.random.default_rng(10))
    captured = []

    def capture(ids):
        captured.extend(ids)
        return [f"snapshot-{index}" for index in ids]

    for step in range(100):
        bank.capture(
            [{"env_id": index, "elapsed_steps": step} for index in range(2)], capture
        )
    assert len(bank.records()) == 8
    assert len(captured) < 100  # Rejected states never cross the simulator boundary.
    assert max(record["elapsed_steps"] for record in bank.records()) > 50
    assert bank.seen.tolist() == [100, 100]
    assert [len(states) for states in bank.states] == [4, 4]


def test_reservoir_does_not_duration_weight_trajectories():
    bank = ARTStateReservoir(2, 4, np.random.default_rng(12))
    for step in range(100):
        bank.capture([{"env_id": 0, "elapsed_steps": step}], lambda ids: ids)
        if step < 10:
            bank.capture([{"env_id": 1, "elapsed_steps": step}], lambda ids: ids)
    assert [len(states) for states in bank.states] == [4, 4]


@pytest.mark.parametrize(
    "sampler", ["legacy", "uniform", "success_mixture", "early_failure"]
)
def test_task_matched_sampling_and_outcome_preferences(sampler):
    states = [
        {"task_id": task, "elapsed_steps": step, "source_success": success}
        for task in (0, 1)
        for step in (0, 25, 50, 75)
        for success in (False, True)
    ]
    tasks = np.repeat([0, 1], 4)
    selected = select_art_states(
        states, tasks, np.random.default_rng(7), sampler, 100, success_fraction=0.75
    )
    assert [state["task_id"] for state in selected] == tasks.tolist()
    if sampler == "success_mixture":
        assert sum(state["source_success"] for state in selected) == 6
    if sampler == "early_failure":
        assert all(
            state["source_success"] or state["elapsed_steps"] <= 50
            for state in selected
        )


def test_success_sampler_falls_back_without_success_and_handles_small_pools():
    states = [{"task_id": 0, "elapsed_steps": 0, "source_success": False}]
    selected = select_art_states(
        states, np.zeros(8), np.random.default_rng(8), "success_mixture", 100
    )
    assert len(selected) == 8 and all(state is states[0] for state in selected)


def test_legacy_sampling_preserves_original_rng_and_temporal_strata():
    states = [{"task_id": 0, "elapsed_steps": step} for step in range(100)]
    expected_rng, actual_rng = np.random.default_rng(9), np.random.default_rng(9)
    expected = [
        states[expected_rng.choice(stratum)]
        for stratum in np.array_split(np.arange(100), 8)
    ]
    actual = select_art_states(states, np.zeros(8), actual_rng, "legacy", 100)
    assert actual == expected
    assert expected_rng.integers(10000) == actual_rng.integers(10000)


@pytest.mark.parametrize("master", [False, True])
def test_parameter_probe_separates_bf16_rounding_from_fp32_accumulation(master):
    model = torch.nn.Module()
    model.register_parameter(
        "expert", torch.nn.Parameter(torch.full((16,), 0.001, dtype=torch.bfloat16))
    )
    model.register_parameter("projection", torch.nn.Parameter(torch.ones(2)))
    optimizer_type = FP32MasterAdamW if master else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=5e-7, weight_decay=0)
    warmup_optimizer_state(optimizer)
    probe = ARTParameterProbe(model, optimizer)
    for _ in range(20):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.001)
        optimizer.step()
    stats = probe.metrics("param", torch.device("cpu"))
    assert stats["param_bf16_numel"] == 16
    assert stats["param_fp32_update_rms"] > 0
    if master:
        assert stats["param_master_update_rms"] > 0
        assert stats["param_bf16_changed_frac"] == 1
    else:
        assert stats["param_bf16_changed_frac"] == 0


def test_mask_counts_ignore_empty_microbatches():
    empty = torch.zeros(4)
    active = torch.tensor([2.0, 1.0, 1.0, 0.04])
    result = _art_mask_count_metrics(empty + active, "actor")
    assert result["actor/mask_frac_global"] == 0.5
    assert result["actor/movement_score_active"] == pytest.approx(0.02)
    assert _art_mask_count_metrics(empty, "actor")["actor/mask_frac_global"] == 0


class _VelocityPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(4, 4, 3))

    def forward(self, forward_type, forward_inputs, nft_inputs, compute_values):
        ids = forward_inputs["observation/state"][:, 0].long()
        return {"v_theta": nft_inputs["x_t"] + self.offset[ids]}


@pytest.fixture
def cpu_art_backend(monkeypatch):
    # Replace GPU/distributed boundaries only; execute the real worker kernels,
    # loss, optimizer, scheduler, and reference construction below.
    platform = SimpleNamespace(
        current_device=lambda: torch.device("cpu"),
        synchronize=lambda: None,
        ipc_collect=lambda: None,
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(Worker, "torch_platform", platform)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return platform


def _cpu_actor(tmp_path, platform, probe_only=False):
    actor = object.__new__(EmbodiedARTFSDPPolicy)
    actor._rank, actor._world_size, actor.version = 0, 1, 0
    actor.device, actor.torch_platform = torch.device("cpu"), platform
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": 2,
                "art_first_chunk_only": not probe_only,
                "art_state_sampling": probe_only,
                "art_updates_per_rollout": 2,
                "art_timestep_fraction": 1.0,
                "art_probe_only": probe_only,
                "art_reference_source": "actor",
                "art_replay_diagnostics": True,
                "art_replay_batch_sizes": [1, 2],
                "art_parameter_diagnostics": True,
            },
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "global_batch_size": 2,
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 2,
                    "action_dim": 2,
                },
            },
            "env": {"train": {"max_episode_steps": 100}},
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "20261028_test",
                }
            },
        }
    )
    states = torch.arange(4.0).view(1, 1, 4, 1, 1).expand(1, 4, 4, 4, 3).clone()
    inputs = {
        "art_endpoint": torch.arange(16.0).reshape(1, 4, 2, 2),
        "art_states": states,
        "art_old_velocity": states + 10,
        "art_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25]).expand(1, 4, 4),
        "art_sigmas_next": torch.tensor([0.75, 0.5, 0.25, 0.0]).expand(1, 4, 4),
        "observation/state": torch.arange(4).view(1, 4, 1),
        "art_branch_valid": torch.ones(1, 4, dtype=torch.bool),
        "art_branch_state_error": torch.zeros(1, 4),
        "art_branch_image_error": torch.zeros(1, 4),
        "art_state_id": torch.tensor([[0, 0, 1, 1]]),
        "art_state_steps": torch.tensor([[20, 20, 40, 40]]),
        "art_source_success": torch.tensor([[1, 1, 0, 0]]),
        "art_source_trajectory_id": torch.tensor([[3, 3, 7, 7]]),
        "art_source_reset_state_id": torch.tensor([[5, 5, 9, 9]]),
    }
    actor.rollout_batch = {
        "forward_inputs": inputs,
        "rewards": torch.tensor([[[1.0, 0], [0, 0], [0, 0], [0, 1.0]]]),
        "dones": torch.zeros(2, 4, 2, dtype=torch.bool),
    }
    actor.model = _VelocityPolicy()
    actor.optimizer = torch.optim.AdamW(
        actor.model.parameters(), lr=0.01, weight_decay=0
    )
    actor.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor.optimizer, lambda step: 1
    )
    actor.amp_context = nullcontext()
    actor.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
    actor.is_weight_offloaded, actor.is_optimizer_offloaded = False, False
    actor.before_micro_batch = lambda *args, **kwargs: nullcontext()

    def optimizer_step():
        norm = torch.nn.utils.clip_grad_norm_(actor.model.parameters(), 1)
        actor.optimizer.step()
        return norm.item(), [0.01]

    actor.optimizer_step = optimizer_step
    return actor


def test_actor_reference_is_frozen_across_two_real_updates(tmp_path, cpu_art_backend):
    actor = _cpu_actor(tmp_path, cpu_art_backend)
    metrics = inspect.unwrap(EmbodiedARTFSDPPolicy.run_training)(actor)
    inputs = actor.rollout_batch["forward_inputs"]
    torch.testing.assert_close(inputs["art_old_velocity"], inputs["art_states"])
    assert metrics["actor/replay_mse"] == pytest.approx(100)
    assert metrics["actor/reference_pre_mse"] == 0
    assert metrics["actor/u1/mask_frac_global"] == 0
    assert metrics["actor/update_u1_exec_rms"] > 0
    assert metrics["actor/update_u2_unexec_rms"] == 0
    assert metrics["actor/param_u2_fp32_changed_frac"] > 0
    assert metrics["actor/batch_mb2_mse"] == 0
    assert actor.lr_scheduler.last_epoch == 2


def test_probe_logs_source_outcomes_without_optimizer_or_backward(
    tmp_path, cpu_art_backend
):
    actor = _cpu_actor(tmp_path, cpu_art_backend, probe_only=True)
    before = actor.model.offset.detach().clone()
    metrics = inspect.unwrap(EmbodiedARTFSDPPolicy.run_training)(actor)
    assert metrics["actor/probe_only"] == 1
    torch.testing.assert_close(actor.model.offset, before, rtol=0, atol=0)
    assert actor.model.offset.grad is None
    assert not actor.optimizer.state
    assert actor.lr_scheduler.last_epoch == 0
    path = next(tmp_path.rglob("*_art_probe_rank0.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.parent.name == "art_probe"
    assert len(rows) == 2
    assert rows[0]["art_source_success"] == 1 and rows[1]["art_source_success"] == 0
    assert rows[0]["remaining_steps"] == 80
    assert rows[1]["returns"] == [0, 1]
    summarize_probes(tmp_path)


def test_experiment_matrices_preserve_controls_and_matched_seeds():
    diagnose = experiment_matrix("diagnose", None, 42, [42, 43])
    assert {(arm["master"], arm["reference"]) for arm in diagnose} == {
        (False, "rollout"),
        (True, "rollout"),
        (False, "actor"),
        (True, "actor"),
    }
    assert len(experiment_matrix("probe", None, 42, [])) == 4
    confirm = experiment_matrix("confirm", "uniform", 42, [42, 43])
    assert [(arm["seed"], arm["sampler"]) for arm in confirm] == [
        (42, "legacy"),
        (42, "uniform"),
        (43, "legacy"),
        (43, "uniform"),
    ]
    with pytest.raises(ValueError, match="probe results"):
        experiment_matrix("train", None, 42, [])


def test_paired_eval_seeds_restore_training_rng():
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank, worker.device, worker.torch_device_type = (
        0,
        torch.device("cpu"),
        "cpu",
    )

    async def evaluate(*args):
        return torch.randn(4)

    worker._evaluate = evaluate
    torch.manual_seed(97)
    before = torch.random.get_rng_state()
    first = asyncio.run(
        inspect.unwrap(MultiStepRolloutWorker.evaluate)(worker, None, None, seed=9)
    )
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    second = asyncio.run(
        inspect.unwrap(MultiStepRolloutWorker.evaluate)(worker, None, None, seed=9)
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    other = asyncio.run(
        inspect.unwrap(MultiStepRolloutWorker.evaluate)(worker, None, None, seed=10)
    )
    assert not torch.equal(first, other)


def test_repeated_evaluation_counts_episodes_and_pairs_noise_seeds():
    class Endpoint:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def evaluate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(wait=lambda: self.result)

    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"eval_repeats": 3, "eval_seed": 123}})
    runner.env_channel, runner.rollout_channel = None, None
    runner.env = Endpoint([{"success_once": torch.tensor([0.0, 1.0])}])
    runner.rollout = Endpoint(None)
    metrics = runner.evaluate()
    assert len(runner.env.calls) == 3
    assert [call["seed"] for call in runner.rollout.calls] == [123, 124, 125]
    assert metrics["success_once"] == 0.5
    assert metrics["num_trajectories"] == 6
