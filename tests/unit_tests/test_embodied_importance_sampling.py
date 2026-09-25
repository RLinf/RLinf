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

import inspect
import math
from contextlib import nullcontext

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import policy_loss
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class _ReplayModel:
    def __init__(self) -> None:
        self.batch_sizes = []
        self.kwargs = []

    def __call__(self, *, forward_inputs, **kwargs):
        expected = forward_inputs["expected_logprobs"]
        self.batch_sizes.append(expected.shape[0])
        self.kwargs.append(kwargs)
        return {"logprobs": expected + 0.25}


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


@pytest.mark.parametrize("logprob_type", ["token_level", "action_level", "chunk_level"])
def test_embodied_importance_sampling_uses_loss_logprob_granularity(logprob_type):
    rollout_logprobs = torch.zeros(2, 1, 2, dtype=torch.float32)
    per_element_gap = math.log(2.0)
    if logprob_type in {"action_level", "chunk_level"}:
        per_element_gap /= 2.0
    recomputed_logprobs = torch.full_like(rollout_logprobs, per_element_gap)
    advantages = (
        torch.ones(2, 1, dtype=torch.float32)
        if logprob_type == "token_level"
        else torch.ones(2, dtype=torch.float32)
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
        advantages=advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        importance_sampling_clip=1.5,
    )

    assert loss.item() == pytest.approx(-1.5)
    assert metrics["actor/ratio"] == pytest.approx(1.0)
    assert metrics["actor/importance_sampling_weight"] == pytest.approx(2.0)
    assert metrics["actor/importance_sampling_weight_max"] == pytest.approx(2.0)
    assert metrics["actor/importance_sampling_clip_fraction"] == pytest.approx(1.0)
    assert metrics["actor/recomputed_logprob_abs_diff"] == pytest.approx(math.log(2.0))


def test_embodied_importance_sampling_metrics_broadcast_loss_mask():
    rollout_logprobs = torch.zeros(1, 2, 2, dtype=torch.float32)
    recomputed_logprobs = torch.full_like(rollout_logprobs, math.log(2.0))

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
        importance_sampling_clip=2.0,
    )

    assert metrics["actor/importance_sampling_weight"] == pytest.approx(2.0)
    assert metrics["actor/importance_sampling_weight_max"] == pytest.approx(2.0)


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
