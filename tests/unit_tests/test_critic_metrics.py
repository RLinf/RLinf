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

import torch

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    CRITIC_EXPLAINED_VARIANCE_STAT_KEYS,
    compute_critic_explained_variance_from_stats,
    compute_critic_explained_variance_stats,
    pop_critic_explained_variance_stats,
)


def _sum_stats(stats_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([stats[key] for stats in stats_list]).sum()
        for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS
    }


def _direct_explained_variance(
    returns: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    errors = returns - values
    return 1 - torch.var(errors) / torch.var(returns)


def test_critic_explained_variance_stats_recover_global_batch_value():
    returns = torch.tensor([0.0, 1.0, 10.0, 11.0])
    values = torch.tensor([0.0, 1.0, 9.0, 12.0])

    microbatch_stats = [
        compute_critic_explained_variance_stats(returns[:2], values[:2]),
        compute_critic_explained_variance_stats(returns[2:], values[2:]),
    ]
    microbatch_explained_variances = torch.stack(
        [
            compute_critic_explained_variance_from_stats(stats)
            for stats in microbatch_stats
        ]
    )

    global_from_summed_stats = compute_critic_explained_variance_from_stats(
        _sum_stats(microbatch_stats)
    )
    global_direct = _direct_explained_variance(returns, values)

    assert not torch.isclose(microbatch_explained_variances.mean(), global_direct)
    assert torch.isclose(global_from_summed_stats, global_direct)


def test_critic_explained_variance_stats_respect_loss_mask():
    returns = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    values = torch.tensor([[0.0, 2.0, 1.0], [9.0, 12.0, 12.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])

    stats = compute_critic_explained_variance_stats(
        returns=returns,
        values=values,
        loss_mask=mask,
    )

    expected = _direct_explained_variance(returns[mask], values[mask])
    actual = compute_critic_explained_variance_from_stats(stats)
    assert torch.isclose(actual, expected)


def test_critic_explained_variance_stats_nan_for_degenerate_inputs():
    values = torch.tensor([0.0, 1.0, 2.0])
    constant_returns = torch.tensor([1.0, 1.0, 1.0])
    constant_stats = compute_critic_explained_variance_stats(
        returns=constant_returns,
        values=values,
    )
    assert torch.isnan(compute_critic_explained_variance_from_stats(constant_stats))

    short_stats = compute_critic_explained_variance_stats(
        returns=torch.tensor([0.0, 1.0]),
        values=torch.tensor([0.0, 0.0]),
        loss_mask=torch.tensor([True, False]),
    )
    assert torch.isnan(compute_critic_explained_variance_from_stats(short_stats))


def test_ppo_critic_loss_returns_stats_that_match_full_batch_metric():
    returns = torch.tensor([0.0, 1.0, 10.0, 11.0])
    values = torch.tensor([0.0, 1.0, 9.0, 12.0])
    prev_values = torch.zeros_like(values)
    loss_mask = torch.ones_like(values, dtype=torch.bool)

    _, full_metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=100.0,
        huber_delta=10000,
        loss_mask=loss_mask,
    )
    microbatch_metrics = []
    for start, end in ((0, 2), (2, 4)):
        _, metrics = compute_ppo_critic_loss(
            values=values[start:end],
            returns=returns[start:end],
            prev_values=prev_values[start:end],
            value_clip=100.0,
            huber_delta=10000,
            loss_mask=loss_mask[start:end],
        )
        microbatch_metrics.append(metrics)

    summed_stats = _sum_stats(
        [
            {key: metrics[key] for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS}
            for metrics in microbatch_metrics
        ]
    )
    global_from_microbatches = compute_critic_explained_variance_from_stats(
        summed_stats
    )
    averaged_microbatch_metric = torch.stack(
        [
            compute_critic_explained_variance_from_stats(
                {key: metrics[key] for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS}
            )
            for metrics in microbatch_metrics
        ]
    ).mean()
    full_batch_metric = compute_critic_explained_variance_from_stats(
        {key: full_metrics[key] for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS}
    )

    assert all(key in full_metrics for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS)
    assert CRITIC_EXPLAINED_VARIANCE_KEY not in full_metrics
    assert not torch.isclose(
        averaged_microbatch_metric,
        full_batch_metric,
    )
    assert torch.isclose(
        global_from_microbatches,
        full_batch_metric,
    )


def test_pop_critic_explained_variance_stats_removes_hidden_and_public_keys():
    metrics = {
        "critic/value_loss": [torch.tensor(1.0), torch.tensor(3.0)],
        CRITIC_EXPLAINED_VARIANCE_KEY: [torch.tensor(1.0), torch.tensor(-3.0)],
    }
    for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS:
        metrics[key] = [torch.tensor(1.0), torch.tensor(2.0)]

    stats = pop_critic_explained_variance_stats(metrics)

    assert CRITIC_EXPLAINED_VARIANCE_KEY not in metrics
    assert all(key not in metrics for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            metrics["critic/value_loss"],
            [torch.tensor(1.0), torch.tensor(3.0)],
            strict=True,
        )
    )
    assert all(torch.equal(value, torch.tensor(3.0)) for value in stats.values())
