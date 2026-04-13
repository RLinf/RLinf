# Copyright 2025 The RLinf Authors.
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

from typing import TypedDict

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import compute_grpo_advantages
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.data.embodied_io_struct import GT_PREFIX_LEARNING_EXCLUDED_KEY
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class _RolloutBatch(TypedDict):
    dones: torch.Tensor
    forward_inputs: dict[str, torch.Tensor]


def _build_actor(*, reward_type: str = "action_level") -> EmbodiedFSDPActor:
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "rollout_epoch": 1,
                "reward_type": reward_type,
                "filter_rewards": False,
            },
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "max_episode_steps": 4,
                }
            },
        }
    )
    return actor


def _build_rollout_batch(
    *, dones: torch.Tensor, gt_prefix_learning_excluded: torch.Tensor
) -> _RolloutBatch:
    return {
        "dones": dones,
        "forward_inputs": {
            GT_PREFIX_LEARNING_EXCLUDED_KEY: gt_prefix_learning_excluded,
        },
    }


def test_actor_loss_mask_and_gt_prefix_masks_only_gt_prefix_steps():
    actor = _build_actor()
    rollout_batch = _build_rollout_batch(
        dones=torch.tensor(
            [
                [[False], [False]],
                [[False], [False]],
                [[False], [False]],
            ],
            dtype=torch.bool,
        ),
        gt_prefix_learning_excluded=torch.tensor(
            [
                [[True], [False]],
                [[False], [False]],
            ],
            dtype=torch.bool,
        ),
    )

    processed = actor._process_received_rollout_batch(rollout_batch)

    assert torch.equal(
        processed["loss_mask"],
        torch.tensor(
            [
                [[False], [True]],
                [[True], [True]],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        processed["loss_mask_sum"],
        torch.tensor(
            [
                [[1], [2]],
                [[1], [2]],
            ],
            dtype=torch.int64,
        ),
    )


def test_actor_loss_mask_and_gt_prefix_keeps_post_done_steps_invalid():
    actor = _build_actor()
    rollout_batch = _build_rollout_batch(
        dones=torch.tensor(
            [
                [[False]],
                [[True]],
                [[False]],
            ],
            dtype=torch.bool,
        ),
        gt_prefix_learning_excluded=torch.tensor(
            [
                [[False]],
                [[True]],
            ],
            dtype=torch.bool,
        ),
    )

    processed = actor._process_received_rollout_batch(rollout_batch)

    assert torch.equal(
        processed["loss_mask"],
        torch.tensor(
            [
                [[True]],
                [[False]],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        processed["loss_mask_sum"],
        torch.tensor(
            [
                [[1]],
                [[1]],
            ],
            dtype=torch.int64,
        ),
    )


def test_actor_loss_mask_and_gt_prefix_all_prefix_episode_stays_finite():
    actor = _build_actor()
    rewards = torch.tensor(
        [
            [[1.0], [3.0]],
            [[2.0], [4.0]],
        ],
        dtype=torch.float32,
    )
    rollout_batch = _build_rollout_batch(
        dones=torch.tensor(
            [
                [[False], [False]],
                [[False], [False]],
                [[False], [False]],
            ],
            dtype=torch.bool,
        ),
        gt_prefix_learning_excluded=torch.ones((2, 2, 1), dtype=torch.bool),
    )

    processed = actor._process_received_rollout_batch(rollout_batch)

    assert not processed["loss_mask"].any()
    assert torch.equal(
        processed["loss_mask_sum"],
        torch.zeros((2, 2, 1), dtype=torch.int64),
    )

    scores = rewards.transpose(1, 2).reshape(-1, 2).sum(dim=0).reshape(1, 2)
    advantages, _ = compute_grpo_advantages(
        rewards=scores,
        loss_mask=processed["loss_mask"].reshape(-1, 2),
        group_size=2,
    )
    flat_loss_mask = processed["loss_mask"].reshape(-1, 1)
    flat_loss_mask_sum = processed["loss_mask_sum"].reshape(-1, 1)
    flat_advantages = advantages.reshape(-1, 1)

    loss, metrics = compute_ppo_actor_loss(
        logprobs=torch.zeros_like(flat_advantages),
        old_logprobs=torch.zeros_like(flat_advantages),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=flat_advantages,
        loss_mask=flat_loss_mask,
        loss_mask_sum=flat_loss_mask_sum,
        max_episode_steps=4,
    )

    assert torch.equal(flat_advantages, torch.zeros_like(flat_advantages))
    assert torch.isfinite(loss)
    for value in metrics.values():
        assert torch.isfinite(value)
