import pytest
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.utils.metric_utils import (
    compute_pipeline_actor_split_num,
    compute_pipeline_expected_actor_recv_num,
    compute_pipeline_micro_batch_env_size,
    flatten_embodied_rollout_batch_for_train,
)
from rlinf.workers.actor.fsdp_actor_worker import _merge_pipeline_train_batches


def test_merge_pipeline_train_batches_recursive():
    batch_a = {
        "advantages": torch.arange(2, dtype=torch.float32).reshape(2, 1),
        "forward_inputs": {
            "action": torch.arange(6, dtype=torch.float32).reshape(2, 3)
        },
    }
    batch_b = {
        "advantages": torch.arange(2, 4, dtype=torch.float32).reshape(2, 1),
        "forward_inputs": {
            "action": torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        },
    }

    merged = _merge_pipeline_train_batches([batch_a, batch_b])

    assert merged["advantages"].shape == (4, 1)
    assert merged["forward_inputs"]["action"].shape == (4, 3)
    assert torch.equal(merged["advantages"][:2], batch_a["advantages"])
    assert torch.equal(merged["advantages"][2:], batch_b["advantages"])


def test_flatten_embodied_rollout_batch_for_train_drops_bootstrap_step():
    rollout_batch = {
        "advantages": torch.tensor(
            [[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.float32
        ),
        "prev_logprobs": torch.tensor(
            [[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float32
        ),
        "returns": torch.tensor(
            [[[5.0], [6.0]], [[7.0], [8.0]]], dtype=torch.float32
        ),
        "dones": torch.zeros((3, 2, 1), dtype=torch.bool),
        "prev_values": torch.arange(6, dtype=torch.float32).reshape(3, 2, 1),
        "forward_inputs": {
            "action": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        },
    }

    flattened = flatten_embodied_rollout_batch_for_train(
        rollout_batch,
        torch.arange(4),
    )

    assert flattened["advantages"].shape == (4, 1)
    assert flattened["prev_logprobs"].shape == (4, 1)
    assert flattened["returns"].shape == (4, 1)
    assert flattened["dones"].shape == (4, 1)
    assert flattened["prev_values"].shape == (4, 1)
    assert flattened["forward_inputs"]["action"].shape == (4, 3)


def test_compute_pipeline_micro_batch_env_size():
    assert (
        compute_pipeline_micro_batch_env_size(
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )
        == 2
    )


def test_compute_pipeline_micro_batch_env_size_with_per_epoch_flush():
    assert (
        compute_pipeline_micro_batch_env_size(
            micro_batch_size=32,
            rollout_epoch=2,
            n_train_chunk_steps=16,
            rollout_epochs_per_flush=1,
        )
        == 2
    )


def test_compute_pipeline_micro_batch_env_size_requires_group_aligned_split():
    with pytest.raises(AssertionError):
        compute_pipeline_micro_batch_env_size(
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
            group_size=3,
        )


def test_compute_pipeline_actor_split_num_and_expected_recv_num():
    assert (
        compute_pipeline_actor_split_num(
            train_num_envs_per_stage=320,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )
        == 160
    )
    assert (
        compute_pipeline_expected_actor_recv_num(
            total_num_envs=320,
            actor_world_size=2,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )
        == 80
    )


def test_compute_pipeline_actor_split_num_supports_group_preserving_splits():
    assert (
        compute_pipeline_actor_split_num(
            train_num_envs_per_stage=32,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=4,
            group_size=4,
        )
        == 4
    )


def test_compute_pipeline_actor_split_num_with_per_epoch_flush():
    assert (
        compute_pipeline_actor_split_num(
            train_num_envs_per_stage=20,
            micro_batch_size=32,
            rollout_epoch=2,
            n_train_chunk_steps=16,
            rollout_epochs_per_flush=1,
        )
        == 10
    )


def test_calculate_adv_and_returns_embodied_grpo_dynamic_single_turn_trajs():
    adv_and_ret = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo_dynamic",
        rewards=torch.tensor([[[1.0], [3.0], [2.0], [6.0]]], dtype=torch.float32),
        dones=torch.tensor(
            [[[False], [False], [False], [False]], [[True], [True], [True], [True]]]
        ),
        values=None,
        gamma=1.0,
        gae_lambda=1.0,
        group_size=2,
        reward_type="chunk_level",
        loss_mask=torch.ones((1, 4, 1), dtype=torch.bool),
        loss_mask_sum=None,
        idx_to_traj=[0, 1, 2, 3],
        normalize_advantages=False,
    )

    advantages = adv_and_ret["advantages"].squeeze(-1).squeeze(0)
    assert adv_and_ret["returns"] is None
    assert advantages.shape == (4,)
    assert torch.allclose(advantages[:2], torch.tensor([-0.7071, 0.7071]), atol=1e-3)
    assert torch.allclose(advantages[2:], torch.tensor([-0.7071, 0.7071]), atol=1e-3)
