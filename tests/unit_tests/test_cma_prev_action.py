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

from typing import Any, cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    GT_PREFIX_LEARNING_EXCLUDED_KEY,
    ChunkStepResult,
    EmbodiedRolloutResult,
    convert_trajectories_to_batch,
)
from rlinf.envs.habitat.gt_prefix import (
    HABITAT_CURRENT_STEP_KEY,
    HABITAT_GT_ACTION_VALID_KEY,
    HABITAT_GT_CURRENT_ACTION_KEY,
    build_gt_prefix_metadata,
)
from rlinf.models.embodiment.cma.cma_action_model import CMAConfig, CMAPolicy
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class _DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = type("StateEncoder", (), {"num_recurrent_layers": 1})()
        self.second_state_encoder = type(
            "SecondStateEncoder", (), {"num_recurrent_layers": 1}
        )()
        self.calls = []

    def forward(self, env_obs, pre_rnn_states, prev_actions, step_masks):
        self.calls.append(
            {
                "pre_rnn_states": pre_rnn_states.clone(),
                "prev_actions": prev_actions.clone(),
                "step_masks": step_masks.clone(),
            }
        )
        batch_size = pre_rnn_states.shape[0]
        hidden_size = pre_rnn_states.shape[-1]
        features = torch.full(
            (batch_size, hidden_size),
            fill_value=float(len(self.calls)),
            device=pre_rnn_states.device,
        )
        next_states = pre_rnn_states + len(self.calls)
        return features, next_states

    __call__ = forward


class _DummyDistribution:
    def __init__(self, sampled_action, mode_action):
        self.sampled_action = sampled_action
        self.mode_action = mode_action
        self.log_prob_actions = []

    def sample(self):
        return self.sampled_action.clone()

    def mode(self):
        return self.mode_action.clone()

    def log_probs(self, action):
        self.log_prob_actions.append(action.clone())
        return action.float() + 0.5


def _build_test_policy(sampled_actions, *, use_gt_prefix=True, gt_prefix_length=1):
    policy = CMAPolicy.__new__(CMAPolicy)
    nn.Module.__init__(policy)
    policy.cfg = CMAConfig(
        hidden_size=4,
        num_action_classes=4,
        use_gt_prefix=use_gt_prefix,
        gt_prefix_length=gt_prefix_length,
    )
    policy.rnn_states = None
    policy.prev_actions = None
    policy.prev_episode_id = None
    policy.prev_mode = None
    policy.action_map = {
        0: "stop",
        1: "move_forward",
        2: "turn_left",
        3: "turn_right",
    }
    policy.register_parameter("_device_anchor", nn.Parameter(torch.zeros(1)))

    dummy_net = _DummyNet()
    distributions = []

    class _DummyActionDistributionFactory:
        def __call__(self, features):
            action = sampled_actions[len(distributions)]
            distribution = _DummyDistribution(sampled_action=action, mode_action=action)
            distributions.append(distribution)
            return distribution

    policy.policy = cast(
        Any,
        type(
            "PolicyStub",
            (),
            {
                "net": dummy_net,
                "action_distribution": _DummyActionDistributionFactory(),
            },
        )(),
    )
    return policy, dummy_net, distributions


def _build_env_obs(*, episode_id, current_step, gt_action_valid, gt_current_action):
    return {
        "main_images": torch.zeros(1, 4, 4, 3),
        "extra_view_images": torch.zeros(1, 1, 4, 4, 1),
        "wrist_images": [[1, 2, 3]],
        "states": torch.tensor([episode_id], dtype=torch.int64),
        HABITAT_CURRENT_STEP_KEY: torch.tensor([current_step], dtype=torch.int64),
        HABITAT_GT_ACTION_VALID_KEY: torch.tensor([gt_action_valid], dtype=torch.bool),
        HABITAT_GT_CURRENT_ACTION_KEY: torch.tensor(
            [gt_current_action], dtype=torch.int64
        ),
    }


def _build_env_obs_from_metadata(*, episode_id, metadata):
    return {
        "main_images": torch.zeros(1, 4, 4, 3),
        "extra_view_images": torch.zeros(1, 1, 4, 4, 1),
        "wrist_images": [[1, 2, 3]],
        "states": torch.tensor([episode_id], dtype=torch.int64),
        **metadata,
    }


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


def test_cma_gt_prefix_length_zero_keeps_model_action_even_with_valid_gt_metadata():
    policy, dummy_net, distributions = _build_test_policy(
        sampled_actions=[torch.tensor([[3]], dtype=torch.long)],
        gt_prefix_length=0,
    )

    action, result = policy.predict_action_batch(
        _build_env_obs(
            episode_id=7,
            current_step=1,
            gt_action_valid=True,
            gt_current_action=1,
        ),
        mode="train",
    )

    assert policy.prev_actions is not None
    assert torch.equal(action, torch.tensor([[3]], dtype=torch.long))
    assert torch.equal(policy.prev_actions, torch.tensor([[3]], dtype=torch.long))
    assert torch.equal(result["forward_inputs"]["action"], torch.tensor([[3]]))
    assert torch.equal(
        result["forward_inputs"][GT_PREFIX_LEARNING_EXCLUDED_KEY],
        torch.tensor([[False]]),
    )
    assert torch.equal(dummy_net.calls[0]["prev_actions"], torch.tensor([[0]]))
    assert torch.equal(distributions[0].log_prob_actions[0], torch.tensor([[3]]))


def test_cma_prev_action_uses_gt_executed_action_on_first_gt_prefix_step():
    policy, dummy_net, distributions = _build_test_policy(
        sampled_actions=[torch.tensor([[3]], dtype=torch.long)],
        gt_prefix_length=1,
    )

    action, result = policy.predict_action_batch(
        _build_env_obs(
            episode_id=7,
            current_step=1,
            gt_action_valid=True,
            gt_current_action=1,
        ),
        mode="train",
    )

    assert policy.prev_actions is not None
    assert torch.equal(action, torch.tensor([[1]], dtype=torch.long))
    assert torch.equal(policy.prev_actions, torch.tensor([[1]], dtype=torch.long))
    assert torch.equal(result["forward_inputs"]["action"], torch.tensor([[1]]))
    assert torch.equal(
        result["forward_inputs"][GT_PREFIX_LEARNING_EXCLUDED_KEY],
        torch.tensor([[True]]),
    )
    assert torch.equal(dummy_net.calls[0]["prev_actions"], torch.tensor([[0]]))
    assert torch.equal(dummy_net.calls[0]["step_masks"], torch.tensor([[False]]))
    assert torch.equal(distributions[0].log_prob_actions[0], torch.tensor([[1]]))


def test_cma_recurrent_boundary_keeps_state_and_prev_action_across_gt_to_model_boundary():
    policy, dummy_net, distributions = _build_test_policy(
        sampled_actions=[
            torch.tensor([[2]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
        ],
        gt_prefix_length=1,
    )

    first_action, first_result = policy.predict_action_batch(
        _build_env_obs(
            episode_id=5,
            current_step=1,
            gt_action_valid=True,
            gt_current_action=1,
        ),
        mode="train",
    )
    second_action, second_result = policy.predict_action_batch(
        _build_env_obs(
            episode_id=5,
            current_step=2,
            gt_action_valid=True,
            gt_current_action=2,
        ),
        mode="train",
    )

    assert policy.prev_actions is not None
    assert torch.equal(first_action, torch.tensor([[1]], dtype=torch.long))
    assert torch.equal(second_action, torch.tensor([[3]], dtype=torch.long))
    assert torch.equal(
        first_result["forward_inputs"][GT_PREFIX_LEARNING_EXCLUDED_KEY],
        torch.tensor([[True]]),
    )
    assert torch.equal(
        second_result["forward_inputs"][GT_PREFIX_LEARNING_EXCLUDED_KEY],
        torch.tensor([[False]]),
    )
    assert torch.equal(dummy_net.calls[1]["prev_actions"], torch.tensor([[1]]))
    assert torch.equal(dummy_net.calls[1]["step_masks"], torch.tensor([[True]]))
    assert torch.equal(
        second_result["forward_inputs"]["pre_rnn_states"],
        first_result["forward_inputs"]["pre_rnn_states"] + 1,
    )
    assert torch.equal(distributions[1].log_prob_actions[0], torch.tensor([[3]]))
    assert torch.equal(policy.prev_actions, torch.tensor([[3]], dtype=torch.long))


def test_cma_prev_action_resets_only_on_episode_reset():
    policy, dummy_net, _ = _build_test_policy(
        sampled_actions=[
            torch.tensor([[2]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
        ],
        gt_prefix_length=1,
    )

    policy.predict_action_batch(
        _build_env_obs(
            episode_id=9,
            current_step=1,
            gt_action_valid=True,
            gt_current_action=1,
        ),
        mode="train",
    )
    policy.predict_action_batch(
        _build_env_obs(
            episode_id=10,
            current_step=1,
            gt_action_valid=True,
            gt_current_action=2,
        ),
        mode="train",
    )

    assert torch.equal(dummy_net.calls[1]["prev_actions"], torch.tensor([[0]]))
    assert torch.equal(dummy_net.calls[1]["step_masks"], torch.tensor([[False]]))
    assert torch.equal(dummy_net.calls[1]["pre_rnn_states"], torch.zeros(1, 2, 4))


def test_cma_gt_prefix_integration_preserves_trajectory_and_actor_mask():
    policy, _, _ = _build_test_policy(
        sampled_actions=[
            torch.tensor([[2]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
        ],
        gt_prefix_length=1,
    )

    first_step_metadata = build_gt_prefix_metadata(
        episode_ids=[5],
        elapsed_steps=[0],
        episode_gt_action_sequences=[(1, 2)],
        valid_action_ids={0, 1, 2, 3},
    )
    second_step_metadata = build_gt_prefix_metadata(
        episode_ids=[5],
        elapsed_steps=[1],
        episode_gt_action_sequences=[(1, 2)],
        valid_action_ids={0, 1, 2, 3},
    )

    first_action, first_result = policy.predict_action_batch(
        _build_env_obs_from_metadata(episode_id=5, metadata=first_step_metadata),
        mode="train",
    )
    second_action, second_result = policy.predict_action_batch(
        _build_env_obs_from_metadata(episode_id=5, metadata=second_step_metadata),
        mode="train",
    )

    rollout = EmbodiedRolloutResult(max_episode_length=2)
    rollout.append_step_result(
        ChunkStepResult(
            actions=first_action,
            prev_logprobs=first_result["prev_logprobs"],
            prev_values=first_result["prev_values"],
            dones=torch.tensor([[False]], dtype=torch.bool),
            forward_inputs=first_result["forward_inputs"],
        )
    )
    rollout.append_step_result(
        ChunkStepResult(
            actions=second_action,
            prev_logprobs=second_result["prev_logprobs"],
            prev_values=second_result["prev_values"],
            dones=torch.tensor([[False]], dtype=torch.bool),
            forward_inputs=second_result["forward_inputs"],
        )
    )
    rollout.dones.append(torch.tensor([[False]], dtype=torch.bool))

    trajectory = rollout.to_trajectory()
    rollout_batch = convert_trajectories_to_batch([trajectory])
    processed = _build_actor()._process_received_rollout_batch(rollout_batch)

    assert torch.equal(first_action, torch.tensor([[1]], dtype=torch.long))
    assert torch.equal(second_action, torch.tensor([[3]], dtype=torch.long))
    assert torch.equal(
        trajectory.actions,
        torch.tensor([[[1]], [[3]]], dtype=torch.long),
    )
    assert torch.equal(
        trajectory.forward_inputs[GT_PREFIX_LEARNING_EXCLUDED_KEY],
        torch.tensor([[[True]], [[False]]], dtype=torch.bool),
    )
    assert torch.equal(
        processed["loss_mask"],
        torch.tensor([[[False]], [[True]]], dtype=torch.bool),
    )
    assert torch.equal(
        processed["loss_mask_sum"],
        torch.tensor([[[1]], [[1]]], dtype=torch.int64),
    )
