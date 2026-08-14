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

from dataclasses import replace

import numpy as np
import torch

from rlinf.data.schema.embodied_types import (
    DummyPolicyInput,
    EmbodiedRolloutResult,
    EnvResult,
    PolicyCompletion,
    PolicyInput,
    PolicyOutput,
    TrajectoryKey,
    TrajectorySource,
    merge_episode_data,
    merge_policy_inputs,
    split_episode_data,
    split_policy_input,
)
from rlinf.scheduler.channel.trajectory_channel.data import PolicyStep
from rlinf.scheduler.cluster.utils import (
    TensorPlaceholder,
    pack_dataclass_tensors,
    unpack_dataclass_tensors,
)


def test_policy_output_supports_tensor_transport_skeleton():
    output = PolicyOutput(actions=torch.ones(2, 3))

    skeleton = replace(output, actions=None)
    restored = replace(skeleton, actions=output.actions)

    assert skeleton.actions is None
    assert torch.equal(restored.actions, output.actions)


def test_transport_results_store_contiguous_cpu_tensors():
    non_contiguous = torch.arange(12).reshape(3, 4).T

    env_result = EnvResult(rewards=non_contiguous)
    rollout_result = EmbodiedRolloutResult(
        actions=non_contiguous,
        forward_inputs={"states": non_contiguous},
    )

    assert env_result.rewards.device.type == "cpu"
    assert env_result.rewards.is_contiguous()
    assert rollout_result.actions.device.type == "cpu"
    assert rollout_result.actions.is_contiguous()
    assert rollout_result.forward_inputs["states"].device.type == "cpu"


def test_nested_dataclass_transport_separates_tensors_from_skeleton():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    shared = torch.arange(4).reshape(2, 2)
    event = PolicyStep(
        sources=[TrajectorySource(key, 2)],
        obs={"states": shared, "task_descriptions": ["a", "b"]},
        rollout_result=EmbodiedRolloutResult(
            actions=shared,
            forward_inputs={"nested": {"states": shared}},
            prev_values=torch.ones(2, 1),
        ),
    )

    skeleton, tensors = pack_dataclass_tensors(event)
    restored = unpack_dataclass_tensors(skeleton, tensors)

    assert isinstance(skeleton.obs["states"], TensorPlaceholder)
    assert isinstance(skeleton.rollout_result.actions, TensorPlaceholder)
    assert isinstance(
        skeleton.rollout_result.forward_inputs["nested"]["states"],
        TensorPlaceholder,
    )
    assert len(tensors) == 2
    assert restored.obs["states"] is restored.rollout_result.actions
    assert torch.equal(restored.rollout_result.prev_values, torch.ones(2, 1))


def test_policy_input_split_merge_preserves_sources_and_nested_payloads():
    keys = [TrajectoryKey(0, 0, rank, 0, 2) for rank in range(2)]
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(12).reshape(4, 3),
            "labels": np.arange(4),
        },
        rlt_switch_flags=torch.tensor([False, True, False, True]),
        sources=[TrajectorySource(key, 2) for key in keys],
    )

    shards = split_policy_input(policy_input, [1, 3])
    merged = merge_policy_inputs(shards)

    assert shards[0].sources == [TrajectorySource(keys[0], 1)]
    assert shards[1].sources == [
        TrajectorySource(keys[0], 1, offset=1),
        TrajectorySource(keys[1], 2),
    ]
    assert merged.sources == policy_input.sources
    assert torch.equal(merged.obs["states"], policy_input.obs["states"])
    assert np.array_equal(merged.obs["labels"], policy_input.obs["labels"])
    assert torch.equal(merged.rlt_switch_flags, policy_input.rlt_switch_flags)


def test_dummy_policy_input_split_merge_preserves_actions_and_type():
    key = TrajectoryKey(0, 0, 0, 0, 1)
    policy_input = DummyPolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        actions=torch.arange(24).reshape(4, 2, 3),
        sources=[TrajectorySource(key, 4)],
    )

    shards = split_policy_input(policy_input, [1, 3])
    merged = merge_policy_inputs(shards)

    assert all(isinstance(shard, DummyPolicyInput) for shard in shards)
    assert isinstance(merged, DummyPolicyInput)
    assert torch.equal(merged.actions, policy_input.actions)
    assert merged.sources == policy_input.sources


def test_online_lerobot_payload_survives_source_routing():
    episode_data = {
        "chunk_actions": torch.arange(12).reshape(3, 2, 2),
        "obs_list": [
            {"images": torch.arange(6).reshape(3, 2)},
            {"images": torch.arange(6, 12).reshape(3, 2)},
        ],
        "terminations": torch.tensor([False, True, False]),
        "truncations": torch.tensor([False, False, True]),
        "infos_list": [
            {"score": np.arange(3)},
            {"score": np.arange(3, 6)},
        ],
    }
    split_data = split_episode_data(episode_data, [1, 2])
    merged_data = merge_episode_data(split_data)

    assert merged_data is not None
    assert torch.equal(merged_data["chunk_actions"], episode_data["chunk_actions"])
    assert torch.equal(
        merged_data["obs_list"][1]["images"],
        episode_data["obs_list"][1]["images"],
    )
    assert np.array_equal(
        merged_data["infos_list"][0]["score"],
        episode_data["infos_list"][0]["score"],
    )


def test_policy_completion_split_merge_preserves_offsets():
    current_key = TrajectoryKey(1, 2, 3, 0, 4)
    previous_key = TrajectoryKey(1, 2, 3, 0, 3)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        sources=[TrajectorySource(current_key, 4)],
        completions=[
            PolicyCompletion(
                sources=[TrajectorySource(previous_key, 4)],
                env_result=EnvResult(rewards=torch.arange(4).reshape(4, 1)),
                next_obs={"states": torch.arange(12).reshape(4, 3)},
                requires_inference=False,
            )
        ],
    )

    shards = split_policy_input(policy_input, [1, 3])
    merged = merge_policy_inputs(shards)

    assert shards[1].sources == [TrajectorySource(current_key, 3, offset=1)]
    assert shards[1].completions[0].sources == [
        TrajectorySource(previous_key, 3, offset=1)
    ]
    assert merged.request_sizes == [1, 3]
    assert torch.equal(
        merged.completions[1].next_obs["states"],
        torch.arange(12).reshape(4, 3)[1:],
    )
