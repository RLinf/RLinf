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

"""Schema-layer entrypoints for data module."""

from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
    EmbodiedTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
    DummyPolicyInput,
    EmbodiedRolloutResult,
    EnvOutput,
    EnvResult,
    PolicyCompletion,
    PolicyInput,
    PolicyOutput,
    RTCActionResponse,
    RTCRequest,
    Trajectory,
    TrajectoryKey,
    TrajectorySource,
    convert_trajectories_to_batch,
    get_model_weights_id,
    merge_policy_inputs,
    split_policy_input,
)
from rlinf.data.schema.reasoning_requests import (
    FinishReasonEnum,
    RolloutRequest,
    SeqGroupInfo,
    get_batch_size,
    get_seq_length,
)
from rlinf.data.schema.reasoning_results import (
    DynamicRolloutResult,
    RolloutResult,
)

__all__ = [
    "ChunkStepResult",
    "convert_trajectories_to_batch",
    "DynamicRolloutResult",
    "DummyPolicyInput",
    "EmbodiedLerobotTrajectoryBuilder",
    "EmbodiedRolloutResult",
    "EnvResult",
    "PolicyCompletion",
    "PolicyInput",
    "PolicyOutput",
    "TrajectoryKey",
    "TrajectorySource",
    "EmbodiedTrajectoryBuilder",
    "EnvOutput",
    "FinishReasonEnum",
    "RTCActionResponse",
    "RTCRequest",
    "RolloutResult",
    "RolloutRequest",
    "SeqGroupInfo",
    "Trajectory",
    "get_batch_size",
    "get_model_weights_id",
    "get_seq_length",
    "merge_policy_inputs",
    "split_policy_input",
]
