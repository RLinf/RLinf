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

from dataclasses import dataclass

import pytest
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.scheduler.cluster.utils import (
    extract_dataclass_tensor_fields,
    unflatten_dataclass_tensor_fields,
)
from rlinf.workers.trajectory.compression import (
    TrajectoryCompressionConfig,
    compress_trajectory,
    decompress_trajectory,
)
from rlinf.workers.trajectory.data import (
    Actions,
    Observations,
    TrajectoryEnvelope,
)
from rlinf.workers.trajectory.transport import TransportEndpoint


@dataclass
class NestedPayload:
    values: dict[str, object]


@dataclass
class Payload:
    nested: NestedPayload
    items: list[object]


def _observations(*, current_step: int, task: str = "pick") -> Observations:
    return Observations(
        global_step=7,
        rank=2,
        current_epoch=3,
        current_step=current_step,
        stage_id=1,
        slot_ids=(4, 5),
        obs={
            "images": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
            "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        task_descriptions=[task, "place"],
    )


def test_extracts_and_restores_nested_dataclass_tensor_leaves():
    payload = Payload(
        nested=NestedPayload(values={"image": torch.arange(4), "caption": "pick cube"}),
        items=[torch.ones(2), ("metadata", torch.zeros(3))],
    )

    skeleton, tensors, metadata = extract_dataclass_tensor_fields(payload)
    restored = unflatten_dataclass_tensor_fields(skeleton, metadata, tensors)

    assert skeleton.nested.values["image"] is None
    assert skeleton.items[0] is None
    assert skeleton.items[1][1] is None
    assert len(metadata) == 3
    assert torch.equal(restored.nested.values["image"], payload.nested.values["image"])
    assert torch.equal(restored.items[0], payload.items[0])
    assert torch.equal(restored.items[1][1], payload.items[1][1])
    assert restored.nested.values["caption"] == "pick cube"


def test_extracts_nested_trajectory_envelope_tensors():
    envelope = TrajectoryEnvelope(
        record=Observations(
            global_step=1,
            rank=0,
            current_epoch=0,
            obs={"image": torch.ones(2, 3), "nested": {"state": torch.zeros(2)}},
        ),
        batch_index="batch-0",
    )

    skeleton, tensors, metadata = extract_dataclass_tensor_fields(envelope)
    restored = unflatten_dataclass_tensor_fields(skeleton, metadata, tensors)

    assert skeleton.record.obs["image"] is None
    assert skeleton.record.obs["nested"]["state"] is None
    assert torch.equal(restored.record.obs["image"], envelope.record.obs["image"])
    assert torch.equal(
        restored.record.obs["nested"]["state"],
        envelope.record.obs["nested"]["state"],
    )


def test_compressed_trajectory_round_trip_preserves_nested_tensor_fields():
    trajectory = Trajectory(
        max_episode_length=240,
        model_weights_id="weights-7",
        actions=torch.arange(60, dtype=torch.float32).reshape(5, 3, 4),
        rewards=torch.arange(15, dtype=torch.float32).reshape(5, 3, 1),
        curr_obs={"main_images": torch.arange(30, dtype=torch.uint8).reshape(5, 3, 2)},
        next_obs={"state": torch.arange(15, dtype=torch.float32).reshape(5, 3, 1)},
        forward_inputs={"action": torch.ones(5, 3, 4)},
    )

    compressed = compress_trajectory(
        trajectory, TrajectoryCompressionConfig(enabled=True, level=1, chunk_steps=2)
    )
    restored = decompress_trajectory(compressed)

    assert restored.max_episode_length == trajectory.max_episode_length
    assert restored.model_weights_id == trajectory.model_weights_id
    assert torch.equal(restored.actions, trajectory.actions)
    assert torch.equal(restored.rewards, trajectory.rewards)
    assert torch.equal(
        restored.curr_obs["main_images"], trajectory.curr_obs["main_images"]
    )
    assert torch.equal(restored.next_obs["state"], trajectory.next_obs["state"])
    assert torch.equal(
        restored.forward_inputs["action"], trajectory.forward_inputs["action"]
    )
    assert len(compressed.blocks) > 1
    assert compressed.payload.dtype == torch.uint8

    skeleton, tensors, metadata = extract_dataclass_tensor_fields(compressed)
    assert len(tensors) == 1
    transported = unflatten_dataclass_tensor_fields(skeleton, metadata, tensors)
    assert torch.equal(decompress_trajectory(transported).actions, trajectory.actions)


def test_compressed_trajectory_rejects_disabled_codec():
    trajectory = Trajectory(actions=torch.ones(1, 1, 1))

    with pytest.raises(ValueError, match="disabled"):
        compress_trajectory(trajectory, TrajectoryCompressionConfig())


def test_transport_endpoint_round_trips_fixed_tensor_frame():
    sender = TransportEndpoint()
    receiver = TransportEndpoint()
    first = _observations(current_step=0)
    sender.bootstrap(first)
    receiver.bootstrap(first)

    payload = _observations(current_step=1)
    frame = sender.encode(payload)

    assert frame is not None
    assert frame.header.tolist() == [7, 2, 3, 1, 1, 2, 4, 5]
    buffers = receiver.receive_buffers()
    for buffer, tensor in zip(buffers, frame.tensors, strict=True):
        buffer.copy_(tensor)

    restored = receiver.decode(frame.header, buffers)

    assert restored.global_step == payload.global_step
    assert restored.current_step == payload.current_step
    assert restored.slot_ids == payload.slot_ids
    assert restored.task_descriptions == payload.task_descriptions
    assert torch.equal(restored.obs["images"], payload.obs["images"])
    assert torch.equal(restored.obs["state"], payload.obs["state"])


def test_transport_endpoint_falls_back_when_non_tensor_layout_changes():
    endpoint = TransportEndpoint()
    endpoint.bootstrap(_observations(current_step=0, task="pick"))

    assert endpoint.encode(_observations(current_step=1, task="push")) is None


def test_transport_endpoint_rejects_non_contiguous_tensor_lanes():
    endpoint = TransportEndpoint()
    prototype = _observations(current_step=0)
    endpoint.bootstrap(prototype)
    payload = _observations(current_step=1)
    payload.obs["images"] = payload.obs["images"].transpose(2, 3)

    assert endpoint.encode(payload) is None


def test_transport_endpoint_preserves_leaf_order_across_dtype_lanes():
    """Tensor lanes must restore traversal order, not dtype-group order."""
    payload = Actions(
        global_step=7,
        rank=2,
        current_epoch=3,
        current_step=1,
        stage_id=1,
        slot_ids=(4, 5),
        actions=torch.full((2, 1, 2), 1.0, dtype=torch.float64),
        prev_logprobs=torch.full((2, 1, 2), 2.0, dtype=torch.float32),
        prev_values=torch.full((2, 1), 3.0, dtype=torch.float32),
        versions=torch.full((2, 1, 2), 4.0, dtype=torch.float32),
        forward_inputs={
            "token_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "action": torch.full((2, 1, 2), 5.0, dtype=torch.float64),
            "mask": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
            "model_action": torch.full((2, 1, 3), 6.0, dtype=torch.float32),
        },
    )
    sender = TransportEndpoint()
    receiver = TransportEndpoint()
    sender.bootstrap(payload)
    receiver.bootstrap(payload)

    frame = sender.encode(payload)
    assert frame is not None
    buffers = receiver.receive_buffers()
    for buffer, tensor in zip(buffers, frame.tensors, strict=True):
        buffer.copy_(tensor)
    restored = receiver.decode(frame.header, buffers)

    assert torch.equal(restored.actions, payload.actions)
    assert torch.equal(restored.prev_logprobs, payload.prev_logprobs)
    assert torch.equal(restored.prev_values, payload.prev_values)
    assert torch.equal(restored.versions, payload.versions)
    for name, tensor in payload.forward_inputs.items():
        assert torch.equal(restored.forward_inputs[name], tensor)
