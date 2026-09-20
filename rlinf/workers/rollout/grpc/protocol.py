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

"""Versioned RLinf policy RPC messages; numeric arrays retain shape and dtype.

This is not LeRobot's stateful AsyncInference protocol. Each Predict request
owns its observation batch and response; no shared observation queue or pickle
payload is used. Transport is restricted to trusted networks or SSH tunnels.
"""

from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np
import torch

VERSION = 1
SERVICE = "rlinf.policy.v1.Policy"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
OPTIONS = (
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.enable_retries", 0),
)


def _encode(value: Any) -> msgpack.ExtType:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "buif":
        raise TypeError(f"Unsupported RPC value: {type(value).__name__}")
    return msgpack.ExtType(
        1,
        msgpack.packb(
            (value.dtype.str, value.shape, value.tobytes()), use_bin_type=True
        ),
    )


def _decode(code: int, payload: bytes) -> np.ndarray:
    if code != 1:
        raise ValueError(f"Unsupported RPC extension: {code}")
    dtype_name, shape, data = msgpack.unpackb(payload, raw=False)
    dtype = np.dtype(dtype_name)
    if dtype.kind not in "buif" or not isinstance(shape, list) or len(shape) > 8:
        raise ValueError("Invalid numeric array")
    if any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("Invalid array shape")
    if math.prod(shape) * dtype.itemsize != len(data):
        raise ValueError("Array shape does not match payload")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def pack_message(message: dict[str, Any]) -> bytes:
    """Serialize an RPC mapping without executable Python objects."""
    data = msgpack.packb(message, default=_encode, use_bin_type=True)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Policy RPC message exceeds 64 MiB")
    return data


def unpack_message(data: bytes) -> dict[str, Any]:
    """Decode an RPC mapping, rejecting unsupported versions."""
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Policy RPC message exceeds 64 MiB")
    message = msgpack.unpackb(data, raw=False, ext_hook=_decode)
    if not isinstance(message, dict) or message.get("version") != VERSION:
        raise ValueError("Incompatible policy protocol version")
    return message


def validate_observations(obs: dict[str, Any]) -> int:
    """Validate the batched image/state policy input and return its batch size."""
    if not isinstance(obs, dict):
        raise ValueError("Observations must be a mapping")
    states = obs.get("states")
    if not isinstance(states, (np.ndarray, torch.Tensor)) or states.ndim != 2:
        raise ValueError("states must have shape [batch, state_dim]")
    batch_size = states.shape[0]
    if (
        batch_size < 1
        or not np.isfinite(
            np.asarray(
                states.detach().cpu() if isinstance(states, torch.Tensor) else states
            )
        ).all()
    ):
        raise ValueError("states must be nonempty and finite")
    tasks = obs.get("task_descriptions")
    if (
        not isinstance(tasks, (list, tuple))
        or len(tasks) != batch_size
        or not all(isinstance(task, str) for task in tasks)
    ):
        raise ValueError("task_descriptions must contain one string per observation")
    for key in ("main_images", "wrist_images", "extra_view_images"):
        value = obs.get(key)
        if value is not None:
            if (
                not isinstance(value, (np.ndarray, torch.Tensor))
                or value.ndim < 4
                or value.shape[0] != batch_size
            ):
                raise ValueError(f"{key} must have a matching batch dimension")
    return batch_size


def validate_actions(actions: np.ndarray, batch: int, chunk: int, dim: int) -> None:
    """Reject malformed or nonfinite actions without changing their units."""
    if not isinstance(actions, np.ndarray) or actions.shape != (batch, chunk, dim):
        raise ValueError(f"Expected actions shape {(batch, chunk, dim)}")
    if actions.dtype.kind != "f" or not np.isfinite(actions).all():
        raise ValueError("Actions must be finite floating-point values")
