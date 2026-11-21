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

"""Versioned Stage1 input/output contract for the Tianji robot."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..tianji_contract import DEFAULT_CONTRACT, MAX_ACTION_HORIZON

PROTOCOL = "rlinf.tianji.stage1.v2"
DEFAULT_CAMERA_ROLES = ("head", "wrist_A", "wrist_B")


@dataclass(frozen=True)
class Stage1Contract:
    """Model-facing dimensions; values are explicit so A1 assumptions cannot leak in."""

    camera_roles: tuple[str, ...] = DEFAULT_CAMERA_ROLES
    state_dim: int = 20
    action_dim: int = 20
    action_chunk_length: int = DEFAULT_CONTRACT.action_chunk_length
    state_units: str = "m,rot6_columns,commanded_closure_0_1"
    action_units: str = "m,rot6_columns,absolute_closure_0_1"
    action_encoding: str = "absolute_arm_local_tcp_160mm"
    max_image_bytes: int = 4 * 1024 * 1024

    def __post_init__(self):
        if tuple(self.camera_roles) != DEFAULT_CAMERA_ROLES:
            raise ValueError("Tianji Stage1 camera roles must be head,wrist_A,wrist_B")
        if self.state_dim != 20 or self.action_dim != 20:
            raise ValueError("Tianji Stage1 contract requires state_dim=20/action_dim=20")
        if (
            not 2 <= self.action_chunk_length <= MAX_ACTION_HORIZON
            or self.max_image_bytes < 1024
        ):
            raise ValueError("invalid Stage1 chunk/image limits")


@dataclass
class Stage1Request:
    """Decoded one-frame request passed to a predictor plugin."""

    images: dict[str, bytes]
    state: np.ndarray
    prompt: str
    metadata: dict = field(default_factory=dict)


def validate_request(
    payload: Mapping, contract: Stage1Contract = Stage1Contract()
) -> Stage1Request:
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Stage1 protocol mismatch")
    images = payload.get("images")
    if not isinstance(images, Mapping) or set(images) != set(contract.camera_roles):
        raise ValueError(f"images must contain exactly {list(contract.camera_roles)}")
    decoded = {}
    for role in contract.camera_roles:
        value = images[role]
        if not isinstance(value, str):
            raise ValueError(f"image {role} must be base64 text")
        try:
            data = base64.b64decode(value.encode("ascii"), validate=True)
        except Exception as exc:
            raise ValueError(f"image {role} is not valid base64") from exc
        if not data or len(data) > contract.max_image_bytes or data[:2] != b"\xff\xd8":
            raise ValueError(f"image {role} must be a bounded JPEG")
        decoded[role] = data
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (contract.state_dim,) or not np.isfinite(state).all():
        raise ValueError(f"state must be finite float32[{contract.state_dim}]")
    for index in (9, 19):
        if not 0.0 <= float(state[index]) <= 1.0:
            raise ValueError("state gripper closure must be in [0,1]")
    from ..tianji_contract import DEFAULT_CONTRACT
    DEFAULT_CONTRACT.validate_state(state)
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        raise ValueError("prompt must be a non-empty bounded string")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    return Stage1Request(decoded, np.ascontiguousarray(state), prompt, dict(metadata))


def validate_response(
    actions, contract: Stage1Contract = Stage1Contract()
) -> np.ndarray:
    """Validate physical absolute action chunks and return contiguous float32 data."""
    array = np.asarray(actions, dtype=np.float32)
    expected = (contract.action_chunk_length, contract.action_dim)
    if array.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {array.shape}")
    from ..tianji_contract import project_pose
    return np.ascontiguousarray(project_pose(array))



def request_to_json(
    request: Stage1Request, contract: Stage1Contract = Stage1Contract()
) -> dict:
    """Encode a request for the framed JSON wire protocol."""
    return {
        "protocol": PROTOCOL,
        "images": {
            role: base64.b64encode(request.images[role]).decode("ascii")
            for role in contract.camera_roles
        },
        "state": request.state.astype(np.float32).tolist(),
        "prompt": request.prompt,
        "metadata": dict(request.metadata),
    }
