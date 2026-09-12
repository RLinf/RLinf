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

"""What a policy reads, key by key.

A policy's observation is a handful of named vectors and some camera frames.
Each :class:`StateKey` says which parts it comes from, in which order, and
through which encoding, so a key that carries both arms' poses or a rotation
written as six numbers is a declaration rather than code inside an env.

An :class:`ObservationSpec` collects the keys and the camera settings. A
checkpoint is trained against exactly one of these, which is why the colour
order and the frame size live here beside the state.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.tasks.requirements import Needs, Reading
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.utils.rot6d import matrix_to_rot6d

#: Turns one part's field into what the policy reads.
Encoder = Callable[[np.ndarray], np.ndarray]


def pose_from_euler(value: np.ndarray) -> np.ndarray:
    """Rewrite ``xyz`` plus Euler angles as ``xyz`` plus an ``xyzw`` quaternion.

    For an arm whose controller reports roll, pitch and yaw, where a policy
    was trained on quaternions. Anything past the sixth number is dropped.
    """
    value = np.asarray(value, dtype=np.float64)
    return np.concatenate([value[:3], R.from_euler("xyz", value[3:6]).as_quat()])


def pose_as_rot6d(value: np.ndarray) -> np.ndarray:
    """Rewrite ``xyz`` plus a quaternion as ``xyz`` plus a six-number rotation.

    The six numbers are the first two columns of the rotation matrix, which a
    policy can regress without the sign ambiguity a quaternion carries.
    """
    value = np.asarray(value, dtype=np.float64)
    rotation = matrix_to_rot6d(R.from_quat(value[3:7]).as_matrix())
    return np.concatenate([value[:3], np.asarray(rotation).reshape(-1)])


@dataclass(frozen=True)
class Source:
    """One part's field, as it contributes to a state key.

    Attributes:
        field: Name the part reports it under.
        role: The role whose part reports it.
        end_effector: Read it from the end effector the role's part carries.
        encode: Rewrite the field into what the policy reads.
        absent: What to report when the role carries no end effector, so a rig
            without one keeps the policy's layout. Only for a key with this
            one source.
    """

    field: str
    role: str = "arm"
    end_effector: bool = False
    encode: Optional[Encoder] = None
    absent: Optional[float] = None

    def requirements(self) -> Mapping[str, Needs]:
        """What the role's part must report for this source to exist."""
        if self.end_effector:
            return (
                {}
                if self.absent is not None
                else {self.role: Needs(end_effector="any")}
            )
        return {self.role: Needs(observes=frozenset({self.field}))}

    def read(self, view: Reading) -> Optional[np.ndarray]:
        """This source's numbers from one whole-robot reading, or ``None``."""
        if self.end_effector:
            reported = view.end_effector(self.role)
        else:
            reported = view.part(self.role)
        if reported is None:
            return None
        value = np.asarray(reported[self.field], dtype=np.float64).reshape(-1)
        return self.encode(value) if self.encode is not None else value


@dataclass(frozen=True)
class StateKey:
    """One named vector the policy reads.

    Attributes:
        key: Name in the observation.
        shape: Shape the policy sees, after every source is concatenated.
        sources: The parts it is built from, in the order they concatenate.
        low: Lower bound of the space.
        high: Upper bound of the space.
        dtype: Type the policy reads it as.
    """

    key: str
    shape: tuple[int, ...]
    sources: tuple[Source, ...]
    low: float = -np.inf
    high: float = np.inf
    dtype: Any = np.float32

    def requirements(self) -> list[Mapping[str, Needs]]:
        """What every source needs of the robot."""
        return [source.requirements() for source in self.sources]

    def read(self, view: Reading) -> np.ndarray:
        """Build this key's value from one whole-robot reading.

        Raises:
            RequirementError: If a source is missing and no substitute for it
                was declared.
        """
        from rlinf.envs.real.tasks.requirements import RequirementError

        chunks = []
        for source in self.sources:
            value = source.read(view)
            if value is None:
                if len(self.sources) == 1 and source.absent is not None:
                    return np.full(self.shape, source.absent, dtype=self.dtype)
                raise RequirementError(
                    f"State key {self.key!r} reads the end effector of role "
                    f"{source.role!r}, which carries none."
                )
            chunks.append(value)
        return np.concatenate(chunks).astype(self.dtype).reshape(self.shape)


@dataclass(frozen=True)
class ObservationSpec:
    """Everything a policy reads: state keys, then camera frames.

    Attributes:
        state: The named vectors, in no particular order; ``flatten`` decides
            what order the runner concatenates them in.
        cameras: The cameras whose frames the policy sees.
        frame_size: ``(height, width)`` every frame is resized to.
        frame_order: ``"rgb"`` or ``"bgr"``, the channel order the policy was
            trained on. Drivers deliver BGR; ``"rgb"`` flips it.
        min_cameras: Cameras this layout needs, checked before any hardware is
            touched.
        order: The order the runner concatenates the state keys in. ``None``
            sorts them by name, which is the order every shipped checkpoint
            was trained on.
    """

    state: tuple[StateKey, ...]
    cameras: tuple[CameraInfo, ...] = ()
    frame_size: tuple[int, int] = (128, 128)
    frame_order: str = "rgb"
    min_cameras: int = 0
    order: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.frame_order not in ("rgb", "bgr"):
            raise ValueError(
                f"frame_order must be 'rgb' or 'bgr', got {self.frame_order!r}."
            )
        if self.order is not None and sorted(self.order) != sorted(
            key.key for key in self.state
        ):
            raise ValueError(
                f"order {list(self.order)} must name every state key exactly "
                f"once; the keys are {sorted(key.key for key in self.state)}."
            )

    @property
    def to_rgb(self) -> bool:
        """Whether a driver's BGR frame is flipped before the policy sees it."""
        return self.frame_order == "rgb"

    def key(self, name: str) -> StateKey:
        """The state key called ``name``."""
        return next(key for key in self.state if key.key == name)

    def flatten(self) -> tuple[str, ...]:
        """The order the runner concatenates the state keys in.

        A policy is trained against one order, so a spec states it rather than
        leaving it to however a dictionary happens to iterate.
        """
        if self.order is not None:
            return tuple(self.order)
        return tuple(sorted(key.key for key in self.state))

    def requirements(self) -> list[Mapping[str, Needs]]:
        """What every state key needs of the robot."""
        return [needs for key in self.state for needs in key.requirements()]

    def read(self, view: Reading) -> dict[str, np.ndarray]:
        """Build the policy's state from one whole-robot reading."""
        return {key.key: key.read(view) for key in self.state}
