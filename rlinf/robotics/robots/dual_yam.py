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

"""Built-in dual-arm YAM station and its hardware configuration.

This module carries configuration and discovery only. It deliberately does not
import the YAM/i2rt runtime, so a cluster config naming a YAM station can be
parsed on a node that has no robot SDK installed. The environment owns the
i2rt connection; see ``rlinf.envs.real.yam``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from rlinf.scheduler.hardware import HardwareConfig, HardwareResource

from ..discovery import RobotConfig, RobotDiscovery
from ..robot import Robot

_DEVICE_FIELDS = (
    "left_follower",
    "right_follower",
)
_NUM_ARM_JOINTS = 6
_SUPPORTED_CAMERA_TYPES = frozenset({"realsense"})


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    """Return a Hydra/Python mapping or raise a useful config error."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"'{field_name}' in DualYam config must be a mapping, "
            f"but got {type(value)}."
        )
    return value


def _as_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    """Return a non-string Hydra/Python sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"'{field_name}' in DualYam config must be a sequence, "
            f"but got {type(value)}."
        )
    return value


def _finite_float(value: Any, *, field_name: str) -> float:
    """Coerce a numeric config value to a finite float."""
    if isinstance(value, bool):
        raise TypeError(f"'{field_name}' must be a number, but got bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{field_name}' must be a number, but got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"'{field_name}' must be finite, but got {value!r}.")
    return result


def _optional_joint_vector(
    value: Optional[Sequence[float]], *, field_name: str, nonnegative: bool
) -> Optional[list[float]]:
    """Validate an optional six-joint tuning vector."""
    if value is None:
        return None
    vector = list(_as_sequence(value, field_name=field_name))
    if len(vector) != _NUM_ARM_JOINTS:
        raise ValueError(
            f"'{field_name}' must contain {_NUM_ARM_JOINTS} joint values, "
            f"but got {len(vector)}."
        )
    result = [
        _finite_float(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(vector)
    ]
    if nonnegative and any(item < 0.0 for item in result):
        raise ValueError(f"'{field_name}' values must be non-negative: {result}.")
    return result


@dataclass
class YamDeviceConfig:
    """Configuration for one YAM follower arm.

    The three optional tuning vectors intentionally default to ``None``.
    ``None`` means that the hardware backend should preserve its pinned SDK
    defaults; an explicit six-element vector means that RLinf should override
    those defaults for this device only.
    """

    channel: str
    """SocketCAN interface name, for example ``"can0"``."""

    gripper_type: str
    """Backend-specific follower gripper type."""

    ee_mass: Optional[float]
    """Optional end-effector mass override in kilograms.

    ``None`` preserves the mass encoded by the pinned i2rt robot model. Use a
    measured value only when that device's end effector has been calibrated.
    """

    gripper_limits: Optional[tuple[float, float]]
    """Optional motor-radian stops ordered as ``(closed, open)``.

    The order carries the motor direction and therefore must not be sorted.
    ``None`` asks i2rt to detect both stops when the device connects.
    """

    arm_type: str = "yam"
    """Robot model understood by the installed low-level backend."""

    gravity_comp_factor: Optional[list[float]] = None
    """Optional per-joint gravity compensation multipliers."""

    grav_comp_kd: Optional[list[float]] = None
    """Optional per-joint damping used in gravity-compensation mode."""

    coulomb_friction: Optional[list[float]] = None
    """Optional per-joint Coulomb-friction compensation values."""

    use_coulomb_friction: bool = False
    """Whether the low-level backend should apply Coulomb compensation."""

    enable_auto_recovery: bool = False
    """Opt into SDK motor auto-recovery; fail-fast remains the safe default."""

    def __post_init__(self) -> None:
        """Normalize Hydra values and validate device-local safety settings."""
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError(
                f"'channel' in YamDeviceConfig must be a non-empty string, "
                f"but got {self.channel!r}."
            )
        self.channel = self.channel.strip()

        if not isinstance(self.arm_type, str) or not self.arm_type.strip():
            raise ValueError(
                f"'arm_type' in YamDeviceConfig must be a non-empty string, "
                f"but got {self.arm_type!r}."
            )
        self.arm_type = self.arm_type.strip()

        if not isinstance(self.gripper_type, str) or not self.gripper_type.strip():
            raise ValueError(
                "'gripper_type' in YamDeviceConfig must be a non-empty string, "
                f"but got {self.gripper_type!r}."
            )
        self.gripper_type = self.gripper_type.strip()

        if self.ee_mass is not None:
            self.ee_mass = _finite_float(self.ee_mass, field_name="ee_mass")
            if self.ee_mass < 0.0:
                raise ValueError(f"'ee_mass' must be non-negative, got {self.ee_mass}.")

        if self.gripper_limits is not None:
            limits = list(
                _as_sequence(self.gripper_limits, field_name="gripper_limits")
            )
            if len(limits) != 2:
                raise ValueError(
                    "'gripper_limits' must contain exactly [closed, open], "
                    f"but got {len(limits)} values."
                )
            closed = _finite_float(limits[0], field_name="gripper_limits[0]")
            opened = _finite_float(limits[1], field_name="gripper_limits[1]")
            if closed == opened:
                raise ValueError(
                    "'gripper_limits' closed and open stops must differ, "
                    f"but both are {closed}."
                )
            self.gripper_limits = (closed, opened)

        self.gravity_comp_factor = _optional_joint_vector(
            self.gravity_comp_factor,
            field_name="gravity_comp_factor",
            nonnegative=True,
        )
        self.grav_comp_kd = _optional_joint_vector(
            self.grav_comp_kd,
            field_name="grav_comp_kd",
            nonnegative=True,
        )
        self.coulomb_friction = _optional_joint_vector(
            self.coulomb_friction,
            field_name="coulomb_friction",
            nonnegative=True,
        )
        for field_name in (
            "use_coulomb_friction",
            "enable_auto_recovery",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(
                    f"'{field_name}' in YamDeviceConfig must be a bool, "
                    f"but got {type(getattr(self, field_name))}."
                )


@dataclass
class YamCameraConfig:
    """Configuration for one named camera in a YAM station."""

    name: str
    """Stable observation key, for example ``"top_rgb"``."""

    serial: str
    """Camera device serial number."""

    camera_type: str = "realsense"
    """Camera backend. The first YAM integration supports RealSense only."""

    resolution: tuple[int, int] = (640, 480)
    """Color image resolution as ``(width, height)``."""

    fps: int = 30
    """Camera frame rate."""

    enable_depth: bool = False
    """Whether to enable the aligned depth stream."""

    def __post_init__(self) -> None:
        """Normalize Hydra values and validate the camera identity/format."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"'name' in YamCameraConfig must be a non-empty string, "
                f"but got {self.name!r}."
            )
        self.name = self.name.strip()

        if self.serial is None:
            raise ValueError("'serial' in YamCameraConfig must not be None.")
        self.serial = str(self.serial).strip()
        if not self.serial:
            raise ValueError("'serial' in YamCameraConfig must not be empty.")

        if not isinstance(self.camera_type, str) or not self.camera_type.strip():
            raise ValueError(
                "'camera_type' in YamCameraConfig must be a non-empty string, "
                f"but got {self.camera_type!r}."
            )
        self.camera_type = self.camera_type.strip().lower()
        if self.camera_type not in _SUPPORTED_CAMERA_TYPES:
            raise ValueError(
                f"Unsupported YAM camera type {self.camera_type!r}; currently "
                f"supported types: {sorted(_SUPPORTED_CAMERA_TYPES)}."
            )

        resolution = list(_as_sequence(self.resolution, field_name="resolution"))
        if len(resolution) != 2:
            raise ValueError(
                "'resolution' must contain exactly [width, height], "
                f"but got {len(resolution)} values."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in resolution
        ):
            raise TypeError(
                f"'resolution' values must be integers, but got {resolution!r}."
            )
        if any(value <= 0 for value in resolution):
            raise ValueError(
                f"'resolution' values must be positive, but got {resolution!r}."
            )
        self.resolution = (resolution[0], resolution[1])

        if isinstance(self.fps, bool) or not isinstance(self.fps, int):
            raise TypeError(f"'fps' must be an integer, but got {self.fps!r}.")
        if self.fps <= 0:
            raise ValueError(f"'fps' must be positive, but got {self.fps}.")
        if not isinstance(self.enable_depth, bool):
            raise TypeError(
                f"'enable_depth' must be a bool, but got {type(self.enable_depth)}."
            )
        if self.enable_depth:
            raise ValueError(
                "DualYamJointEnv currently records RGB only; set enable_depth=false."
            )


class DualYamRobot(Robot):
    """A dual-follower YAM station.

    The station is driven by the environment rather than composed from arm
    parts, so this class carries no builder: ``rlinf.envs.real.yam`` opens the
    i2rt devices itself. Registration is what links the ``DualYam`` hardware
    type to :class:`DualYamConfig`.
    """

    ROBOT_TYPE = "DualYam"


@dataclass
class DualYamConfig(RobotConfig):
    """Hardware configuration for a dual-follower YAM station."""

    # Every field after ``node_rank`` needs a default because ``RobotConfig``
    # declares one itself. ``None`` still means "required": ``__post_init__``
    # rejects it before the config can be enumerated.
    left_follower: Optional[YamDeviceConfig] = None
    """Left follower arm and gripper configuration."""

    right_follower: Optional[YamDeviceConfig] = None
    """Right follower arm and gripper configuration."""

    cameras: Optional[list[YamCameraConfig]] = None
    """Named cameras exposed by the YAM environment."""

    def __post_init__(self) -> None:
        """Recursively convert Hydra mappings and validate station resources."""
        super().__post_init__()

        for field_name in _DEVICE_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                raise ValueError(
                    f"'{field_name}' in DualYam config must name a CAN device."
                )
            if not isinstance(value, YamDeviceConfig):
                mapping = _as_mapping(value, field_name=field_name)
                value = YamDeviceConfig(**dict(mapping))
                setattr(self, field_name, value)

        cameras = list(_as_sequence(self.cameras or (), field_name="cameras"))
        if not cameras:
            raise ValueError(
                "'cameras' in DualYam config must contain at least one camera."
            )
        converted_cameras: list[YamCameraConfig] = []
        for index, camera in enumerate(cameras):
            if isinstance(camera, YamCameraConfig):
                converted_cameras.append(camera)
                continue
            mapping = _as_mapping(camera, field_name=f"cameras[{index}]")
            converted_cameras.append(YamCameraConfig(**dict(mapping)))
        self.cameras = converted_cameras

        channels = [device.channel for device in self.devices.values()]
        if len(set(channels)) != len(channels):
            raise ValueError(
                f"The two YAM follower CAN channels must be unique, but got {channels}."
            )

        camera_names = [camera.name for camera in self.cameras]
        if len(set(camera_names)) != len(camera_names):
            raise ValueError(
                f"YAM camera names must be unique, but got {camera_names}."
            )
        camera_serials = [camera.serial for camera in self.cameras]
        if len(set(camera_serials)) != len(camera_serials):
            raise ValueError(
                f"YAM camera serials must be unique, but got {camera_serials}."
            )

    @property
    def devices(self) -> dict[str, YamDeviceConfig]:
        """Return both followers keyed by their stable role names."""
        return {field_name: getattr(self, field_name) for field_name in _DEVICE_FIELDS}


class DualYamDiscovery(RobotDiscovery):
    """Enumerate explicitly configured YAM stations on one node.

    Enumeration stays configuration-only: opening SocketCAN or importing i2rt
    here would make cluster discovery mutate robot state.
    """

    HW_TYPE = "DualYam"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Return the YAM stations assigned to ``node_rank``."""
        resource = super().enumerate(node_rank, configs)
        if resource is not None:
            _validate_station_resources([info.config for info in resource.infos])
        return resource


def _validate_station_resources(configs: Sequence[DualYamConfig]) -> None:
    """Reject CAN/camera resource reuse across stations on one node."""
    channels: dict[str, str] = {}
    serials: dict[str, str] = {}
    for station_index, config in enumerate(configs):
        for device_name, device in config.devices.items():
            owner = f"station[{station_index}].{device_name}"
            previous = channels.get(device.channel)
            if previous is not None:
                raise ValueError(
                    f"YAM CAN channel {device.channel!r} is shared by "
                    f"{previous} and {owner}."
                )
            channels[device.channel] = owner
        for camera in config.cameras:
            owner = f"station[{station_index}].cameras[{camera.name}]"
            previous = serials.get(camera.serial)
            if previous is not None:
                raise ValueError(
                    f"YAM camera serial {camera.serial!r} is shared by "
                    f"{previous} and {owner}."
                )
            serials[camera.serial] = owner


DualYamRobot.register_type(DualYamConfig, DualYamDiscovery)

__all__ = [
    "DualYamConfig",
    "DualYamDiscovery",
    "DualYamRobot",
    "YamCameraConfig",
    "YamDeviceConfig",
]
