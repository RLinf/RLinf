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

"""Kuavo robot composition and scheduler configuration."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..discovery import RobotConfig
from ..parts.base import PartGroup, RobotPart
from ..robot import Robot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..parts.arms.kuavo import KuavoConnection

_PLATFORMS = frozenset({"4pro", "5", "5w"})
_ARM_SELECTIONS = frozenset({"left", "right", "both"})
_END_EFFECTORS = frozenset({"leju_claw", "qiangnao", "rq2f85"})
_ARM_JOINT_RANGES = {
    "4pro": (12, 26),
    "5": (13, 27),
    "5w": (4, 18),
}


def _default_obs_key_map() -> dict[str, dict[str, Any]]:
    """Return the complete Kuavo ROS observation routing table."""
    return {
        "head_cam_h": {
            "topic": "/cam_h/color/image_raw/compressed",
            "msg_type": "CompressedImage",
            "frequency": 30,
        },
        "wrist_cam_l": {
            "topic": "/cam_l/color/image_raw/compressed",
            "msg_type": "CompressedImage",
            "frequency": 30,
        },
        "wrist_cam_r": {
            "topic": "/cam_r/color/image_raw/compressed",
            "msg_type": "CompressedImage",
            "frequency": 30,
        },
        "joint_q": {
            "topic": "/sensors_data_raw",
            "msg_type": "sensorsData",
            "frequency": 500,
        },
        "leju_claw": {
            "topic": "/leju_claw_state",
            "msg_type": "lejuClawState",
            "frequency": 500,
        },
        "qiangnao": {
            "topic": "/dexhand/state",
            "msg_type": "JointState",
            "frequency": 500,
        },
        "rq2f85": {
            "topic": "/gripper/state",
            "msg_type": "JointState",
            "frequency": 500,
        },
    }


class KuavoRobot(Robot):
    """Composable Kuavo robot with one shared dual-arm SDK connection."""

    ROBOT_TYPE = "Kuavo"

    @classmethod
    def declare_connection(
        cls,
        *,
        which_arm: str,
        end_effector_type: str,
        head_position: list[float] | None,
        joint_limits_min: list[float],
        joint_limits_max: list[float],
        end_effector_limits_min: list[float],
        end_effector_limits_max: list[float],
        node_rank: int | None,
        name: str,
    ) -> "KuavoConnection":
        """Declare the deferred SDK connection used by all Kuavo parts."""
        from ..parts.arms.kuavo import KuavoConnection

        return KuavoConnection(
            which_arm=which_arm,
            end_effector_type=end_effector_type,
            head_position=None if head_position is None else tuple(head_position),
            joint_limits_min=joint_limits_min,
            joint_limits_max=joint_limits_max,
            end_effector_limits_min=end_effector_limits_min,
            end_effector_limits_max=end_effector_limits_max,
            node_rank=node_rank,
            worker_name=name,
        )

    @classmethod
    def build_arms(
        cls, connection: "KuavoConnection", *, which_arm: str
    ) -> dict[str, RobotPart]:
        """Return the configured arm groups exported by the shared connection."""
        sides = ("left", "right") if which_arm == "both" else (which_arm,)
        return {
            side: PartGroup(
                arm=connection.part(side),
                gripper=connection.part(f"{side}_end_effector"),
            )
            for side in sides
        }

    @classmethod
    def build(
        cls,
        *,
        which_arm: str = "both",
        end_effector_type: str = "leju_claw",
        head_position: list[float] | None = None,
        joint_limits_min: list[float] | None = None,
        joint_limits_max: list[float] | None = None,
        end_effector_limits_min: list[float] | None = None,
        end_effector_limits_max: list[float] | None = None,
        env_idx: int = 0,
        node_rank: int | None = None,
        controller_node_rank: int | None = None,
        worker_rank: int = 0,
    ) -> "KuavoRobot":
        """Compose a Kuavo robot without opening its hardware connection."""
        connection = cls.declare_connection(
            which_arm=which_arm,
            end_effector_type=end_effector_type,
            head_position=head_position,
            joint_limits_min=(
                [-3.14] * 14 if joint_limits_min is None else joint_limits_min
            ),
            joint_limits_max=(
                [3.14] * 14 if joint_limits_max is None else joint_limits_max
            ),
            end_effector_limits_min=(
                [0.0, 0.0]
                if end_effector_limits_min is None
                else end_effector_limits_min
            ),
            end_effector_limits_max=(
                [1.0, 1.0]
                if end_effector_limits_max is None
                else end_effector_limits_max
            ),
            node_rank=(
                node_rank if controller_node_rank is None else controller_node_rank
            ),
            name=f"KuavoConnection-{worker_rank}-{env_idx}",
        )
        return cls(**cls.build_arms(connection, which_arm=which_arm))


@dataclass
class KuavoRobotConfig(RobotConfig):
    """Hardware, control, observation, and safety configuration for Kuavo."""

    platform_type: str = "4pro"
    """Kuavo platform revision: ``4pro``, ``5``, or ``5w``."""

    which_arm: str = "both"
    """Arm selection exposed to the environment: left, right, or both."""

    end_effector_type: str = "leju_claw"
    """Installed end effector: Leju claw, Qiangnao hand, or RQ2F85."""

    only_arm: bool = True
    """Whether this allocation controls only the arms and end effectors."""

    control_mode: str = "joint"
    """RLinf currently exposes Kuavo through joint-position control."""

    direct_to_wbc: bool = False
    """Direct WBC control is reserved for a dedicated controller integration."""

    qiangnao_dof_needed: int = 1
    """Number of Qiangnao state dimensions exposed per hand."""

    is_binary: bool = False
    """Whether to collapse end-effector state and commands to open/closed."""

    head_position: list[float] | None = None
    """Optional initial head yaw and pitch in radians."""

    ros_rate: int = 10
    """ROS observation and action frequency in hertz."""

    control_rate: int = 100
    """Robot-side command frequency in hertz."""

    controller_node_rank: int | None = None
    """Node that opens the SDK; ``None`` uses the environment worker node."""

    image_size: list[int] = field(default_factory=lambda: [848, 480])
    """Camera image size as ``[width, height]``."""

    depth_range: list[int] = field(default_factory=lambda: [0, 1500])
    """Accepted depth interval in millimetres."""

    obs_key_map: dict[str, dict[str, Any]] = field(
        default_factory=_default_obs_key_map
    )
    """ROS topic, message type, and source frequency for each observation."""

    arm_state_keys: list[str] = field(
        default_factory=lambda: ["joint_q", "gripper"]
    )
    """Ordered state entries concatenated for each selected arm."""

    joint_limits_min: list[float] = field(default_factory=lambda: [-3.14] * 14)
    joint_limits_max: list[float] = field(default_factory=lambda: [3.14] * 14)
    end_effector_limits_min: list[float] = field(default_factory=lambda: [0.0, 0.0])
    end_effector_limits_max: list[float] = field(default_factory=lambda: [1.0, 1.0])

    def __post_init__(self) -> None:
        """Validate choices and fixed-width hardware contracts."""
        assert isinstance(self.node_rank, int), (
            "'node_rank' in Kuavo config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.platform_type not in _PLATFORMS:
            raise ValueError(
                f"Unsupported Kuavo platform_type {self.platform_type!r}; "
                f"choose one of {sorted(_PLATFORMS)}."
            )
        if self.which_arm not in _ARM_SELECTIONS:
            raise ValueError(
                f"Unsupported Kuavo which_arm {self.which_arm!r}; "
                f"choose one of {sorted(_ARM_SELECTIONS)}."
            )
        if self.end_effector_type not in _END_EFFECTORS:
            raise ValueError(
                "Unsupported Kuavo end_effector_type "
                f"{self.end_effector_type!r}; choose one of "
                f"{sorted(_END_EFFECTORS)}."
            )
        if self.control_mode != "joint":
            raise ValueError("Kuavo currently supports control_mode='joint'.")
        if not self.only_arm:
            raise ValueError("Kuavo currently supports only_arm=true.")
        if self.direct_to_wbc:
            raise ValueError("Kuavo direct_to_wbc control is not yet supported.")
        if self.qiangnao_dof_needed != 1:
            raise ValueError("Kuavo currently supports qiangnao_dof_needed=1.")
        if self.is_binary:
            raise ValueError("Kuavo binary end-effector control is not yet supported.")
        if self.head_position is not None:
            self._validate_numeric_values("head_position", self.head_position, 2)
        self._validate_pair("image_size", self.image_size, 2)
        self._validate_numeric_values("depth_range", self.depth_range, 2)
        self._validate_limits(
            "joint_limits", self.joint_limits_min, self.joint_limits_max, 14
        )
        self._validate_limits(
            "end_effector_limits",
            self.end_effector_limits_min,
            self.end_effector_limits_max,
            2,
        )
        if any(value < 0.0 for value in self.end_effector_limits_min) or any(
            value > 1.0 for value in self.end_effector_limits_max
        ):
            raise ValueError("Kuavo end_effector_limits must stay within 0..1.")
        if not all(
            isinstance(rate, (int, float)) and np.isfinite(rate) and rate > 0
            for rate in (self.ros_rate, self.control_rate)
        ):
            raise ValueError("Kuavo control frequencies must be positive.")
        if self.controller_node_rank is not None and not isinstance(
            self.controller_node_rank, int
        ):
            raise ValueError("Kuavo controller_node_rank must be an integer or None.")
        if self.depth_range[0] >= self.depth_range[1]:
            raise ValueError("Kuavo depth_range must be strictly increasing.")
        if any(not isinstance(size, int) or size <= 0 for size in self.image_size):
            raise ValueError("Kuavo image_size values must be positive.")
        self._validate_obs_key_map()

    def _validate_obs_key_map(self) -> None:
        """Validate observation routes and attach image processing dimensions."""
        required = {"head_cam_h", "joint_q", self.end_effector_type}
        if self.which_arm in {"left", "both"}:
            required.add("wrist_cam_l")
        if self.which_arm in {"right", "both"}:
            required.add("wrist_cam_r")
        missing = sorted(required - self.obs_key_map.keys())
        if missing:
            raise ValueError(f"Kuavo obs_key_map is missing required keys {missing}.")

        for key, route in self.obs_key_map.items():
            if not isinstance(route, dict):
                raise ValueError(f"Kuavo obs_key_map[{key!r}] must be a mapping.")
            absent = {"topic", "msg_type", "frequency"} - route.keys()
            if absent:
                raise ValueError(
                    f"Kuavo obs_key_map[{key!r}] is missing {sorted(absent)}."
                )
            if not route["topic"] or not route["msg_type"]:
                raise ValueError(
                    f"Kuavo obs_key_map[{key!r}] needs a topic and message type."
                )
            frequency = route["frequency"]
            if (
                not isinstance(frequency, (int, float))
                or not np.isfinite(frequency)
                or frequency <= 0
            ):
                raise ValueError(
                    f"Kuavo obs_key_map[{key!r}] frequency must be positive."
                )
        allowed_state_keys = {"joint_q", "gripper"}
        if not self.arm_state_keys or len(set(self.arm_state_keys)) != len(
            self.arm_state_keys
        ):
            raise ValueError("Kuavo arm_state_keys must be non-empty and unique.")
        unknown = sorted(set(self.arm_state_keys) - allowed_state_keys)
        if unknown:
            raise ValueError(f"Unsupported Kuavo arm_state_keys {unknown}.")

    def active_obs_key_map(self) -> dict[str, dict[str, Any]]:
        """Return only routes used by the selected arms and end effector."""
        active_keys = {"head_cam_h", "joint_q", self.end_effector_type}
        if self.which_arm in {"left", "both"}:
            active_keys.add("wrist_cam_l")
        if self.which_arm in {"right", "both"}:
            active_keys.add("wrist_cam_r")

        active = {
            key: deepcopy(route)
            for key, route in self.obs_key_map.items()
            if key in active_keys
        }
        active["gripper"] = active.pop(self.end_effector_type)
        for key, route in active.items():
            params = route.setdefault("handle", {}).setdefault("params", {})
            if key in {"head_cam_h", "wrist_cam_l", "wrist_cam_r"}:
                params["resize_wh"] = list(self.image_size)
            if "depth" in key:
                params["depth_range"] = list(self.depth_range)
        active["joint_q"]["handle"]["params"]["slice"] = self.joint_q_slices
        active["gripper"]["handle"]["params"]["slice"] = self.gripper_slices
        return active

    @property
    def joint_q_slices(self) -> list[list[int]]:
        """Return raw joint-state slices for the configured platform and arms."""
        start, end = _ARM_JOINT_RANGES[self.platform_type]
        middle = start + 7
        return {
            "left": [[start, middle]],
            "right": [[middle, end]],
            "both": [[start, middle], [middle, end]],
        }[self.which_arm]

    @property
    def gripper_slices(self) -> list[list[int]]:
        """Return end-effector state slices for the configured device and arms."""
        right_start = 6 if self.end_effector_type == "qiangnao" else 1
        return {
            "left": [[0, 1]],
            "right": [[right_start, right_start + 1]],
            "both": [[0, 1], [right_start, right_start + 1]],
        }[self.which_arm]

    @staticmethod
    def _validate_pair(name: str, values: list[float], size: int) -> None:
        if len(values) != size:
            raise ValueError(f"Kuavo {name} must contain {size} values.")

    @staticmethod
    def _validate_numeric_values(
        name: str, values: list[float], size: int
    ) -> None:
        KuavoRobotConfig._validate_pair(name, values, size)
        if not all(
            isinstance(value, (int, float)) and np.isfinite(value)
            for value in values
        ):
            raise ValueError(f"Kuavo {name} values must be finite numbers.")

    @staticmethod
    def _validate_limits(
        name: str, minimum: list[float], maximum: list[float], size: int
    ) -> None:
        if len(minimum) != size or len(maximum) != size:
            raise ValueError(f"Kuavo {name} must contain {size} min/max values.")
        if not all(
            isinstance(value, (int, float)) and np.isfinite(value)
            for value in (*minimum, *maximum)
        ):
            raise ValueError(f"Kuavo {name} values must be finite numbers.")
        if any(low > high for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError(f"Kuavo {name} minimum exceeds its maximum.")

    def hardware_model(self, robot_type: str) -> str:
        """Include the platform, arm selection, and end effector in discovery."""
        return (
            f"{robot_type}_{self.platform_type}_{self.which_arm}_"
            f"{self.end_effector_type}"
        )


KuavoRobot.register_type(KuavoRobotConfig)
