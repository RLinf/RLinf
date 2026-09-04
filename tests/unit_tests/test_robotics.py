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

"""Tests for robot parts, composition, placement, and layer boundaries."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest

import rlinf.robotics.robots.franka as franka_module
from rlinf.robotics import (
    Arm,
    Camera,
    Connection,
    ControllablePart,
    DOSW1Robot,
    DOSW1RobotConfig,
    DualFrankaRobot,
    EndEffector,
    FrankaRobot,
    GimArmConfig,
    LegacyObservationAdapter,
    MethodArm,
    MethodEndEffector,
    PartGroup,
    Robot,
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotPart,
    Turtle2Config,
    VectorActionAdapter,
    VectorActionBinding,
    register_robot,
)
from rlinf.robotics.parts.arms import (
    FrankaROSArm,
    FrankyArm,
    GimArm,
    Turtle2Connection,
)
from rlinf.robotics.parts.arms.franka import FrankaRobotState
from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareResource,
    NodeHardwareConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class FakePart(RobotPart):
    """Record lifecycle calls through the standard driver hooks."""

    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    @property
    def observation_features(self) -> dict[str, dict]:
        return {"state": {"shape": (1,)}}

    def _open(self) -> Any:
        self.events.append(f"connect:{self.name}")
        return f"device:{self.name}"

    def get_observation(self) -> dict[str, np.ndarray]:
        return {"state": np.array([1.0])}

    def _release(self, device: Any) -> None:
        self.events.append(f"disconnect:{self.name}")


class FakeControllablePart(FakePart, ControllablePart):
    @property
    def action_features(self) -> dict[str, dict]:
        return {"target": {"shape": (1,)}}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return action


class FakeEndEffector(FakePart, EndEffector):
    """Minimal end effector with the shared category interface."""

    state_dim = 1
    action_dim = 1
    control_mode = "binary"

    def get_state(self) -> np.ndarray:
        return np.array([1.0])

    def command(self, action: np.ndarray) -> bool:
        return True


class FakeCamera(FakePart, Camera):
    pass


class FakeRemoteResult:
    def __init__(self, value: Any):
        self.value = value

    def wait(self) -> list[Any]:
        return [self.value]


class FakeMethodDriver:
    """Expose arm and gripper operations as named methods."""

    def __init__(self):
        self.state = FrankaRobotState(gripper_position=0, gripper_open=False)
        self.calls: list[tuple[str, Any]] = []
        self.is_connected = True

    def get_state(self) -> FrankaRobotState:
        return self.state

    def move_arm(self, target: np.ndarray) -> None:
        self.calls.append(("move_arm", target))

    def open_gripper(self) -> None:
        self.calls.append(("open_gripper", None))

    def close_gripper(self) -> None:
        self.calls.append(("close_gripper", None))

    def is_robot_up(self) -> bool:
        """Return a driver-specific status used by real environments."""
        return True


class FakeWorkerGroup:
    """Record calls forwarded to a single hosted connection."""

    def __init__(self, values: Optional[dict[str, Any]] = None):
        self.calls: list[tuple[str, Any]] = []
        self.values = values or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> FakeRemoteResult:
            self.calls.append((name, args[0] if len(args) == 1 else args or None))
            if name == "attribute":
                return FakeRemoteResult(self.values.get(args[0]))
            return FakeRemoteResult(self.values.get(name))

        return call

    def _close(self) -> None:
        self.calls.append(("_close", None))


def test_robot_composes_and_namespaces_parts():
    events: list[str] = []
    arm = PartGroup(
        arm=FakeControllablePart("arm", events),
        gripper=FakeEndEffector("gripper", events),
        wrist=FakeCamera("wrist", events),
    )
    robot = Robot(left=arm, front=FakeCamera("front", events))

    robot.connect()

    assert robot.is_connected
    assert events == [
        "connect:arm",
        "connect:gripper",
        "connect:wrist",
        "connect:front",
    ]
    # Observation and action namespaces follow the composed part names.
    assert set(robot.observation_features) == {"left", "front"}
    assert set(robot.action_features) == {"left"}, "a camera takes no action"
    action = {
        "left": {
            "arm": {"target": np.array([0.5])},
            "gripper": {"target": np.array([1.0])},
        }
    }
    assert robot.send_action(action) == action
    assert set(robot.parts_of_type(Camera)) == {"left.wrist", "front"}

    robot.disconnect()
    assert events[-4:] == [
        "disconnect:front",
        "disconnect:wrist",
        "disconnect:gripper",
        "disconnect:arm",
    ]


def test_robot_rejects_actions_for_observation_only_parts():
    robot = Robot(camera=FakeCamera("camera", []))

    with pytest.raises(TypeError, match="not controllable"):
        robot.send_action({"camera": {"target": np.array([0.5])}})


def test_robot_disconnects_remaining_arm_parts_after_camera_failure():
    events: list[str] = []
    camera = FakeCamera("wrist", events)
    arm = PartGroup(arm=FakeControllablePart("driver", events), wrist=camera)
    robot = Robot(arm=arm)
    robot.connect()
    camera.disconnect()

    robot.disconnect()

    assert "disconnect:driver" in events


def test_driver_views_expose_composed_part_api():
    driver = FakeMethodDriver()
    arm = MethodArm(
        driver,
        commands={"tcp_pose": "move_arm"},
        state_fields=("tcp_pose", "arm_joint_position"),
    )
    end_effector = MethodEndEffector(driver, state_field="gripper_position")
    target = np.ones(7)

    assert set(arm.get_observation()) == {"tcp_pose", "arm_joint_position"}
    assert end_effector.get_observation()["state"].tolist() == [0]
    assert arm.send_action({"tcp_pose": target})["tcp_pose"] is target
    end_effector.send_action({"target": np.array([1.0])})
    end_effector.send_action({"target": np.array([-1.0])})

    assert [name for name, _ in driver.calls] == [
        "move_arm",
        "open_gripper",
        "close_gripper",
    ]


def test_letting_a_placed_connection_go_closes_it_before_killing_it():
    from rlinf.robotics.placement import shutdown

    group = FakeWorkerGroup()
    shutdown(group)

    assert [name for name, _ in group.calls] == ["disconnect", "_close"]


def test_a_placed_connection_forwards_off_interface_driver_methods():
    from rlinf.robotics.placement import remote_view_of

    view = remote_view_of(FakeMethodDriver)
    placed = object.__new__(view)
    placed._group = FakeWorkerGroup({"is_robot_up": True})

    assert placed.is_robot_up() is True


def test_robot_requires_non_empty_string_part_names():
    with pytest.raises(ValueError, match="non-empty strings"):
        Robot(parts={0: FakePart("camera", [])})  # type: ignore[dict-item]


def test_builtin_robots_expose_standard_composition_layouts():
    events: list[str] = []
    left_arm = PartGroup(
        arm=FakeControllablePart("left_arm", events),
        gripper=FakeEndEffector("left_gripper", events),
    )
    right_arm = PartGroup(
        arm=FakeControllablePart("right_arm", events),
        gripper=FakeEndEffector("right_gripper", events),
    )
    third_arm = PartGroup(arm=FakeControllablePart("third_arm", events))

    single = FrankaRobot(arm=left_arm, front_camera=FakeCamera("front", events))
    dual = DualFrankaRobot(
        left=left_arm, right=right_arm, base_camera=FakeCamera("base", events)
    )
    # Composition supports arbitrary names and part counts.
    triple = FrankaRobot(left=left_arm, right=right_arm, third=third_arm)

    assert set(single.children) == {"arm", "front_camera"}
    assert set(single.parts_of_type(PartGroup)) == {"arm"}
    assert set(single.parts_of_type(EndEffector)) == {"arm.gripper"}
    assert set(single.parts_of_type(Camera)) == {"front_camera"}
    assert set(dual.children) == {"left", "right", "base_camera"}
    assert set(dual.parts_of_type(PartGroup)) == {"left", "right"}
    assert set(triple.children) == {"left", "right", "third"}
    assert set(triple.parts_of_type(PartGroup)) == {"left", "right", "third"}


def test_register_robot_registers_policy_and_config(monkeypatch):
    monkeypatch.setattr(RobotDiscovery, "registry", RobotDiscovery.registry.copy())
    monkeypatch.setattr(Hardware, "hw_types", Hardware.hw_types.copy())
    monkeypatch.setattr(Hardware, "policy_registry", Hardware.policy_registry.copy())
    monkeypatch.setattr(
        NodeHardwareConfig,
        "_hardware_config_registry",
        NodeHardwareConfig._hardware_config_registry.copy(),
    )

    @dataclass
    class TestRobotConfig(RobotConfig):
        connection: str = "loopback"

    class TestRobot(Robot):
        ROBOT_TYPE = "TestRobot"

    class TestRobotDiscovery(RobotDiscovery):
        HW_TYPE = "TestRobot"

        @classmethod
        def enumerate(
            cls,
            node_rank: int,
            configs: Optional[list[HardwareConfig]] = None,
        ) -> Optional[HardwareResource]:
            return None

    registered = register_robot(TestRobotConfig, TestRobot)(TestRobotDiscovery)
    parsed = NodeHardwareConfig(
        type="TestRobot",
        configs=cast(Any, [{"node_rank": 3, "connection": "robot.local"}]),
    )

    assert registered is TestRobotDiscovery
    assert RobotDiscovery.registry["TestRobot"].robot_cls is TestRobot
    assert TestRobotDiscovery in Hardware.policy_registry
    assert isinstance(parsed.configs[0], TestRobotConfig)
    assert parsed.configs[0].connection == "robot.local"

    with pytest.raises(ValueError, match="already registered"):
        register_robot(TestRobotConfig, TestRobot)(TestRobotDiscovery)
    assert NodeHardwareConfig._hardware_config_registry["TestRobot"] is TestRobotConfig


def test_scheduler_does_not_export_concrete_robot_types():
    import rlinf.scheduler as scheduler

    assert not hasattr(scheduler, "FrankaConfig")
    assert not hasattr(scheduler, "FrankaHWInfo")


def test_robot_auto_config_supports_pep604_optional(monkeypatch):
    @dataclass
    class TestConfig(RobotConfig):
        port: int | None = None

    config = TestConfig(node_rank=0)
    monkeypatch.setenv("PORT", "5000")

    assert RobotAutoConfig.resolve([config])[0].port == 5000


def test_namespaces_follow_the_composition():
    robot = Robot(arm=PartGroup(arm=FakeControllablePart("arm", [])))
    robot.connect()

    observation = robot.get_observation()
    action = {"arm": {"arm": {"target": np.array([0.25])}}}

    assert observation["arm"]["arm"]["state"].shape == (1,)
    assert robot.send_action(action) == action
    robot.disconnect()


def test_a_connection_hands_out_the_part_it_backs_not_a_controllable_one():
    class CameraOnlyHost(FakePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"wrist": FakeCamera("wrist", [])}

    wrist = CameraOnlyHost("host", []).part("wrist")

    assert isinstance(wrist, Camera)
    assert not isinstance(wrist, ControllablePart)
    with pytest.raises(TypeError, match="not controllable"):
        Robot(wrist=wrist).send_action({"wrist": {}})


def test_legacy_adapters_preserve_policy_facing_layouts():
    canonical_observation = {
        "arms": {
            "left": {"state": {"joint_position": np.arange(6)}},
            "right": {"state": {"joint_position": np.arange(6, 12)}},
        },
        "cameras": {
            "front": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        },
    }
    observation_adapter = LegacyObservationAdapter(
        state_fields={
            "left_joint_position": ("arms", "left", "state", "joint_position"),
            "right_joint_position": (
                "arms",
                "right",
                "state",
                "joint_position",
            ),
        },
        frame_fields={"front": ("cameras", "front", "rgb")},
    )
    action_adapter = VectorActionAdapter(
        action_dim=12,
        bindings=[
            VectorActionBinding(("arms", "left", "arm", "target"), 0, 6),
            VectorActionBinding(("arms", "right", "arm", "target"), 6, 12),
        ],
    )

    legacy_observation = observation_adapter.adapt(canonical_observation)
    canonical_action = action_adapter.adapt(np.arange(12))

    assert set(legacy_observation) == {"state", "frames"}
    assert legacy_observation["state"]["left_joint_position"].tolist() == list(range(6))
    assert canonical_action["arms"]["right"]["arm"]["target"].tolist() == list(
        range(6, 12)
    )


def test_all_builtin_configs_construct_from_a_node_rank_alone():
    configs = [
        GimArmConfig(node_rank=0),
        DOSW1RobotConfig(node_rank=0),
        Turtle2Config(node_rank=0),
    ]

    assert all(config.node_rank == 0 for config in configs)


def test_every_registered_robot_can_skip_the_enumeration_probe():
    """Validation is uniform across robots, so the opt-out must be too.

    Enumeration checks the cameras a config names, which needs the camera SDK
    on the enumerating node. Every robot config has to offer the same way out,
    or a node without that SDK cannot declare the robot at all.
    """
    registry = RobotDiscovery.registry

    without = sorted(
        name
        for name, reg in registry.items()
        if "disable_validate" not in {f.name for f in fields(reg.config_cls)}
    )
    assert without == []

    # The flag has to reach enumeration, not merely exist on the config.
    discovery = registry["DOSW1"].discovery_cls
    probed = []

    class Probing(discovery):
        @classmethod
        def validate(cls, config, node_rank):
            probed.append(config)

    Probing.enumerate(0, [DOSW1RobotConfig(node_rank=0, camera_serials=["cam"])])
    assert len(probed) == 1

    Probing.enumerate(
        0,
        [DOSW1RobotConfig(node_rank=0, camera_serials=["cam"], disable_validate=True)],
    )
    assert len(probed) == 1


def test_every_registered_robot_carries_a_builder():
    registry = RobotDiscovery.registry

    assert set(registry) >= {"Franka", "DualFranka", "GimArm", "Turtle2", "DOSW1"}
    missing = sorted(name for name, reg in registry.items() if reg.build is None)
    assert missing == []


def test_dosw1_dummy_runtime_uses_composed_dual_arm_interface():
    robot = DOSW1Robot.build(is_dummy=True)
    robot.connect()

    assert set(robot.children) == {"left", "right"}
    observation = robot.get_observation()
    assert observation["left"]["arm"]["joint_position"].shape == (6,)
    robot.disconnect()
    assert not robot.is_connected


def test_pure_drivers_construct_without_scheduler_or_vendor_sdks():
    from rlinf.robotics.parts.base import Connection

    # Single-arm connections are themselves controllable parts.
    arms = [
        FrankaROSArm("10.0.0.1"),
        FrankyArm("10.0.0.1"),
        GimArm("can0", "gim_arm_xl", True, "parallel"),
    ]
    # Multi-part buses are connections, not parts.
    buses = [Turtle2Connection()]

    assert all(isinstance(driver, ControllablePart) for driver in arms)
    # Both forms participate in the connection lifecycle; only arms are readable.
    assert all(isinstance(driver, Connection) for driver in arms + buses)
    assert all(isinstance(driver, RobotPart) for driver in arms)
    assert not any(isinstance(driver, RobotPart) for driver in buses)
    assert all(not driver.is_connected for driver in arms + buses)
    # A connection exports the parts it genuinely backs. A Franka arm backs
    # none: its end effector answers on its own endpoint and is composed
    # beside it. A GimArm gripper shares the arm's bus, so the arm exports it.
    assert not FrankaROSArm("10.0.0.1").parts
    assert not FrankyArm("10.0.0.1").parts
    assert all(driver.parts for driver in buses)
    assert GimArm("can0", "gim_arm_xl", True, "parallel").parts


class _BareArm(Arm):
    """Minimal arm used by placement and backend-selection tests."""

    @classmethod
    def declare(cls, address, **settings):
        """Declare the arm while preserving placement settings."""
        placement = {
            name: settings.pop(name)
            for name in ("node_rank", "worker_name")
            if name in settings
        }
        return cls(address, **placement)

    @property
    def observation_features(self) -> dict:
        return {}

    @property
    def action_features(self) -> dict:
        return {}

    def get_observation(self) -> dict:
        return {}

    def send_action(self, action):
        return action


def _opens_here(monkeypatch):
    """Force declared connections to open in the current process."""
    from dataclasses import replace

    from rlinf.robotics.parts.base import Connection

    connect = Connection.connect

    def connect_here(self):
        if self._remote_info is not None and self._remote_info.node_rank is not None:
            self._remote_info = replace(self._remote_info, node_rank=None)
        connect(self)

    monkeypatch.setattr(Connection, "connect", connect_here)


def _fake_arm_backend(monkeypatch, *, failing_ip=None, disconnected=None):
    """Register a fake arm backend and select it for Franka robots."""
    from rlinf.robotics.parts.arms.base import Arm

    _opens_here(monkeypatch)

    class FakeArm(_BareArm):
        def __init__(self, robot_ip, *_args, **_kwargs):
            self.robot_ip = robot_ip

        @property
        def parts(self):
            return {}

        @property
        def observation_features(self):
            return {"state": {"shape": (1,)}}

        @property
        def action_features(self):
            return {"target": {"shape": (1,)}}

        def get_observation(self):
            return {"state": np.array([1.0])}

        def send_action(self, action):
            return action

        def _open(self):
            if failing_ip is not None and self.robot_ip == failing_ip:
                raise RuntimeError("right arm is unreachable")
            return f"arm:{self.robot_ip}"

        def _release(self, device):
            if disconnected is not None:
                disconnected.append(self.robot_ip)

    # Populate the registry before monkeypatch installs the temporary backend.
    Arm.backends()
    monkeypatch.setitem(Arm.__dict__["_BACKENDS"], "bench", FakeArm)
    monkeypatch.setattr(franka_module.FrankaRobot, "BACKEND", "bench")
    return FakeArm


def test_an_arm_backend_is_selected_from_the_registry_like_any_driver():
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robots import DualFrankaRobot, FrankaRobot

    assert Arm.backend("franka_ros") is FrankaROSArm
    assert Arm.backend("franky") is FrankyArm
    assert {"franka_ros", "franky"} <= set(Arm.backends())

    # The robot selects the backend by its registry name.
    assert FrankaRobot.BACKEND == "franka_ros"
    assert DualFrankaRobot.BACKEND == "franky"
    for robot in (FrankaRobot, DualFrankaRobot):
        assert Arm.backend(robot.BACKEND) is not None

    with pytest.raises(ValueError, match="Unsupported Arm backend"):
        Arm.backend("no_such_stack")


def test_a_backend_maps_the_robot_settings_onto_its_own_constructor():
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.robots import FrankaRobot

    ros = FrankaRobot.declare_arm(
        "10.0.0.2", node_rank=1, name="arm", backend="franka_ros"
    )
    assert type(ros).__name__ == "FrankaROSArm"
    assert ros.node_rank == 1, "placement must survive the mapping"

    franky = FrankaRobot.declare_arm(
        "10.0.0.3", node_rank=2, name="arm", backend="franky"
    )
    assert type(franky).__name__ == "FrankyArm"
    assert franky.node_rank == 2

    # An end effector is declared on its own, and takes its own placement.
    hand = FrankaRobot.declare_end_effector(
        "10.0.0.3", node_rank=4, name="hand", end_effector_type="ruiyan_hand"
    )
    assert type(hand).__name__ == "RuiyanHand"
    assert hand.node_rank == 4, "an end effector is placed independently"

    # End-effector settings reach the end effector, not the arm.
    with pytest.raises(TypeError, match="does not take"):
        FrankaRobot.declare_arm(
            "10.0.0.3",
            node_rank=0,
            name="arm",
            backend="franky",
            gripper_type="robotiq",
        )

    # A backend with no matching options also rejects them.
    class Plain(Arm):
        def __init__(self, address):
            self.address = address

        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        def _open(self):
            return "device"

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

    assert Plain.declare("addr", node_rank=3).node_rank == 3
    with pytest.raises(TypeError, match="does not take"):
        Plain.declare("addr", gripper_connection="/dev/ttyUSB0")


def test_every_canonical_arm_reports_the_same_fields_from_one_place():
    import inspect

    from rlinf.robotics.parts.arms.base import ARM_STATE_FIELDS, BaseArm
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.parts.arms.gim_arm import GimArm

    for driver in (FrankaROSArm, FrankyArm, GimArm):
        assert issubclass(driver, BaseArm)
        for inherited in ("observation_features", "get_observation"):
            assert inherited not in vars(driver), (
                f"{driver.__name__} writes its own {inherited}; the three of "
                "them had the same body three times"
            )
        assert set(driver.STATE_FIELDS) == set(ARM_STATE_FIELDS)
        assert "get_state" in vars(driver), f"{driver.__name__} must supply state"

    # Robot builders depend on the category interface.
    assert inspect.isabstract(BaseArm)


def test_declaring_arms_opens_nothing_until_connect(monkeypatch):
    from rlinf.robotics.parts.arms.base import Arm

    class NeverOpens(_BareArm):
        def __init__(self, *_args, **_kwargs):
            pass

        def _open(self):
            raise AssertionError("nothing may be opened while composing")

    Arm.backends()
    monkeypatch.setitem(Arm.__dict__["_BACKENDS"], "bench", NeverOpens)
    monkeypatch.setattr(franka_module.FrankaRobot, "BACKEND", "bench")

    robot = FrankaRobot(
        arm=FrankaRobot.declare_arm("10.0.0.1", node_rank=0, name="left")
    )

    assert not robot.is_connected


def test_connect_tears_down_parts_already_opened(monkeypatch):
    disconnected: list[str] = []
    _fake_arm_backend(monkeypatch, failing_ip="10.0.0.2", disconnected=disconnected)

    robot = FrankaRobot(
        left=FrankaRobot.declare_arm("10.0.0.1", node_rank=0, name="left"),
        right=FrankaRobot.declare_arm("10.0.0.2", node_rank=0, name="right"),
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert disconnected == ["10.0.0.1"]


def test_declaring_arms_scales_past_two(monkeypatch):
    _fake_arm_backend(monkeypatch)

    robot = FrankaRobot(
        **{
            name: FrankaRobot.declare_arm(f"10.0.0.{index}", node_rank=0, name=name)
            for index, name in enumerate(("left", "right", "third"), start=1)
        }
    )
    robot.connect()

    assert list(robot.children) == ["left", "right", "third"]
    assert robot.is_connected


def test_one_connection_is_opened_once_however_often_it_is_named():
    opens: list[str] = []

    class Riding(ControllablePart):
        """Borrow the connection opened by its host."""

        @property
        def observation_features(self) -> dict:
            return {}

        @property
        def action_features(self) -> dict:
            return {}

        def get_observation(self) -> dict:
            return {}

        def send_action(self, action):
            return action

    class RidingCamera(Riding, Camera):
        pass

    class CoupledHardware(Connection):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"left": Riding(), "right": Riding(), "wrist": RidingCamera()}

        def _open(self):
            opens.append("open")
            return "link"

    hardware = CoupledHardware()
    robot = Robot(
        left=hardware.part("left"),
        right=hardware.part("right"),
        wrist=hardware.part("wrist"),
    )
    robot.connect()

    assert opens == ["open"], "the shared connection was opened more than once"
    assert isinstance(robot.child("wrist"), Camera)
    assert robot.is_connected


def test_a_robot_composes_an_arm_and_gets_what_rides_on_it():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.robots import FrankaRobot, GimArmRobot

        arm = FrankaRobot.declare_arm("10.0.0.2", node_rank=0, name="arm")
        assert list(arm.parts) == [], (
            "a Franka arm backs no end effector: the hand answers on its own "
            "endpoint, so it is composed beside the arm rather than under it"
        )
        hand = FrankaRobot.declare_end_effector(
            "10.0.0.2",
            node_rank=0,
            name="hand",
            gripper_type="robotiq",
            gripper_connection="/dev/ttyUSB0",
        )

        robot = FrankaRobot(arm=arm, end_effector=hand)
        assert list(robot.children) == ["arm", "end_effector"]
        assert robot.child("arm") is arm
        assert robot.child("end_effector") is hand
        # Neither owns the other, so either can be placed on its own node.
        assert hand.owner is hand and arm.owner is arm

        # Readings preserve the composed tree structure.
        assert set(robot.observation_features["arm"]) >= {"tcp_pose"}
        assert set(robot.observation_features["end_effector"]) == {"state"}
        assert set(robot.action_features["arm"]) == {"tcp_pose"}
        assert set(robot.action_features["end_effector"]) == {"target"}

        config = {
            "node_rank": 0,
            "can_interface": "can0",
            "arm_variant": "arm6",
            "gripper_type": "default",
            "control_mode": "position",
            "env_idx": 0,
            "worker_rank": 0,
        }
        fitted = GimArmRobot.build(enable_gripper=True, **config)
        bare = GimArmRobot.build(enable_gripper=False, **config)

    assert list(fitted.child("arm").children) == ["end_effector"]
    assert list(bare.child("arm").children) == [], (
        "the arm knows whether a gripper is fitted; the robot must not decide "
        "that a second time"
    )


def test_every_env_reaches_a_real_connection_through_the_tree():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.robots import (
            DualFrankaRobot,
            FrankaRobot,
            GimArmRobot,
            Turtle2Robot,
        )

        # Exercise each driver path used by the shipped environments.
        reached = [
            (
                FrankaRobot.build(
                    robot_ip="10.0.0.2", node_rank=0, env_idx=0, worker_rank=0
                )
                .child("arm")
                .owner,
                "get_state",
            ),
            (
                GimArmRobot.build(
                    node_rank=0,
                    can_interface="can0",
                    arm_variant="arm6",
                    enable_gripper=True,
                    gripper_type="default",
                    control_mode="position",
                    env_idx=0,
                    worker_rank=0,
                )
                .child("arm")
                .owner,
                "get_state",
            ),
            (
                Turtle2Robot.build(
                    frequency=50,
                    camera_ids=(1, 2),
                    env_idx=0,
                    node_rank=0,
                    worker_rank=0,
                )
                .child("left")
                .child("arm")
                .owner,
                "get_cams",
            ),
        ]
        dual = DualFrankaRobot.build(
            left_robot_ip="1.2.3.4",
            right_robot_ip="1.2.3.5",
            left_gripper_connection="/dev/a",
            right_gripper_connection="/dev/b",
            env_idx=0,
            worker_rank=0,
            node_rank=0,
        )
        reached.append((dual.child("left").child("arm").owner, "clear_errors"))

        for connection, method in reached:
            assert not isinstance(connection, PartGroup), (
                f"the path reached a {type(connection).__name__}, not a driver"
            )
            assert hasattr(connection, method), (
                f"{type(connection).__name__} has no {method}(), which the env calls"
            )

        # Groups do not claim ownership of a connection.
        with pytest.raises(TypeError, match="rides no connection"):
            dual.child("left").owner


def _arm_with_a_camera_of_its_own(log):
    """Build an arm with an independently connected wrist camera."""

    class WristCamera(Camera):
        def _open(self):
            log.append("open:camera")
            return "usb"

        def _release(self, device):
            log.append("close:camera")

        @property
        def observation_features(self):
            return {"frame": {}}

        def get_observation(self):
            return {"frame": "IMAGE"}

    class ArmWithCamera(ControllablePart):
        def __init__(self, *args, **kwargs):
            self._camera = WristCamera()

        def _open(self):
            log.append("open:arm")
            return "arm"

        def _release(self, device):
            log.append("close:arm")

        @property
        def observation_features(self):
            return {"q": {}}

        @property
        def action_features(self):
            return {}

        def get_observation(self):
            return {"q": 0}

        def send_action(self, action):
            return action

        @property
        def parts(self):
            return {"wrist": self._camera}

    return ArmWithCamera()


def test_a_rider_may_not_shadow_one_of_its_carriers_own_fields():
    class Rider(RobotPart):
        @property
        def observation_features(self):
            return {"v": {}}

        def get_observation(self):
            return {"v": "RIDER"}

    class Shadowed(ControllablePart):
        def _open(self):
            return "arm"

        @property
        def observation_features(self):
            return {"tcp_pose": {}}

        @property
        def action_features(self):
            return {"tcp_pose": {}}

        def get_observation(self):
            return {"tcp_pose": "ARM"}

        def send_action(self, action):
            return action

        @property
        def parts(self):
            return {"tcp_pose": Rider()}

    with pytest.raises(ValueError, match="also its own observation or action"):
        Shadowed().children


def test_a_constructor_reaches_no_hardware_and_no_vendor_library():
    import sys
    import warnings

    from rlinf.robotics.parts.arms.gim_arm import GimArm
    from rlinf.robotics.parts.end_effectors.hands.ruiyan import RuiyanHand

    sys.modules.pop("rlinf_dexhand", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hand = RuiyanHand(port="/dev/ttyUSB9", node_rank=1)
        arm = GimArm("can-not-here", "arm6", True, "default", "position", node_rank=2)

    assert "rlinf_dexhand" not in sys.modules, (
        "RuiyanHand imported its vendor SDK while composing, so a hand bound "
        "for another node needs that package on this one"
    )
    assert not [w for w in caught if "CAN interface" in str(w.message)], (
        "GimArm warned about the composing machine's CAN bus for an arm that "
        "is going to node 2"
    )
    assert not hand.is_connected and not arm.is_connected


def test_a_rider_holding_its_own_link_is_opened_by_the_robot():
    log: list[str] = []
    arm = _arm_with_a_camera_of_its_own(log)
    robot = Robot(arm=arm)

    assert len(robot.owners()) == 2, (
        "the camera rides no connection but its own, so it is a second one to "
        f"open; owners() found {[type(o).__name__ for o in robot.owners()]}"
    )

    robot.connect()
    assert log == ["open:arm", "open:camera"]
    assert arm.child("wrist").is_connected

    robot.disconnect()
    assert log[-2:] == ["close:camera", "close:arm"], "closed newest first"


def test_the_tree_holds_the_same_objects_every_time_it_is_walked():
    log: list[str] = []
    robot = Robot(arm=_arm_with_a_camera_of_its_own(log))

    once = robot.child("arm").child("wrist")
    assert robot.child("arm").child("wrist") is once
    assert robot.named_parts["arm.wrist"] is once
    assert robot.parts_of_type(Camera)["arm.wrist"] is once

    robot.connect()
    try:
        assert once.is_connected, "the object the tree hands out is the one opened"
    finally:
        robot.disconnect()


def test_a_failed_connect_leaves_nothing_open():
    released: list[Any] = []

    class Balky(RobotPart):
        def _open(self):
            return "device"

        def _opened(self):
            raise RuntimeError("the capture loop would not start")

        def _release(self, device):
            released.append(device)

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    part = Balky()
    with pytest.raises(RuntimeError, match="capture loop"):
        part.connect()

    assert released == ["device"], "the device was not handed back"
    assert not part.is_connected


def test_a_device_is_released_even_when_teardown_throws():
    released: list[Any] = []

    class Balky(RobotPart):
        def _open(self):
            return "device"

        def _closing(self):
            raise RuntimeError("the capture loop would not stop")

        def _release(self, device):
            released.append(device)

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    part = Balky()
    part.connect()
    with pytest.raises(RuntimeError, match="capture loop"):
        part.disconnect()

    assert released == ["device"]
    assert not part.is_connected


def test_a_handle_that_is_falsy_is_still_a_handle():
    for handle in (0, "", 0.0, []):

        class Zero(RobotPart):
            def _open(self):
                return handle

            @property
            def observation_features(self):
                return {}

            def get_observation(self):
                return {}

        part = Zero()
        part.connect()
        assert part._device is handle, f"{handle!r} was replaced by the part"
        assert part.is_connected
        part.disconnect()


def test_rollback_closes_every_connection_even_if_one_will_not():
    log: list[str] = []

    def part(tag, fail_open=False, fail_close=False):
        class Flaky(RobotPart):
            def _open(self):
                if fail_open:
                    raise RuntimeError("hardware unreachable")
                log.append(f"open:{tag}")
                return tag

            def _release(self, device):
                if fail_close:
                    raise RuntimeError("this one will not close")
                log.append(f"close:{tag}")

            @property
            def observation_features(self):
                return {}

            def get_observation(self):
                return {}

        return Flaky()

    group = PartGroup(
        first=part("first"),
        second=part("second", fail_close=True),
        third=part("third", fail_open=True),
    )

    with pytest.raises(RuntimeError):
        group.connect()

    assert "close:first" in log, (
        f"the rollback stopped at the connection that would not close: {log}"
    )


def test_a_device_with_its_own_link_keeps_it_when_a_connection_lists_it():
    events: list[str] = []

    class WristCamera(FakeCamera):
        pass

    class ArmWithCamera(FakeControllablePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"arm": self, "wrist": WristCamera("wrist", events, node_rank=3)}

    arm = ArmWithCamera("arm", events)
    wrist = arm.part("wrist")

    assert wrist.owner is wrist, "the camera was adopted by the arm"
    assert wrist.node_rank == 3, "the camera lost the node it named"

    arm.connect()
    assert arm.is_connected
    assert not wrist.is_connected, (
        "the camera reported itself connected off the back of the arm's link"
    )

    # Views without their own connection are adopted by the host.
    gripper = MethodEndEffector(arm, state_field="gripper_position")
    assert gripper.owner is arm


def test_a_connection_answers_its_parts_before_it_is_opened():
    events: list[str] = []

    class HostWithSubparts(FakeControllablePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"arm": self, "end_effector": FakeEndEffector("ee", events)}

    host = HostWithSubparts("host", events)

    assert not host.is_connected
    assert set(host.parts) == {"arm", "end_effector"}
    assert isinstance(host.part("arm"), RobotPart)
    assert isinstance(host.part("end_effector"), EndEffector)
    assert events == [], "asking what a connection backs must not open it"


def test_any_connection_can_be_placed_not_only_arms():
    events: list[str] = []
    camera = FakeCamera("wrist", events, node_rank=2)

    assert camera.node_rank == 2
    assert FakeCamera("bench", events).node_rank is None
    assert events == [], "declaring where a camera runs must not open it"


def test_every_robot_owns_its_construction():
    registry = RobotDiscovery.registry

    for name, registration in registry.items():
        build = registration.build
        assert build is not None, f"{name} registered no builder"
        assert getattr(build, "__self__", None) is registration.robot_cls, (
            f"{name}'s builder is not bound to {registration.robot_cls.__name__}"
        )


def test_dual_franka_inherits_declaration_from_franka():
    assert issubclass(DualFrankaRobot, FrankaRobot)
    # DualFranka inherits arm declaration and changes only arm construction.
    assert DualFrankaRobot.declare_arm.__func__ is FrankaRobot.declare_arm.__func__
    assert DualFrankaRobot.build_arms.__func__ is not FrankaRobot.build_arms.__func__, (
        "only the arm count differs, and that is what build_arms says"
    )
    # The backend selection applies independently of arm count.
    assert (FrankaRobot.BACKEND, DualFrankaRobot.BACKEND) == ("franka_ros", "franky")
    # Arm construction contains the remaining single/dual distinction.
    overridden = [
        name
        for name in ("declare_arm", "build_arms", "build_cameras", "build")
        if getattr(DualFrankaRobot, name).__func__
        is not getattr(FrankaRobot, name).__func__
    ]
    assert overridden == ["build_arms"]


def test_every_part_places_independently_whatever_it_is():
    def fake(base):
        class Fake(base):
            state_dim = action_dim = 1
            control_mode = "binary"

            def __init__(self, *args, **kwargs):
                pass

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def _open(self):
                return "device"

            def get_observation(self):
                return {}

            def get_state(self):
                return np.array([0.0])

            def send_action(self, action):
                return action

            def command(self, action):
                return True

        return Fake

    arm = fake(ControllablePart)("10.0.0.1", node_rank=1)
    gripper = fake(EndEffector)(port="/dev/ttyUSB0", node_rank=2)
    wrist = fake(Camera)(node_rank=3)
    robot = Robot(arm=PartGroup(arm=arm, gripper=gripper, wrist=wrist))

    assert [part.node_rank for part in (arm, gripper, wrist)] == [1, 2, 3]
    # Three independent connections require three opens.
    assert len(robot.owners()) == 3
    assert [part.owner for part in (arm, gripper, wrist)] == [arm, gripper, wrist]


def test_a_leaf_part_placed_remotely_is_still_the_part_it_was():
    from rlinf.robotics.placement import remote_view_of

    class Leaf(Camera):
        @property
        def observation_features(self):
            return {"frame": {}}

        def _open(self):
            return object()

        def get_observation(self):
            return {"frame": None}

    view = remote_view_of(Leaf)

    assert issubclass(view, Leaf) and issubclass(view, Camera)
    assert "get_observation" in view.__dict__, "the reading call must travel"
    assert isinstance(view.__dict__["observation_features"], property), (
        "a property is not callable, so a worker group would not bind it; the "
        "view has to read it through the attribute call instead"
    )
    assert "parts" not in view.__dict__, "composition is answered here"
    assert Leaf().parts == {}


def test_declaring_cameras_needs_no_config_class():
    from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo

    info = CameraInfo(name="scene", serial_number="123", camera_type="realsense")
    declared = Camera.declare({"scene": info}, node_rank=4)

    assert set(declared) == {"scene"}
    assert declared["scene"].node_rank == 4
    assert type(declared["scene"]).__name__ == "RealSenseCamera"
    assert not declared["scene"].is_connected, "declaring a camera must not open it"
    assert Camera.declare(None) == {}
    # Backend resolution uses the category registry.
    assert set(BaseCamera.backends()) >= {"realsense", "zed", "lumos"}
    with pytest.raises(ValueError, match="Unsupported BaseCamera backend"):
        BaseCamera.backend("no-such-camera")


def test_failed_connect_can_be_retried():
    opened: list[str] = []
    state = {"failing": True}

    def maker(tag, flaky=False):
        class Made(ControllablePart):
            def __init__(self, *args, **kwargs):
                pass

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def _open(self):
                if flaky and state["failing"]:
                    raise RuntimeError("hardware unreachable")
                opened.append(tag)
                return object()

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

        return Made

    left = maker("left")()
    robot = Robot(left=left, right=maker("right", flaky=True)())

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert robot.child("left") is left, "the tree did not go back to what it held"
    assert not left.is_connected, (
        "the arm that opened was left connected with nobody holding it"
    )

    state["failing"] = False
    robot.connect()

    assert opened == ["left", "left", "right"], "the retry did not re-open"
    assert robot.is_connected

    robot.disconnect()
    assert not left.is_connected
    robot.connect()
    assert robot.is_connected, "a disconnected robot must be connectable again"


def test_a_robot_owned_camera_is_opened_and_closed_like_any_other_part(monkeypatch):
    events: list[str] = []

    class Cam(Camera):
        def __init__(self, *args, **kwargs):
            pass

        @property
        def observation_features(self):
            return {"frame": {}}

        def _open(self):
            events.append(f"open@{self.node_rank}")
            return "camera"

        def _release(self, device):
            events.append("close")

        def get_observation(self):
            return {"frame": None}

    camera = Cam(node_rank=5)
    robot = Robot(arm=PartGroup(arm=FakeControllablePart("arm", []), wrist=camera))

    assert camera.node_rank == 5, "the camera did not keep the node it named"

    _opens_here(monkeypatch)
    robot.connect()
    robot.disconnect()

    assert events == ["open@None", "close"]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


#: Every package that resolves its exports through a lazy ``__getattr__``.
#: Each needs a TYPE_CHECKING block, or an editor cannot follow the import.
_LAZY_PACKAGES = (
    ("rlinf/robotics/__init__.py", "rlinf.robotics"),
    ("rlinf/robotics/parts/arms/__init__.py", "rlinf.robotics.parts.arms"),
    ("rlinf/envs/real/__init__.py", "rlinf.envs.real"),
)


def _typed_names(source: str) -> set[str]:
    """Return the names declared for a type checker in a TYPE_CHECKING block."""
    typed: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)):
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.ImportFrom):
                typed |= {alias.asname or alias.name for alias in statement.names}
    return typed


def _module_defined(source: str) -> set[str]:
    """Return the names the module defines outright, which need no declaring."""
    defined: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
    return defined


@pytest.mark.parametrize(("path", "package"), _LAZY_PACKAGES)
def test_a_lazy_package_still_types_what_it_exports(path, package):
    """A name resolved at run time must also be declared for a type checker.

    Without the declaration the import still works, but go-to-definition and
    completion do not, because nothing runs ``__getattr__`` at edit time.
    """
    import importlib

    source = (_ROOT / path).read_text()
    typed = _typed_names(source)
    exported = set(importlib.import_module(package).__all__)

    # A name the module defines itself is already visible to a type checker.
    missing = sorted(exported - typed - _module_defined(source))
    assert missing == [], (
        f"exported but invisible to a type checker: {missing}. "
        f"Add them to the TYPE_CHECKING block in {path}."
    )

    extra = sorted(typed - exported)
    assert extra == [], f"declared but not exported: {extra}"


@pytest.mark.parametrize(("path", "package"), _LAZY_PACKAGES)
def test_a_lazy_package_exports_only_names_that_exist(path, package):
    """``__all__`` is written out, so nothing can drift out from under it.

    A computed ``__all__`` agrees with its own map by construction, which is
    how ``DOSW1ConnectionConfig`` survived in this list after the class was
    gone. Ruff's F822 catches that only against a literal.
    """
    import importlib

    module = importlib.import_module(package)

    assert module.__all__ == sorted(module.__all__), f"{path}: __all__ is not sorted"
    dangling = [name for name in module.__all__ if not hasattr(module, name)]
    assert dangling == [], f"{path}: exported but does not exist: {dangling}"


def test_scheduler_has_no_robotics_dependency():
    scheduler_dir = _ROOT / "rlinf" / "scheduler"
    offenders = {
        path.relative_to(_ROOT): module
        for path in scheduler_dir.rglob("*.py")
        for module in _imports(path)
        if module == "rlinf.robotics" or module.startswith("rlinf.robotics.")
    }

    assert offenders == {}


_SCHEDULER_BRIDGE = Path("rlinf") / "robotics" / "placement" / "handles.py"


def test_robotics_devices_do_not_depend_on_the_scheduler_or_gym():
    robotics_dir = _ROOT / "rlinf" / "robotics"
    device_paths = [
        robotics_dir / "robot.py",
        robotics_dir / "adapters.py",
        *robotics_dir.joinpath("parts").rglob("*.py"),
    ]
    forbidden = ("gymnasium", "rlinf.scheduler")
    offenders = {
        path.relative_to(_ROOT): module
        for path in device_paths
        if path.relative_to(_ROOT) != _SCHEDULER_BRIDGE
        for module in _imports(path)
        if module == forbidden or module.startswith(forbidden)
    }

    assert offenders == {}


def test_scheduler_use_is_confined_to_the_composition_layer():
    robotics_dir = _ROOT / "rlinf" / "robotics"
    allowed = {
        _SCHEDULER_BRIDGE,
        Path("rlinf") / "robotics" / "discovery" / "registry.py",
    }
    importers = {
        path.relative_to(_ROOT)
        for path in robotics_dir.rglob("*.py")
        for module in _imports(path)
        if module == "rlinf.scheduler" or module.startswith("rlinf.scheduler.")
    }
    leaks = {
        path
        for path in importers
        if path not in allowed and path.parent.name != "robots"
    }

    assert leaks == set()


def test_realworld_environments_do_not_own_controller_workers():
    realworld_dir = _ROOT / "rlinf" / "envs" / "realworld"
    legacy_controller_files = {
        "franka/franka_controller.py",
        "franka/franky_controller.py",
        "gim_arm/gim_arm_controller.py",
        "xsquare/turtle2_smooth_controller.py",
    }

    assert not any(
        realworld_dir.joinpath(path).exists() for path in legacy_controller_files
    )


def test_env_packages_live_under_sim_or_real():
    envs = _ROOT / "rlinf" / "envs"
    # Ignore stale bytecode left by a moved package.
    stray = sorted(
        path.name
        for path in envs.iterdir()
        if path.is_dir()
        and path.name not in {"sim", "real", "venv", "wrappers", "__pycache__"}
        and any(path.rglob("*.py"))
    )

    assert stray == []
    assert "_MovedEnvFinder" not in (envs / "__init__.py").read_text()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib\n"
            "for name in ('rlinf.envs.realworld', 'rlinf.envs.maniskill'):\n"
            "    try:\n"
            "        importlib.import_module(name)\n"
            "    except ModuleNotFoundError:\n"
            "        continue\n"
            "    raise AssertionError(name + ' still resolves')\n"
            "import rlinf.envs.real, rlinf.envs.sim.maniskill, rlinf.envs.utils\n",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_teleop_devices_are_parts_like_any_other_hardware():
    from rlinf.robotics.parts.base import RobotPart
    from rlinf.robotics.parts.teleop import Gello, Glove, Pico, SpaceMouse

    devices = (SpaceMouse, Gello, Glove, Pico)

    assert all(issubclass(device, RobotPart) for device in devices)
    assert not (_ROOT / "rlinf" / "envs" / "real" / "teleop" / "devices").exists()

    # Declaration remains hardware-free and supports offline description.
    mouse = SpaceMouse()
    assert not mouse.is_connected
    assert sorted(mouse.observation_features) == ["buttons", "twist"]


def test_the_guide_names_every_device_that_runs_standalone():
    """The list in the guide is what an operator reaches for first.

    It named three devices while four have an entry point, so the SO-101
    leader could not be checked on its own even though it can be.
    """
    devices = _ROOT / "rlinf" / "robotics" / "parts" / "teleop"
    standalone = {
        path.stem
        for path in devices.glob("*.py")
        if '__name__ == "__main__"' in path.read_text()
    }
    guide = (
        _ROOT / "docs" / "source-en" / "rst_source" / "guides" / "teleoperation.rst"
    ).read_text()
    section = guide.split("Check a Device First")[1].split("Compose Devices")[0]

    missing = sorted(name for name in standalone if f"``{name}``" not in section)
    assert missing == [], (
        f"these run standalone but the guide does not say so: {missing}"
    )


def test_teleop_devices_do_not_import_gymnasium():
    devices = _ROOT / "rlinf" / "robotics" / "parts" / "teleop"
    modules = sorted(devices.glob("*.py"))
    assert len(modules) > 5, "teleop device modules moved; update this guard"
    offenders = {
        path.name
        for path in modules
        if re.search(r"^\s*(import|from)\s+gymnasium\b", path.read_text(), re.M)
    }

    assert offenders == set()


# DOSW1 production composition through its hardware-free dummy mode


def _dosw1_robot():
    from rlinf.robotics.robots import DOSW1Robot

    return DOSW1Robot.build(is_dummy=True)


def test_building_a_real_robot_touches_no_hardware():
    from rlinf.robotics.parts.base import PartGroup, RobotPart

    robot = _dosw1_robot()

    assert not robot.is_connected
    # Every exported part shares the same DOSW1 SDK session.
    leaves = [
        leaf
        for part in robot.children.values()
        for leaf in (part.children.values() if isinstance(part, PartGroup) else [part])
    ]
    assert leaves, "the robot composed no parts"
    assert all(isinstance(leaf, RobotPart) for leaf in leaves)
    # Ownership is based on connection identity, not constructor equality.
    sessions = [leaf.owner for leaf in leaves]
    assert all(session is sessions[0] for session in sessions)
    assert not sessions[0].is_connected, "composing the robot opened the session"


def test_parts_sharing_a_session_are_never_read_concurrently():
    robot = _dosw1_robot()

    assert len(robot.owners()) == 1, "DOSW1 should present exactly one session"
    assert robot._batches() == [["left", "right"]], (
        f"both sides ride one session but were batched as {robot._batches()}"
    )


def test_parts_on_separate_connections_still_run_together():
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot

    robot = DualFrankaRobot.build(
        left_robot_ip="1.2.3.4",
        right_robot_ip="1.2.3.5",
        left_gripper_connection="/dev/ttyUSB0",
        right_gripper_connection="/dev/ttyUSB1",
        env_idx=0,
        worker_rank=0,
        node_rank=0,
    )

    # Four connections now: an arm and a hand on each side.
    assert len(robot.owners()) == 4
    assert robot._batches() == [["left"], ["right"]]
    # The hand no longer shares the arm's connection, so the two run at once.
    assert robot.child("left")._batches() == [["arm"], ["end_effector"]]
    assert set(robot.child("left").children) >= {"arm", "end_effector"}


def test_a_group_spanning_two_sessions_pulls_both_into_one_batch():
    class Riding(RobotPart):
        @property
        def observation_features(self) -> dict:
            return {}

        def get_observation(self) -> dict:
            return {}

    class Session(RobotPart):
        @property
        def observation_features(self) -> dict:
            return {}

        def get_observation(self) -> dict:
            return {}

        def _open(self):
            return "link"

        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"a": Riding(), "b": Riding()}

    first, second = Session(), Session()
    tree = PartGroup(
        x=first.part("a"),
        bridge=PartGroup(p=first.part("b"), q=second.part("a")),
        y=second.part("b"),
    )

    assert len(tree.owners()) == 2
    assert tree._batches() == [["x", "bridge", "y"]]


def test_real_robot_lifecycle_without_hardware():
    robot = _dosw1_robot()

    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected

    observation = robot.get_observation()
    assert set(observation) == {"left", "right"}
    assert set(observation["left"]) == {"arm", "gripper"}
    assert observation["left"]["arm"]["joint_position"].shape == (6,)

    applied = robot.send_action(
        {
            "left": {
                "arm": {"joint_position": observation["left"]["arm"]["joint_position"]}
            }
        }
    )
    assert set(applied) == {"left"}

    robot.reset()
    robot.disconnect()
    assert not robot.is_connected

    # Disconnect restores declarations for a later reconnect.
    robot.connect()
    assert robot.is_connected
    robot.disconnect()


def test_one_connection_is_opened_once_for_every_part_it_drives():
    robot = _dosw1_robot()

    owners = robot.owners()
    assert len(owners) == 1, (
        f"four parts on one session produced {len(owners)} connections to open"
    )
    left = robot.child("left").child("arm")
    right = robot.child("right").child("arm")
    assert left.owner is right.owner is owners[0]

    robot.connect()
    assert left.is_connected and right.is_connected
    robot.disconnect()
    assert not left.is_connected


def test_coupled_hardware_exposes_its_components_as_parts():
    from rlinf.robotics import EndEffector

    robot = _dosw1_robot()
    robot.connect()

    assert sorted(robot.named_parts) == [
        "left",
        "left.arm",
        "left.gripper",
        "right",
        "right.arm",
        "right.gripper",
    ]
    assert sorted(robot.parts_of_type(EndEffector)) == ["left.gripper", "right.gripper"]
    assert type(robot.child("left").child("arm")).__name__ == "DOSW1Arm"

    robot.disconnect()


def test_observation_tree_follows_the_composition_on_real_parts():
    robot = _dosw1_robot()
    robot.connect()

    observation = robot.get_observation()
    paths = {
        f"{group}.{part}" for group, parts in observation.items() for part in parts
    }

    assert paths == set(robot.named_parts) - set(robot.children)

    robot.disconnect()


# Teleoperation composition


def _scripted_device(reading, produces=(), value=None, driving=True):
    """Build a teleop device that reads fixedly and fills fixed parts."""
    from rlinf.robotics.parts.teleop import TeleopAction, TeleopDevice

    class Scripted(TeleopDevice):
        PRODUCES = _kinds(*produces)

        def _open(self):
            return object()

        @property
        def observation_features(self):
            return {key: {} for key in reading}

        def get_observation(self):
            return dict(reading)

        def action(self, reading, context):
            return TeleopAction(parts=dict.fromkeys(produces, value), driving=driving)

    device = Scripted()
    device.connect()
    return device


def test_disconnecting_a_spacemouse_stops_its_thread_and_closes_the_handle():
    """A daemon thread is not cleanup: it keeps polling a device nobody owns.

    ``TeleopPart._release`` only calls ``close`` or ``stop``, so a reader that
    defines neither leaks both its thread and its HID handle, and the next
    connect finds the puck taken.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import SpaceMouse

        mouse = SpaceMouse()
        mouse.connect()
        reader = mouse._device
        handle = made["pyspacemouse"].devices[-1]
        assert reader.thread.is_alive()

        mouse.disconnect()

        assert not reader.thread.is_alive()
        assert handle.closed


def test_disconnecting_a_gello_stops_its_thread_and_releases_the_port():
    """The same leak on the leader arm, which holds a serial port open."""
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import Gello

        gello = Gello(port="/dev/mock-gello")
        gello.connect()
        reader = gello._device
        agent = made["gello_teleop.gello_teleop_agent"].last_agent
        assert reader.thread.is_alive()

        gello.disconnect()

        assert not reader.thread.is_alive()
        assert agent.closed


def test_disconnecting_a_gello_joint_releases_the_port_as_well():
    """It already stopped its thread but left the serial port open."""
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.teleop import GelloJoint

        gello = GelloJoint(port="/dev/mock-gello")
        gello.connect()
        reader = gello._device
        agent = made["gello_teleop.gello_teleop_agent"].last_agent

        gello.disconnect()

        assert not reader.thread.is_alive()
        assert agent.closed


def test_a_device_can_be_reconnected_after_release():
    """Release must leave a device reopenable, not a dead thread to restart."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop import SpaceMouse

        mouse = SpaceMouse()
        mouse.connect()
        first = mouse._device
        mouse.disconnect()

        # A fresh reader, because threading.Thread cannot be restarted.
        mouse.connect()
        try:
            second = mouse._device
            assert second is not first
            assert second.thread.is_alive()
            assert mouse.get_observation()["twist"].shape == (6,)
        finally:
            mouse.disconnect()


def _counting_device(produces=("arm",)):
    """Build a teleop device that reports how many times it was read."""
    from rlinf.robotics.parts.teleop import TeleopAction, TeleopDevice

    class Counting(TeleopDevice):
        PRODUCES = _kinds(*produces)

        def __init__(self):
            super().__init__()
            self.reads = 0

        def _open(self):
            return object()

        @property
        def observation_features(self):
            return {"sample": {}}

        def get_observation(self):
            self.reads += 1
            # A changing value, so two entries mapping different reads of the
            # same instant show up as different actions rather than passing.
            return {"sample": np.full(6, self.reads, dtype=float)}

        def action(self, reading, context):
            parts = dict.fromkeys(produces, np.asarray(reading["sample"]))
            return TeleopAction(parts=parts, driving=True)

    device = Counting()
    device.connect()
    return device


def test_two_entries_sharing_a_device_read_it_once():
    """Both entries must map the same instant, not two reads of it.

    A shared device read per entry gives each a different sample of the same
    moment, so one controller would drive two arms from torn readings.
    """
    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    device = _counting_device()
    group = TeleopGroup(
        [
            TeleopEntry(device, drives="left"),
            TeleopEntry(device, drives="right"),
        ],
        available=_kinds("left.arm", "right.arm"),
    )

    parts, _, _ = group.action({})

    assert device.reads == 1
    assert np.array_equal(parts["left.arm"], parts["right.arm"])


def test_a_device_of_its_own_is_read_inside_drive():
    """The shared-read path must not cost an unshared device a second read."""
    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    left, right = _counting_device(), _counting_device()
    group = TeleopGroup(
        [TeleopEntry(left, drives="left"), TeleopEntry(right, drives="right")],
        available=_kinds("left.arm", "right.arm"),
    )

    group.action({})

    assert (left.reads, right.reads) == (1, 1)


def _kinds(*names):
    """Build the action-kind mapping expected by a robot."""
    from rlinf.robotics.actions import ActionKind

    per_name = {
        "hand": ActionKind.HAND,
        "end_effector": ActionKind.GRIPPER,
    }
    return {
        name: per_name.get(name.rsplit(".", 1)[-1], ActionKind.CARTESIAN_DELTA)
        for name in names
    }


def test_two_devices_merge_into_one_action():
    import numpy as np

    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(_scripted_device({"twist": 1}, ("arm",), np.ones(6))),
            TeleopEntry(_scripted_device({"angles": 2}, ("hand",), np.ones(6) * 2)),
        ],
        available=_kinds(*("arm", "hand")),
    )

    parts, driving, _ = group.action({})

    assert sorted(parts) == ["arm", "hand"]
    assert driving
    assert np.allclose(parts["hand"], 2.0)


def test_a_part_the_robot_lacks_is_not_filled():
    import numpy as np

    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(
                _scripted_device({"twist": 1}, ("arm", "end_effector"), np.ones(6)),
            )
        ],
        available=_kinds(*("arm", "hand")),
    )

    assert group.parts == ("arm",)


def test_two_devices_claiming_one_part_is_refused():
    import numpy as np
    import pytest

    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="both drive"):
        TeleopGroup(
            [
                TeleopEntry(_scripted_device({}, ("arm",), np.ones(6))),
                TeleopEntry(_scripted_device({}, ("arm",), np.ones(6))),
            ],
            available=_kinds(*("arm",)),
        )


def test_a_device_that_fills_nothing_is_refused():
    import numpy as np
    import pytest

    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="fills none"):
        TeleopGroup(
            [TeleopEntry(_scripted_device({}, ("hand",), np.ones(6)))],
            available=_kinds(*("arm", "end_effector")),
        )


def test_drives_separates_two_identical_leaders():
    import numpy as np

    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(_scripted_device({}, ("arm",), np.ones(7)), drives="left"),
            TeleopEntry(_scripted_device({}, ("arm",), np.ones(7) * 2), drives="right"),
        ],
        available=_kinds(*("left.arm", "right.arm")),
    )

    parts, _, _ = group.action({})

    assert sorted(parts) == ["left.arm", "right.arm"]
    assert np.allclose(parts["right.arm"], 2.0)


def test_one_device_fills_every_part_it_produces_from_one_reading():
    """A device says what it fills, so it is listed and read once."""
    import numpy as np

    from rlinf.robotics.parts.teleop import TeleopEntry, TeleopGroup

    device = _scripted_device({"twist": 1}, ("arm", "end_effector"), np.ones(6))
    group = TeleopGroup([TeleopEntry(device)], available=_kinds("arm", "end_effector"))

    assert len(group.devices) == 1
    assert group.parts == ("arm", "end_effector")

    reads = []
    plain = device.get_observation
    device.get_observation = lambda: (reads.append(1), plain())[1]
    parts, _, _ = group.action({})
    assert len(reads) == 1, "the device was read more than once for one step"
    assert sorted(parts) == ["arm", "end_effector"]


def test_a_teleop_rig_waits_until_every_reader_is_ready():
    from rlinf.robotics.parts.teleop import TeleopDevice, TeleopEntry, TeleopGroup

    class Starting(TeleopDevice):
        PRODUCES = _kinds("arm")

        def _open(self):
            return object()

        def action(self, reading, context):
            raise AssertionError("an unready reader must not be mapped")

        @property
        def ready(self):
            return False

        @property
        def observation_features(self):
            return {"twist": {}}

        def get_observation(self):
            raise AssertionError("an unready reader must not be sampled")

    device = Starting()
    device.connect()
    group = TeleopGroup(
        [TeleopEntry(device)],
        available=_kinds("arm"),
    )

    assert group.action({}) == ({}, False, {})


def test_spacemouse_reset_resyncs_the_gripper_and_reports_its_buttons():
    from rlinf.robotics.parts.teleop import SpaceMouse

    binding = SpaceMouse()
    opened = binding.action(
        {"twist": np.zeros(6), "buttons": [False, False]},
        {"gripper_open": True},
    )
    binding.on_reset()
    closed = binding.action(
        {"twist": np.zeros(6), "buttons": [True, False]},
        {"gripper_open": False},
    )

    assert opened.parts["end_effector"].item() > 0
    assert closed.parts["end_effector"].item() < 0
    assert closed.info == {"left": True, "right": False}
    dex = SpaceMouse(dexterous_hand=True).action(
        {"twist": np.zeros(6), "buttons": [True, False]},
        {"gripper_open": True},
    )
    assert dex.info == {"left": False, "right": True}


def test_leader_arm_only_takes_control_for_motion_or_an_active_gripper():
    from rlinf.robotics.parts.teleop import Gello

    context = {
        "tcp_pose": np.array([0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0]),
        "action_scale": np.array([0.05, 0.3, 1.0]),
    }
    idle = {
        "position": context["tcp_pose"][:3],
        "orientation": context["tcp_pose"][3:],
        "grip": np.array([0.5]),
    }

    assert not Gello(port="/dev/unused").action(idle, context).driving
    assert (
        not Gello(port="/dev/unused", gripper=False)
        .action({**idle, "grip": np.array([0.0])}, context)
        .driving
    )
    moved = {**idle, "position": idle["position"] + np.array([0.01, 0.0, 0.0])}
    assert Gello(port="/dev/unused").action(moved, context).driving


def test_leader_joint_uses_the_legacy_motion_and_gripper_thresholds():
    from rlinf.robotics.parts.teleop import GelloJoint

    binding = GelloJoint(port="/dev/unused", side=0)
    current = np.zeros((2, 7))
    idle = {"joint_position": np.zeros(7), "grip": np.array([0.5])}

    assert not binding.action(idle, {"joint_positions": current}).driving
    moved = {**idle, "joint_position": np.full(7, 0.01)}
    assert binding.action(moved, {"joint_positions": current}).driving
    gripped = {**idle, "grip": np.array([0.0])}
    assert binding.action(gripped, {"joint_positions": current}).driving


def test_the_glove_holds_what_the_operator_posed():
    import numpy as np

    from rlinf.robotics.parts.teleop import Glove

    glove = Glove()
    glove.on_reset({"hand_reset_pose": np.zeros(6)})

    posed = glove.action({"angles": np.full(6, 0.4)}, {"hand_driving": True}).parts[
        "hand"
    ]
    released = glove.action({"angles": np.full(6, 0.9)}, {"hand_driving": False}).parts[
        "hand"
    ]

    assert np.allclose(posed, released)


def test_the_glove_tracks_only_while_its_gate_is_held():
    import numpy as np

    from rlinf.robotics.parts.teleop import Glove, SpaceMouse

    mouse, glove = SpaceMouse(), Glove()
    glove.on_reset({"hand_reset_pose": np.zeros(6)})

    released = mouse.publish({"twist": np.zeros(6), "buttons": [False, False]})
    held = mouse.publish({"twist": np.zeros(6), "buttons": [False, True]})

    assert released == {"hand_driving": False}
    assert held == {"hand_driving": True}

    # The first active reading sets the reference; the next applies motion.
    glove.action({"angles": np.zeros(6)}, held)
    moved = glove.action({"angles": np.full(6, 0.3)}, held).parts["hand"]
    assert np.allclose(moved, 0.3)

    # Releasing control preserves the last commanded hand pose.
    assert np.allclose(
        glove.action({"angles": np.full(6, 0.9)}, released).parts["hand"], 0.3
    )


# PICO device and binding


class _ScriptedController:
    """Replay a fixed sequence of controller readings."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._index = 0
        self.is_connected = True

    def get_observation(self):
        import numpy as np

        moved, turned, held, grip = self._steps[min(self._index, len(self._steps) - 1)]
        self._index += 1
        return {
            "held": held,
            "ready": True,
            "hand": "right",
            "calibrated": True,
            "control_value": 1.0 if held else 0.0,
            "position_delta": np.asarray(moved, dtype=np.float64),
            "rotation_delta": np.asarray(turned, dtype=np.float64),
            "grip_close": grip < 0,
            "grip_open": grip > 0,
        }

    def connect(self):
        pass

    def disconnect(self):
        pass


def _pico_context():
    import numpy as np

    return {
        "tcp_pose": np.array([0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "action_scale": np.array([0.05, 0.3, 1.0]),
    }


def test_a_controller_drives_the_arm_to_a_pose_it_can_reach():
    import numpy as np

    from rlinf.robotics.parts.teleop import PicoTcp

    binding = PicoTcp(gripper=True, side=0)
    device = _ScriptedController([((0.025, 0.0, 0.0), (0.0, 0.0, 0.0), True, -1)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    # Operator motion is relative to the pose captured on takeover.
    assert parts["arm"].size == 9
    assert np.isclose(parts["arm"][0], 0.3 + 0.025)
    assert np.isclose(parts["end_effector"][0], -1.0)
    assert binding.action(device.get_observation(), _pico_context()).driving


def test_releasing_the_grip_mid_chunk_holds_the_arm_where_it_was_left():
    import numpy as np

    from rlinf.robotics.parts.teleop import PicoTcp

    binding = PicoTcp(gripper=True, side=0, hold_current_when_inactive=False)
    device = _ScriptedController(
        [
            ((0.025, 0.0, 0.0), (0.0, 0.0, 0.0), True, -1),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0),
        ]
    )
    context = _pico_context()

    driven = binding.action(device.get_observation(), context).parts["arm"].copy()
    held = binding.action(device.get_observation(), context).parts["arm"]
    assert np.allclose(held, driven)

    # Policy-only input releases the intervention.
    binding.on_action_chunk_begin()
    assert binding.action(device.get_observation(), context).parts == {}


def test_holding_the_current_pose_leaves_the_gripper_to_the_policy():
    from rlinf.robotics.parts.teleop import PicoTcp

    binding = PicoTcp(gripper=True, side=0, hold_current_when_inactive=True)
    device = _ScriptedController([((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    assert set(parts) == {"arm"}


def _pico_frame(position, grip=0.0, buttons=None, headset=None):
    """One publisher frame, in the schema the shipped PICO configs describe."""
    import json

    return json.dumps(
        {
            "headset_pose": headset or [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            "right_controller": {
                "position": list(position),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "grip": grip,
                "trigger": 0.0,
            },
            "buttons": buttons or {},
        }
    )


def _stream_until(zmq, device, packet, wanted, tries=80):
    """Publish one frame repeatedly until the reading reflects it.

    The transport drops readings older than ``max_stale_s``, so a single frame
    is not enough; a real publisher streams continuously.
    """
    import time

    for _ in range(tries):
        zmq.packets.append(packet)
        time.sleep(0.004)
        reading = device.get_observation()
        if wanted(reading):
            return reading
    return device.get_observation()


def test_the_pico_transport_links_calibrates_and_tracks_motion():
    """Drive the ZMQ transport for real, rather than only its mapping.

    The device's action mapping was covered, but nothing had ever opened the
    socket, parsed a publisher frame, or exercised the calibration gate that
    stands between a linked headset and a moving robot.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop.pico import PicoTcp
        from rlinf.robotics.parts.transports import pico as transport

        # The transport binds ``zmq`` as a module global at import time, so it
        # keeps whichever fake was installed when it was first imported. Read
        # it back from there rather than from ``made``, or this test only
        # passes when it runs first.
        zmq = transport.zmq
        pico = PicoTcp(gripper=True, side=0, hand="right", zmq_addr="tcp://x:1")
        pico.connect()
        try:
            socket = zmq.sockets[-1]
            assert socket.address == "tcp://x:1"
            assert socket.hwm == 10, "a stale-frame backlog would lag the operator"

            # A headset that is linked but reporting garbage must not
            # calibrate, and must not let a squeezed grip drive the robot.
            broken = _pico_frame([0, 0, 0], grip=1.0, headset=[0.0] * 7)
            reading = _stream_until(zmq, pico, broken, lambda r: r["ready"])
            assert reading["ready"], "the link is up"
            assert not reading["calibrated"], "a zero quaternion is not a pose"
            assert not reading["held"], "uncalibrated must not drive"

            # A valid headset pose calibrates on its own at start-up.
            reading = _stream_until(
                zmq, pico, _pico_frame([0, 0, 0]), lambda r: r["calibrated"]
            )
            assert reading["calibrated"]

            # Grip is what hands control over.
            held = _pico_frame([0, 0, 0], grip=1.0)
            reading = _stream_until(zmq, pico, held, lambda r: r["held"])
            assert reading["held"]
            assert reading["control_value"] == pytest.approx(1.0)

            # 10 cm along the controller's -Z is 10 cm along the robot's +X.
            moved = _pico_frame([0, 0, -0.10], grip=1.0)
            reading = _stream_until(
                zmq,
                pico,
                moved,
                lambda r: float(np.linalg.norm(r["position_delta"])) > 1e-6,
            )
            assert reading["position_delta"] == pytest.approx([0.1, 0.0, 0.0], abs=1e-6)

            released = _pico_frame([0, 0, -0.10], grip=0.0)
            reading = _stream_until(zmq, pico, released, lambda r: not r["held"])
            assert not reading["held"]
        finally:
            pico.disconnect()

        assert socket.closed, "the subscriber socket outlived the device"


def test_a_stale_pico_link_stops_driving_the_robot():
    """No frames means no motion, not the last motion repeated forever."""
    import time

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.teleop.pico import PicoTcp
        from rlinf.robotics.parts.transports import pico as transport

        zmq = transport.zmq
        pico = PicoTcp(gripper=True, side=0, hand="right", zmq_addr="tcp://x:1")
        pico.connect()
        try:
            held = _pico_frame([0, 0, 0], grip=1.0)
            assert _stream_until(zmq, pico, held, lambda r: r["held"])["held"]

            # Publisher goes quiet for longer than the transport tolerates.
            time.sleep(pico._device.max_stale_s + 0.05)
            reading = pico.get_observation()

            assert reading["stale"] is True
            assert not reading["held"]
        finally:
            pico.disconnect()


def test_pico_builds_the_variant_the_arm_can_accept():
    """One config name, two meanings; the arm's declared kind picks which.

    Sending an absolute pose to an arm that expects a delta is the difference
    between a few millimetres of motion and the arm jumping across its
    workspace, so this is the decision worth pinning.
    """
    from types import SimpleNamespace

    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import Pico, PicoDelta, PicoTcp, TeleopDevice

    def built_for(kind):
        facts = SimpleNamespace(kinds={"arm": kind})
        return TeleopDevice.named("pico").from_config({}, {}, facts).device

    delta = built_for(ActionKind.CARTESIAN_DELTA)
    pose = built_for(ActionKind.CARTESIAN_POSE)

    assert type(delta) is PicoDelta
    assert type(pose) is PicoTcp
    # The config name resolves to the family, so either one is a Pico.
    assert isinstance(delta, Pico) and isinstance(pose, Pico)


def test_the_pico_name_resolves_to_the_family_not_a_variant():
    """Whatever from_config returns must be an instance of what was named."""
    import inspect

    from rlinf.robotics.parts.teleop import Pico, PicoDelta, PicoTcp, TeleopDevice

    assert TeleopDevice.named("pico") is Pico
    # Abstract, so the family cannot be mistaken for one of its variants.
    assert inspect.isabstract(Pico)
    # Siblings: neither variant is a kind of the other.
    assert not issubclass(PicoTcp, PicoDelta)
    assert not issubclass(PicoDelta, PicoTcp)


def test_no_device_shadows_a_method_of_the_contract():
    """A constructor argument must not overwrite a method the group calls.

    The glove took a ``hold`` argument and stored it as ``self.hold``, which
    replaced the ``hold()`` the group calls on every device. Any rig with a
    glove in it then raised ``TypeError: 'bool' object is not callable`` from
    ``TeleopGroup.hold``, and the caller in ``RealEnv.get_hold_actions`` only
    catches ``AttributeError``, so it surfaced as a crash mid-episode.
    """
    from rlinf.robotics.parts.teleop import TeleopDevice

    contract = (
        "hold",
        "action",
        "drive",
        "publish",
        "get_observation",
        "on_reset",
        "on_action_chunk_begin",
    )
    for name in TeleopDevice.names():
        device_cls = TeleopDevice.named(name)
        for method in contract:
            attr = getattr(device_cls, method, None)
            assert attr is None or callable(attr), (
                f"{device_cls.__name__}.{method} is {type(attr).__name__}, not callable"
            )


def test_a_glove_rig_can_be_asked_what_to_hold():
    """The glove fills a part while idle but has no absolute pose to hold."""
    from types import SimpleNamespace

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.actions import ActionKind
        from rlinf.robotics.parts.teleop import TeleopDevice, TeleopGroup

        facts = SimpleNamespace(
            kinds={"hand": ActionKind.HAND},
            joint_action_scale=0.1,
            direct_stream=False,
            layout={},
        )
        entry = TeleopDevice.named("glove").from_config(
            {"glove_config": {"left_port": "/dev/mock-glove"}}, {}, facts
        )
        group = TeleopGroup([entry], available={"hand": ActionKind.HAND})
        group.connect()
        try:
            # Empty, not an exception: the hand holds itself through
            # APPLIES_WHILE_IDLE rather than through an absolute pose.
            assert group.hold({"hand_reset_pose": np.zeros(6)}) == {}
        finally:
            group.disconnect()


def _dosw1_arm_with_recorded_commands():
    """Return a left DOSW1 arm and the go_joint calls it issues."""
    from rlinf.robotics.parts.arms.dosw1 import DOSW1Connection

    connection = DOSW1Connection(robot_url="localhost", is_dummy=True)
    connection.connect()
    issued: list[tuple[tuple[float, ...], float]] = []
    original = connection.left_go_joint

    def record(joints, gripper, **kwargs):
        issued.append((tuple(round(v, 4) for v in joints), round(gripper, 4)))
        return original(joints, gripper, **kwargs)

    connection.left_go_joint = record
    return connection.parts["left"], issued


def test_one_dosw1_action_is_one_hardware_command():
    """Joints and gripper reach the arm together, as the SDK takes them.

    They used to be separate parts, each restating the other half from a
    read-back. Driving both issued two go_joint calls, and the second
    countermanded the first with joints read mid-motion.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        arm, issued = _dosw1_arm_with_recorded_commands()

        arm.send_action(
            {
                "joint_position": np.full(6, 0.25),
                "gripper_width": np.array([0.03]),
            }
        )

        assert issued == [((0.25,) * 6, 0.03)], issued


def test_a_dosw1_action_without_a_gripper_holds_the_current_width():
    """Omitting the gripper keeps it where it is, as it did before."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        arm, issued = _dosw1_arm_with_recorded_commands()
        held = float(arm.sdk.get_left_joint()[6])

        arm.send_action({"joint_position": np.full(6, 0.4)})

        assert issued == [((0.4,) * 6, round(held, 4))], issued


def test_a_dosw1_arm_still_refuses_an_action_it_does_not_have():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        arm, _ = _dosw1_arm_with_recorded_commands()

        with pytest.raises(KeyError):
            arm.send_action({"tcp_pose": np.zeros(6)})


def test_an_end_effector_answers_for_its_own_kind():
    """A part reports what it is, so an env need not trust its config.

    The kind used to live only on the EndEffectorType enum, which meant the
    Franka env branched on the config it built the part from and then called
    BaseGripper-only members through a BaseEndEffector.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector
        from rlinf.robotics.parts.end_effectors.base import BaseEndEffector

        gripper = EndEffector.of("robotiq_gripper", port="/dev/mock-gripper")

        assert gripper.is_gripper
        assert not gripper.is_hand
        # The verbs an env drives it with are on the declared type, not on a
        # subclass it would have to downcast to.
        for name in ("open", "close", "is_open", "is_gripper", "is_hand"):
            assert hasattr(BaseEndEffector, name), name


def test_a_grasp_is_not_a_move_to_zero_width():
    """close() holds an object; command() only travels to a width.

    Routing binary close through send_action would call move() and quietly
    drop the grasp force, so the two stay separate verbs.
    """
    import inspect

    from rlinf.robotics.parts.end_effectors.grippers.franka import FrankaGripper

    close = inspect.getsource(FrankaGripper.close)
    move = inspect.getsource(FrankaGripper.move)

    assert "force" in close, "closing must request a grasp force"
    assert "Grasp" in close, "closing goes through the grasp action"
    assert "force" not in move, "moving positions without force"


def test_a_delta_binding_has_no_pose_to_hold():
    from rlinf.robotics.parts.teleop import PicoDelta, PicoTcp

    assert PicoDelta(gripper=True).hold(_pico_context()) == {}
    assert "arm" in PicoTcp(gripper=True).hold(_pico_context())


def test_absolute_commands_are_clipped_but_deltas_are_not():
    from rlinf.robotics.parts.teleop import PicoDelta, PicoTcp, SpaceMouse

    assert PicoTcp.CLIPS_TO_ACTION_SPACE
    assert not PicoDelta.CLIPS_TO_ACTION_SPACE
    assert not SpaceMouse.CLIPS_TO_ACTION_SPACE


def test_each_side_reports_its_own_state():
    from rlinf.robotics.actions import ActionKind
    from rlinf.robotics.parts.teleop import PicoTcp, TeleopEntry, TeleopGroup

    layout = {
        "left.arm": ActionKind.CARTESIAN_POSE,
        "left.end_effector": ActionKind.GRIPPER,
        "right.arm": ActionKind.CARTESIAN_POSE,
        "right.end_effector": ActionKind.GRIPPER,
    }

    def scripted(side, index):
        """A PICO whose reading is replayed rather than read from hardware."""
        replay = _ScriptedController(
            [((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), side == "left", 1)]
        )

        class Scripted(PicoTcp):
            def get_observation(self):
                return replay.get_observation()

        device = Scripted(gripper=True, side=index)
        device._device = replay
        return device

    entries = [
        TeleopEntry(scripted(side, index), drives=side)
        for index, side in enumerate(("left", "right"))
    ]
    group = TeleopGroup(entries, available=layout)

    _, driving, info = group.action(_pico_context())

    assert driving
    assert info["left_pico_active"] is True
    assert info["right_pico_active"] is False


def test_reading_a_controller_does_not_need_the_robot():
    import inspect

    from rlinf.robotics.parts.teleop import PicoDelta
    from rlinf.robotics.parts.transports import pico

    source = inspect.getsource(pico.PicoExpert)
    for name in ("tcp_pose", "_ref_tcp"):
        assert name not in source, f"the reader still refers to {name!r}"

    observation = PicoDelta.get_observation
    assert "get_reading" in inspect.getsource(observation)


def test_a_controller_reading_is_data_not_a_handle():
    from rlinf.robotics.parts.teleop import PicoDelta

    features = PicoDelta(hand="right").observation_features

    assert set(features) == {
        "held",
        "position_delta",
        "rotation_delta",
        "grip_close",
        "grip_open",
    }


def test_the_arm_anchors_where_it_was_when_the_operator_took_hold():
    import numpy as np

    from rlinf.robotics.parts.teleop import PicoTcp

    binding = PicoTcp(gripper=True, side=0)
    device = _ScriptedController(
        [
            ((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), True, 0),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0),
            ((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), True, 0),
        ]
    )
    context = _pico_context()

    first = binding.action(device.get_observation(), context).parts["arm"][0]
    binding.action(device.get_observation(), context)
    again = binding.action(device.get_observation(), context).parts["arm"][0]

    # Repeated motion from the same measured pose produces the same command.
    assert np.isclose(first, again)


# Controller packet reader


def _pico_packet(x, y, z, yaw, grip, close=False, open_=False):
    """Build one controller frame in the headset wire format."""
    from scipy.spatial.transform import Rotation as R

    return {
        "right_controller": {
            "position": [x, y, z],
            "orientation": list(R.from_euler("xyz", [0.0, 0.0, yaw]).as_quat()),
            "grip": grip,
            "trigger": 0.0,
        },
        "buttons": {"A": close, "B": open_},
    }


def _pico_reader(**overrides):
    """Build a PICO reader backed by scripted packets."""
    from rlinf.robotics.parts.transports.pico import PicoExpert

    config = {
        "hand": "right",
        "control_trigger": "grip",
        "control_threshold": 0.85,
        "gripper_close_button": "A",
        "gripper_open_button": "B",
        "position_scale": 1.0,
        "rotation_scale": 0.5,
        "calibration": {"enabled": False, "required": False},
    }
    config.update(overrides)
    PicoExpert.start = lambda self: None
    return PicoExpert(**config)


def test_a_reading_is_empty_until_the_operator_takes_hold():
    import numpy as np

    reader = _pico_reader()
    reader._snapshot = lambda: _pico_packet(0.1, 0.2, 0.3, 0.4, grip=0.0)

    reading = reader.get_reading()

    assert reading["held"] is False
    assert np.allclose(reading["position_delta"], 0.0)


def test_motion_is_measured_from_where_the_operator_took_hold():
    import numpy as np

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    reader._snapshot = lambda: _pico_packet(0.5, 0.5, 0.5, 0.0, grip=0.95)
    reader.get_reading()  # Establish the control anchor at this pose.

    reader._snapshot = lambda: _pico_packet(0.5, 0.5, 0.47, 0.0, grip=0.95)
    reading = reader.get_reading()

    assert reading["held"] is True
    assert np.allclose(reading["position_delta"], [0.03, 0.0, 0.0], atol=1e-9)


def test_letting_go_and_grabbing_again_re_anchors():
    import numpy as np

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    for position, grip in (
        ((0.0, 0.0, 0.0), 0.95),
        ((0.30, 0.0, 0.0), 0.95),
        ((0.30, 0.0, 0.0), 0.0),
        ((0.30, 0.0, 0.0), 0.95),
    ):
        reader._snapshot = lambda p=position, g=grip: _pico_packet(*p, 0.0, grip=g)
        reading = reader.get_reading()

    # Retaking control resets the anchor without applying motion.
    assert reading["held"] is True
    assert np.allclose(reading["position_delta"], 0.0, atol=1e-9)


def test_the_gripper_buttons_are_reported_separately():
    reader = _pico_reader()
    reader._snapshot = lambda: _pico_packet(0, 0, 0, 0, grip=0.95, close=True)

    reading = reader.get_reading()

    assert reading["grip_close"] is True
    assert reading["grip_open"] is False


def test_a_dropped_link_reports_stale_rather_than_stale_motion():
    reader = _pico_reader()
    reader._snapshot = lambda: None

    reading = reader.get_reading()

    assert reading["held"] is False
    assert reading["ready"] is False
    assert reading["stale"] is True


def test_the_binding_turns_a_reading_into_a_command_for_this_arm():
    import numpy as np

    from rlinf.robotics.parts.teleop import PicoTcp

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    binding = PicoTcp(gripper=True, side=0)
    context = _pico_context()

    reader._snapshot = lambda: _pico_packet(0.0, 0.0, 0.0, 0.0, grip=0.95)
    binding.action(reader.get_reading(), context)

    reader._snapshot = lambda: _pico_packet(0.0, 0.0, -0.04, 0.0, grip=0.95)
    reading = reader.get_reading()
    parts = binding.action(reading, context).parts

    # The target combines the takeover pose with operator motion.
    expected = np.asarray(context["tcp_pose"][:3]) + reading["position_delta"]
    assert np.allclose(parts["arm"][:3], expected, atol=1e-6)
    assert np.isclose(reading["position_delta"][0], 0.04, atol=1e-9)


# Shared lifecycle across part categories


def test_every_device_family_is_shaped_the_same_way():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import rlinf.robotics.parts.cameras  # noqa: F401  - registers drivers
        import rlinf.robotics.parts.end_effectors  # noqa: F401
        import rlinf.robotics.parts.teleop  # noqa: F401
        from rlinf.robotics.parts.arms.base import Arm, BaseArm
        from rlinf.robotics.parts.cameras.base import BaseCamera, Camera
        from rlinf.robotics.parts.end_effectors.base import BaseEndEffector, EndEffector
        from rlinf.robotics.parts.teleop import TeleopPart

        # Configured backend names resolve through their category registry.
        for category in (Arm, Camera, EndEffector):
            assert category.backends(), (
                f"{category.__name__} has no registered backends; a config "
                "naming one would have nothing to resolve against"
            )

        # Driver classes implement both lifecycle hooks unless the category
        # provides one shared release implementation.
        for base in (BaseArm, BaseCamera, BaseEndEffector, TeleopPart):
            assert base._open.__isabstractmethod__, (
                f"{base.__name__} lets a driver inherit _open, so one that "
                "never wrote it fails at the first connect instead of at "
                "class definition"
            )
        for base in (BaseArm, BaseCamera, BaseEndEffector):
            assert base._release.__isabstractmethod__, base.__name__
        assert not getattr(TeleopPart._release, "__isabstractmethod__", False), (
            "TeleopPart releases every reader the same way, so it does it once"
        )


def test_every_part_presents_a_device_category():
    import inspect

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import rlinf.robotics.parts.teleop  # noqa: F401
        import rlinf.robotics.parts.views  # noqa: F401
        import rlinf.robotics.robots  # noqa: F401
        from rlinf.robotics.parts.arms.base import Arm
        from rlinf.robotics.parts.base import PartGroup, RobotPart
        from rlinf.robotics.parts.cameras.base import Camera
        from rlinf.robotics.parts.end_effectors.base import EndEffector
        from rlinf.robotics.parts.mobility.base import MobileBase
        from rlinf.robotics.parts.teleop import TeleopPart

        Arm.backends()  # Load every arm backend module.

        def descendants(cls):
            for child in cls.__subclasses__():
                yield child
                yield from descendants(child)

        categories = (Arm, Camera, EndEffector, MobileBase, TeleopPart)
        homeless = sorted(
            cls.__name__
            for cls in descendants(RobotPart)
            if cls.__module__.startswith("rlinf.")  # Exclude local fakes.
            and not inspect.isabstract(cls)
            and not issubclass(cls, PartGroup)
            and not any(issubclass(cls, category) for category in categories)
        )

    assert homeless == [], f"parts belonging to no device category: {homeless}"


def test_every_part_family_opens_and_closes_the_same_way():
    import inspect

    from rlinf.robotics.parts.cameras.base import BaseCamera
    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector
    from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper
    from rlinf.robotics.parts.teleop import TeleopPart

    for family in (TeleopPart, BaseCamera, BaseEndEffector, BaseGripper):
        assert hasattr(family, "_open"), f"{family.__name__} has no _open"
        assert hasattr(family, "_release"), f"{family.__name__} has no _release"

    # Retired lifecycle hook names are no longer accepted.
    for family in (BaseCamera, BaseEndEffector):
        source = inspect.getsource(family)
        for retired in ("_close_device", "def initialize", "def shutdown"):
            assert retired not in source, f"{family.__name__} still has {retired}"

    # Placement-aware connect/disconnect remain owned by Connection.
    for family in (TeleopPart, BaseCamera, BaseEndEffector, BaseGripper):
        for public in ("connect", "disconnect"):
            assert public not in vars(family), (
                f"{family.__name__} overrides {public}; a part placed on another "
                "node would then never be rebuilt there"
            )

    # Categories extend drivers through local lifecycle hooks.
    assert "_opened" in vars(BaseCamera), (
        "BaseCamera starts its capture loop, and that has to run beside the "
        "camera rather than beside whoever is holding it"
    )


def test_a_gripper_is_an_end_effector_rather_than_a_second_kind_of_one():
    import inspect

    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector, EndEffector
    from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper
    from rlinf.robotics.parts.views import MethodEndEffector

    assert issubclass(BaseGripper, BaseEndEffector)

    # Grippers and other end effectors share one lifecycle declaration.
    for hook in ("_open", "_release"):
        assert BaseGripper.__dict__.get(hook) is None, (
            f"BaseGripper re-declares {hook}; it should inherit the one contract"
        )
        assert getattr(BaseEndEffector, hook).__isabstractmethod__, (
            f"{hook} must be required, or a driver that never wrote one fails "
            "at the first connect instead of at class definition"
        )

    # ``reset`` has one signature across the category.
    assert list(inspect.signature(BaseGripper.reset).parameters) == list(
        inspect.signature(BaseEndEffector.reset).parameters
    )

    # A hosted view borrows its connection and therefore needs no ``_open``.
    assert issubclass(MethodEndEffector, EndEffector)
    assert not issubclass(MethodEndEffector, BaseEndEffector)


def test_a_gripper_is_commanded_in_the_units_it_reports():
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.end_effectors import EndEffector

    with mocked_sdks():
        gripper = EndEffector.of("robotiq_gripper", port="/dev/mock-gripper")
        gripper.connect()
        try:
            assert gripper.max_width > 0

            gripper.open()
            assert gripper.position == pytest.approx(gripper.max_width, abs=1e-3)
            gripper.close()
            assert gripper.position == pytest.approx(0.0, abs=1e-3)

            # Round-trip width may differ by one quantized register count.
            quantum = gripper.max_width / 255
            for width in (0.0, gripper.max_width / 2, gripper.max_width):
                gripper.move(width)
                assert gripper.position == pytest.approx(width, abs=quantum)

            # The canonical part API uses the same physical unit.
            gripper.move(gripper.max_width / 3)
            state = gripper.get_state()
            gripper.open()
            gripper.command(state)
            assert gripper.position == pytest.approx(state[0], abs=quantum)

            # Values beyond the stroke clamp instead of wrapping.
            gripper.move(gripper.max_width * 10)
            assert gripper.position == pytest.approx(gripper.max_width, abs=quantum)
        finally:
            gripper.disconnect()


def test_a_franka_hand_is_commanded_in_metres_on_the_wire():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franka_gripper")
        gripper.connect()
        try:
            widths: list[float] = []
            put = gripper._ros.put_channel

            def record(channel, message):
                widths.append(float(message.goal.width))
                return put(channel, message)

            gripper._ros.put_channel = record

            gripper.open()
            gripper.move(0.05)
            gripper.move(gripper.max_width * 10)
        finally:
            gripper.disconnect()

    assert widths == pytest.approx([gripper.max_width, 0.05, gripper.max_width])


def test_every_end_effector_answers_the_same_questions():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors.base import EndEffector
        from rlinf.robotics.robots import DOSW1Robot, FrankaRobot

        def franka(end_effector_type=None):
            robot = FrankaRobot.build(
                robot_ip="10.0.0.2",
                node_rank=0,
                env_idx=0,
                worker_rank=0,
                end_effector_type=end_effector_type,
            )
            return robot.child("end_effector")

        every = {
            "hand on a bus": franka("ruiyan_hand"),
            "gripper on a bus": franka(),
            "gripper on a session": DOSW1Robot.build(is_dummy=True)
            .child("left")
            .child("gripper"),
            "gripper on its own port": EndEffector.of(
                "robotiq_gripper", port="/dev/mock-gripper"
            ),
        }

        for label, part in every.items():
            assert isinstance(part, EndEffector), label
            assert part.state_dim >= 1 and part.action_dim >= 1, label
            assert part.control_mode in {"binary", "continuous"}, label
            assert callable(part.get_state) and callable(part.command), label
            assert set(part.observation_features) == {"state"}, label
            assert set(part.action_features) == {"target"}, label

        # A six-finger hand composes exactly where a gripper would.
        assert every["hand on a bus"].action_dim == 6


def test_every_end_effector_reports_its_state_under_the_same_name():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors.base import EndEffectorType

        gripper = EndEffector.of("robotiq_gripper", port="/dev/mock-gripper")
        hand = EndEffector.of(EndEffectorType.RUIYAN_HAND, port="/dev/mock-hand")

        for part in (gripper, hand):
            name = type(part).__name__
            assert set(part.observation_features) == {"state"}, (
                f"{name} reports {sorted(part.observation_features)}, not 'state'"
            )
            assert set(part.action_features) == {"target"}, name
            assert part.state_dim >= 1 and part.action_dim >= 1, name
            assert part.control_mode in {"binary", "continuous"}, name

            part.connect()
            try:
                observation = part.get_observation()
                assert set(observation) == {"state"}, name
                assert observation["state"].shape == (part.state_dim,), name
                target = np.zeros(part.action_dim, dtype=np.float32)
                assert set(part.send_action({"target": target})) == {"target"}, name
                part.reset()
                part.reset(target)
            finally:
                part.disconnect()
            assert not part.is_connected, name


class _Part(RobotPart):
    """Record lifecycle events in its device handle."""

    def __init__(self):
        self.log = []

    def _open(self):
        self.log.append("open")
        return "device"

    def _release(self, device):
        self.log.append("close")

    @property
    def observation_features(self):
        return {}

    def get_observation(self):
        return {}


def test_connecting_twice_opens_once():
    part = _Part()
    part.connect()
    part.connect()

    assert part.log == ["open"]
    assert part.is_connected


def test_disconnecting_what_was_never_opened_does_nothing():
    part = _Part()
    part.disconnect()

    assert part.log == []
    assert not part.is_connected


def test_a_part_is_released_with_the_device_it_opened():
    part = _Part()
    part.connect()

    assert part.is_connected
    part.disconnect()

    assert part.log == ["open", "close"]
    assert not part.is_connected
    # Repeated disconnect does not release a device twice.
    part.disconnect()
    assert part.log == ["open", "close"]


def test_a_part_that_says_nothing_about_its_hardware_fails_clearly():
    class Bare(RobotPart):
        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    with pytest.raises(NotImplementedError, match="does not say how to open"):
        Bare().connect()


# Minimal robot composition


def test_a_robot_is_named_parts_and_nothing_else():
    from rlinf.robotics import Robot

    class Gripper(_Part):
        @property
        def observation_features(self):
            return {"width": {"shape": (1,), "dtype": "float32"}}

        def get_observation(self):
            return {"width": 0.5}

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    robot = Bench(arm=_Part(), hand=Gripper())
    robot.connect()

    assert robot.is_connected
    assert set(robot.get_observation()) == {"arm", "hand"}
    assert robot.get_observation()["hand"] == {"width": 0.5}

    robot.disconnect()
    assert not robot.is_connected


def test_build_is_only_for_robots_reached_by_name():
    from rlinf.robotics import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    with pytest.raises(NotImplementedError, match="Construct Bench"):
        Bench.build()


# Connection and part separation


def test_a_connection_backing_several_parts_is_not_observable():
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, ControllablePart

    assert issubclass(Turtle2Connection, Connection)
    assert not issubclass(Turtle2Connection, ControllablePart)

    hardware = Turtle2Connection()
    assert set(hardware.parts) == {
        "left",
        "left_end_effector",
        "right",
        "right_end_effector",
    }
    # Connections omit the observation interface and cannot enter the part tree.
    assert not isinstance(hardware, RobotPart)
    assert not hasattr(hardware, "get_observation")
    assert not hasattr(hardware, "observation_features")


def test_every_robot_composes_from_named_parts():
    from rlinf.robotics.parts.base import Connection, RobotPart
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot
    from rlinf.robotics.robots.franka import FrankaRobot
    from rlinf.robotics.robots.gim_arm import GimArmRobot

    built = [
        FrankaRobot.build_arms(robot_ip="1.2.3.4", node_rank=0),
        GimArmRobot.build_arms(
            node_rank=0,
            can_interface="can0",
            arm_variant="xl",
            enable_gripper=True,
            gripper_type="default",
            control_mode="joint",
        ),
        DualFrankaRobot.build_arms(
            left_robot_ip="1.2.3.4",
            right_robot_ip="1.2.3.5",
            left_gripper_connection="/dev/ttyUSB0",
            right_gripper_connection="/dev/ttyUSB1",
        ),
    ]
    for arms in built:
        for name, value in arms.items():
            values = (
                list(value.children.values()) if hasattr(value, "children") else [value]
            )
            # Each leaf must be a readable capability, not its shared session.
            assert all(isinstance(v, RobotPart) for v in values), (
                f"{name} holds {[type(v).__name__ for v in values]} rather than "
                "parts picked out of its connection"
            )
            assert not any(
                isinstance(v, Connection) and not isinstance(v, RobotPart)
                for v in values
            )


def test_one_connection_backs_every_part_that_names_it():
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot
    from rlinf.robotics.robots.franka import FrankaRobot

    single = FrankaRobot.build_arms(
        robot_ip="1.2.3.4", node_rank=0, gripper_connection="/dev/ttyUSB0"
    )
    # The arm and the hand each answer on their own endpoint, so each owns its
    # own connection rather than borrowing one.
    assert len({id(part.owner) for part in single.values()}) == len(single)

    dual = DualFrankaRobot.build_arms(
        left_robot_ip="1.2.3.4",
        right_robot_ip="1.2.3.5",
        left_gripper_connection="/dev/ttyUSB0",
        right_gripper_connection="/dev/ttyUSB1",
    )
    sides = {
        side: {id(part.owner) for part in group.children.values()}
        for side, group in dual.items()
    }
    # No connection is shared between the two sides.
    assert not (sides["left"] & sides["right"])


def test_a_connection_with_several_parts_hands_each_of_them_out_by_name():
    class Pair(RobotPart):
        def _open(self):
            return "link"

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

        @property
        def parts(self):
            return {"a": _Part(), "b": _Part()}

    pair = Pair()

    assert set(pair.parts) == {"a", "b"}
    assert isinstance(pair.part("a"), RobotPart)
    with pytest.raises(KeyError):
        pair.part("c")


# Bench-check integration


def _fake_robot_type(broken=None):
    """Register a robot made of fakes, and yield its type name."""
    import numpy as np

    from rlinf.robotics.discovery import RobotConfig, RobotDiscovery, register_robot
    from rlinf.robotics.parts.base import Connection, ControllablePart, PartGroup
    from rlinf.robotics.parts.views import MethodArm, MethodEndEffector
    from rlinf.robotics.robot import Robot

    class State:
        def to_dict(self):
            return {"tcp_pose": np.zeros(7), "gripper_position": np.zeros(1)}

    class Bus(Connection):
        def _open(self):
            return "bus"

        def get_state(self):
            return State()

        def move(self, target):
            return target

        @property
        def parts(self):
            return {
                "left": MethodArm(
                    self, commands={"tcp_pose": "move"}, state_fields=("tcp_pose",)
                ),
                "left_end_effector": MethodEndEffector(
                    self, state_field="gripper_position"
                ),
            }

    class Liar(ControllablePart):
        def _open(self):
            return "liar"

        @property
        def observation_features(self):
            return {"tcp_pose": {}}

        @property
        def action_features(self):
            return {}

        def get_observation(self):
            return {"joint_position": np.zeros(7)}

        def send_action(self, action):
            return action

    class Bench(Robot):
        ROBOT_TYPE = f"BenchFake{broken or ''}"

        @classmethod
        def build(cls, **_):
            if broken == "Mismatch":
                return cls(arm=Liar())
            if broken == "Connection":
                return cls(bus=Bus())
            bus = Bus()
            return cls(
                left=PartGroup(
                    arm=bus.part("left"),
                    gripper=bus.part("left_end_effector"),
                )
            )

    class Config(RobotConfig):
        pass

    class Discovery(RobotDiscovery):
        HW_TYPE = Bench.ROBOT_TYPE

        @classmethod
        def enumerate(cls, node_rank, configs=None):
            return None

    register_robot(Config, Bench, build=Bench.build)(Discovery)
    return Bench.ROBOT_TYPE


def _run_bench(robot_type):
    from toolkits.realworld_check.check_robot_parts import check

    return check(robot_type, {})


def test_the_bench_check_passes_a_healthy_robot():
    assert _run_bench(_fake_robot_type()) == 0


def test_the_bench_check_catches_an_observation_that_was_never_declared():
    assert _run_bench(_fake_robot_type("Mismatch")) == 1


def test_a_connection_left_in_the_tree_is_refused_at_composition():
    with pytest.raises(TypeError, match="backs parts without being one"):
        _run_bench(_fake_robot_type("Connection"))


# Production parts with fake SDKs


def test_no_part_hook_collides_with_the_worker_group():
    import inspect

    from rlinf.robotics.parts.base import RobotPart
    from rlinf.scheduler.worker.worker_group import WorkerGroup

    taken = {
        name
        for name in dir(WorkerGroup)
        if name.startswith("_") and not name.startswith("__")
    }
    for family in (RobotPart, *RobotPart.__subclasses__()):
        ours = {
            name
            for name, value in inspect.getmembers(family, callable)
            if name.startswith("_") and not name.startswith("__")
        }
        assert not (ours & taken), (
            f"{family.__name__} defines {sorted(ours & taken)}, which the "
            "scheduler already attaches to a WorkerGroup"
        )


def test_a_real_camera_runs_against_a_faked_sdk():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.cameras.base import CameraInfo
        from rlinf.robotics.parts.cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(
            CameraInfo(
                name="wrist",
                serial_number="MOCK0001",
                camera_type="realsense",
                fps=30,
                resolution=(64, 48),
            )
        )
        assert not camera.is_connected
        camera.connect()
        assert camera.is_connected

        frame = camera.get_observation()["frame"]
        assert frame.shape == (48, 64, 3)

        camera.disconnect()
        assert not camera.is_connected


def test_a_real_arm_runs_against_a_faked_sdk():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.franky import FrankyArm

        arm = FrankyArm("10.0.0.1")
        arm.connect()

        assert arm.is_connected
        # The arm backs nothing else: its end effector is composed beside it.
        assert not arm.parts and not arm.children

        observation = arm.get_observation()
        assert observation["tcp_pose"].shape == (7,)

        arm.send_action({"joint_position": [0.0] * 7})

        arm.disconnect()
        assert not arm.is_connected


def test_a_franky_arm_takes_no_end_effector_settings():
    from rlinf.robotics.parts.arms.franky import FrankyArm

    arm = FrankyArm.declare("10.0.0.1")
    assert not arm.parts

    # Settings the arm cannot honour are refused rather than dropped: the hand
    # they describe is composed beside the arm, so they belong to it.
    for unsupported in ("gripper_type", "gripper_connection", "end_effector_type"):
        with pytest.raises(TypeError, match="does not take"):
            FrankyArm.declare("10.0.0.1", **{unsupported: "whatever"})


def test_the_bench_check_runs_a_whole_robot_on_fakes():
    from robot_mocks import mocked_sdks

    from toolkits.realworld_check.check_robot_parts import check

    with mocked_sdks():
        import rlinf.robotics.robots  # noqa: F401

        assert (
            check(
                "DualFranka",
                {
                    "left_robot_ip": "10.0.0.1",
                    "right_robot_ip": "10.0.0.2",
                    "left_gripper_connection": "/dev/ttyUSB0",
                    "right_gripper_connection": "/dev/ttyUSB1",
                },
            )
            == 0
        )


def test_every_shipped_robot_runs_on_faked_sdks():
    from robot_mocks import mocked_sdks

    from toolkits.realworld_check.check_robot_parts import check

    robots = {
        "Franka": {"robot_ip": "10.0.0.1", "node_rank": 0},
        "DualFranka": {
            "left_robot_ip": "10.0.0.1",
            "right_robot_ip": "10.0.0.2",
            "left_gripper_connection": "/dev/ttyUSB0",
            "right_gripper_connection": "/dev/ttyUSB1",
        },
        "GimArm": {
            "node_rank": 0,
            "can_interface": "can0",
            "arm_variant": "gim_arm_xl",
            "enable_gripper": True,
            "gripper_type": "parallel",
            "control_mode": "momentum_observer",
            "env_idx": 0,
            "worker_rank": 0,
        },
        "Turtle2": {
            "frequency": 50,
            "camera_ids": [1],
            "env_idx": 0,
            "node_rank": 0,
            "worker_rank": 0,
        },
        "DOSW1": {
            "node_rank": 0,
            "robot_url": "localhost",
            "left_arm_port": 1,
            "right_arm_port": 2,
            "left_lead_port": 3,
            "right_lead_port": 4,
            "enable_human_in_loop": False,
            "is_dummy": False,
            "gripper_width_max": 0.08,
        },
    }
    with mocked_sdks():
        import rlinf.robotics.robots  # noqa: F401

        failed = [name for name, args in robots.items() if check(name, args) != 0]

    assert failed == []


def test_a_worker_installs_the_fakes_for_itself():
    import subprocess
    import sys

    from robot_mocks import _reach_worker_processes

    environment = {**os.environ, **_reach_worker_processes()}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, psutil;"
            "print('pyrealsense2' in sys.modules, type(psutil).__name__)",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd="/tmp",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True _Psutil"


def test_a_wrapper_that_narrows_the_action_declares_it():
    from types import SimpleNamespace

    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.wrappers.transforms import GripperCloseEnv

    inner = SimpleNamespace(
        action_parts=lambda: FrankaEnv.action_parts(SimpleNamespace(_is_hand=False))
    )
    wrapper = GripperCloseEnv.__new__(GripperCloseEnv)
    wrapper.env = SimpleNamespace(get_wrapper_attr=lambda name: getattr(inner, name))

    assert [part.name for part in wrapper.action_parts()] == ["arm"]
    assert sum(part.width for part in wrapper.action_parts()) == 6


def test_a_dual_arm_robot_is_given_its_wrist_cameras():
    import inspect

    from rlinf.robotics.robots.dual_franka import DualFrankaRobot

    accepted = set()
    for method in (DualFrankaRobot.build, DualFrankaRobot.build_arms):
        accepted |= set(inspect.signature(method).parameters)

    assert "arm_cameras" in accepted, (
        "DualFrankaEnv passes arm_cameras to build(); nothing accepts it, so "
        "wrist cameras never reach the robot"
    )


def test_two_parts_of_one_class_on_one_node_get_different_names():
    from rlinf.robotics.parts.cameras.realsense import RealSenseCamera
    from rlinf.robotics.placement import PartWorkerHost

    names = {PartWorkerHost.default_name(RealSenseCamera, 0) for _ in range(8)}

    assert len(names) == 8, f"names repeat: {sorted(names)}"
    assert all(name.startswith("RealSenseCamera-node0-") for name in names), (
        f"a name should still say what it is and where: {sorted(names)}"
    )


def test_every_real_task_is_registered_through_the_shared_factory():
    import re

    offenders = []
    for path in (_ROOT / "rlinf" / "envs" / "real").rglob("__init__.py"):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*register\(", line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == [], f"these register a task outside register_tasks: {offenders}"


def test_an_address_is_checked_by_the_arm_that_dials_it():
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig
    from rlinf.robotics.robots.franka import FrankaConfig

    # Enumeration may still resolve the address from the node environment.
    assert (
        DualFrankaConfig(node_rank=0, left_robot_ip="LEFT_ROBOT_IP").left_robot_ip
        == "LEFT_ROBOT_IP"
    )
    assert FrankaConfig(node_rank=0, robot_ip="ROBOT_IP").robot_ip == "ROBOT_IP"

    for arm_cls in (FrankaROSArm, FrankyArm):
        with pytest.raises(ValueError, match="to be an IP address"):
            arm_cls("LEFT_ROBOT_IP")
        with pytest.raises(ValueError, match="needs a 'robot_ip'"):
            arm_cls("")
        assert arm_cls("10.0.0.1")._robot_ip == "10.0.0.1"


def test_a_connection_cannot_be_composed_into_a_robot():
    from rlinf.robotics.parts.arms.dosw1 import DOSW1Connection
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, RobotPart

    for cls in (Turtle2Connection, DOSW1Connection):
        assert issubclass(cls, Connection), f"{cls.__name__} must still be placeable"
        assert not issubclass(cls, RobotPart), (
            f"{cls.__name__} backs parts without being one; it must not be "
            "composable as a part"
        )
        for absent in ("get_observation", "observation_features"):
            assert not hasattr(cls, absent), (
                f"{cls.__name__}.{absent} exists, so a robot can compose it"
            )


def test_no_robot_builder_absorbs_a_setting_it_does_not_use():
    import inspect

    from rlinf.robotics.robots import (
        DOSW1Robot,
        DualFrankaRobot,
        FrankaRobot,
        GimArmRobot,
        Turtle2Robot,
    )

    # Limit the check to shipped robots because other tests register fakes.
    for robot in (
        FrankaRobot,
        DualFrankaRobot,
        GimArmRobot,
        Turtle2Robot,
        DOSW1Robot,
    ):
        # Forwarding catch-alls are valid only when a downstream API validates keys.
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            robot.build(no_such_setting=True)

        # Retired configuration objects are rejected as well.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        DOSW1Robot.build(config=object())

    # DOSW1 session placement follows the configured node.
    assert "node_rank" in inspect.signature(DOSW1Robot.build).parameters


def test_disconnect_releases_before_it_forgets_the_handle():
    from rlinf.robotics.parts.teleop import TeleopPart

    class Reader:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Device(TeleopPart):
        def __init__(self):
            self.reader = Reader()

        def _open(self):
            return self.reader

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    device = Device()
    device.connect()
    device.disconnect()

    assert device.reader.closed, "the reader was left open"
    assert not device.is_connected


def test_a_camera_can_be_opened_again_after_it_is_closed():
    import numpy as np

    from rlinf.robotics.parts.cameras.base import BaseCamera, CameraInfo

    class Fake(BaseCamera):
        def __init__(self, info):
            super().__init__(info)
            self.opens = 0
            self.releases = 0

        def _open(self):
            self.opens += 1
            return object()

        def _read_frame(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def _release(self, device):
            self.releases += 1

    camera = Fake(CameraInfo(name="c", serial_number="X", camera_type="realsense"))
    camera.connect()
    camera.reopen()

    assert camera.opens == 2
    assert camera.releases == 1
    assert camera.is_connected
    assert camera.get_frame(timeout=2.0).shape == (4, 4, 3)

    camera.disconnect()
    assert camera.releases == 2


def test_a_part_that_would_break_its_worker_is_refused_before_placement():
    from rlinf.robotics.placement.handles import PartWorkerHost

    class Clashing(ControllablePart):
        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        def _open(self):
            return "device"

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

        def attribute(self, name):
            """Conflict with the worker RPC reserved for property access."""

    with pytest.raises(TypeError, match="share a name with the worker"):
        PartWorkerHost.worker_class(Clashing)

        # Shipped drivers avoid worker-method name collisions.
    for driver in (FrankyArm, FrankaROSArm, GimArm):
        assert PartWorkerHost.worker_class(driver) is not None


def test_a_hosted_camera_reopens_on_the_node_that_holds_it():
    from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
    from rlinf.robotics.placement import remote_view_of

    class Fake(BaseCamera):
        def _open(self):
            return object()

        def _read_frame(self):
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def _release(self, device):
            pass

    view = remote_view_of(Fake)
    placed = object.__new__(view)
    group = FakeWorkerGroup()
    placed._group = group
    placed._device = group

    placed.reopen()

    assert [name for name, _ in group.calls] == ["reopen"], (
        "reopen has to travel, or a stalled hosted camera can never recover"
    )
    assert "connect" not in view.__dict__ and "disconnect" not in view.__dict__

    # Local cameras use the same recovery operation directly.
    info = CameraInfo(name="c", serial_number="X", camera_type="realsense")
    assert callable(Fake(info).reopen)


def test_a_held_hand_resumes_from_the_pose_the_env_reset_it_to():
    import numpy as np

    from rlinf.robotics.parts.teleop import Glove

    configured = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    glove = Glove()
    glove.on_reset({"hand_reset_pose": configured})

    first = glove.action({"angles": np.zeros(6)}, {}).parts["hand"]

    assert np.allclose(first, configured), (
        f"the hand starts at {first}, not the configured {configured}"
    )


def test_backed_parts_and_children_are_different_questions():
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, PartGroup, RobotPart

    connection = Turtle2Connection()

    assert set(connection.parts) >= {"left", "left_end_effector"}
    assert not hasattr(connection, "children"), (
        "a connection composes nothing; it only offers"
    )

    robot = PartGroup(arm=connection.part("left"))

    assert set(robot.children) == {"arm"}
    assert robot.parts == {}, "a group backs nothing of its own"
    assert isinstance(connection, Connection)
    assert not isinstance(connection, RobotPart)


def test_describe_says_where_a_part_runs_before_anything_is_opened():
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    from rlinf.robotics.parts.end_effectors import EndEffector

    robot = Bench(
        arm=FrankyArm("10.0.0.1", node_rank=2),
        end_effector=EndEffector.of("franky_gripper", robot_ip="10.0.0.1", node_rank=5),
    )

    described = robot.describe()

    assert "Bench" in described
    assert "arm" in described and "end_effector" in described
    # Each part reports the node it was placed on, before anything opens.
    assert "node=2" in described and "node=5" in described, described
    # Separate connections, so neither is listed as owned by the other.
    assert described.count("FrankyArm#1") == 1, described


def test_a_robot_can_be_disconnected_twice():
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    from rlinf.robotics.parts.end_effectors import EndEffector

    robot = Bench(
        arm=FrankyArm("10.0.0.1"),
        end_effector=EndEffector.of("robotiq_gripper", port="/dev/mock-gripper"),
    )

    with mocked_sdks():
        robot.connect()
        robot.disconnect()
        robot.disconnect()

        assert not robot.is_connected


def test_the_franka_hand_is_reachable_over_libfranka():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector, FrankyGripper

        assert EndEffector.backend("franky_gripper") is FrankyGripper
        # Two drivers for one Franka Hand, not two names for one driver:
        # franka_gripper publishes over ROS, franky_gripper opens libfranka.
        assert EndEffector.backend("franka_gripper") is not FrankyGripper

        gripper = EndEffector.of("franky_gripper", robot_ip="10.0.0.1")
        assert isinstance(gripper, FrankyGripper)
        assert not gripper.is_connected

        # Before connecting, the nominal stroke stands in.
        assert gripper.max_width == pytest.approx(0.08)

        gripper.connect()
        assert gripper.is_connected
        assert gripper.is_ready()
        # Connecting adopts the stroke this hand reports for itself.
        assert gripper.max_width == pytest.approx(gripper._gripper.max_width)

        gripper.disconnect()
        assert not gripper.is_connected
        assert not gripper.is_ready()


def test_the_franka_hand_needs_the_arm_ip_it_hangs_from():
    from rlinf.robotics.parts.end_effectors import FrankyGripper

    with pytest.raises(ValueError, match="arm's own IP"):
        FrankyGripper.declare(port="/dev/ttyUSB0")


def test_the_franka_hand_commands_widths_in_metres():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky_gripper", robot_ip="10.0.0.1")
        gripper.connect()
        sdk = gripper._gripper

        gripper.move(0.03)
        assert sdk.commands[-1][:2] == ("move", 0.03)
        assert gripper.position == pytest.approx(0.03)
        assert not gripper.is_open

        # Beyond the stroke clamps rather than raising.
        gripper.move(0.5)
        assert sdk.commands[-1][1] == pytest.approx(gripper.max_width)
        assert gripper.is_open

        # The end-effector interface reads and writes the same metre axis.
        assert gripper.get_state() == pytest.approx([gripper.max_width])
        gripper.command(np.asarray([0.02], dtype=np.float32))
        assert gripper.position == pytest.approx(0.02)


def test_the_franka_hand_grasps_within_the_force_it_can_apply():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky_gripper", robot_ip="10.0.0.1")
        gripper.connect()
        sdk = gripper._gripper

        # A force written on the Robotiq scale is served at the hand's own.
        gripper.close()
        assert sdk.commands[-1][0] == "grasp"
        assert sdk.commands[-1][3] == pytest.approx(40.0)

        # A force the hand can apply is passed through.
        gripper.close(force=25.0)
        assert sdk.commands[-1][3] == pytest.approx(25.0)

        # Normalized speed maps into the hand's m/s band, never past it.
        gripper.open(speed=1.0)
        assert sdk.commands[-1] == ("open", pytest.approx(0.1))
        gripper.open(speed=0.0)
        assert sdk.commands[-1] == ("open", pytest.approx(0.01))


def test_a_refused_grasp_does_not_end_the_episode():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky_gripper", robot_ip="10.0.0.1")
        gripper.connect()
        sdk = gripper._gripper

        def refuse(*_args, **_kwargs):
            raise RuntimeError("libfranka: command rejected")

        sdk.grasp = refuse
        # Closing on air raises in libfranka; the hand stops and moves instead.
        gripper.close()
        assert [c[0] for c in sdk.commands[-2:]] == ["stop", "move"]
        assert not gripper.is_open

        sdk.open = refuse
        gripper.open()
        assert [c[0] for c in sdk.commands[-2:]] == ["stop", "move"]
        assert gripper.is_open


def test_a_robot_composes_the_hand_its_config_names():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import (
            FrankaGripper,
            FrankyGripper,
            RobotiqGripper,
        )
        from rlinf.robotics.robots import FrankaRobot

        def hand_of(**settings):
            return FrankaRobot.declare_end_effector(
                "10.0.0.1", node_rank=0, name="hand", **settings
            )

        # The built-in hand is one device with two drivers, and the arm backend
        # the robot is built on decides which of them reaches it.
        assert isinstance(hand_of(gripper_type="franka"), FrankaGripper)
        assert isinstance(
            hand_of(gripper_type="franka", backend="franky"), FrankyGripper
        )
        # A config that names a driver outright is taken at its word.
        assert isinstance(hand_of(end_effector_type="franky_gripper"), FrankyGripper)
        assert isinstance(
            hand_of(gripper_type="robotiq", gripper_connection="/dev/ttyUSB0"),
            RobotiqGripper,
        )


def test_a_franka_robot_composes_the_backend_and_hand_it_is_given():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import FrankyGripper
        from rlinf.robotics.robots import FrankaRobot

        robot = FrankaRobot.build(
            robot_ip="10.0.0.1",
            node_rank=0,
            backend="franky",
            gripper_type="franka",
        )
        assert type(robot.child("arm")).__name__ == "FrankyArm"
        assert isinstance(robot.child("end_effector"), FrankyGripper)

        robot.connect()
        assert set(robot.get_observation()) == {"arm", "end_effector"}
        assert robot.child("end_effector").is_connected
        robot.disconnect()
        assert not robot.child("end_effector").is_connected


def test_reading_the_hand_twice_costs_one_round_trip():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky_gripper", robot_ip="10.0.0.1")
        gripper.connect()

        reads = []
        sdk = gripper._gripper
        sdk._width = sdk.width
        type(sdk).width = property(
            lambda self: (reads.append(1), self._width)[1],
            lambda self, value: setattr(self, "_width", value),
        )

        # An observation asks for both, and libfranka cannot produce a newer
        # width within one poll period.
        _, _ = gripper.position, gripper.is_open
        assert len(reads) == 1

        # Commanding the fingers means the cached width no longer describes them.
        gripper.move(0.02)
        _ = gripper.position
        assert len(reads) == 2


def test_one_part_at_a_time_holds_a_hardware_endpoint(tmp_path, monkeypatch):
    from rlinf.robotics.parts import claims
    from rlinf.robotics.parts.claims import DeviceClaim

    monkeypatch.setattr(claims, "_CLAIM_DIR", str(tmp_path))

    held = DeviceClaim("franky-arm:10.0.0.1", "FrankyArm")
    held.acquire()

    # A second part reaching for the same endpoint is told who has it, rather
    # than being left to fail later inside a vendor SDK.
    with pytest.raises(RuntimeError, match="FrankyArm"):
        DeviceClaim("franky-arm:10.0.0.1", "OtherArm").acquire()

    # An arm and the hand mounted on it answer on different endpoints, so
    # holding one says nothing about the other.
    beside = DeviceClaim("franky-hand:10.0.0.1", "FrankyGripper")
    beside.acquire()
    beside.release()

    held.release()
    # Releasing hands the endpoint on.
    DeviceClaim("franky-arm:10.0.0.1", "OtherArm").acquire()


def test_a_claim_survives_a_part_that_fails_to_open(tmp_path, monkeypatch):
    from rlinf.robotics.parts import claims
    from rlinf.robotics.parts.claims import DeviceClaim

    monkeypatch.setattr(claims, "_CLAIM_DIR", str(tmp_path))

    claim = DeviceClaim("robotiq:/dev/ttyUSB0", "RobotiqGripper")
    try:
        with claim:
            raise RuntimeError("the port was there but the gripper was not")
    except RuntimeError:
        pass

    # A failed open must not strand the endpoint for the rest of the session.
    DeviceClaim("robotiq:/dev/ttyUSB0", "RobotiqGripper").acquire()


def test_ros_parts_share_one_session_per_process():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.transports.ros import ROSController

        ROSController._shared = None
        try:
            first = ROSController.shared()
            # ROS 1 gives a process one node, so asking twice has to answer
            # with the session that node already belongs to.
            assert ROSController.shared() is first
        finally:
            ROSController._shared = None


def test_a_ros_hand_opens_without_an_arm_to_hand_it_a_session():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector, FrankaGripper

        hand = EndEffector.of("franka_gripper")
        assert isinstance(hand, FrankaGripper)

        # No arm involved: the hand joins the session itself.
        hand.connect()
        try:
            assert hand.is_connected
            assert hand._ros is not None
        finally:
            hand.disconnect()
        assert not hand.is_connected
        # A hand that has let go of the session is not ready, whatever the
        # topic is still delivering to whoever else is listening.
        assert not hand.is_ready()


def test_a_later_subscriber_does_not_unready_an_existing_one():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.transports.ros import ROSController

        ROSController._shared = None
        try:
            session = ROSController.shared()
            session.connect_ros_channel("/topic", object, lambda _msg: None)
            session._input_channel_status["/topic"] = True

            # A second part on the same session subscribes to the same topic.
            session.connect_ros_channel("/topic", object, lambda _msg: None)

            # The topic has been delivering all along, so the part already
            # reading it must not be told it is no longer ready.
            assert session.get_input_channel_status("/topic")
        finally:
            ROSController._shared = None


def test_child_says_what_a_part_is_expected_to_be():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.franky import FrankyArm
        from rlinf.robotics.parts.end_effectors import BaseGripper
        from rlinf.robotics.robots import FrankaRobot

        robot = FrankaRobot.build(
            robot_ip="10.0.0.1",
            node_rank=0,
            backend="franky",
            gripper_type="franka",
        )

        # Naming the class is what lets a caller reach the driver's own methods
        # with an editor able to resolve them.
        hand = robot.child("end_effector", BaseGripper)
        assert callable(hand.open) and callable(hand.close)
        assert robot.child("arm", FrankyArm) is robot.child("arm")

        # A composition that does not match is reported here, by name, rather
        # than as a missing attribute somewhere later.
        with pytest.raises(TypeError, match="not the FrankyArm"):
            robot.child("end_effector", FrankyArm)


def test_a_stalled_camera_is_reopened_before_the_caller_sees_the_error():
    import queue

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.cameras import Camera, CameraInfo

        camera = Camera.of(CameraInfo(name="wrist", serial_number="MOCK0001"))
        camera.connect()
        try:
            reopens = []
            camera.reopen = lambda: reopens.append(1)

            def stalled(timeout=None):
                raise queue.Empty

            camera._frame_queue.get = stalled

            # Reading is where a stall is discovered, so recovering happens
            # here rather than in every loop that reads a camera.
            with pytest.raises(queue.Empty, match="after 3 attempts"):
                camera.get_frame(timeout=0.01, attempts=3)
            assert len(reopens) == 3
        finally:
            camera.disconnect()


def test_a_franka_env_commands_the_arm_through_the_robot():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.base import Arm
        from rlinf.robotics.robots import FrankaRobot

        # Both Franka arm backends accept a Cartesian target through the part
        # interface, which is what lets one env drive either of them.
        for backend in ("franka_ros", "franky"):
            robot = FrankaRobot.build(
                robot_ip="10.0.0.1",
                node_rank=0,
                backend=backend,
                gripper_type="franka",
            )
            robot.connect()
            try:
                assert "tcp_pose" in robot.child("arm", Arm).action_features
                pose = np.array([0.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])
                applied = robot.send_action({"arm": {"tcp_pose": pose}})
                assert set(applied) == {"arm"}
            finally:
                robot.disconnect()


def test_an_arm_reports_compliance_settings_it_cannot_apply(caplog):
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.parts.base import Features

    class PlainArm(Arm):
        def __init__(self, address: str) -> None:
            self.address = address

        @property
        def observation_features(self) -> Features:
            return {}

        @property
        def action_features(self) -> Features:
            return {}

        def _open(self):
            return "device"

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

    arm = PlainArm("10.0.0.1")
    arm.reconfigure_compliance_params({})
    with caplog.at_level("WARNING"):
        arm.reconfigure_compliance_params({"translational_stiffness": 500})
    assert "cannot change compliance" in caplog.text


def test_every_arm_answers_the_operations_the_contract_names():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.base import Arm

        for name in sorted(Arm.backends()):
            arm = Arm.backend(name)
            # A caller reaches for these without asking which backend it has.
            assert callable(arm.is_robot_up)
            assert callable(arm.clear_errors)
            assert callable(arm.reset_joint)
            assert callable(arm.reconfigure_compliance_params)


def test_an_arm_that_cannot_reset_its_joints_says_so():
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.parts.base import Features

    class PlainArm(Arm):
        def __init__(self, address: str) -> None:
            self.address = address

        @property
        def observation_features(self) -> Features:
            return {}

        @property
        def action_features(self) -> Features:
            return {}

        def _open(self):
            return "device"

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

    arm = PlainArm("10.0.0.1")

    # Readiness follows the connection when the backend has no other signal.
    assert not arm.is_robot_up()
    arm.connect()
    assert arm.is_robot_up()
    # Nothing latched, nothing to clear.
    arm.clear_errors()

    # A reset that cannot happen is refused rather than silently skipped, so a
    # caller is never left believing the arm moved.
    with pytest.raises(NotImplementedError, match="cannot reset its joints"):
        arm.reset_joint([0.0] * 7)
    arm.disconnect()


def test_so101_reports_joints_in_radians_and_the_gripper_as_a_fraction():
    """lerobot speaks degrees and 0..100; every other arm here speaks radians."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.so101 import SO101Arm

        arm = SO101Arm("/dev/mock-so101", calibration_id="bench")
        arm.connect()
        arm._robot.positions.update(
            {"shoulder_pan.pos": 90.0, "elbow_flex.pos": -45.0, "gripper.pos": 25.0}
        )

        state = arm.get_state()
        assert state.arm_joint_position[0] == pytest.approx(np.pi / 2)
        assert state.arm_joint_position[2] == pytest.approx(-np.pi / 4)
        assert state.gripper_position[0] == pytest.approx(0.25)

        # The arm reports joints only: it has no pose, force, or Jacobian.
        assert set(arm.observation_features) == {"arm_joint_position"}
        assert set(arm.get_observation()) == {"arm_joint_position"}

        arm.disconnect()


def test_so101_commands_are_converted_back_to_degrees():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.so101 import SO101Arm

        arm = SO101Arm("/dev/mock-so101")
        arm.connect()

        sent = arm.send_action({"joint_position": [np.pi / 2, 0.0, 0.0, 0.0, 0.0]})

        wire = arm._robot.sent[-1]
        assert wire["shoulder_pan.pos"] == pytest.approx(90.0)
        # send_action reports what actually reached the arm, in radians.
        assert sent["joint_position"][0] == pytest.approx(np.pi / 2)

        arm.disconnect()


def test_so101_gripper_rides_the_arm_connection():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.so101 import SO101Arm

        arm = SO101Arm("/dev/mock-so101")
        arm.connect()

        gripper = arm.children["end_effector"]
        # It borrows the arm's connection rather than opening its own.
        assert gripper.owner is arm

        # Continuous, not binary: a partial opening reaches the servo.
        assert gripper.control_mode == "continuous"
        gripper.command(np.array([0.4]))
        assert arm._robot.sent[-1]["gripper.pos"] == pytest.approx(40.0)

        arm._robot.positions["gripper.pos"] = 60.0
        assert gripper.get_state()[0] == pytest.approx(0.6)

        arm.disconnect()


def test_so101_never_prompts_for_calibration_and_refuses_an_uncalibrated_arm():
    """Calibrating asks the operator to move the arm, so a worker cannot."""
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.arms.so101 import SO101Arm

        follower = made["lerobot.robots.so101_follower"].SO101Follower

        arm = SO101Arm("/dev/mock-so101", calibration_id="bench")
        arm.connect()
        # lerobot's calibrate() blocks on input(); it must never be reached.
        assert arm._robot.calibrate_calls == 0
        arm.disconnect()

        follower.calibrated = False
        try:
            uncalibrated = SO101Arm("/dev/mock-so101", calibration_id="missing")
            with pytest.raises(RuntimeError, match="not calibrated"):
                uncalibrated.connect()
            assert not uncalibrated.is_connected
        finally:
            follower.calibrated = True


def test_so101_reset_waits_for_the_servos_to_stop_moving():
    """The servo bus returns while the arm travels, so a reset waits for it."""
    from types import SimpleNamespace

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.so101 import SO101Arm

        arm = SO101Arm.declare("/dev/ttyACM0", calibration_id="test")
        arm.connect()
        # Three frames of travel, then a pose it holds a little short of the
        # target, as the servos do under gravity.
        frames = [[0.0] * 5, [0.3] * 5, [0.6] * 5] + [[0.62] * 5] * 20
        seen: list[list[float]] = []

        def report():
            frame = frames[min(len(seen), len(frames) - 1)]
            seen.append(frame)
            return SimpleNamespace(arm_joint_position=np.array(frame, dtype=float))

        arm.get_state = report
        arm.reset_joint([0.0] * 5)
        # It read on until two frames agreed, rather than reporting the pose
        # the arm was leaving.
        assert len(seen) == 5

        arm.disconnect()


def test_piper_reset_waits_for_the_arm_to_stop_moving():
    """move_j returns while the arm is still travelling, so a reset waits."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()
        frames = [[0.0] * 6, [0.3] * 6, [0.6] * 6] + [[0.62] * 6] * 20
        seen: list[list[float]] = []

        def report():
            frame = frames[min(len(seen), len(frames) - 1)]
            seen.append(frame)
            return np.array(frame, dtype=float)

        arm._joint_reading = report
        arm.reset_joint([0.0] * 6)
        assert len(seen) == 5

        arm.disconnect()


def test_an_arm_that_never_stops_gives_up_instead_of_blocking():
    """A jammed or driven arm must not hold a reset open for ever."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()

        def climbing():
            climbing.angle += 0.5
            return np.full(6, climbing.angle)

        climbing.angle = 0.0
        arm._joint_reading = climbing

        start = time.monotonic()
        arm.wait_until_still(0.2)
        assert 0.2 <= time.monotonic() - start < 1.0

        arm.disconnect()


def test_piper_reports_radians_metres_and_a_quaternion():
    """pyAgxArm speaks radians and metres; only the pose needs converting."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()
        arm._robot.joints = [0.1, 0.5, -0.4, 0.2, 0.3, -0.1]
        # A quarter turn about Z, in the SDK's extrinsic xyz Euler.
        arm._robot.pose = [0.3, 0.05, 0.25, 0.0, 0.0, np.pi / 2]
        arm._gripper.status.value = 0.035

        state = arm.get_state()
        assert state.arm_joint_position[1] == pytest.approx(0.5)
        assert state.tcp_pose.shape == (7,)
        assert state.tcp_pose[:3] == pytest.approx([0.3, 0.05, 0.25])
        assert state.tcp_pose[3:] == pytest.approx(
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
        )
        # A fraction of the stroke, not a width.
        assert state.gripper_position[0] == pytest.approx(0.5)
        assert set(arm.observation_features) == {"arm_joint_position", "tcp_pose"}

        arm.disconnect()


def test_piper_commands_go_out_in_radians_and_are_held_to_the_travel():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()

        sent = arm.send_action({"joint_position": [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]})
        assert arm._robot.sent[-1] == pytest.approx([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
        assert sent["joint_position"][1] == pytest.approx(1.0)

        # Joints 2 and 3 travel one way only, so a request through zero is
        # clipped rather than faulting the arm.
        arm.send_action({"joint_position": [0.0, -5.0, 5.0, 0.0, 0.0, 0.0]})
        assert arm._robot.sent[-1][1] == pytest.approx(0.0)
        assert arm._robot.sent[-1][2] == pytest.approx(0.0)

        arm.disconnect()


def test_piper_gripper_rides_the_arm_connection():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()

        gripper = arm.children["end_effector"]
        # Borrows the arm's CAN session rather than opening its own.
        assert gripper.owner is arm
        assert gripper.control_mode == "continuous"

        gripper.command(np.array([0.5]))
        assert arm._gripper.sent[-1] == {"value": 0.035, "force": 1.0}

        arm._gripper.status.value = 0.07
        assert gripper.get_state()[0] == pytest.approx(1.0)

        arm.disconnect()


def test_piper_without_a_gripper_exports_no_end_effector():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0", with_gripper=False)
        arm.connect()
        try:
            assert "end_effector" not in arm.children
            with pytest.raises(RuntimeError, match="without a "):
                arm.move_gripper([1.0])
        finally:
            arm.disconnect()


def test_piper_refuses_to_hand_over_an_arm_that_never_enabled():
    """Enabling is what makes a command take effect, so it has to succeed."""
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.arms.piper import PiperArm

        fake = made["pyAgxArm"].AgxArmFactory.create_arm({}).__class__
        fake.enables = False
        try:
            arm = PiperArm.declare("can0")
            arm.ENABLE_TIMEOUT_S = 0.05
            with pytest.raises(RuntimeError, match="did not enable"):
                arm.connect()
            assert not arm.is_connected
            # A refused open must not leave the channel claimed.
            second = PiperArm.declare("can0")
            second.ENABLE_TIMEOUT_S = 0.05
            with pytest.raises(RuntimeError, match="did not enable"):
                second.connect()
        finally:
            fake.enables = True


def test_piper_clearing_errors_does_not_cut_motor_power():
    """The SDK's reset() and disable() drop a raised arm; neither is used."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()

        arm.clear_errors()
        assert arm._robot.cleared == 1
        assert arm._robot.enabled, "clearing a fault must leave the motors on"

        driver = arm._robot
        arm.disconnect()
        assert not driver.is_connected()
        assert driver.enabled


def test_a_single_backend_robot_does_not_load_every_arm_driver():
    """SO-101 has one driver, so it names it instead of asking the registry.

    A registry lookup imports every registered arm module to populate itself,
    which is worth paying only where a config can select among them.
    """
    import sys

    from robot_mocks import mocked_sdks

    def loaded():
        return {name for name in sys.modules if ".parts.arms." in name}

    with mocked_sdks():
        from rlinf.robotics.robots.so101 import SO101Robot

        before = loaded()
        robot = SO101Robot.build(port="/dev/mock-so101", node_rank=0)
        try:
            pulled = {name.split(".")[-1] for name in loaded() - before}
            assert pulled <= {"so101"}, f"also imported {sorted(pulled - {'so101'})}"
        finally:
            robot.disconnect()


def test_piper_asks_the_arm_which_protocol_it_speaks():
    """The profile frames the CAN messages, so guessing it talks nonsense."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0")
        arm.connect()
        try:
            # The mock arm reports S-V1.8-9, which resolves to v189 -- not the
            # "default" profile, which covers S-V1.8-2 and older.
            assert arm._robot.config["firmeware_version"] == "v189"
        finally:
            arm.disconnect()


def test_piper_takes_the_firmware_profile_it_is_given():
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.arms.piper import PiperArm

        arm = PiperArm.declare("can0", firmware="v183")
        arm.connect()
        try:
            assert arm._robot.config["firmeware_version"] == "v183"
            # Pinning it means the arm is never asked.
            assert arm._robot.get_firmware()["software_version"] == "S-V1.8-9"
        finally:
            arm.disconnect()
        del made


def test_piper_refuses_to_guess_when_the_arm_will_not_say():
    from robot_mocks import mocked_sdks

    with mocked_sdks() as made:
        from rlinf.robotics.parts.arms.piper import PiperArm

        fake = made["pyAgxArm"].AgxArmFactory.create_arm({}).__class__
        fake.firmware_version = None
        try:
            arm = PiperArm.declare("can0")
            with pytest.raises(RuntimeError, match="did not report a firmware"):
                arm.connect()
            assert not arm.is_connected
        finally:
            fake.firmware_version = "S-V1.8-9"


def test_piper_backend_is_named_for_its_sdk():
    """A second driver for the same arm must be able to register beside it."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms import Arm
        from rlinf.robotics.parts.arms.piper import PiperArm
        from rlinf.robotics.robots.piper import PiperRobot

        # Named for the SDK, so "piper" stays free for another driver.
        assert Arm.backend("pyagxarm") is PiperArm
        assert PiperRobot.BACKEND == "pyagxarm"
        # Lookup is case-insensitive, so a config may spell it pyAgxArm.
        assert Arm.backend("pyAgxArm") is PiperArm
        with pytest.raises(ValueError, match="Unsupported"):
            Arm.backend("piper")


def test_a_piper_robot_takes_the_backend_its_config_names():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.piper import PiperArm
        from rlinf.robotics.robots.piper import PiperRobot

        robot = PiperRobot.build(can_channel="can0", node_rank=0, backend="pyAgxArm")
        try:
            assert isinstance(robot.child("arm"), PiperArm)
        finally:
            robot.disconnect()

        with pytest.raises(ValueError, match="Unsupported"):
            PiperRobot.build(can_channel="can0", node_rank=0, backend="piper_sdk")


def test_piper_takes_no_separately_wired_end_effector():
    """The AgxGripper is on the arm's own bus, not a fitted device."""
    from rlinf.robotics.parts.arms.piper import PiperArm

    with pytest.raises(TypeError, match="gripper_connection"):
        PiperArm.declare("can0", gripper_connection="/dev/ttyUSB0")


def test_so101_takes_no_separately_wired_end_effector():
    """The gripper is a servo on the arm's own bus, not a fitted device."""
    from rlinf.robotics.parts.arms.so101 import SO101Arm

    with pytest.raises(TypeError, match="gripper_connection"):
        SO101Arm.declare("/dev/mock-so101", gripper_connection="/dev/ttyUSB0")


def test_an_arm_takes_the_compliance_its_robot_was_configured_with():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.base import CartesianCompliance
        from rlinf.robotics.robots import FrankaRobot

        settings = CartesianCompliance(translational_stiffness=900.0, max_step=0.02)
        robot = FrankaRobot.build(
            robot_ip="10.0.0.1",
            node_rank=0,
            backend="franky",
            gripper_type="franka",
            compliance=settings,
        )
        arm = robot.child("arm")
        # The settings belong to the arm, so a second arm may hold others and a
        # remotely placed one carries them to the node it opens on.
        assert arm._compliance is settings
        assert arm._cart_k_t == 900.0

        # An arm whose controller owns its gains is offered the same settings
        # and ignores them, rather than refusing to be built.
        ros = FrankaRobot.declare_arm(
            "10.0.0.1",
            node_rank=0,
            name="arm",
            backend="franka_ros",
            compliance=settings,
        )
        assert type(ros).__name__ == "FrankaROSArm"


def test_a_compliance_request_is_held_to_what_the_backend_can_run():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.base import CartesianCompliance
        from rlinf.robotics.parts.arms.franky import FrankyArm

        arm = FrankyArm.declare(
            "10.0.0.1",
            compliance=CartesianCompliance(stiffness_cap=1200.0, clip_floor=0.005),
        )
        arm.connect()
        try:
            # A task written for a real-time controller asks for more stiffness
            # and tighter clips than a client-side loop can hold.
            arm.reconfigure_compliance_params(
                {
                    "translational_stiffness": 2000,
                    "rotational_stiffness": 150,
                    "translational_clip_x": 0.003,
                    "translational_clip_neg_x": 0.001,
                    "rotational_clip_x": 0.001,
                }
            )
            assert arm._cart_k_t == 1200.0
            assert arm._cart_k_r == 80.0
            # The looser of each direction pair, floored.
            assert arm._cart_trans_clip[0] == pytest.approx(0.005)
            assert arm._cart_rot_clip[0] == pytest.approx(0.02)

            # Nothing asked for, nothing changed.
            before = arm._cart_k_t
            arm.reconfigure_compliance_params({})
            assert arm._cart_k_t == before
        finally:
            arm.disconnect()


def test_a_yaml_compliance_mapping_becomes_settings():
    from omegaconf import OmegaConf

    from rlinf.robotics.parts.arms.base import CartesianCompliance
    from rlinf.robotics.robots.franka import FrankaConfig

    default = FrankaConfig(node_rank=0, robot_ip="172.16.0.2")
    assert isinstance(default.compliance, CartesianCompliance)

    for mapping in (
        {"translational_stiffness": 1000, "rotational_stiffness": 50},
        OmegaConf.create({"translational_stiffness": 1000, "rotational_stiffness": 50}),
    ):
        config = FrankaConfig(node_rank=0, robot_ip="172.16.0.2", compliance=mapping)
        assert isinstance(config.compliance, CartesianCompliance)
        assert config.compliance.translational_stiffness == pytest.approx(1000)
        assert config.compliance.rotational_stiffness == pytest.approx(50)
        # Unstated settings keep their defaults.
        assert config.compliance.max_step == pytest.approx(
            CartesianCompliance().max_step
        )

    with pytest.raises(KeyError, match="translational_stifness"):
        FrankaConfig(
            node_rank=0,
            robot_ip="172.16.0.2",
            compliance={"translational_stifness": 1000},
        )
