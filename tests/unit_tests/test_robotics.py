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

"""The robotics layer: parts, composition, placement, and its boundaries."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest

import rlinf.robotics.robots.franka as franka_module
from rlinf.robotics import (
    Camera,
    ControllablePart,
    DOSW1Robot,
    DOSW1RobotConfig,
    DualFrankaRobot,
    EndEffector,
    Endpoint,
    FrankaRobot,
    GimArmConfig,
    Group,
    LegacyObservationAdapter,
    MethodArm,
    MethodGripper,
    RemotePartHandle,
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
from rlinf.robotics.placement.specs import PartSpec
from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareResource,
    NodeHardwareConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- from test_robotics.py --------------------------------------------


class FakePart(RobotPart):
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def observation_features(self) -> dict[str, dict]:
        return {"state": {"shape": (1,)}}

    def connect(self) -> None:
        self.events.append(f"connect:{self.name}")
        self._connected = True

    def get_observation(self) -> dict[str, np.ndarray]:
        return {"state": np.array([1.0])}

    def disconnect(self) -> None:
        self.events.append(f"disconnect:{self.name}")
        self._connected = False


class FakeControllablePart(FakePart, ControllablePart):
    @property
    def action_features(self) -> dict[str, dict]:
        return {"target": {"shape": (1,)}}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return action


class FakeEndEffector(FakePart, EndEffector):
    @property
    def action_features(self) -> dict[str, dict]:
        return {"target": {"shape": (1,)}}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return action


class FakeCamera(FakePart, Camera):
    pass


class FakeRemoteResult:
    def __init__(self, value: Any):
        self.value = value

    def wait(self) -> list[Any]:
        return [self.value]


class FakeMethodDriver:
    """A driver whose components are reachable only through named methods."""

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


class FakeWorkerGroup:
    """Stand-in for the one-worker group behind a RemotePartHandle."""

    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def shutdown(self) -> FakeRemoteResult:
        self.calls.append(("shutdown", None))
        return FakeRemoteResult(None)

    def part_observation(self, name: str) -> FakeRemoteResult:
        self.calls.append(("part_observation", name))
        return FakeRemoteResult({"tcp_pose": np.zeros(7)})

    def part_action(self, name: str, action: Any) -> FakeRemoteResult:
        self.calls.append(("part_action", (name, action)))
        return FakeRemoteResult(action)

    def is_robot_up(self) -> FakeRemoteResult:
        self.calls.append(("is_robot_up", None))
        return FakeRemoteResult(True)

    def _close(self) -> None:
        self.calls.append(("_close", None))


def test_robot_composes_and_namespaces_parts():
    events: list[str] = []
    arm = Group(
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
    # The observation mirrors the composition: names, not fixed categories.
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
    arm = Group(arm=FakeControllablePart("driver", events), wrist=camera)
    robot = Robot(arm=arm)
    robot.connect()
    camera.disconnect()

    robot.disconnect()

    assert "disconnect:driver" in events


def test_driver_views_expose_composed_part_api():
    """A method-shaped driver decomposes into parts without any remoting."""
    driver = FakeMethodDriver()
    arm = MethodArm(
        driver,
        commands={"tcp_pose": "move_arm"},
        state_fields=("tcp_pose", "arm_joint_position"),
    )
    end_effector = MethodGripper(driver, state_field="gripper_position")
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


def test_remote_handle_releases_its_worker_group():
    group = FakeWorkerGroup()
    handle = RemotePartHandle(
        group,
        {"arm": {"kind": "controllable", "observation": {}, "action": {}}},
    )

    handle.disconnect()
    handle.disconnect()  # idempotent

    assert [name for name, _ in group.calls] == ["shutdown", "_close"]


def test_remote_handle_forwards_off_interface_driver_methods():
    """Methods outside the part interface reach the driver unchanged."""
    group = FakeWorkerGroup()
    handle = RemotePartHandle(group, {})

    assert handle.is_robot_up().wait()[0] is True


def test_robot_requires_non_empty_string_part_names():
    with pytest.raises(ValueError, match="non-empty strings"):
        Robot(parts={0: FakePart("camera", [])})  # type: ignore[dict-item]


def test_builtin_robots_expose_standard_composition_layouts():
    events: list[str] = []
    left_arm = Group(
        arm=FakeControllablePart("left_arm", events),
        gripper=FakeEndEffector("left_gripper", events),
    )
    right_arm = Group(
        arm=FakeControllablePart("right_arm", events),
        gripper=FakeEndEffector("right_gripper", events),
    )
    third_arm = Group(arm=FakeControllablePart("third_arm", events))

    single = FrankaRobot(arm=left_arm, front_camera=FakeCamera("front", events))
    dual = DualFrankaRobot(
        left=left_arm, right=right_arm, base_camera=FakeCamera("base", events)
    )
    # Names are the composition, so nothing caps the count or the kind.
    triple = FrankaRobot(left=left_arm, right=right_arm, third=third_arm)

    assert set(single.children) == {"arm", "front_camera"}
    assert set(single.parts_of_type(Group)) == {"arm"}
    assert set(single.parts_of_type(EndEffector)) == {"arm.gripper"}
    assert set(single.parts_of_type(Camera)) == {"front_camera"}
    assert set(dual.children) == {"left", "right", "base_camera"}
    assert set(dual.parts_of_type(Group)) == {"left", "right"}
    assert set(triple.children) == {"left", "right", "third"}
    assert set(triple.parts_of_type(Group)) == {"left", "right", "third"}


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
        endpoint: str = "loopback"

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
        configs=cast(Any, [{"node_rank": 3, "endpoint": "robot.local"}]),
    )

    assert registered is TestRobotDiscovery
    assert RobotDiscovery.registry["TestRobot"].robot_cls is TestRobot
    assert TestRobotDiscovery in Hardware.policy_registry
    assert isinstance(parsed.configs[0], TestRobotConfig)
    assert parsed.configs[0].endpoint == "robot.local"

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
    """Observation and action keys are the names the robot was composed with."""
    robot = Robot(arm=Group(arm=FakeControllablePart("arm", [])))
    robot.connect()

    observation = robot.get_observation()
    action = {"arm": {"arm": {"target": np.array([0.25])}}}

    assert observation["arm"]["arm"]["state"].shape == (1,)
    assert robot.send_action(action) == action
    robot.disconnect()


def test_robot_releases_driver_handles_after_parts():
    """Parts borrow a connection; the robot closes it once they are done."""
    events: list[str] = []

    class FakeHandle:
        def disconnect(self) -> None:
            events.append("handle")

    robot = Robot(arm=Group(arm=FakeControllablePart("driver", events)))
    robot.handles["arm"] = FakeHandle()
    robot.connect()
    robot.disconnect()

    assert events[-1] == "handle"
    assert "disconnect:driver" in events


def test_driver_rejects_actions_for_observation_only_parts():
    class CameraOnlyHost(FakePart):
        @property
        def exports(self) -> dict[str, RobotPart]:
            return {"wrist": FakeCamera("wrist", [])}

    with pytest.raises(TypeError, match="not controllable"):
        CameraOnlyHost("host", []).part_action("wrist", {})


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
    """Every built-in robot config is usable with only its placement set."""
    configs = [
        GimArmConfig(node_rank=0),
        DOSW1RobotConfig(node_rank=0),
        Turtle2Config(node_rank=0),
    ]

    assert all(config.node_rank == 0 for config in configs)


def test_every_registered_robot_carries_a_builder():
    """Registration is the single source of truth for a robot type.

    A robot module registers its config, robot, discovery, and builder in one
    call, so composing by type name never needs a central dispatch table.
    """
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

    # A link to one arm is that arm, and commands it.
    arms = [
        FrankaROSArm("10.0.0.1"),
        FrankyArm("10.0.0.1"),
        GimArm("can0", "gim_arm_xl", True, "parallel"),
    ]
    # A bus driving several components is no one of them.
    buses = [Turtle2Connection()]

    assert all(isinstance(driver, ControllablePart) for driver in arms)
    assert all(isinstance(driver, Connection) for driver in buses)
    # Every one of them is an endpoint -- declared, placed, opened, closed --
    # but only the arms are parts, because only they mean something when read.
    assert all(isinstance(driver, Endpoint) for driver in arms + buses)
    assert all(isinstance(driver, RobotPart) for driver in arms)
    assert not any(isinstance(driver, RobotPart) for driver in buses)
    assert all(not driver.is_connected for driver in arms + buses)
    # Each declares the parts riding on its connection.
    assert all(driver.exports for driver in arms + buses)


def _fake_arm_backend(monkeypatch, *, failing_ip=None, disconnected=None):
    """Point FrankaRobot at a fake arm class that records what it places."""

    class FakeHandle:
        def __init__(self, name: str):
            self.name = name
            self.exports = {"arm": FakeControllablePart(name, [])}

        def part_named(self, _name: str):
            return self.exports["arm"]

        def disconnect(self) -> None:
            if disconnected is not None:
                disconnected.append(self.name)

    class FakeArm:
        @classmethod
        def at(cls, *args, node_rank=None, name=None, **kwargs):
            return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

        @staticmethod
        def spawn(robot_ip, *args, node_rank=None, name=None, **kwargs):
            if failing_ip is not None and robot_ip == failing_ip:
                raise RuntimeError("right arm is unreachable")
            return FakeHandle(robot_ip)

    monkeypatch.setattr(franka_module, "franka_arm_cls", lambda backend: FakeArm)
    return FakeArm


def test_declaring_arms_places_nothing_until_connect(monkeypatch):
    """Declarations are inert. ``connect`` is what touches hardware."""

    class NeverSpawns:
        @classmethod
        def at(cls, *args, node_rank=None, name=None, **kwargs):
            return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

        @staticmethod
        def spawn(*args, **kwargs):
            raise AssertionError("spawn must not run while declaring")

    monkeypatch.setattr(franka_module, "franka_arm_cls", lambda backend: NeverSpawns)

    robot = FrankaRobot(
        arm=FrankaRobot.declare_arm("10.0.0.1", node_rank=0, name="left")
    )

    assert not robot.is_connected


def test_connect_tears_down_parts_already_placed(monkeypatch):
    """A half-placed robot is never left behind when a later part fails."""
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
    """Nothing in declaration or placement is specific to one or two arms."""
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


def test_one_declaration_is_placed_once_however_often_referenced():
    """A connection backing several components opens exactly once."""
    placements: list[str] = []

    class FakeHandle:
        def __init__(self):
            self.exports = {
                "left": FakeControllablePart("left", []),
                "right": FakeControllablePart("right", []),
                "wrist": FakeCamera("wrist", []),
            }

        def part_named(self, name):
            return self.exports[name]

        def disconnect(self):
            pass

    class CoupledHardware:
        @classmethod
        def at(cls, *args, node_rank=None, name=None, **kwargs):
            return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

        @staticmethod
        def spawn(*args, **kwargs):
            placements.append("placed")
            return FakeHandle()

    hardware = CoupledHardware.at(node_rank=0)
    robot = Robot(
        left=hardware.export("left"),
        right=hardware.export("right"),
        wrist=hardware.export("wrist"),
    )
    robot.connect()

    assert placements == ["placed"], "the shared connection was opened more than once"
    assert isinstance(robot.child("wrist"), Camera)
    assert robot.is_connected


def test_local_handle_subpart_returns_a_part_not_a_forwarded_call():
    """The handle's accessors must win over its catch-all forwarding.

    ``LocalPartHandle.__getattr__`` forwards unknown names to the hosted part
    and wraps the result for call-shape symmetry. If an accessor like
    ``subpart`` were missing from the handle, that forwarding would silently
    return a result wrapper instead of a part, and the failure would surface
    far away as a missing attribute on the composed arm.
    """
    events: list[str] = []

    class HostWithSubparts(FakeControllablePart):
        @property
        def exports(self) -> dict[str, RobotPart]:
            return {"arm": self, "end_effector": FakeEndEffector("ee", events)}

    handle = HostWithSubparts.spawn("host", events)

    assert isinstance(handle.part_named("arm"), RobotPart)
    assert isinstance(handle.part_named("end_effector"), EndEffector)
    assert set(handle.exports) == {"arm", "end_effector"}
    # Off-interface names still forward, and still wrap.
    assert handle.get_observation().wait()[0]["state"].shape == (1,)


def test_any_part_can_be_placed_not_only_arms():
    """Placement is a property of parts, so a camera can be spawned alone."""
    events: list[str] = []
    handle = FakeCamera.spawn("wrist", events)

    assert handle.part.is_connected
    assert "connect:wrist" in events
    handle.disconnect()
    assert "disconnect:wrist" in events


def test_every_robot_owns_its_construction():
    """Construction and registration are the robot class's own behaviour.

    ``build`` must be a classmethod bound to the registered class, not a loose
    module function handed to the registry, so a subclass overriding it is what
    ``build_robot`` dispatches to.
    """
    registry = RobotDiscovery.registry

    for name, registration in registry.items():
        build = registration.build
        assert build is not None, f"{name} registered no builder"
        assert getattr(build, "__self__", None) is registration.robot_cls, (
            f"{name}'s builder is not bound to {registration.robot_cls.__name__}"
        )


def test_dual_franka_inherits_declaration_from_franka():
    """Arm count and backend are the only differences between the two."""
    assert issubclass(DualFrankaRobot, FrankaRobot)
    # declare_arm is inherited; only build_arms differs.
    assert DualFrankaRobot.declare_arm.__func__ is FrankaRobot.declare_arm.__func__
    assert DualFrankaRobot.build_arms.__func__ is not FrankaRobot.build_arms.__func__, (
        "only the arm count differs, and that is what build_arms says"
    )
    # Switching the control backend is one class attribute, and it serves any
    # arm count because declare_arm is shared.
    assert (FrankaRobot.BACKEND, DualFrankaRobot.BACKEND) == ("franka_ros", "franky")
    # build_arms carries the entire difference; everything else is inherited.
    overridden = [
        name
        for name in ("declare_arm", "build_arms", "build_cameras", "build")
        if getattr(DualFrankaRobot, name).__func__
        is not getattr(FrankaRobot, name).__func__
    ]
    assert overridden == ["build_arms"]


def test_every_part_kind_places_independently():
    """Arm, end effector, and camera are separate parts, each with its own node.

    A Robotiq gripper is a serial device of its own and a camera holds its own
    USB link, so neither has to ride the arm's connection or its node.
    """
    placed: dict[str, int] = {}

    class Handle:
        exports: dict[str, RobotPart] = {}

        def __init__(self, part):
            self._part = part

        @property
        def part(self):
            return self._part

        def part_named(self, name):
            raise KeyError(name)

        def disconnect(self):
            pass

    def fake(kind, base):
        class Fake(base):
            def __init__(self, *args, **kwargs):
                self._connected = False

            @property
            def is_connected(self):
                return self._connected

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def connect(self):
                self._connected = True

            def disconnect(self):
                self._connected = False

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

            @classmethod
            def spawn(cls, *args, node_rank=None, name=None, **kwargs):
                placed[kind] = node_rank
                part = cls()
                part.connect()
                return Handle(part)

        return Fake

    robot = Robot(
        **{
            "arm": Group(
                arm=fake("arm", ControllablePart).at("10.0.0.1", node_rank=1),
                gripper=fake("gripper", EndEffector).at(
                    port="/dev/ttyUSB0", node_rank=2
                ),
                wrist=fake("camera", Camera).at(node_rank=3),
            )
        }
    )
    robot.connect()

    assert placed == {"arm": 1, "gripper": 2, "camera": 3}
    arm = robot.child("arm")
    assert isinstance(arm.child("gripper"), EndEffector)
    assert isinstance(arm.child("wrist"), Camera)
    assert robot.is_connected


def test_a_leaf_part_placed_remotely_still_exposes_itself():
    """A camera has no parts, so its handle must proxy the part itself.

    Without this, declaring a camera on its own node would resolve to nothing.
    """

    class Leaf(Camera):
        @property
        def is_connected(self):
            return True

        @property
        def observation_features(self):
            return {"frame": {}}

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_observation(self):
            return {"frame": None}

    described = Leaf().describe_self()

    assert described["kind"] == "camera"
    assert described["observation"] == {"frame": {}}
    assert Leaf().exports == {}


def test_declaring_cameras_needs_no_config_class():
    """A camera descriptor plus a node is all a declaration needs."""
    from rlinf.robotics.parts.cameras import CameraInfo, declare_cameras

    info = CameraInfo(name="scene", serial_number="123", camera_type="realsense")
    declared = declare_cameras({"scene": info}, node_rank=4)

    assert set(declared) == {"scene"}
    assert declared["scene"].node_rank == 4
    assert declared["scene"].part_cls.__name__ == "RealSenseCamera"
    assert declare_cameras(None) == {}


def test_failed_connect_can_be_retried():
    """A failed connect restores declarations instead of poisoning the robot.

    Without this, an arm that placed successfully would keep a part whose
    handle was released during rollback, and a retry would place nothing.
    """
    placed: list[str] = []
    state = {"failing": True}

    class Handle:
        exports: dict = {}

        def __init__(self, part):
            self._part = part

        @property
        def part(self):
            return self._part

        def part_named(self, name):
            raise KeyError(name)

        def disconnect(self):
            pass

    def maker(tag, flaky=False):
        class Made(ControllablePart):
            def __init__(self, *args, **kwargs):
                self._connected = False

            @property
            def is_connected(self):
                return self._connected

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def connect(self):
                self._connected = True

            def disconnect(self):
                self._connected = False

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

            @classmethod
            def spawn(cls, *args, node_rank=None, name=None, **kwargs):
                if flaky and state["failing"]:
                    raise RuntimeError("hardware unreachable")
                placed.append(tag)
                part = cls()
                part.connect()
                return Handle(part)

        return Made

    robot = Robot(
        **{
            "left": maker("left").at(node_rank=1),
            "right": maker("right", flaky=True).at(node_rank=2),
        }
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert robot.handles == {}, "handles from the aborted attempt were kept"
    assert isinstance(robot.child("left"), PartSpec), (
        "the successful arm kept a part whose handle was released"
    )

    state["failing"] = False
    robot.connect()

    assert placed == ["left", "left", "right"], "the retry did not re-place"
    assert robot.is_connected

    robot.disconnect()
    assert isinstance(robot.child("left"), PartSpec)
    robot.connect()
    assert robot.is_connected, "a disconnected robot must be connectable again"


def test_a_robot_owned_camera_is_placed_opened_and_closed():
    """Cameras follow the same lifecycle as every other part.

    They used to be built by the environment and bolted on after connect, which
    meant a camera could never sit on the node it was plugged into.
    """
    events: list[str] = []

    class Cam(Camera):
        def __init__(self, *args, **kwargs):
            self._connected = False

        @property
        def is_connected(self):
            return self._connected

        @property
        def observation_features(self):
            return {"frame": {}}

        def connect(self):
            self._connected = True
            events.append("open")

        def disconnect(self):
            self._connected = False
            events.append("close")

        def get_observation(self):
            return {"frame": None}

        @classmethod
        def spawn(cls, *args, node_rank=None, name=None, **kwargs):
            events.append(f"placed@{node_rank}")
            part = cls()

            class Handle:
                exports: dict = {}

                @property
                def part(self):
                    return part

                def part_named(self, name):
                    raise KeyError(name)

                def disconnect(self):
                    events.append("released")

            return Handle()

    robot = Robot(
        arm=Group(arm=FakeControllablePart("arm", []), wrist=Cam.at(node_rank=5))
    )
    robot.connect()
    robot.disconnect()

    assert events == ["placed@5", "open", "close", "released"]


# --- from test_robotics_boundaries.py ---------------------------------


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_scheduler_has_no_robotics_dependency():
    scheduler_dir = _ROOT / "rlinf" / "scheduler"
    offenders = {
        path.relative_to(_ROOT): module
        for path in scheduler_dir.rglob("*.py")
        for module in _imports(path)
        if module == "rlinf.robotics" or module.startswith("rlinf.robotics.")
    }

    assert offenders == {}


def test_pure_driver_import_does_not_load_scheduler():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from rlinf.robotics.parts.arms.franky import FrankyArm; "
                "assert 'rlinf.scheduler' not in sys.modules; "
                "assert not FrankyArm('10.0.0.1').is_connected"
            ),
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


_SCHEDULER_BRIDGE = Path("rlinf") / "robotics" / "placement" / "handles.py"


def test_robotics_devices_do_not_depend_on_scheduler_ray_or_gym():
    robotics_dir = _ROOT / "rlinf" / "robotics"
    device_paths = [
        robotics_dir / "robot.py",
        robotics_dir / "adapters.py",
        *robotics_dir.joinpath("parts").rglob("*.py"),
    ]
    forbidden = ("ray", "gymnasium", "rlinf.scheduler")
    offenders = {
        path.relative_to(_ROOT): module
        for path in device_paths
        if path.relative_to(_ROOT) != _SCHEDULER_BRIDGE
        for module in _imports(path)
        if module == forbidden or module.startswith(forbidden)
    }

    assert offenders == {}


def test_scheduler_use_is_confined_to_the_composition_layer():
    """Only composition code may see the scheduler; hardware code never does.

    ``rlinf.robotics`` is two layers. Hardware -- parts: arms, cameras, end
    effectors, teleop devices -- is scheduler-free so it runs from plain
    scripts.
    Composition -- the placement bridge, robot builders, and hardware discovery
    -- is allowed to use the scheduler. This pins the boundary between them, so
    a new driver cannot quietly reach for ``Cluster``.
    """
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
    """Every environment sits in one of the two halves, with no alias behind it.

    The pre-split paths were kept working by a ``sys.meta_path`` alias while
    callers moved over. Nothing imports them now, so the alias is gone and the
    old names resolve to nothing.
    """
    envs = _ROOT / "rlinf" / "envs"
    # Only directories carrying source count. A package that moved leaves its
    # __pycache__ behind, and stale bytecode on one machine is not a stray env
    # package -- it would fail this for whoever had checked out the old tree.
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
    """A leader arm is an arm with encoders, so it is a part.

    What makes something robotics here is that it talks to hardware without
    importing Ray or Gymnasium, not whether a policy observes it. As parts these
    devices get placement, the connect and disconnect lifecycle, and one
    connection opened once -- none of which they had as env-side helpers.
    """
    from rlinf.robotics.parts.base import RobotPart
    from rlinf.robotics.parts.teleop import (
        Glove,
        PicoController,
        SpaceMouse,
        TeleopLeaderArm,
    )

    devices = (SpaceMouse, TeleopLeaderArm, Glove, PicoController)

    assert all(issubclass(device, RobotPart) for device in devices)
    assert not (_ROOT / "rlinf" / "envs" / "real" / "teleop" / "devices").exists()

    # Declaring one opens nothing, so a device can be described anywhere.
    mouse = SpaceMouse()
    assert not mouse.is_connected
    assert sorted(mouse.observation_features) == ["buttons", "twist"]


def test_teleop_readers_do_not_import_gymnasium():
    """A reader only talks to its vendor SDK, so a bench script can drive it."""
    readers = _ROOT / "rlinf" / "robotics" / "parts" / "teleop" / "readers"
    offenders = {
        path.name
        for path in readers.glob("*.py")
        if re.search(r"^\s*(import|from)\s+gymnasium\b", path.read_text(), re.M)
    }

    assert offenders == set()


def test_importing_a_teleop_device_does_not_load_the_env_stack():
    """A bench script driving one serial device should not need the env stack.

    ``toolkits/realworld_check`` runs on the machine the device is plugged into,
    which may have no Gymnasium, no OpenCV, and no cluster. Reaching a reader
    goes through ``rlinf.robotics`` alone now, and that package loads lazily.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import rlinf.robotics.parts.teleop.readers.gello\n"
            "leaked = {m.split('.')[0] for m in sys.modules} & {'gymnasium', 'cv2', 'ray'}\n"
            "assert not leaked, sorted(leaked)\n",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# --- the real composition, exercised through DOSW1's dummy SDK ---------------
#
# Everything above this line drives fakes, which pins the contracts but cannot
# catch a robot whose own parts, placement, or lifecycle is wrong. DOSW1 ships a
# dummy mode that runs the real arm, gripper, and SDK-adapter code without
# hardware, so these tests go through production paths end to end.


def _dosw1_robot():
    from rlinf.robotics.robots import DOSW1Robot

    return DOSW1Robot.build(is_dummy=True)


def test_building_a_real_robot_touches_no_hardware():
    """``build`` leaves declarations, so a robot can be described anywhere.

    The machine assembling a robot config is often not the machine holding the
    hardware, so nothing may be opened until ``connect``.
    """
    from rlinf.robotics.parts.base import Group
    from rlinf.robotics.placement.specs import PartSpec, SubpartRef

    robot = _dosw1_robot()

    assert not robot.is_connected
    # Both arms reference one declaration, because one SDK session drives them
    # all; the references resolve against it when connect places it.
    leaves = [
        leaf
        for part in robot.children.values()
        for leaf in (part.children.values() if isinstance(part, Group) else [part])
    ]
    assert leaves, "the robot declared no parts"
    assert all(isinstance(leaf, (PartSpec, SubpartRef)) for leaf in leaves)
    # Compared by identity, not equality: a PartSpec carries its constructor
    # kwargs, which routinely hold unhashable values, so Placement dedupes on
    # id() too.
    specs = [leaf.spec for leaf in leaves if isinstance(leaf, SubpartRef)]
    assert specs and all(spec is specs[0] for spec in specs)


def test_real_robot_lifecycle_without_hardware():
    """Connect, read, command, reset, disconnect, and connect again."""
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

    # A disconnected robot goes back to its declarations and can run again.
    robot.connect()
    assert robot.is_connected
    robot.disconnect()


def test_one_connection_is_opened_once_for_every_part_it_drives():
    """DOSW1 drives both arms over a single SDK session.

    Placing the declaration twice would open that session twice, which the
    hardware does not allow. Both arms therefore share one handle.
    """
    robot = _dosw1_robot()
    robot.connect()

    assert robot.handles["left"] is robot.handles["right"]

    robot.disconnect()


def test_coupled_hardware_exposes_its_components_as_parts():
    """One connection, four components, named by the robot that composed them."""
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
    """Every path in the observation is a name the robot was composed with."""
    robot = _dosw1_robot()
    robot.connect()

    observation = robot.get_observation()
    paths = {
        f"{group}.{part}" for group, parts in observation.items() for part in parts
    }

    assert paths == set(robot.named_parts) - set(robot.children)

    robot.disconnect()


# --- teleop composition ------------------------------------------------------


def _scripted_device(reading):
    """A teleop part that always reports `reading`."""
    from rlinf.robotics.parts.teleop.devices import TeleopPart

    class Scripted(TeleopPart):
        def _open(self):
            return object()

        @property
        def observation_features(self):
            return {key: {} for key in reading}

        def get_observation(self):
            return dict(reading)

    device = Scripted()
    device.connect()
    return device


def _kinds(*names):
    """What a robot with these parts expects, for a group's `available`."""
    from rlinf.robotics.teleop import ActionKind

    per_name = {
        "hand": ActionKind.HAND,
        "end_effector": ActionKind.GRIPPER,
    }
    return {
        name: per_name.get(name.rsplit(".", 1)[-1], ActionKind.CARTESIAN_DELTA)
        for name in names
    }


def _binding(produces, value, driving=True):
    from rlinf.robotics.teleop import TeleopBinding

    class Fixed(TeleopBinding):
        PRODUCES = _kinds(*produces)

        def action(self, reading, context):
            from rlinf.robotics.teleop import TeleopAction

            return TeleopAction(parts=dict.fromkeys(produces, value), driving=driving)

        def is_driving(self, reading):
            return driving

    return Fixed()


def test_two_devices_merge_into_one_action():
    """A dex-hand rig is a spacemouse on the arm and a glove on the hand."""
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(_scripted_device({"twist": 1}), _binding(("arm",), np.ones(6))),
            TeleopEntry(
                _scripted_device({"angles": 2}), _binding(("hand",), np.ones(6) * 2)
            ),
        ],
        available=_kinds(*("arm", "hand")),
    )

    parts, driving, _ = group.action({})

    assert sorted(parts) == ["arm", "hand"]
    assert driving
    assert np.allclose(parts["hand"], 2.0)


def test_a_part_the_robot_lacks_is_not_filled():
    """A spacemouse offers a gripper; an arm carrying a hand has none."""
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(
                _scripted_device({"twist": 1}),
                _binding(("arm", "end_effector"), np.ones(6)),
            )
        ],
        available=_kinds(*("arm", "hand")),
    )

    assert group.parts == ("arm",)


def test_two_devices_claiming_one_part_is_refused():
    """Silently letting one win would move the robot from the wrong device."""
    import numpy as np
    import pytest

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="both drive"):
        TeleopGroup(
            [
                TeleopEntry(_scripted_device({}), _binding(("arm",), np.ones(6))),
                TeleopEntry(_scripted_device({}), _binding(("arm",), np.ones(6))),
            ],
            available=_kinds(*("arm",)),
        )


def test_a_device_that_fills_nothing_is_refused():
    """A rig that cannot drive anything is a configuration mistake."""
    import numpy as np
    import pytest

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="fills none"):
        TeleopGroup(
            [TeleopEntry(_scripted_device({}), _binding(("hand",), np.ones(6)))],
            available=_kinds(*("arm", "end_effector")),
        )


def test_drives_separates_two_identical_leaders():
    """Both leaders produce an arm; `drives` says which branch each fills."""
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(
                _scripted_device({}), _binding(("arm",), np.ones(7)), drives="left"
            ),
            TeleopEntry(
                _scripted_device({}), _binding(("arm",), np.ones(7) * 2), drives="right"
            ),
        ],
        available=_kinds(*("left.arm", "right.arm")),
    )

    parts, _, _ = group.action({})

    assert sorted(parts) == ["left.arm", "right.arm"]
    assert np.allclose(parts["right.arm"], 2.0)


def test_one_device_listed_twice_is_read_once():
    """A spacemouse drives an arm and a gripper over one HID handle."""
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    device = _scripted_device({"twist": 1})
    group = TeleopGroup(
        [
            TeleopEntry(device, _binding(("arm",), np.ones(6))),
            TeleopEntry(device, _binding(("end_effector",), np.ones(1))),
        ],
        available=_kinds(*("arm", "end_effector")),
    )

    assert len(group.devices) == 1
    assert group.parts == ("arm", "end_effector")


def test_the_glove_holds_what_the_operator_posed():
    """Releasing the arm device leaves the hand where it was put.

    Under composition an unbound part would keep the policy's value, so the
    hold the dex-hand setup has always had is stated in the binding.
    """
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding

    glove = GloveBinding()
    glove.reset({"hand_reset_pose": np.zeros(6)})

    posed = glove.action({"angles": np.full(6, 0.4)}, {"hand_driving": True}).parts[
        "hand"
    ]
    released = glove.action({"angles": np.full(6, 0.9)}, {"hand_driving": False}).parts[
        "hand"
    ]

    assert np.allclose(posed, released)


def test_the_glove_tracks_only_while_its_gate_is_held():
    """A spacemouse button is what puts the glove in control.

    Devices in one rig are not independent. Stating the gate as published
    context keeps that ordered and visible, rather than hidden inside a class
    that reads both devices.
    """
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding, SpaceMouseBinding

    mouse, glove = SpaceMouseBinding(), GloveBinding()
    glove.reset({"hand_reset_pose": np.zeros(6)})

    released = mouse.publish({"twist": np.zeros(6), "buttons": [False, False]})
    held = mouse.publish({"twist": np.zeros(6), "buttons": [False, True]})

    assert released == {"hand_driving": False}
    assert held == {"hand_driving": True}

    # Held: the first reading re-zeros, a later one moves the hand.
    glove.action({"angles": np.zeros(6)}, held)
    moved = glove.action({"angles": np.full(6, 0.3)}, held).parts["hand"]
    assert np.allclose(moved, 0.3)

    # Released: the hand stays where it was posed.
    assert np.allclose(
        glove.action({"angles": np.full(6, 0.9)}, released).parts["hand"], 0.3
    )


# --- PICO, as a device and a binding ----------------------------------


class _ScriptedController:
    """A controller part replaying a fixed sequence of readings."""

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
    """An absolute binding turns a delta into where the arm should end up."""
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0)
    device = _ScriptedController([((0.025, 0.0, 0.0), (0.0, 0.0, 0.0), True, -1)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    # The arm goes where the operator moved from where it was when they grabbed.
    assert parts["arm"].size == 9
    assert np.isclose(parts["arm"][0], 0.3 + 0.025)
    assert np.isclose(parts["end_effector"][0], -1.0)
    assert binding.action(device.get_observation(), _pico_context()).driving


def test_releasing_the_grip_mid_chunk_holds_the_arm_where_it_was_left():
    """Snapping back to the policy's pose would jerk the arm mid-motion."""
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0, hold_current_when_inactive=False)
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

    # The next chunk of policy actions releases it.
    binding.on_action_chunk_begin()
    assert binding.action(device.get_observation(), context).parts == {}


def test_holding_the_current_pose_leaves_the_gripper_to_the_policy():
    """The operator is not touching the gripper, so the policy still owns it."""
    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0, hold_current_when_inactive=True)
    device = _ScriptedController([((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    assert set(parts) == {"arm"}


def test_a_delta_binding_has_no_pose_to_hold():
    """Zero motion already holds a robot that takes deltas."""
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding

    assert PicoBinding(gripper=True).hold(_pico_context()) == {}
    assert "arm" in PicoTcpBinding(gripper=True).hold(_pico_context())


def test_absolute_commands_are_clipped_but_deltas_are_not():
    """A pose can leave the action space; a normalised delta cannot."""
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding, SpaceMouseBinding

    assert PicoTcpBinding.CLIPS_TO_ACTION_SPACE
    assert not PicoBinding.CLIPS_TO_ACTION_SPACE
    assert not SpaceMouseBinding.CLIPS_TO_ACTION_SPACE


def test_each_side_reports_its_own_state():
    """Two controllers report separately, so a collector can tell them apart."""
    from rlinf.robotics.teleop import (
        ActionKind,
        PicoTcpBinding,
        TeleopEntry,
        TeleopGroup,
    )

    layout = {
        "left.arm": ActionKind.CARTESIAN_POSE,
        "left.end_effector": ActionKind.GRIPPER,
        "right.arm": ActionKind.CARTESIAN_POSE,
        "right.end_effector": ActionKind.GRIPPER,
    }
    entries = [
        TeleopEntry(
            _ScriptedController(
                [((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), side == "left", 1)]
            ),
            PicoTcpBinding(gripper=True, side=index),
            drives=side,
        )
        for index, side in enumerate(("left", "right"))
    ]
    group = TeleopGroup(entries, available=layout)

    _, driving, info = group.action(_pico_context())

    assert driving
    assert info["left_pico_active"] is True
    assert info["right_pico_active"] is False


def test_reading_a_controller_does_not_need_the_robot():
    """The part reports data, so nothing about the robot reaches the reader."""
    import inspect

    from rlinf.robotics.parts.teleop.devices import PicoController
    from rlinf.robotics.parts.teleop.readers import pico

    source = inspect.getsource(pico.PicoExpert)
    for name in ("tcp_pose", "_ref_tcp"):
        assert name not in source, f"the reader still refers to {name!r}"

    observation = PicoController.get_observation
    assert "get_reading" in inspect.getsource(observation)


def test_a_controller_reading_is_data_not_a_handle():
    """An observation another part could record, rather than an object."""
    from rlinf.robotics.parts.teleop.devices import PicoController

    features = PicoController(hand="right").observation_features

    assert set(features) == {
        "held",
        "position_delta",
        "rotation_delta",
        "grip_close",
        "grip_open",
    }


def test_the_arm_anchors_where_it_was_when_the_operator_took_hold():
    """Re-taking hold re-anchors, so the arm does not jump."""
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0)
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

    # Same motion from the same measured pose, so the same command both times.
    assert np.isclose(first, again)


# --- the controller reader, from packets ------------------------------


def _pico_packet(x, y, z, yaw, grip, close=False, open_=False):
    """One frame as the headset publishes it."""
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
    """A reader wired to scripted packets rather than to a headset."""
    from rlinf.robotics.parts.teleop.readers.pico import PicoExpert

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
    """Not from the origin: grabbing at an arbitrary pose must not jump.

    The headset's axes are not the robot's. Its ``-z`` is the robot's ``+x``,
    so this also pins down the mapping the reader applies.
    """
    import numpy as np

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    reader._snapshot = lambda: _pico_packet(0.5, 0.5, 0.5, 0.0, grip=0.95)
    reader.get_reading()  # takes hold here, at (0.5, 0.5, 0.5)

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

    # The last frame re-took hold where it stands, so nothing has moved yet.
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
    """The whole path, from packet to the numbers the env receives."""
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    binding = PicoTcpBinding(gripper=True, side=0)
    context = _pico_context()

    reader._snapshot = lambda: _pico_packet(0.0, 0.0, 0.0, 0.0, grip=0.95)
    binding.action(reader.get_reading(), context)

    reader._snapshot = lambda: _pico_packet(0.0, 0.0, -0.04, 0.0, grip=0.95)
    reading = reader.get_reading()
    parts = binding.action(reading, context).parts

    # The arm is asked for where it was when they grabbed, plus their motion.
    expected = np.asarray(context["tcp_pose"][:3]) + reading["position_delta"]
    assert np.allclose(parts["arm"][:3], expected, atol=1e-6)
    assert np.isclose(reading["position_delta"][0], 0.04, atol=1e-9)


# --- one lifecycle, whatever kind of part it is -----------------------


def test_every_part_family_opens_and_closes_the_same_way():
    """A part author learns _open/_release once, not once per device family."""
    import inspect

    from rlinf.robotics.parts.base import RobotPart
    from rlinf.robotics.parts.cameras.base import BaseCamera
    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector
    from rlinf.robotics.parts.teleop.devices import TeleopPart

    for family in (TeleopPart, BaseCamera, BaseEndEffector):
        assert hasattr(family, "_open"), f"{family.__name__} has no _open"
        assert hasattr(family, "_release"), f"{family.__name__} has no _release"

    # The names the families used to use are gone.
    for family in (BaseCamera, BaseEndEffector):
        source = inspect.getsource(family)
        for retired in ("_close_device", "def initialize", "def shutdown"):
            assert retired not in source, f"{family.__name__} still has {retired}"

    # Teardown belongs to the endpoint, so a connection gets it too.
    assert RobotPart.shutdown.__qualname__ == "Endpoint.shutdown"
    assert BaseEndEffector.shutdown.__qualname__ == "Endpoint.shutdown"


class _Part(RobotPart):
    """A part whose hardware is a list of what happened to it."""

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


def test_teardown_goes_through_disconnect():
    """RobotPart.shutdown is worker teardown, not a hardware method.

    An end effector used to override it, so teardown released the hardware
    without ever passing through disconnect: the part stayed 'connected'.
    """
    part = _Part()
    part.connect()
    part.shutdown()

    assert part.log == ["open", "close"]
    assert not part.is_connected


def test_a_part_that_says_nothing_about_its_hardware_fails_clearly():
    class Bare(RobotPart):
        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    with pytest.raises(NotImplementedError, match="does not say how to open"):
        Bare().connect()


# --- the smallest robot that works ------------------------------------


def test_a_robot_is_named_parts_and_nothing_else():
    """The whole of a working robot, with no registry and no hardware config.

    The shipped robots all declare discovery and a hardware config because they
    are reached by type name from a config file. A robot you construct yourself
    needs neither, and this is what that costs.
    """
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
    """A robot you construct yourself never needs build(), and is told so."""
    from rlinf.robotics import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    with pytest.raises(NotImplementedError, match="Construct Bench"):
        Bench.build()


# --- a connection is not a part ---------------------------------------


def test_a_connection_backing_several_parts_is_not_observable():
    """One ROS node driving two arms is no part, and says so.

    It used to satisfy the part interface with an observation of the whole
    robot and a coupled two-arm action that nothing called.
    """
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, ControllablePart

    assert issubclass(Turtle2Connection, Connection)
    assert not issubclass(Turtle2Connection, ControllablePart)

    hardware = Turtle2Connection()
    assert set(hardware.exports) == {
        "left",
        "left_end_effector",
        "right",
        "right_end_effector",
    }
    # Not "refuses to be read" but "is not the kind of thing you read": the
    # methods are absent, so a robot cannot compose it into the tree and find
    # out at the first step.
    assert not isinstance(hardware, RobotPart)
    assert not hasattr(hardware, "get_observation")
    assert not hasattr(hardware, "observation_features")


def test_every_robot_composes_from_named_parts():
    """No robot puts a whole connection in its tree.

    A connection that also happens to be a part -- a ROS link to one arm --
    then lands under its own name rather than under a name that repeats it.
    """
    from rlinf.robotics.placement.specs import PartSpec
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
        DualFrankaRobot.build_arms(left_robot_ip="1.2.3.4", right_robot_ip="1.2.3.5"),
    ]
    for arms in built:
        for name, value in arms.items():
            values = (
                list(value.children.values()) if hasattr(value, "children") else [value]
            )
            assert not any(isinstance(v, PartSpec) for v in values), (
                f"{name} holds a whole connection rather than its parts"
            )


def test_one_connection_backs_every_part_that_names_it():
    """Both halves of an arm share a link; two arms do not share one."""
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot
    from rlinf.robotics.robots.franka import FrankaRobot

    single = FrankaRobot.build_arms(robot_ip="1.2.3.4", node_rank=0)
    assert len({id(ref.spec) for ref in single.values()}) == 1

    dual = DualFrankaRobot.build_arms(left_robot_ip="1.2.3.4", right_robot_ip="1.2.3.5")
    sides = {
        side: {id(ref.spec) for ref in group.children.values()}
        for side, group in dual.items()
    }
    assert len(sides["left"]) == 1 and len(sides["right"]) == 1
    assert sides["left"] != sides["right"]


def test_a_connection_with_several_parts_resolves_to_all_of_them():
    """The path a robot takes when it does hold a whole declaration.

    Its import was wrong from the move that split placement into a package, so
    this raised ModuleNotFoundError rather than composing anything.
    """
    from rlinf.robotics.parts.base import Group
    from rlinf.robotics.placement.specs import Placement

    class Pair(RobotPart):
        def _open(self):
            return "link"

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

        @property
        def exports(self):
            return {"a": _Part(), "b": _Part()}

    resolved = Placement().resolve(Pair.at())
    assert isinstance(resolved, Group)
    assert set(resolved.children) == {"a", "b"}


# --- the bench check, checked ------------------------------------------


def _fake_robot_type(broken=None):
    """Register a robot made of fakes, and yield its type name."""
    import numpy as np

    from rlinf.robotics.discovery import RobotConfig, RobotDiscovery, register_robot
    from rlinf.robotics.parts.base import Connection, ControllablePart, Group
    from rlinf.robotics.parts.views import MethodArm, MethodGripper
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
        def exports(self):
            return {
                "left": MethodArm(
                    self, commands={"tcp_pose": "move"}, state_fields=("tcp_pose",)
                ),
                "left_end_effector": MethodGripper(
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
            bus = Bus.at()
            return cls(
                left=Group(
                    arm=bus.export("left"),
                    gripper=bus.export("left_end_effector"),
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
    """The script the bench runs, run here against fakes."""
    assert _run_bench(_fake_robot_type()) == 0


def test_the_bench_check_catches_an_observation_that_was_never_declared():
    assert _run_bench(_fake_robot_type("Mismatch")) == 1


def test_the_bench_check_catches_a_connection_left_in_the_tree():
    assert _run_bench(_fake_robot_type("Connection")) == 1


# --- real parts, faked SDKs -------------------------------------------


def test_no_part_hook_collides_with_the_worker_group():
    """A placed part becomes a worker, and its private methods are attached.

    `_close` was the obvious name for the lifecycle hook and it is taken:
    `WorkerGroup._close` already exists, so every remotely placed part failed
    to launch. Only placement caught it, which no unit test did.
    """
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
    """The camera class itself, from connect to frame to disconnect."""
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
    """The arm class itself: connect, the parts it backs, state, commands."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.franky import FrankyArm

        arm = FrankyArm("10.0.0.1", gripper_connection="/dev/ttyUSB0")
        arm.connect()

        assert arm.is_connected
        assert set(arm.exports) == {"arm", "end_effector"}

        observation = arm.get_observation()
        assert observation["tcp_pose"].shape == (7,)
        assert arm.exports["end_effector"].get_observation()["state"].shape == (1,)

        arm.send_action({"joint_position": [0.0] * 7})

        arm.disconnect()
        assert not arm.is_connected


def test_the_bench_check_runs_a_whole_robot_on_fakes():
    """The command a bench runs, run here on the same code path."""
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
    """All five, through compose, connect, observe and disconnect.

    Building the fakes is itself the check: each field a part reads has to be
    there, with the shape the part expects, or this fails.
    """
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
    """A part placed on a node is built in a process that never saw them.

    ``sitecustomize`` in the mock package is what closes that gap, so a mocked
    run can exercise real placement rather than only in-process construction.
    """
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
    """The teleop layout is read from the wrapped env, not the bare one.

    ``GripperCloseEnv`` holds the gripper shut and drops that channel, so an
    env declaring seven numbers presents six. Composing teleop above it raised
    "the declaration and step() disagree" until the wrapper said what it took
    away -- which only a full stack shows, because that is where wrappers are
    applied before teleop.
    """
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
    """``arm_cameras`` is declared, passed, and consumed by nobody.

    ``DualFrankaEnv`` splits its cameras into per-arm and robot-level, then
    hands the per-arm ones to ``DualFrankaRobot.build(arm_cameras=...)``. That
    builder takes ``**config`` and forwards it to ``build_arms``, which ignores
    what it does not recognise, so the wrist cameras are dropped in silence and
    the env's observation space asks for frames no part produces.

    Found by an end-to-end run on faked SDKs, where reset returned no
    observation at all.
    """
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
    """A dual-arm robot places a camera per wrist, both on the same node.

    A name built from the class and the node alone is the same string twice,
    and Ray refuses the second actor rather than placing it, so the robot comes
    up with one wrist camera and an observation space asking for two.
    """
    from rlinf.robotics.parts.cameras.realsense import RealSenseCamera
    from rlinf.robotics.placement import PartWorkerHost

    names = {PartWorkerHost.default_name(RealSenseCamera, 0) for _ in range(8)}

    assert len(names) == 8, f"names repeat: {sorted(names)}"
    assert all(name.startswith("RealSenseCamera-node0-") for name in names), (
        f"a name should still say what it is and where: {sorted(names)}"
    )


def test_every_real_task_is_registered_through_the_shared_factory():
    """A task registered by hand does not fit the call the env worker makes.

    ``task_factory`` is what passes ``env_cfg`` to the wrapper stack and keeps
    it out of the env's constructor. A package that calls ``register`` itself
    builds an entry point taking different arguments, so the task can be
    constructed by hand and never by the runner.
    """
    import re

    offenders = []
    for path in (_ROOT / "rlinf" / "envs" / "real").rglob("__init__.py"):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*register\(", line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == [], f"these register a task outside register_tasks: {offenders}"


def test_a_dual_arm_config_can_switch_its_address_check_off():
    """Every other robot config can; this one rejected the field outright.

    A config whose addresses are placeholders, or whose arms are faked, has
    nothing to validate, and being unable to say so is what stopped a dual-arm
    config from being written at all.
    """
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig

    with pytest.raises(ValueError, match="valid IP address"):
        DualFrankaConfig(node_rank=0, left_robot_ip="LEFT_ROBOT_IP")

    relaxed = DualFrankaConfig(
        node_rank=0, left_robot_ip="LEFT_ROBOT_IP", disable_validate=True
    )

    assert relaxed.left_robot_ip == "LEFT_ROBOT_IP"


def test_a_connection_cannot_be_composed_into_a_robot():
    """The tree holds parts. A connection backs them and is not one.

    Before the split this was a policy: ``Connection`` inherited
    ``get_observation`` and overrode it to raise, so a connection composed into
    a robot by mistake type-checked fine and failed at the first read. Now the
    method does not exist on it, and the two shapes that used to disagree --
    a connection that refused to be read, and one that answered by repeating
    its own parts -- are the same shape.
    """
    from rlinf.robotics.parts.arms.dosw1 import DOSW1Connection
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, Endpoint, RobotPart

    for cls in (Turtle2Connection, DOSW1Connection):
        assert issubclass(cls, Connection)
        assert issubclass(cls, Endpoint), f"{cls.__name__} must still be placeable"
        assert not issubclass(cls, RobotPart), (
            f"{cls.__name__} is a connection; it must not be composable as a part"
        )
        for absent in ("get_observation", "observation_features"):
            assert not hasattr(cls, absent), (
                f"{cls.__name__}.{absent} exists, so a robot can compose it"
            )


def test_a_retired_dosw1_config_object_is_refused_not_ignored():
    """``build`` absorbs unknown keywords, which is how a caller goes quiet.

    It used to take one config object. A call still passing ``config=`` would
    land in ``**_`` and leave every setting at its default -- ``is_dummy``
    included, so a session meant to skip the SDK would reach for it and fail
    somewhere else entirely.
    """
    from rlinf.robotics.robots import DOSW1Robot

    with pytest.raises(TypeError, match="no longer takes a config object"):
        DOSW1Robot.build(config=object())


def test_disconnect_releases_before_it_forgets_the_handle():
    """``_release`` is what closes the vendor object, so it needs it.

    Clearing ``_device`` first left every implementation reading it -- which is
    where the handle is documented to live, and what ``TeleopPart`` uses to
    close a reader by whichever name its vendor gave the method -- closing
    nothing. The part reported itself disconnected while the reader, its
    thread and its serial port stayed open, for GELLO, gloves, PICO and the
    spacemouse alike.
    """
    from rlinf.robotics.parts.teleop.devices import TeleopPart

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
    """Stall recovery closes a camera and opens it again.

    A thread runs once, so a camera holding one from ``__init__`` raised
    ``RuntimeError: threads can only be started once`` the second time -- on
    the recovery path both Franka envs take when a camera stops producing
    frames.
    """
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


def test_a_hosted_camera_reopens_on_the_node_that_holds_it():
    """``connect`` and ``disconnect`` are no-ops on a proxy.

    The handle owns a hosted part's lifetime, so calling those two on a stalled
    remote camera would do nothing at all. ``reopen`` is the operation that
    crosses the boundary.
    """
    from rlinf.robotics.placement.handles import RemoteCamera

    class Result:
        def wait(self):
            return [None]

    class Group:
        def __init__(self):
            self.calls = []

        def reopen(self):
            self.calls.append("reopen")
            return Result()

    group = Group()
    RemoteCamera(group, None, {}).reopen()

    assert group.calls == ["reopen"]


def test_a_held_hand_resumes_from_the_pose_the_env_reset_it_to():
    """The env homes the hand, then the binding has to agree with it.

    Resetting with no context zeroed the commanded pose, so the first takeover
    command moved the hand away from where the env had just put it. On hardware
    that is an abrupt command.
    """
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding

    configured = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    glove = GloveBinding()
    glove.reset({"hand_reset_pose": configured})

    first = glove.action({"angles": np.zeros(6)}, {}).parts["hand"]

    assert np.allclose(first, configured), (
        f"the hand starts at {first}, not the configured {configured}"
    )


def test_exports_and_children_are_different_questions():
    """One word used to answer both, and they are not the same question.

    ``exports`` is what a hardware session offers; ``children`` is what a robot
    was composed of. Composition is where the two meet -- it picks exports and
    names them -- and nowhere else. Sharing a name meant a reader could not
    tell which of the two a given call site meant, and the docs got it wrong.
    """
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, Group

    connection = Turtle2Connection()

    assert set(connection.exports) >= {"left", "left_end_effector"}
    assert not hasattr(connection, "children"), (
        "a connection composes nothing; it only offers"
    )

    robot = Group(arm=connection.export("left"))

    assert set(robot.children) == {"arm"}
    assert robot.exports == {}, "a group offers nothing of its own"
    assert isinstance(connection, Connection)


def test_describe_says_where_a_part_runs_before_anything_is_opened():
    """The point of describing a robot is to check it without one present.

    Placement and ownership live in the declaration, so both are answerable
    before connecting -- and a connected part no longer carries a declaration,
    which is why the robot keeps the snapshot and reads it either side.
    """
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    declared = FrankyArm.at("10.0.0.1", node_rank=2)
    robot = Bench(arm=declared.export("arm"), gripper=declared.export("end_effector"))

    described = robot.describe()

    assert "Bench" in described
    assert "arm" in described and "gripper" in described
    assert "node=2" in described, described
    # Both names come from one declaration, so both say the same connection.
    assert described.count("FrankyArm#1") == 2, described


def test_a_robot_can_be_disconnected_twice():
    """Teardown calls disconnect when it is not sure the robot came up.

    Disconnecting restores the declarations, so the second call walked a tree
    holding those and asked a ``SubpartRef`` whether it was connected. The
    ``AttributeError`` that produced would surface from inside a ``finally``,
    replacing whatever error the caller was actually handling.

    Found by the robot conformance suite, against every shipped robot at once.
    """
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    declared = FrankyArm.at("10.0.0.1", gripper_connection="/dev/mock-gripper")
    robot = Bench(arm=declared.export("arm"), gripper=declared.export("end_effector"))

    with mocked_sdks():
        robot.connect()
        robot.disconnect()
        robot.disconnect()

        assert not robot.is_connected
