# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS 2 transport adapter for the main dual-Franka environment."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, TypeVar

import numpy as np

from rlinf.envs.realworld.common.gripper import create_gripper
from rlinf.envs.realworld.franka.franka_robot_state import FrankaRobotState
from rlinf.utils.logging import get_logger

FR3_JOINT_LIMITS_LOWER = np.array(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
)
FR3_JOINT_LIMITS_UPPER = np.array(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
)

_T = TypeVar("_T")


class _Result(Generic[_T]):
    """Small local equivalent of the Ray result used by FrankyController."""

    def __init__(self, get_value: Callable[[], _T]):
        self._get_value = get_value

    def wait(self) -> list[_T]:
        return [self._get_value()]


@dataclass
class Ros2ControllerConfig:
    left_robot_ip: str
    right_robot_ip: str
    left_gripper_type: str
    right_gripper_type: str
    left_gripper_connection: str
    right_gripper_connection: str
    ros_discovery_timeout: float = 2.0
    ros_wait_timeout: float = 30.0
    ros_state_max_age: float = 0.5
    controlled_motion_tolerance: float = 0.03
    controlled_motion_timeout: float = 180.0
    controlled_motion_stable_time: float = 0.5
    gripper_poll_interval: float = 0.1
    controller_health_period: float = 0.1
    controller_health_timeout: float = 0.5
    joint_names: list[str] = field(
        default_factory=lambda: [f"fr3_joint{i}" for i in range(1, 8)]
    )


class Ros2DualFrankaBackend:
    """Own two namespaced ROS 2 controllers and two serial grippers."""

    def __init__(self, config: Ros2ControllerConfig):
        self.config = config
        self._logger = get_logger()
        self._lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._states = {"left": None, "right": None}
        self._state_arrivals = {"left": 0.0, "right": 0.0}
        self._processes: dict[str, subprocess.Popen] = {}
        self._logs = {}
        self._grippers = {}
        self._gripper_positions = {"left": None, "right": None}
        self._gripper_open = {"left": True, "right": True}
        self._gripper_targets = {"left": None, "right": None}
        self._gripper_errors = {"left": None, "right": None}
        self._gripper_running = False
        self._gripper_threads = []
        self._controller_health_futures = {"left": None, "right": None}
        self._controller_health_requested = {"left": 0.0, "right": 0.0}
        self._controller_last_active = {"left": 0.0, "right": 0.0}
        self._controller_seen_active = {"left": False, "right": False}
        self._controller_health_errors = {"left": None, "right": None}
        self._started_at = time.monotonic()
        self._closed = False

        try:
            self._start_ros()
            for side in ("left", "right"):
                self._grippers[side] = create_gripper(
                    gripper_type=getattr(config, f"{side}_gripper_type"),
                    port=getattr(config, f"{side}_gripper_connection"),
                )
            for side, gripper in self._grippers.items():
                if not gripper.is_ready():
                    raise RuntimeError(f"{side} gripper did not become ready")
            self._start_gripper_threads()
            for side in ("left", "right"):
                self._start_controller(side, getattr(config, f"{side}_robot_ip"))
                while not self.is_robot_up(side):
                    time.sleep(0.05)
        except Exception:
            self.close()
            raise

    def _start_ros(self) -> None:
        try:
            import rclpy
            from controller_manager_msgs.srv import ListControllers
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import JointState
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "ROS 2 Franka collection requires the sourced Humble and "
                "franka_ros2 environments"
            ) from exc

        self._joint_state_type = JointState
        self._list_controllers_type = ListControllers
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            f"rlinf_dual_franka_{os.getpid()}", context=self._context
        )
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscriptions = []
        self._command_publishers = {}
        self._reset_publishers = {}
        self._controller_health_clients = {}
        for side in ("left", "right"):
            self._subscriptions.append(
                self._node.create_subscription(
                    JointState,
                    f"/{side}/franka/joint_states",
                    lambda msg, arm=side: self._on_state(arm, msg),
                    state_qos,
                )
            )
            self._command_publishers[side] = self._node.create_publisher(
                JointState, f"/{side}/rlinf/joint_targets", command_qos
            )
            self._reset_publishers[side] = self._node.create_publisher(
                JointState, f"/{side}/rlinf/reset_joint_target", command_qos
            )
            self._controller_health_clients[side] = self._node.create_client(
                ListControllers, f"/{side}/controller_manager/list_controllers"
            )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._ros_thread = threading.Thread(
            target=self._executor.spin,
            name="Ros2DualFrankaBackend",
            daemon=True,
        )
        self._ros_thread.start()

    def _start_controller(self, side: str, robot_ip: str) -> None:
        service = f"/{side}/controller_manager/list_controllers"
        if self._controller_health_clients[side].wait_for_service(
            timeout_sec=self.config.ros_discovery_timeout
        ):
            raise RuntimeError(f"controller_manager already active at {service}")

        repo_root = Path(__file__).resolve().parents[4]
        overlay = repo_root / "ros2_ws" / "install" / "rlinf_franka_controller"
        if not overlay.exists():
            raise RuntimeError(
                f"RLinf ROS 2 controller is not built under {repo_root / 'ros2_ws/install'}"
            )
        log_path = repo_root / "logs" / f"ros2_controller_{side}_{os.getpid()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        self._logs[side] = log_file
        process = subprocess.Popen(
            [
                "/usr/bin/setpriv",
                "--pdeathsig",
                "TERM",
                "--",
                "/opt/ros/humble/bin/ros2",
                "launch",
                "rlinf_franka_controller",
                "single_fr3.launch.py",
                f"robot_ip:={robot_ip}",
                f"namespace:={side}",
                f"arm_prefix:={side}",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "LD_LIBRARY_PATH": "/opt/openrobots/lib:"
                + os.environ.get("LD_LIBRARY_PATH", ""),
            },
            start_new_session=True,
        )
        self._processes[side] = process

    def _ordered_joints(self, joint_state, values: str) -> np.ndarray:
        by_name = dict(zip(joint_state.name, getattr(joint_state, values)))
        result = []
        for expected in self.config.joint_names:
            matches = [
                value for name, value in by_name.items() if name.endswith(expected)
            ]
            if len(matches) != 1:
                raise ValueError(f"missing or ambiguous ROS joint {expected}")
            result.append(matches[0])
        return np.asarray(result, dtype=np.float64)

    def _on_state(self, side: str, message) -> None:
        try:
            state = FrankaRobotState(
                arm_joint_position=self._ordered_joints(message, "position"),
                arm_joint_velocity=self._ordered_joints(message, "velocity"),
            )
        except (AttributeError, ValueError) as exc:
            self._logger.warning("Ignoring invalid %s Franka state: %s", side, exc)
            return
        with self._lock:
            self._states[side] = state
            self._state_arrivals[side] = time.monotonic()

    def _check_process(self, side: str) -> None:
        self._poll_controller_health(side)
        process = self._processes.get(side)
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"{side} ROS 2 controller exited with status {process.returncode}"
            )
        if self._gripper_errors[side] is not None:
            raise RuntimeError(
                f"{side} gripper worker failed"
            ) from self._gripper_errors[side]
        if self._controller_health_errors[side] is not None:
            raise RuntimeError(f"{side} ROS 2 controller is not active") from (
                self._controller_health_errors[side]
            )

    def _poll_controller_health(self, side: str) -> None:
        now = time.monotonic()
        with self._health_lock:
            future = self._controller_health_futures[side]
            if future is not None and future.done():
                self._controller_health_futures[side] = None
                try:
                    response = future.result()
                    state = next(
                        (
                            controller.state
                            for controller in response.controller
                            if controller.name == "joint_impedance_controller"
                        ),
                        None,
                    )
                    if state == "active":
                        self._controller_seen_active[side] = True
                        self._controller_last_active[side] = now
                        self._controller_health_errors[side] = None
                    elif self._controller_seen_active[side]:
                        raise RuntimeError(f"controller state is {state!r}")
                except Exception as exc:
                    self._controller_health_errors[side] = exc

            if (
                self._controller_health_futures[side] is None
                and now - self._controller_health_requested[side]
                >= self.config.controller_health_period
                and self._controller_health_clients[side].service_is_ready()
            ):
                self._controller_health_futures[side] = self._controller_health_clients[
                    side
                ].call_async(self._list_controllers_type.Request())
                self._controller_health_requested[side] = now

            future = self._controller_health_futures[side]
            if future is not None and (
                now - self._controller_health_requested[side]
                > self.config.controller_health_timeout
            ):
                self._controller_health_errors[side] = TimeoutError(
                    "controller health response timed out"
                )

    def is_robot_up(self, side: str) -> bool:
        self._check_process(side)
        with self._lock:
            has_state = self._states[side] is not None
            has_gripper = self._gripper_positions[side] is not None
        has_controller = self._controller_seen_active[side]
        if not (has_state and has_gripper and has_controller) and (
            time.monotonic() - self._started_at > self.config.ros_wait_timeout
        ):
            raise TimeoutError(
                f"timed out waiting for {side} Franka, gripper, and controller"
            )
        return has_state and has_gripper and has_controller

    def get_state(self, side: str) -> FrankaRobotState:
        self._check_process(side)
        now = time.monotonic()
        with self._lock:
            state = self._states[side]
            age = now - self._state_arrivals[side]
            gripper_position = self._gripper_positions[side]
            gripper_open = self._gripper_open[side]
        if state is None:
            raise RuntimeError(f"{side} ROS 2 Franka state is not initialized")
        if age > self.config.ros_state_max_age:
            raise RuntimeError(f"{side} ROS 2 Franka state is stale ({age:.3f}s)")
        if gripper_position is None:
            raise RuntimeError(f"{side} gripper state is not initialized")
        result = FrankaRobotState(
            arm_joint_position=state.arm_joint_position.copy(),
            arm_joint_velocity=state.arm_joint_velocity.copy(),
            gripper_position=gripper_position,
            gripper_open=gripper_open,
        )
        return result

    def _joint_positions(self, side: str) -> np.ndarray:
        self._check_process(side)
        now = time.monotonic()
        with self._lock:
            state = self._states[side]
            age = now - self._state_arrivals[side]
            positions = None if state is None else state.arm_joint_position.copy()
        if positions is None:
            raise RuntimeError(f"{side} ROS 2 Franka state is not initialized")
        if age > self.config.ros_state_max_age:
            raise RuntimeError(f"{side} ROS 2 Franka state is stale ({age:.3f}s)")
        return positions

    def _wait_for_subscriber(self, side: str, publisher) -> None:
        deadline = time.monotonic() + self.config.ros_wait_timeout
        while publisher.get_subscription_count() == 0:
            self._check_process(side)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no ROS 2 controller subscriber for {side}")
            time.sleep(0.05)

    def _publish(self, side: str, target: np.ndarray, reset: bool) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"{side} joint target must contain seven finite values")
        if np.any(target < FR3_JOINT_LIMITS_LOWER) or np.any(
            target > FR3_JOINT_LIMITS_UPPER
        ):
            raise ValueError(f"{side} joint target is outside FR3 joint limits")
        publisher = (
            self._reset_publishers[side] if reset else self._command_publishers[side]
        )
        self._wait_for_subscriber(side, publisher)
        message = self._joint_state_type()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = self.config.joint_names
        message.position = target.tolist()
        publisher.publish(message)
        return target

    def reset_joint(self, side: str, target: np.ndarray) -> _Result[None]:
        target = self._publish(side, target, reset=True)
        return _Result(lambda: self._wait_for_target(side, target))

    def move_joints(self, side: str, target: np.ndarray) -> _Result[None]:
        self._publish(side, target, reset=False)
        return _Result(lambda: None)

    def _wait_for_target(self, side: str, target: np.ndarray) -> None:
        deadline = time.monotonic() + self.config.controlled_motion_timeout
        reached_since = None
        while time.monotonic() < deadline:
            current = self._joint_positions(side)
            error = float(np.max(np.abs(current - target)))
            if error <= self.config.controlled_motion_tolerance:
                reached_since = reached_since or time.monotonic()
                if (
                    time.monotonic() - reached_since
                    >= self.config.controlled_motion_stable_time
                ):
                    return
            else:
                reached_since = None
            time.sleep(0.02)
        raise TimeoutError(f"{side} controlled motion did not reach target")

    def clear_errors(self, side: str) -> None:
        self._check_process(side)

    def open_gripper(self, side: str) -> None:
        with self._lock:
            self._gripper_targets[side] = True

    def close_gripper(self, side: str) -> None:
        with self._lock:
            self._gripper_targets[side] = False

    def _start_gripper_threads(self) -> None:
        self._gripper_running = True
        for side in ("left", "right"):
            thread = threading.Thread(
                target=self._gripper_loop,
                args=(side,),
                name=f"Ros2FrankaGripper-{side}",
                daemon=True,
            )
            thread.start()
            self._gripper_threads.append(thread)

    def _gripper_loop(self, side: str) -> None:
        gripper = self._grippers[side]
        applied_target = None
        while self._gripper_running:
            started = time.monotonic()
            try:
                with self._lock:
                    target = self._gripper_targets[side]
                if target is not None and target != applied_target:
                    if target:
                        gripper.open(speed=1.0)
                    else:
                        gripper.close(speed=1.0)
                    applied_target = target
                position = gripper.position
                with self._lock:
                    self._gripper_positions[side] = position
                    self._gripper_open[side] = gripper.is_open
            except Exception as exc:
                with self._lock:
                    self._gripper_errors[side] = exc
                return
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, self.config.gripper_poll_interval - elapsed))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process in self._processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in self._processes.values():
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2.0)
        for log_file in self._logs.values():
            log_file.close()
        self._gripper_running = False
        for thread in self._gripper_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        for gripper in self._grippers.values():
            try:
                gripper.cleanup()
            except Exception as exc:
                self._logger.warning("Failed to close gripper: %s", exc)
        if hasattr(self, "_executor"):
            self._executor.shutdown(timeout_sec=1.0)
            self._node.destroy_node()
            self._context.shutdown()
            if self._ros_thread.is_alive():
                self._ros_thread.join(timeout=1.0)


class Ros2FrankaControllerProxy:
    """Per-arm API expected by the main DualFrankaEnv and GELLO wrapper."""

    def __init__(self, backend: Ros2DualFrankaBackend, side: str):
        self._backend = backend
        self._side = side

    def is_robot_up(self) -> _Result[bool]:
        return _Result(lambda: self._backend.is_robot_up(self._side))

    def get_state(self) -> _Result[FrankaRobotState]:
        return _Result(lambda: self._backend.get_state(self._side))

    def reset_joint(self, target) -> _Result[None]:
        return self._backend.reset_joint(self._side, target)

    def move_joints(self, target) -> _Result[None]:
        return self._backend.move_joints(self._side, target)

    def clear_errors(self) -> _Result[None]:
        return _Result(lambda: self._backend.clear_errors(self._side))

    def open_gripper(self) -> None:
        self._backend.open_gripper(self._side)

    def close_gripper(self) -> None:
        self._backend.close_gripper(self._side)
