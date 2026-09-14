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

"""Dual-arm Franka env driven through ``FrankyArm`` (libfranka)."""

from __future__ import annotations

import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
from itertools import cycle
from typing import Any, Callable, Optional, TypeVar

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import (
    Camera,
    DualFrankaConfig,
    DualFrankaRobot,
    Robot,
    RobotInfo,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.base import CartesianCompliance
from rlinf.robotics.parts.arms.franka import FrankaRobotState
from rlinf.robotics.parts.arms.franky import FrankyArm
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.robotics.parts.end_effectors import EndEffector
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

# Keep frame reads shorter than the 10 Hz control period.
_CAMERA_FRAME_TIMEOUT_S = 0.5

_ArmResult = TypeVar("_ArmResult")


@dataclass
class DualFrankaEnvConfig:
    """Configuration for the dual-arm Franka environment."""

    enable_camera_player: bool = False
    # Whether the cameras also capture depth, alongside every colour frame.
    enable_camera_depth: bool = False
    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0

    # Two-row pose arrays store the left arm first and right arm second.
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    joint_reset_qpos: list[list[float]] = field(
        default_factory=lambda: [[0, 0, 0, -1.9, 0, 2, 0]] * 2
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    action_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    enable_random_reset: bool = False
    random_xy_range: float = 0.0
    random_rz_range: float = 0.0

    # Per-arm Cartesian safety limits.
    ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), -np.inf)
    )
    ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), np.inf)
    )

    compliance_param: dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    joint_reset_cycle: int = 20000
    task_description: str = ""
    success_hold_steps: int = 1

    def __post_init__(self) -> None:
        self.target_ee_pose = np.array(self.target_ee_pose).reshape(2, 6)
        self.reset_ee_pose = np.array(self.reset_ee_pose).reshape(2, 6)
        self.reward_threshold = np.array(self.reward_threshold).reshape(2, 6)
        self.action_scale = np.array(self.action_scale)
        self.ee_pose_limit_min = np.array(self.ee_pose_limit_min).reshape(2, 6)
        self.ee_pose_limit_max = np.array(self.ee_pose_limit_max).reshape(2, 6)


class DualFrankaEnv(gym.Env):
    """Dual-arm Franka env driven through ``FrankyArm`` (libfranka).

    Abstract base. Subclasses set ``PER_ARM_ACTION_DIM`` / ``GRIPPER_IDX_IN_ARM``
    and implement ``_init_action_obs_spaces`` + ``_get_observation`` +
    ``_dispatch_arm_motion``.
    """

    #: Teleoperation devices compatible with the dual-arm action layouts.
    TELEOP = ("gello_joint", "pico")
    TELEOP_DEFAULT = "none"
    TELEOP_MARK_FLAG = True

    #: Reject the single-arm gripper-removal flag for dual-arm actions.
    ACTION_WRAPPERS = ()
    REFUSE_FLAGS = ("no_gripper",)
    REFUSE_DEFAULTS = {"no_gripper": True}

    TRANSFORMS = ()

    CONFIG_CLS: type[DualFrankaEnvConfig] = DualFrankaEnvConfig
    PER_ARM_ACTION_DIM: int = 0
    GRIPPER_IDX_IN_ARM: int = 0

    def arm_action_kind(self) -> ActionKind:
        """Return the semantic type of each arm action."""
        raise NotImplementedError

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return mirrored arm and end-effector action parts."""
        from rlinf.envs.real.wrappers.teleop.layout import mirrored

        gripper_at = self.GRIPPER_IDX_IN_ARM
        return mirrored(
            (
                ActionPart("arm", gripper_at, self.arm_action_kind()),
                ActionPart(
                    "end_effector",
                    self.PER_ARM_ACTION_DIM - gripper_at,
                    ActionKind.GRIPPER,
                ),
            ),
            ("left", "right"),
        )

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Optional[RobotInfo[DualFrankaConfig]] = None,
        env_idx: int = 0,
    ) -> None:
        config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self.config = config
        self.hardware = get_hardware_config(
            DualFrankaConfig, robot_info, is_dummy=config.is_dummy
        )
        self._task_description = config.task_description
        self.robot_info = robot_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._left_state = FrankaRobotState()
        self._right_state = FrankaRobotState()

        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)
        self._success_hold_counter = 0
        self.robot: Robot | None = None

        if not self.config.is_dummy:
            self._setup_hardware()
            # Each arm keeps its own serial command queue
            self._arm_executors = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="franka-left"),
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="franka-right"),
            )

        all_serials = self._all_camera_serials()
        assert len(all_serials) > 0, (
            "At least one camera serial must be provided for DualFrankaEnv."
        )
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for an initial valid state from each arm.
        for label, ctrl in [("left", self._left_arm), ("right", self._right_arm)]:
            t0 = time.time()
            while not ctrl.is_robot_up():
                time.sleep(0.5)
                if time.time() - t0 > 30:
                    self._logger.warning(
                        "Waited %.0fs for %s Franka to be ready.",
                        time.time() - t0,
                        label,
                    )

        self._left_state, self._right_state = self._run_arm_calls(
            self._left_arm.get_state,
            self._right_arm.get_state,
        )

        # Retain the latest valid reading while an individual camera recovers.
        self._last_camera_frame: dict[str, dict[str, np.ndarray]] = {}

        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    @property
    def task_description(self) -> str:
        return self._task_description

    def close(self) -> None:
        if hasattr(self, "_cameras"):
            self._close_cameras()
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if hasattr(self, "_arm_executors"):
            for executor in self._arm_executors:
                executor.shutdown(wait=True)
        if self.robot is not None:
            self.robot.disconnect()

    # Camera handling.

    def _all_camera_specs(self) -> list[tuple[str, str, str]]:
        """Return named camera specifications in policy-compatible order.

        Per-slot ``*_camera_type`` falls back to the global ``camera_type``.
        """
        default_ct = self.hardware.camera_type or "realsense"
        specs: list[tuple[str, str, str]] = []
        if self.hardware.base_camera_serials:
            ct = self.hardware.base_camera_type or default_ct
            for j, serial in enumerate(self.hardware.base_camera_serials):
                specs.append((f"base_{j}_rgb", serial, ct))
        for arm, serials, slot_ct in (
            (
                "left",
                self.hardware.left_camera_serials,
                self.hardware.left_camera_type,
            ),
            (
                "right",
                self.hardware.right_camera_serials,
                self.hardware.right_camera_type,
            ),
        ):
            if not serials:
                continue
            ct = slot_ct or default_ct
            for j, serial in enumerate(serials):
                specs.append((f"{arm}_wrist_{j}_rgb", serial, ct))
        return specs

    def _all_camera_serials(self) -> list[str]:
        return [serial for _, serial, _ in self._all_camera_specs()]

    def _camera_infos(self) -> list[CameraInfo]:
        """Return declarations for all wrist and base cameras."""
        return [
            CameraInfo(
                name=name,
                serial_number=serial,
                camera_type=ct,
                enable_depth=self.config.enable_camera_depth,
            )
            for name, serial, ct in self._all_camera_specs()
        ]

    def _camera_declarations(self) -> tuple[dict[str, dict], dict]:
        """Separate wrist cameras from robot-level cameras."""
        per_arm: dict[str, dict] = {"left": {}, "right": {}}
        robot_level: dict = {}
        for info in self._camera_infos():
            if info.name.startswith("left_wrist_"):
                per_arm["left"][info.name] = info
            elif info.name.startswith("right_wrist_"):
                per_arm["right"][info.name] = info
            else:
                robot_level[info.name] = info
        return per_arm, robot_level

    def _open_cameras(self) -> None:
        """Use cameras connected and placed by the robot runtime.

        Camera paths are reduced to their declared leaf names because the
        observation space already encodes the arm side.
        """
        self._cameras: dict[str, BaseCamera] = {}
        for path, camera in self.robot.parts_of_type(Camera).items():
            declared = path.rsplit(".", 1)[-1]
            assert declared not in self._cameras, (
                f"Two cameras on this robot are both called {declared!r} "
                f"({path} collides). Camera names come from the env's own "
                "declarations, which have to be unique across the robot."
            )
            self._cameras[declared] = camera

    def _close_cameras(self) -> None:
        """Drop the camera references; the robot closes what it opened."""
        self._cameras = {}

    def _crop_frame(
        self,
        frame: np.ndarray,
        reshape_size: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
        resized = cv2.resize(cropped, reshape_size, interpolation=interpolation)
        return cropped, resized

    def _get_camera_observation(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read all cameras and use the latest cached frame during recovery.

        A camera that stalls before producing its first frame raises an error.
        """
        frames: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        display_frames: dict[str, np.ndarray] = {}

        for name, camera in self._cameras.items():
            try:
                reading = camera.get_observation(timeout=_CAMERA_FRAME_TIMEOUT_S)
            except queue.Empty:
                # get_frame has already reopened it; keep the loop at 10 Hz by
                # reusing the last good frame while it comes back.
                cached = self._last_camera_frame.get(name)
                if cached is None:
                    raise RuntimeError(
                        f"Camera {name} stalled with no cached frame to fall back to."
                    )
                self._logger.error("Camera %s stalled; using the last frame.", name)
                reading = cached

            reshape_size = self.observation_space["frames"][name].shape[:2][::-1]
            cropped, resized = self._crop_frame(reading["frame"], reshape_size)
            frames[name] = resized[..., ::-1]
            display_frames[name] = resized
            display_frames[f"{name}_full"] = cropped
            if "depth" in reading:
                # Averaging a depth map invents distances between an object and
                # whatever is behind it, so this resamples by nearest instead.
                _, depths[name] = self._crop_frame(
                    reading["depth"], reshape_size, interpolation=cv2.INTER_NEAREST
                )
            self._last_camera_frame[name] = reading

        self.camera_player.put_frame(display_frames)
        return frames, depths

    # Hardware setup.

    def _resolve_controller_node_ranks(self) -> tuple[int, int]:
        """Return controller node ranks with hardware overrides applied."""
        hardware = self.hardware
        left_node = hardware.left_controller_node_rank
        right_node = hardware.right_controller_node_rank
        return (
            self.node_rank if left_node is None else left_node,
            self.node_rank if right_node is None else right_node,
        )

    def _side_compliance(self, side: str) -> Optional[CartesianCompliance]:
        """Return one arm's impedance settings, or the shared ones."""
        hardware = self.hardware
        return getattr(hardware, f"{side}_compliance") or hardware.compliance

    def _setup_hardware(self) -> None:
        assert self.env_idx >= 0, f"env_idx must be set for {type(self).__name__}."

        left_node, right_node = self._resolve_controller_node_ranks()

        arm_cameras, base_cameras = self._camera_declarations()
        self.robot = DualFrankaRobot.build(
            left_robot_ip=self.hardware.left_robot_ip,
            right_robot_ip=self.hardware.right_robot_ip,
            env_idx=self.env_idx,
            left_node_rank=left_node,
            right_node_rank=right_node,
            worker_rank=self.env_worker_rank,
            left_gripper_type=self.hardware.left_gripper_type,
            right_gripper_type=self.hardware.right_gripper_type,
            left_gripper_connection=self.hardware.left_gripper_connection,
            right_gripper_connection=self.hardware.right_gripper_connection,
            left_compliance=self._side_compliance("left"),
            right_compliance=self._side_compliance("right"),
            arm_cameras=arm_cameras,
            cameras=base_cameras,
        )
        self.robot.connect()
        # Naming the class each part is expected to be keeps the driver's own
        # methods resolvable from here, and reports a mismatched composition at
        # this line rather than as a missing attribute mid-episode.
        self._left_arm: FrankyArm = self.robot.child("left").child("arm", FrankyArm)
        self._right_arm: FrankyArm = self.robot.child("right").child("arm", FrankyArm)
        # Each hand is a part beside its arm, with its own connection.
        self._left_hand: EndEffector = self.robot.child("left").child(
            "end_effector", EndEffector
        )
        self._right_hand: EndEffector = self.robot.child("right").child(
            "end_effector", EndEffector
        )

    # Gymnasium reset and step.

    def _run_arm_calls(
        self,
        left: Callable[[], _ArmResult],
        right: Callable[[], _ArmResult],
    ) -> tuple[_ArmResult, _ArmResult]:
        """Run one ordered command on each arm and wait for both results."""
        futures = (
            self._arm_executors[0].submit(left),
            self._arm_executors[1].submit(right),
        )
        wait(futures)
        return futures[0].result(), futures[1].result()

    def _submit_arm_calls(
        self,
        left: Callable[[], Any],
        right: Callable[[], Any],
    ) -> tuple[Future[Any], Future[Any]]:
        """Queue one command per arm without delaying the control loop."""
        futures = (
            self._submit_arm_call(0, left),
            self._submit_arm_call(1, right),
        )
        return futures

    def _submit_arm_call(
        self, arm: int, call: Callable[[], _ArmResult]
    ) -> Future[_ArmResult]:
        """Queue one arm command and report failures without waiting."""
        future = self._arm_executors[arm].submit(call)
        side = "left" if arm == 0 else "right"
        future.add_done_callback(
            lambda completed: self._log_background_failure(side, completed)
        )
        return future

    def _log_background_failure(self, side: str, future: Future[Any]) -> None:
        """Report an asynchronous arm command that failed."""
        try:
            future.result()
        except Exception as exc:
            self._logger.warning("%s Franka command failed: %s", side, exc)

    def _go_to_rest(self, joint_reset: bool = False) -> None:
        del joint_reset
        self._submit_arm_calls(
            self._left_hand.open,
            self._right_hand.open,
        )
        self._submit_arm_calls(
            lambda: self._left_arm.reset_joint(self.config.joint_reset_qpos[0]),
            lambda: self._right_arm.reset_joint(self.config.joint_reset_qpos[1]),
        )
        time.sleep(0.5)
        # State reads follow the reset on each arm's queue and therefore wait
        # for it, while the fixed settle delay overlaps the reset as before.
        self._left_state, self._right_state = self._run_arm_calls(
            self._left_arm.get_state,
            self._right_arm.get_state,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset both arms unless teleoperation requests pose continuity."""
        # A run with no hardware samples this space instead of reading,
        # so seeding it is what makes such a run reproducible.
        seed_sampled_spaces(seed, self.observation_space)
        del seed
        skip_reset_to_home = bool((options or {}).get("skip_reset_to_home", False))
        self._num_steps = 0
        self._success_hold_counter = 0

        if self.config.is_dummy:
            return self._get_observation(), {}

        self._reconfigure_compliance()

        joint_cycle = next(self._joint_reset_cycle)
        joint_reset = joint_cycle == 0
        if joint_reset:
            self._logger.info(
                "Number of resets reached %d, resetting joints.",
                self.config.joint_reset_cycle,
            )

        if skip_reset_to_home:
            self._logger.info(
                "skip_reset_to_home=True: holding arms at episode-end pose "
                "(teleop wrapper will realign to device)."
            )
        else:
            self._go_to_rest(joint_reset)
        self._clear_errors()

        self._left_state, self._right_state = self._run_arm_calls(
            self._left_arm.get_state,
            self._right_arm.get_state,
        )
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = action.reshape(2, self.PER_ARM_ACTION_DIM)

        is_gripper_effective = [True, True]

        if not self.config.is_dummy:
            states = [self._left_state, self._right_state]
            ctrls = [self._left_arm, self._right_arm]
            hands = [self._left_hand, self._right_hand]
            dt = 1.0 / self.config.step_frequency

            # Queue gripper commands before the next motion on each arm. The
            # command is intentionally not awaited in the 10 Hz control loop.
            for arm in range(2):
                gripper_val = (
                    actions[arm, self.GRIPPER_IDX_IN_ARM] * self.config.action_scale[2]
                )
                is_gripper_effective[arm] = self._gripper_action(
                    arm, hands[arm], gripper_val
                )

            self._dispatch_arm_motion(actions, states, ctrls, dt)

        self._num_steps += 1
        if not self.config.is_dummy:
            if self._pace_between_action_and_state_read():
                step_time = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
            self._left_state, self._right_state = self._run_arm_calls(
                self._left_arm.get_state,
                self._right_arm.get_state,
            )

        observation = self._get_observation()
        reward = self._calc_step_reward(is_gripper_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    def _clear_errors(self) -> None:
        self._run_arm_calls(
            self._left_arm.clear_errors,
            self._right_arm.clear_errors,
        )

    def _reconfigure_compliance(self) -> None:
        params = self.config.compliance_param
        self._run_arm_calls(
            partial(self._left_arm.reconfigure_compliance_params, params),
            partial(self._right_arm.reconfigure_compliance_params, params),
        )

    # Gripper and state helpers.

    def _gripper_action(self, arm: int, hand: Any, position: float) -> bool:
        threshold = self.config.binary_gripper_threshold
        if position <= -threshold and hand.is_open:
            self._submit_arm_call(arm, hand.close)
            return True
        elif position >= threshold and not hand.is_open:
            self._submit_arm_call(arm, hand.open)
            return True
        return False

    def get_tcp_pose(self) -> np.ndarray:
        """Return concatenated TCP poses ``(14,)`` for both arms."""
        self._left_state, self._right_state = self._run_arm_calls(
            self._left_arm.get_state,
            self._right_arm.get_state,
        )
        return np.concatenate([self._left_state.tcp_pose, self._right_state.tcp_pose])

    def get_action_scale(self) -> np.ndarray:
        """Return the action scaling factors used by teleop wrappers."""
        return self.config.action_scale

    def get_joint_positions(self) -> np.ndarray:
        """Return cached joint positions for both arms with shape ``(2, 7)``."""
        return np.stack(
            [
                self._left_state.arm_joint_position.copy(),
                self._right_state.arm_joint_position.copy(),
            ]
        )

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def target_ee_pose(self) -> np.ndarray:
        """Return concatenated target poses ``(14,)`` in quaternion form."""
        poses = []
        for arm in range(2):
            euler = self.config.target_ee_pose[arm]
            poses.append(
                np.concatenate(
                    [
                        euler[:3],
                        R.from_euler("xyz", euler[3:].copy()).as_quat(),
                    ]
                )
            )
        return np.concatenate(poses)

    def _cartesian_safety_boxes(self) -> None:
        self._xyz_safe_spaces = []
        self._rpy_safe_spaces = []
        for arm in range(2):
            self._xyz_safe_spaces.append(
                gym.spaces.Box(
                    low=self.config.ee_pose_limit_min[arm, :3],
                    high=self.config.ee_pose_limit_max[arm, :3],
                    dtype=np.float64,
                )
            )
            self._rpy_safe_spaces.append(
                gym.spaces.Box(
                    low=self.config.ee_pose_limit_min[arm, 3:],
                    high=self.config.ee_pose_limit_max[arm, 3:],
                    dtype=np.float64,
                )
            )

    def _build_camera_spaces(self) -> dict[str, gym.spaces.Dict]:
        """Return the camera part of the observation space.

        ``depths`` appears only where a camera captures depth, so a rig
        without one keeps the schema its policy already reads.
        """
        camera_infos = self._camera_infos()
        spaces: dict[str, gym.spaces.Dict] = {
            "frames": gym.spaces.Dict(
                {
                    info.name: gym.spaces.Box(
                        0, 255, shape=(224, 224, 3), dtype=np.uint8
                    )
                    for info in camera_infos
                }
            )
        }
        depth_cameras = [info for info in camera_infos if info.enable_depth]
        if depth_cameras:
            spaces["depths"] = gym.spaces.Dict(
                {
                    info.name: gym.spaces.Box(
                        0.0, np.inf, shape=(224, 224), dtype=np.float32
                    )
                    for info in depth_cameras
                }
            )
        return spaces

    def _build_observation_space(self, joint_position_dim: int) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(2 * 7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(2 * 6,)),
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(joint_position_dim,)
                        ),
                        "joint_velocity": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2 * 7,)
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(2,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(2 * 3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(2 * 3,)),
                    }
                ),
            }
            | self._build_camera_spaces()
        )

    # Reward calculation.

    def _calc_step_reward(self, is_gripper_effective: list[bool]) -> float:
        if self.config.is_dummy:
            return 0.0

        all_in_zone = True
        dense_sq_sum = 0.0
        for arm, state in enumerate([self._left_state, self._right_state]):
            euler = np.abs(R.from_quat(state.tcp_pose[3:].copy()).as_euler("xyz"))
            position = np.hstack([state.tcp_pose[:3], euler])
            delta = np.abs(position - self.config.target_ee_pose[arm])
            if not np.all(delta[:3] <= self.config.reward_threshold[arm, :3]):
                all_in_zone = False
                dense_sq_sum += np.sum(np.square(delta[:3]))

        if all_in_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(np.exp(-500 * dense_sq_sum))
            else:
                reward = 0.0

        if self.config.enable_gripper_penalty:
            for eff in is_gripper_effective:
                if eff:
                    reward -= self.config.gripper_penalty
        return reward

    # Subclass hooks.

    def _init_action_obs_spaces(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_action_obs_spaces"
        )

    def _get_observation(self) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_observation"
        )

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        """Dispatch arm motion using the subclass control representation."""
        del actions, states, ctrls, dt

    def _pace_between_action_and_state_read(self) -> bool:
        return True
