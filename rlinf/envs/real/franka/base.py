# Copyright 2025 The RLinf Authors.
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

import copy
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import (
    Camera,
    FrankaConfig,
    FrankaRobot,
    Robot,
    RobotInfo,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.base import BaseArm
from rlinf.robotics.parts.arms.franka import FrankaRobotState
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.robotics.parts.end_effectors import BaseEndEffector
from rlinf.robotics.parts.end_effectors.base import (
    EndEffectorType,
    normalize_end_effector_type,
)
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from ..utils.pose import (
    clip_euler_to_target_window,
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
    quat_slerp,
)

#: Default Cartesian impedance gains shared by Franka tasks.
COMPLIANCE_DEFAULTS: dict[str, float] = {
    "translational_stiffness": 1000,
    "translational_damping": 89,
    "rotational_stiffness": 150,
    "rotational_damping": 7,
    "translational_Ki": 0,
    "rotational_Ki": 0,
    "translational_clip_x": 0.003,
    "translational_clip_y": 0.003,
    "translational_clip_z": 0.01,
    "translational_clip_neg_x": 0.003,
    "translational_clip_neg_y": 0.003,
    "translational_clip_neg_z": 0.01,
    "rotational_clip_x": 0.02,
    "rotational_clip_y": 0.02,
    "rotational_clip_z": 0.02,
    "rotational_clip_neg_x": 0.02,
    "rotational_clip_neg_y": 0.02,
    "rotational_clip_neg_z": 0.02,
}


def compliance(**overrides: float) -> dict[str, float]:
    """Return :data:`COMPLIANCE_DEFAULTS` with ``overrides`` applied.

    Raises:
        KeyError: If an override is not a supported controller gain.
    """
    unknown = set(overrides) - set(COMPLIANCE_DEFAULTS)
    if unknown:
        raise KeyError(
            f"Unknown compliance gains {sorted(unknown)}. "
            f"Known: {sorted(COMPLIANCE_DEFAULTS)}."
        )
    return {**COMPLIANCE_DEFAULTS, **overrides}


#: Maximum reopen attempts for a stalled camera.
_CAMERA_REOPEN_ATTEMPTS = 3

#: Seconds to wait before reopening a stalled camera.
_CAMERA_REOPEN_WAIT_S = 5.0


@dataclass
class FrankaRobotConfig:
    robot_ip: Optional[str] = None
    #: Arm backend this robot runs, such as ``"franka_ros"`` or ``"franky"``.
    #: ``None`` leaves the choice to the robot's own default.
    backend: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    camera_names: Optional[dict[str, str]] = None
    camera_type: Optional[str] = None
    gripper_type: Optional[str] = None
    gripper_connection: Optional[str] = None
    enable_camera_player: bool = True
    # Per-camera [top, left, bottom, right] crop fractions, keyed by serial.
    camera_crop_regions: Optional[dict[str, list[float]]] = None

    is_dummy: bool = False
    use_dense_reward: bool = False
    reward_scale: float = 1.0  # Scale applied to the dense reward.
    step_frequency: float = 10.0  # Maximum environment steps per second.

    use_reward_model: bool = False
    reward_worker_cfg: Optional[dict] = None
    reward_worker_hardware_rank: Optional[int] = None
    reward_worker_node_rank: Optional[int] = None
    reward_worker_node_group: Optional[str] = None
    reward_image_key: Optional[str] = None

    # Poses use xyz position and xyz Euler orientation; the env converts to quaternions.
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0, 0, 0, -1.9, -0, 2, 0]
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # Translation, orientation, and gripper scales.
    enable_random_reset: bool = False

    random_xy_range: float = 0.0
    random_rz_range: float = 0.0  # Maximum yaw perturbation, in radians.

    # Cartesian limits use xyz position followed by xyz Euler orientation.
    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(6))
    compliance_param: dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    joint_reset_cycle: int = 20000  # Episode resets between full joint resets.
    task_description: str = ""
    success_hold_steps: int = 1  # Consecutive successful steps required.

    # End-effector selection and control parameters.
    end_effector_type: str = "franka_gripper"
    # Extra kwargs forwarded to the end-effector constructor.
    end_effector_config: dict = field(default_factory=dict)
    # Target hand pose used for dense-reward success criteria (6-D).
    hand_target_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Default hand pose after ``reset()`` (6-D).
    hand_reset_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Hand action scale (for continuous hand control).
    hand_action_scale: float = 1.0
    # Max per-step change for hand joints (set to inf to disable).
    hand_max_delta_per_step: float = float("inf")

    def __post_init__(self) -> None:
        """Convert list fields from YAML/Hydra to numpy arrays."""
        if self.camera_names is not None:
            self.camera_names = {
                str(serial): str(camera_name)
                for serial, camera_name in self.camera_names.items()
            }
        if self.camera_crop_regions is not None:
            self.camera_crop_regions = {
                str(serial): crop_region
                for serial, crop_region in self.camera_crop_regions.items()
            }
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = np.array(self.reset_ee_pose)
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array(self.action_scale)
        self.ee_pose_limit_min = np.array(self.ee_pose_limit_min)
        self.ee_pose_limit_max = np.array(self.ee_pose_limit_max)
        self.hand_target_state = np.array(self.hand_target_state)
        self.hand_reset_state = np.array(self.hand_reset_state)


class FrankaEnv(gym.Env):
    """Franka robot arm environment."""

    #: Supported teleoperation devices and the default selection.
    TELEOP = ("spacemouse", "gello", "glove", "pico")
    TELEOP_DEFAULT = "spacemouse"

    #: Action wrappers applied before teleoperation.
    ACTION_WRAPPERS = ("GripperCloseEnv",)

    #: Representation transforms applied after episode wrappers.
    TRANSFORMS = ("RelativeFrame", "Quat2EulerWrapper")

    CONFIG_CLS: type[FrankaRobotConfig] = FrankaRobotConfig

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo],
        robot_info: Optional[RobotInfo[FrankaConfig]],
        env_idx: int,
    ) -> None:
        config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self.config = config
        self.config.end_effector_type = normalize_end_effector_type(
            self.config.end_effector_type,
            self.config.gripper_type,
        ).value
        self._task_description = config.task_description
        self.robot_info = robot_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._franka_state = FrankaRobotState()
        #: Cropped frames from the most recent whole-robot reading.
        self._last_frames: dict[str, np.ndarray] = {}
        if not self.config.is_dummy:
            self._reset_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:].copy()).as_quat(),
                ]
            ).copy()
        else:
            self._reset_pose = np.zeros(7)
        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)  # Start the first cycle after zero.

        self._success_hold_counter = 0
        self._last_hand_command: np.ndarray | None = None
        self._reward_worker = None
        self.robot: Robot | None = None

        if not self.config.is_dummy:
            self._setup_hardware()
            self._setup_reward_worker()
        else:
            self._camera_infos = self._build_camera_infos()

        # Initialize spaces after camera declarations are available.
        assert self._camera_infos, (
            "At least one camera serial must be provided for FrankaEnv."
        )
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the controller's first valid state.
        start_time = time.time()
        while not self._arm.is_robot_up():
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
                )

        # Take the cameras before the first whole-robot read, which carries
        # their frames along with the arm state.
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

        self._interpolate_move(self._reset_pose)
        time.sleep(1.0)
        self._franka_state = self._read_robot()

    @property
    def task_description(self) -> str:
        return self._task_description

    def _setup_hardware(self) -> None:
        assert self.env_idx >= 0, "env_idx must be set for FrankaEnv."

        # Fill unset connection fields from enumerated hardware configuration.
        assert isinstance(self.robot_info, RobotInfo) and isinstance(
            self.robot_info.config, FrankaConfig
        ), f"robot_info must contain a FrankaConfig, but got {type(self.robot_info)}."
        if self.config.robot_ip is None:
            self.config.robot_ip = self.robot_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.robot_info.config.camera_serials
        if self.config.camera_type is None:
            self.config.camera_type = getattr(
                self.robot_info.config, "camera_type", "realsense"
            )
        if self.config.backend is None:
            self.config.backend = getattr(self.robot_info.config, "backend", None)
        if self.config.gripper_type is None:
            self.config.gripper_type = getattr(
                self.robot_info.config, "gripper_type", "franka"
            )
        if self.config.gripper_connection is None:
            self.config.gripper_connection = getattr(
                self.robot_info.config, "gripper_connection", None
            )
        self.config.end_effector_type = normalize_end_effector_type(
            self.config.end_effector_type,
            self.config.gripper_type,
        ).value
        # Hardware discovery may be the source of the camera serials. Build the
        # declarations only after those values have been applied.
        self._camera_infos = self._build_camera_infos()

        # Default the arm controller to the environment worker's node.
        controller_node_rank = getattr(
            self.robot_info.config, "controller_node_rank", None
        )
        if controller_node_rank is None:
            controller_node_rank = self.node_rank
        # The composed robot owns camera placement and lifecycle.
        camera_node_rank = getattr(self.robot_info.config, "camera_node_rank", None)
        self.robot = FrankaRobot.build(
            robot_ip=self.config.robot_ip,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            backend=self.config.backend,
            compliance=getattr(self.robot_info.config, "compliance", None),
            end_effector_type=self.config.end_effector_type,
            end_effector_config=self.config.end_effector_config,
            gripper_connection=self.config.gripper_connection,
            cameras={info.name: info for info in self._camera_infos},
            camera_node_rank=camera_node_rank,
        )
        self.robot.connect()
        # Naming the class each part is expected to be keeps the driver's own
        # methods resolvable from here, and reports a mismatched composition at
        # this line rather than as a missing attribute mid-episode.
        self._arm: BaseArm = self.robot.child("arm", BaseArm)
        # The end effector is a part beside the arm, with its own connection.
        # A mismatch here means the declared spaces do not fit the hardware
        # that turned up, which surfaces later as a shape error mid-episode.
        self._end_effector: BaseEndEffector = self.robot.child(
            "end_effector", BaseEndEffector
        )
        if self._end_effector.is_hand is not self._ee_type.is_hand:
            raise ValueError(
                f"end_effector_type={self.config.end_effector_type!r} declares a "
                f"{'hand' if self._ee_type.is_hand else 'gripper'}, but the robot "
                f"built a {type(self._end_effector).__name__}. The action and "
                "observation spaces were sized from the config, so they would "
                "not fit this part."
            )

    def _setup_reward_worker(self) -> None:
        if not self.config.use_reward_model:
            return
        if self.config.reward_worker_cfg is None:
            raise ValueError(
                "use_reward_model=True but reward_worker_cfg is not provided in env override_cfg."
            )

        from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

        reward_node_rank = self.config.reward_worker_node_rank
        if reward_node_rank is None:
            reward_node_rank = self.node_rank

        self._reward_worker = EmbodiedRewardWorker.launch_for_realworld(
            reward_cfg=self.config.reward_worker_cfg,
            node_rank=reward_node_rank,
            node_group_label=self.config.reward_worker_node_group,
            hardware_rank=self.config.reward_worker_hardware_rank,
            env_idx=self.env_idx,
            worker_rank=self.env_worker_rank,
        )
        self._reward_worker.init_worker().wait()

    def transform_action_ee_to_base(self, action: np.ndarray) -> np.ndarray:
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        For gripper end-effectors (7-D action)::

            [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper_action]

        For dexterous hand (12-D action)::

            [
                x_delta,
                y_delta,
                z_delta,
                rx_delta,
                ry_delta,
                rz_delta,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
            ]
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.next_position = self._franka_state.tcp_pose.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self.config.action_scale[0]
        )

        is_ee_action_effective = True
        if not self.config.is_dummy:
            self.next_position[3:] = (
                R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                * R.from_quat(self._franka_state.tcp_pose[3:].copy())
            ).as_quat()

            # Apply the end-effector command before moving the arm.
            ee_action = action[6:]
            is_ee_action_effective = self._end_effector_action(ee_action)

            self._move_action(self._clip_position_to_safety_box(self.next_position))

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._franka_state = self._read_robot()
        else:
            self._franka_state = self._franka_state
        observation = self._get_observation()

        # Reward evaluation also updates the success hold counter.
        reward = self._calc_step_reward(observation, is_ee_action_effective)

        # Terminate after the target has been held for the configured duration.
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )

        truncated = self._num_steps >= self.config.max_num_steps
        reward *= self.config.reward_scale
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def get_tcp_pose(self) -> np.ndarray:
        """Return the current TCP pose ``[x, y, z, qx, qy, qz, qw]``."""
        self._franka_state = self._read_robot()
        return self._franka_state.tcp_pose

    def get_action_scale(self) -> np.ndarray:
        """Return the action scale ``[pos_scale, ori_scale, gripper_scale]``."""
        return self.config.action_scale

    def get_hand_reset_pose(self) -> Optional[np.ndarray]:
        """Return the dexterous-hand pose applied during reset."""
        if not self._is_hand:
            return None
        return np.asarray(self.config.hand_reset_state, dtype=np.float64)

    def get_gripper_open(self) -> bool:
        """Return whether the gripper is currently open."""
        return bool(self._end_effector.is_open)

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return the Cartesian arm and configured end-effector action parts."""
        if self._is_hand:
            return (
                ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
                ActionPart("hand", 6, ActionKind.HAND),
            )
        return (
            ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
            ActionPart("end_effector", 1, ActionKind.GRIPPER),
        )

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray | FrankaRobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute reward from the current robot state and camera frames.

        Args:
            observation: Current environment observation.
            is_gripper_action_effective: Whether the gripper state changed.
        """
        if self.config.use_reward_model:
            reward = self._compute_reward_model(observation)
            if reward >= 1.0:
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty
            return reward

        if not self.config.is_dummy:
            # Compare orientation in the Euler representation used by the config.
            euler_angles = np.abs(
                R.from_quat(self._franka_state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._franka_state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)

            # Check whether the current state is within the success threshold.
            is_in_target_zone = np.all(
                target_delta[:3] <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                self._success_hold_counter += 1
                reward = 1.0
            else:
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria. Target delta: {target_delta}, "
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            if (
                self.config.enable_gripper_penalty
                and not self._is_hand
                and is_gripper_action_effective
            ):
                reward -= self.config.gripper_penalty

            return reward
        else:
            return 0.0

    def _compute_reward_model(
        self, observation: dict[str, np.ndarray | FrankaRobotState]
    ) -> float:
        if self._reward_worker is None:
            raise RuntimeError("Reward worker is not initialized.")

        frames = observation.get("frames", {})
        if not frames:
            raise ValueError("No frames available for reward model inference.")

        image_key = self.config.reward_image_key
        if image_key is None:
            image_key = sorted(frames.keys())[0]
        if image_key not in frames:
            raise KeyError(
                f"reward_image_key '{image_key}' not found in frames. "
                f"Available keys: {list(frames.keys())}"
            )

        image_batch = np.expand_dims(frames[image_key], axis=0)
        reward_output = self._reward_worker.compute_reward(
            {"main_images": image_batch}
        ).wait()[0]
        if hasattr(reward_output, "detach"):
            reward_output = reward_output.detach().cpu().numpy()
        reward_array = np.asarray(reward_output).reshape(-1)
        return float(reward_array[0])

    def reset(
        self,
        joint_reset: bool = False,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # A run with no hardware samples this space instead of reading,
        # so seeding it is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._base_observation_space)
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        self._success_hold_counter = 0

        self._arm.reconfigure_compliance_params(self.config.compliance_param)

        # Periodically return the joints to their configured reset positions.
        joint_reset_cycle = next(self._joint_reset_cycle)
        joint_reset = False
        if joint_reset_cycle == 0:
            self._logger.info(
                f"Number of resets reached {self.config.joint_reset_cycle}, resetting joints to initial position."
            )
            joint_reset = True

        self.go_to_rest(joint_reset)

        self._clear_error()
        self._num_steps = 0
        self._franka_state = self._read_robot()
        observation = self._get_observation()

        return observation, {}

    def go_to_rest(self, joint_reset: bool = False) -> None:
        if joint_reset:
            self._arm.reset_joint(self.config.joint_reset_qpos)
            time.sleep(0.5)

        # Move the arm to a fixed or randomized Cartesian reset pose.
        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose[:2] += np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            euler_random = self.config.target_ee_pose[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range
            )
            reset_pose[3:] = R.from_euler("xyz", euler_random).as_quat()
        else:
            reset_pose = self._reset_pose.copy()

        self._franka_state = self._read_robot()
        cnt = 0
        while not np.allclose(self._franka_state.tcp_pose[:3], reset_pose[:3], 0.02):
            cnt += 1
            self._interpolate_move(reset_pose)
            self._franka_state = self._read_robot()
            if cnt > 2:
                break

        # Reset dexterous hands; individual tasks remain responsible for grippers.
        if self._is_hand:
            self._end_effector.reset(np.asarray(self.config.hand_reset_state))
            self._last_hand_command = (
                np.array(self.config.hand_reset_state, dtype=np.float64)
                * self.config.hand_action_scale
            )

    @property
    def _ee_type(self) -> EndEffectorType:
        """Cached end-effector type enum."""
        return EndEffectorType(self.config.end_effector_type)

    @property
    def _is_hand(self) -> bool:
        """Whether the active end-effector is a dexterous hand.

        Asked of the part once one exists. Gymnasium spaces are declared
        before hardware is set up, and a dummy env never sets any up, so the
        configured type answers until then. :meth:`_setup_hardware` checks the
        two agree rather than letting them drift.
        """
        end_effector = getattr(self, "_end_effector", None)
        if end_effector is None:
            return self._ee_type.is_hand
        return end_effector.is_hand

    def _init_action_obs_spaces(self) -> None:
        """Initialize spaces and Cartesian safety limits.

        The action dimension adapts to the active end-effector:
        - Gripper: 7-D (6 arm + 1 gripper)
        - Dexterous hand: 12-D (6 arm + 6 hand DOFs)
        """
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )

        # The arm has six Cartesian values; the end-effector size varies.
        ee_action_dim = 6 if self._is_hand else 1
        total_action_dim = 6 + ee_action_dim
        self.action_space = gym.spaces.Box(
            np.ones((total_action_dim,), dtype=np.float32) * -1,
            np.ones((total_action_dim,), dtype=np.float32),
        )

        obs_tcp_pose_dim = 7
        # Match the state field and dimension to the selected end effector.
        if self._is_hand:
            ee_state_key = "hand_position"
            ee_state_dim = 6
            ee_low, ee_high = 0.0, 1.0
        else:
            ee_state_key = "gripper_position"
            ee_state_dim = 1
            ee_low, ee_high = -1.0, 1.0

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(obs_tcp_pose_dim,)
                        ),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        ee_state_key: gym.spaces.Box(
                            ee_low, ee_high, shape=(ee_state_dim,)
                        ),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        camera_info.name: gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for camera_info in self._camera_infos
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    @staticmethod
    def _normalize_crop_region(
        crop_region: Any,
        *,
        camera_name: str,
        serial: str,
    ) -> tuple[float, float, float, float]:
        """Validate and normalize a crop region from the config."""
        if not isinstance(crop_region, (list, tuple)) or len(crop_region) != 4:
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): expected "
                "[top, left, bottom, right]."
            )

        try:
            top_pct, left_pct, bottom_pct, right_pct = tuple(
                float(value) for value in crop_region
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): expected numeric values, "
                f"got {crop_region!r}."
            ) from exc

        normalized_crop_region = (top_pct, left_pct, bottom_pct, right_pct)
        if not all(0.0 <= value <= 1.0 for value in normalized_crop_region):
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): values must be within "
                f"[0, 1], got {crop_region!r}."
            )
        if bottom_pct <= top_pct or right_pct <= left_pct:
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): expected "
                "bottom > top and right > left, "
                f"got {crop_region!r}."
            )

        return normalized_crop_region

    def _build_camera_infos(self) -> list[CameraInfo]:
        if self.config.camera_serials is None:
            return []

        ordered_serials = [str(serial) for serial in self.config.camera_serials]

        default_camera_type = self.config.camera_type or "realsense"
        camera_names = self.config.camera_names or {}
        camera_crop_regions = self.config.camera_crop_regions or {}
        camera_infos: list[CameraInfo] = []
        for camera_index, serial in enumerate(ordered_serials, start=1):
            default_name = f"wrist_{camera_index}"
            name = camera_names.get(serial, default_name)

            crop_region = camera_crop_regions.get(serial)
            if crop_region is not None:
                crop_region = self._normalize_crop_region(
                    crop_region,
                    camera_name=name,
                    serial=serial,
                )

            camera_infos.append(
                CameraInfo(
                    name=name,
                    serial_number=serial,
                    camera_type=default_camera_type,
                    crop_region=crop_region,
                )
            )

        return camera_infos

    def _open_cameras(self) -> None:
        """Take the cameras the robot composed and connected.

        The env never builds a camera of its own: one is hardware the robot
        owns, places, and releases, and a second object for the same device
        would open it twice.
        """
        self._cameras: dict[str, BaseCamera] = {
            path.rsplit(".", 1)[-1]: camera
            for path, camera in self.robot.parts_of_type(Camera).items()
        }

    def close(self) -> None:
        """Release all hardware resources including cameras and video player."""
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if not self.config.is_dummy and hasattr(self, "_cameras"):
            self._close_cameras()
        if self.robot is not None:
            self.robot.disconnect()
        super().close()

    def _close_cameras(self) -> None:
        """Drop the camera references; the robot closes what it opened."""
        self._cameras = {}

    def _crop_frame(
        self,
        frame: np.ndarray,
        reshape_size: tuple[int, int],
        crop_region: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop the frame and resize.

        Args:
            frame: Raw camera frame ``(H, W, C)``.
            reshape_size: Target ``(width, height)`` after resize.
            crop_region: Optional relative crop ``(top, left, bottom, right)``
                where each value is in ``[0, 1]``.  ``None`` falls back to the
                default centre-square crop.

        Returns:
            A tuple of ``(cropped_frame, resized_frame)``.
        """
        h, w, _ = frame.shape
        if crop_region is not None:
            top_pct, left_pct, bottom_pct, right_pct = crop_region
            y1 = int(h * top_pct)
            x1 = int(w * left_pct)
            y2 = int(h * bottom_pct)
            x2 = int(w * right_pct)
            cropped_frame = frame[y1:y2, x1:x2]
        else:
            crop_size = min(h, w)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            cropped_frame = frame[
                start_y : start_y + crop_size, start_x : start_x + crop_size
            ]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return cropped_frame, resized_frame

    def _frames_from(self, reading: dict[str, Any]) -> dict[str, np.ndarray]:
        """Crop and resize the frames one robot reading carried."""
        crops = {info.name: info.crop_region for info in self._camera_infos}
        frames = {}
        display_frames = {}
        for name in self._cameras:
            frame = np.asarray(reading[name]["frame"])

            reshape_size = self.observation_space["frames"][name].shape[:2][::-1]
            cropped_frame, resized_frame = self._crop_frame(
                frame, reshape_size, crop_region=crops.get(name)
            )
            frames[name] = resized_frame[..., ::-1]  # Policy input in BGR.
            display_frames[name] = resized_frame  # Display frame in RGB.
            display_frames[f"{name}_full"] = cropped_frame  # Full crop for display.

        self.camera_player.put_frame(display_frames)
        return frames

    # Robot action helpers.

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")
        euler = clip_euler_to_target_window(
            euler=euler,
            target_euler=self.config.target_ee_pose[3:],
            lower_euler=self._rpy_safe_space.low,
            upper_euler=self._rpy_safe_space.high,
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self) -> None:
        self._arm.clear_errors()

    def _binary_gripper_action(self, position: float) -> bool:
        """Execute a scaled binary gripper command."""
        if (
            position <= -self.config.binary_gripper_threshold
            and self._end_effector.is_open
        ):
            self._end_effector.close()
            time.sleep(0.6)
            return True
        if (
            position >= self.config.binary_gripper_threshold
            and not self._end_effector.is_open
        ):
            self._end_effector.open()
            time.sleep(0.6)
            return True
        return False

    def _end_effector_action(self, ee_action: np.ndarray) -> bool:
        """Dispatch an action to the active end-effector.

        For gripper end-effectors the action is a scalar binary signal;
        for dexterous hands it is a 6-D continuous target.

        Args:
            ee_action: End-effector portion of the action vector (after the
                first 6 arm DOFs).

        Returns:
            ``True`` if the action caused a meaningful state change.
        """
        if not self._is_hand:
            # Preserve the established binary gripper action contract.
            position = float(ee_action[0]) * self.config.action_scale[2]
            return self._binary_gripper_action(position)
        else:
            scaled = (
                np.asarray(ee_action, dtype=np.float64) * self.config.hand_action_scale
            )
            if self._last_hand_command is not None:
                delta = scaled - self._last_hand_command
                max_d = self.config.hand_max_delta_per_step
                scaled = self._last_hand_command + np.clip(delta, -max_d, max_d)
            self._last_hand_command = scaled.copy()
            self._end_effector.command(scaled)
            return True

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5) -> None:
        num_steps = int(timeout * self.config.step_frequency)
        self._franka_state: FrankaRobotState = self._read_robot()
        pos_path = np.linspace(
            self._franka_state.tcp_pose[:3], pose[:3], int(num_steps) + 1
        )
        quat_path = quat_slerp(
            self._franka_state.tcp_pose[3:], pose[3:], int(num_steps) + 1
        )

        for pos, quat in zip(pos_path[1:], quat_path[1:]):
            pose = np.concatenate([pos, quat])
            self._move_action(pose.astype(np.float32))
            time.sleep(1.0 / self.config.step_frequency)

        self._franka_state: FrankaRobotState = self._read_robot()

    def _move_action(self, position: np.ndarray) -> None:
        if self.config.is_dummy:
            self._logger.info("Executing dummy action towards %s.", position)
            return
        self._clear_error()
        # Commanding the named part rather than the driver is what lets this
        # env run on any arm backend that accepts a Cartesian target.
        self.robot.send_action({"arm": {"tcp_pose": position.astype(np.float32)}})

    def _read_robot(self) -> FrankaRobotState:
        """Read every part once and refresh the cached arm and frame state.

        Reading the robot rather than its drivers is what keeps this env
        independent of which backend the arm runs, and parts on separate
        connections answer at the same time instead of in turn.
        """
        reading = self.robot.get_observation()
        self._franka_state = self._state_from(reading)
        self._last_frames = self._frames_from(reading)
        return self._franka_state

    def _state_from(self, reading: dict[str, Any]) -> FrankaRobotState:
        """Fill the arm state this env carries from one robot reading."""
        state = FrankaRobotState()
        for name, value in reading["arm"].items():
            if hasattr(state, name):
                setattr(state, name, value)
        end_effector = np.asarray(reading["end_effector"]["state"])
        if self._is_hand:
            state.hand_position = end_effector
        else:
            state.gripper_position = float(end_effector[0])
            state.gripper_open = bool(self._end_effector.is_open)
        return state

    def _get_observation(self) -> dict[str, Any]:
        if not self.config.is_dummy:
            frames = self._last_frames
            state: dict = {
                "tcp_pose": self._franka_state.tcp_pose,
                "tcp_vel": self._franka_state.tcp_vel,
                "tcp_force": self._franka_state.tcp_force,
                "tcp_torque": self._franka_state.tcp_torque,
            }
            # Use the state field declared in observation_space.
            if self._is_hand:
                hand_pos = self._franka_state.hand_position
                if hand_pos is None:
                    hand_pos = np.zeros(6)
                state["hand_position"] = hand_pos
            else:
                state["gripper_position"] = np.array(
                    [self._franka_state.gripper_position]
                )
            state = {
                key: np.asarray(value, dtype=np.float32) for key, value in state.items()
            }
            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            obs = self._base_observation_space.sample()
            return obs

    def transform_obs_base_to_ee(self, state: dict[str, Any]) -> dict[str, Any]:
        self.adjoint_matrix = construct_adjoint_matrix(self._franka_state.tcp_pose)
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)

        state["tcp_vel"] = adjoint_inv @ state["tcp_vel"]

        T_b_o = construct_homogeneous_matrix(self._franka_state.tcp_pose)
        T_r_o = self.T_b_r_inv @ T_b_o

        p_r_o = T_r_o[:3, 3]
        quat_r_o = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
        state["tcp_pose"] = np.concatenate([p_r_o, quat_r_o], axis=0)

        return state

    @property
    def target_ee_pose(self) -> np.ndarray:
        tgt = np.concatenate(
            [
                self.config.target_ee_pose[:3],
                R.from_euler("xyz", self.config.target_ee_pose[3:].copy()).as_quat(),
            ]
        ).copy()
        return tgt
