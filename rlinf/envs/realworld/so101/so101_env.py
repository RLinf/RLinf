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

"""SO101 6-DOF robot environment powered by LeRobot.

Wraps LeRobot's :class:`~lerobot.robots.so_follower.SO101Follower` as a
``gym.Env`` for RLinf training, data collection, and evaluation. LeRobot's
synchronous Python API is used directly, so no distributed controller worker
is required.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .so101_robot_state import SO101RobotState

# SO101 joints in bus-ID order, matching the motor names LeRobot uses for
# observation and action keys (suffixed with ``.pos``).
_SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
_SO101_GRIPPER = "gripper"
_SO101_MOTOR_NAMES = (*_SO101_ARM_JOINTS, _SO101_GRIPPER)
_NUM_ARM_JOINTS = len(_SO101_ARM_JOINTS)

# Conservative defaults for arm joints (degrees). Final ranges come from
# LeRobot calibration on first connect.
_DEFAULT_JOINT_LIMIT_LOW_DEG = np.array(
    [-150.0, -90.0, -150.0, -90.0, -150.0], dtype=np.float64
)
_DEFAULT_JOINT_LIMIT_HIGH_DEG = np.array(
    [150.0, 90.0, 150.0, 90.0, 150.0], dtype=np.float64
)
_DEFAULT_GRIPPER_LIMIT_LOW = 0.0
_DEFAULT_GRIPPER_LIMIT_HIGH = 90.0


def _to_lerobot_action(arm_targets: np.ndarray, gripper_target: float) -> dict:
    """Build the dict format LeRobot's ``send_action`` expects.

    LeRobot filters incoming keys via ``key.endswith(".pos")``; missing the
    suffix silently drops the motor.
    """
    action = {f"{name}.pos": float(arm_targets[i]) for i, name in enumerate(_SO101_ARM_JOINTS)}
    action[f"{_SO101_GRIPPER}.pos"] = float(gripper_target)
    return action


@dataclass
class SO101RobotConfig:
    """Configuration for :class:`SO101Env`.

    Hardware connection fields (``port``, ``leader_port``, ``camera_serials``,
    ``camera_type``, ``calibration_id``) are auto-filled from
    :class:`~rlinf.scheduler.SO101HWInfo` when left as ``None``.
    """

    port: Optional[str] = None
    leader_port: Optional[str] = None
    calibration_id: Optional[str] = None
    leader_calibration_id: Optional[str] = None
    """Calibration id for the leader arm. When ``None``, defaults to
    ``f"{calibration_id}_leader"``."""
    arm_variant: str = "so101"

    camera_serials: Optional[list] = None
    camera_type: Optional[str] = None

    use_degrees: bool = True
    max_relative_target: Optional[float] = None
    """Per-step joint movement cap in degrees (LeRobot ``max_relative_target``)."""

    auto_calibrate: bool = True
    """When ``True``, pass ``calibrate=True`` to LeRobot's ``connect()``: if the
    motor EEPROM already matches the loaded calibration JSON, this is a no-op;
    otherwise LeRobot will prompt on stdin to recalibrate. Set to ``False`` to
    skip calibration entirely (use only when the arm is already calibrated and
    you don't want any interactive prompt)."""

    step_frequency: float = 30.0

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW_DEG.copy()
    )
    """Lower limits ``(5,)`` in degrees, one per arm joint (no gripper)."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH_DEG.copy()
    )
    """Upper limits ``(5,)`` in degrees, one per arm joint (no gripper)."""

    gripper_limit_low: float = _DEFAULT_GRIPPER_LIMIT_LOW
    gripper_limit_high: float = _DEFAULT_GRIPPER_LIMIT_HIGH

    max_num_steps: int = 200
    """Episode truncation horizon when ``max_episode_steps`` is unset."""

    max_episode_steps: Optional[int] = None
    """Override for ``max_num_steps`` from env-level config."""

    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0, -45.0, 90.0, 0.0, 0.0, 45.0]
    )
    """Reset configuration ``[q1..q5, gripper]`` in degrees."""

    target_joint_qpos: Optional[np.ndarray] = None
    """Target ``[q1..q5, gripper]`` for reward; ``None`` returns 0 reward."""

    reward_threshold_deg: float = 5.0
    binary_gripper_threshold: float = 45.0

    is_dummy: bool = False
    """When ``True``, skip all hardware calls (useful for offline training)."""

    enable_teleop: bool = False
    """When ``True``, override actions with leader-arm readings each step."""

    manual_episode_control_only: bool = False
    """When ``True``, episodes terminate only on keyboard events
    (``s``/``r``/``q``); reward-based termination is disabled. Used for
    teleoperation data collection where the demo is human-defined."""

    def __post_init__(self):
        self.joint_limit_low = np.asarray(self.joint_limit_low, dtype=np.float64)
        self.joint_limit_high = np.asarray(self.joint_limit_high, dtype=np.float64)
        if self.joint_limit_low.shape != (_NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_limit_low must have shape ({_NUM_ARM_JOINTS},); "
                f"got {self.joint_limit_low.shape}"
            )
        if self.joint_limit_high.shape != (_NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_limit_high must have shape ({_NUM_ARM_JOINTS},); "
                f"got {self.joint_limit_high.shape}"
            )
        if (self.joint_limit_low >= self.joint_limit_high).any():
            raise ValueError(
                "Each joint_limit_low must be strictly less than joint_limit_high; "
                f"got low={self.joint_limit_low}, high={self.joint_limit_high}"
            )
        if self.gripper_limit_low >= self.gripper_limit_high:
            raise ValueError(
                f"gripper_limit_low ({self.gripper_limit_low}) must be < "
                f"gripper_limit_high ({self.gripper_limit_high})"
            )
        if len(self.reset_joint_qpos) != len(_SO101_MOTOR_NAMES):
            raise ValueError(
                f"reset_joint_qpos must have {len(_SO101_MOTOR_NAMES)} entries "
                f"(arm joints + gripper); got {len(self.reset_joint_qpos)}"
            )
        if self.target_joint_qpos is not None:
            self.target_joint_qpos = np.asarray(self.target_joint_qpos, dtype=np.float64)
            if self.target_joint_qpos.shape != (len(_SO101_MOTOR_NAMES),):
                raise ValueError(
                    f"target_joint_qpos must have shape ({len(_SO101_MOTOR_NAMES)},); "
                    f"got {self.target_joint_qpos.shape}"
                )


class SO101Env(gym.Env):
    """SO101 6-DOF robot environment with joint-space actions.

    Action space: ``Box((6,))`` — ``[q1..q5, gripper]`` in degrees, where
    ``q1..q5`` are absolute joint position targets and ``gripper`` is the
    gripper opening. All targets are clipped to the configured limits before
    being sent to the motor bus.

    Observation space: ``Dict`` with ``state`` containing ``joint_position``
    (5 arm joints) and ``gripper_position`` (1,). Camera frames are only
    present in the observation when ``camera_serials`` is configured *and* the
    cameras are reachable; subclasses are responsible for populating
    ``self._cameras`` accordingly.
    """

    def __init__(
        self,
        config: SO101RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info,
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._state = SO101RobotState()
        self._num_steps = 0
        self._robot = None
        self._leader = None
        self._cameras: dict = {}
        self._pynput_listener = None
        self._key_state_lock = None
        self._key_state: dict[str, bool] = {
            "episode_success": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }

        if not self.config.is_dummy:
            self._setup_hardware()
            self._start_keyboard_listener()
            self._connect_robot()

        self._init_action_obs_spaces()

    def _setup_hardware(self):
        """Fill connection fields from hardware_info when not set by the user."""
        from rlinf.scheduler.hardware.robots.so101 import SO101HWInfo

        if not isinstance(self.hardware_info, SO101HWInfo):
            return
        hw = self.hardware_info.config
        if self.config.port is None:
            self.config.port = hw.port
        if self.config.leader_port is None:
            self.config.leader_port = getattr(hw, "leader_port", None)
        if self.config.camera_serials is None:
            self.config.camera_serials = getattr(hw, "camera_serials", None) or []
        if self.config.camera_type is None:
            self.config.camera_type = getattr(hw, "camera_type", "opencv")
        if self.config.calibration_id is None:
            self.config.calibration_id = getattr(hw, "calibration_id", "default")
        if self.config.leader_calibration_id is None:
            self.config.leader_calibration_id = getattr(hw, "leader_calibration_id", None)

    def _connect_robot(self):
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        robot_cfg = SO101FollowerConfig(
            port=self.config.port,
            id=self.config.calibration_id or "default",
            use_degrees=self.config.use_degrees,
            max_relative_target=self.config.max_relative_target,
            disable_torque_on_disconnect=True,
        )
        self._robot = SO101Follower(robot_cfg)
        self._robot.connect(calibrate=self.config.auto_calibrate)

        if self.config.enable_teleop and self.config.leader_port:
            self._connect_leader()

        self._state.is_connected = True
        self._logger.info(
            f"SO101 robot connected on port {self.config.port} "
            f"(arm_variant={self.config.arm_variant})"
        )

    def _connect_leader(self):
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        leader_id = self.config.leader_calibration_id or (
            f"{self.config.calibration_id or 'default'}_leader"
        )
        leader_cfg = SO101LeaderConfig(
            port=self.config.leader_port,
            id=leader_id,
            use_degrees=self.config.use_degrees,
        )
        self._leader = SO101Leader(leader_cfg)
        self._leader.connect(calibrate=self.config.auto_calibrate)
        self._logger.info(
            f"SO101 leader connected on port {self.config.leader_port} "
            f"(calibration_id={leader_id})"
        )

    def _start_keyboard_listener(self) -> None:
        """Start a ``pynput`` listener for keyboard-driven episode control.

        Key bindings match LeRobot's convention used during teleoperation
        recording:
          - ``s`` / →   ``episode_success``
          - ``r`` / ←   ``rerecord_episode``
          - ``q`` / Esc ``stop_recording``

        Only started in teleop mode. If ``pynput`` is missing or the listener
        fails to start, episodes terminate by timeout only.
        """
        if not self.config.enable_teleop:
            return

        try:
            from pynput import keyboard as pynput_keyboard
        except ImportError:
            self._logger.info(
                "[SO101Env] pynput unavailable — falling back to timeout-only episodes."
            )
            return

        import threading

        self._key_state_lock = threading.Lock()
        key_to_event = {
            pynput_keyboard.Key.right: "episode_success",
            pynput_keyboard.Key.left: "rerecord_episode",
            pynput_keyboard.Key.esc: "stop_recording",
        }
        char_to_event = {
            "s": "episode_success",
            "r": "rerecord_episode",
            "q": "stop_recording",
        }

        def _on_press(key):
            event = key_to_event.get(key)
            if event is None and getattr(key, "char", None):
                event = char_to_event.get(key.char)
            if event is None:
                return
            with self._key_state_lock:
                self._key_state[event] = True
            self._logger.info(f"[SO101Env] {event} triggered")

        try:
            self._pynput_listener = pynput_keyboard.Listener(on_press=_on_press)
            self._pynput_listener.start()
            self._logger.info(
                "[SO101Env] Keyboard ready: 's'/→ save, 'r'/← rerecord, 'q'/Esc stop."
            )
        except Exception as e:
            self._pynput_listener = None
            self._logger.info(
                f"[SO101Env] Keyboard listener failed to start ({e}); "
                "episodes will end on timeout only."
            )

    def _init_action_obs_spaces(self):
        self._joint_limit_low = np.asarray(self.config.joint_limit_low, dtype=np.float64)
        self._joint_limit_high = np.asarray(self.config.joint_limit_high, dtype=np.float64)

        action_low = np.append(
            self._joint_limit_low, self.config.gripper_limit_low
        ).astype(np.float32)
        action_high = np.append(
            self._joint_limit_high, self.config.gripper_limit_high
        ).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        state_space = gym.spaces.Dict(
            {
                "joint_position": gym.spaces.Box(
                    -np.inf, np.inf, shape=(_NUM_ARM_JOINTS,), dtype=np.float32
                ),
                "gripper_position": gym.spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        # ``RealWorldEnv._wrap_obs`` always reads ``frames[main_image_key]``,
        # so ensure at least one camera slot exists. When no cameras are
        # configured we fall back to a single blank ``camera_0`` placeholder
        # to keep downstream wrappers happy.
        num_cameras = max(1, len(self.config.camera_serials or []))
        frame_space = gym.spaces.Dict(
            {
                f"camera_{k}": gym.spaces.Box(
                    0, 255, shape=(128, 128, 3), dtype=np.uint8
                )
                for k in range(num_cameras)
            }
        )
        self.observation_space = gym.spaces.Dict(
            {"state": state_space, "frames": frame_space}
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Args:
            action: ``(6,)`` float array — ``[q1..q5, gripper]`` in degrees.
                Arm targets are clipped to the configured joint limits and the
                gripper to ``[gripper_limit_low, gripper_limit_high]``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.time()
        expected_dim = _NUM_ARM_JOINTS + 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != expected_dim:
            raise ValueError(
                f"action must have {expected_dim} entries; got {action.shape[0]}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        robot_action = _to_lerobot_action(action[:_NUM_ARM_JOINTS], action[_NUM_ARM_JOINTS])

        if not self.config.is_dummy:
            if self.config.enable_teleop and self._leader is not None:
                try:
                    robot_action = self._leader.get_action()
                except Exception as e:
                    self._logger.warning(f"Failed to read leader action: {e}")
            self._robot.send_action(robot_action)

        self._num_steps += 1

        observation = self._get_observation()
        reward = self._calc_step_reward(observation)

        if self.config.enable_teleop and self.config.manual_episode_control_only:
            terminated = any(self._key_state.values())
        else:
            terminated = bool(reward >= 1.0)

        max_steps = (
            self.config.max_episode_steps
            if self.config.max_episode_steps is not None
            else self.config.max_num_steps
        )
        truncated = self._num_steps >= max_steps

        info: dict = {}
        if self.config.enable_teleop:
            if self._leader is not None:
                intervene_action = np.array(
                    [robot_action.get(f"{name}.pos", 0.0) for name in _SO101_MOTOR_NAMES],
                    dtype=np.float32,
                )
                info["intervene_action"] = intervene_action
            info["episode_success"] = self._key_state["episode_success"]
            info["rerecord_episode"] = self._key_state["rerecord_episode"]
            info["stop_recording"] = self._key_state["stop_recording"]
            info["manual_done"] = terminated

        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to ``reset_joint_qpos`` and clear counters."""
        self._num_steps = 0
        for k in self._key_state:
            self._key_state[k] = False

        if self.config.is_dummy:
            return self._get_observation(), {}

        self._go_to_rest()
        self._update_state()
        return self._get_observation(), {}

    def _go_to_rest(self):
        rest = self.config.reset_joint_qpos
        rest_action = _to_lerobot_action(
            np.asarray(rest[:_NUM_ARM_JOINTS], dtype=np.float64),
            float(rest[_NUM_ARM_JOINTS]),
        )
        self._robot.send_action(rest_action)
        time.sleep(1.0)

    def _update_state(self):
        if self.config.is_dummy:
            return
        try:
            obs = self._robot.get_observation()
        except Exception as e:
            self._logger.warning(f"Failed to read robot observation: {e}")
            return
        self._state.joint_position = np.array(
            [obs.get(f"{name}.pos", 0.0) for name in _SO101_ARM_JOINTS],
            dtype=np.float64,
        )
        self._state.joint_velocity = np.array(
            [obs.get(f"{name}.vel", 0.0) for name in _SO101_ARM_JOINTS],
            dtype=np.float64,
        )
        self._state.gripper_position = float(obs.get(f"{_SO101_GRIPPER}.pos", 0.0))
        self._state.gripper_open = (
            self._state.gripper_position > self.config.binary_gripper_threshold
        )

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        self._update_state()
        return copy.deepcopy(
            {
                "state": {
                    "joint_position": self._state.joint_position.astype(np.float32),
                    "gripper_position": np.array(
                        [self._state.gripper_position], dtype=np.float32
                    ),
                },
                "frames": self._get_camera_frames(),
            }
        )

    def _get_camera_frames(self) -> dict:
        """Return frames keyed by ``camera_<i>``.

        Returns one blank-image entry per slot in the declared observation
        space (at least ``camera_0``) so consumers like
        ``RealWorldEnv._wrap_obs`` always find ``main_image_key``. Real
        cameras override the blanks via ``self._cameras``.
        """
        frames: dict = {}
        slots = list(self.observation_space.spaces["frames"].spaces.keys())
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        for slot in slots:
            frames[slot] = blank.copy()
        for name, camera in self._cameras.items():
            try:
                frames[name] = camera.get_frame()
            except Exception as e:
                self._logger.warning(f"Failed to read camera '{name}': {e}")
        return frames

    def _calc_step_reward(self, observation: dict) -> float:
        """Reward = exp falloff on L2 joint error, saturating at 1.0 within tolerance."""
        if self.config.is_dummy or self.config.target_joint_qpos is None:
            return 0.0
        arm_pos = observation["state"]["joint_position"]
        target_arm = self.config.target_joint_qpos[:_NUM_ARM_JOINTS]
        error = float(np.linalg.norm(arm_pos - target_arm))
        if error < self.config.reward_threshold_deg:
            return 1.0
        return float(np.exp(-error / self.config.reward_threshold_deg))

    def close(self):
        if self._pynput_listener is not None:
            try:
                if self._pynput_listener.is_alive():
                    self._pynput_listener.stop()
            except Exception:
                pass
            self._pynput_listener = None

        if self._robot is not None:
            try:
                self._robot.disconnect()
                self._logger.info("SO101 robot disconnected.")
            except Exception as e:
                self._logger.warning(f"Error disconnecting robot: {e}")
            self._robot = None

        if self._leader is not None:
            try:
                self._leader.disconnect()
                self._logger.info("SO101 leader disconnected.")
            except Exception as e:
                self._logger.warning(f"Error disconnecting leader: {e}")
            self._leader = None

        self._state.is_connected = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def robot(self):
        return self._robot

    @property
    def leader(self):
        return self._leader
