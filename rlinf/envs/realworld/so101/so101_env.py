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

import contextlib
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

    Hardware connection fields (``port``, ``leader_port``,
    ``calibration_id``, ``camera_cfgs``) are auto-filled from
    :class:`~rlinf.scheduler.SO101HWInfo` when left as ``None``.
    """

    port: Optional[str] = None
    leader_port: Optional[str] = None
    calibration_id: Optional[str] = None
    leader_calibration_id: Optional[str] = None
    """Calibration id for the leader arm. When ``None``, defaults to
    ``f"{calibration_id}_leader"``."""
    arm_variant: str = "so101"

    camera_cfgs: dict[str, dict] = field(default_factory=dict)
    """Camera configurations keyed by camera name.  Each value is a dict
    forwarded to LeRobot's ``CameraConfig`` constructor.  The ``width``,
    ``height``, and ``fps`` keys inside each dict are the **capture**
    parameters accepted by the camera hardware, NOT the stored resolution.
    To shrink stored frames, set :attr:`image_width` and
    :attr:`image_height` — frames are resized via OpenCV after capture.
    Empty = a single blank ``camera_0`` placeholder.  Example::

        camera_cfgs = {
            "front": {"type": "intelrealsense",
                      "serial_number_or_name": "409122274720",
                      "width": 640, "height": 480, "fps": 30,
                      "use_depth": True},
            "handeye": {"type": "opencv",
                        "index_or_path": "/dev/video8",
                        "width": 640, "height": 480, "fps": 30},
        }
    """

    image_width: int = 128
    """Output image width in pixels.  Camera frames larger than this are
    resized (OpenCV bilinear) before storage.  Set to ``0`` to keep the
    native capture resolution."""

    image_height: int = 128
    """Output image height in pixels.  Paired with :attr:`image_width`."""

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
    """Fallback joint-angle target ``[q1..q5, gripper]`` in degrees.
    Only used when ``target_ee_pose`` is ``None``.  Prefer
    ``target_ee_pose`` for real tasks — it matches how a human reasons
    about the arm (\"gripper to position X in the workspace\")."""

    target_ee_pose: tuple[float, ...] | None = None
    """End-effector target ``(x, y, z)`` in **metres**.  When set, the
    per-step reward is computed from the gripper's 3‑D Euclidean distance
    to the target (via forward kinematics), replacing the joint-angle
    path entirely.  ``None`` falls back to ``target_joint_qpos``."""

    urdf_path: str | None = None
    """Path to the SO101 URDF file for forward kinematics.  ``None``
    (the default) picks up the calibration URDF bundled with lerobot.
    Only needed when ``target_ee_pose`` is set and the env is not
    running in dummy mode."""

    reward_threshold_m: float = 0.03
    """EE-space success radius in **metres**.  When the gripper is within
    this Euclidean distance of ``target_ee_pose`` the reward saturates."""

    reward_threshold_deg: float = 5.0

    success_hold_steps: int = 3
    """Consecutive steps within the threshold required for success.
    Prevents transient noise from triggering a false-positive success."""

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
        if self.target_ee_pose is not None:
            self.target_ee_pose = tuple(float(v) for v in self.target_ee_pose)
            if len(self.target_ee_pose) != 3:
                raise ValueError(
                    f"target_ee_pose must have 3 elements (x, y, z); "
                    f"got {len(self.target_ee_pose)}"
                )
        if self.reward_threshold_m <= 0:
            raise ValueError(
                f"reward_threshold_m must be > 0; got {self.reward_threshold_m}"
            )


class SO101Env(gym.Env):
    """SO101 6-DOF robot environment with joint-space actions.

    Action space: ``Box((6,))`` — ``[q1..q5, gripper]`` in degrees, where
    ``q1..q5`` are absolute joint position targets and ``gripper`` is the
    gripper opening. All targets are clipped to the configured limits before
    being sent to the motor bus.

    Observation space: ``Dict`` with ``state`` containing ``joint_position``
    (5 arm joints) and ``gripper_position`` (1,). Camera frames are only
    present in the observation when ``camera_cfgs`` is configured *and* the
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
        self._success_hold_counter = 0
        self._robot = None
        self._leader = None
        self._camera_frames: dict = {}
        self._evdev_device = None
        self._key_state_lock = None
        self._key_state: dict[str, bool] = {
            "start_episode": False,
            "end_episode": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }
        self._recording = False

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
        if not self.config.camera_cfgs:
            self.config.camera_cfgs = getattr(hw, "camera_cfgs", None) or {}
        if self.config.calibration_id is None:
            self.config.calibration_id = getattr(hw, "calibration_id", "default")
        if self.config.leader_calibration_id is None:
            self.config.leader_calibration_id = getattr(hw, "leader_calibration_id", None)

    @staticmethod
    def _build_camera_configs(raw: dict[str, dict]) -> dict:
        """Build LeRobot ``CameraConfig`` instances from user-facing dicts.

        Each value in *raw* must contain a ``"type"`` key (``"intelrealsense"``
        or ``"opencv"``) along with the arguments expected by the corresponding
        :class:`~lerobot.cameras.configs.CameraConfig` subclass.  The ``"type"``
        key is consumed and not forwarded.
        """
        # CameraConfig subclasses register themselves at module import time;
        # LeRobot does not auto-import them, so we must import the modules
        # that contain the registrations before get_choice_class() can find them.
        import lerobot.cameras.opencv.configuration_opencv as _  # registers "opencv"
        import lerobot.cameras.realsense.configuration_realsense as _  # registers "intelrealsense"
        from lerobot.cameras.configs import CameraConfig

        result: dict = {}
        for name, cfg in raw.items():
            cfg = dict(cfg)
            cam_type = cfg.pop("type")
            cls = CameraConfig.get_choice_class(cam_type)
            result[name] = cls(**cfg)
        return result

    def _connect_robot(self):
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        camera_configs = self._build_camera_configs(self.config.camera_cfgs)

        robot_cfg = SO101FollowerConfig(
            port=self.config.port,
            id=self.config.calibration_id or "default",
            use_degrees=self.config.use_degrees,
            max_relative_target=self.config.max_relative_target,
            disable_torque_on_disconnect=False,
            cameras=camera_configs,
        )
        self._robot = SO101Follower(robot_cfg)
        self._robot.connect(calibrate=self.config.auto_calibrate)

        if camera_configs:
            names = ", ".join(camera_configs)
            self._logger.info(f"SO101 cameras configured: {names}")

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

    def _start_evdev_listener(self) -> bool:
        """Find and start an ``evdev``-based keyboard listener.

        Scans ``/dev/input/event*`` for the first keyboard device that
        supports both ``KEY_S`` and ``KEY_Q``, starts a daemon read-loop
        thread, and returns ``True``.  Returns ``False`` if no usable
        keyboard is found (``evdev`` missing, no permission, or no
        matching input device).
        """
        try:
            import evdev
        except ImportError:
            self._logger.info("[SO101Env] evdev not installed.")
            return False

        # Map evdev key codes → SO101 event name.
        _ev_key_s = evdev.ecodes.KEY_S
        _ev_key_r = evdev.ecodes.KEY_R
        _ev_key_q = evdev.ecodes.KEY_Q
        _ev_key_right = evdev.ecodes.KEY_RIGHT
        _ev_key_left = evdev.ecodes.KEY_LEFT
        _ev_key_esc = evdev.ecodes.KEY_ESC

        code_to_event = {
            _ev_key_s: "start_episode",
            _ev_key_r: "rerecord_episode",
            _ev_key_q: "stop_recording",
            _ev_key_right: "start_episode",
            _ev_key_left: "rerecord_episode",
            _ev_key_esc: "stop_recording",
            evdev.ecodes.KEY_E: "end_episode",
        }

        # Find a keyboard device (has KEY_S and KEY_Q in its capabilities).
        # We glob /dev/input/event* ourselves because evdev's is_device()
        # requires R_OK|W_OK — we only need read access for the listener.
        import glob as _glob
        _paths = sorted(_glob.glob("/dev/input/event*"))
        if not _paths:
            self._logger.info("[SO101Env] evdev: no /dev/input/event* devices found.")
            return False

        device = None
        for path in _paths:
            try:
                _dev = evdev.InputDevice(path)
            except PermissionError:
                continue
            try:
                caps = _dev.capabilities(verbose=False)
                key_caps = set(caps.get(evdev.ecodes.EV_KEY, []))
                if _ev_key_s in key_caps and _ev_key_q in key_caps:
                    device = _dev
                    break
            finally:
                if device is not _dev:
                    _dev.close()

        if device is None:
            self._logger.info("[SO101Env] evdev: no keyboard with KEY_S+KEY_Q found.")
            return False

        import threading

        self._key_state_lock = threading.Lock()

        def _evdev_loop():
            try:
                for event in device.read_loop():
                    if event.type != evdev.ecodes.EV_KEY:
                        continue
                    # Only act on key-down (value 1) and auto-repeat (value 2).
                    if event.value not in (1, 2):
                        continue
                    event_name = code_to_event.get(event.code)
                    if event_name is None:
                        continue
                    with self._key_state_lock:
                        self._key_state[event_name] = True
                    self._logger.info(f"[SO101Env] evdev: {event_name} triggered")
            except Exception as e:
                self._logger.warning(f"[SO101Env] evdev loop exited: {e}")

        t = threading.Thread(target=_evdev_loop, daemon=True, name="so101-kbd")
        t.start()
        self._evdev_device = device  # keep alive for close()
        self._logger.info(
            f"[SO101Env] Keyboard ready (evdev: {device.name}): "
            "'s'/→ start, 'e' end, 'r'/← rerecord, 'q'/Esc stop."
        )
        return True

    def _start_keyboard_listener(self) -> None:
        """Start a keyboard listener for manual episode control.

        Reads directly from the Linux input event layer via ``evdev``
        — no X11 or graphical session needed.

        Key bindings:
          - ``s`` / →     start a NEW episode (begin recording).
          - ``e``         END an episode (save the current recording).
          - ``r`` / ←     RE-RECORD (discard the current episode).
          - ``q`` / Esc   QUIT data collection.

        Only started in teleop mode.  If ``evdev`` is not available or no
        readable keyboard device is found, episodes terminate by timeout
        only.
        """
        if not self.config.enable_teleop:
            return

        if self._start_evdev_listener():
            return

        self._logger.info(
            "[SO101Env] no keyboard device found — "
            "episodes will end via timeout only (max_episode_steps)."
            "To enable keyboard control, ensure the runtime user has read "
            "access to /dev/input/event* (e.g. group 'input')."
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

        # Use the configured LeRobot camera names and the *output* image
        # size (image_width / image_height), NOT the camera capture
        # resolution.  Frames are resized on read if the sizes differ.
        # When no cameras are configured, emit a single blank ``camera_0``
        # so ``RealWorldEnv._wrap_obs`` keeps working.
        out_h = self.config.image_height or 128
        out_w = self.config.image_width or 128
        frame_entries: dict = {}
        if self.config.camera_cfgs:
            for name in self.config.camera_cfgs:
                frame_entries[name] = gym.spaces.Box(
                    0, 255, shape=(out_h, out_w, 3), dtype=np.uint8
                )
        else:
            frame_entries["camera_0"] = gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )

        self.observation_space = gym.spaces.Dict(
            {"state": state_space, "frames": gym.spaces.Dict(frame_entries)}
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _handle_teleop_controls(
        self, robot_action: dict
    ) -> tuple[bool, dict]:
        """Snapshot the evdev key state and return termination + info.

        Called once per ``step()`` from the teleop fast-path.  The
        background evdev thread only ever sets flags to ``True`` so we
        snapshot-and-clear under the lock to deliver each press exactly
        once.

        Key bindings (assigned by :meth:`_start_evdev_listener`):

        * ``s`` / →   **start** — begin recording a new episode.  The
          env stays where the operator placed the arm; no reset.
        * ``e``       **end**   — save the current episode as success.
        * ``r`` / ←   **rerecord** — discard the current episode.
        * ``q`` / Esc **stop**  — exit data collection entirely.

        Returns:
            ``(terminated, info_dict)``.  The caller merges *info_dict*
            into the step's ``info`` and forwards *terminated*.
        """
        with self._key_state_lock if self._key_state_lock else contextlib.nullcontext():
            pressed = dict(self._key_state)
            for k in self._key_state:
                self._key_state[k] = False

        stop    = bool(pressed.get("stop_recording"))
        rerec   = bool(pressed.get("rerecord_episode"))
        start   = bool(pressed.get("start_episode"))
        end_ep  = bool(pressed.get("end_episode"))

        terminated = False
        info: dict = {}

        if stop:
            self._recording = False
            info.update(keyboard_event="stop", keyboard_phase="pre",
                        stop_recording=True, rerecord_episode=False,
                        episode_success=False, record_reset=True)
            terminated = True
        elif self._recording and rerec:
            self._recording = False
            info.update(keyboard_event="abort", keyboard_phase="pre",
                        stop_recording=False, rerecord_episode=True,
                        episode_success=False, record_reset=True)
            terminated = True
        elif self._recording and end_ep:
            self._recording = False
            info.update(keyboard_event="end_success", keyboard_phase="pre",
                        stop_recording=False, rerecord_episode=False,
                        episode_success=True, record_reset=True)
            terminated = True
        elif not self._recording and start:
            self._recording = True
            info.update(keyboard_event="start", keyboard_phase="rec",
                        stop_recording=False, rerecord_episode=False,
                        episode_success=False, record_reset=True)
        else:
            phase = "rec" if self._recording else "pre"
            info.update(keyboard_event=None, keyboard_phase=phase,
                        stop_recording=False, rerecord_episode=False,
                        episode_success=False, record_reset=False)

        if self._leader is not None:
            info["intervene_action"] = np.array(
                [robot_action.get(f"{name}.pos", 0.0) for name in _SO101_MOTOR_NAMES],
                dtype=np.float32,
            )

        return terminated, info

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
            terminated, info = self._handle_teleop_controls(robot_action)
        else:
            terminated = bool(reward >= 1.0)
            info = {}

        max_steps = (
            self.config.max_episode_steps
            if self.config.max_episode_steps is not None
            else self.config.max_num_steps
        )
        truncated = self._num_steps >= max_steps

        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to ``reset_joint_qpos`` and clear counters."""
        self._num_steps = 0
        self._success_hold_counter = 0
        self._recording = False
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

        # Camera frames are stored in the observation dict under their
        # configured names (e.g. "front", "handeye", "front_depth").
        # Motor keys all end with ".pos" or ".vel" — camera keys don't.
        # Collect everything that isn't a known motor suffix.
        motor_affixes = frozenset({".pos", ".vel"})
        self._camera_frames = {}
        for key, val in obs.items():
            if not any(isinstance(key, str) and key.endswith(sfx) for sfx in motor_affixes):
                self._camera_frames[key] = np.asarray(val)

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        self._update_state()
        return {
            "state": {
                "joint_position": self._state.joint_position.astype(np.float32),
                "gripper_position": np.array(
                    [self._state.gripper_position], dtype=np.float32
                ),
            },
            "frames": self._get_camera_frames(),
        }

    def _get_camera_frames(self) -> dict:
        """Return camera frames, resized to ``image_width × image_height``.

        When LeRobot cameras are configured, frames are read from
        ``self._camera_frames`` (populated by ``_update_state`` from
        ``robot.get_observation()``) and optionally resized via OpenCV.
        When no cameras are configured, blank ``camera_0`` placeholder
        frames are emitted.
        """
        if self._camera_frames:
            tw = self.config.image_width or 0
            th = self.config.image_height or 0

            # Expand obs space lazily for any extra keys LeRobot emits
            # (e.g. depth) that we didn't declare at init time.
            declared = set(self.observation_space["frames"].spaces)
            actual = set(self._camera_frames)
            new = actual - declared
            if new:
                spaces = dict(self.observation_space["frames"].spaces)
                for key in sorted(new):
                    arr = self._camera_frames[key]
                    dtype = np.uint16 if arr.dtype == np.uint16 else np.uint8
                    low, high = (0, 65535) if dtype == np.uint16 else (0, 255)
                    h, w = (th, tw) if (tw and th) else arr.shape[:2]
                    spaces[key] = gym.spaces.Box(
                        low, high, shape=(h, w) + arr.shape[2:], dtype=dtype
                    )
                self.observation_space.spaces["frames"] = gym.spaces.Dict(spaces)
                self._base_observation_space = copy.deepcopy(self.observation_space)
                self._logger.info(f"SO101 obs space expanded with: {sorted(new)}")

            if not tw or not th:
                return dict(self._camera_frames)

            result: dict = {}
            for key, arr in self._camera_frames.items():
                if arr.ndim >= 2 and (arr.shape[0] != th or arr.shape[1] != tw):
                    import cv2
                    interp = (
                        cv2.INTER_NEAREST
                        if arr.ndim == 3 and arr.shape[2] == 1
                        else cv2.INTER_LINEAR
                    )
                    arr = cv2.resize(arr, (tw, th), interpolation=interp)
                    if arr.ndim == 2:
                        arr = arr[..., None]
                result[key] = arr
            return result

        # No cameras — emit blank placeholder.
        out_h = self.config.image_height or 128
        out_w = self.config.image_width or 128
        blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        return {"camera_0": blank.copy()}

    def _calc_step_reward(self, observation: dict) -> float:
        """Joint-angle reward — exponential falloff, saturating at 1.0.

        When the arm is within ``reward_threshold_deg`` for
        ``success_hold_steps`` consecutive steps, return 1.0
        (success confirmed).  Otherwise return 0.5 (close) or an
        exponential-decay scalar.
        """
        if self.config.is_dummy or self.config.target_joint_qpos is None:
            return 0.0
        arm_pos = observation["state"]["joint_position"]
        target_arm = self.config.target_joint_qpos[:_NUM_ARM_JOINTS]
        error = float(np.linalg.norm(arm_pos - target_arm))
        if error < self.config.reward_threshold_deg:
            self._success_hold_counter += 1
            if self._success_hold_counter >= self.config.success_hold_steps:
                return 1.0
            return 0.5
        self._success_hold_counter = 0
        return float(np.exp(-error / self.config.reward_threshold_deg))

    def _shutdown_arm_safely(self):
        """Park the arm at the rest pose and leave torque enabled.

        We deliberately do NOT disable torque on close. With torque enabled
        and Goal_Position set to rest, the motors actively hold the rest
        pose until power-cycled — the user-visible "alive" state that
        re-running the data collection script produces. Disabling torque
        here would make the arm feel "frozen" (limp + stiff gear-train
        friction), which collaborators consistently report as a regression.

        Sequence:
          1. Send rest pose via send_action.
          2. Wait a fixed 3 s so motors physically reach rest before we
             return — without this, the script can exit while the arm is
             still mid-traverse and the motors then hold *that* pose
             instead of rest.
        """
        try:
            self._go_to_rest()
        except Exception as e:
            self._logger.warning(f"Failed to send rest target on close: {e}")
        time.sleep(3.0)
        self._logger.info("SO101 parked at rest (torque held).")

    def close(self):
        if self._evdev_device is not None:
            try:
                self._evdev_device.close()
            except Exception:
                pass
            self._evdev_device = None

        if self._robot is not None and not self.config.is_dummy:
            self._shutdown_arm_safely()

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
