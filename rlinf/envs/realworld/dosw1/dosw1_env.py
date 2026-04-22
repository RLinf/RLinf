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

"""DOSW1 dual-arm gymnasium environment with human-in-the-loop support."""

from __future__ import annotations

import copy
import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import DOSW1HWInfo, WorkerInfo
from rlinf.utils.logging import get_logger

from .dosw1_robot_state import DOSW1RobotState
from .dosw1_sdk import DOSW1SDKAdapter


class ControlMode(enum.IntEnum):
    """Data-collection tag written with each recorded transition."""

    MODEL = 0
    PAUSE = 1
    TELEOP = 2


NUM_JOINTS = 6
ACTION_DIM = 14
IMAGE_H, IMAGE_W = 128, 128


@dataclass
class DOSW1Config:
    """Configuration for one DOSW1 dual-arm robot instance."""

    robot_url: str = "localhost"
    left_arm_port: int = 50051
    right_arm_port: int = 50053
    left_lead_port: int = 50050
    right_lead_port: int = 50052

    camera_serials: Optional[list[str]] = None
    camera_names: list[str] = field(
        default_factory=lambda: ["cam_front", "cam_left", "cam_right"]
    )
    enable_camera_player: bool = True
    is_dummy: bool = False

    left_reset_joint: list[float] = field(
        default_factory=lambda: [-0.75, 0.0, 0.0, 1.57, 0.0, -1.57]
    )
    right_reset_joint: list[float] = field(
        default_factory=lambda: [0.75, 0.0, 0.0, -1.57, 0.0, 1.57]
    )
    left_reset_gripper: float = 0.0
    right_reset_gripper: float = 0.0

    gripper_width_min: float = 0.0
    gripper_width_max: float = 0.07

    joint_limit_min: np.ndarray = field(
        default_factory=lambda: np.full(NUM_JOINTS, -3.14)
    )
    joint_limit_max: np.ndarray = field(
        default_factory=lambda: np.full(NUM_JOINTS, 3.14)
    )

    max_joint_delta: float = float("inf")
    action_scale: float = 1.0

    left_ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf)
    )
    left_ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf)
    )
    right_ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf)
    )
    right_ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf)
    )

    step_frequency: float = 30.0
    max_num_steps: int = 1000

    enable_human_in_loop: bool = False
    manual_episode_control_only: bool = False
    gripper_factor: float = 0.07 / 0.048
    gripper_teleop_scale: float = 5.0

    save_video_path: Optional[str] = None


class DOSW1Env(gym.Env):
    """Dual-arm DOSW1 gymnasium environment with optional human-in-the-loop."""

    metadata = {"render_modes": []}
    supports_relative_frame = False
    supports_leader_follower_keyboard_intervention = True

    def __init__(
        self,
        config: DOSW1Config,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[DOSW1HWInfo],
        env_idx: int,
    ) -> None:
        self._logger = get_logger()
        self.config = config
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self.sdk: DOSW1SDKAdapter | None = None
        if not config.is_dummy:
            self._apply_hardware_info(hardware_info)
            self.sdk = DOSW1SDKAdapter(config)
            self.sdk.connect()
            self._go_to_home()
            time.sleep(1.0)

        self.robot_state = DOSW1RobotState()
        self._num_steps = 0

        self.control_mode = ControlMode.MODEL
        self.in_free_teleop = False
        self.start_episode_requested = False
        self._keyboard = None
        self._keyboard_event_callback: Callable[[bool], object] | None = None
        self._teleop_init_lead_left: np.ndarray | None = None
        self._teleop_init_lead_right: np.ndarray | None = None
        self._teleop_init_follow_left: np.ndarray | None = None
        self._teleop_init_follow_right: np.ndarray | None = None
        self.teleop_target_left_gripper: float | None = None
        self.manual_done: bool = False
        self._leader_follow_enabled: bool = False
        if config.enable_human_in_loop:
            self._keyboard = KeyboardListener()
            self.in_free_teleop = True
            self._leader_follow_enabled = True

        self._init_action_obs_spaces()

        self._cameras: list[BaseCamera] = []
        if not config.is_dummy:
            self._open_cameras()
        self._camera_player = VideoPlayer(config.enable_camera_player)

        if not config.is_dummy:
            self.robot_state = self.sdk.get_state()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        joint_reset: bool = False,
    ) -> tuple[dict, dict]:
        if self.config.is_dummy:
            return self._get_observation(), {}

        options = options or {}
        skip_wait_for_start = bool(options.get("skip_wait_for_start", False))

        if self.config.enable_human_in_loop:
            self.in_free_teleop = True
            self.start_episode_requested = False
            self._set_leader_follow_enabled(
                enabled=True, source="reset_enter_free_teleop"
            )
            if skip_wait_for_start:
                # Caller explicitly asked not to block on the 's' key, e.g.
                # the final reset in collect_real_data.py after the last episode.
                self._logger.info(
                    "[DOSW1Env] Skipping free-teleop start wait "
                    "(options.skip_wait_for_start=True)."
                )
            else:
                self._logger.info(
                    "[DOSW1Env] FreeTeleop mode active. "
                    "Move arms freely via leader arm. Press 's' to start episode."
                )
                self._free_teleop_loop()
            self.snapshot_teleop_init()
            manual_episode_control_only = bool(
                getattr(self.config, "manual_episode_control_only", False)
            )
            next_mode = (
                ControlMode.TELEOP if manual_episode_control_only else ControlMode.MODEL
            )
            self.set_control_mode(next_mode, source="reset_after_start_key")
        else:
            self._go_to_home()
            self.set_control_mode(ControlMode.MODEL, source="reset_no_human_in_loop")
        self._num_steps = 0
        self.manual_done = False
        self.robot_state = self.sdk.get_state()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        t0 = time.time()
        action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)

        if self.config.is_dummy:
            self._num_steps += 1
            obs = self._get_observation()
            truncated = self._num_steps >= self.config.max_num_steps
            reward = self._calc_step_reward(obs, gripper_changed=False)
            return obs, reward, False, truncated, {"control_mode": 0}

        prev_left_gripper = self.robot_state.left_gripper
        prev_right_gripper = self.robot_state.right_gripper
        actual_action = self._dispatch_action(action)
        self._num_steps += 1

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1.0 / self.config.step_frequency - elapsed))

        self.robot_state = self.sdk.get_state()
        obs = self._get_observation()
        gripper_changed = (
            abs(self.robot_state.left_gripper - prev_left_gripper) > 1e-6
            or abs(self.robot_state.right_gripper - prev_right_gripper) > 1e-6
        )
        reward = self._calc_step_reward(obs, gripper_changed=gripper_changed)
        terminated = bool(self.manual_done)
        truncated = self._num_steps >= self.config.max_num_steps

        info: dict = {
            "control_mode": self.control_mode.value,
            "manual_done": self.manual_done,
            "success": bool(self.manual_done),
        }
        if self.control_mode == ControlMode.TELEOP:
            info["intervene_action"] = actual_action
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_cameras()
        if self._keyboard is not None:
            listener = getattr(self._keyboard, "listener", None)
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
            self._keyboard = None
        if self.sdk is not None:
            self.sdk.disconnect()

    def set_keyboard_event_callback(
        self, callback: Callable[[bool], object] | None
    ) -> None:
        self._keyboard_event_callback = callback

    @property
    def task_description(self) -> str:
        return "Perform the DOSW1 dual-arm manipulation task."

    def set_control_mode(self, mode: ControlMode, *, source: str = "unknown") -> None:
        self.control_mode = mode
        self._set_leader_follow_enabled(
            enabled=bool(self.in_free_teleop or mode == ControlMode.TELEOP),
            source=f"{source}:{getattr(mode, 'name', mode)}",
        )

    def _set_leader_follow_enabled(self, *, enabled: bool, source: str) -> None:
        enabled = bool(enabled)
        self._leader_follow_enabled = enabled
        if self.sdk is not None:
            set_enabled = getattr(self.sdk, "set_leader_arm_enabled", None)
            if callable(set_enabled):
                try:
                    set_enabled(enabled)
                except Exception:
                    self._logger.exception(
                        "[DOSW1Env] Failed to toggle leader follow to %s",
                        enabled,
                    )

    def _dispatch_action(self, policy_action: np.ndarray) -> np.ndarray:
        if self.control_mode == ControlMode.MODEL:
            return self._execute_model_action(policy_action)
        if self.control_mode == ControlMode.PAUSE:
            return self._execute_pause_action()
        if self.control_mode == ControlMode.TELEOP:
            return self._execute_teleop_action()
        return policy_action

    def _clip_gripper_width(self, width: float) -> float:
        return float(
            np.clip(
                width,
                float(self.config.gripper_width_min),
                float(self.config.gripper_width_max),
            )
        )

    def _clip_joint_to_ee_safety_box(
        self,
        current_joint: np.ndarray,
        target_joint: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """Clip target joints so the resulting ee_pose stays within the safety box.

        Uses binary search along the current -> target line in joint space,
        checking ee_pose via FK at each midpoint.
        """
        cfg = self.config
        if side == "left":
            lo_limit = np.asarray(cfg.left_ee_pose_limit_min, dtype=np.float64)
            hi_limit = np.asarray(cfg.left_ee_pose_limit_max, dtype=np.float64)
        else:
            lo_limit = np.asarray(cfg.right_ee_pose_limit_min, dtype=np.float64)
            hi_limit = np.asarray(cfg.right_ee_pose_limit_max, dtype=np.float64)

        if np.all(np.isinf(lo_limit)) and np.all(np.isinf(hi_limit)):
            return target_joint

        ee = self.sdk.forward_kinematics(target_joint.tolist())
        if self._ee_in_box(ee, lo_limit, hi_limit):
            return target_joint

        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            interp = current_joint + mid * (target_joint - current_joint)
            ee = self.sdk.forward_kinematics(interp.tolist())
            if self._ee_in_box(ee, lo_limit, hi_limit):
                lo = mid
            else:
                hi = mid

        return current_joint + lo * (target_joint - current_joint)

    @staticmethod
    def _ee_in_box(ee: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
        n = min(len(lo), len(ee))
        return bool(np.all(ee[:n] >= lo[:n]) and np.all(ee[:n] <= hi[:n]))

    def _execute_model_action(self, action: np.ndarray) -> np.ndarray:
        cfg = self.config
        cur_left = self.robot_state.left_joint_positions
        cur_right = self.robot_state.right_joint_positions

        left_target = cur_left + cfg.action_scale * (action[:6] - cur_left)
        right_target = cur_right + cfg.action_scale * (action[7:13] - cur_right)

        left_target = np.clip(
            left_target,
            cur_left - cfg.max_joint_delta,
            cur_left + cfg.max_joint_delta,
        )
        right_target = np.clip(
            right_target,
            cur_right - cfg.max_joint_delta,
            cur_right + cfg.max_joint_delta,
        )

        left_joint = np.clip(left_target, cfg.joint_limit_min, cfg.joint_limit_max)
        right_joint = np.clip(right_target, cfg.joint_limit_min, cfg.joint_limit_max)

        left_joint = self._clip_joint_to_ee_safety_box(
            cur_left, left_joint, side="left"
        )
        right_joint = self._clip_joint_to_ee_safety_box(
            cur_right, right_joint, side="right"
        )

        left_gripper = self._clip_gripper_width(float(action[6]))
        right_gripper = self._clip_gripper_width(float(action[13]))

        self.sdk.left_go_joint(left_joint.tolist(), left_gripper)
        self.sdk.right_go_joint(right_joint.tolist(), right_gripper)

        actual = np.empty(ACTION_DIM, dtype=np.float64)
        actual[:6] = left_joint
        actual[6] = left_gripper
        actual[7:13] = right_joint
        actual[13] = right_gripper
        return actual

    def _execute_pause_action(self) -> np.ndarray:
        state = self.robot_state
        actual = np.empty(ACTION_DIM, dtype=np.float64)
        actual[:6] = state.left_joint_positions
        actual[6] = state.left_gripper
        actual[7:13] = state.right_joint_positions
        actual[13] = state.right_gripper
        return actual

    def snapshot_teleop_init(self) -> None:
        self._teleop_init_lead_left = self.sdk.get_left_lead_joint().copy()
        self._teleop_init_lead_right = self.sdk.get_right_lead_joint().copy()
        self._teleop_init_follow_left = self.sdk.get_left_joint().copy()
        self._teleop_init_follow_right = self.sdk.get_right_joint().copy()

    def _compute_teleop_command(
        self,
        cur_left: np.ndarray,
        cur_right: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Compute teleop target joints/grippers from leader-arm deltas.

        Returns (left_joint, left_gripper, right_joint, right_gripper).
        """
        cfg = self.config
        lead_left = self.sdk.get_left_lead_joint()
        lead_right = self.sdk.get_right_lead_joint()
        init_lead_left = self._teleop_init_lead_left
        init_lead_right = self._teleop_init_lead_right
        init_follow_left = self._teleop_init_follow_left
        init_follow_right = self._teleop_init_follow_right

        gripper_scale = cfg.gripper_teleop_scale * cfg.gripper_factor

        delta_left_joint = lead_left[:6] - init_lead_left[:6]
        delta_left_gripper = gripper_scale * (lead_left[6] - init_lead_left[6])
        left_joint = np.clip(
            init_follow_left[:6] + delta_left_joint,
            cfg.joint_limit_min,
            cfg.joint_limit_max,
        )
        left_joint = np.clip(
            left_joint,
            cur_left - cfg.max_joint_delta,
            cur_left + cfg.max_joint_delta,
        )
        left_joint = self._clip_joint_to_ee_safety_box(
            cur_left, left_joint, side="left"
        )
        left_gripper = self._clip_gripper_width(
            float(init_follow_left[6] + delta_left_gripper)
        )

        delta_right_joint = lead_right[:6] - init_lead_right[:6]
        delta_right_gripper = gripper_scale * (lead_right[6] - init_lead_right[6])
        right_joint = np.clip(
            init_follow_right[:6] + delta_right_joint,
            cfg.joint_limit_min,
            cfg.joint_limit_max,
        )
        right_joint = np.clip(
            right_joint,
            cur_right - cfg.max_joint_delta,
            cur_right + cfg.max_joint_delta,
        )
        right_joint = self._clip_joint_to_ee_safety_box(
            cur_right, right_joint, side="right"
        )
        right_gripper = self._clip_gripper_width(
            float(init_follow_right[6] + delta_right_gripper)
        )

        return left_joint, left_gripper, right_joint, right_gripper

    def _execute_teleop_action(self) -> np.ndarray:
        cur_left = self.robot_state.left_joint_positions
        cur_right = self.robot_state.right_joint_positions
        left_joint, left_gripper, right_joint, right_gripper = (
            self._compute_teleop_command(cur_left, cur_right)
        )

        self.sdk.left_go_joint(left_joint.tolist(), left_gripper)
        self.sdk.right_go_joint(right_joint.tolist(), right_gripper)
        self.teleop_target_left_gripper = left_gripper

        actual = np.empty(ACTION_DIM, dtype=np.float64)
        actual[:6] = left_joint
        actual[6] = left_gripper
        actual[7:13] = right_joint
        actual[13] = right_gripper
        return actual

    _FREE_TELEOP_LOG_INTERVAL_S = 10.0

    def _free_teleop_loop(self) -> None:
        self.snapshot_teleop_init()
        last_log = time.time()
        while True:
            self._poll_keyboard_event(reset_phase=True)
            if self.start_episode_requested:
                self.start_episode_requested = False
                self.in_free_teleop = False
                break
            now = time.time()
            if now - last_log >= self._FREE_TELEOP_LOG_INTERVAL_S:
                self._logger.info("[DOSW1Env] FreeTeleop waiting for 's' key")
                last_log = now
            self._forward_leader_to_follower()
            time.sleep(1.0 / self.config.step_frequency)

    def _poll_keyboard_event(self, reset_phase: bool = False) -> None:
        if self._keyboard_event_callback is not None:
            self._keyboard_event_callback(reset_phase=reset_phase)
            return

        # Fallback when wrapper is disabled: keep reset-phase "s to start".
        if (
            reset_phase
            and self._keyboard is not None
            and self._keyboard.get_key() == "s"
        ):
            self.start_episode_requested = True

    def _forward_leader_to_follower(self) -> None:
        if not self._leader_follow_enabled:
            return
        cur_left = self.sdk.get_left_joint()[:6]
        cur_right = self.sdk.get_right_joint()[:6]
        left_joint, left_gripper, right_joint, right_gripper = (
            self._compute_teleop_command(cur_left, cur_right)
        )
        self.sdk.left_go_joint(left_joint.tolist(), left_gripper)
        self.sdk.right_go_joint(right_joint.tolist(), right_gripper)

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self.observation_space.sample()

        state = {
            "left_joint_positions": self.robot_state.left_joint_positions.copy(),
            "left_gripper": np.array([self.robot_state.left_gripper], dtype=np.float64),
            "right_joint_positions": self.robot_state.right_joint_positions.copy(),
            "right_gripper": np.array(
                [self.robot_state.right_gripper],
                dtype=np.float64,
            ),
        }
        return copy.deepcopy({"state": state, "frames": self._get_camera_frames()})

    def _calc_step_reward(self, obs: dict, gripper_changed: bool = False) -> float:
        del obs, gripper_changed
        return 0.0

    def _init_action_obs_spaces(self) -> None:
        camera_names = self.effective_camera_names()
        gripper_low = float(self.config.gripper_width_min)
        gripper_high = float(self.config.gripper_width_max)
        action_low = np.full(ACTION_DIM, -np.pi, dtype=np.float32)
        action_high = np.full(ACTION_DIM, np.pi, dtype=np.float32)
        action_low[6] = action_low[13] = gripper_low
        action_high[6] = action_high[13] = gripper_high
        self.action_space = gym.spaces.Box(low=action_low, high=action_high)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "left_joint_positions": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(NUM_JOINTS,),
                        ),
                        "left_gripper": gym.spaces.Box(
                            gripper_low,
                            gripper_high,
                            shape=(1,),
                        ),
                        "right_joint_positions": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(NUM_JOINTS,),
                        ),
                        "right_gripper": gym.spaces.Box(
                            gripper_low,
                            gripper_high,
                            shape=(1,),
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0,
                            255,
                            shape=(IMAGE_H, IMAGE_W, 3),
                            dtype=np.uint8,
                        )
                        for name in camera_names
                    }
                ),
            }
        )

    def _go_to_home(self) -> None:
        self.sdk.left_go_joint(
            self.config.left_reset_joint,
            self.config.left_reset_gripper,
            interp=True,
        )
        self.sdk.right_go_joint(
            self.config.right_reset_joint,
            self.config.right_reset_gripper,
            interp=True,
        )
        time.sleep(3.0)

    def effective_camera_names(self) -> list[str]:
        serials = self.config.camera_serials or []
        names = self.config.camera_names or []
        return names[: len(serials)] if serials else names

    def _open_cameras(self) -> None:
        serials = self.config.camera_serials or self._discover_camera_serials()
        self.config.camera_serials = list(serials)
        names = self.config.camera_names or []
        for index, serial in enumerate(serials):
            name = names[index] if index < len(names) else f"cam_{index}"
            camera = create_camera(CameraInfo(name=name, serial_number=serial))
            camera.open()
            self._cameras.append(camera)

    def _close_cameras(self) -> None:
        for camera in self._cameras:
            camera.close()
        self._cameras.clear()

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        display_frames: dict[str, np.ndarray] = {}
        for camera in self._cameras:
            frame_rgb = camera.get_frame()
            height, width = frame_rgb.shape[:2]
            crop = min(height, width)
            start_x = (width - crop) // 2
            start_y = (height - crop) // 2
            cropped = frame_rgb[start_y : start_y + crop, start_x : start_x + crop]
            resized = cv2.resize(cropped, (IMAGE_W, IMAGE_H))
            frames[camera.name] = resized[..., ::-1]
            display_frames[camera.name] = resized
        self._camera_player.put_frame(display_frames)
        return frames

    @staticmethod
    def _discover_camera_serials() -> list[str]:
        try:
            import pyrealsense2 as rs
        except ImportError:
            return []

        serials: list[str] = []
        for device in rs.context().devices:
            serials.append(device.get_info(rs.camera_info.serial_number))
        return serials

    def _apply_hardware_info(self, hardware_info: Optional[DOSW1HWInfo]) -> None:
        if hardware_info is None:
            return
        assert isinstance(hardware_info, DOSW1HWInfo), (
            f"hardware_info must be DOSW1HWInfo, got {type(hardware_info)}."
        )
        hw = hardware_info.config
        if hw.camera_serials:
            self.config.camera_serials = list(hw.camera_serials)
        if hw.robot_url and str(hw.robot_url).strip():
            self.config.robot_url = str(hw.robot_url).strip()
        for attr in (
            "left_arm_port",
            "right_arm_port",
            "left_lead_port",
            "right_lead_port",
        ):
            value = getattr(hw, attr, None)
            if value is not None:
                setattr(self.config, attr, int(value))
