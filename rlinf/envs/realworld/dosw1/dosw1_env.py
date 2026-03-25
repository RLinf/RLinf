"""DOSW1 dual-arm gymnasium environment with human-in-the-loop support."""

from __future__ import annotations

import copy
import enum
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .dosw1_data_recorder import DOSW1DataRecorder
from .dosw1_robot_state import DOSW1RobotState
from .dosw1_sdk import DOSW1SDKAdapter


class ControlMode(enum.IntEnum):
    """Data-collection tag written with each recorded transition."""

    MODEL = 0
    PAUSE = 1
    TELEOP = 2


_NUM_JOINTS = 6
_ACTION_DIM = 14
_IMAGE_H, _IMAGE_W = 128, 128


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
        default_factory=lambda: ["cam_left", "cam_right", "cam_front"]
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
        default_factory=lambda: np.full(_NUM_JOINTS, -3.14)
    )
    joint_limit_max: np.ndarray = field(
        default_factory=lambda: np.full(_NUM_JOINTS, 3.14)
    )

    step_frequency: float = 30.0
    max_num_steps: int = 1000

    enable_human_in_loop: bool = False
    gripper_factor: float = 0.07 / 0.048
    gripper_teleop_scale: float = 5.0

    enable_data_persistence: bool = False
    persist_data_dir: str = "./dosw1_data"

    save_video_path: Optional[str] = None


class DOSW1Env(gym.Env):
    """Dual-arm DOSW1 gymnasium environment with optional human-in-the-loop."""

    metadata = {"render_modes": []}
    supports_relative_frame = False

    def __init__(
        self,
        config: DOSW1Config,
        worker_info: Optional[WorkerInfo],
        hardware_info,
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

        self._apply_hardware_info(hardware_info)

        self._sdk: DOSW1SDKAdapter | None = None
        if not config.is_dummy:
            self._sdk = DOSW1SDKAdapter(config)
            self._sdk.connect()
            self._go_to_home()
            time.sleep(1.0)

        self._robot_state = DOSW1RobotState()
        self._num_steps = 0

        self._control_mode = ControlMode.MODEL
        self._in_free_teleop = False
        self._keyboard = None
        self._teleop_init_lead_left: np.ndarray | None = None
        self._teleop_init_lead_right: np.ndarray | None = None
        self._teleop_init_follow_left: np.ndarray | None = None
        self._teleop_init_follow_right: np.ndarray | None = None
        if config.enable_human_in_loop:
            from rlinf.envs.realworld.common.keyboard.keyboard_listener import (
                KeyboardListener,
            )

            self._keyboard = KeyboardListener()
            self._in_free_teleop = True

        self._recorder: DOSW1DataRecorder | None = None
        if config.enable_data_persistence:
            self._recorder = DOSW1DataRecorder(config.persist_data_dir)

        self._init_action_obs_spaces()

        self._cameras: list[Camera] = []
        if not config.is_dummy:
            self._open_cameras()
        self._camera_player = VideoPlayer(config.enable_camera_player)

        if not config.is_dummy:
            self._robot_state = self._sdk.get_state()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        joint_reset: bool = False,
    ) -> tuple[dict, dict]:
        if self.config.is_dummy:
            return self._get_observation(), {}

        if self._recorder is not None:
            self._recorder.end_episode()

        if self.config.enable_human_in_loop:
            self._in_free_teleop = True
            self._logger.info(
                "[DOSW1Env] FreeTeleop mode active. "
                "Move arms freely via leader arm. Press 's' to start episode."
            )
            self._free_teleop_loop()

        if not self.config.enable_human_in_loop:
            self._go_to_home()
        self._num_steps = 0
        self._control_mode = ControlMode.MODEL
        self._robot_state = self._sdk.get_state()

        if self._recorder is not None:
            self._recorder.start_episode()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        t0 = time.time()
        action = np.asarray(action, dtype=np.float64).reshape(_ACTION_DIM)

        if self.config.is_dummy:
            self._num_steps += 1
            obs = self._get_observation()
            truncated = self._num_steps >= self.config.max_num_steps
            reward = self._calc_step_reward(obs, gripper_changed=False)
            return obs, reward, False, truncated, {"control_mode": 0}

        truncated_by_free = False
        if self.config.enable_human_in_loop:
            truncated_by_free = self._handle_keyboard()

        if truncated_by_free:
            obs = self._get_observation()
            if self._recorder is not None:
                self._recorder.end_episode()
            return obs, 0.0, False, True, {"control_mode": self._control_mode.value}

        prev_left_gripper = self._robot_state.left_gripper
        prev_right_gripper = self._robot_state.right_gripper
        actual_action = self._dispatch_action(action)
        self._num_steps += 1

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1.0 / self.config.step_frequency - elapsed))

        self._robot_state = self._sdk.get_state()
        obs = self._get_observation()
        gripper_changed = (
            abs(self._robot_state.left_gripper - prev_left_gripper) > 1e-6
            or abs(self._robot_state.right_gripper - prev_right_gripper) > 1e-6
        )
        reward = self._calc_step_reward(obs, gripper_changed=gripper_changed)
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps

        if self._recorder is not None:
            self._recorder.record_step(
                obs=obs,
                action=actual_action,
                reward=reward,
                control_mode=self._control_mode.value,
                timestamp=time.time(),
            )

        info: dict = {"control_mode": self._control_mode.value}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._recorder is not None:
            self._recorder.end_episode()
        self._close_cameras()
        if self._keyboard is not None:
            try:
                self._keyboard.listener.stop()
            except Exception:
                pass
            self._keyboard = None
        if self._sdk is not None:
            self._sdk.disconnect()

    @property
    def task_description(self) -> str:
        return "Perform the DOSW1 dual-arm manipulation task."

    def _dispatch_action(self, policy_action: np.ndarray) -> np.ndarray:
        if self._control_mode == ControlMode.MODEL:
            return self._execute_model_action(policy_action)
        if self._control_mode == ControlMode.PAUSE:
            return self._execute_pause_action()
        if self._control_mode == ControlMode.TELEOP:
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

    def _execute_model_action(self, action: np.ndarray) -> np.ndarray:
        left_joint = np.clip(
            action[:6],
            self.config.joint_limit_min,
            self.config.joint_limit_max,
        )
        left_gripper = self._clip_gripper_width(float(action[6]))
        right_joint = np.clip(
            action[7:13],
            self.config.joint_limit_min,
            self.config.joint_limit_max,
        )
        right_gripper = self._clip_gripper_width(float(action[13]))

        self._sdk.left_go_joint(left_joint.tolist(), left_gripper)
        self._sdk.right_go_joint(right_joint.tolist(), right_gripper)

        actual = np.empty(_ACTION_DIM, dtype=np.float64)
        actual[:6] = left_joint
        actual[6] = left_gripper
        actual[7:13] = right_joint
        actual[13] = right_gripper
        return actual

    def _execute_pause_action(self) -> np.ndarray:
        state = self._robot_state
        actual = np.empty(_ACTION_DIM, dtype=np.float64)
        actual[:6] = state.left_joint_positions
        actual[6] = state.left_gripper
        actual[7:13] = state.right_joint_positions
        actual[13] = state.right_gripper
        return actual

    def _snapshot_teleop_init(self) -> None:
        self._teleop_init_lead_left = self._sdk.get_left_lead_joint().copy()
        self._teleop_init_lead_right = self._sdk.get_right_lead_joint().copy()
        self._teleop_init_follow_left = self._sdk.get_left_joint().copy()
        self._teleop_init_follow_right = self._sdk.get_right_joint().copy()

    def _execute_teleop_action(self) -> np.ndarray:
        cfg = self.config
        lead_left = self._sdk.get_left_lead_joint()
        lead_right = self._sdk.get_right_lead_joint()
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
        left_gripper = self._clip_gripper_width(float(init_follow_left[6] + delta_left_gripper))

        delta_right_joint = lead_right[:6] - init_lead_right[:6]
        delta_right_gripper = gripper_scale * (lead_right[6] - init_lead_right[6])
        right_joint = np.clip(
            init_follow_right[:6] + delta_right_joint,
            cfg.joint_limit_min,
            cfg.joint_limit_max,
        )
        right_gripper = self._clip_gripper_width(
            float(init_follow_right[6] + delta_right_gripper)
        )

        self._sdk.left_go_joint(left_joint.tolist(), left_gripper)
        self._sdk.right_go_joint(right_joint.tolist(), right_gripper)

        actual = np.empty(_ACTION_DIM, dtype=np.float64)
        actual[:6] = left_joint
        actual[6] = left_gripper
        actual[7:13] = right_joint
        actual[13] = right_gripper
        return actual

    def _free_teleop_loop(self) -> None:
        self._snapshot_teleop_init()
        while True:
            key = self._keyboard.get_key() if self._keyboard else None
            if key == "s":
                self._in_free_teleop = False
                break
            self._forward_leader_to_follower()
            time.sleep(1.0 / self.config.step_frequency)

    def _forward_leader_to_follower(self) -> None:
        cfg = self.config
        lead_left = self._sdk.get_left_lead_joint()
        lead_right = self._sdk.get_right_lead_joint()
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
        left_gripper = self._clip_gripper_width(float(init_follow_left[6] + delta_left_gripper))

        delta_right_joint = lead_right[:6] - init_lead_right[:6]
        delta_right_gripper = gripper_scale * (lead_right[6] - init_lead_right[6])
        right_joint = np.clip(
            init_follow_right[:6] + delta_right_joint,
            cfg.joint_limit_min,
            cfg.joint_limit_max,
        )
        right_gripper = self._clip_gripper_width(
            float(init_follow_right[6] + delta_right_gripper)
        )

        self._sdk.left_go_joint(left_joint.tolist(), left_gripper)
        self._sdk.right_go_joint(right_joint.tolist(), right_gripper)

    def _handle_keyboard(self) -> bool:
        if self._keyboard is None:
            return False

        key = self._keyboard.get_key()
        if key is None:
            return False

        if key == "r":
            self._logger.info("[DOSW1Env] -> FreeTeleop (episode aborted)")
            self._in_free_teleop = True
            return True

        if key == "p" and self._control_mode in (ControlMode.MODEL, ControlMode.TELEOP):
            self._logger.info("[DOSW1Env] %s -> PAUSE", self._control_mode.name)
            self._control_mode = ControlMode.PAUSE
        elif key == "t" and self._control_mode == ControlMode.PAUSE:
            self._logger.info("[DOSW1Env] PAUSE -> TELEOP")
            self._snapshot_teleop_init()
            self._control_mode = ControlMode.TELEOP
        elif key == "m" and self._control_mode == ControlMode.PAUSE:
            self._logger.info("[DOSW1Env] PAUSE -> MODEL")
            self._control_mode = ControlMode.MODEL

        return False

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self.observation_space.sample()

        state = {
            "left_joint_positions": self._robot_state.left_joint_positions.copy(),
            "left_gripper": np.array([self._robot_state.left_gripper], dtype=np.float64),
            "right_joint_positions": self._robot_state.right_joint_positions.copy(),
            "right_gripper": np.array(
                [self._robot_state.right_gripper],
                dtype=np.float64,
            ),
        }
        return copy.deepcopy({"state": state, "frames": self._get_camera_frames()})

    def _calc_step_reward(self, obs: dict, gripper_changed: bool = False) -> float:
        del obs, gripper_changed
        return 0.0

    def _init_action_obs_spaces(self) -> None:
        camera_names = self._effective_camera_names()
        gripper_low = float(self.config.gripper_width_min)
        gripper_high = float(self.config.gripper_width_max)
        action_low = np.full(_ACTION_DIM, -np.pi, dtype=np.float32)
        action_high = np.full(_ACTION_DIM, np.pi, dtype=np.float32)
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
                            shape=(_NUM_JOINTS,),
                        ),
                        "left_gripper": gym.spaces.Box(
                            gripper_low,
                            gripper_high,
                            shape=(1,),
                        ),
                        "right_joint_positions": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(_NUM_JOINTS,),
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
                            shape=(_IMAGE_H, _IMAGE_W, 3),
                            dtype=np.uint8,
                        )
                        for name in camera_names
                    }
                ),
            }
        )

    def _go_to_home(self) -> None:
        self._sdk.left_go_joint(
            self.config.left_reset_joint,
            self.config.left_reset_gripper,
            interp=True,
        )
        self._sdk.right_go_joint(
            self.config.right_reset_joint,
            self.config.right_reset_gripper,
            interp=True,
        )
        time.sleep(3.0)

    def _effective_camera_names(self) -> list[str]:
        serials = self.config.camera_serials or []
        names = self.config.camera_names or []
        return names[: len(serials)] if serials else names

    def _open_cameras(self) -> None:
        serials = self.config.camera_serials or self._discover_camera_serials()
        self.config.camera_serials = list(serials)
        names = self.config.camera_names or []
        for index, serial in enumerate(serials):
            name = names[index] if index < len(names) else f"cam_{index}"
            camera = Camera(CameraInfo(name=name, serial_number=serial))
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
            resized = cv2.resize(cropped, (_IMAGE_W, _IMAGE_H))
            frames[camera._camera_info.name] = resized[..., ::-1]
            display_frames[camera._camera_info.name] = resized
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

    def _apply_hardware_info(self, hardware_info) -> None:
        if hardware_info is None:
            return
        config = getattr(hardware_info, "config", None)
        if config is not None:
            serials = getattr(config, "camera_serials", None)
            if serials:
                self.config.camera_serials = list(serials)
            robot_url = getattr(config, "robot_url", None)
            if robot_url and str(robot_url).strip():
                self.config.robot_url = str(robot_url).strip()
            for attr in (
                "left_arm_port",
                "right_arm_port",
                "left_lead_port",
                "right_lead_port",
            ):
                value = getattr(config, attr, None)
                if value is not None:
                    setattr(self.config, attr, int(value))
        elif hasattr(hardware_info, "camera_serials") and not self.config.camera_serials:
            self.config.camera_serials = list(hardware_info.camera_serials)
