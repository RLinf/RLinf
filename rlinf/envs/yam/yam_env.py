# Copyright 2026 Shirui Chen
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

"""YAM bimanual robot environment for RLinf.

This module provides the ``YAMEnv`` gymnasium wrapper around the
``yam_realtime`` robot stack.  The environment is driven on the robot
controller node; the RLinf rollout workers on GPU nodes communicate with it
through the normal EnvWorker → Channel pipeline.

Architecture
~~~~~~~~~~~~
::

    GPU node (actor/rollout)   <──gRPC channel──>   Robot node (env worker)
                                                          │
                                                     YAMEnv (gym.Env)
                                                          │
                                               yam_realtime.RobotEnv (dm_env)
                                                     ┌────┴────┐
                                                  left arm   right arm
                                                  cameras    cameras

Observation space
~~~~~~~~~~~~~~~~~
``states``        - concatenated joint positions of both arms (14D).
``main_images``   - uint8 RGB image (H, W, 3) from the primary camera.
``wrist_images``  - optional uint8 RGB image stack from wrist cameras.
``extra_view_images`` - optional uint8 RGB image stack from additional cameras.

Action space
~~~~~~~~~~~~
14D float32 vector: 7 joint positions for left arm followed by 7 for right arm.
"""

import time
from contextlib import contextmanager
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo, YAMHWInfo
from rlinf.utils.logging import get_logger

# Number of joints per arm (YAM has 6 DoF + 1 gripper)
_JOINTS_PER_ARM = 7
_STATE_DIM = _JOINTS_PER_ARM * 2  # left + right

# Default camera resolution (can be overridden via cfg)
_DEFAULT_IMG_H = 224
_DEFAULT_IMG_W = 224


class YAMEnv(gym.Env):
    """Gymnasium wrapper for a YAM bimanual robot environment.

    Parameters
    ----------
    cfg:
        Hydra/OmegaConf DictConfig containing at least:
        - ``init_params.id`` (str): task identifier, e.g. "yam_pick_and_place"
        - ``control_rate_hz`` (float, default 10): environment step rate
        - ``max_episode_steps`` (int, default 100): episode length
        - ``is_dummy`` (bool, default False): when True runs in simulation mode
          without connecting to real hardware
        - ``img_height``, ``img_width`` (int): camera resolution
        - ``main_camera`` (str): key of the primary camera in observations
    num_envs:
        Must be 1 (each worker manages one real environment instance).
    seed_offset:
        Unused – kept for API compatibility with other RLinf envs.
    total_num_processes:
        Total number of env workers across all nodes.
    worker_info:
        WorkerInfo from the RLinf scheduler containing hardware information.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Optional[WorkerInfo],
    ):
        assert num_envs == 1, (
            f"YAMEnv supports exactly 1 environment per worker, got {num_envs}."
        )
        self._logger = get_logger()
        self.cfg = cfg
        self.num_envs = num_envs
        self.worker_info = worker_info

        self._is_dummy = bool(cfg.get("is_dummy", False))
        self._max_episode_steps = int(cfg.get("max_episode_steps", 100))
        self._control_rate_hz = float(cfg.get("control_rate_hz", 10.0))
        self._img_h = int(cfg.get("img_height", _DEFAULT_IMG_H))
        self._img_w = int(cfg.get("img_width", _DEFAULT_IMG_W))
        self._main_camera = str(cfg.get("main_camera", "top_camera"))
        self.main_image_key = self._main_camera
        self._wrist_camera_patterns = self._normalize_camera_patterns(
            cfg.get("wrist_cameras", None)
        )
        camera_cfgs_raw = cfg.get("camera_cfgs", None)
        self._camera_names = list(camera_cfgs_raw.keys()) if camera_cfgs_raw else []
        self._main_camera_name: Optional[str] = None
        self._wrist_camera_names: list[str] = []
        self._extra_view_camera_names: list[str] = []

        self.auto_reset = bool(cfg.get("auto_reset", False))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", True))
        self._move_to_reset_pose_on_reset = bool(
            cfg.get("move_to_reset_pose_on_reset", True)
        )
        self._reset_motion_steps = max(1, int(cfg.get("reset_motion_steps", 50)))
        self._reset_motion_duration_s = max(
            0.0, float(cfg.get("reset_motion_duration_s", 2.0))
        )
        self._reset_motion_tolerance = max(
            0.0, float(cfg.get("reset_motion_tolerance", 0.05))
        )
        self._reset_motion_settle_timeout_s = max(
            0.0, float(cfg.get("reset_motion_settle_timeout_s", 0.75))
        )
        self._reset_motion_poll_interval_s = max(
            0.0, float(cfg.get("reset_motion_poll_interval_s", 0.05))
        )
        self._configured_reset_joint_positions = self._load_reset_joint_positions(
            cfg.get("reset_joint_positions", None)
        )
        self._reset_joint_positions: dict[str, np.ndarray] = {}

        self._num_steps = 0
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        # Metrics (mirrors RealWorldEnv for compatibility)
        self._init_metrics()

        # Hardware / robot setup
        self._yam_hw_info: Optional[YAMHWInfo] = None
        if worker_info is not None and len(worker_info.hardware_infos) > 0:
            hw = worker_info.hardware_infos[0]
            if isinstance(hw, YAMHWInfo):
                self._yam_hw_info = hw

        self._robot_env = None
        self._dummy_cameras: dict = {}
        if not self._is_dummy:
            self._setup_robot()
        else:
            self._setup_dummy_cameras()
        self._resolve_camera_roles(self._camera_names)
        self._capture_reset_joint_positions()

        # Gym spaces
        obs_space = {
            "states": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_STATE_DIM,), dtype=np.float32
            ),
            "main_images": gym.spaces.Box(
                low=0,
                high=255,
                shape=(self._img_h, self._img_w, 3),
                dtype=np.uint8,
            ),
        }
        if self._wrist_camera_names:
            obs_space["wrist_images"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(len(self._wrist_camera_names), self._img_h, self._img_w, 3),
                dtype=np.uint8,
            )
        if self._extra_view_camera_names:
            obs_space["extra_view_images"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(len(self._extra_view_camera_names), self._img_h, self._img_w, 3),
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(obs_space)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_STATE_DIM,), dtype=np.float32
        )

        # Mutable task description — can be updated mid-episode by a VLM planner.
        self._task_description: str = str(cfg.get("task_description", ""))

        # Episode tracking
        self._is_start = True

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_robot(self):
        """Connect to the YAM robot hardware via yam_realtime.

        Robots are created via ``yam_realtime.utils.launch_utils.initialize_robots``
        using config file paths supplied in ``cfg.robot_cfgs``.  Cameras are
        optional and are created via ``initialize_sensors`` using
        ``cfg.camera_cfgs``.  Both follow the standard yam_realtime config
        format (``_target_`` dicts or YAML file path lists).
        """
        try:
            from omegaconf import OmegaConf
            from yam_realtime.envs.robot_env import RobotEnv
            from yam_realtime.robots.utils import Rate
            from yam_realtime.utils.launch_utils import (
                initialize_robots,
                initialize_sensors,
            )
        except ImportError as e:
            raise ImportError(
                "yam_realtime must be installed to run YAMEnv with real hardware. "
                f"Original error: {e}"
            ) from e

        self._server_processes: list = []

        # Robot configs: mapping of arm name to config file path list(s).
        # E.g. robot_cfgs: {left: ["/path/to/left.yaml"], right: ["/path/to/right.yaml"]}
        robots_cfg_raw = self.cfg.get("robot_cfgs", None)
        if robots_cfg_raw is None:
            raise ValueError(
                "YAMEnv requires 'robot_cfgs' in the env config specifying robot YAML paths. "
                "E.g. robot_cfgs: {left: ['/path/to/left.yaml'], right: ['/path/to/right.yaml']}"
            )
        robots_cfg = OmegaConf.to_container(robots_cfg_raw, resolve=True)

        self._logger.info(f"[YAMEnv] Initializing robots from configs: {robots_cfg}")
        robot_dict = initialize_robots(robots_cfg, self._server_processes)

        # Camera configs (optional): mapping of camera name to CameraNode config dict.
        # E.g. camera_cfgs: {cam_front: {_target_: ..., camera: {_target_: ...,
        #                                 device_path: "/dev/video0"}}}
        camera_cfgs_raw = self.cfg.get("camera_cfgs", None)
        camera_dict: dict = {}
        if camera_cfgs_raw is not None:
            sensors_cfg = {
                "cameras": OmegaConf.to_container(camera_cfgs_raw, resolve=True)
            }
            camera_dict, _ = initialize_sensors(sensors_cfg, self._server_processes)
            self._camera_names = list(camera_dict.keys())
            self._logger.info(
                f"[YAMEnv] Cameras initialized: {list(camera_dict.keys())}"
            )

        rate = Rate(self._control_rate_hz)
        self._robot_env = RobotEnv(
            robot_dict=robot_dict,
            camera_dict=camera_dict,
            control_rate_hz=rate,
            use_joint_state_as_action=False,
        )
        self._logger.info("[YAMEnv] Robot connected.")

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None, reset_state_ids=None, env_idx=None):
        """Reset the environment.

        Returns
        -------
        obs : dict
            Processed observation tensors.
        info : dict
            Empty dict (reserved for future use).
        """
        self._num_steps = 0
        self._elapsed_steps[:] = 0
        self._reset_metrics()
        self._is_start = True

        if self._is_dummy:
            raw_obs = self._dummy_obs()
        else:
            self._move_robots_to_reset_pose()
            raw_obs = self._robot_env.reset()

        return self._wrap_obs(raw_obs), {}

    def _load_reset_joint_positions(
        self, reset_joint_positions
    ) -> dict[str, np.ndarray]:
        """Build optional per-arm reset overrides from config."""
        if reset_joint_positions is None:
            return {}

        if isinstance(reset_joint_positions, DictConfig):
            reset_joint_positions = dict(reset_joint_positions)

        if isinstance(reset_joint_positions, dict):
            targets = {}
            for side, target in reset_joint_positions.items():
                if side not in {"left", "right"}:
                    continue
                target_array = np.asarray(target, dtype=np.float32)
                if target_array.shape != (_JOINTS_PER_ARM,):
                    raise ValueError(
                        f"reset_joint_positions.{side} must have shape "
                        f"({_JOINTS_PER_ARM},), got {target_array.shape}."
                    )
                targets[side] = target_array
            return targets

        target_array = np.asarray(reset_joint_positions, dtype=np.float32)
        if target_array.shape != (_JOINTS_PER_ARM,):
            raise ValueError(
                "reset_joint_positions must be either a 7D joint target list or "
                "a mapping with 'left'/'right' 7D joint target lists."
            )
        return {
            "left": target_array.copy(),
            "right": target_array.copy(),
        }

    def _capture_reset_joint_positions(self) -> None:
        """Capture the startup joint positions and use them as the reset home."""
        if self._is_dummy:
            self._reset_joint_positions = {
                side: target.copy()
                for side, target in self._configured_reset_joint_positions.items()
            }
            return

        assert self._robot_env is not None, "Robot environment is not initialized."
        robot_names = tuple(self._robot_env.get_all_robots().keys())
        reset_joint_positions = {}
        for name in robot_names:
            current_pos = self._read_robot_joint_position(name)
            if current_pos.shape != (_JOINTS_PER_ARM,):
                raise ValueError(
                    f"Robot '{name}' startup joint position must have shape "
                    f"({_JOINTS_PER_ARM},), got {current_pos.shape}."
                )
            reset_joint_positions[name] = current_pos.copy()

        for side, target in self._configured_reset_joint_positions.items():
            reset_joint_positions[side] = target.copy()

        self._reset_joint_positions = reset_joint_positions
        self._logger.info(
            f"[YAMEnv] Captured startup home joint positions: "
            f"{ {side: target.tolist() for side, target in self._reset_joint_positions.items()} }"
        )

    @contextmanager
    def _use_robot_command_futures(self, robot_names: tuple[str, ...]):
        """Temporarily enable future-returning RPC calls for follower clients."""
        assert self._robot_env is not None, "Robot environment is not initialized."
        robots = [self._robot_env.robot(name) for name in robot_names]
        previous_states = []
        try:
            for robot in robots:
                set_use_future = getattr(robot, "set_use_future", None)
                use_future = getattr(robot, "use_future", None)
                if callable(set_use_future) and isinstance(use_future, bool):
                    previous_states.append((robot, use_future))
                    set_use_future(True)
            yield
        finally:
            for robot, previous_state in reversed(previous_states):
                robot.set_use_future(previous_state)

    def _read_robot_joint_position(self, robot_name: str) -> np.ndarray:
        """Read a robot joint vector in the 7D command/state ordering."""
        assert self._robot_env is not None, "Robot environment is not initialized."
        robot = self._robot_env.robot(robot_name)

        observations_getter = getattr(robot, "get_observations", None)
        if callable(observations_getter):
            observations = observations_getter()
            if isinstance(observations, dict):
                components = []
                joint_pos = observations.get("joint_pos")
                if joint_pos is not None:
                    components.append(
                        np.asarray(joint_pos, dtype=np.float32).reshape(-1)
                    )
                gripper_pos = observations.get("gripper_pos")
                if gripper_pos is not None:
                    components.append(
                        np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
                    )
                if components:
                    current_pos = np.concatenate(components)[:_JOINTS_PER_ARM]
                    if current_pos.shape == (_JOINTS_PER_ARM,):
                        return current_pos.copy()

        current_pos = np.asarray(robot.get_joint_pos(), dtype=np.float32).reshape(-1)
        if current_pos.shape != (_JOINTS_PER_ARM,):
            raise ValueError(
                f"Robot '{robot_name}' joint position must have shape "
                f"({_JOINTS_PER_ARM},), got {current_pos.shape}."
            )
        return current_pos.copy()

    def _move_robots_to_reset_pose(self) -> None:
        """Linearly interpolate all connected arms back to their startup home pose."""
        if not self._move_to_reset_pose_on_reset:
            return

        assert self._robot_env is not None, "Robot environment is not initialized."
        robot_names = tuple(self._robot_env.get_all_robots().keys())
        if not robot_names:
            return

        missing_targets = [
            name for name in robot_names if name not in self._reset_joint_positions
        ]
        if missing_targets:
            raise RuntimeError(
                f"[YAMEnv] Missing startup home targets for robots: {missing_targets}."
            )

        current_joint_positions = {
            name: self._read_robot_joint_position(name) for name in robot_names
        }
        sleep_s = self._reset_motion_duration_s / self._reset_motion_steps
        for step_idx in range(self._reset_motion_steps + 1):
            alpha = step_idx / self._reset_motion_steps
            with self._use_robot_command_futures(robot_names):
                for name, current_pos in current_joint_positions.items():
                    target_pos = self._reset_joint_positions[name]
                    interpolated_pos = (1.0 - alpha) * current_pos + alpha * target_pos
                    self._robot_env.robot(name).command_joint_pos(interpolated_pos)
            if sleep_s > 0:
                time.sleep(sleep_s)

        deadline = time.monotonic() + self._reset_motion_settle_timeout_s
        final_joint_positions = {
            name: self._read_robot_joint_position(name) for name in robot_names
        }
        while time.monotonic() <= deadline:
            failed_robots = [
                name
                for name, final_pos in final_joint_positions.items()
                if not np.allclose(
                    final_pos,
                    self._reset_joint_positions[name],
                    atol=self._reset_motion_tolerance,
                    rtol=0.0,
                )
            ]
            if not failed_robots:
                self._logger.info(
                    "[YAMEnv] Startup home pose reached for robots: "
                    f"{list(robot_names)}."
                )
                return

            if self._reset_motion_poll_interval_s > 0:
                time.sleep(self._reset_motion_poll_interval_s)
            final_joint_positions = {
                name: self._read_robot_joint_position(name) for name in robot_names
            }

        failure_details = {
            name: float(
                np.max(
                    np.abs(
                        final_joint_positions[name] - self._reset_joint_positions[name]
                    )
                )
            )
            for name in robot_names
            if not np.allclose(
                final_joint_positions[name],
                self._reset_joint_positions[name],
                atol=self._reset_motion_tolerance,
                rtol=0.0,
            )
        }
        failed_robots = list(failure_details.keys())
        if failed_robots:
            raise RuntimeError(
                "[YAMEnv] Failed to move robots to startup home pose: "
                f"{failed_robots}. Max abs diff: {failure_details}."
            )
        self._logger.info(
            f"[YAMEnv] Startup home pose reached for robots: {list(robot_names)}."
        )

    def step(self, actions=None, auto_reset=True):
        """Execute one environment step.

        Parameters
        ----------
        actions : np.ndarray | torch.Tensor, shape (1, 14) or (14,)
            Concatenated left (7) + right (7) joint positions.
        auto_reset : bool
            If True, automatically reset when the episode ends.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 2:
                actions = actions[0]  # unwrap batch dim

        self._elapsed_steps += 1
        self._num_steps += 1
        truncated = self._elapsed_steps >= self._max_episode_steps

        if self._is_dummy:
            raw_obs = self._dummy_obs()
        else:
            action_dict = self._format_action(actions)
            raw_obs = self._robot_env.step(action_dict)

        reward = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)

        obs = self._wrap_obs(raw_obs)
        infos = self._record_metrics(reward, terminated, np.zeros_like(terminated), {})

        if auto_reset and (np.any(terminated) or np.any(truncated)):
            if self._is_dummy:
                raw_obs = self._dummy_obs()
            else:
                raw_obs = self._robot_env.reset()
            obs = self._wrap_obs(raw_obs)
            self._reset_metrics()
            self._num_steps = 0
            self._elapsed_steps[:] = 0

        return obs, reward, terminated, truncated, infos

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions and return stacked results.

        Follows the same contract as ``RealWorldEnv.chunk_step()``.

        Parameters
        ----------
        chunk_actions : np.ndarray | torch.Tensor
            Shape ``(num_envs, chunk_size, action_dim)``.

        Returns
        -------
        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list
        """
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        chunk_size = chunk_actions.shape[1]

        obs_list = []
        infos_list = []
        raw_chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            raw_chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(
            [
                to_tensor(r) if not isinstance(r, torch.Tensor) else r
                for r in raw_chunk_rewards
            ],
            dim=1,
        )  # [num_envs, chunk_size]
        raw_chunk_terminations = torch.stack(
            [
                to_tensor(t) if not isinstance(t, torch.Tensor) else t
                for t in raw_chunk_terminations
            ],
            dim=1,
        )
        raw_chunk_truncations = torch.stack(
            [
                to_tensor(t) if not isinstance(t, torch.Tensor) else t
                for t in raw_chunk_truncations
            ],
            dim=1,
        )

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def close(self):
        if self._robot_env is not None:
            try:
                self._robot_env.close()
            except Exception:
                pass
        for cam_node in getattr(self, "_dummy_cameras", {}).values():
            try:
                cam_node.close()
            except Exception:
                pass
        self._dummy_cameras = {}
        for proc in getattr(self, "_server_processes", []):
            try:
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()
            except Exception:
                pass

    def enter_zero_torque_mode(self) -> None:
        """Switch connected robots into zero-torque / zero-gravity mode."""
        if self._is_dummy or self._robot_env is None:
            return

        robot_names = tuple(self._robot_env.get_all_robots().keys())
        if not robot_names:
            return

        failed_robots = []
        for name in robot_names:
            robot = self._robot_env.robot(name)
            zero_torque_mode = getattr(robot, "zero_torque_mode", None)
            if not callable(zero_torque_mode):
                failed_robots.append(name)
                continue
            try:
                zero_torque_mode()
            except Exception:
                failed_robots.append(name)

        if failed_robots:
            raise RuntimeError(
                "[YAMEnv] Failed to enter zero-torque mode for robots: "
                f"{failed_robots}."
            )

        self._logger.info(
            f"[YAMEnv] Entered zero-torque mode for robots: {list(robot_names)}."
        )

    # ------------------------------------------------------------------
    # Observation / action helpers
    # ------------------------------------------------------------------

    def _format_action(self, action: np.ndarray) -> dict:
        """Convert flat action vector to YAM robot action dict."""
        left_pos = action[:_JOINTS_PER_ARM]
        right_pos = action[_JOINTS_PER_ARM:]
        return {
            "left": {"pos": left_pos},
            "right": {"pos": right_pos},
        }

    @staticmethod
    def _normalize_camera_patterns(patterns) -> list[str]:
        if patterns is None:
            return []
        if isinstance(patterns, str):
            return [patterns]
        return [str(pattern) for pattern in patterns]

    @staticmethod
    def _name_matches_pattern(name: str, pattern: str) -> bool:
        name_lower = name.lower()
        pattern_lower = pattern.lower()
        return (
            name_lower == pattern_lower
            or name_lower.startswith(pattern_lower)
            or pattern_lower in name_lower
        )

    def _match_camera_patterns(
        self, patterns: list[str], camera_names: list[str], exclude: set[str]
    ) -> list[str]:
        matched: list[str] = []
        for pattern in patterns:
            for camera_name in camera_names:
                if camera_name in exclude or camera_name in matched:
                    continue
                if self._name_matches_pattern(camera_name, pattern):
                    matched.append(camera_name)
        return matched

    @staticmethod
    def _is_wrist_camera(camera_name: str) -> bool:
        camera_name = camera_name.lower()
        return any(
            token in camera_name
            for token in ("wrist", "hand", "gripper", "eye_in_hand", "eef", "tcp")
        )

    def _resolve_camera_roles(self, camera_names: list[str]) -> None:
        if not camera_names:
            self._main_camera_name = None
            self._wrist_camera_names = []
            self._extra_view_camera_names = []
            return

        main_camera_name = None
        for camera_name in camera_names:
            if self._name_matches_pattern(camera_name, self._main_camera):
                main_camera_name = camera_name
                break
        if main_camera_name is None:
            main_camera_name = camera_names[0]

        used_names = {main_camera_name}
        wrist_camera_names = self._match_camera_patterns(
            self._wrist_camera_patterns, camera_names, used_names
        )
        if not wrist_camera_names:
            wrist_camera_names = [
                camera_name
                for camera_name in camera_names
                if camera_name not in used_names and self._is_wrist_camera(camera_name)
            ]
        used_names.update(wrist_camera_names)
        extra_view_camera_names = [
            camera_name for camera_name in camera_names if camera_name not in used_names
        ]

        self._main_camera_name = main_camera_name
        self.main_image_key = main_camera_name
        self._wrist_camera_names = wrist_camera_names
        self._extra_view_camera_names = extra_view_camera_names
        self._logger.info(
            "[YAMEnv] Camera roles resolved: "
            f"main={self._main_camera_name}, "
            f"wrist={self._wrist_camera_names}"
        )

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        if image.shape != (self._img_h, self._img_w, 3):
            import cv2

            image = cv2.resize(image, (self._img_w, self._img_h))
        return image

    def _extract_camera_images(self, raw_obs: dict) -> dict[str, np.ndarray]:
        camera_images: dict[str, np.ndarray] = {}
        for cam_key, cam_data in raw_obs.items():
            if cam_key in {"left", "right"}:
                continue

            image = None
            if isinstance(cam_data, dict):
                images = cam_data.get("images")
                if isinstance(images, dict) and images:
                    if len(images) == 1:
                        image = next(iter(images.values()))
                        if image is not None:
                            camera_images[cam_key] = self._normalize_image(image)
                            continue
                    for image_name, image_value in images.items():
                        if image_value is None:
                            continue
                        derived_name = (
                            image_name
                            if image_name in self._camera_names
                            else f"{cam_key}/{image_name}"
                        )
                        camera_images[derived_name] = self._normalize_image(image_value)
                    continue
                image = cam_data.get("rgb", cam_data.get("color", None))
            elif isinstance(cam_data, np.ndarray):
                image = cam_data

            if image is not None:
                camera_images[cam_key] = self._normalize_image(image)
        return camera_images

    def _wrap_obs(self, raw_obs: dict) -> dict:
        """Convert raw yam_realtime observation dict to RLinf-compatible format.

        Parameters
        ----------
        raw_obs : dict
            Output from ``RobotEnv.get_obs()`` or ``reset()``.

        Returns
        -------
        dict with keys ``states`` (tensor) and ``main_images`` (tensor).
        """
        # --- joint states ---
        left_state = np.zeros(_JOINTS_PER_ARM, dtype=np.float32)
        right_state = np.zeros(_JOINTS_PER_ARM, dtype=np.float32)
        if "left" in raw_obs and raw_obs["left"] is not None:
            jpos = np.asarray(
                raw_obs["left"].get("joint_pos", np.zeros(6)), dtype=np.float32
            )
            gpos = np.asarray(
                raw_obs["left"].get("gripper_pos", np.zeros(1)), dtype=np.float32
            )
            left_state = np.concatenate([jpos, gpos])[:_JOINTS_PER_ARM]
        if "right" in raw_obs and raw_obs["right"] is not None:
            jpos = np.asarray(
                raw_obs["right"].get("joint_pos", np.zeros(6)), dtype=np.float32
            )
            gpos = np.asarray(
                raw_obs["right"].get("gripper_pos", np.zeros(1)), dtype=np.float32
            )
            right_state = np.concatenate([jpos, gpos])[:_JOINTS_PER_ARM]
        states = np.concatenate([left_state, right_state])  # (14,)
        # Add batch dimension for vectorised env compatibility
        states = states[np.newaxis, :]  # (1, 14)

        camera_images = self._extract_camera_images(raw_obs)
        zero_image = np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)

        main_image = (
            camera_images.get(self._main_camera_name)
            if self._main_camera_name is not None
            else None
        )
        if main_image is None and camera_images:
            main_image = next(iter(camera_images.values()))
        if main_image is None:
            main_image = zero_image

        main_image = main_image[np.newaxis, :]  # (1, H, W, 3)

        obs = {
            "states": to_tensor(states),
            "main_images": to_tensor(main_image),
            "task_descriptions": [self._task_description],
        }
        if self._wrist_camera_names:
            wrist_images = [
                camera_images.get(camera_name, zero_image)
                for camera_name in self._wrist_camera_names
            ]
            obs["wrist_images"] = to_tensor(
                np.stack(wrist_images, axis=0)[np.newaxis, :]
            )
        if self._extra_view_camera_names:
            extra_view_images = [
                camera_images.get(camera_name, zero_image)
                for camera_name in self._extra_view_camera_names
            ]
            obs["extra_view_images"] = to_tensor(
                np.stack(extra_view_images, axis=0)[np.newaxis, :]
            )
        return obs

    def _setup_dummy_cameras(self):
        """Initialize cameras without robot hardware for dummy mode.

        When camera_cfgs is provided in the config, real camera frames are
        captured even though robot arms are not connected.
        """
        camera_cfgs_raw = self.cfg.get("camera_cfgs", None)
        if camera_cfgs_raw is None:
            return

        try:
            from omegaconf import OmegaConf
            from yam_realtime.utils.launch_utils import initialize_sensors
        except ImportError:
            self._logger.warning(
                "[YAMEnv] yam_realtime not installed; dummy mode will use black images."
            )
            return

        self._server_processes: list = getattr(self, "_server_processes", [])
        sensors_cfg = {"cameras": OmegaConf.to_container(camera_cfgs_raw, resolve=True)}
        try:
            camera_dict, _ = initialize_sensors(sensors_cfg, self._server_processes)
            self._dummy_cameras = camera_dict
            self._camera_names = list(camera_dict.keys())
            self._logger.info(
                f"[YAMEnv] Dummy-mode cameras initialized: {self._camera_names}"
            )
        except Exception as e:
            self._logger.warning(
                f"[YAMEnv] Failed to initialize cameras in dummy mode: {e}. "
                "Falling back to black images."
            )

    def _read_dummy_cameras(self) -> dict:
        """Read frames from dummy-mode cameras and return as observation keys."""
        camera_obs: dict = {}
        for cam_name, cam_node in self._dummy_cameras.items():
            try:
                cam_data = cam_node.read()
                camera_obs[cam_name] = cam_data
            except Exception as e:
                self._logger.warning(
                    f"[YAMEnv] Failed to read camera '{cam_name}': {e}"
                )
        return camera_obs

    def _dummy_obs(self) -> dict:
        """Return observation dict for dummy/simulation mode.

        If cameras were initialized, captures real frames. Otherwise returns
        zero joint states with no image data (resulting in black images).
        """
        obs: dict = {
            "left": {"joint_pos": np.zeros(_JOINTS_PER_ARM, dtype=np.float32)},
            "right": {"joint_pos": np.zeros(_JOINTS_PER_ARM, dtype=np.float32)},
        }
        if self._dummy_cameras:
            obs.update(self._read_dummy_cameras())
        return obs

    # ------------------------------------------------------------------
    # Metrics (mirrors RealWorldEnv for compatibility with runners)
    # ------------------------------------------------------------------

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)
        self.prev_step_reward = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0.0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, intervene_current_step, infos):
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
            "reward": np.where(
                self._elapsed_steps > 0, self.returns / self._elapsed_steps, 0.0
            ),
            "intervened_once": self.intervened_once,
            "intervened_steps": self.intervened_steps,
            "success_no_intervened": self.success_once.copy() & (~self.intervened_once),
        }
        infos["episode"] = to_tensor(episode_info)
        return infos

    # ------------------------------------------------------------------
    # Properties expected by the runner
    # ------------------------------------------------------------------

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool):
        self._is_start = value

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def total_num_group_envs(self) -> int:
        return np.iinfo(np.uint8).max // 2  # mirrors RealWorldEnv

    @property
    def task_description(self) -> str:
        return self._task_description

    @task_description.setter
    def task_description(self, value: str) -> None:
        self._task_description = str(value)

    @property
    def task_descriptions(self) -> list[str]:
        return [self._task_description]

    def update_reset_state_ids(self) -> None:
        """No-op. YAMEnv is a single-episode env; no episode-state IDs to update."""
