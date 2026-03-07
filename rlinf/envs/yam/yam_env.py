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
``states``        – concatenated joint positions of both arms (14D).
``main_images``   – uint8 RGB image (H, W, 3) from the primary camera.

Action space
~~~~~~~~~~~~
14D float32 vector: 7 joint positions for left arm followed by 7 for right arm.
"""

import time
from collections import OrderedDict
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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
        if not self._is_dummy:
            self._setup_robot()

        # Gym spaces
        self.observation_space = gym.spaces.Dict(
            {
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
        )
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
            from yam_realtime.utils.launch_utils import initialize_robots, initialize_sensors
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
            sensors_cfg = {"cameras": OmegaConf.to_container(camera_cfgs_raw, resolve=True)}
            camera_dict, _ = initialize_sensors(sensors_cfg, self._server_processes)
            self._logger.info(f"[YAMEnv] Cameras initialized: {list(camera_dict.keys())}")

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
            raw_obs = self._robot_env.reset()

        return self._wrap_obs(raw_obs), {}

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

    def close(self):
        if self._robot_env is not None:
            try:
                self._robot_env.close()
            except Exception:
                pass
        for proc in getattr(self, "_server_processes", []):
            try:
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()
            except Exception:
                pass

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
            jpos = raw_obs["left"].get("joint_pos", np.zeros(_JOINTS_PER_ARM))
            left_state = np.asarray(jpos, dtype=np.float32)[:_JOINTS_PER_ARM]
        if "right" in raw_obs and raw_obs["right"] is not None:
            jpos = raw_obs["right"].get("joint_pos", np.zeros(_JOINTS_PER_ARM))
            right_state = np.asarray(jpos, dtype=np.float32)[:_JOINTS_PER_ARM]
        states = np.concatenate([left_state, right_state])  # (14,)
        # Add batch dimension for vectorised env compatibility
        states = states[np.newaxis, :]  # (1, 14)

        # --- camera image ---
        # yam_realtime CameraNode.read() returns {"images": {name: ndarray}, "timestamp": ...}
        img = None
        for cam_key in raw_obs:
            if self._main_camera in cam_key or "rgb" in cam_key.lower():
                cam_data = raw_obs[cam_key]
                if isinstance(cam_data, dict):
                    # yam_realtime format: {"images": {name: ndarray}, "timestamp": ...}
                    images = cam_data.get("images")
                    if isinstance(images, dict) and images:
                        img = next(iter(images.values()))
                    else:
                        img = cam_data.get("rgb", cam_data.get("color", None))
                elif isinstance(cam_data, np.ndarray):
                    img = cam_data
                if img is not None:
                    break

        if img is None:
            img = np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)

        img = np.asarray(img, dtype=np.uint8)
        if img.shape != (self._img_h, self._img_w, 3):
            import cv2
            img = cv2.resize(img, (self._img_w, self._img_h))
        img = img[np.newaxis, :]  # (1, H, W, 3)

        obs = {
            "states": states,
            "main_images": img,
            "task_descriptions": [self._task_description],
        }
        return to_tensor(obs)

    def _dummy_obs(self) -> dict:
        """Return a zeroed observation dict for dummy/simulation mode."""
        return {
            "left": {"joint_pos": np.zeros(_JOINTS_PER_ARM, dtype=np.float32)},
            "right": {"joint_pos": np.zeros(_JOINTS_PER_ARM, dtype=np.float32)},
        }

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
