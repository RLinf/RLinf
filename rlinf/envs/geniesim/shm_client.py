# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# GenieSimShmClient — RLinf-native client for the GeneSim simulation container.
#
# Thin client that communicates with the sim-side GenieSimVectorEnv via SHM:
#   Frame SHM  — camera images (read-only, written by Isaac Sim renderer)
#   Ctrl SHM   — per-env states (read-only, written by MuJoCo node)
#   Step SHM   — request-reply channel for step/reset (shared with GenieSimVectorEnv)
#
# The sim-side GenieSimVectorEnv is responsible for organising obs, reward,
# terminated, truncated, info.  This client simply writes actions and reads
# the pre-organised results.
#
# NO geniesim, rclpy, or ROS dependencies are required on the host.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from multiprocessing import resource_tracker as _resource_tracker
from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np

from rlinf.envs.geniesim.shm_layout import (
    BODY_POSE_DIM,
    CTRL_HEADER_BYTES,
    SHM_HEADER_BYTES,
    STEP_HEADER_BYTES,
    STEP_PHASE_CLOSE,
    STEP_PHASE_IDLE,
    STEP_PHASE_RESET_DONE,
    STEP_PHASE_RESET_REQUEST,
    STEP_PHASE_STEP_DONE,
    STEP_PHASE_STEP_REQUEST,
    ctrl_shm_name,
    ctrl_total_bytes,
    shm_total_bytes,
    step_shm_name,
    step_total_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# Config dataclass (mirrors GenieSimVectorEnvConfig from the geniesim package)
# ---------------------------------------------------------------------------- #


@dataclass
class GenieSimVectorEnvConfig:
    mjcf_path: str = ""
    scene_usd: str = ""
    robot_usd: str = ""
    robot_prim: str = "/robot"
    task_file: str = ""
    task_name: str = ""
    task_description: str = ""
    robot_cfg: str = "G2_omnipicker"
    robot_type: str = "G2"
    task_instance_id: int = 0

    num_envs: int = 1

    max_episode_steps: int = 300
    enable_reward: bool = False
    reward_coef: float = 1.0
    use_rel_reward: bool = False
    ignore_terminations: bool = False
    auto_reset: bool = True

    cameras: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "main", "prim": "/camera_main", "width": 640, "height": 480},
        ]
    )

    cam_width: int = 640
    cam_height: int = 480
    main_cam_prim: str = "/camera_main"
    wrist_cam_prim: str = ""

    shm_name: str = "geniesim_frames"
    shm_open_timeout_sec: int = 180

    physics_hz: float = 1000.0
    render_hz: float = 30.0
    headless: bool = True
    ros_domain_id: int = 0
    isaac_python: str = "/isaac-sim/python.sh"
    mujoco_python: str = ""

    state_dim: int = 28
    action_dim: int = 14
    state_joint_offset: int = 0
    ctrl_offset: int = 0
    ctrl_offset_r: int = -1
    control_mode: str = "joint"
    gripper_ctrl_l: int = -1
    gripper_ctrl_r: int = -1
    ee_body_l: str = "arm_l_link7"
    ee_body_r: str = "arm_r_link7"
    ik_max_iter: int = 10
    ik_damp: float = 0.05

    randomization_cfg_json: str = ""
    init_qpos_json: str = ""
    reset_ee_r_json: str = ""
    seed: int = 42

    info_body_names: list[str] = field(default_factory=list)

    sync_mode: bool = True
    steps_per_step: int = 33

    attach_to_running: bool = False


# ---------------------------------------------------------------------------- #
# SHM client
# ---------------------------------------------------------------------------- #


class GenieSimShmClient:
    """
    Lightweight environment client that talks to the sim-side GenieSimVectorEnv
    exclusively via shared memory.

    Communication protocol:
      1. Client writes actions into step SHM and sets STEP_PHASE_STEP_REQUEST.
      2. GenieSimVectorEnv (sim side) reads actions, runs physics, organises
         obs/reward/terminated/truncated/info, writes results to step SHM, and
         sets STEP_PHASE_STEP_DONE.
      3. Client reads results from step SHM and frame SHM (images).

    Interface::

        reset(env_idx=None) -> (obs_dict, info_dict)
        step(actions)       -> (obs, rewards, terminated, truncated, infos)
        close()
    """

    def __init__(self, cfg: GenieSimVectorEnvConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs

        self._cameras = list(cfg.cameras) if cfg.cameras else []
        if not self._cameras and cfg.main_cam_prim:
            self._cameras = [
                {
                    "name": "main",
                    "prim": cfg.main_cam_prim,
                    "width": cfg.cam_width,
                    "height": cfg.cam_height,
                }
            ]
            if cfg.wrist_cam_prim:
                self._cameras.append(
                    {
                        "name": "wrist",
                        "prim": cfg.wrist_cam_prim,
                        "width": cfg.cam_width,
                        "height": cfg.cam_height,
                    }
                )
        self._num_cams = len(self._cameras)
        self._cam_h = self._cameras[0]["height"] if self._cameras else cfg.cam_height
        self._cam_w = self._cameras[0]["width"] if self._cameras else cfg.cam_width

        self._info_body_names: list[str] = list(cfg.info_body_names or [])
        self._info_dim = len(self._info_body_names) * BODY_POSE_DIM

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._frames: Optional[np.ndarray] = None
        self._frame_counter: Optional[np.ndarray] = None
        self._open_frame_shm(max_attempts=cfg.shm_open_timeout_sec)

        self._ctrl_shms: list[shared_memory.SharedMemory] = []
        self._ctrl_states_bufs: list[np.ndarray] = []
        self._open_ctrl_shms(max_attempts=cfg.shm_open_timeout_sec)

        self._step_shm: Optional[shared_memory.SharedMemory] = None
        self._step_phase: Optional[np.ndarray] = None
        self._step_actions: Optional[np.ndarray] = None
        self._step_rewards: Optional[np.ndarray] = None
        self._step_terminated: Optional[np.ndarray] = None
        self._step_truncated: Optional[np.ndarray] = None
        self._step_elapsed: Optional[np.ndarray] = None
        self._step_returns: Optional[np.ndarray] = None
        self._step_success: Optional[np.ndarray] = None
        self._step_info_poses: Optional[np.ndarray] = None
        self._open_step_shm(max_attempts=cfg.shm_open_timeout_sec)

        logger.info(
            "Initialised | num_envs=%d state_dim=%d action_dim=%d info_dim=%d",
            self.num_envs,
            cfg.state_dim,
            cfg.action_dim,
            self._info_dim,
        )

    # ---------------------------------------------------------------------- #
    # SHM attachment
    # ---------------------------------------------------------------------- #

    def _open_frame_shm(self, max_attempts: int = 180):
        h, w = self._cam_h, self._cam_w
        shm_bytes = shm_total_bytes(self.num_envs, h, w, num_cams=self._num_cams)
        for _ in range(max_attempts):
            try:
                self._shm = shared_memory.SharedMemory(
                    name=self.cfg.shm_name, create=False, size=shm_bytes
                )
                _resource_tracker.unregister(f"/{self.cfg.shm_name}", "shared_memory")
                break
            except FileNotFoundError:
                time.sleep(1.0)
        else:
            raise RuntimeError(
                f"[GenieSimShmClient] Frame SHM '{self.cfg.shm_name}' "
                f"not available after {max_attempts}s"
            )

        self._frames = np.ndarray(
            (self.num_envs, self._num_cams, h, w, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=SHM_HEADER_BYTES,
        )
        self._frame_counter = np.ndarray(
            (1,), dtype=np.uint32, buffer=self._shm.buf, offset=0
        )

    def _open_ctrl_shms(self, max_attempts: int = 180):
        _total = ctrl_total_bytes(
            self.cfg.state_dim, self.cfg.action_dim, self._info_dim
        )
        for i in range(self.num_envs):
            name = ctrl_shm_name(self.cfg.shm_name, i)
            shm = None
            for _ in range(max_attempts):
                try:
                    shm = shared_memory.SharedMemory(
                        name=name, create=False, size=_total
                    )
                    break
                except FileNotFoundError:
                    time.sleep(1.0)
            if shm is None:
                raise RuntimeError(
                    f"[GenieSimShmClient] Ctrl SHM '{name}' "
                    f"not available after {max_attempts}s"
                )
            _resource_tracker.unregister(f"/{name}", "shared_memory")
            self._ctrl_shms.append(shm)
            self._ctrl_states_bufs.append(
                np.ndarray(
                    (self.cfg.state_dim,),
                    dtype=np.float32,
                    buffer=shm.buf,
                    offset=CTRL_HEADER_BYTES,
                )
            )
        logger.info(
            "Ctrl SHMs attached | state_dim=%d action_dim=%d",
            self.cfg.state_dim,
            self.cfg.action_dim,
        )

    def _open_step_shm(self, max_attempts: int = 180):
        N = self.num_envs
        A = self.cfg.action_dim
        _total = step_total_bytes(N, A, self._info_dim)
        name = step_shm_name(self.cfg.shm_name)
        for _ in range(max_attempts):
            try:
                self._step_shm = shared_memory.SharedMemory(
                    name=name, create=False, size=_total
                )
                break
            except FileNotFoundError:
                time.sleep(1.0)
        if self._step_shm is None:
            raise RuntimeError(
                f"[GenieSimShmClient] Step SHM '{name}' "
                f"not available after {max_attempts}s"
            )
        _resource_tracker.unregister(f"/{name}", "shared_memory")
        off = 0
        self._step_phase = np.ndarray(
            (1,), dtype=np.uint32, buffer=self._step_shm.buf, offset=off
        )
        off += STEP_HEADER_BYTES
        self._step_reset_mask = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_actions = np.ndarray(
            (N, A), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * A * 4
        self._step_rewards = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_terminated = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_truncated = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_elapsed = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_returns = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_success = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        if self._info_dim > 0:
            self._step_info_poses = np.ndarray(
                (N, self._info_dim),
                dtype=np.float32,
                buffer=self._step_shm.buf,
                offset=off,
            )
        logger.info("Step SHM attached: %s", name)

    # ---------------------------------------------------------------------- #
    # Observation helpers
    # ---------------------------------------------------------------------- #

    def _get_obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for cam_idx, cam_cfg in enumerate(self._cameras):
            name = cam_cfg["name"]
            key = f"{name}_images"
            obs[key] = np.copy(self._frames[:, cam_idx])
        if not self._cameras:
            h, w = self.cfg.cam_height, self.cfg.cam_width
            obs["main_images"] = np.zeros((self.num_envs, h, w, 3), dtype=np.uint8)
        states = np.stack([np.copy(b) for b in self._ctrl_states_bufs], axis=0)
        obs["states"] = states
        obs["task_descriptions"] = [self.cfg.task_description] * self.num_envs
        return obs

    # ---------------------------------------------------------------------- #
    # Step SHM request-reply
    # ---------------------------------------------------------------------- #

    def _wait_step_done(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            phase = int(self._step_phase[0])
            if phase == STEP_PHASE_STEP_DONE:
                self._step_phase[0] = STEP_PHASE_IDLE
                return True
            time.sleep(0.0001)
        logger.warning("Step timeout")
        return False

    def _wait_reset_done(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            phase = int(self._step_phase[0])
            if phase == STEP_PHASE_RESET_DONE:
                self._step_phase[0] = STEP_PHASE_IDLE
                return True
            time.sleep(0.0001)
        logger.warning("Reset timeout")
        return False

    def _read_body_poses(self) -> dict[str, np.ndarray]:
        if self._info_dim == 0 or self._step_info_poses is None:
            return {}
        poses = {}
        for idx, bname in enumerate(self._info_body_names):
            off = idx * BODY_POSE_DIM
            poses[bname] = np.copy(self._step_info_poses[:, off : off + BODY_POSE_DIM])
        return poses

    def _read_step_infos(self) -> dict:
        infos: dict = {
            "episode": {
                "success_once": np.copy(self._step_success).astype(bool),
                "return": np.copy(self._step_returns),
                "episode_len": np.copy(self._step_elapsed).astype(np.int32),
                "reward": np.where(
                    self._step_elapsed > 0,
                    self._step_returns / np.maximum(self._step_elapsed, 1),
                    0.0,
                ),
            },
            "task_progress": [[] for _ in range(self.num_envs)],
        }
        if self._info_dim > 0:
            infos["body_poses"] = self._read_body_poses()
        return infos

    # ---------------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------------- #

    def reset(self, env_idx=None) -> tuple[dict, dict]:
        if env_idx is None:
            self._step_reset_mask[:] = 1.0
        else:
            self._step_reset_mask[:] = 0.0
            if isinstance(env_idx, int):
                self._step_reset_mask[env_idx] = 1.0
            else:
                for i in env_idx:
                    self._step_reset_mask[i] = 1.0
        self._step_phase[0] = STEP_PHASE_RESET_REQUEST
        self._wait_reset_done()
        obs = self._get_obs()
        infos = self._read_step_infos()
        return obs, infos

    # ---------------------------------------------------------------------- #
    # Step
    # ---------------------------------------------------------------------- #

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        np.copyto(self._step_actions, actions.astype(np.float32))
        self._step_phase[0] = STEP_PHASE_STEP_REQUEST
        self._wait_step_done()

        rewards = np.copy(self._step_rewards)
        terminated = np.copy(self._step_terminated).astype(bool)
        truncated = np.copy(self._step_truncated).astype(bool)
        obs = self._get_obs()
        infos = self._read_step_infos()

        return obs, rewards, terminated, truncated, infos

    # ---------------------------------------------------------------------- #
    # Cleanup
    # ---------------------------------------------------------------------- #

    def close(self):
        if self._step_phase is not None:
            self._step_phase[0] = STEP_PHASE_CLOSE
        if self._step_shm is not None:
            try:
                self._step_shm.close()
            except Exception:
                pass
            self._step_shm = None
        for shm in self._ctrl_shms:
            try:
                shm.close()
            except Exception:
                pass
        self._ctrl_shms.clear()
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
