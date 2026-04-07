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
#
# BaseSimManager — abstract base class for GenieSim simulation backend managers.
#
# Two concrete implementations exist:
#   SimContainerManager  (container_manager.py)  — launches sim inside a Docker container
#   SimProcessManager    (process_manager.py)    — launches sim as a local subprocess
#
# Both share the same sentinel-file handshake, SHM cleanup, config serialisation,
# and _wait_ready() readiness polling loop.  Only the process/container lifecycle
# methods differ.

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_READY_FILE    = ".geniesim_ready"
_SIM_CFG_FILE  = ".sim_server_config.json"
_PROGRESS_FILE = ".geniesim_progress"
_STOP_FILE     = ".geniesim_stop"
_START_FILE    = ".geniesim_start"
_IDLE_FILE     = ".geniesim_idle"
_ERROR_FILE    = ".geniesim_error"


class SimStartupError(RuntimeError):
    """Raised when the simulation backend fails to start or become ready."""


# Backward-compat alias so existing code that catches ContainerStartupError still works.
ContainerStartupError = SimStartupError


class BaseSimManager(ABC):
    """
    Abstract base for GenieSim simulation backend managers.

    Subclasses implement the backend-specific lifecycle methods
    (ensure_running, health_check, shutdown) and log/process introspection
    hooks used by the shared _wait_ready() polling loop.

    Shared responsibilities handled here:
      - Sentinel-file path setup
      - SHM cleanup (basic unlink; subclasses may override for privilege escalation)
      - ProcessManager config serialisation (_write_sim_config)
      - pm_kwargs_from_vec_cfg static helper
      - _wait_ready() readiness loop (template method pattern)
    """

    # ---------------------------------------------------------------------- #
    # Known startup failure patterns → human-readable messages
    # (shared across all backends)
    # ---------------------------------------------------------------------- #
    _FAILURE_PATTERNS: List[tuple] = [
        (r"CUDA.*[Ee]rror|[Ee]rror.*CUDA|GPU.*not found|no CUDA-capable device",
         "GPU/CUDA initialization failed — run `nvidia-smi` to verify GPU accessibility"),
        (r"Failed to find display|cannot connect to X|_XSERVTransmkdir",
         "Display error — ensure `headless: true` in the sim config"),
        (r"ModuleNotFoundError.*rclpy|ImportError.*rclpy",
         "rclpy not importable — entrypoint may have failed to source ROS"),
        (r"ModuleNotFoundError.*geniesim_rl_interfaces|No module named.*geniesim_rl_interfaces",
         "geniesim_rl_interfaces not built — colcon build failed; check entrypoint logs"),
        (r"No such file or directory.*\.xml|FileNotFoundError.*\.xml",
         "MJCF file not found — check mjcf_path in the config"),
        (r"No such file or directory.*\.usd|FileNotFoundError.*\.usd",
         "USD scene/robot file not found — check scene_usd/robot_usd in the config"),
        (r"[Ss]hared.?[Mm]emory.*[Ee]rror|FileExistsError.*shm|/dev/shm",
         "Shared memory conflict — a stale session may hold the SHM segment; "
         "run `ipcs -m` and `ipcrm` to clean up, or change shm_name"),
        (r"[Oo]ut [Oo]f [Mm]emory|OOMKilled|OutOfMemoryError|ENOMEM",
         "Out of memory — reduce num_envs or image resolution"),
        (r"[Pp]ermission denied",
         "Permission denied — check file ownership and ACL setup"),
        (r"entrypoint.*failed|setup\.bash.*[Ee]rror|colcon.*[Ee]rror",
         "Startup script failed — inspect logs for colcon build errors"),
        (r"ROS_DOMAIN_ID.*conflict|multiple.*ROS.*nodes.*same.*domain",
         "ROS domain ID conflict — change ros_domain_id in the config"),
    ]

    # Stage labels written by sim_server.py to .geniesim_progress
    _STAGE_MSGS = {
        "mujoco_launching": "Launching MuJoCo physics nodes...",
        "mujoco_ready":     "MuJoCo physics nodes ready.",
        "isaac_launching":  "Launching Isaac Sim renderer (cold start ~2 min)...",
        "isaac_loading":    "Isaac Sim loading USD scene and GPU renderer...",
        "isaac_ready":      "Isaac Sim renderer ready — finalising...",
    }

    # Substrings in log lines worth surfacing to the user during startup wait.
    _ISAAC_LOG_PATTERNS = [
        "entrypoint", "geniesim_rl_interfaces", "Installing geniesim",
        "sim_server", "ProcessManager", "MuJoCo", "ready",
        "Isaac Sim", "Loading", "Startup", "startup", "RTX",
        "renderer", "SHM", "shm",
    ]

    def __init__(
        self,
        geniesim_root: Path,
        keep_alive: bool,
        reuse_running: bool,
        startup_timeout_sec: int,
        ros_domain_id: int,
    ) -> None:
        self.geniesim_root = geniesim_root
        self.keep_alive = keep_alive
        self.reuse_running = reuse_running
        self.startup_timeout_sec = startup_timeout_sec
        self.ros_domain_id = ros_domain_id

        self._ready_file    = geniesim_root / _READY_FILE
        self._cfg_file      = geniesim_root / _SIM_CFG_FILE
        self._progress_file = geniesim_root / _PROGRESS_FILE
        self._stop_file     = geniesim_root / _STOP_FILE
        self._start_file    = geniesim_root / _START_FILE
        self._idle_file     = geniesim_root / _IDLE_FILE
        self._error_file    = geniesim_root / _ERROR_FILE

    # ---------------------------------------------------------------------- #
    # Public API (abstract)
    # ---------------------------------------------------------------------- #

    @abstractmethod
    def ensure_running(self, pm_kwargs: Dict) -> None:
        """Start the simulation backend and wait for readiness."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is running and the sim is live."""

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully stop the simulation backend."""

    # ---------------------------------------------------------------------- #
    # Template-method hooks (abstract — implemented by subclasses)
    # ---------------------------------------------------------------------- #

    @abstractmethod
    def _get_log_tail(self, tail: int = 40) -> str:
        """Return the last *tail* log lines from the sim backend."""

    @abstractmethod
    def _check_alive(self) -> Tuple[bool, str]:
        """
        Check whether the backend process/container is still alive.

        Returns
        -------
        (is_alive, error_msg)
            is_alive  — True if still running
            error_msg — human-readable message when is_alive is False
        """

    @abstractmethod
    def _classify_failure(self) -> str:
        """
        Scan backend logs for known failure patterns.

        Returns a human-readable string with the likely cause and a log tail.
        """

    @abstractmethod
    def _error_hint(self) -> str:
        """
        Return a backend-specific hint string for the .geniesim_error fast path.

        Example:
            "Check sim_server_local.log:\n  cat /path/sim_server_local.log | tail -50"
        """

    # ---------------------------------------------------------------------- #
    # Shared: SHM cleanup (basic; Docker subclass overrides for uid-escalation)
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _check_shm_accessible(shm_name: str) -> bool:
        """Return True if /dev/shm/<shm_name> exists AND is readable."""
        shm_path = Path("/dev/shm") / shm_name
        if not shm_path.exists():
            return False
        try:
            shm_path.open("rb").close()
            return True
        except PermissionError:
            return False

    def _cleanup_stale_shm(self, shm_name: str) -> None:
        """Remove a stale POSIX SHM segment.  Subclasses may override."""
        shm_path = Path("/dev/shm") / shm_name
        if not shm_path.exists():
            return
        try:
            shm_path.unlink()
            logger.info("Removed stale SHM '%s'", shm_name)
        except PermissionError:
            owner_uid = shm_path.stat().st_uid
            logger.warning(
                "Stale SHM '%s' owned by uid %d — run `sudo rm /dev/shm/%s` if startup fails",
                shm_name, owner_uid, shm_name,
            )

    def _cleanup_stale_ctrl_shms(self, shm_name: str) -> None:
        """Remove stale per-env ctrl SHM segments.  Subclasses may override."""
        shm_dir = Path("/dev/shm")
        prefix = f"{shm_name}_ctrl_"
        for entry in shm_dir.iterdir():
            if not entry.name.startswith(prefix):
                continue
            try:
                entry.unlink()
                logger.info("Removed stale ctrl SHM '%s'", entry.name)
            except PermissionError:
                owner_uid = entry.stat().st_uid
                logger.warning(
                    "Stale ctrl SHM '%s' owned by uid %d — run `sudo rm /dev/shm/%s`",
                    entry.name, owner_uid, entry.name,
                )

    def _cleanup_stale_step_shm(self, shm_name: str) -> None:
        """Remove a stale step SHM segment."""
        from rlinf.envs.geniesim.shm_layout import step_shm_name
        name = step_shm_name(shm_name)
        shm_path = Path("/dev/shm") / name
        if not shm_path.exists():
            return
        try:
            shm_path.unlink()
            logger.info("Removed stale step SHM '%s'", name)
        except PermissionError:
            owner_uid = shm_path.stat().st_uid
            logger.warning(
                "Stale step SHM '%s' owned by uid %d — run `sudo rm /dev/shm/%s`",
                name, owner_uid, name,
            )

    # ---------------------------------------------------------------------- #
    # Shared: config serialisation
    # ---------------------------------------------------------------------- #

    def _write_sim_config(self, pm_kwargs: Dict) -> None:
        """
        Write the ProcessManager config JSON to geniesim_root/.sim_server_config.json.

        Subclasses may override to apply path overrides before writing
        (e.g. SimContainerManager applies container_paths here).
        """
        self._cfg_file.write_text(json.dumps(dict(pm_kwargs), indent=2))

    @staticmethod
    def pm_kwargs_from_vec_cfg(vec_cfg) -> Dict:
        """
        Extract ProcessManager.__init__() kwargs from a GenieSimVectorEnvConfig.

        These are the fields forwarded to sim_server.py via the JSON config.
        Training-wrapper-only fields (enable_reward, max_episode_steps,
        attach_to_running, …) are excluded.
        """
        return {
            "num_envs":           vec_cfg.num_envs,
            "mjcf_path":          vec_cfg.mjcf_path,
            "scene_usd":          vec_cfg.scene_usd,
            "robot_usd":          vec_cfg.robot_usd,
            "robot_prim":         vec_cfg.robot_prim,
            "shm_name":           vec_cfg.shm_name,
            "physics_hz":         vec_cfg.physics_hz,
            "render_hz":          vec_cfg.render_hz,
            "cam_width":          vec_cfg.cam_width,
            "cam_height":         vec_cfg.cam_height,
            "main_cam_prim":      vec_cfg.main_cam_prim,
            "wrist_cam_prim":     vec_cfg.wrist_cam_prim,
            "cameras_json":       json.dumps(getattr(vec_cfg, "cameras", []) or []),
            "headless":           vec_cfg.headless,
            "ros_domain_id":      vec_cfg.ros_domain_id,
            "isaac_python":       vec_cfg.isaac_python,
            "mujoco_python":      vec_cfg.mujoco_python or "",
            "task_name":          vec_cfg.task_name,
            "robot_type":         vec_cfg.robot_type,
            "task_instance_id":   vec_cfg.task_instance_id,
            "state_joint_offset": vec_cfg.state_joint_offset,
            "ctrl_offset":        vec_cfg.ctrl_offset,
            "ctrl_offset_r":      getattr(vec_cfg, "ctrl_offset_r", -1),
            "state_dim":          vec_cfg.state_dim,
            "action_dim":         vec_cfg.action_dim,
            "control_mode":       vec_cfg.control_mode,
            "gripper_ctrl_l":     vec_cfg.gripper_ctrl_l,
            "gripper_ctrl_r":     vec_cfg.gripper_ctrl_r,
            "ee_body_l":          vec_cfg.ee_body_l,
            "ee_body_r":          vec_cfg.ee_body_r,
            "ik_max_iter":        vec_cfg.ik_max_iter,
            "ik_damp":            vec_cfg.ik_damp,
            "randomization_cfg_json": vec_cfg.randomization_cfg_json,
            "init_qpos_json":     getattr(vec_cfg, "init_qpos_json", ""),
            "reset_ee_r_json":    getattr(vec_cfg, "reset_ee_r_json", ""),
            "seed":               getattr(vec_cfg, "seed", 42),
            "info_body_names":    getattr(vec_cfg, "info_body_names", []),
            "sync_mode":          getattr(vec_cfg, "sync_mode", True),
            "steps_per_step":     getattr(vec_cfg, "steps_per_step", 33),
            "max_episode_steps":  getattr(vec_cfg, "max_episode_steps", 300),
            "enable_reward":      getattr(vec_cfg, "enable_reward", False),
            "reward_coef":        getattr(vec_cfg, "reward_coef", 1.0),
            "ignore_terminations": getattr(vec_cfg, "ignore_terminations", False),
            "auto_reset":         getattr(vec_cfg, "auto_reset", True),
            "task_description":   getattr(vec_cfg, "task_description", ""),
        }

    # ---------------------------------------------------------------------- #
    # Shared: readiness polling loop (template method)
    # ---------------------------------------------------------------------- #

    def _wait_ready(self) -> None:
        """
        Block until {geniesim_root}/.geniesim_ready is written or timeout.

        Uses abstract hooks:
          _get_log_tail()   — backend log lines for display
          _check_alive()    — whether the backend process is still up
          _classify_failure() — pattern-matched error diagnosis
          _error_hint()     — log inspection command for the error-file fast path
        """
        deadline = time.time() + self.startup_timeout_sec
        t_start = time.time()
        last_stage = ""
        last_heartbeat = time.time()
        _HEARTBEAT_SEC = 5.0
        _shown_log_lines: set = set()

        logger.info("Waiting for simulation readiness (timeout=%ds)...", self.startup_timeout_sec)

        while time.time() < deadline:
            if self._ready_file.exists():
                elapsed = time.time() - t_start
                logger.info("Simulation ready (%.0fs elapsed)", elapsed)
                return

            # ---- Stage label from sim_server.py ----
            try:
                stage = self._progress_file.read_text().strip()
                if stage and stage != last_stage:
                    msg = self._STAGE_MSGS.get(stage, stage)
                    elapsed = time.time() - t_start
                    logger.info("  › %s  (%.0fs)", msg, elapsed)
                    last_stage = stage
                    last_heartbeat = time.time()
            except (FileNotFoundError, OSError):
                pass

            # ---- Renderer crash fast path ----
            if self._error_file.exists():
                try:
                    detail = self._error_file.read_text().strip()
                except OSError:
                    detail = "(could not read error file)"
                raise SimStartupError(
                    f"Isaac Sim renderer crashed during startup:\n  {detail}\n\n"
                    + self._error_hint()
                    + "\n\n"
                    + self._classify_failure()
                )

            # ---- Backend alive check ----
            is_alive, err_msg = self._check_alive()
            if not is_alive:
                raise SimStartupError(err_msg + "\n\n" + self._classify_failure())

            # ---- Tail logs for new relevant lines ----
            try:
                log_tail = self._get_log_tail(tail=40)
                for line in log_tail.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped in _shown_log_lines:
                        continue
                    if any(pat.lower() in stripped.lower() for pat in self._ISAAC_LOG_PATTERNS):
                        _shown_log_lines.add(stripped)
                        logger.info("    %s", stripped)
            except Exception:
                pass

            # ---- Periodic heartbeat ----
            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_SEC:
                elapsed = now - t_start
                remaining = self.startup_timeout_sec - elapsed
                if last_stage in ("", "mujoco_launching"):
                    hint = "entrypoint + MuJoCo init..."
                elif last_stage == "mujoco_ready":
                    hint = "waiting for Isaac Sim renderer to start..."
                elif last_stage in ("isaac_launching", "isaac_loading"):
                    hint = "Isaac Sim GPU init + USD scene load (this is the long part)..."
                else:
                    hint = "finalising..."
                logger.info("  › %s  elapsed=%.0fs  remaining≤%.0fs", hint, elapsed, remaining)
                last_heartbeat = now

            time.sleep(1.0)

        raise SimStartupError(
            f"Simulation not ready after {self.startup_timeout_sec}s.\n\n"
            + self._classify_failure()
        )

    # ---------------------------------------------------------------------- #
    # Shared: failure classification helper
    # ---------------------------------------------------------------------- #

    def _scan_logs_for_failure(self, logs: str, log_label: str) -> str:
        """
        Search *logs* for known failure patterns and return a human-readable string.

        Used by subclasses in their _classify_failure() implementation.
        """
        for pattern, message in self._FAILURE_PATTERNS:
            if re.search(pattern, logs, re.IGNORECASE):
                return (
                    f"Likely cause: {message}\n\n"
                    f"--- {log_label} ---\n{logs[-3000:]}"
                )
        return (
            f"Cause unknown — no recognised failure pattern found.\n\n"
            f"--- {log_label} ---\n{logs[-3000:]}"
        )
