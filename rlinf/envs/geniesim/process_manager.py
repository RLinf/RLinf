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
# SimProcessManager — launches GenieSim sim_server.py as a local subprocess.
#
# Use this when both the RLinf training code and GenieSim run inside the same
# environment (e.g. the merged geniesim-rlinf-train container).  No Docker
# daemon access is required.
#
# Config fields (container_cfg.mode == 'local'):
#
#   geniesim_root    (str)  Path to rlinf_open_source/  (must be readable)
#   keep_alive       (bool) Don't stop sim on shutdown() [default False]
#   reuse_running    (bool) Reuse an already-running sim  [default True]
#   startup_timeout_sec (int)  [default 300]
#   ros_domain_id    (int)  ROS_DOMAIN_ID                [default 0]
#   ros_ws_install   (str)  colcon install dir            [default /geniesim/ros_ws_build/install]
#

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging

from rlinf.envs.geniesim.sim_manager_base import BaseSimManager, SimStartupError

logger = logging.getLogger(__name__)


class SimProcessManager(BaseSimManager):
    """
    Manages GenieSim sim_server.py as a direct subprocess.

    Typical call sequence
    ---------------------
    mgr = SimProcessManager(container_cfg)
    mgr.ensure_running(pm_kwargs)   # start + handshake
    # ... training ...
    mgr.shutdown()

    Reuse logic
    -----------
    When reuse_running=True (default), an already-running sim_server.py (detected
    via sentinel files) is reused without spawning a new process.  This handles
    repeated GenieSimBaseEnv instantiations within the same session, as well as
    resuming a training run that previously left keep_alive=True.
    """

    def __init__(self, container_cfg) -> None:
        geniesim_root = Path(container_cfg.geniesim_root).expanduser().resolve()
        super().__init__(
            geniesim_root=geniesim_root,
            keep_alive=bool(getattr(container_cfg, "keep_alive", False)),
            reuse_running=bool(getattr(container_cfg, "reuse_running", True)),
            startup_timeout_sec=int(getattr(container_cfg, "startup_timeout_sec", 300)),
            ros_domain_id=int(getattr(container_cfg, "ros_domain_id", 0)),
        )
        _rws = getattr(container_cfg, "ros_ws_install", None)
        self.ros_ws_install: str = str(_rws) if _rws else "/geniesim/ros_ws_build/install"

        self._local_proc: Optional[subprocess.Popen] = None
        self._log_path = geniesim_root / "sim_server_local.log"

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def ensure_running(self, pm_kwargs: Dict) -> None:
        """
        Ensure sim_server.py is running and ready.

        Reuse logic (when reuse_running=True):
          - If .geniesim_ready + SHM accessible → reuse immediately.
          - If .geniesim_idle exists → a keep_alive process is waiting;
            write new config + signal restart via .geniesim_start.
          - Otherwise → fresh start.
        """
        shm_name = pm_kwargs.get("shm_name", "geniesim_frames")

        if self.reuse_running:
            if self._ready_file.exists():
                if self._check_shm_accessible(shm_name):
                    logger.info("Reusing ready local sim_server process")
                    return
                logger.info(
                    "Ready file exists but SHM '%s' not accessible — restarting local sim_server...",
                    shm_name,
                )
            elif self._idle_file.exists():
                if self._local_proc is not None and self._local_proc.poll() is None:
                    logger.info("Local sim_server is idle — restarting sim processes...")
                    self._cleanup_stale_shm(shm_name)
                    self._cleanup_stale_ctrl_shms(shm_name)
                    self._cleanup_stale_step_shm(shm_name)
                    self._idle_file.unlink(missing_ok=True)
                    self._ready_file.unlink(missing_ok=True)
                    self._progress_file.unlink(missing_ok=True)
                    self._error_file.unlink(missing_ok=True)
                    self._write_sim_config(pm_kwargs)
                    self._start_file.write_text("start")
                    self._wait_ready()
                    return
                logger.info(
                    "Stale .geniesim_idle found but no sim_server process — "
                    "cleaning up and starting fresh..."
                )
                self._idle_file.unlink(missing_ok=True)

        # Terminate any process tracked by this instance.
        if self._local_proc is not None and self._local_proc.poll() is None:
            self._local_proc.terminate()
            try:
                self._local_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._local_proc.kill()
            self._local_proc = None

        # Kill orphan sim_server.py processes left by previous crashed runs.
        self._kill_orphan_sim_servers()

        # Fresh start.
        self._ready_file.unlink(missing_ok=True)
        self._error_file.unlink(missing_ok=True)
        self._cleanup_stale_shm(shm_name)
        self._cleanup_stale_ctrl_shms(shm_name)
        self._cleanup_stale_step_shm(shm_name)
        self._progress_file.unlink(missing_ok=True)
        self._write_sim_config(pm_kwargs)
        self._launch(pm_kwargs)
        self._wait_ready()

    def health_check(self) -> bool:
        """Return True if sim_server.py is running and the sim is live."""
        if self._local_proc is None or self._local_proc.poll() is not None:
            # No proc tracked by this instance — check sentinel files, as the
            # process may have been started by a previous instance.
            return self._ready_file.exists()
        return self._ready_file.exists()

    def shutdown(self) -> None:
        """
        Gracefully stop the local sim_server.py.

        If keep_alive=True: stop the sim processes via sentinel file so the
        next run can restart without Isaac Sim cold-start overhead.
        If keep_alive=False: terminate the subprocess and clean up.
        """
        if self.keep_alive:
            proc_alive = self._local_proc is not None and self._local_proc.poll() is None
            sim_active = self._ready_file.exists() or proc_alive
            if not sim_active or self._idle_file.exists():
                return
            logger.info(
                "Stopping sim processes (sim_server.py stays alive for quick restart)..."
            )
            self._stop_file.write_text("stop")
            deadline = time.time() + 60
            while time.time() < deadline:
                if self._idle_file.exists():
                    logger.info("Local sim processes stopped.")
                    return
                time.sleep(0.5)
            logger.warning("Local sim processes may not have stopped cleanly (60 s timeout).")
            return

        logger.info("Shutting down local sim_server.py...")
        self._ready_file.unlink(missing_ok=True)
        if self._local_proc is not None and self._local_proc.poll() is None:
            self._local_proc.terminate()
            try:
                self._local_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._local_proc.kill()
            self._local_proc = None
            logger.info("Local sim_server.py stopped.")
        self._kill_orphan_sim_servers()

    # ---------------------------------------------------------------------- #
    # BaseSimManager abstract hooks
    # ---------------------------------------------------------------------- #

    def _get_log_tail(self, tail: int = 40) -> str:
        try:
            lines = self._log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-tail:])
        except OSError:
            return ""

    def _check_alive(self) -> Tuple[bool, str]:
        if self._local_proc is not None and self._local_proc.poll() is not None:
            return (
                False,
                f"Local sim_server.py exited unexpectedly "
                f"(rc={self._local_proc.returncode}).",
            )
        return True, ""

    def _classify_failure(self) -> str:
        logs = self._get_log_tail(tail=200)
        return self._scan_logs_for_failure(logs, f"Last sim_server_local.log")

    def _error_hint(self) -> str:
        return (
            f"Check sim_server_local.log:\n"
            f"  cat {self._log_path} | tail -50"
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _launch(self, pm_kwargs: Dict) -> None:
        """Spawn sim_server.py as a subprocess with the correct environment."""
        sim_server_path = (
            self.geniesim_root / "source" / "geniesim" / "rl" / "scripts" / "sim_server.py"
        )

        # Source ROS + colcon workspace, then exec sim_server.py.
        # The colcon install dir must be sourced for geniesim_rl_interfaces.
        bash_cmd = (
            f"source /opt/ros/jazzy/setup.bash 2>/dev/null ; "
            f"[ -f {self.ros_ws_install}/setup.bash ] && "
            f"source {self.ros_ws_install}/setup.bash 2>/dev/null ; "
            f"exec python3 {sim_server_path}"
            f" --config-json {self._cfg_file}"
            f" --ready-file {self._ready_file}"
        )

        env = os.environ.copy()
        # Deactivate any active Python venv so sim_server.py uses the system
        # Python that has ROS Jazzy rclpy (not the RLinf training venv).
        _venv = env.pop("VIRTUAL_ENV", "")
        env.pop("VIRTUAL_ENV_PROMPT", None)
        if _venv:
            env["PATH"] = ":".join(
                p for p in env.get("PATH", "").split(":")
                if not p.startswith(_venv)
            )
        env.update({
            "SIM_REPO_ROOT":           str(self.geniesim_root),
            "ROS_DOMAIN_ID":           str(self.ros_domain_id),
            "ROS_LOCALHOST_ONLY":      "1",
            "GENIESIM_ROS_WS_INSTALL": self.ros_ws_install,
        })

        logger.info(
            "Starting local sim_server.py\n"
            "  sim_server    : %s\n"
            "  geniesim_root : %s\n"
            "  ros_ws_install: %s\n"
            "  log           : %s",
            sim_server_path, self.geniesim_root, self.ros_ws_install, self._log_path,
        )

        # pylint: disable=consider-using-with
        log_file = open(self._log_path, "w")  # noqa: WPS515 — kept open for proc lifetime
        self._local_proc = subprocess.Popen(
            ["bash", "-c", bash_cmd],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    def _kill_orphan_sim_servers(self) -> None:
        """
        Kill any sim_server.py processes left over from previous crashed runs.

        Only targets truly orphaned processes (no sentinel files present).
        Skips PID 1 (container init) and the current process.
        Uses /proc scanning — no shell spawn required.
        """
        # Leave managed processes alone.
        if self._idle_file.exists() or self._ready_file.exists():
            return

        killed: List[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid <= 1 or pid == os.getpid():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    errors="replace"
                )
            except OSError:
                continue
            if "sim_server.py" in cmdline:
                try:
                    os.kill(pid, 9)
                    killed.append(pid)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    logger.warning(
                        "Cannot kill orphan sim_server pid=%d (permission denied) "
                        "— run `sudo kill -9 %d`",
                        pid, pid,
                    )
        if killed:
            logger.info(
                "Killed %d orphan sim_server.py process(es): %s",
                len(killed), killed,
            )
            time.sleep(0.5)
