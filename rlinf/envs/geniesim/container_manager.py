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
# SimContainerManager — manages the GenieSim simulation Docker container lifecycle.
#
# Architecture:
#   HOST  GenieSimBaseEnv ──(SHM via --ipc=host)──┐
#                                                   │
#   CONTAINER  sim_server.py → ProcessManager → Isaac Sim + MuJoCo ───┘
#
# Handshake: the container writes {geniesim_root}/.geniesim_ready when all
# simulation processes are up.  The host polls this file.
#
# For running sim_server.py as a local subprocess (inside the merged container),
# use SimProcessManager (process_manager.py) instead.
#

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from rlinf.envs.geniesim.sim_manager_base import (
    BaseSimManager,
    ContainerStartupError,   # backward-compat re-export
    SimStartupError,
    _SIM_CFG_FILE,
    _READY_FILE,
)

__all__ = ["SimContainerManager", "ContainerStartupError"]


class SimContainerManager(BaseSimManager):
    """
    Manages the lifecycle of the GenieSim simulation Docker container.

    Typical call sequence
    ---------------------
    mgr = SimContainerManager(container_cfg)
    mgr.ensure_running(pm_kwargs)   # start + handshake
    # ... training ...
    mgr.shutdown()                  # graceful stop

    Readiness handshake
    -------------------
    The container's sim_server.py writes {geniesim_root}/.geniesim_ready
    after all MuJoCo envs and the Isaac Sim renderer are ready.  The host
    polls for this file.

    Container reuse
    ---------------
    When reuse_running=True (default), a running container whose ready file
    already exists is reused without restart, making repeated
    GenieSimBaseEnv instantiations fast (e.g. during training restarts).

    Parameters (container_cfg)
    --------------------------
    image            (str)  Docker image name
    name             (str)  Container name
    geniesim_root    (str)  Path to rlinf_open_source/ on the host
    keep_alive       (bool) Don't stop container on shutdown()   [default False]
    reuse_running    (bool) Reuse already-running sim            [default True]
    startup_timeout_sec (int)                                    [default 300]
    ros_domain_id    (int) ROS_DOMAIN_ID                         [default 0]
    extra_docker_args (list) Extra `docker run` flags            [default []]
    container_paths  (dict) Override paths inside container      [default {}]
    isaac_cache_root (str)  Isaac Sim cache directory on host    [default ~/docker/isaac-sim]
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
        self.image = str(getattr(container_cfg, "image", ""))
        self.name  = str(getattr(container_cfg, "name", "geniesim_local"))
        self.extra_docker_args: List[str] = list(
            getattr(container_cfg, "extra_docker_args", []) or []
        )
        self.container_paths: Dict[str, str] = dict(
            getattr(container_cfg, "container_paths", {}) or {}
        )
        _icr = getattr(container_cfg, "isaac_cache_root", None)
        self.isaac_cache_root: Path = (
            Path(_icr).expanduser().resolve() if _icr
            else Path.home() / "docker" / "isaac-sim"
        )
        self._container_geniesim_root: str = str(
            getattr(container_cfg, "container_geniesim_root", "/geniesim/main")
        )
        self._headless: bool = True  # updated in ensure_running()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def ensure_running(self, pm_kwargs: Dict) -> None:
        """
        Ensure the simulation container is running and ready.

        Reuse logic (when reuse_running=True):
          - Running + ready file + SHM accessible → reuse immediately.
          - Running + idle file → send restart signal via sentinel.
          - Running but neither ready nor idle → stop and restart fresh.
          - Not running → start fresh.
        """
        self._check_docker_available()

        status = self._get_status()

        if status == "running" and self.reuse_running:
            if self._ready_file.exists():
                shm_name = pm_kwargs.get("shm_name", "geniesim_frames")
                if self._check_shm_accessible(shm_name):
                    logger.info("Reusing ready container '%s'", self.name)
                    return
                logger.warning(
                    "Ready file exists but SHM '%s' is not accessible — "
                    "restarting container (stale ready file)...", shm_name,
                )
            elif self._idle_file.exists():
                logger.info(
                    "Container '%s' is idle — restarting sim processes...", self.name,
                )
                shm_name = pm_kwargs.get("shm_name", "geniesim_frames")
                self._cleanup_stale_shm(shm_name)
                self._cleanup_stale_ctrl_shms(shm_name)
                self._cleanup_stale_step_shm(shm_name)
                self._idle_file.unlink(missing_ok=True)
                self._ready_file.unlink(missing_ok=True)
                self._progress_file.unlink(missing_ok=True)
                self._error_file.unlink(missing_ok=True)
                self._headless = bool(pm_kwargs.get("headless", True))
                self._write_sim_config(pm_kwargs)
                self._start_file.write_text("start")
                self._wait_ready()
                return
            else:
                logger.warning(
                    "Container '%s' is running but has no ready file — "
                    "stopping for a fresh start with sim_server.py...", self.name,
                )
            self._run_docker(["stop", "--time", "10", self.name], check=False)
            self._run_docker(["rm", "-f", self.name], check=False)
            status = None

        if status in ("running", "paused", "created"):
            logger.info("Stopping existing container '%s'...", self.name)
            self._run_docker(["stop", "--time", "10", self.name], check=False)

        if status is not None:
            self._run_docker(["rm", "-f", self.name], check=False)

        shm_name = pm_kwargs.get("shm_name", "geniesim_frames")
        self._ready_file.unlink(missing_ok=True)
        self._error_file.unlink(missing_ok=True)
        self._cleanup_stale_shm(shm_name)
        self._cleanup_stale_ctrl_shms(shm_name)
        self._cleanup_stale_step_shm(shm_name)
        self._headless = bool(pm_kwargs.get("headless", True))
        self._progress_file.unlink(missing_ok=True)
        self._write_sim_config(pm_kwargs)
        self._start_container()
        self._wait_ready()

    def health_check(self) -> bool:
        """Return True if the container is running and the sim is live."""
        if self._get_status() != "running":
            return False
        return self._ready_file.exists()

    def shutdown(self) -> None:
        """
        Gracefully stop the simulation container.

        If keep_alive=True: sim processes are stopped via sentinel file but the
        container stays alive so the next run can restart without a cold start.

        If keep_alive=False: the container is stopped with SIGTERM → SIGKILL
        (15 s grace period) and the ready file is cleaned up.
        """
        if self.keep_alive:
            if self._get_status() != "running":
                return
            if self._idle_file.exists():
                return
            logger.info(
                "Stopping sim processes (container '%s' stays alive for quick restart)...",
                self.name,
            )
            self._stop_file.write_text("stop")
            deadline = time.time() + 60
            while time.time() < deadline:
                if self._idle_file.exists():
                    logger.info("Sim processes stopped. Container '%s' is idle.", self.name)
                    return
                time.sleep(0.5)
            logger.warning(
                "Sim processes may not have stopped cleanly "
                "(60s timeout waiting for idle state)."
            )
            return

        logger.info("Shutting down simulation processes...")
        self._ready_file.unlink(missing_ok=True)
        if self._get_status() == "running":
            logger.info("Stopping container '%s'...", self.name)
            self._run_docker(["stop", "--time", "15", self.name], check=False)
            logger.info("Container stopped.")

    # ---------------------------------------------------------------------- #
    # BaseSimManager abstract hooks
    # ---------------------------------------------------------------------- #

    def _get_log_tail(self, tail: int = 40) -> str:
        return self._get_logs(tail=tail)

    def _check_alive(self) -> Tuple[bool, str]:
        status = self._get_status()
        if status != "running":
            return (
                False,
                f"Container '{self.name}' exited unexpectedly (status={status!r}).",
            )
        return True, ""

    def _classify_failure(self) -> str:
        logs = self._get_logs(tail=200)
        return self._scan_logs_for_failure(logs, "Last container logs")

    def _error_hint(self) -> str:
        return (
            f"Check renderer.log inside the container:\n"
            f"  docker exec {self.name} cat /tmp/geniesim_logs/renderer.log | tail -50"
        )

    # ---------------------------------------------------------------------- #
    # Override: _write_sim_config applies container path translation
    # ---------------------------------------------------------------------- #

    def _write_sim_config(self, pm_kwargs: Dict) -> None:
        cfg = dict(pm_kwargs)
        host_root = str(self.geniesim_root)
        container_root = self._container_geniesim_root
        _PATH_KEYS = ("mjcf_path", "scene_usd", "robot_usd", "task_file")
        for key in _PATH_KEYS:
            val = cfg.get(key, "")
            if val and host_root in val:
                cfg[key] = val.replace(host_root, container_root)
        if not cfg.get("isaac_python"):
            cfg["isaac_python"] = "/isaac-sim/python.sh"
        if not cfg.get("mujoco_python"):
            cfg["mujoco_python"] = "python3"
        cfg.update(self.container_paths)
        self._cfg_file.write_text(json.dumps(cfg, indent=2))

    # ---------------------------------------------------------------------- #
    # Override: _cleanup_stale_shm — adds Docker uid-escalation fallback
    # ---------------------------------------------------------------------- #

    def _cleanup_stale_shm(self, shm_name: str) -> None:
        """
        Remove stale POSIX SHM.  When direct unlink fails due to permissions
        (SHM owned by the container user uid 1234), a temporary container
        invocation is used so uid 1234 can remove the file.
        """
        shm_path = Path("/dev/shm") / shm_name
        if not shm_path.exists():
            return
        try:
            shm_path.unlink()
            logger.info("Removed stale SHM '%s'", shm_name)
            return
        except PermissionError:
            pass

        owner_uid = shm_path.stat().st_uid
        logger.info(
            "Stale SHM '%s' owned by uid %d — removing via temporary container...",
            shm_name, owner_uid,
        )
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--ipc", "host",
                "--user", str(owner_uid),
                "--entrypoint", "/bin/rm",
                self.image,
                f"/dev/shm/{shm_name}",
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Stale SHM '%s' removed via container", shm_name)
        else:
            logger.warning(
                "Could not remove stale SHM '%s': %s — "
                "run `sudo rm /dev/shm/%s` if startup fails",
                shm_name, result.stderr.strip(), shm_name,
            )

    def _cleanup_stale_ctrl_shms(self, shm_name: str) -> None:
        """
        Remove stale per-env ctrl SHM segments.  Uses Docker uid-escalation
        fallback when direct unlink fails.
        """
        shm_dir = Path("/dev/shm")
        prefix = f"{shm_name}_ctrl_"
        for entry in shm_dir.iterdir():
            if not entry.name.startswith(prefix):
                continue
            try:
                entry.unlink()
                logger.info("Removed stale ctrl SHM '%s'", entry.name)
                continue
            except PermissionError:
                pass

            owner_uid = entry.stat().st_uid
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--ipc", "host",
                    "--user", str(owner_uid),
                    "--entrypoint", "/bin/rm",
                    self.image,
                    f"/dev/shm/{entry.name}",
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info("Removed stale ctrl SHM '%s' via container", entry.name)
            else:
                logger.warning(
                    "Could not remove ctrl SHM '%s': %s",
                    entry.name, result.stderr.strip(),
                )

    # ---------------------------------------------------------------------- #
    # Container startup
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _ensure_vulkan_icd() -> None:
        """
        Verify that the Vulkan ICD paths in the CDI spec exist on the host.

        The CDI spec (/var/run/cdi/nvidia.yaml) is generated by nvidia-ctk and
        may embed stale paths.  If referenced ICD files are absent, runc fails
        at container init.  This method attempts to regenerate the CDI spec or
        create a symlink as a fallback.
        """
        _CDI_SPEC = Path("/var/run/cdi/nvidia.yaml")
        _ICD_CANDIDATES = [
            Path("/etc/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/local/share/vulkan/icd.d/nvidia_icd.json"),
        ]

        if not _CDI_SPEC.exists():
            return
        try:
            spec_text = _CDI_SPEC.read_text()
        except OSError:
            return

        import re as _re
        missing = [
            p for p in _re.findall(r"hostPath:\s*(\S+nvidia_icd[^\s]*)", spec_text)
            if not Path(p).exists()
        ]
        if not missing:
            return

        valid_icd = next((p for p in _ICD_CANDIDATES if p.exists()), None)
        if valid_icd is None:
            logger.warning(
                "Vulkan ICD paths in CDI spec are missing (%s) and no valid ICD found "
                "in standard locations. GPU rendering may fail.", missing,
            )
            return

        logger.info(
            "CDI spec references missing Vulkan ICD path(s): %s\n"
            "Valid ICD found at: %s\n"
            "Regenerating CDI spec via: sudo nvidia-ctk cdi generate --output /var/run/cdi/nvidia.yaml",
            missing, valid_icd,
        )
        result = subprocess.run(
            ["sudo", "nvidia-ctk", "cdi", "generate", "--output", str(_CDI_SPEC)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("CDI spec regenerated successfully.")
            return

        logger.warning(
            "CDI spec regeneration failed (rc=%d): %s — falling back to symlink creation...",
            result.returncode, result.stderr.strip(),
        )
        icd_dir = Path("/etc/vulkan/icd.d")
        icd_link = icd_dir / "nvidia_icd.json"
        r2 = subprocess.run(["sudo", "mkdir", "-p", str(icd_dir)], capture_output=True, text=True)
        if r2.returncode == 0:
            r3 = subprocess.run(
                ["sudo", "ln", "-s", str(valid_icd), str(icd_link)],
                capture_output=True, text=True,
            )
            if r3.returncode == 0:
                logger.info("Symlink created: %s -> %s", icd_link, valid_icd)
                return
        logger.warning(
            "Automatic fix failed. To fix manually, run ONE of:\n"
            "  sudo nvidia-ctk cdi generate --output /var/run/cdi/nvidia.yaml\n"
            "Or:\n"
            "  sudo mkdir -p /etc/vulkan/icd.d\n"
            "  sudo ln -s %s /etc/vulkan/icd.d/nvidia_icd.json",
            valid_icd,
        )

    def _start_container(self) -> None:
        self._ensure_vulkan_icd()
        self._run_docker_start()

    def _run_docker_start(self) -> None:
        mounts: List[str] = [
            f"{self.geniesim_root}:/geniesim/main:rw",
        ]

        _isaac_cache_map = {
            "pkg":                "/isaac-sim/.local/share/ov/pkg",
            "cache/main":         "/isaac-sim/.cache",
            "cache/computecache": "/isaac-sim/.nv/ComputeCache",
            "logs":               "/isaac-sim/.nvidia-omniverse/logs",
            "config":             "/isaac-sim/.nvidia-omniverse/config",
            "data":               "/isaac-sim/.local/share/ov/data",
        }
        for rel, container_dst in _isaac_cache_map.items():
            host_path = self.isaac_cache_root / rel
            if host_path.exists():
                mounts.append(f"{host_path}:{container_dst}:rw")

        env_vars: List[str] = [
            "SIM_REPO_ROOT=/geniesim/main",
            f"ROS_DOMAIN_ID={self.ros_domain_id}",
            "ROS_LOCALHOST_ONLY=1",
            "GENIESIM_ROS_WS_INSTALL=/geniesim/ros_ws_build/install",
        ]

        sim_server_cmd = (
            f"python3 /geniesim/main/source/geniesim/rl/scripts/sim_server.py"
            f" --config-json /geniesim/main/{_SIM_CFG_FILE}"
            f" --ready-file /geniesim/main/{_READY_FILE}"
        )

        cmd: List[str] = [
            "run", "--detach",
            "--name", self.name,
            "--network", "host",
            "--ipc",     "host",
            "--gpus",    "all",
            "--device",  "/dev/input:/dev/input",
        ]

        _display = os.environ.get("DISPLAY", "")
        if not self._headless:
            if _display:
                cmd += ["-e", f"DISPLAY={_display}",
                        "-v", "/tmp/.X11-unix:/tmp/.X11-unix"]
                subprocess.run(["xhost", "+local:docker"], capture_output=True, check=False)
            else:
                logger.warning(
                    "headless=false but $DISPLAY is not set. "
                    "Isaac Sim 3D window will not appear.  Set the DISPLAY env var "
                    "(e.g. export DISPLAY=:0) before launching, or set headless=true.",
                )

        for m in mounts:
            cmd += ["-v", m]
        for e in env_vars:
            cmd += ["-e", e]
        cmd += self.extra_docker_args
        cmd += [
            self.image,
            "/entrypoint_geniesim_rlinf.sh",
            "bash", "-c", sim_server_cmd,
        ]

        _vis = "headless" if self._headless else f"display={_display or '(no $DISPLAY)'}"
        logger.info(
            "Starting '%s' from image '%s'  network=host  ipc=host  gpus=all  mode=%s",
            self.name, self.image, _vis,
        )
        self._run_docker(cmd)

    # ---------------------------------------------------------------------- #
    # Docker helpers
    # ---------------------------------------------------------------------- #

    def _run_docker(self, args: List[str], check: bool = True) -> str:
        result = subprocess.run(
            ["docker"] + args,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise SimStartupError(
                f"`docker {args[0]}` failed (rc={result.returncode}):\n"
                + (result.stderr or result.stdout or "(no output)").strip()
            )
        return result.stdout.strip()

    def _get_status(self) -> Optional[str]:
        """Return container State.Status or None if the container doesn't exist."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", self.name],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _get_logs(self, tail: int = 150) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self.name],
            capture_output=True,
            text=True,
        )
        return (result.stdout + result.stderr).strip()

    @staticmethod
    def _check_docker_available() -> None:
        result = subprocess.run(["docker", "info"], capture_output=True)
        if result.returncode != 0:
            raise SimStartupError(
                "Docker daemon is not running or not accessible. "
                "Start Docker and ensure the current user is in the `docker` group."
            )
