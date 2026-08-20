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

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


class EmbodiedRuntimeCLI:
    """Client for the embodied-runtime command-line tools."""

    _BIN_ENV = {
        "camctr": "RLINF_EMBODIED_CAMERA_CLI",
        "rosctr": "RLINF_EMBODIED_ROS_CLI",
    }
    _ENABLED_ENV = {
        "camctr": "RLINF_EMBODIED_CAMERA_ENABLED",
        "rosctr": "RLINF_EMBODIED_ROS_ENABLED",
    }

    def __init__(self, tool: str, executable: Optional[str] = None):
        """Initialize a client for ``camctr`` or ``rosctr``."""
        if tool not in self._BIN_ENV:
            raise ValueError(f"Unsupported embodied-runtime tool: {tool}")
        self.tool = tool
        resolved_executable = executable or self.find_executable(tool)
        if resolved_executable is None:
            raise FileNotFoundError(f"Could not find embodied-runtime CLI '{tool}'.")
        self.executable = resolved_executable

    @classmethod
    def find_executable(cls, tool: str) -> Optional[str]:
        """Find an embodied-runtime CLI executable."""
        if tool not in cls._BIN_ENV:
            raise ValueError(f"Unsupported embodied-runtime tool: {tool}")
        explicit = os.environ.get(cls._BIN_ENV[tool])
        if explicit:
            return explicit if Path(explicit).is_file() else shutil.which(explicit)

        executable = shutil.which(tool)
        if executable:
            return executable

        runtime_path = os.environ.get("RLINF_EMBODIED_PATH")
        if runtime_path:
            candidate = Path(runtime_path) / tool
            if candidate.is_file():
                return str(candidate)
        for directory in ("/opt/rlinf/bin", "/usr/local/bin"):
            candidate = Path(directory) / tool
            if candidate.is_file():
                return str(candidate)
        return None

    @classmethod
    def is_installed(cls, tool: str) -> bool:
        """Return whether an embodied-runtime CLI is installed."""
        return cls.find_executable(tool) is not None

    @classmethod
    def is_enabled(cls, tool: str) -> bool:
        """Return whether the corresponding runtime controller is enabled."""
        if tool not in cls._ENABLED_ENV:
            raise ValueError(f"Unsupported embodied-runtime tool: {tool}")
        if os.environ.get("RLINF_EMBODIED_RUNTIME_ENABLED"):
            return os.environ.get(cls._ENABLED_ENV[tool]) == "1"
        return cls.is_installed(tool)

    @classmethod
    def is_available(cls, tool: str) -> bool:
        """Return whether the CLI and its controller socket are available."""
        if not cls.is_enabled(tool):
            return False
        executable = cls.find_executable(tool)
        if executable is None:
            return False
        try:
            subprocess.run(
                [executable, "list", "-o", "json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return True

    def run_json(self, *args: str, timeout: float = 10) -> dict[str, Any]:
        """Run a CLI command and decode its JSON output."""
        result = self._run(*args, "-o", "json", timeout=timeout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"embodied-runtime command '{self.tool} {' '.join(args)}' "
                "returned invalid JSON."
            ) from exc

    def run(self, *args: str, timeout: float = 10) -> None:
        """Run a CLI command and require a successful exit status."""
        self._run(*args, timeout=timeout)

    def _run(self, *args: str, timeout: float) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                [self.executable, *args],
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            stderr = stderr.strip()
            raise RuntimeError(
                f"embodied-runtime command '{self.tool} {' '.join(args)}' failed: "
                f"{stderr or exc.returncode}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"embodied-runtime command '{self.tool} {' '.join(args)}' "
                f"timed out after {timeout} seconds."
            ) from exc

    def list_cameras(self) -> list[dict[str, Any]]:
        """List cameras registered with camera-controller."""
        return self.run_json("list").get("cameras", [])

    def list_robots(self) -> list[dict[str, Any]]:
        """List robots registered with ros-controller."""
        return self.run_json("list").get("robots", [])

    def resolve_camera_id(self, identifier: str) -> str:
        """Resolve a camera ID from an ID, serial number, or name."""
        cameras = self.list_cameras()
        matches = [
            camera["cameraId"]
            for camera in cameras
            if identifier
            in (
                camera.get("cameraId"),
                camera.get("serialNumber"),
                camera.get("name"),
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Camera {identifier!r} is ambiguous in embodied-runtime. "
                f"Matching camera IDs: {matches}."
            )
        available = [camera.get("cameraId") for camera in cameras]
        raise ValueError(
            f"Camera {identifier!r} is not managed by embodied-runtime. "
            f"Available camera IDs: {available}."
        )

    def resolve_robot_id(self, robot_ip: Optional[str] = None) -> str:
        """Resolve a robot ID from its IP or an unambiguous inventory."""
        robots = self.list_robots()
        if robot_ip:
            matches = [
                robot["robotId"]
                for robot in robots
                if robot.get("params", {}).get("robot_ip") == robot_ip
            ]
            if len(matches) == 1:
                return matches[0]
            available = [robot.get("robotId") for robot in robots]
            raise ValueError(
                f"Could not select an embodied-runtime robot for "
                f"robot_ip={robot_ip!r}. Available robot IDs: {available}."
            )
        if len(robots) == 1:
            return robots[0]["robotId"]
        available = [robot.get("robotId") for robot in robots]
        raise ValueError(
            f"Could not select an embodied-runtime robot for robot_ip={robot_ip!r}. "
            f"Available robot IDs: {available}."
        )
