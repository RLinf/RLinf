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

import logging
import os
import re
import shlex
import subprocess
import time
from enum import IntEnum
from typing import Optional

from .phone_proxy import PhoneEnv


# Commonly used key code constants
class KeyCode(IntEnum):
    """Android key code constants"""

    BACK = 4
    HOME = 3
    MENU = 82
    ENTER = 66
    POWER = 26
    VOLUME_UP = 24
    VOLUME_DOWN = 25
    CAMERA = 27
    SEARCH = 84


class ADBEnv(PhoneEnv):
    """ADB Env implementation based on `PhoneEnv`"""

    def __init__(self, device_id: Optional[str] = None, adb_path: str = "adb"):
        self.device_id = device_id
        self.adb_path = adb_path
        self.logger = logging.getLogger(__name__)

        if not self._check_adb():
            raise RuntimeError(
                "ADB is not available, please ensure ADB is installed and in PATH"
            )

        if not self._check_device():
            raise RuntimeError(
                f"Device not connected: {device_id if device_id else 'default device'}"
            )

    def _run_command(self, command: list[str], timeout: int = 10) -> tuple[bool, str]:
        try:
            cmd = [self.adb_path]
            if self.device_id:
                cmd.extend(["-s", self.device_id])
            cmd.extend(command)

            self.logger.debug(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                self.logger.warning(f"Command execution failed: {result.stderr}")
                return False, result.stderr.strip()

        except subprocess.TimeoutExpired:
            self.logger.error(f"Command execution timed out: {' '.join(command)}")
            return False, "Command execution timed out"
        except Exception as e:
            self.logger.error(f"Command execution exception: {str(e)}")
            return False, str(e)

    def _check_adb(self) -> bool:
        success, _ = self._run_command(["version"])
        return success

    def _check_device(self) -> bool:
        success, output = self._run_command(["devices"])
        if not success:
            return False

        lines = output.split("\n")[1:]  # Skip the first line "List of devices attached"
        devices = [
            line.split("\t")[0] for line in lines if line.strip() and "\tdevice" in line
        ]

        if not devices:
            self.logger.warning("No devices found in ADB devices output")

        if self.device_id:
            return self.device_id in devices
        else:
            return len(devices) > 0

    def get_device_info(self) -> dict[str, str]:
        info = {}

        success, model = self._run_command(["shell", "getprop", "ro.product.model"])
        if success:
            info["model"] = model

        success, version = self._run_command(
            ["shell", "getprop", "ro.build.version.release"]
        )
        if success:
            info["android_version"] = version

        success, sdk = self._run_command(["shell", "getprop", "ro.build.version.sdk"])
        if success:
            info["sdk_version"] = sdk

        success, resolution = self._run_command(["shell", "wm", "size"])
        if success:
            info["resolution"] = resolution

        success, device_id = self._run_command(["shell", "getprop", "ro.serialno"])
        if success:
            info["device_id"] = device_id

        return info

    def get_screen_size(self) -> tuple[int, int]:
        success, output = self._run_command(["shell", "wm", "size"])
        if success:
            try:
                size_str = output.split(":")[1].strip()
                width, height = map(int, size_str.split("x"))
                return width, height
            except (ValueError, IndexError) as e:
                self.logger.warning(
                    f"Failed to parse screen size: {output}, error: {e}"
                )
        return 0, 0

    def click(self, x: int, y: int, duration: float = 0.1) -> bool:
        success, _ = self._run_command(["shell", "input", "tap", str(x), str(y)])

        if success and duration > 0.1:
            time.sleep(duration)

        return success

    def long_press(self, x: int, y: int, duration: float = 1.0) -> bool:
        success, _ = self._run_command(
            [
                "shell",
                "input",
                "swipe",
                str(x),
                str(y),
                str(x),
                str(y),
                str(int(duration * 1000)),
            ]
        )
        return success

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> bool:
        success, _ = self._run_command(
            [
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(int(duration * 1000)),
            ]
        )
        return success

    def input_text(self, text: str) -> bool:
        escaped_text = text.replace(" ", "%s").replace("&", "\\&").replace("'", "\\'")
        success, _ = self._run_command(["shell", "input", "text", escaped_text])
        return success

    def input_keyevent(self, keycode: int) -> bool:
        success, _ = self._run_command(["shell", "input", "keyevent", str(keycode)])
        return success

    def back(self) -> bool:
        return self.input_keyevent(KeyCode.BACK)

    def home(self) -> bool:
        return self.input_keyevent(KeyCode.HOME)

    def menu(self) -> bool:
        return self.input_keyevent(KeyCode.MENU)

    def screenshot(self, save_path: str = "screenshot.png") -> bool:
        device_path = "/sdcard/screenshot_temp.png"
        success, _ = self._run_command(["shell", "screencap", "-p", device_path])

        if not success:
            return False

        try:
            success, _ = self._run_command(["pull", device_path, save_path])
            return success
        finally:
            self._run_command(["shell", "rm", device_path])

    def install_app(self, apk_path: str, replace: bool = True) -> bool:
        if not os.path.exists(apk_path):
            self.logger.error(f"APK file does not exist: {apk_path}")
            return False

        cmd = ["install"]
        if replace:
            cmd.append("-r")
        cmd.append(apk_path)

        success, output = self._run_command(cmd, timeout=60)
        return success and "Success" in output

    def uninstall_app(self, package_name: str) -> bool:
        success, output = self._run_command(["uninstall", package_name], timeout=30)
        return success and "Success" in output

    def start_app(self, package_name: str, activity_name: Optional[str] = None) -> bool:
        if activity_name:
            component = f"{package_name}/{activity_name}"
        else:
            component = package_name

        success, _ = self._run_command(["shell", "am", "start", "-n", component])
        return success

    def stop_app(self, package_name: str) -> bool:
        success, _ = self._run_command(["shell", "am", "force-stop", package_name])
        return success

    def get_current_activity(self) -> Optional[str]:
        success, output = self._run_command(
            ["shell", "dumpsys", "window", "windows"], timeout=10
        )

        if not success or not output:
            return None

        for line in output.split("\n"):
            if "mCurrentFocus" in line:
                try:
                    match = re.search(r"([a-zA-Z0-9_.]+/[a-zA-Z0-9_.]+)", line)
                    if match:
                        activity = match.group(1)
                        return activity.split("/")[-1]
                except (ValueError, AttributeError) as e:
                    self.logger.warning(
                        f"Failed to parse activity from line: {line}, error: {e}"
                    )

        return None

    def get_installed_packages(self) -> list[str]:
        success, output = self._run_command(["shell", "pm", "list", "packages"])

        if success:
            packages = []
            for line in output.split("\n"):
                if line.startswith("package:"):
                    packages.append(line.replace("package:", "").strip())
            return packages

        return []

    def push_file(self, local_path: str, remote_path: str) -> bool:
        if not os.path.exists(local_path):
            self.logger.error(f"Local file does not exist: {local_path}")
            return False

        success, _ = self._run_command(["push", local_path, remote_path], timeout=60)
        return success

    def pull_file(self, remote_path: str, local_path: str) -> bool:
        success, _ = self._run_command(["pull", remote_path, local_path], timeout=60)
        return success

    def shell_command(self, command: str) -> tuple[bool, str]:
        try:
            command_parts = shlex.split(command)
        except ValueError:
            self.logger.warning(
                f"Failed to parse command with shlex, using simple split: {command}"
            )
            command_parts = command.split()

        return self._run_command(["shell"] + command_parts)

    def wait_for_device(self, timeout: int = 30) -> bool:
        success, _ = self._run_command(["wait-for-device"], timeout=timeout)
        return success

    def reboot(self, mode: str = "system") -> bool:
        cmd_map = {
            "system": "reboot",
            "bootloader": "reboot-bootloader",
            "recovery": "reboot-recovery",
        }

        cmd = cmd_map.get(mode, "reboot")
        success, _ = self._run_command([cmd], timeout=10)
        return success
