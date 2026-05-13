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

import errno
import os
import threading
import time
from collections import deque


class KeyboardListener:
    """Headless keyboard listener backed by Linux evdev input devices."""

    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_Q")

    def __init__(self):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as exc:
            raise RuntimeError(
                "KeyboardListener requires the 'evdev' package. "
                "Install the real-world extras with evdev support."
            ) from exc

        self._input_device_cls = InputDevice
        self._ecodes = ecodes
        self._list_devices = list_devices

        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}
        # Edge-press queue so a sub-period tap isn't missed (get_key() only reports the held key).
        self._press_events: deque[str] = deque()
        self.device = self._open_keyboard_device()

        self.listener = threading.Thread(
            target=self._listen_loop,
            name=f"KeyboardListener:{self.device.path}",
            daemon=True,
        )
        self.listener.start()
        self.last_intervene = 0

    def _open_keyboard_device(self):
        override_path = os.environ.get("RLINF_KEYBOARD_DEVICE")
        if override_path:
            device = self._open_device(override_path, is_override=True)
            if not self._is_keyboard_device(device):
                device.close()
                raise RuntimeError(
                    "KeyboardListener device set by "
                    f"RLINF_KEYBOARD_DEVICE='{override_path}' does not look like a "
                    "keyboard device. Point it to the correct /dev/input/eventX path."
                )
            return device

        permission_denied_paths: list[str] = []
        keyboards: list = []  # (path, device) for every device that has KEY_A/B/C/Q
        for device_path in sorted(self._list_devices()):
            try:
                device = self._open_device(device_path)
            except PermissionError:
                permission_denied_paths.append(device_path)
                continue

            if self._is_keyboard_device(device):
                keyboards.append((device_path, device))
            else:
                device.close()

        if len(keyboards) == 1:
            return keyboards[0][1]
        if len(keyboards) > 1:
            for _, dev in keyboards:
                dev.close()
            listing = "\n".join(
                f"  {path}  name={dev.name!r}" for path, dev in keyboards
            )
            raise RuntimeError(
                "Multiple keyboard-capable devices on /dev/input/event*; "
                "set RLINF_KEYBOARD_DEVICE to the intended one (prefer a "
                "/dev/input/by-id/... path so it survives reboots).\n"
                f"Candidates:\n{listing}"
            )

        if permission_denied_paths:
            denied = ", ".join(permission_denied_paths)
            raise RuntimeError(
                "KeyboardListener could not open any readable keyboard device under "
                f"/dev/input/event*. Permission denied for: {denied}. Grant the runtime "
                "user read access via the input group or udev rules, or set "
                "RLINF_KEYBOARD_DEVICE to a readable keyboard event device."
            )

        raise RuntimeError(
            "KeyboardListener could not find a readable keyboard device under "
            "/dev/input/event*. Ensure a physical keyboard is connected, the runtime "
            "user has access to input devices, or set RLINF_KEYBOARD_DEVICE to the "
            "correct /dev/input/eventX path."
        )

    def _open_device(self, device_path: str, is_override: bool = False):
        try:
            return self._input_device_cls(device_path)
        except FileNotFoundError as exc:
            if is_override:
                raise RuntimeError(
                    f"KeyboardListener override path '{device_path}' does not exist."
                ) from exc
            raise
        except PermissionError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener cannot read the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}'. Grant the runtime user "
                    "read access via the input group or udev rules."
                ) from exc
            raise
        except OSError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener failed to open the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}': {exc}"
                ) from exc
            raise RuntimeError(
                f"KeyboardListener failed to open input device '{device_path}': {exc}"
            ) from exc

    def _is_keyboard_device(self, device) -> bool:
        required_codes = {
            getattr(self._ecodes, key_name) for key_name in self.REQUIRED_KEY_NAMES
        }
        capabilities = device.capabilities(verbose=False)
        supported_key_codes = set(capabilities.get(self._ecodes.EV_KEY, []))
        return required_codes.issubset(supported_key_codes)

    def _listen_loop(self) -> None:
        # Cache path so we can reopen after a USB hiccup (errno=19 ENODEV); pedal shares a flaky bus with Lumos.
        device_path = self.device.path
        while True:
            try:
                for event in self.device.read_loop():
                    if event.type != self._ecodes.EV_KEY:
                        continue

                    key = self._event_to_key(event.code)
                    if key is None:
                        continue

                    if event.value == 1:
                        # Initial press only; autorepeat (value==2) does not re-enqueue.
                        with self.state_lock:
                            self.latest_data["key"] = key
                            self._press_events.append(key)
                    elif event.value == 2:
                        with self.state_lock:
                            self.latest_data["key"] = key
                    elif event.value == 0:
                        with self.state_lock:
                            if self.latest_data["key"] == key:
                                self.latest_data["key"] = None
            except OSError as exc:
                if exc.errno != errno.ENODEV:
                    raise
                with self.state_lock:
                    self.latest_data["key"] = None
                try:
                    self.device.close()
                except Exception:
                    pass
                # Reopen forever; daemon thread dies with the process.
                while True:
                    time.sleep(0.5)
                    try:
                        self.device = self._input_device_cls(device_path)
                        break
                    except (FileNotFoundError, OSError):
                        continue

    def _event_to_key(self, key_code: int) -> str | None:
        key_name = self._ecodes.bytype[self._ecodes.EV_KEY].get(key_code)
        if isinstance(key_name, list):
            key_name = key_name[0]
        if not isinstance(key_name, str):
            return None

        if key_name.startswith("KEY_"):
            normalized_key = key_name.removeprefix("KEY_").lower()
            if len(normalized_key) == 1:
                return normalized_key
            return f"Key.{normalized_key}"
        return key_name.lower()

    def get_key(self) -> str | None:
        """Return the currently-held key, or None.

        Only reflects held state; fast taps may be missed between polls.
        Use :meth:`pop_pressed_keys` when you need lossless press detection.
        """
        with self.state_lock:
            return self.latest_data["key"]

    def pop_pressed_keys(self) -> list[str]:
        """Drain and return every key that has seen an initial press since the
        last call.  Autorepeat is collapsed to a single entry per physical
        keystroke.  Thread-safe and non-blocking.
        """
        with self.state_lock:
            if not self._press_events:
                return []
            pressed = list(self._press_events)
            self._press_events.clear()
            return pressed
