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
import select
import threading
from collections import deque
from typing import Any

from rlinf.utils.logging import get_logger

_logger = get_logger()


class KeyboardListener:
    """Read operator commands from a Linux input device."""

    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_Q")

    def __init__(self) -> None:
        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}
        self._press_events: deque[str] = deque()
        self._stop = threading.Event()
        self.device = None
        self.listener = None
        self.last_intervene = 0

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

        self.device = self._open_keyboard_device()

        self.listener = threading.Thread(
            target=self._listen_loop,
            name=f"KeyboardListener:{self.device.path}",
            daemon=True,
        )
        self.listener.start()

    def _open_keyboard_device(self) -> Any:
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

    def _open_device(self, device_path: str, is_override: bool = False) -> Any:
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

    def _is_keyboard_device(self, device: Any) -> bool:
        required_codes = {
            getattr(self._ecodes, key_name) for key_name in self.REQUIRED_KEY_NAMES
        }
        capabilities = device.capabilities(verbose=False)
        supported_key_codes = set(capabilities.get(self._ecodes.EV_KEY, []))
        return required_codes.issubset(supported_key_codes)

    def _listen_loop(self) -> None:
        # Retain the path so the listener can recover after USB disconnects.
        device_path = self.device.path
        try:
            while not self._stop.is_set():
                try:
                    readable, _, _ = select.select([self.device], [], [], 0.1)
                    if not readable:
                        continue
                    for event in self.device.read():
                        if event.type != self._ecodes.EV_KEY:
                            continue
                        key = self._event_to_key(event.code)
                        if key is None:
                            continue
                        with self.state_lock:
                            if event.value == 1:
                                self.latest_data["key"] = key
                                self._press_events.append(key)
                            elif event.value == 2:
                                self.latest_data["key"] = key
                            elif event.value == 0 and self.latest_data["key"] == key:
                                self.latest_data["key"] = None
                except BlockingIOError:
                    continue
                except OSError as exc:
                    if exc.errno != errno.ENODEV:
                        _logger.exception("Keyboard device %s read failed", device_path)
                        return
                    _logger.warning(
                        "Keyboard device %s disconnected; reopening", device_path
                    )
                    with self.state_lock:
                        self.latest_data["key"] = None
                        self._press_events.clear()
                    self.device.close()
                    while not self._stop.wait(0.5):
                        try:
                            self.device = self._input_device_cls(device_path)
                            break
                        except OSError:
                            continue
                    else:
                        return
                    _logger.info("Keyboard device %s reopened.", device_path)
        finally:
            self.device.close()

    def close(self) -> None:
        """Stop listening and reconnecting, then release the input device."""
        self._stop.set()
        assert self.listener is not None
        self.listener.join()
        with self.state_lock:
            self.latest_data["key"] = None
            self._press_events.clear()

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
        """Return the currently held key, or ``None``.

        Only reflects held state; fast taps may be missed between polls.
        Use :meth:`pop_pressed_keys` when you need lossless press detection.
        """
        with self.state_lock:
            return self.latest_data["key"]

    def pop_pressed_keys(self) -> list[str]:
        """Return and clear key presses recorded since the previous call.

        Key-repeat events are omitted. The operation is thread-safe and
        non-blocking.
        """
        with self.state_lock:
            if not self._press_events:
                return []
            pressed = list(self._press_events)
            self._press_events.clear()
            return pressed
