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

import threading


class KeyboardListener:
    def __init__(self):
        from pynput import keyboard

        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}

        self.listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.listener.start()
        self.last_intervene = 0

    def on_key_press(self, key):
        with self.state_lock:
            self.latest_data["key"] = key.char if hasattr(key, "char") else str(key)

    def on_key_release(self, key):
        with self.state_lock:
            self.latest_data["key"] = None

    def get_key(self) -> str | None:
        """Returns the latest key pressed."""
        with self.state_lock:
            return self.latest_data["key"]
