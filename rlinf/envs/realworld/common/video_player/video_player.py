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

import os
import queue
import threading
import warnings

import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, enable: bool = True):
        self.queue = queue.Queue()
        self.is_running = False
        if not enable:
            return
        self._run_thread = threading.Thread(target=self._play, daemon=True)
        self._run_thread.start()

    def put_frame(self, frame):
        if self.is_running:
            self.queue.put(frame)

    def stop(self):
        """Stop the video player and release X11 resources."""
        if not self.is_running:
            return
        self.is_running = False
        # Drain the queue then send exit signal
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(None)
        if hasattr(self, "_run_thread"):
            self._run_thread.join(timeout=3)

    def _play(self):
        display = os.environ.get("DISPLAY")
        if not display:
            # Try common fallback values for headful environments
            for candidate in [":0", ":1", ":4"]:
                try:
                    os.environ["DISPLAY"] = candidate
                    # Quick test: can we open a window?
                    import subprocess
                    ret = subprocess.run(
                        ["xdpyinfo"], capture_output=True, timeout=2,
                    )
                    if ret.returncode == 0:
                        display = candidate
                        break
                except Exception:
                    continue

            if not display:
                warnings.warn(
                    "No display found. VideoPlayer will not run. Set DISPLAY environment variable to enable."
                )
                return

        self.is_running = True
        try:
            while True:
                img_array = self.queue.get()  # retrieve an image from the queue
                if img_array is None:  # None is our signal to exit
                    break

                frame = np.concatenate(
                    [v for k, v in img_array.items() if "full" not in k], axis=0
                )

                cv2.imshow("RealSense Cameras", frame)
                cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process pending GUI events after destroy
