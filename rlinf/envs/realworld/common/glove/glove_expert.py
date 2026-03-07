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

"""PSI data-glove expert that reads finger angles in a background thread.

The design mirrors :class:`SpaceMouseExpert` — a single daemon thread polls
the hardware driver and updates shared state behind a lock; the public
:meth:`get_angles` method is non-blocking and returns the latest reading.

The underlying ``PSIGloveStandalone`` driver is imported lazily so that
machines without the hardware dependency can still import the module
(useful for linting, tests, etc.).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

logger = get_logger()


class GloveExpert:
    """Non-blocking expert that streams finger angles from a PSI data-glove.

    Args:
        left_port: Serial device for the left-hand glove, e.g.
            ``"/dev/ttyACM0"``.  Pass ``None`` to disable.
        right_port: Serial device for the right-hand glove.  Pass ``None``
            to disable.
        frequency: Polling frequency in Hz passed to the driver.
        config_file: Path to the glove calibration YAML.  When ``None`` the
            driver's built-in default is used.

    Example::

        expert = GloveExpert(left_port="/dev/ttyACM0", right_port=None)
        angles = expert.get_angles()  # np.ndarray of shape (6,)
        expert.close()
    """

    _NUM_DOFS: int = 6

    def __init__(
        self,
        left_port: Optional[str] = "/dev/ttyACM0",
        right_port: Optional[str] = None,
        frequency: int = 60,
        config_file: Optional[str] = None,
    ) -> None:
        self._left_port = left_port
        self._right_port = right_port
        self._frequency = frequency
        self._config_file = config_file

        self._lock = threading.Lock()
        self._angles: np.ndarray = np.zeros(self._NUM_DOFS, dtype=np.float64)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_angles(self) -> np.ndarray:
        """Return the latest 6-D finger angles (normalised to [0, 1]).

        The order follows the glove driver convention:
        ``[thumb_side, thumb_back, index, middle, ring, pinky]``.
        """
        with self._lock:
            return self._angles.copy()

    def close(self) -> None:
        """Stop the reader thread and release resources."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Background reader — runs in a daemon thread."""
        try:
            controller = self._create_driver()
        except Exception:
            logger.error(
                "Failed to initialise PSI glove driver.  "
                "Glove expert will return zeros.",
                exc_info=True,
            )
            return

        interval = 1.0 / max(self._frequency, 1)
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                results = controller.get_hand_action()
                angles = self._extract_angles(results)
                if angles is not None:
                    with self._lock:
                        self._angles[:] = angles
            except Exception:
                logger.debug("Glove read error; keeping last values.", exc_info=True)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

        # Clean up
        try:
            controller.disconnect()
        except Exception:
            pass

    def _create_driver(self):
        """Lazily import and instantiate :class:`PSIGloveStandalone`."""
        try:
            from .psi_glove_driver.node import PSIGloveStandalone
        except ImportError as exc:
            raise ImportError(
                "Cannot import PSIGloveStandalone from the in-tree driver "
                "module (rlinf.envs.realworld.common.glove.psi_glove_driver)."
            ) from exc

        kwargs: dict = {
            "left_port": self._left_port,
            "right_port": self._right_port,
            "frequency": self._frequency,
            "auto_connect": True,
        }
        if self._config_file is not None:
            kwargs["config_file"] = self._config_file

        return PSIGloveStandalone(**kwargs)

    @staticmethod
    def _extract_angles(results: dict) -> Optional[np.ndarray]:
        """Pick the first available hand from the driver result dict."""
        for hand_key in ("left", "right"):
            if hand_key in results:
                processed = results[hand_key].get("processed")
                if processed is not None:
                    return np.asarray(processed, dtype=np.float64)
        return None


if __name__ == "__main__":

    def _test_glove() -> None:
        expert = GloveExpert(left_port="/dev/ttyACM0", right_port=None, frequency=60)
        with np.printoptions(precision=3, suppress=True):
            while True:
                angles = expert.get_angles()
                print(f"Glove angles: {angles}")
                time.sleep(0.1)

    _test_glove()
