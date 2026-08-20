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

"""Franka parallel-jaw gripper via ``franky.Gripper`` (libfranka, no ROS)."""

from __future__ import annotations

import numpy as np

from rlinf.utils.logging import get_logger

from .base_gripper import BaseGripper

# Match ROS FrankaGripper.open width / close grasp target.
_MAX_WIDTH_M = 0.09
_CLOSE_WIDTH_M = 0.01
# franky expects m/s; map normalized [0, 1] into a practical band.
_MIN_SPEED_MS = 0.01
_MAX_SPEED_MS = 0.1
_OPEN_WIDTH_THRESHOLD_M = 0.05
_DEFAULT_GRASP_FORCE_N = 40.0


def _normalized_speed_to_ms(speed: float) -> float:
    s = float(np.clip(speed, 0.0, 1.0))
    return _MIN_SPEED_MS + s * (_MAX_SPEED_MS - _MIN_SPEED_MS)


class FrankyFrankaGripper(BaseGripper):
    """Franka Emika Hand controlled through ``franky.Gripper``.

    Args:
        robot_ip: Same IP as the arm (libfranka gripper endpoint).
        max_width: Fully-open width in metres (default 0.09).
        grasp_force: Default grasp force in Newtons.
    """

    def __init__(
        self,
        robot_ip: str,
        max_width: float = _MAX_WIDTH_M,
        grasp_force: float = _DEFAULT_GRASP_FORCE_N,
    ):
        import franky

        self._logger = get_logger()
        self._max_width = float(max_width)
        self._grasp_force = float(grasp_force)
        self._gripper = franky.Gripper(robot_ip)
        self._is_open_flag = True
        self._logger.info("FrankyFrankaGripper connected at %s", robot_ip)

    def _speed_ms(self, speed: float) -> float:
        return _normalized_speed_to_ms(speed)

    def _stop_quiet(self) -> None:
        try:
            self._gripper.stop()
        except Exception:
            pass

    def open(self, speed: float = 0.3) -> None:
        try:
            self._gripper.open(self._speed_ms(speed))
        except Exception as exc:
            # libfranka may reject open while a prior grasp/move is active.
            self._logger.warning(
                "Franky gripper open failed (%s); stop + retry via move", exc
            )
            self._stop_quiet()
            try:
                self._gripper.move(self._max_width, self._speed_ms(speed))
            except Exception as move_exc:
                self._logger.warning(
                    "Franky gripper move(open) also failed (%s); continuing", move_exc
                )
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        # franky grasp force is in Newtons; map oversized ROS-style defaults down.
        grasp_force = float(force) if force <= 100.0 else self._grasp_force
        speed_ms = self._speed_ms(speed)
        try:
            # grasp() often raises CommandException when closing on air / already
            # closed / transient libfranka errors; that must not kill eval.
            self._gripper.grasp(
                _CLOSE_WIDTH_M,
                speed_ms,
                grasp_force,
                epsilon_inner=1.0,
                epsilon_outer=1.0,
            )
        except Exception as exc:
            self._logger.warning(
                "Franky gripper grasp failed (%s); falling back to move(close)", exc
            )
            self._stop_quiet()
            try:
                self._gripper.move(_CLOSE_WIDTH_M, speed_ms)
            except Exception as move_exc:
                self._logger.warning(
                    "Franky gripper move(close) also failed (%s); continuing", move_exc
                )
        self._is_open_flag = False

    def move(self, position: float, speed: float = 0.3) -> None:
        # Same 0–255 → metres mapping as the ROS FrankaGripper.
        width = float(position) / (255.0 * 10.0)
        width = float(np.clip(width, 0.0, self._max_width))
        self._gripper.move(width, self._speed_ms(speed))
        self._is_open_flag = width >= _OPEN_WIDTH_THRESHOLD_M

    @property
    def position(self) -> float:
        # Expose width in metres; callers that expect 0–255 (Robotiq) should
        # not use this backend interchangeably without conversion.
        try:
            return float(self._gripper.width)
        except Exception:
            return 0.0

    @property
    def is_open(self) -> bool:
        try:
            return float(self._gripper.width) >= _OPEN_WIDTH_THRESHOLD_M
        except Exception:
            return self._is_open_flag

    def is_ready(self) -> bool:
        try:
            _ = self._gripper.width
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        try:
            self._gripper.stop()
        except Exception:
            pass
