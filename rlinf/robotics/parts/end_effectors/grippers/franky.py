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

"""Franka parallel-jaw gripper controlled through libfranka by Franky."""

import time
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.utils.logging import get_logger

from ..base import EndEffector
from .base import BaseGripper

#: Nominal stroke of the Franka Hand, in metres. Each hand reports its own
#: calibrated stroke once connected, which is a millimetre or two off this.
_MAX_WIDTH_M = 0.08

#: Width :meth:`FrankyGripper.close` grasps to, in metres.
_CLOSE_WIDTH_M = 0.01

#: Width at or above which the hand counts as open, in metres.
_OPEN_WIDTH_THRESHOLD_M = 0.05

#: Franky takes finger speed in m/s; normalized speed maps into this band.
_MIN_SPEED_MS = 0.01
_MAX_SPEED_MS = 0.1

#: Grasp force substituted when a caller asks for more than the hand allows.
_DEFAULT_GRASP_FORCE_N = 40.0

#: Above this, a force argument is treated as another gripper's scale.
_FORCE_SCALE_LIMIT_N = 100.0

#: libfranka serves gripper state from its own poll, roughly every 30 ms, and a
#: read costs that whole round trip. Two reads inside one period return the same
#: number, so the second is served from the first.
_STATE_PERIOD_S = 0.03


@EndEffector.register("franky_gripper")
class FrankyGripper(BaseGripper):
    """Franka Emika Hand reached over libfranka rather than ROS.

    Same hardware as :class:`~.franka.FrankaGripper`, different transport: that
    one publishes on the arm's ROS session, this one opens its own libfranka
    gripper session from the arm's IP. libfranka gives the hand a separate
    endpoint from arm control, so the two sessions are independent, and an arm
    on any backend can build this gripper as long as it knows the robot's IP.

    The arm that builds it owns its lifecycle: it is connected when the arm
    connects and released when the arm releases.

    Args:
        robot_ip: Address of the arm this hand is mounted on.
        max_width: Stroke in metres, used until the hand reports its own
            calibrated stroke on connect. It bounds the axis :meth:`move` and
            :attr:`position` share, and ``open()`` travels to it.
        grasp_force: Force in Newtons used when a caller asks for more than
            the hand can apply.
    """

    @classmethod
    def declare(
        cls,
        *,
        ros: Optional[Any] = None,
        port: Optional[str] = None,
        robot_ip: Optional[str] = None,
        **settings: Any,
    ) -> "FrankyGripper":
        """Declare a hand that opens its own libfranka session."""
        if not robot_ip:
            raise ValueError(
                "A Franka Hand on the libfranka backend is reached at the "
                "arm's own IP, so one has to be passed. Only an arm that "
                "knows that address can build this gripper."
            )
        return cls(robot_ip=robot_ip, **settings)

    def __init__(
        self,
        robot_ip: str,
        max_width: float = _MAX_WIDTH_M,
        grasp_force: float = _DEFAULT_GRASP_FORCE_N,
    ) -> None:
        self._logger = get_logger()
        self._robot_ip = robot_ip
        # The hand answers on its own libfranka port, so it does not contend
        # with arm control -- only with a second hand session.
        self._claim = DeviceClaim(f"franky-hand:{robot_ip}", type(self).__name__)
        self._max_width = float(max_width)
        self._grasp_force = float(grasp_force)
        self._gripper = None
        self._is_open_flag = True
        self._width_value = 0.0
        self._width_read_at: Optional[float] = None

    def _open(self) -> Any:
        """Open the libfranka gripper session and prove it answers."""
        import franky

        self._claim.acquire()
        gripper = franky.Gripper(self._robot_ip)
        # Reading width fails here rather than inside the first command if the
        # hand is absent or FCI is not released to this host.
        _ = gripper.width
        # Calibration leaves each hand a slightly different stroke, and the
        # clamp in move() is only honest against the one in front of us.
        reported = getattr(gripper, "max_width", None)
        if reported:
            self._max_width = float(reported)
        self._gripper = gripper
        self._logger.info(f"FrankyGripper connected at {self._robot_ip}")
        return gripper

    def _release(self, device: Any) -> None:
        """Stop motion in flight and drop the session."""
        try:
            self._stop_quiet()
        finally:
            self._gripper = None
            self._width_read_at = None
            self._claim.release()

    # BaseGripper interface

    def open(self, speed: float = 0.3) -> None:
        """Travel to :attr:`max_width`."""
        try:
            self._gripper.open(self._speed_ms(speed))
        except Exception as error:
            # libfranka rejects open() while an earlier grasp still holds the
            # hand; stopping first turns that into a plain move.
            self._logger.warning(
                f"FrankyGripper open failed ({error}); retrying as a move"
            )
            self._stop_quiet()
            self._move_quiet(self._max_width, speed, "open")
        self._invalidate_width()
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        """Grasp at ``force``, in Newtons.

        The Franka Hand applies far less force than the Robotiq scale the
        shared default is written in, so an oversized request is served at
        :attr:`grasp_force` instead of being passed through or rejected.
        """
        grasp_force = (
            float(force) if force <= _FORCE_SCALE_LIMIT_N else self._grasp_force
        )
        try:
            # Epsilon of 1 m accepts any final width: closing on air or on an
            # object narrower than expected is a normal outcome here, not a
            # failed episode.
            self._gripper.grasp(
                _CLOSE_WIDTH_M,
                self._speed_ms(speed),
                grasp_force,
                epsilon_inner=1.0,
                epsilon_outer=1.0,
            )
        except Exception as error:
            self._logger.warning(
                f"FrankyGripper grasp failed ({error}); retrying as a move"
            )
            self._stop_quiet()
            self._move_quiet(_CLOSE_WIDTH_M, speed, "close")
        self._invalidate_width()
        self._is_open_flag = False

    def move(self, width: float, speed: float = 0.3) -> None:
        """Move to an opening width in metres."""
        target = float(np.clip(width, 0.0, self._max_width))
        self._gripper.move(target, self._speed_ms(speed))
        self._invalidate_width()
        self._is_open_flag = target >= _OPEN_WIDTH_THRESHOLD_M

    @property
    def position(self) -> float:
        """Current opening width in metres."""
        try:
            return self._width()
        except Exception:
            return 0.0

    @property
    def max_width(self) -> float:
        """Stroke of the Franka Hand."""
        return self._max_width

    @property
    def is_open(self) -> bool:
        """Whether the fingers are at or beyond the open threshold.

        Falls back to the last commanded state when the width cannot be read,
        so a dropped reading does not report the hand as closed.
        """
        try:
            return self._width() >= _OPEN_WIDTH_THRESHOLD_M
        except Exception:
            return self._is_open_flag

    def is_ready(self) -> bool:
        """Whether the session answers a width query."""
        if self._gripper is None:
            return False
        try:
            _ = float(self._gripper.width)
            return True
        except Exception:
            return False

    # Internals

    def _width(self) -> float:
        """Return the opening width, reusing a reading the SDK cannot beat.

        An observation asks for the width and the open state together, and each
        libfranka read costs a full poll period. Serving the second from the
        first keeps a whole-robot observation inside one gripper round trip
        rather than two.
        """
        now = time.monotonic()
        if (
            self._width_read_at is not None
            and now - self._width_read_at < _STATE_PERIOD_S
        ):
            return self._width_value
        self._width_value = float(self._gripper.width)
        self._width_read_at = now
        return self._width_value

    def _invalidate_width(self) -> None:
        """Drop the cached width after commanding the fingers to move."""
        self._width_read_at = None

    @staticmethod
    def _speed_ms(speed: float) -> float:
        """Map normalized speed onto the hand's practical m/s band."""
        return _MIN_SPEED_MS + float(np.clip(speed, 0.0, 1.0)) * (
            _MAX_SPEED_MS - _MIN_SPEED_MS
        )

    def _stop_quiet(self) -> None:
        """Stop motion, ignoring a session that is already gone."""
        try:
            self._gripper.stop()
        except Exception:
            pass
        finally:
            self._invalidate_width()

    def _move_quiet(self, width: float, speed: float, doing: str) -> None:
        """Last-resort move; a failure here is logged, not raised.

        Both callers are already handling a failed command. Raising would end
        an episode over a hand that is merely out of position.
        """
        try:
            self._gripper.move(width, self._speed_ms(speed))
        except Exception as error:
            self._logger.warning(
                f"FrankyGripper move({doing}) also failed ({error}); continuing"
            )
