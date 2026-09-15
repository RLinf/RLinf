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

"""Data glove driving a dexterous hand, relative to a resettable baseline."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice


@TeleopDevice.register("glove")
class Glove(TeleopDevice):
    """A data glove reporting finger angles.

    Args:
        left_port: Serial port of the left glove.
        right_port: Serial port of the right glove.
        frequency: Polling rate in Hz.
        config_file: Optional calibration file.
    """

    PRODUCES = {"hand": ActionKind.HAND}

    #: The hand keeps its pose while the operator drives the arm, so its part
    #: still applies on a step where the glove is not the one driving.
    APPLIES_WHILE_IDLE = True

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
        self._baseline: Optional[np.ndarray] = None
        self._commanded = np.zeros(6, dtype=np.float64)
        self._base = np.zeros(6, dtype=np.float64)
        self._rebaseline = False

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Merge the shared glove config with this entry's own options."""
        from .group import TeleopEntry

        # Per-entry options override the shared glove configuration.
        glove_cfg = dict(cfg.get("glove_config", {}))
        glove_cfg.update(options)
        return TeleopEntry(
            cls(
                left_port=glove_cfg.get("left_port", "/dev/ttyACM0"),
                right_port=glove_cfg.get("right_port"),
                frequency=int(glove_cfg.get("frequency", 60)),
                config_file=glove_cfg.get("config_file"),
            ),
            drives=options.get("drives"),
        )

    # Hardware.

    def _open(self) -> Any:
        from rlinf_dexhand.glove import GloveExpert

        return GloveExpert(
            left_port=self._left_port,
            right_port=self._right_port,
            frequency=self._frequency,
            config_file=self._config_file,
        )

    @property
    def observation_features(self) -> Features:
        """Return one angle feature per finger joint."""
        return {"angles": {"shape": (6,), "dtype": "float32"}}

    def get_observation(self) -> Observation:
        """Read the operator's finger angles."""
        return {"angles": np.asarray(self._device.get_angles(), dtype=np.float32)}

    # Driving the robot.

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Initialize the held hand pose from the post-reset context."""
        start = context.get("hand_reset_pose")
        self._commanded = (
            np.zeros(6, dtype=np.float64)
            if start is None
            else np.asarray(start, dtype=np.float64).reshape(-1).copy()
        )
        self._base = self._commanded.copy()
        self._baseline = None

    def rebaseline(self) -> None:
        """Re-zero the glove against the hand's current pose."""
        self._rebaseline = True

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Track the operator's fingers, or hold where they left them."""
        angles = np.asarray(reading["angles"], dtype=np.float64)
        # Hold the latest command while glove control is inactive.
        if context.get("hand_driving", False):
            if self._rebaseline or self._baseline is None:
                # Rebase at the control edge to avoid a command discontinuity.
                self._baseline = angles.copy()
                self._base = self._commanded.copy()
                self._rebaseline = False
            self._commanded = np.clip(self._base + (angles - self._baseline), 0.0, 1.0)
        else:
            self._baseline = None  # Rebase when control is next taken.
        # The arm device's button decides who is driving, not the glove.
        return TeleopAction(parts={"hand": self._commanded.copy()}, driving=False)


# The vendor SDK this device speaks to.
