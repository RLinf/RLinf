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

"""Servo-bus plumbing shared by the LeKiwi bring-up commands.

These commands talk to the wheels directly rather than through
:class:`~rlinf.robotics.parts.mobility.LeKiwiBase`, because assigning motor IDs
and reading raw velocities are exactly what they check.
"""

from __future__ import annotations

import logging

from rlinf.robotics.parts.mobility.lekiwi import (
    WHEEL_IDS,
    WHEEL_ORDER,
    make_wheel_bus,
    start_velocity_mode,
    stop_and_disconnect,
)
from rlinf.utils.logging import get_logger

__all__ = [
    "WHEEL_IDS",
    "WHEEL_ORDER",
    "cli_logger",
    "make_wheel_bus",
    "start_velocity_mode",
    "stop_and_disconnect",
]


def cli_logger() -> logging.Logger:
    """Return the RLinf logger for command-line output."""
    return get_logger()
