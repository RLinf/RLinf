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

"""What a device means for the robot it drives, and how devices compose.

A device reports what the operator did; a :class:`TeleopBinding` says which
parts of the action that fills; :class:`TeleopGroup` merges several of them
into one action. Composing here is what removes the hand-written classes that
used to splice a spacemouse and a glove, or two leader arms, together.
"""

from .binding import CONTEXT_KEYS, TeleopAction, TeleopBinding
from .bindings import (
    GloveBinding,
    LeaderArmBinding,
    LeaderJointBinding,
    PicoBinding,
    PicoTcpBinding,
    SpaceMouseBinding,
    jittered_grip,
)
from .group import TeleopEntry, TeleopGroup
from .kinds import ActionKind, ActionPart

__all__ = [
    "CONTEXT_KEYS",
    "ActionKind",
    "TeleopAction",
    "ActionPart",
    "GloveBinding",
    "LeaderArmBinding",
    "LeaderJointBinding",
    "PicoBinding",
    "PicoTcpBinding",
    "SpaceMouseBinding",
    "TeleopBinding",
    "TeleopEntry",
    "TeleopGroup",
    "jittered_grip",
]
