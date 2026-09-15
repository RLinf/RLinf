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

"""Operator input devices, the part category an operator drives.

Importing this package registers every shipped device, so a config can name
one. Adding a device is one module here plus one line in ``_MODULES``.
"""

from importlib import import_module

from .base import CONTEXT_KEYS, TeleopAction, TeleopDevice, TeleopPart
from .group import TeleopEntry, TeleopGroup

#: Modules holding registered devices, imported for their side effect.
_MODULES = (
    ".gello",
    ".gello_joint",
    ".glove",
    ".pico",
    ".so101_leader",
    ".spacemouse",
)

for _module in _MODULES:
    import_module(_module, __name__)

from .gello import Gello  # noqa: E402
from .gello_joint import GelloJoint  # noqa: E402
from .glove import Glove  # noqa: E402
from .pico import Pico, PicoDelta, PicoTcp  # noqa: E402
from .so101_leader import SO101Leader  # noqa: E402
from .spacemouse import SpaceMouse  # noqa: E402

__all__ = [
    "CONTEXT_KEYS",
    "Gello",
    "GelloJoint",
    "Glove",
    "Pico",
    "PicoDelta",
    "PicoTcp",
    "SO101Leader",
    "SpaceMouse",
    "TeleopAction",
    "TeleopDevice",
    "TeleopEntry",
    "TeleopGroup",
    "TeleopPart",
]
