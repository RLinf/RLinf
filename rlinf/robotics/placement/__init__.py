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

"""Declaring where a part runs, and running it there.

A part whose vendor SDK only exists on one machine cannot be built here and
moved. So placement takes a declaration -- a class, its constructor arguments,
and a node -- and builds the part on that node, handing back a handle that
behaves the same whether the part ended up local or remote.

:mod:`.specs` is the declaring half and imports nothing from the scheduler.
:mod:`.handles` is the running half, and is the one module in
``rlinf.robotics`` allowed to import it.
"""

from .handles import (
    LocalPartHandle,
    PartHandle,
    PartWorkerHost,
    RemoteCamera,
    RemoteControllablePart,
    RemoteEndEffector,
    RemotePart,
    RemotePartHandle,
)
from .specs import PartSpec, Placement, SubpartRef

__all__ = [
    "LocalPartHandle",
    "PartHandle",
    "PartSpec",
    "Placement",
    "RemoteCamera",
    "RemoteControllablePart",
    "RemoteEndEffector",
    "RemotePart",
    "RemotePartHandle",
    "SubpartRef",
    "PartWorkerHost",
]
