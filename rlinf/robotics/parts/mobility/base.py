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

"""Interfaces for controllable mobile bases."""

from rlinf.robotics.parts.base import ControllablePart


class MobileBase(ControllablePart):
    """Base class for wheeled, tracked, or legged mobile platforms.

    Drivers declare their observation fields, action fields, and units through
    the standard part interface.
    """
