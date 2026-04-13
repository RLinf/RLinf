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

from typing import TYPE_CHECKING

from . import gt_prefix

if TYPE_CHECKING:
    from .habitat_env import HabitatEnv

__all__ = ["HabitatEnv", "gt_prefix"]


def __getattr__(name: str):
    if name == "HabitatEnv":
        from .habitat_env import HabitatEnv

        return HabitatEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
