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

"""RLinf DreamZero embodiment tags.

When adding a new embodiment, register it in ``data_transforms/__init__.py`` and
add the corresponding member below.
"""

from enum import Enum


class EmbodimentTag(Enum):
    """Embodiment tags supported by RLinf DreamZero SFT / eval."""

    LIBERO_SIM = "libero_sim"
    OXE_DROID = "oxe_droid"
    FRANKA_PNP = "franka_pnp"
