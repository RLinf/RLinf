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

"""Patched EmbodimentTag enum (includes LIBERO_SIM) for DreamZero + Libero."""

from enum import Enum


class EmbodimentTag(Enum):
    REAL_GR1_ARMS_ONLY = "real_gr1_arms_only"
    """
    The real GR1 robot embodiment with arms only.
    """

    OXE_DROID = "oxe_droid"
    """
    The Open X-Embodiment droid dataset.
    """

    UNKNOWN = "unknown"

    GR1_UNIFIED_SEGMENTATION = "gr1_unified_segmentation"
    """
    The GR1 unified dataset with segmentation.
    """

    LIBERO_SIM = "libero_sim"
    """
    The Libero Sim dataset.
    """

    DROID_SIM = "droid_sim"
    """
    The Droid dataset in sim.
    """

    MECKA_HANDS = "mecka_hands"
    """
    The Mecka robot with hands.
    """