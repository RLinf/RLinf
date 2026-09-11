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

"""Reproducible observations for an environment with no hardware to read."""

from typing import Optional

import gymnasium as gym


def seed_sampled_spaces(seed: Optional[int], *spaces: gym.Space) -> None:
    """Seed the spaces a hardware-free environment samples observations from.

    A dummy environment has nothing to read, so it samples its declared space
    instead. A Gymnasium space carries its own generator seeded from entropy,
    which leaves two runs of one configuration disagreeing on every value, so
    a dummy end-to-end run can only catch a change of shape and never one of
    content.

    Each space is offset so that two of them do not draw the same numbers.

    Args:
        seed: The seed the caller was given, or ``None`` to leave the spaces
            as they are, which is what Gymnasium expects of a reset that asks
            for no particular episode.
        spaces: The spaces this environment samples when it has no hardware.
    """
    if seed is None:
        return
    for offset, space in enumerate(spaces):
        space.seed(seed + offset)
