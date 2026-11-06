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

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PrefixAlgorithmRequirements:
    """Head capabilities required by an off-policy algorithm component."""

    actor_distribution: Literal["stochastic", "deterministic"]
    min_q_heads: int = 2
    requires_action_noise: bool = False
    uses_target_actor: bool = False
