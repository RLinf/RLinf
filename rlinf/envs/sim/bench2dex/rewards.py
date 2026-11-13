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

"""One-time rewards for existing stage completion events."""

import math
from collections.abc import Mapping, Sequence


class StageCompletionReward:
    """Award each stage once when its existing completion predicate is met.

    Predicates and dependency gating belong to the caller. Pass their current
    completion flags to ``update``. No additional duration condition is applied.
    """

    def __init__(self, stage_ids: Sequence[str], *, bonus: float):
        self.stage_ids = tuple(stage_ids)
        if not self.stage_ids or len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValueError("Expected nonempty, unique stage IDs")
        if not math.isfinite(bonus) or bonus < 0:
            raise ValueError(f"Expected finite nonnegative stage bonus, got {bonus}")
        self.bonus = float(bonus)
        self.reset()

    def reset(self, initial_completed: Mapping[str, bool] | None = None) -> None:
        """Start an episode; stages already complete cannot earn a bonus."""
        initial_completed = initial_completed or {}
        self._paid = {
            sid for sid in self.stage_ids if initial_completed.get(sid, False)
        }
        self.total_reward = 0.0
        self.awarded_count = 0

    def update(self, current_completed: Mapping[str, bool]) -> float:
        """Return bonuses immediately for stages not rewarded in this episode."""
        reward = 0.0
        for sid in self.stage_ids:
            if sid in self._paid or not current_completed.get(sid, False):
                continue
            self._paid.add(sid)
            self.awarded_count += 1
            reward += self.bonus
        self.total_reward += reward
        return reward

    def complete(self) -> float:
        """Pay remaining stage bonuses once, including initially complete stages."""
        remaining = len(self.stage_ids) - self.awarded_count
        reward = remaining * self.bonus
        self._paid.update(self.stage_ids)
        self.awarded_count += remaining
        self.total_reward += reward
        return reward
