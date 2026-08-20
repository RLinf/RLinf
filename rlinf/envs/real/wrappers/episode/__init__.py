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
"""Wrappers that let an operator drive the episode rather than the robot.

None of these touch the action. They decide when an episode starts, when it ends,
what reward it earned, and which policy is in charge -- the judgements a person
watching the robot makes, which no sensor reports. Teleop is the other half of
that story and lives in :mod:`rlinf.envs.real.wrappers.teleop`.
"""

from .eval_control import KeyboardEvalControlWrapper
from .leader_follower import LeaderFollowerKeyboardIntervention
from .policy_switch import KeyboardRLTPolicySwitchWrapper
from .reward_done import (
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
)
from .start_end import KeyboardStartEndWrapper

__all__ = [
    "KeyboardEvalControlWrapper",
    "KeyboardRLTPolicySwitchWrapper",
    "KeyboardRewardDoneMultiStageWrapper",
    "KeyboardRewardDoneWrapper",
    "KeyboardStartEndWrapper",
    "LeaderFollowerKeyboardIntervention",
]
