# Copyright 2026 Shirui Chen
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

"""Subtask-level reward for stage-aware embodied RL.

For embodied RL with YAM robot, subtask rewards are returned directly from
``YAMEnv.step()`` after the VLM planner evaluates each subtask boundary.
This class is a lightweight wrapper registered in the reward registry for
logging and configuration purposes.

Usage in YAML:
    reward:
      use_reward_model: False
      reward_type: subtask
      success_reward: 1.0
      failure_reward: 0.0
"""

from omegaconf import DictConfig


class SubtaskReward:
    """Binary reward based on VLM-evaluated subtask completion.

    Rewards are expected to already be present in the rollout result (provided
    by the environment).  This class acts as a pass-through registry entry and
    exposes configuration for the success/failure reward magnitudes used by
    ``YAMEnv`` when constructing subtask rewards.

    Args:
        config: Reward config DictConfig.  Recognised keys:
            - ``success_reward`` (float, default 1.0): reward when subtask is
              completed successfully.
            - ``failure_reward`` (float, default 0.0): reward when the subtask
              fails or times out.
    """

    def __init__(self, config: DictConfig):
        self.success_reward = float(config.get("success_reward", 1.0))
        self.failure_reward = float(config.get("failure_reward", 0.0))

    def get_reward(self, completions, answers, **kwargs) -> list[float]:
        """Not used for embodied RL — rewards come from env.step().

        For embodied RL, the VLM planner evaluates subtask completion inside
        ``YAMEnv.step()`` and returns rewards directly.  This method exists
        solely so SubtaskReward is compatible with the reward-registry interface
        used by the text-based reward worker.

        Args:
            completions: Unused.
            answers: Unused.
            **kwargs: Unused.

        Returns:
            Empty list.
        """
        # TODO(agent): implement model-side reward computation if a separate
        # reward worker is wired up for embodied subtask evaluation.
        return []
