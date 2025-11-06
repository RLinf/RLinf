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

from omegaconf import DictConfig

from toolkits.code_verifier.verify import (
    fim_llm_as_judge_verify_call,
)


class CodeRewardOffline:
    def __init__(self, config: DictConfig):
        self.scale = config.get("reward_scale", 1.0)

    def get_reward(
        self,
        response: list[str],
        reference: list[list[str]],
        prompts: list[str],
    ) -> list[float]:
        rewards = fim_llm_as_judge_verify_call(response, reference, prompts)
        return [float(reward) * self.scale for reward in rewards]
