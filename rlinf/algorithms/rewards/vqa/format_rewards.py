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

import re


def think_format_reward(completions, answers) -> list[float]:
    """
    Think format reward function compatible with GRPO training.

    Reward function that checks if reasoning is enclosed within <think></think> tags.

    Args:
        completions: List of model completions (text strings)

    Returns:
        List of reward scores (1.0 for correct format, 0.0 otherwise)
    """
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    rewards = []

    for completion in completions:
        completion_text = str(completion).strip()
        match = re.match(pattern, completion_text, re.DOTALL | re.MULTILINE)
        rewards.append(1.0 if match else 0.0)

    return rewards


def answer_format_reward(completions, answers) -> list[float]:
    """
    Reward function that checks for proper answer formatting.

    Expected format: <answer>X. content</answer> where X is a choice letter.

    Args:
        completions: List of model completions (text strings)

    Returns:
        List of reward scores (1.0 for correct format, 0.0 otherwise)
    """
    rewards = []

    for completion in completions:
        completion_text = str(completion).strip()

        # Check for proper answer format: <answer>X. content</answer>
        answer_pattern = r"<answer>\s*[A-E]\.\s*.+?\s*</answer>"
        has_proper_answer = bool(
            re.search(answer_pattern, completion_text, re.DOTALL | re.IGNORECASE)
        )

        rewards.append(1.0 if has_proper_answer else 0.0)

    return rewards
