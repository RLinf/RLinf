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


def qa_accuracy_reward(completions: list[str], answers: list[dict]) -> list[float]:
    """
    Reward function that evaluates question-answering accuracy for VQA tasks.

    Based on TRL's accuracy_reward pattern but adapted for multiple choice VQA.

    Args:
        completions: List of model completions (text strings)
        answers: List of correct answers (dict)

    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []

    for completion, answer in zip(completions, answers):
        completion_text = str(completion).strip()

        # Extract answer from completion - look for <answer>X. content</answer>
        answer_match = re.search(
            r"<answer>\s*([A-E])\.\s*(.*?)\s*</answer>",
            completion_text,
            re.DOTALL | re.IGNORECASE,
        )

        if not answer_match:
            rewards.append(0.0)
            continue

        predicted_letter = answer_match.group(1).upper()
        predicted_content = answer_match.group(2).strip()

        # Get ground truth from kwargs
        correct_answer = answer.get("correct_answer", None)
        choices = answer.get("choices", None)

        if correct_answer is None or choices is None:
            rewards.append(0.0)
            continue

        # Normalize correct_answer to letter format
        if isinstance(correct_answer, int):
            correct_letter = chr(65 + correct_answer)  # 0->A, 1->B, etc.
        elif isinstance(correct_answer, str):
            correct_letter = correct_answer.strip().upper()
        else:
            rewards.append(0.0)
            continue

        # Parse choices if string format
        if isinstance(choices, str):
            try:
                import ast

                choices = ast.literal_eval(choices)
            except (ValueError, SyntaxError):
                choices = [str(choices)]

        # Get correct choice content
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        if correct_letter in letter_to_idx and letter_to_idx[correct_letter] < len(
            choices
        ):
            correct_content = choices[letter_to_idx[correct_letter]].strip()
        else:
            rewards.append(0.0)
            continue

        # Check accuracy: both letter and content must match
        letter_match = predicted_letter == correct_letter
        content_match = _compare_choice_content(predicted_content, correct_content)

        rewards.append(1.0 if (letter_match and content_match) else 0.0)

    return rewards


def _compare_choice_content(predicted: str, correct: str) -> bool:
    """Compare predicted choice content with correct content."""
    # Simple normalized comparison
    pred_normalized = predicted.lower().strip()
    correct_normalized = correct.lower().strip()

    # Direct match
    if pred_normalized == correct_normalized:
        return True

    # Partial match for more flexibility
    if pred_normalized in correct_normalized or correct_normalized in pred_normalized:
        return True

    return False
