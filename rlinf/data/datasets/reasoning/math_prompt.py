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

_OPENING = "Solve the following math problem step by step. "
_REASONING = "You must conduct reasoning inside <think> and </think> firstly. "
_ANSWER = "The last line of your response should be of the form Answer: "
_REMINDER = 'Remember to put your answer on its own line after "Answer:".'


def format_dapo_math_prompt(question: str) -> str:
    """Format a raw problem or a DAPO user message with a boxed final answer."""
    header, separator, body = question.partition("\n\n")
    if separator and header.startswith(
        (_OPENING + _ANSWER, _OPENING + _REASONING + _ANSWER)
    ):
        question = body.removesuffix("\n\n" + _REMINDER)
    return (
        f"{_OPENING}{_REASONING}{_ANSWER}"
        r"\boxed{...}, where ... is your final answer."
        f"\n\n{question}\n\n{_REMINDER}"
    )
