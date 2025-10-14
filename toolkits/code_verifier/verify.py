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

from typing import List

try:
    from fuzzywuzzy import fuzz

    FUZZY_AVAILABLE = True
except ImportError:
    fuzz = None
    FUZZY_AVAILABLE = False


def fim_verify_call(
    responses: List[str],
    references: List[str],
) -> List:
    assert FUZZY_AVAILABLE, "fuzzywuzzy is not installed"
    assert len(responses) == len(references), (
        len(responses),
        len(references),
    )

    rewards = []
    for resp, ref in zip(responses, references):
        fuzzy_sim = fuzz.ratio(resp.strip(), ref.strip()) / 100
        rewards.append(fuzzy_sim)
    return rewards
