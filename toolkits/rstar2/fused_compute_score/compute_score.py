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

from .math_verify import compute_score as math_verify_compute_score
from .prime_math import compute_score as prime_compute_score
import base64

def compute_score(model_output: str, ground_truth: str) -> float:
    try:
        # prime_compute_score returns a tuple, we need the first element.
        if prime_compute_score(model_output, ground_truth)[0]:
            return 1.0
    except Exception:
        # If prime_compute_score fails, fall back to math_verify_compute_score.
        pass

    try:
        # print(f"math_verify_compute_score: {model_output}, {ground_truth}")
        if math_verify_compute_score(model_output, ground_truth):
            return 1.0
    except Exception:
        # If math_verify_compute_score also fails, return 0.0.
        return 0.0

    # If both ran successfully but did not return a positive score.
    return 0.0
