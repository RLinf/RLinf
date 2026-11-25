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

"""Evaluate a Go2 C checkpoint with the original frozen baseline and seed-0 label."""

if __package__:
    from .evaluate import main as evaluate
else:
    from evaluate import main as evaluate


def main() -> None:
    """Run the locomotion evaluator for task C."""
    evaluate(legacy_stairs=True)


if __name__ == "__main__":
    main()
