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

from pathlib import Path


def test_async_embodied_runner_run_has_single_checkpoint_call():
    """Async embodied runner should not save checkpoint twice in one run loop."""
    source = Path("rlinf/runners/async_embodied_runner.py").read_text(encoding="utf-8")
    _, _, run_body = source.partition("def run(self):")
    assert run_body.count("self._save_checkpoint()") == 1
