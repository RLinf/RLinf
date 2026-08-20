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

"""What the test suites share.

``robot_mocks`` sits here rather than inside ``unit_tests`` because a faked
robot is as useful to an end-to-end run as to a unit test: the same fakes let a
whole training loop run against a robot that is not there.

How each suite reaches it differs, because they are run differently. Pytest
suites add this directory in their own ``conftest.py``, since each carries its
own ``pytest.ini`` and pytest never looks above a rootdir. Script-driven suites
put it on ``PYTHONPATH``, which is also what a worker process needs -- see
``robot_mocks._reach_worker_processes``.
"""

import sys
from pathlib import Path

for _path in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
