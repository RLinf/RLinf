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

"""Put what the suites share on the path.

``pytest.ini`` lives in this directory, so pytest roots itself here and never
reaches ``tests/``. Adding it is what lets a unit test write
``from robot_mocks import mocked_sdks``; see ``tests/conftest.py`` for why the
mocks live a level up.
"""

import sys
from pathlib import Path

for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
