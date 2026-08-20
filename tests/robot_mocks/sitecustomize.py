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

"""Install the fake SDKs in a worker process, before anything imports them.

Python imports ``sitecustomize`` at interpreter start if it is importable, so a
scheduler worker picks the fakes up before the part it hosts constructs. This
is what lets a mocked run place parts on nodes like a real one, rather than
building everything in the driver.

It does nothing unless ``RLINF_ROBOT_MOCKS`` is set, so putting this directory
on ``PYTHONPATH`` is harmless on its own.
"""

import os

if os.environ.get("RLINF_ROBOT_MOCKS"):
    try:
        import sys

        from robot_mocks import sdk_modules

        # Assigned rather than set as a default: building the fakes imports
        # real psutil, so a default would keep the real one and the worker
        # would go on to launch processes.
        for name, fake in sdk_modules().items():
            sys.modules[name] = fake
    except Exception as error:  # pragma: no cover - a worker must still start
        print(f"[robot_mocks] could not install fake SDKs: {error!r}")
