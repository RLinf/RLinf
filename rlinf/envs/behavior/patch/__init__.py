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

SUPPORTED_OMNIGIBSON_VERSION = ["3.7.1", "3.7.2"]

_INSTALLED = False


def install_patch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import importlib

    import omnigibson as og

    if og.__version__ not in SUPPORTED_OMNIGIBSON_VERSION:
        raise RuntimeError(
            "RLinf BEHAVIOR patch only supports OmniGibson "
            f"{SUPPORTED_OMNIGIBSON_VERSION}, got {og.__version__}."
        )

    importlib.import_module("rlinf.envs.behavior.patch.task")
    importlib.import_module("rlinf.envs.behavior.patch.scene")
    from rlinf.envs.behavior.patch import monkey_patch

    monkey_patch.apply()
    _INSTALLED = True
