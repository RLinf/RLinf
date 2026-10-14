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

import importlib
import pathlib
import pkgutil


def import_all_tasks():
    package_name = __name__ + ".tasks"
    package_path = pathlib.Path(__file__).parent / "tasks"

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"{package_name}.{module_name}")

    try:
        importlib.import_module("mani_skill_gs")
    except (ModuleNotFoundError, ImportError):
        pass


import_all_tasks()
