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

REGISTER_GENIESIM_ENVS = {}

def register_geniesim_env(task_id: str):
    """Decorator to register a GenieSimEnv subclass under a task ID."""
    def _register(cls):
        REGISTER_GENIESIM_ENVS[task_id] = cls
        return cls
    return _register


def _import_all_tasks():
    """Import all task modules to trigger @register_geniesim_env decorators."""
    from rlinf.envs.geniesim.tasks import (  # noqa: F401
        PlaceWorkpieceEnv,
    )


_import_all_tasks()
