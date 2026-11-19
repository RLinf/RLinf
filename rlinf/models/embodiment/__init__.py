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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlinf.models.embodiment.cosmos3.dagger import Cosmos3DAggerAdapter


def get_dagger_adapter(model_type: str) -> "type[Cosmos3DAggerAdapter] | None":
    """Return the model-specific DAgger adapter when the policy needs one."""
    if model_type == "cosmos3":
        from rlinf.models.embodiment.cosmos3.dagger import Cosmos3DAggerAdapter

        return Cosmos3DAggerAdapter
    return None
