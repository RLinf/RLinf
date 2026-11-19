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

"""RoboDojo dual-arm JSON action schema for a VLM policy.

Execution belongs to Isaac Sim and cuRobo. This module validates the shared
JSON contract before that executor is attached.
"""

from __future__ import annotations

from typing import Any

from rlinf.models.embodiment.qwen_vlm.protocol import (
    ROBODOJO_SCHEMA,
    check_action,
)

SCHEMA = ROBODOJO_SCHEMA


def validate_action(action: dict[str, Any]) -> dict[str, Any]:
    """Validate one dual-arm relative action and return it unchanged."""
    check_action(action, SCHEMA)
    return action


def execute_action(action: dict[str, Any]) -> None:
    """Execute a validated action through the Isaac/cuRobo controller."""
    validate_action(action)
    raise NotImplementedError(
        "RoboDojo VLM execution requires the Isaac Sim scene and cuRobo "
        "waypoint controller. The JSON schema is validated, but the simulator "
        "bridge is not part of the LIBERO MVP."
    )
