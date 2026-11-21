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

"""Text protocol for a Qwen VLM policy: prompts, JSON actions, LIBERO schema."""

from __future__ import annotations

import json
import math
import re
from typing import Any

LIBERO_SCHEMA: dict[str, tuple[str, ...]] = {
    "arm": ("x", "y", "z", "roll", "pitch", "yaw"),
    "gripper": ("width",),
}
ROBODOJO_SCHEMA: dict[str, tuple[str, ...]] = {
    "left_arm": ("x", "y", "z", "roll", "pitch", "yaw"),
    "right_arm": ("x", "y", "z", "roll", "pitch", "yaw"),
    "left_gripper": ("width",),
    "right_gripper": ("width",),
}
LIBERO_CAMERAS: tuple[str, ...] = ("agentview", "wrist")
ROBODOJO_CAMERAS: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
STATE_DECIMALS: dict[str, int] = {
    "x": 1,
    "y": 1,
    "z": 1,
    "roll": 2,
    "pitch": 2,
    "yaw": 2,
    "width": 1,
}
TURN_REMINDER = (
    "The current camera frames follow. End your response with one json block "
    "containing the next action."
)
JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_THINK_CLOSE = "</think>"
# Qwen3.8 chat template at revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0:
# when enable_thinking is true or omitted, reasoning_effort defaults to xhigh.
REASONING_EFFORTS = ("xhigh", "medium", "low")
DEFAULT_REASONING_EFFORT = "xhigh"
NATIVE_STEP_BUDGET = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
_MIRRORED_SIDE = {"left": "right", "right": "left"}


def chat_template_kwargs(
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, Any]:
    """Keyword arguments for Qwen3.8 ``apply_chat_template``.

    The official template resolves an omitted effort to ``xhigh``. Callers
    still pass it explicitly so a server-side template cannot pick another
    default.
    """
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {REASONING_EFFORTS}, "
            f"got {reasoning_effort!r}"
        )
    return {"enable_thinking": True, "reasoning_effort": reasoning_effort}


def reset_wait_steps(vlm_action_mode: bool, configured: Any = None) -> int:
    """VLM-policy episodes settle for 10 steps. Other LIBERO runs keep 15."""
    if configured is not None:
        return int(configured)
    return 10 if vlm_action_mode else 15


def native_step_budget(suite: str) -> int:
    """Return the native-step budget for one LIBERO suite."""
    key = str(suite).lower()
    if key not in NATIVE_STEP_BUDGET:
        raise KeyError(f"no native-step budget for suite {suite!r}")
    return NATIVE_STEP_BUDGET[key]


def mirror_instruction(text: str) -> str:
    """Swap left and right so the instruction matches a 180-degree image."""
    return re.sub(
        r"\b(left|right)\b",
        lambda match: _MIRRORED_SIDE[match.group(1)],
        text,
    )


LIBERO_ENVIRONMENT = """- agentview: a fixed camera; the robot's base is at the top of its image, +x points toward the bottom of the image and +y toward its left.
- wrist: a camera on the gripper; with the gripper pointing straight down at yaw 0, +x points toward the top of its image and +y toward its right.
- the episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet.
"""

ROBODOJO_ENVIRONMENT = """- head, left_wrist and right_wrist are the three cameras.
- left_arm and right_arm are the two arms. left_gripper and right_gripper are their grippers.
- the episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet.
"""

_SYSTEM_TEMPLATE = """You are controlling a robot. Each turn you see the robot's current state and its camera images, and you respond with the next action.

## OBSERVATION

The robot is made of named parts, and each part has named quantities you can set. Every turn begins with the current value of every such quantity, grouped by part, followed by the camera images:

{"<part>": {"<quantity>": <value>, ...}, ...}

Lengths are in centimeters and angles in radians; any other quantity is a plain number. Nothing beyond these names is given about what each quantity corresponds to.

## ACTION

The action changes the listed quantities by these amounts:

{"<part>": {"<quantity>": <change>, ...}, ...}

Write the change of every quantity; 0 leaves a quantity as it is.

## EXECUTION

The action ends once every target is reached and the robot is at rest. If something keeps a target from being reached, the robot stops there and the action ends. Either way, you then receive the next observation.

## RESPONSE FORMAT

End your response with exactly one fenced json block containing the action, and nothing after it:

```json
<action>
```

{environment}## TASK

{instruction}
"""


class InvalidAction(ValueError):
    """A JSON block that is not an action for the active schema."""


def system_prompt(instruction: str, schema: dict[str, tuple[str, ...]]) -> str:
    """Return the system prompt for one embodiment and task instruction."""
    if schema == ROBODOJO_SCHEMA:
        environment = f"## ENVIRONMENT\n\n{ROBODOJO_ENVIRONMENT}\n"
    else:
        environment = f"## ENVIRONMENT\n\n{LIBERO_ENVIRONMENT}\n"
    return _SYSTEM_TEMPLATE.replace("{environment}", environment).replace(
        "{instruction}", instruction
    )


def cameras_for_schema(schema: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return ordered camera names matching PEAR's EnvironmentTool.CAMERAS."""
    if schema == ROBODOJO_SCHEMA:
        return ROBODOJO_CAMERAS
    return LIBERO_CAMERAS


def round_state(
    state: dict[str, dict[str, float]],
    decimals: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """Round state values to PEAR's display precision, avoiding -0.0 and fp32 drift."""
    precision = STATE_DECIMALS if decimals is None else decimals
    return {
        part: {
            name: (
                round(float(value), precision[name]) + 0.0
                if name in precision
                else float(value) + 0.0
            )
            for name, value in quantities.items()
        }
        for part, quantities in state.items()
    }


def turn_text(state: dict[str, dict[str, float]]) -> str:
    """Return the user-turn text that precedes the camera frames."""
    return json.dumps(round_state(state)) + "\n\n" + TURN_REMINDER


def build_user_content(
    state: dict[str, dict[str, float]],
    images: list[Any],
    cameras: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build PEAR-format user content: JSON state + reminder, then ``[camera]`` + frame."""
    camera_names = LIBERO_CAMERAS if cameras is None else tuple(cameras)
    if len(images) != len(camera_names):
        raise ValueError(
            f"Expected {len(camera_names)} images for cameras {camera_names}, "
            f"got {len(images)}"
        )
    blocks: list[dict[str, Any]] = [{"type": "text", "text": turn_text(state)}]
    for name, image in zip(camera_names, images):
        blocks.append({"type": "text", "text": f"[{name}]"})
        blocks.append({"type": "image", "image": image})
    return blocks


def answer_text(text: str) -> str:
    """Return the text after the last thinking close tag."""
    if _THINK_CLOSE in text:
        return text.rsplit(_THINK_CLOSE, 1)[-1]
    return text


def extract_json_block(text: str) -> dict[str, Any]:
    """Return the last fenced JSON object in the post-thinking answer."""
    matches = list(JSON_BLOCK_PATTERN.finditer(answer_text(text)))
    if not matches:
        raise InvalidAction("the answer has no fenced json block")
    try:
        value = json.loads(matches[-1].group(1))
    except json.JSONDecodeError as exc:
        raise InvalidAction("the json block is not valid json") from exc
    if not isinstance(value, dict):
        raise InvalidAction("the json block must be an object")
    return value


def check_action(action: Any, schema: dict[str, tuple[str, ...]]) -> None:
    """Raise InvalidAction unless every schema quantity is a finite number."""
    if not isinstance(action, dict) or set(action) != set(schema):
        raise InvalidAction(
            f"the json block must map exactly the parts {', '.join(schema)}"
        )
    for part, quantities in schema.items():
        values = action[part]
        if not isinstance(values, dict) or set(values) != set(quantities):
            raise InvalidAction(f"{part} must give exactly {', '.join(quantities)}")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidAction(f"{part}.{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise InvalidAction(f"{part}.{name} must be a finite number")


def parse_action(
    text: str, schema: dict[str, tuple[str, ...]] | None = None
) -> dict[str, Any]:
    """Parse and validate one JSON macro action."""
    schema = LIBERO_SCHEMA if schema is None else schema
    action = extract_json_block(text)
    check_action(action, schema)
    return action


def action_to_vector(
    action: dict[str, Any] | None,
    schema: dict[str, tuple[str, ...]] | None = None,
) -> list[float]:
    """Flatten an action and append 1, or zeros and 0 when it is invalid.

    The trailing flag is not a simulator command. Zero means the environment
    must terminate the episode instead of executing a zero motion.
    """
    schema = LIBERO_SCHEMA if schema is None else schema
    width = sum(len(quantities) for quantities in schema.values()) + 1
    if action is None:
        return [0.0] * width
    try:
        check_action(action, schema)
    except InvalidAction:
        return [0.0] * width
    values: list[float] = []
    for part, quantities in schema.items():
        values.extend(float(action[part][name]) for name in quantities)
    values.append(1.0)
    return values
