# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Ordered subtask prompts, following the A1 ``stage_prompts`` design.

Subtask ``k`` (1-based) feeds ``prompts[k - 1]`` to the model's language
input and is also the stage number (``rlt_switch_flags``) that selects the
Stage2 actor head and the proprio stage one-hot.
"""

from __future__ import annotations

import json
from pathlib import Path

# Stage2 appends an RLT_NUM_STAGES=4 one-hot to the actor/critic state
# (launcher/tianji_schedule_env.py; model yaml stage_onehot_dim).
MAX_SUBTASKS = 4


def load_subtask_prompts(path, *, limit=MAX_SUBTASKS) -> tuple[str, ...]:
    """Read prompts in subtask order.

    Accepted layouts: ``["...", ...]``; ``{"subtasks"|"prompts"|"stages": [...]}``
    whose items are strings or objects with ``"prompt"`` or ``"en"`` (the A1
    ``stage_prompts_14.json`` layout).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        key = next((k for k in ("subtasks", "prompts", "stages") if k in data), None)
        if key is None:
            raise ValueError(
                "subtask JSON object needs a 'subtasks', 'prompts' or 'stages' list"
            )
        data = data[key]
    if not isinstance(data, list) or not 1 <= len(data) <= limit:
        raise ValueError(f"subtask JSON must list 1..{limit} prompts in order")
    prompts = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, dict):
            item = item.get("prompt", item.get("en"))
        if not isinstance(item, str) or not item.strip() or len(item) > 4096:
            raise ValueError(f"subtask {index} needs a non-empty prompt string")
        prompts.append(item)
    return tuple(prompts)
