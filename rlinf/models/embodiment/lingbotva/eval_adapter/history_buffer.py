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

"""Episode state helpers for LingBot-VA chunked rollout on Libero."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LingbotVAEpisodeState:
    """Per-environment state tracked across LingBot-VA inference chunks."""

    prompt: str | None = None
    first_obs: dict[str, Any] | None = None
    first_chunk: bool = True
    action_queue: deque[np.ndarray] = field(default_factory=deque)
    prev_model_action: np.ndarray | None = None
    last_action_per_frame: int | None = None
    kv_cache_history: list[tuple[list[dict[str, Any]], np.ndarray]] = field(
        default_factory=list
    )

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self.first_obs = None
        self.first_chunk = True
        self.action_queue.clear()
        self.prev_model_action = None
        self.last_action_per_frame = None
        self.kv_cache_history.clear()
