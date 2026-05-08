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

"""Helpers for naming scheduler channels."""

from __future__ import annotations

from typing import Any


def channel_name(cfg: Any, base_name: str) -> str:
    """Return a channel name with an optional runner-level prefix."""
    runner_cfg = cfg.get("runner", {}) if hasattr(cfg, "get") else {}
    prefix = (
        runner_cfg.get("channel_prefix", None) if hasattr(runner_cfg, "get") else None
    )
    return f"{prefix}_{base_name}" if prefix else base_name
