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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_turtle2_deploy_action_mode(
    override_cfg: Mapping[str, Any] | None,
) -> str:
    """Resolve Turtle2 deploy action mode from override_cfg."""
    override_cfg = override_cfg or {}
    action_mode = str(override_cfg.get("action_mode", "relative_pose")).lower()
    if action_mode not in {"relative_pose", "absolute_pose"}:
        raise ValueError(
            f"Unsupported Turtle2 deploy action_mode={action_mode!r}. "
            "Expected one of {'relative_pose', 'absolute_pose'}."
        )
    return action_mode


def validate_turtle2_takeover_requirements(
    env_cfg: Mapping[str, Any],
    override_cfg: Mapping[str, Any] | None,
    *,
    split_label: str | None = None,
) -> None:
    """Validate Turtle2 master-takeover invariants.

    No-op when ``use_master_takeover`` is not enabled. ``split_label`` (e.g.
    ``"eval"``) is only used to make error messages point at the offending
    config path.
    """
    if not env_cfg.get("use_master_takeover", False):
        return
    prefix = f"env.{split_label}." if split_label else ""
    action_mode = resolve_turtle2_deploy_action_mode(override_cfg)
    if action_mode != "absolute_pose":
        raise ValueError(
            f"{prefix}use_master_takeover=True requires "
            "override_cfg.action_mode='absolute_pose'. "
            "Master takeover poses are absolute dual-arm commands and "
            "cannot be interpreted as relative_pose actions."
        )
    override_cfg = override_cfg or {}
    pose_control_backend = str(
        override_cfg.get("pose_control_backend", "smooth")
    ).lower()
    if pose_control_backend != "hybrid":
        raise ValueError(
            f"{prefix}use_master_takeover=True requires "
            "override_cfg.pose_control_backend='hybrid'."
        )
