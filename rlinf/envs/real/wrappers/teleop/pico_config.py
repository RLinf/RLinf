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

"""Split shared PICO configuration into left and right controller settings."""

from __future__ import annotations

from typing import Any, Mapping


def _as_dict(config: Any) -> dict[str, Any]:
    return {} if config is None else dict(config)


def split_dual_config(pico_config: Mapping[str, Any]) -> tuple[dict, dict]:
    """Return the left and right controller configs, in that order."""
    cfg = _as_dict(pico_config)
    shared = {k: v for k, v in cfg.items() if k not in ("hand", "left", "right")}
    shared_calibration = _as_dict(shared.get("calibration"))

    sides = []
    for side in ("left", "right"):
        override = _as_dict(cfg.get(side))
        side_cfg = shared.copy()
        side_cfg.update(override)
        if shared_calibration or "calibration" in override:
            calibration = shared_calibration.copy()
            calibration.update(_as_dict(override.get("calibration")))
            side_cfg["calibration"] = calibration
        # Bind each controller to the matching robot side.
        side_cfg["hand"] = side
        sides.append(side_cfg)
    return sides[0], sides[1]
