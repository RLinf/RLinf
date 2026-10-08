# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SO-101 env adapter for the canonical OpenPI data contract."""

from __future__ import annotations

from typing import Callable


def repack_env_obs(env_obs: dict, *, select_state: Callable) -> dict:
    """Repack canonical SO-101 observations for OpenPI."""
    states = select_state(env_obs["states"])
    return {
        "observation/image": env_obs["main_images"],
        "prompt": env_obs["task_descriptions"],
        "observation/state": states,
        **(
            {"observation/wrist_image": env_obs["wrist_images"]}
            if env_obs.get("wrist_images") is not None
            else {}
        ),
        **(
            {"observation/extra_view_image": env_obs["extra_view_images"]}
            if env_obs.get("extra_view_images") is not None
            else {}
        ),
    }
