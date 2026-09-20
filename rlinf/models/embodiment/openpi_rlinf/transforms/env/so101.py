# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SO-101 env adapter for the OpenPI degree-like data contract."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def repack_env_obs(env_obs: dict, *, select_state: Callable) -> dict:
    """Convert radians/normalized gripper observations to OpenPI units."""
    states = select_state(env_obs["states"])
    if torch.is_tensor(states):
        states = states.clone()
        states[..., :5] *= float(180.0 / np.pi * 0.01)
    else:
        states = np.asarray(states, dtype=np.float32).copy()
        states[..., :5] *= np.float32(180.0 / np.pi * 0.01)
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
