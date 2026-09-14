# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CALVIN env-obs adapter: split ee pose / rotation / gripper."""

from __future__ import annotations

from typing import Callable


def repack_env_obs(env_obs: dict, *, select_state: Callable) -> dict:
    """Map CALVIN env observations onto split ``observation/state_*`` keys."""
    env_states = select_state(env_obs["states"])
    processed_obs = {
        "observation/image": env_obs["main_images"],
        "prompt": env_obs["task_descriptions"],
        "observation/state_ee_pos": env_states[:, :3],
        "observation/state_ee_rot": env_states[:, 3:6],
        "observation/state_gripper": env_states[:, 6:7],
    }
    wrist_images = env_obs.get("wrist_images")
    if wrist_images is not None:
        processed_obs["observation/wrist_image"] = wrist_images
    extra_view_images = env_obs.get("extra_view_images")
    if extra_view_images is not None:
        processed_obs["observation/extra_view_image"] = extra_view_images
    return processed_obs
