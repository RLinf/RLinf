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

import json
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.env_access import get_behavior_robot


def trace_robot_joints_enabled(cfg: DictConfig) -> bool:
    cfg_value = bool(OmegaConf.select(cfg, "trace_robot_joints", default=False))
    env_value = os.environ.get("RLINF_BEHAVIOR_TRACE_JOINTS", "")
    return cfg_value or env_value.lower() in {"1", "true", "yes", "on"}


def tensor_to_float_list(value: Any, digits: int = 6) -> list[float]:
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [round(float(item), digits) for item in value]


def robot_joint_names(robot: Any) -> list[str]:
    joints = getattr(robot, "joints", None)
    if isinstance(joints, dict):
        return list(joints.keys())
    return []


class RobotJointTracer:
    def __init__(
        self,
        *,
        enabled: bool,
        replay_seed_offset: int,
    ):
        self.enabled = enabled
        self.replay_seed_offset = replay_seed_offset
        self.frame_idx = 0

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        *,
        replay_seed_offset: int,
    ) -> "RobotJointTracer":
        return cls(
            enabled=trace_robot_joints_enabled(cfg),
            replay_seed_offset=replay_seed_offset,
        )

    def log_env(
        self,
        child_env,
        local_env_idx: int,
        event: str,
        chunk_step_idx: int | None = None,
        action=None,
    ) -> None:
        if not self.enabled:
            return

        robot = get_behavior_robot(child_env)
        if robot is None:
            return

        get_joint_velocities = getattr(robot, "get_joint_velocities", None)
        if callable(get_joint_velocities):
            joint_velocities = get_joint_velocities()
        else:
            joint_velocities = None
        robot_pos, robot_quat = robot.get_position_orientation(frame="world")
        record = {
            "event": event,
            "replay_seed_offset": self.replay_seed_offset,
            "local_env_idx": int(local_env_idx),
            "frame_idx": int(self.frame_idx),
            "chunk_step_idx": None if chunk_step_idx is None else int(chunk_step_idx),
            "robot_name": getattr(robot, "name", None),
            "joint_names": robot_joint_names(robot),
            "joint_positions": tensor_to_float_list(robot.get_joint_positions()),
            "joint_velocities": tensor_to_float_list(joint_velocities),
            "robot_position_world": tensor_to_float_list(robot_pos),
            "robot_quat_world": tensor_to_float_list(robot_quat),
        }
        if action is not None:
            record["action"] = tensor_to_float_list(action)
        print(
            f"RLINF_BEHAVIOR_JOINT_TRACE {json.dumps(record, sort_keys=True)}",
            flush=True,
        )

    def log_all(
        self,
        child_envs: list,
        event: str,
        env_indices: list[int] | None = None,
        chunk_step_idx: int | None = None,
        actions=None,
    ) -> None:
        if not self.enabled:
            return

        if env_indices is None:
            env_indices = list(range(len(child_envs)))
        for env_idx in env_indices:
            action = None
            if actions is not None:
                action = actions[env_idx]
            self.log_env(
                child_envs[env_idx],
                local_env_idx=env_idx,
                event=event,
                chunk_step_idx=chunk_step_idx,
                action=action,
            )
        self.frame_idx += 1
