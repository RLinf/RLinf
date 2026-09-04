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

"""Dual-arm Franka environment with absolute TCP waypoint actions.

Only ``rotation_repr='rot6d'`` is currently supported. Each action contains
left and right position, rot6d orientation, and gripper commands. Per-arm
Cartesian impedance trackers receive the converted quaternion poses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.actions import ActionKind
from rlinf.utils.rot6d import matrix_to_rot6d, rot6d_to_quat_xyzw_safe

from .dual_base import DualFrankaEnv, DualFrankaRobotConfig

ACTION_DIM_PER_ARM = 10  # xyz, 6D rotation, and gripper.
PROPRIO_DIM_PER_ARM = 9  # xyz and 6D rotation; gripper uses a separate slot.


@dataclass
class DualFrankaTCPRobotConfig(DualFrankaRobotConfig):
    """Config for :class:`DualFrankaTCPEnv`."""

    # Other orientation representations are rejected during space initialization.
    rotation_repr: str = "rot6d"


class DualFrankaTCPEnv(DualFrankaEnv):
    """Dual-arm Franka environment with absolute TCP waypoint actions."""

    CONFIG_CLS: type[DualFrankaTCPRobotConfig] = DualFrankaTCPRobotConfig

    PER_ARM_ACTION_DIM = ACTION_DIM_PER_ARM
    GRIPPER_IDX_IN_ARM = 9  # Gripper channel after xyz and 6D rotation.

    def arm_action_kind(self) -> ActionKind:
        """Return absolute Cartesian-pose semantics."""
        return ActionKind.CARTESIAN_POSE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Retain quaternion hemispheres across steps for smooth interpolation.
        self._prev_step_quat = [None, None]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        self._prev_step_quat = [None, None]
        return super().reset(seed=seed, options=options)

    # Spaces.

    def _init_action_obs_spaces(self) -> None:
        if self.config.rotation_repr != "rot6d":
            raise NotImplementedError(
                f"DualFrankaTCPEnv currently only supports rotation_repr='rot6d', "
                f"got {self.config.rotation_repr!r}."
            )
        self._cartesian_safety_boxes()

        # Leave headroom for normalization before Gram-Schmidt projection.
        rot6d_low = -1.5 * np.ones(6, dtype=np.float32)
        rot6d_high = 1.5 * np.ones(6, dtype=np.float32)
        left_low = np.concatenate(
            [self.config.ee_pose_limit_min[0, :3], rot6d_low, np.array([-1.0])]
        )
        left_high = np.concatenate(
            [self.config.ee_pose_limit_max[0, :3], rot6d_high, np.array([1.0])]
        )
        right_low = np.concatenate(
            [self.config.ee_pose_limit_min[1, :3], rot6d_low, np.array([-1.0])]
        )
        right_high = np.concatenate(
            [self.config.ee_pose_limit_max[1, :3], rot6d_high, np.array([1.0])]
        )
        act_low = np.concatenate([left_low, right_low]).astype(np.float32)
        act_high = np.concatenate([left_high, right_high]).astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high)

        camera_specs = self._all_camera_specs()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(2,)),
                        "tcp_pose_rot6d": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(2 * PROPRIO_DIM_PER_ARM,),
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(224, 224, 3), dtype=np.uint8
                        )
                        for name, _, _ in camera_specs
                    }
                ),
            }
        )

    # Motion dispatch.

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        del dt

        targets = []
        for arm in range(2):
            xyz = actions[arm, 0:3]
            rot6d = actions[arm, 3:9]

            prev_quat = self._prev_step_quat[arm]
            if prev_quat is None:
                prev_quat = states[arm].tcp_pose[3:]
            quat = rot6d_to_quat_xyzw_safe(rot6d, fallback_quat_xyzw=prev_quat)
            if float(np.dot(quat, prev_quat)) < 0.0:
                quat = -quat
            self._prev_step_quat[arm] = quat
            targets.append(np.concatenate([xyz, quat]).astype(np.float64))

        # Commanding the named part rather than the driver keeps this task
        # working on any arm backend that accepts Cartesian targets.
        self._run_arm_calls(
            lambda: ctrls[0].send_action({"tcp_pose": targets[0]}),
            lambda: ctrls[1].send_action({"tcp_pose": targets[1]}),
        )

    # Observation and pose helpers.

    def _get_observation(self) -> dict[str, Any]:
        if self.config.is_dummy:
            return self.observation_space.sample()
        frames = self._get_camera_frames()

        state = {
            "gripper_position": np.array(
                [
                    self._left_hand.position,
                    self._right_hand.position,
                ],
                dtype=np.float32,
            ),
            "tcp_pose_rot6d": self._tcp_rot6d_18d(),
        }
        return copy.deepcopy({"state": state, "frames": frames})

    def _tcp_rot6d_18d(self) -> np.ndarray:
        """Return both TCP poses using rot6d orientation."""
        out = np.zeros(18, dtype=np.float32)
        for arm, st in enumerate((self._left_state, self._right_state)):
            base = arm * PROPRIO_DIM_PER_ARM
            out[base : base + 3] = st.tcp_pose[:3]
            mat = R.from_quat(st.tcp_pose[3:]).as_matrix()
            out[base + 3 : base + 9] = matrix_to_rot6d(mat)
        return out
