# Copyright 2025 The RLinf Authors.
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

from typing import List

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.physical.camera import Camera, CameraInfo
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker

from .franka_controller import FrankaController, FrankaRobotState


class FrankaEnv(gym.Env):
    """Franka robot arm environment. This is adapted from SERL's FrankaEnv."""

    def __init__(self, robot_ip: str, gripper_ip: str, camera_info: List[CameraInfo]):
        """Initialize the Franka robot arm environment.

        Args:
            robot_ip (str): The IP address of the robot arm.
            gripper_ip (str): The IP address of the gripper.
        """
        self._franka_state = FrankaRobotState()

        # Launch Franka controller on the same node as the env
        # TODO: Support launching on different nodes
        self._controller = FrankaController.create_group(robot_ip, gripper_ip).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(
                node_ids=[Worker.current_worker._node_id]
            ),
        )

        # Init cameras
        self._cameras: List[Camera] = []
        self._open_cameras(camera_info)

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        action (np.ndarray): The action to take, which is a 7D vector representing the desired end-effector position and orientation, as well as the gripper action.
        """
        action = np.clip(action, self._action_space.low, self._action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        gripper_action_effective = self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        # dt = time.time() - start_time
        # time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob, gripper_action_effective)
        done = self.curr_path_length >= self.max_episode_length or reward == 1
        return ob, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def _init_action_obs_spaces(self):
        self._action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self._obs_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "arm_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "arm_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "arm_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "arm_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "wrist_1": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        "wrist_2": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )

    def _open_cameras(self, camera_info: List[CameraInfo]):
        for info in camera_info:
            camera = Camera(info)
            camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []
