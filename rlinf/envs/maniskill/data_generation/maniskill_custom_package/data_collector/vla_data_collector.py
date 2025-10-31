# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cv2
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from PIL import Image

from rlinf.envs.maniskill.data_generation.maniskill_custom_package.controller_utils.delta_pose import (
    get_numpy,
    to_numpy,
)


class VLADataCollector:
    def __init__(
        self,
        env: BaseEnv,
        camera_name: str,
        is_image_encode: bool = False,
        proprioception_type: str = "qpos",
        *args,
        **kwargs,
    ):
        """
        Args:
        is_image_encode:
            When toggled, compress the raw image tensor into JPEG format (to save memory or facilitate storage/transmission).
        proprioception_type: the type of information needed to describe the robot's proprioceptive states. Supported types: \
            **qpos** (joint angles), **qvel** (joint velocities) for a static robot, \
            and also includes root_pose, root_vel, root_angular_vel for a mobile robot, and all_robot_state for all of them. \
            Currently we do not support returning the controller's states but you can add it from ManiSkill/mani_skill/agents/base_agent.py
        """

        self.env = env.unwrapped
        self.camera_name = camera_name
        self.is_image_encode = is_image_encode

        self.data_dict = self.get_empty_data_dict()
        self.proprioception_type: str = proprioception_type

        # check for plausible proprioception_types
        candidate_proprioception_types = [
            "qpos",
            "qvel",
            "root_pose",
            "root_vel",
            "root_angular_vel",
            "all_robot_state",
        ]
        if self.proprioception_type not in candidate_proprioception_types:
            raise ValueError(
                f"Unsupported proprioception_type:{proprioception_type}, it should be one of {candidate_proprioception_types} !"
            )

    def get_empty_data_dict(self):
        data_dict = {
            "is_image_encode": self.is_image_encode,
            "image": [],
            "instruction": None,
            "action": [],
            "state": [],  # Revised
            "info": [],
        }
        return data_dict

    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()

    def get_data(self):
        return to_numpy(self.data_dict, self.env.unwrapped.device)

    def save_data(self, save_path, is_compressed=False):
        """Save data as .npy file with dictionary structure."""
        saving_data = to_numpy(self.data_dict, self.env.unwrapped.device)
        saving_data["image"] = [
            Image.fromarray(im).convert("RGB") for im in saving_data["image"]
        ]
        if is_compressed:
            np.savez_compressed(save_path, saving_data)
            print(f"save data at {save_path}.npz.")
        else:
            np.save(save_path, saving_data)
            print(f"save data at {save_path}.npy.")
        self.clear_data()

    def update_instruction(self):
        if self.data_dict["instruction"] is None:
            self.data_dict["instruction"] = self.env.get_language_instruction()
        else:
            return

    # should run before env.step()
    def update_image(self, camera_name=None):
        if camera_name is None:
            rgb = self.env.render().squeeze(0).to(torch.uint8)
        else:
            rgb = (
                self.env.get_obs()["sensor_data"][camera_name]["rgb"]
                .squeeze(0)
                .to(torch.uint8)
            )
        if self.is_image_encode:
            success, encoded_rgb = cv2.imencode(
                ".jpeg",
                get_numpy(rgb, self.env.unwrapped.device),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if not success:
                raise ValueError("JPEG encode error.")
            img_bytes = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)
            rgb = img_bytes

        self.data_dict["image"].append(rgb)

    # should run before env.step()
    def update_state(self):
        """
        Write the robot's proprioception, which is a numpy arra and is determined by self.proprioception_type,
        to self.data_dict['state'].
        """
        # Revised
        state = None
        self.env: BaseEnv
        if self.proprioception_type == "qpos":
            state = (
                self.env.agent.robot.get_qpos()
            )  # [N, 6] without grippers, [N,7] or [N,8] with grippers
        elif self.proprioception_type == "qvel":
            state = (
                self.env.agent.robot.get_qvel()
            )  # [N, 6] without grippers, [N,7] or [N,8] with grippers
        elif self.proprioception_type == "root_pose":
            state = self.env.agent.robot.root.pose  # [N, 3]
        elif self.proprioception_type == "root_vel":
            state = self.env.agent.robot.root.get_linear_velocity()  # [N, 3]
        elif self.proprioception_type == "root_angular_vel":
            state = self.env.agent.robot.root.get_angular_velocity()  # [N, 3]
        else:
            state = self.env.agent.robot.get_state()

        # for motionplanning, we run one instance per CPU thread and use the first and only state as the robot's state.
        state = state[0].cpu().numpy()
        self.data_dict["state"].append(state)  # [state_dim,]

    # should run before env.step()
    def update_action(self, action):
        self.data_dict["action"].append(action)

    def updata_info(
        self,
    ):
        info = self.env.get_info()
        self.data_dict["info"].append(info)

    # should run before env.step()
    def update_data_dict(self, action):
        self.update_instruction()
        self.update_image(self.camera_name)
        self.update_state()  # Revised
        self.updata_info()
        self.update_action(action.astype(np.float32))
