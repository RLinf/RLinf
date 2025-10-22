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


import time
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.controller_utils.delta_pose import get_numpy, to_numpy
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix, matrix_to_quaternion,
    )
from mani_skill.utils.structs.pose import Pose
import cv2

class WorldModelDataCollector:
    """
        [Note]: All the xyz, quat, velocity are in the robot base frame, except the robot velocity.
        data_dict: dict
            task_info: dict
                task_name: list[str] -> [1] Description of the entire task set, e.g., "Pick items on the table".
                init_scene_text: list[str] -> [1] Description of the initial scene.
                action_text: list[str] -> [1] Description of the action trajectory, e.g., "Pick the pen on the table".
                skill: list[str] -> [1] Skill used in the task, e.g., Grasping, Pushing, Placing, Throwing.

            timestamp: list[np.int64] -> [N] Timestamp in nanoseconds (using time.time_ns()).

            observation: dict
                rgb: list[np.ndarray] -> [N, 480, 640, 3] RGB image from a third-person camera beside the robot workbench.
                depth: list[np.ndarray] -> [N, 480, 640, 1] Depth image from a third-person camera beside the robot workbench.

            state: dict
                effector: dict
                    position_gripper: list[np.ndarray] -> [N, 1] Gripper joint position in radian.
                end: dict
                    orientation: list[np.ndarray] -> [N, 1, 4] End-effector quaternion [w, x, y, z].
                    position: list[np.ndarray] -> [N, 1, 3] End-effector XYZ position in meters.
                joint: dict
                    position: list[np.ndarray] -> [N, 7] Seven joint angles in radians.
                robot: dict
                    orientation: list[np.ndarray] -> [N, 4] Robot quaternion (yaw-only rotation) [w, x, y, z].
                    position: list[np.ndarray] -> [N, 3] Robot base position in world frame, where z is always 0 (meters).

            action: dict
                effector: dict
                    position_gripper: list[np.ndarray] -> [N, 1]  
                        - AGIBOT: 0 for fully open, 1 for fully closed.  
                        - Ours: 1 for fully open, -1 for fully closed.  
                end: dict
                    orientation: list[np.ndarray] -> [N, 1, 4] Same as /state/end/orientation (wxyz).
                    position: list[np.ndarray] -> [N, 1, 3] Same as /state/end/position (meters).
                joint: dict
                    position: list[np.ndarray] -> [N, 7] Same as /state/joint/position (radians).
                robot: dict
                    velocity: list[np.ndarray] -> [N, 2]  
                        - [:, 0] -> Velocity along the x-axis. in world frame
                        - [:, 1] -> Yaw rate. in world frame

            reserve: dict
                target_object: dict
                    position: list[np.ndarray] -> [N, 3] XYZ position of the target object (meters). in robot base frame.
                    orientation: list[np.ndarray] -> [N, 4] Quaternion (wxyz) of the target object. in robot base frame.
                grasp_point: dict
                    position
                    orientation
    """
    class WorldModelDataStruct:
        def __init__(self,):
            self.data_dict = self.get_empty_data_dict()

        def get_empty_data_dict(self):
            data_dict = {
                "task_info":{
                    "task_name":[],
                    "init_scene_text":[],
                    "action_text":[],
                    "skill":[],
                },
                "timestamp": [],
                "observation":{
                    "rgb": [],
                    "depth": [],
                },
                "state":{
                    "effector":{
                        "position_gripper":[],
                    },
                    "end":{
                        "orientation":[],
                        "position":[],
                    },
                    "joint":{
                        "position":[],
                    },
                    "robot":{
                        "orientation":[],
                        "position":[],
                    },
                },
                "action":{
                    "effector":{
                        "position_gripper":[],
                    },
                    "end":{
                        "orientation":[],
                        "position":[],
                    },
                    "joint":{
                        "position":[],
                    },
                    "robot":{
                        "velocity":[],
                    },
                },
                "reserve":{
                    "target_object":{
                        "orientation": [],
                        "position": [],
                    },
                    "grasp_point":{
                        "orientation": [],
                        "position": [],
                    },
                },
            }
            return data_dict
            
        def clear_data(self):
            """Clear all collected data."""
            self.data_dict = self.get_empty_data_dict()

        def get_data_summary(self):
            """Print a summary of the collected data."""
            total_entries = len(self.data_dict["timestamps"])
            print(f"Total data entries: {total_entries}")
            if total_entries > 0:
                print(f"First timestamp in ns : {self.data_dict['timestamps'][0]}")
                print(f"Last timestamp in ns: {self.data_dict['timestamps'][-1]}")

        def save_to_file(self, save_path):
            """Save data as .npy file with dictionary structure."""

            np.savez_compressed(save_path, self.data_dict)
            print(f"save data at {save_path}.npz.")

            self.clear_data()

        def load_from_file(self, loding_path):
            """Load data and reconstruct as dictionary.
                Just load .npy
            """
            return np.load(loding_path, allow_pickle=True).tolist()

    def __init__(self, env: BaseEnv, *args, **kwargs,):
        self.env = env.unwrapped
        self.real2sim_data = self.WorldModelDataStruct()

    def get_data(self):
        return to_numpy(self.real2sim_data.data_dict, self.env.unwrapped.device)

    def save_data(self, save_path):
        self.real2sim_data.data_dict = to_numpy(self.real2sim_data.data_dict, self.env.unwrapped.device)
        self.real2sim_data.save_to_file(save_path)

    def set_target_object_actor(self, target_object_actor: Actor):
        self.target_object_actor = target_object_actor

    def set_task_info(self, task_name: str = None, skill: str = None):
        self.real2sim_data.data_dict["task_info"]["task_name"] = ["Picking up items on the table."] * self.env.num_envs if task_name==None else [task_name] * self.env.num_envs
        self.real2sim_data.data_dict["task_info"]["init_scene_text"] = self.env.get_scene_description()
        self.real2sim_data.data_dict["task_info"]["action_text"] = self.env.get_language_instruction()
        self.real2sim_data.data_dict["task_info"]["skill"] = ["Pick"] * self.env.num_envs if skill==None else [skill] * self.env.num_envs

    # should run before env.step()
    def update_timestamp(self):
        # time_stamp
        self.real2sim_data.data_dict["timestamp"].append(time.time_ns())

    # should run before env.step()
    def update_observation(self):
        # observation
        rgb = self.env.get_obs()['sensor_data']['base_camera']['rgb'].squeeze(0)
        depth = self.env.get_obs()['sensor_data']['base_camera']['depth'].squeeze(0)
        success, encoded_rgb = cv2.imencode('.jpeg', get_numpy(rgb,self.env.unwrapped.device), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise ValueError("JPEG encode error.")
        img_bytes = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)
        self.real2sim_data.data_dict["observation"]["rgb"].append(rgb)
        self.real2sim_data.data_dict["observation"]["depth"].append(depth)

    # should run before env.step()
    def update_state(self):
        # effecotr
        qpos = self.env.agent.robot.get_qpos().squeeze(0)

        position_gripper = qpos[...,-1:]
        self.real2sim_data.data_dict["state"]["effector"]["position_gripper"].append(position_gripper)
        # end
        orientation = self.env.agent.ee_pose_at_robot_base.q
        position = self.env.agent.ee_pose_at_robot_base.p
        self.real2sim_data.data_dict["state"]["end"]["orientation"].append(orientation)
        self.real2sim_data.data_dict["state"]["end"]["position"].append(position)
        # joint
        joint_pos = qpos[...,:-2]
        self.real2sim_data.data_dict["state"]["joint"]["position"].append(joint_pos)
        # robot
        robot_pose = self.env.agent.robot.get_pose() # pose, root_pose, .root.pose
        self.real2sim_data.data_dict["state"]["robot"]["orientation"].append(robot_pose.get_q().squeeze(0))
        self.real2sim_data.data_dict["state"]["robot"]["position"].append(robot_pose.get_p().squeeze(0))

    # should run before env.step()
    def update_reserve(self,):
        # traget_object
        target_object_pose_at_robot_base = self.env.agent.robot.pose.inv() * self.target_object_actor.pose  
        orientation = target_object_pose_at_robot_base.q.squeeze(0)
        position = target_object_pose_at_robot_base.p.squeeze(0)
        self.real2sim_data.data_dict["reserve"]["target_object"]["orientation"].append(orientation)
        self.real2sim_data.data_dict["reserve"]["target_object"]["position"].append(position)

    # should run before env.step()
    def update_action(self, control_mode, action):
        if isinstance(action, torch.Tensor):
            action.to(self.env.device)
        elif isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.env.device)
        else:
            raise TypeError(f"action should be torch.tensor of np.ndarray instead of {type(action)}")

        # action
        if control_mode in ["pd_joint_pos"]:
            # joint, effector
            joint_pos = action # in radian
            position_gripper = action[-1] # [-1, 1]

            self.real2sim_data.data_dict["action"]["joint"]["position"].append(joint_pos)
            self.real2sim_data.data_dict["action"]["effector"]["position_gripper"].append(position_gripper)

        elif control_mode in ["pd_ee_delta_pose", "pd_ee_target_delta_pose", "pd_ee_pose",]:
            # end, effector
            position = action[:3].unsqueeze(0) # in meters
            orientation = matrix_to_quaternion(euler_angles_to_matrix(action[3:6], "XYZ")).unsqueeze(0)
            position_gripper = action[6:7] # [-1, 1]

            self.real2sim_data.data_dict["action"]["end"]["orientation"].append(orientation)
            self.real2sim_data.data_dict["action"]["end"]["position"].append(position)
            self.real2sim_data.data_dict["action"]["effector"]["position_gripper"].append(position_gripper)
        else:
            raise ValueError(f"real2sim_datacollection not support {control_mode} mode")

    # should run before env.step()
    def update_data_dict(self, control_mode, action):
        # get from os
        self.update_timestamp()
        # get from env camera
        self.update_observation()
        # get from robot sensor
        self.update_state()
        # get from object itself
        self.update_reserve()
        # get from outside
        self.update_action(control_mode, action)

    def update_target_grasp_point(self, grasp_point: Pose):
        orientation = grasp_point.q
        position = grasp_point.p
        self.real2sim_data.data_dict["reserve"]["grasp_point"]["orientation"].append(orientation)
        self.real2sim_data.data_dict["reserve"]["grasp_point"]["position"].append(position)


