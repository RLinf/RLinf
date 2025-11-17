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

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

from rlinf.envs.worldmodel.evac.lvdm.data.statistics import StatisticInfo
from rlinf.envs.worldmodel.evac.lvdm.data.traj_vis_statistics import (
    EEF2CamLeft,
    EEF2CamRight,
)


def get_action_bias_std(domain_name):
    return torch.tensor(StatisticInfo[domain_name]["mean"]).unsqueeze(0), torch.tensor(
        StatisticInfo[domain_name]["std"]
    ).unsqueeze(0)


def quat_to_euler(quat):
    """
    quat: [N, 4] or [4,] 四元数 (w, x, y, z) 或 (x, y, z, w)
    返回: [N, 3] 欧拉角 (roll, pitch, yaw)
    """
    # 如果你的四元数顺序是 (x, y, z, w)，需要转换为 (w, x, y, z)
    # quat = quat[[3, 0, 1, 2]] if quat.shape[-1] == 4 else quat
    rot = R.from_quat(quat)  # 默认(x, y, z, w)
    euler = rot.as_euler("xyz", degrees=False)  # 返回 roll, pitch, yaw
    return euler


def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (
        np.mod(radius, 2 * np.pi) > np.pi
    )
    return radius_normed


def get_delta_action(action_list):
    delta_action = np.zeros((1, 14), dtype=np.float16)  # [N-1, 16]
    print("====enter get_delta_action====")
    print(f"action_list: {action_list.shape}")  # (5, 16)
    print(f"delta_action: {delta_action.shape}")  # (1, 14)
    delta_action[:, 0:3] = action_list[-1, 0:3] - action_list[-5, 0:3]  # xyz
    print(f"四元数减数和被减数: {action_list[-1, 3:7]}, {action_list[-5, 3:7]}")
    print(f"四元数减数和被减数的差值: {action_list[-1, 3:7] - action_list[-5, 3:7]}")
    delta_action[:, 3:6] = normalize_angles(
        quat_to_euler(action_list[-1, 3:7] - action_list[-5, 3:7])
    )  # 四元数残差归一化
    delta_action[:, 6] = (action_list[-1, 7] - action_list[-5, 7]) / 120.0  # 夹爪归一化
    # 右臂
    delta_action[:, 7:10] = action_list[-1, 8:11] - action_list[-5, 8:11]  # xyz
    delta_action[:, 10:13] = normalize_angles(
        quat_to_euler(action_list[-1, 11:15] - action_list[-5, 11:15])
    )  # 四元数残差归一化
    delta_action[:, 13] = (
        action_list[-1:, 15] - action_list[-5, 15]
    ) / 120.0  # 夹爪归一化
    delta_action = torch.FloatTensor(delta_action)

    domain_name = "agibotworld"
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)
    sep = 3.0
    delta_action[:, :6] = (delta_action[:, :6] - sep * delta_act_meanv[:, :6]) / (
        sep * delta_act_stdv[:, :6]
    )
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep * delta_act_meanv[:, 6:]) / (
        sep * delta_act_stdv[:, 6:]
    )
    return delta_action


def get_actions(
    gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None
):
    n = all_ends_p.shape[0]
    slices = list(range(all_ends_p.shape[0]))

    all_left_rpy = []
    all_right_rpy = []
    all_left_quat = []
    all_right_quat = []

    # cam eef 30...CAM_ANGLE...
    # 获取旋转变换对象
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    for i in slices:
        rot_l = Rotation.from_quat(all_ends_o[i, 0])  # [t = i, 4]
        rot_vis_l = rot_l * cvt_vis_l
        # 左末端到相机坐标系的旋转
        left_vis_quat = np.concatenate((all_ends_p[i, 0], rot_vis_l.as_quat()), axis=0)
        # 3 + 4
        left_rpy = np.concatenate(
            (all_ends_p[i, 0], rot_l.as_euler("xyz", degrees=False)), axis=0
        )
        # 3 + 3
        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r * cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i, 1], rot_vis_r.as_quat()), axis=0)
        right_rpy = np.concatenate(
            (all_ends_p[i, 1], rot_r.as_euler("xyz", degrees=False)), axis=0
        )

        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)
        all_left_quat.append(left_vis_quat)
        all_right_quat.append(right_vis_quat)

    # xyz, rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)
    # xyz, xyzw
    all_left_quat = np.stack(all_left_quat)
    all_right_quat = np.stack(all_right_quat)

    # xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 16])
    # xyz, rpy, gripper
    all_delta_actions = np.zeros([n - delta_act_sidx, 14])
    # 前4帧重复

    # T -1
    for i in range(0, n):
        all_abs_actions[i, 0:7] = all_left_quat[i, :7]
        all_abs_actions[i, 7] = gripper[slices[i], 0]
        all_abs_actions[i, 8:15] = all_right_quat[i, :7]
        all_abs_actions[i, 15] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i - delta_act_sidx, 0:6] = (
                all_left_rpy[i, :6] - all_left_rpy[i - 1, :6]
            )
            all_delta_actions[i - delta_act_sidx, 3:6] = normalize_angles(
                all_delta_actions[i - delta_act_sidx, 3:6]
            )
            all_delta_actions[i - delta_act_sidx, 6] = gripper[slices[i], 0] / 120.0
            all_delta_actions[i - delta_act_sidx, 7:13] = (
                all_right_rpy[i, :6] - all_right_rpy[i - 1, :6]
            )
            all_delta_actions[i - delta_act_sidx, 10:13] = normalize_angles(
                all_delta_actions[i - delta_act_sidx, 10:13]
            )
            all_delta_actions[i - delta_act_sidx, 13] = gripper[slices[i], 1] / 120.0

    return all_abs_actions, all_delta_actions


def get_action_sim(abs_act):
    """
    action_list: [N, 16]
    """
    n_previous = 4
    action, delta_action = get_actions(
        gripper=np.stack((abs_act[:, 7], abs_act[:, 15]), axis=1),
        all_ends_p=np.stack((abs_act[:, 0:3], abs_act[:, 8:11]), axis=1),
        all_ends_o=np.stack((abs_act[:, 3:7], abs_act[:, 11:15]), axis=1),
        slices=None,
        delta_act_sidx=n_previous,
    )
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    domain_name = "agibotworld"
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)
    sep = 3
    delta_action[:, :6] = (delta_action[:, :6] - sep * delta_act_meanv[:, :6]) / (
        sep * delta_act_stdv[:, :6]
    )
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep * delta_act_meanv[:, 6:]) / (
        sep * delta_act_stdv[:, 6:]
    )
    return action, delta_action
