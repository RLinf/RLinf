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

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.worldmodel.evac.lvdm.data.traj_vis_statistics import (
    EEF2CamLeft,
    EEF2CamRight,
)


def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (
        np.mod(radius, 2 * np.pi) > np.pi
    )
    return radius_normed


def get_actions(
    gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None
):
    if delta_act_sidx is None:
        delta_act_sidx = 1
    # delta_act_sidx = 4
    if slices is None:
        # the first frame is repeated to fill memory
        n = all_ends_p.shape[0] - 1 + delta_act_sidx
        # n = T + 3
        slices = [
            0,
        ] * (delta_act_sidx - 1) + list(range(all_ends_p.shape[0]))
        # 0 * 3 + [0, 1, 2, ..., T-1]
    else:
        n = len(slices)

    all_left_rpy = []
    all_right_rpy = []
    all_left_quat = []
    all_right_quat = []

    # cam eef 30...CAM_ANGLE...
    # 获取旋转变换对象
    # 末端到相机
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    for i in slices:
        # 0 * 3 + [0, 1, 2, ..., T-1]
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


def get_single_arm_actions(
    gripper, all_ends_p, all_ends_o, slices=None, delta_act_sidx=4
):
    # all_ends_o: [T, 3] or [T, 1, 3]，欧拉角xyz
    # print(f'get_single_arm_actions all_ends_p shape: {all_ends_p.shape}, all_ends_o shape: {all_ends_o.shape}')
    if all_ends_o.ndim == 3:
        all_ends_o = all_ends_o[:, 0]  # 变成 [T, 3]
    if slices is None:
        n = all_ends_p.shape[0] - 1 + delta_act_sidx
        slices = [
            0,
        ] * (delta_act_sidx - 1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_rpy = []

    # all_ends_p = all_ends_p + np.random.uniform(-0.1, 0.1, all_ends_p.shape)

    for i in slices:
        rpy = all_ends_o[i]  # 直接用欧拉角
        all_rpy.append(np.concatenate((all_ends_p[i], rpy), axis=0))  # [3+3]

    all_rpy = np.stack(all_rpy)  # [n, 6]

    all_abs_actions = np.zeros([n, 7])  # [xyz, rpy, gripper]
    all_delta_actions = np.zeros([n - delta_act_sidx, 7])  # [dxyz, drpy, dgripper]

    for i in range(n):
        all_abs_actions[i, 0:6] = all_rpy[i]
        all_abs_actions[i, 6] = gripper[slices[i]]
        if i >= delta_act_sidx:
            all_delta_actions[i - delta_act_sidx, 0:3] = (
                all_rpy[i, :3] - all_rpy[i - 1, :3]
            )
            all_delta_actions[i - delta_act_sidx, 3:6] = normalize_angles(
                all_rpy[i, 3:] - all_rpy[i - 1, 3:]
            )
            all_delta_actions[i - delta_act_sidx, 6] = gripper[slices[i]] / 0.04

    # 提取
    xyz = all_abs_actions[:, 0:3]
    euler = all_abs_actions[:, 3:6]
    gripper = all_abs_actions[:, 6:7]

    # 欧拉角转四元数
    quat = Rotation.from_euler("xyz", euler).as_quat()  # [275, 4]，xyzw

    # 拼接成 [xyz, quat, gripper]
    all_abs_actions_quat = np.concatenate([xyz, quat, gripper], axis=1)  # [275, 8]

    return all_abs_actions_quat, all_delta_actions


def parse_h5(h5_file, slices=None, delta_act_sidx=1):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        all_abs_gripper = np.array(fid["state/effector/position"], dtype=np.float32)
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)

    all_abs_actions, all_delta_actions = get_actions(
        gripper=all_abs_gripper,
        slices=slices,
        delta_act_sidx=delta_act_sidx,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
    )
    return all_abs_actions, all_delta_actions
