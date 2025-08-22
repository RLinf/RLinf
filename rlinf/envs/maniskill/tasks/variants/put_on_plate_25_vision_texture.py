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

import torch
import torch.nn.functional as F
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from rlinf.envs.maniskill.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)
from rlinf.envs.maniskill.tasks.variants.utils import (
    masks_to_boxes_pytorch,
)


@register_env(
    "PutOnPlateInScene25VisionTexture03-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25VisionTexture03(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    overlay_texture_mix_ratio = 0.3

    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            le = 1
            le_offset = 0
        elif self.obj_set == "test":
            le = 16
            le_offset = 1
        elif self.obj_set == "all":
            le = 17
            le_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        lp_offset = 0
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset

    @property
    def total_num_trials(self):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset = (
            self.basic_obj_infos
        )
        ltt = lc * le * lp * lo * l1 * l2
        return ltt

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset = (
            self.basic_obj_infos
        )
        self._reset_episode_idx(env_idx, self.total_num_trials, options)

        self.select_carrot_ids = (
            self.episode_id // (le * lp * lo * l1 * l2) + lc_offset
        )  # [b]
        self.select_extra_ids = (
            self.episode_id // (lp * lo * l1 * l2)
        ) % le + le_offset  # [b]
        self.select_plate_ids = (self.episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (self.episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (self.episode_id // l2) % l1
        self.select_quat_ids = self.episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        self._reset_overlay(env_idx)

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0],
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = (
                torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = (
                torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(
                is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset
            )  # [b, 3]

            q_reset = (
                torch.tensor([0, 0, 0, 1], device=self.device)
                .reshape(1, -1)
                .repeat(b, 1)
            )  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(
                is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset
            )  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        # self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            pass
            # self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(carrot_actor)]
        )  # [b, 4]
        self.plate_q_after_settle = torch.stack(
            [a.pose.q[idx] for idx, a in enumerate(plate_actor)]
        )  # [b, 4]
        corner_signs = torch.tensor(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            device=self.device,
        )

        # carrot
        carrot_bbox_world = torch.stack(
            [self.model_bbox_sizes[n] for n in select_carrot]
        )  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(
            self.carrot_q_after_settle
        )  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(
            c_bbox_corners, c_q_matrix.transpose(1, 2)
        )  # [b, 8, 3]
        c_rotated_bbox_size = (
            c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values
        )  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack(
            [self.model_bbox_sizes[n] for n in select_plate]
        )  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(
            self.plate_q_after_settle
        )  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(
            p_bbox_corners, p_q_matrix.transpose(1, 2)
        )  # [b, 8, 3]
        p_rotated_bbox_size = (
            p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values
        )  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        self._reset_stats(env_idx)

    def _green_sceen_rgb(
        self, rgb, segmentation, overlay_img, overlay_texture, overlay_mix
    ):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        # mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(
                actor_seg.device
            )

        robot_item_ids = torch.concat(
            [
                self.robot_link_ids,
                self.target_object_actor_ids,
            ]
        )
        arm_obj_mask = torch.isin(actor_seg, robot_item_ids)  # [b, H, W]

        mask = (~arm_obj_mask).to(torch.float32).unsqueeze(-1)  # [b, H, W, 1]
        mix = overlay_mix.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 1]
        mix = mix * self.overlay_texture_mix_ratio
        assert rgb.shape == overlay_img.shape
        assert rgb.shape == overlay_texture.shape

        # Step 1
        b, H, W, _ = mask.shape
        boxes = masks_to_boxes_pytorch(arm_obj_mask)  # [b, 4], [xmin, ymin, xmax, ymax]

        # Step 2
        xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(4)]  # [b]
        h_box = (ymax - ymin + 1).clamp(min=1)  # [b]
        w_box = (xmax - xmin + 1).clamp(min=1)  # [b]

        # Step 3
        max_h, max_w = h_box.max().item(), w_box.max().item()
        texture = overlay_texture.permute(0, 3, 1, 2).float()  # [b, 3, H_tex, W_tex]
        texture_resized = F.interpolate(
            texture, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        # [b, 3, max_h, max_w]

        # Step 4
        rgb = rgb.to(torch.float32)
        rgb_ret = overlay_img * mask

        for i in range(b):
            tex_crop = texture_resized[i, :, : h_box[i], : w_box[i]].permute(
                1, 2, 0
            )  # [h_box, w_box, 3]
            y0, y1 = ymin[i].item(), (ymin[i] + h_box[i]).item()
            x0, x1 = xmin[i].item(), (xmin[i] + w_box[i]).item()
            rgb_box = rgb[i, y0:y1, x0:x1, :]  # [h_box, w_box, 3]
            overlay_img_box = overlay_img[i, y0:y1, x0:x1, :]  # [h_box, w_box, 3]
            mask_box = arm_obj_mask[i, y0:y1, x0:x1]  # [h_box, w_box]
            mix_val = mix[i].item()

            mask_box_3 = mask_box.unsqueeze(-1).to(rgb_box.dtype)  # [h_box, w_box, 1]
            blended = tex_crop * mix_val + rgb_box * (1.0 - mix_val)
            out_box = blended * mask_box_3 + overlay_img_box * (1 - mask_box_3)

            rgb_ret[i, y0:y1, x0:x1, :] = out_box

        rgb_ret = torch.clamp(rgb_ret, 0, 255)
        rgb_ret = rgb_ret.to(torch.uint8)

        return rgb_ret


@register_env(
    "PutOnPlateInScene25VisionTexture05-v1",
    max_episode_steps=80,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutOnPlateInScene25VisionTexture05(PutOnPlateInScene25VisionTexture03):
    overlay_texture_mix_ratio = 0.5
