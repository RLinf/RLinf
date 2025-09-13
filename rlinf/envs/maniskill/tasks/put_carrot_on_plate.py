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
import sapien
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene import (
    PutCarrotOnPlateInScene,
)
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

"""
We override the original PutCarrotOnPlateInScene because we do not want the _settle behavior in the _initialize_episode function.
Since the initial states of the carrots and the plates are appropriate, delete the _settle function does not harm simulation quality.
Also, removing the _settle function make the partial reset behavior correct.
"""


@register_env(
    "PutCarrotOnPlateInScene-v2",
    max_episode_steps=60,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PutCarrotOnPlateInSceneV2(PutCarrotOnPlateInScene):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        with torch.device(self.device):
            b = len(env_idx)
            if "episode_id" in options:
                if isinstance(options["episode_id"], int):
                    options["episode_id"] = torch.tensor([options["episode_id"]])
                    assert len(options["episode_id"]) == b
                pos_episode_ids = (
                    options["episode_id"]
                    % (len(self.xyz_configs) * len(self.quat_configs))
                ) // len(self.quat_configs)
                quat_episode_ids = options["episode_id"] % len(self.quat_configs)
            else:
                pos_episode_ids = torch.randint(0, len(self.xyz_configs), size=(b,))
                quat_episode_ids = torch.randint(0, len(self.quat_configs), size=(b,))
            for i, actor in enumerate(self.objs.values()):
                xyz = self.xyz_configs[pos_episode_ids, i]
                actor.set_pose(
                    Pose.create_from_pq(p=xyz, q=self.quat_configs[quat_episode_ids, i])
                )

            # measured values for bridge dataset
            if self.scene_setting == "flat_table":
                qpos = np.array(
                    [
                        -0.01840777,
                        0.0398835,
                        0.22242722,
                        -0.00460194,
                        1.36524296,
                        0.00153398,
                        0.037,
                        0.037,
                    ]
                )

                self.agent.robot.set_pose(
                    sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
                )
            elif self.scene_setting == "sink":
                qpos = np.array(
                    [
                        -0.2600599,
                        -0.12875618,
                        0.04461369,
                        -0.00652761,
                        1.7033415,
                        -0.26983038,
                        0.037,
                        0.037,
                    ]
                )
                self.agent.robot.set_pose(
                    sapien.Pose([0.127, 0.060, 0.85], q=[0, 0, 0, 1])
                )
            self.agent.reset(init_qpos=qpos)

            # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
            self.episode_source_obj_xyz_after_settle = self.objs[
                self.source_obj_name
            ].pose.p
            self.episode_target_obj_xyz_after_settle = self.objs[
                self.target_obj_name
            ].pose.p
            self.episode_obj_xyzs_after_settle = {
                obj_name: self.objs[obj_name].pose.p for obj_name in self.objs.keys()
            }
            self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[
                self.source_obj_name
            ].float()
            self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[
                self.target_obj_name
            ].float()
            self.episode_source_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(
                    self.objs[self.source_obj_name].pose.q
                )
                @ self.episode_source_obj_bbox_world[..., None]
            )[0, :, 0]
            """source object bbox size (3, )"""
            self.episode_target_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(
                    self.objs[self.target_obj_name].pose.q
                )
                @ self.episode_target_obj_bbox_world[..., None]
            )[0, :, 0]
            """target object bbox size (3, )"""

            if getattr(self, "consecutive_grasp", None) is None:
                self.consecutive_grasp = torch.zeros(
                    self.num_envs, dtype=torch.int32
                ).to(self.device)
            if getattr(self, "episode_stats", None) is None:
                self.episode_stats = {
                    "moved_correct_obj": torch.zeros(
                        (self.num_envs,), dtype=torch.bool
                    ).to(self.device),
                    "moved_wrong_obj": torch.zeros(
                        (self.num_envs,), dtype=torch.bool
                    ).to(self.device),
                    "is_src_obj_grasped": torch.zeros(
                        (self.num_envs,), dtype=torch.bool
                    ).to(self.device),
                    "consecutive_grasp": torch.zeros(
                        (self.num_envs,), dtype=torch.bool
                    ).to(self.device),
                }
            # stats to track
            self.consecutive_grasp[env_idx] = 0
            self.episode_stats["moved_correct_obj"][env_idx] = 0
            self.episode_stats["moved_wrong_obj"][env_idx] = 0
            self.episode_stats["is_src_obj_grasped"][env_idx] = 0
            self.episode_stats["consecutive_grasp"][env_idx] = 0
