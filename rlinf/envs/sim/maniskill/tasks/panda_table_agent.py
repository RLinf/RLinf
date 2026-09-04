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


from copy import deepcopy

import numpy as np
import sapien
from mani_skill import ASSET_DIR
from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.sensors.camera import CameraConfig

BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"


@register_agent()
class PandaBridgeDatasetFlatTable(PandaWristCam):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_bridgedataset_flat_table"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_ee_delta_pose_real_root_frame = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,  # -1.0,
            pos_upper=0.1,  # 1.0,
            rot_lower=-0.1,  # -np.pi / 2,
            rot_upper=0.1,  # np.pi / 2,
            stiffness=[
                37.800000000000004,
                29.925,
                48.3,
                48.3,
                2.1284343434343436,
                27.3,
                48.3,
            ],
            damping=[10.5, 10.5, 10.5, 10.5, 0.6353535353535353, 10.5, 10.5],
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_ee_delta_pose_real_root_frame.use_target = True
        arm_pd_ee_delta_pose_real = deepcopy(arm_pd_ee_delta_pose_real_root_frame)
        arm_pd_ee_delta_pose_real.frame = "root_translation:body_aligned_body_rotation"

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )

        controller_configs = {
            "pd_ee_body_target_delta_pose_real": {
                "arm": arm_pd_ee_delta_pose_real,
                "gripper": gripper_pd_joint_pos,
            },
            "pd_ee_body_target_delta_pose_real_root_frame": {
                "arm": arm_pd_ee_delta_pose_real_root_frame,
                "gripper": gripper_pd_joint_pos,
            },
        }

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def ee_pose_at_robot_base(self):  # in robot frame(root frame)
        to_base = self.robot.pose.inv()
        return to_base * (self.tcp.pose)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used in the Bridge dataset
                pose=sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
                * sapien.Pose(
                    [0, -0.16, 0.36],  # 0, -0.16, 0.36
                    [0.8992917, -0.09263245, 0.35892478, 0.23209205],
                ),
                width=640,
                height=480,
                # entity_uid="panda_link0",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
            CameraConfig(
                uid="c19_front_view",
                pose=sapien.Pose(
                    [0.1840, 0.2000, 1.4000], q=[0.2541, 0.3510, 0.1361, -0.8909]
                ),
                width=640,
                height=480,
                fov=0.81,
                near=0.1,
                far=1000,
            ),
        ]
