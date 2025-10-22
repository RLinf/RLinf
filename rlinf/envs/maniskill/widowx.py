import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250S(BaseAgent):
    uid = "widowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]
    ee_link_name = "ee_gripper_link" # Valid

    def get_state(self):
        state = super().get_state()
        state["ee_pose"] = self.robot.find_link_by_name(self.ee_link_name).pose # in maniskill world frame
        return state

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    def _after_init(self):
        self.tcp = self.robot.find_link_by_name(self.ee_link_name)
        # self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    @property
    def ee_pose_at_robot_base(self): # in robot frame(root frame)
        to_base = self.robot.pose.inv()
        return to_base * (self.tcp.pose)

@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SSimpler(WidowX250S):
    uid = "widowx250s_simpler"
    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]

    arm_stiffness = [
        1169.7891719504198,
        730.0,
        808.4601346394447,
        1229.1299089624076,
        1272.2760456418862,
        1056.3326605132252,
    ]
    arm_damping = [
        330.0,
        180.0,
        152.12036565582588,
        309.6215302722146,
        201.04998711007383,
        269.51458932695414,
    ]

    arm_force_limit = [200, 200, 100, 100, 100, 100]
    arm_friction = 0.0
    arm_vel_limit = 1.5
    arm_acc_limit = 2.0

    gripper_stiffness = 1000
    gripper_damping = 200
    gripper_pid_stiffness = 1000
    gripper_pid_damping = 200
    gripper_pid_integral = 300
    gripper_force_limit = 60
    gripper_vel_limit = 0.12
    gripper_acc_limit = 0.50
    gripper_jerk_limit = 5.0

    @property
    def _controller_configs(self):
        arm_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,  # dummy limit, which is unused since normalize_action=False
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="ee_gripper_link",
            urdf_path=self.urdf_path,
            normalize_action=False,
            use_delta=True
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            **arm_common_kwargs, use_target=True
        )

        extra_gripper_clearance = 0.001  # since real gripper is PID, we use extra clearance to mitigate PD small errors; also a trick to have force when grasping
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.015 - extra_gripper_clearance,
            upper=0.037 + extra_gripper_clearance,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        controller = dict(
            arm=arm_pd_ee_target_delta_pose_align2, gripper=gripper_pd_joint_pos
        )
        return dict(arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=controller)

