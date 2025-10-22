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


import numpy as np
import torch
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
import sapien.physx as physx
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.controller_utils.kinematics import Kinematics
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
from mani_skill.utils.structs.pose import Pose
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.controller_utils.delta_pose import controller_delta_pose_calculate
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.controller_utils.qpos_pose_transfer import transfer_qpos_2_ee_pose
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.data_collector.vla_data_collector import VLADataCollector
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.motionplanning.widowx.mplib_non_convex import NonConvexPlanner


OPEN = 1
CLOSED = -1


class WidowXArmMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = False,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        plan_time_step=None):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None
        self.transfer_q = torch.tensor([0.7071068, 0, -0.7071068, 0,]) # -90 deg
        self.plan_time_step = plan_time_step if plan_time_step is not None else self.base_env.control_timestep

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = NonConvexPlanner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="ee_gripper_link",
            joint_vel_limits=np.ones(6) * self.joint_vel_limits,
            joint_acc_limits=np.ones(6) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q) * sapien.Pose(q=self.transfer_q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.plan_time_step,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p , q=pose.q) * sapien.Pose(q=self.transfer_q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.plan_time_step,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.plan_time_step,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self, t=5):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=5, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass

def get_numpy(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        if device == "cpu":
            return data.numpy()
        else:
            return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError("parameter passed is not torch.tensor")

class VLADataCollectWidowXArmMotionPlanningSolver(WidowXArmMotionPlanningSolver):
    def __init__(self, camera_name='3rd_view_camera', proprioception_type='qpos', *args, **kwargs):
        # Revised
        super().__init__(*args, **kwargs)
        if camera_name !='3rd_view_camera':
            raise ValueError(f"Currently we only support 3rd_view_camera, but got {camera_name}!")
        self.camera_name = camera_name  # the Bridge dataset only records 3rd_view_camera. 
        self.proprioception_type = proprioception_type     # Tonghe: by default, pi-zero receives qpos as proprioception type. 
        self.data_collector = VLADataCollector(self.env, self.camera_name, is_image_encode=False, proprioception_type=self.proprioception_type) # Revised
        self.last_abs_action_pose = None
        self.num = 0
        self.kinematics = Kinematics(
            urdf_path=self.env.agent.urdf_path,
            end_link_name=self.env.agent.ee_link_name,
            articulation=self.env.agent.robot,
            active_joint_indices=get_active_joint_indices(self.env.agent.robot, self.env.agent.arm_joint_names),
        )

    def preprocess_qpos(self, abs_qpos):
        """
            transfer raw_qpos(q_pos) to action(delta_ee_pose)
            just for qpos in pd_ee_delta_pose control mode
        """
        if self.last_abs_action_pose == None:
            abs_action_pose = transfer_qpos_2_ee_pose(self.env, self.kinematics, abs_qpos, world_frame = False)
            self.last_abs_action_pose = abs_action_pose

            delta_action = np.hstack([np.zeros((6,)), self.gripper_state])
            abs_action = torch.cat([self.env.agent.ee_pose_at_robot_base.p[0],
                                matrix_to_euler_angles(quaternion_to_matrix(self.env.agent.ee_pose_at_robot_base.q[0]),"XYZ"),
                                torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
        else:
            abs_action_pose = transfer_qpos_2_ee_pose(self.env, self.kinematics, abs_qpos, world_frame = False)
            delta_xyz, delta_euler_angle = controller_delta_pose_calculate(
                                                        self.env.agent.controller.configs["arm"].frame,
                                                        self.last_abs_action_pose.to_transformation_matrix().squeeze(0),
                                                        abs_action_pose.to_transformation_matrix().squeeze(0),
                                                        self.env.device)
            self.last_abs_action_pose = abs_action_pose
            delta_action = torch.cat([delta_xyz, delta_euler_angle,
                                      torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
            abs_action = torch.cat([abs_action_pose.p[0],matrix_to_euler_angles(quaternion_to_matrix(abs_action_pose.q[0]),"XYZ"),
                                torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
        return get_numpy(delta_action,self.env.unwrapped.device), get_numpy(abs_action,self.env.unwrapped.device), abs_action_pose

    def follow_path(self, result, refine_steps: int = 0): 
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            # for widowX.collect with MP, control_model==arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos
            qpos = result["position"][min(i, n_step - 1)]
            # for widowX, qpos is a 6DoF foating-point list.
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            elif self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode in ["pd_ee_delta_pose", "pd_ee_target_delta_pose", "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"] :
                delta_action, abs_action, abs_action_pose = self.preprocess_qpos(qpos) # actually delta action
                action = delta_action
                if self.grasp_pose_visual is not None and self.num % 5 ==0:
                    self.grasp_pose_visual.set_pose(abs_action)
                self.num += 1
            elif self.control_mode == "pd_ee_pose":
                delta_action, abs_action, abs_action_pose = self.preprocess_qpos(qpos)
                action = abs_action
            else:
                raise ValueError(f"motion planning doesn't support control mode {self.control_mode}")
            self.data_collector.update_data_dict(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def open_gripper(self,t=1):
        self.gripper_state = OPEN
        qpos = get_numpy(self.robot.get_qpos()[0, :-2],device=self.env.unwrapped.device)
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_joint_pos_vel":
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            elif self.control_mode in ["pd_ee_delta_pose", "pd_ee_target_delta_pose", "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]:
                action = np.hstack([np.zeros((6,)), self.gripper_state])
            elif self.control_mode == "pd_ee_pose":
                action = torch.cat([self.env.agent.ee_pose_at_robot_base.p[0],
                                    matrix_to_euler_angles(quaternion_to_matrix(self.env.agent.ee_pose_at_robot_base.q[0]),"XYZ"),
                                    torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
            else:
                raise ValueError(f"motion planning doesn't support control mode {self.control_mode}")
            self.data_collector.update_data_dict(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=1, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        qpos = get_numpy(self.robot.get_qpos()[0, :-2], device=self.env.unwrapped.device)

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_joint_pos_vel":
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            elif self.control_mode in ["pd_ee_delta_pose", "pd_ee_target_delta_pose", "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]:
                action = np.hstack([np.zeros((6,)), self.gripper_state])
            elif self.control_mode == "pd_ee_pose":
                action = torch.cat([self.env.agent.ee_pose_at_robot_base.p[0],
                                    matrix_to_euler_angles(quaternion_to_matrix(self.env.agent.ee_pose_at_robot_base.q[0]),"XYZ"),
                                    torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
            else:
                raise ValueError(f"motion planning doesn't support control mode {self.control_mode}")
            self.data_collector.update_data_dict(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

from transforms3d import quaternions


def build_panda_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual
