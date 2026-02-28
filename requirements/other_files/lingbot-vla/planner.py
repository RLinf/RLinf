import traceback

import envs._GLOBAL_CONFIGS as CONFIGS
import mplib
import numpy as np
import toppra as ta
import transforms3d as t3d
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld

try:
    # ********************** CuroboPlanner (optional) **********************

    import torch
    import yaml
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import JointState
    from curobo.util import logger
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )

    logger.setup_logger(level="error", logger_name="curobo")

    class CuroboPlanner:
        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            # motion generation
            if True:
                world_config = {
                    "cuboid": {
                        "table": {
                            "dims": [0.7, 2, 0.04],  # x, y, z
                            "pose": [
                                self.robot_origion_pose.p[1],
                                0.0,
                                0.74 - self.robot_origion_pose.p[2],
                                1,
                                0,
                                0,
                                0.0,
                            ],  # x, y, z, qw, qx, qy, qz
                        },
                    }
                }
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup()
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
                num_graph_seeds=1,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):
            world_base_pose = np.concatenate(
                [
                    np.array(self.robot_origion_pose.p),
                    np.array(self.robot_origion_pose.q),
                ]
            )
            world_target_pose = np.concatenate(
                [np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)]
            )
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                world_base_pose, world_target_pose
            )
            if "aloha-agilex" not in self.yml_path:
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else:  # patch for aloha-agilex
                T_target = t3d.affines.compose(
                    target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1]
                )
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            goal_pose_of_ee = CuroboPose.from_list(
                list(target_pose_p) + list(target_pose_q)
            )
            joint_indices = [
                self.all_joints.index(name)
                for name in self.active_joints_name
                if name in self.all_joints
            ]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [
                round(angle, 5) for angle in joint_angles
            ]  # avoid the precision problem
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(
                        constraint_pose
                    ),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            result = self.motion_gen.plan_single(
                start_joint_states, goal_pose_of_ee, plan_config
            )

            # output
            res_result = dict()
            if result.success.item() == False:
                res_result["status"] = "Fail"
                return res_result
            else:
                res_result["status"] = "Success"
                res_result["position"] = np.array(
                    result.interpolated_plan.position.to("cpu")
                )
                res_result["velocity"] = np.array(
                    result.interpolated_plan.velocity.to("cpu")
                )
                return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            num_poses = len(target_gripper_pose_list)
            # transformation from world to arm's base
            world_base_pose = np.concatenate(
                [
                    np.array(self.robot_origion_pose.p),
                    np.array(self.robot_origion_pose.q),
                ]
            )
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate(
                    [np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)]
                )
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(
                    world_base_pose, world_target_pose
                )

                if "aloha-agilex" not in self.yml_path:
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else:  # patch for aloha-agilex
                    T_target = t3d.affines.compose(
                        base_target_pose_p,
                        t3d.quaternions.quat2mat(base_target_pose_q),
                        [1, 1, 1],
                    )
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(
                    base_target_pose_q
                )
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [
                self.all_joints.index(name)
                for name in self.active_joints_name
                if name in self.all_joints
            ]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [
                round(angle, 5) for angle in joint_angles
            ]  # avoid the precision problem
            joint_angles_cuda = (
                torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1)
            )
            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(
                joint_angles_cuda, joint_names=self.active_joints_name
            )
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(
                        constraint_pose
                    ),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                result = self.motion_gen_batch.plan_batch(
                    start_joint_states, goal_pose_of_ee, plan_config
                )
            except Exception:
                return {"status": ["Failure" for i in range(10)]}

            # output
            res_result = dict()
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(
                ["Success" if s else "Failure" for s in success_array], dtype=object
            )
            res_result["status"] = status_array

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(
                result.interpolated_plan.position.to("cpu")
            )
            res_result["velocity"] = np.array(
                result.interpolated_plan.velocity.to("cpu")
            )
            return res_result

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            """
            transform target pose from world frame to base frame
            base_pose: np.array([x, y, z, qw, qx, qy, qz])
            target_pose: np.array([x, y, z, qw, qx, qy, qz])
            """
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q

except Exception:
    print(
        "[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually."
    )
    print("Exception traceback:")
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(
                now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log
            )

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
