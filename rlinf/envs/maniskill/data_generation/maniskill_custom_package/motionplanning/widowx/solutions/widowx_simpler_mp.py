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


# MIT License

# Copyright (c) 2024 simpler-env

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
import sapien
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene import (
    PutCarrotOnPlateInScene,
    PutEggplantInBasketScene,
    PutSpoonOnTableClothInScene,
    StackGreenCubeOnYellowCubeBakedTexInScene,
)
from transforms3d.euler import euler2quat

from rlinf.envs.maniskill.data_generation.maniskill_custom_package.motionplanning.widowx.motionplanner import (
    VLADataCollectWidowXArmMotionPlanningSolver,
    WidowXArmMotionPlanningSolver,
    get_numpy,
)
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.motionplanning.widowx.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


# put source object to target object
def pick_and_place(
    env,
    seed=None,
    debug=False,
    vis=False,
    use_rrt=False,
    plan_time_step=None,
    simpler_collcet=True,
):
    assert env.unwrapped.control_mode in [
        "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    ], f"Invalid control mode: {env.unwrapped.control_mode}."

    cls = (
        VLADataCollectWidowXArmMotionPlanningSolver
        if simpler_collcet
        else WidowXArmMotionPlanningSolver
    )
    planner = cls(
        camera_name="3rd_view_camera",  # Revised,
        proprioception_type="qpos",  # Revised,
        env=env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        plan_time_step=plan_time_step,
    )
    FINGER_LENGTH = 0.45
    env = env.unwrapped
    source_obb = get_actor_obb(env.objs[env.source_obj_name])
    target_obb = get_actor_obb(env.objs[env.target_obj_name])

    # All the Pose and direction are defined in the world frame,  the conversion to the robot base(root) frame is handled by the motion planner.
    approaching_dir = np.array([0, 0, -1])
    target_closing = get_numpy(
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1], env.device
    )
    # grasp_info = get_grasp_info(actor = env.objs[env.source_obj_name], obb = source_obb,
    #                    depth = FINGER_LENGTH, offset=0, approaching=approaching_dir)
    grasp_info = compute_grasp_info_by_obb(
        source_obb,
        approaching=approaching_dir,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching_dir, closing, center)

    # Search a valid pose
    angles = np.array(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, -np.pi * 3 / 4, -np.pi / 2, -np.pi / 4]
    )  # 7
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = (
            planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
            if not use_rrt
            else planner.move_to_pose_with_RRTConnect(grasp_pose2, dry_run=True)
        )
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")
        return -1, planner

    refine_steps = 0
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
    res = (
        planner.move_to_pose_with_screw(reach_pose, refine_steps=refine_steps)
        if not use_rrt
        else planner.move_to_pose_with_RRTConnect(reach_pose, refine_steps=refine_steps)
    )
    if res == -1:
        return -1, planner
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = (
        planner.move_to_pose_with_screw(grasp_pose, refine_steps=refine_steps)
        if not use_rrt
        else planner.move_to_pose_with_RRTConnect(grasp_pose, refine_steps=refine_steps)
    )
    if res == -1:
        return -1, planner

    res = planner.close_gripper(t=1)
    if res == -1:
        return -1, planner

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    res = (
        planner.move_to_pose_with_screw(lift_pose, refine_steps=refine_steps)
        if not use_rrt
        else planner.move_to_pose_with_RRTConnect(lift_pose, refine_steps=refine_steps)
    )
    if res == -1:
        return -1, planner
    # -------------------------------------------------------------------------- #
    # place
    # -------------------------------------------------------------------------- #
    if isinstance(env, PutEggplantInBasketScene):
        temp_pose_p = env.objs[env.target_obj_name].pose.p.squeeze(0)
        temp_pose_p[2] = lift_pose.p[2].item()
        temp_pose = sapien.Pose(p=get_numpy(temp_pose_p, env.device), q=lift_pose.q)
        res = (
            planner.move_to_pose_with_screw(temp_pose, refine_steps=refine_steps)
            if not use_rrt
            else planner.move_to_pose_with_RRTConnect(
                temp_pose, refine_steps=refine_steps
            )
        )
        # Tonghe changed from end_pose to temp_pose, although use_rrt is False.
        if res == -1:
            return -1, planner
    elif isinstance(env, StackGreenCubeOnYellowCubeBakedTexInScene):
        grasp_info = compute_grasp_info_by_obb(
            target_obb,
            approaching=approaching_dir,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.agent.build_grasp_pose(approaching_dir, closing, center)
        offset = np.array([0, 0, 0.04])
        end_pose = sapien.Pose(p=grasp_pose.p + offset, q=grasp_pose.q)
        res = (
            planner.move_to_pose_with_screw(end_pose, refine_steps=refine_steps)
            if not use_rrt
            else planner.move_to_pose_with_RRTConnect(
                end_pose, refine_steps=refine_steps
            )
        )

        res = planner.open_gripper(t=1)
        return res, planner

    offset = np.array([0, 0, 0.03])
    end_pose = sapien.Pose(
        p=get_numpy(env.objs[env.target_obj_name].pose.p.squeeze(0), env.device)
        + offset,
        q=lift_pose.q,
    )
    res = (
        planner.move_to_pose_with_screw(end_pose, refine_steps=refine_steps)
        if not use_rrt
        else planner.move_to_pose_with_RRTConnect(end_pose, refine_steps=refine_steps)
    )
    if res == -1:
        return -1, planner

    res = planner.open_gripper(t=2)  # must be 2, to get the success signal
    return res, planner


def SolveStackCube(
    env: StackGreenCubeOnYellowCubeBakedTexInScene,
    seed=None,
    debug=False,
    vis=False,
    use_rrt=False,
    plan_time_step=None,
):
    return pick_and_place(env, seed, debug, vis, use_rrt, plan_time_step)


def SolvePutSpoon(
    env: PutSpoonOnTableClothInScene,
    seed=None,
    debug=False,
    vis=False,
    use_rrt=False,
    plan_time_step=None,
):
    return pick_and_place(env, seed, debug, vis, use_rrt, plan_time_step)


def SolvePutCarrot(
    env: PutCarrotOnPlateInScene,
    seed=None,
    debug=False,
    vis=False,
    use_rrt=False,
    plan_time_step=None,
):
    return pick_and_place(env, seed, debug, vis, use_rrt, plan_time_step)


def SolvePutEggplant(
    env: PutEggplantInBasketScene,
    seed=None,
    debug=False,
    vis=False,
    use_rrt=False,
    plan_time_step=None,
):
    return pick_and_place(env, seed, debug, vis, use_rrt, plan_time_step)
