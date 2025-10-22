import torch

from mani_skill.envs.sapien_env import BaseEnv
from rlinf.envs.maniskill.data_generation.maniskill_custom_package.controller_utils.kinematics import Kinematics
from mani_skill.agents.utils import get_active_joint_indices

def transfer_qpos_2_ee_pose(env: BaseEnv, kinematics: Kinematics, qpos, world_frame:bool = False):
    """Transfer joint positions to ee pose in the world frame"""
    qpos = torch.as_tensor(qpos.squeeze())
    qpos_fk = torch.zeros(qpos.shape[0], env.agent.robot.max_dof, # 8
                        dtype=qpos.dtype, device=env.agent.robot.device)
    qpos_fk[:, get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names)] = qpos.to(device=env.agent.robot.device)
    ee_pose = kinematics.compute_fk(qpos_fk)
    if world_frame: 
        return ee_pose * env.agent.robot.root.pose
    return ee_pose
