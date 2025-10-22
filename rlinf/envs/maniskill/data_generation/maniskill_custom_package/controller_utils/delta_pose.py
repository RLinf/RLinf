import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

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

def to_numpy(data, device="cpu",):
    if isinstance(data, torch.Tensor):
        return get_numpy(data, device)
    elif isinstance(data, dict):
        return {key: to_numpy(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        try:
            return np.array([to_numpy(item, device) for item in data])
        except ValueError:
            return [to_numpy(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_numpy(item, device) for item in data)
    elif isinstance(data, bytes):
        return np.frombuffer(data, dtype=np.uint8)
    else:
        return data

def controller_delta_pose_calculate(delta_control_mode, pose0, pose1, device):
    """
    Args:
        delta_control_mode: str, can be one of:
                'root_translation:root_aligned_body_rotation', # default for every agent
                'root_translation:body_aligned_body_rotation',
                'body_translation:root_aligned_body_rotation',
                'body_translation:body_aligned_body_rotation',
        pose0: last pose (4x4 torch tensor). in the robot base frame.
        pose1: current pose (4x4 torch tensor). in the robot base frame.
    Returns:
        delta_xyz:  (torch.tensor, shape (3,)). in m
        delta_euler_angle:  (torch tensor, shape (3,)). in radian
    """
    from scipy.spatial.transform import Rotation as R
    pose0 = get_numpy(pose0, device)
    pose1 = get_numpy(pose1, device)
    if delta_control_mode=='root_translation:root_aligned_body_rotation':
        # calculate delta translation
        delta_xyz = pose1[:3, 3] - pose0[:3, 3]

        # calculate delta rotation
        rot0 = R.from_matrix(pose0[:3, :3])
        rot1 = R.from_matrix(pose1[:3, :3])

        delta_rot = rot1 * rot0.inv() # root aligned rotation
        # get delta euler_angle
        delta_euler_angle = delta_rot.as_euler('xyz', degrees=False)
        # # get delta quaternion
        # delta_quat = delta_rot.as_quat()
        # # get delta axis-angle
        # delta_axis_angle = delta_rot.as_rotvec() # Use as_rotvec for axis-angle
        # delta_angle = np.linalg.norm(delta_axis_angle)
        # if delta_angle > 1e-6:
        #     delta_axis = delta_axis_angle / delta_angle
        # else:
        #     delta_axis = np.array([1, 0, 0])  # any direction is ok
    elif delta_control_mode=='root_translation:body_aligned_body_rotation':
        # calculate delta translation
        delta_xyz = pose1[:3, 3] - pose0[:3, 3]

        # calculate delta rotation
        rot0 = R.from_matrix(pose0[:3, :3])
        rot1 = R.from_matrix(pose1[:3, :3])

        delta_rot = rot0.inv() * rot1 # rotation for body aligned rotation
        # get delta euler_angle
        delta_euler_angle = delta_rot.as_euler('xyz', degrees=False)
    else:
        raise ValueError("""
                         Now we only support or root_translation, but if you want to use
                         body_translation:body_aligned_body_rotation mode, you can use 
                         Homogeneous Transformation Matri(T).
                         The comments below are as demenstration.
                         T_last.inv() * T_now
                         """)
        # # For controller mode: body translation + body aligned rotation
        # delta_pose = (self.last_abs_action_pose.inv() * abs_action_pose)
        # delta_xyz = delta_pose.p[0]
        # delta_euler_angle = matrix_to_euler_angles(quaternion_to_matrix(delta_pose.q[0]),"XYZ")
    return torch.from_numpy(delta_xyz).float(), torch.from_numpy(delta_euler_angle).float()
