import numpy as np
from ..franka_env import FrankaEnv, FrankaRobotConfig
import gymnasium as gym
import copy
import time
from rlinf.utils.utils import euler_2_quat, quat_2_euler


TARGET_POSE = np.array(
    [
        0.5748380998638687,0.05450998790669838,0.016262587587812682,-3.111252051323321,0.0001972290844429736,-0.043024099516195013
    ]
)



class PegInsertionConfig(FrankaRobotConfig):
    def __post_init__(self):
        self.apply_gripper_penalty = False
        self.random_xy_range = 0.05
        self.random_rz_range = np.pi / 6
        self.enable_random_reset = True
        self.compliance_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.003,
            "translational_clip_y": 0.003,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.003,
            "translational_clip_neg_y": 0.003,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.02,
            "rotational_clip_y": 0.02,
            "rotational_clip_z": 0.02,
            "rotational_clip_neg_x": 0.02,
            "rotational_clip_neg_y": 0.02,
            "rotational_clip_neg_z": 0.02,
            "rotational_Ki": 0,
        }
        self.precision_param = {
            "translational_stiffness": 3000,
            "translational_damping": 89,
            "rotational_stiffness": 300,
            "rotational_damping": 9,
            "translational_Ki": 0.1,
            "translational_clip_x": 0.01,
            "translational_clip_y": 0.01,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.01,
            "translational_clip_neg_y": 0.01,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.05,
            "rotational_clip_y": 0.05,
            "rotational_clip_z": 0.05,
            "rotational_clip_neg_x": 0.05,
            "rotational_clip_neg_y": 0.05,
            "rotational_clip_neg_z": 0.05,
            "rotational_Ki": 0.1,
        }
        self.target_ee_pose = TARGET_POSE
        self.reset_ee_pose = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        self.reward_threshold = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
        self.action_scale = np.array([0.02, 0.1, 1])
        self.ee_pose_limit_min = np.array(
        [
            TARGET_POSE[0] - self.random_xy_range,
            TARGET_POSE[1] - self.random_xy_range,
            TARGET_POSE[2],
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - self.random_rz_range,
        ]
    )
        self.ee_pose_limit_max  = np.array(
        [
            TARGET_POSE[0] + self.random_xy_range,
            TARGET_POSE[1] + self.random_xy_range,
            TARGET_POSE[2] + 0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + self.random_rz_range,
        ]
    )
        super().__post_init__()


class PegInsertionEnv(FrankaEnv):
    def __init__(self, override_cfg):
        # Update config according to current env
        config = PegInsertionConfig(**override_cfg)
        super().__init__(config)

    @property
    def task_description(self):
        return "peg and insertion"

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._gripper_action(-1)
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)
        self._franka_state = self._controller.get_state().wait()[0]
        # Move up to clear the slot
        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)

        super().go_to_rest(joint_reset)

