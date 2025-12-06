import numpy as np
from ..franka_env import FrankaEnv, FrankaRobotConfig
import gymnasium as gym
import copy
import time
from rlinf.utils.utils import euler_2_quat, quat_2_euler


TARGET_POSE = np.array(
    [
        0.5906439143742067,
        0.07771711953459341,
        0.0937835826958042,
        3.1099675,
        0.0146619,
        -0.0078615,
    ]
)

class PegInsertionConfig(FrankaRobotConfig):
    apply_gripper_penalty = False
    random_reset = True
    random_xy_range = 0.05
    random_rz_range = np.pi / 6

    def __post_init__(self):
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


class PegInsertionEnv(FrankaEnv):
    def __init__(self, overwride_cfg):
        # Update config according to current env
        config = PegInsertionConfig()
        config.update_from_dict(overwride_cfg)
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

    def step(self, action):
        """
        - action: [6, ]; append 1 to [7, ]
        """
        new_action = np.zeros(7)
        new_action[:6] = action
        return super().step(new_action)
