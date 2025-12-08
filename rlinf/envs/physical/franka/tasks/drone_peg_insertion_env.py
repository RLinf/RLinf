import numpy as np
from ..franka_env import FrankaEnv, FrankaRobotConfig
import gymnasium as gym
import copy
import time
from rlinf.utils.utils import euler_2_quat, quat_2_euler


# [0.506938502936494,0.3954955734380723,0.6411658779176086,-2.8977165916149517,-1.4797545119150894,1.3497943895469893]
# [0.44631335973319547,0.3922720584821552,0.6411032368470745,-2.6936208323365602,-1.4729969356396122,1.1050705957650027]

TARGET_POSE = np.array(
    [
        0.5083962757132434,0.4011011731302831,0.6398856749479279,
        -2.9041451876755344,-1.5498064793662414,1.315375849279958
    ]
)


class DronePegInsertionConfig(FrankaRobotConfig):
    enable_random_reset: bool = True
    random_xz_range: float = 0.05
    random_ry_range: float = 0.01 # np.pi / 6

    enable_gripper_penalty: bool = False

    def __post_init__(self):
        self.joint_reset_qpos = np.array([0, 0, 0, -1.9, -0, 2, 0])
        self.compliance_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.01,
            "translational_clip_y": 0.01,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.01,
            "translational_clip_neg_y": 0.01,
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
        self.reset_ee_pose = TARGET_POSE + np.array([0.0, -0.15, 0.0, 0.0, 0.0, 0.0])
        self.reward_threshold = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
        self.action_scale = np.array([0.05, 0.05, 0.05])
        self.ee_pose_limit_min = np.array(
            [
                TARGET_POSE[0] - self.random_xz_range,
                TARGET_POSE[1] - 0.15, # 0.1
                TARGET_POSE[2] - self.random_xz_range,
                TARGET_POSE[3] - 0.01,
                TARGET_POSE[4] - self.random_ry_range,
                TARGET_POSE[5] - 0.01,
            ]
        )
        self.ee_pose_limit_max = np.array(
            [
                TARGET_POSE[0] + self.random_xz_range,
                TARGET_POSE[1],
                TARGET_POSE[2] + self.random_xz_range,
                TARGET_POSE[3] + 0.01,
                TARGET_POSE[4] + self.random_ry_range,
                TARGET_POSE[5] + 0.01,
            ]
        )
        super().__post_init__()

class DronePegInsertionEnv(FrankaEnv):
    def __init__(self, overwride_cfg):
        # Update config according to current env
        config = DronePegInsertionConfig(**overwride_cfg)
        super().__init__(config)
    
    @property
    def task_description(self):
        return "drone peg and insertion"

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
        reset_pose[1] -= 0.05
        self._move_action(reset_pose)


        joint_reset = False
        if joint_reset:
            raise NotImplementedError("no joint reset")
            self._controller.reset_joint(self._config.joint_reset_qpos).wait()
            time.sleep(0.5)

        # Reset arm
        if self._config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose[[0, 2]] += np.random.uniform(
                -self._config.random_xz_range, self._config.random_xz_range, (2,)
            )
            # euler_random = self._config.target_ee_pose[3:].copy()
            # euler_random[-1] += np.random.uniform(
            #     -self._config.random_rz_range, self._config.random_rz_range
            # )
            
            # reset_pose[3:] = euler_2_quat(self._config.target_ee_pose[3:].copy())
        else:
            reset_pose = self._reset_pose.copy()
        
        self._franka_state = self._controller.get_state().wait()[0]
        cnt = 0
        while not np.allclose(self._franka_state.tcp_pose[:3], reset_pose[:3], 0.02):
            cnt +=1
            delta = self._franka_state.tcp_pose - reset_pose
            print(f"{cnt=}")
            print(f"{self._franka_state.tcp_pose=}")
            print(f"{reset_pose=}")
            print(f"{delta=}")
            self._interpolate_move(reset_pose)
            self._franka_state = self._controller.get_state().wait()[0]
            if cnt > 10:
                break

    def step(self, action):
        """
        - action: [6, ]; append 1 to [7, ]
        """
        # action = np.array([0, 0.1, 0, 0, 0, 0]) #+z

        # action = np.array([0, 0, -0.5, 0, 0, 0]) # +x

        # action = np.array([0.5, 0, 0, 0, 0, 0]) #+y
        new_action = np.zeros(7)
        new_action[:6] = action
        return super().step(new_action)
