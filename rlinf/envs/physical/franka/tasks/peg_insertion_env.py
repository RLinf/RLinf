import numpy as np
from ..franka_env import FrankaEnv
import gymnasium as gym
import copy
import time
from rlinf.utils.utils import euler_2_quat, quat_2_euler

class PegInsertionEnv(FrankaEnv):
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
        reset_pose[1] -= 0.05
        self._move_action(reset_pose)

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
