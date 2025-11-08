import numpy as np
from ..franka_env import FrankaEnv
import gymnasium as gym
import copy
from gym.envs.registration import register
import time
from rlinf.utils.utils import euler_2_quat

class PegInsertionEnv(FrankaEnv):
    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space = gym.spaces.Box(
            low=self._config.position_limit_min[:3],
            high=self._config.position_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self._config.position_limit_min[3:],
            high=self._config.position_limit_max[3:],
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "arm_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "arm_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "arm_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "arm_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "wrist_1": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        "wrist_2": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._gripper_action(-1)
        self._move_action(self._franka_state.arm_position)
        time.sleep(0.5)

        # Move up to clear the slot
        reset_pose = copy.deepcopy(self._franka_state.arm_position)
        reset_pose[1] -= 0.10
        self._interpolate_move(reset_pose, timeout=1)

        if joint_reset:
            self._controller.reset_joint(self._config.joint_reset_pose).wait()
            time.sleep(0.5)

        # Reset arm
        if self._config.enable_random_reset:
            reset_pose = self._config.reset_position.copy()
            reset_pose[[0, 2]] += np.random.uniform(
                -self._config.random_xy_range, self._config.random_xy_range, (2,)
            )
            # euler_random = self._config.target_position[3:].copy()
            # euler_random[-1] += np.random.uniform(
            #     -self._config.random_rz_range, self._config.random_rz_range
            # )
            # reset_pose[3:] = euler_2_quat(euler_random)
            self._interpolate_move(reset_pose)
        else:
            reset_pose = self._config.reset_position.copy()
            self._interpolate_move(reset_pose)

    def step(self, action):
        """
        - action: [6, ]; append 1 to [7, ]
        """
        new_action = np.ones(7)
        new_action[:6] = action
        return super().step(new_action)
    

    def _get_observation(self):
        if not self.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "arm_position": self._franka_state.arm_position, 
                "arm_velocity": self._franka_state.arm_velocity, 
                "arm_force": self._franka_state.arm_force, 
                "arm_torque": self._franka_state.arm_torque
            }
            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            return self.observation_space.sample()
