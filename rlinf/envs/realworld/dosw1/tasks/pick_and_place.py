"""Pick-and-place example task for the DOSW1 robot."""

import time
from dataclasses import dataclass, field

import numpy as np

from rlinf.envs.realworld.dosw1.dosw1_env import DOSW1Config, DOSW1Env


def _default_grasp_joint() -> np.ndarray:
    return np.array([-0.75, 0.0, 0.0, 1.57, 0.0, -1.57], dtype=np.float64)


def _default_place_joint() -> np.ndarray:
    return np.array([-0.65, 0.12, 0.0, 1.35, 0.0, -1.35], dtype=np.float64)


@dataclass
class PickAndPlaceConfig(DOSW1Config):
    """Configuration for the DOSW1 pick-and-place task."""

    target_grasp_left_joint: np.ndarray = field(default_factory=_default_grasp_joint)
    target_place_left_joint: np.ndarray = field(default_factory=_default_place_joint)
    joint_reward_sharpness: float = 2.0
    joint_success_threshold: float = 0.25
    gripper_closed_max_width: float = 0.01
    enable_gripper_penalty: bool = False
    gripper_penalty: float = 0.05
    use_dense_reward: bool = True
    step_frequency: float = 10.0


class PickAndPlaceEnv(DOSW1Env):
    """Reach a grasp pose, close gripper, then move to the place pose."""

    def __init__(
        self,
        override_cfg: dict,
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
    ) -> None:
        super().__init__(
            PickAndPlaceConfig(**override_cfg),
            worker_info,
            hardware_info,
            env_idx,
        )
        self._holding_object = False

    @property
    def task_description(self) -> str:
        return "Pick up the object and place it at the target location."

    def reset(
        self,
        *,
        seed=None,
        options=None,
        joint_reset: bool = False,
    ) -> tuple[dict, dict]:
        obs, info = super().reset(seed=seed, options=options, joint_reset=joint_reset)

        if not self.config.is_dummy:
            self._sdk.open_gripper()
            time.sleep(0.4)
            self._robot_state = self._sdk.get_state()

        self._holding_object = False
        return self._get_observation(), info

    def _calc_step_reward(self, obs: dict, gripper_changed: bool = False) -> float:
        del obs
        if self.config.is_dummy:
            return 0.0

        config = self.config
        left_joint = self._robot_state.left_joint_positions
        left_gripper = self._robot_state.left_gripper
        grasp_joint = np.asarray(config.target_grasp_left_joint, dtype=np.float64).reshape(
            6
        )
        place_joint = np.asarray(config.target_place_left_joint, dtype=np.float64).reshape(
            6
        )
        gripper_closed = left_gripper <= config.gripper_closed_max_width
        sharpness = float(config.joint_reward_sharpness)

        reward = 0.0
        if not gripper_closed:
            distance = float(np.linalg.norm(left_joint - grasp_joint))
            reward += float(np.exp(-sharpness * distance))
        else:
            if gripper_changed:
                self._holding_object = True
                reward += 0.3

            if self._holding_object:
                distance_place = float(np.linalg.norm(left_joint - place_joint))
                reward += float(np.exp(-sharpness * distance_place))
                if distance_place <= config.joint_success_threshold:
                    reward = 1.0

        if config.enable_gripper_penalty and gripper_changed:
            reward -= config.gripper_penalty

        return float(np.clip(reward, -1.0, 1.0))

    def go_to_rest(self) -> None:
        """Open gripper if needed and return to the home joint state."""
        if self.config.is_dummy:
            return

        if self._holding_object:
            self._sdk.open_gripper()
            time.sleep(0.4)
            self._holding_object = False

        self._go_to_home()
