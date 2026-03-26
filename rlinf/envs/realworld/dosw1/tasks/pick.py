"""Single-arm pick task for the DOSW1 robot.

Three-phase task (reach -> grasp -> lift) using left arm only,
modelled after Franka's ``PegInsertionEnv`` but in joint space.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from rlinf.envs.realworld.dosw1.dosw1_env import DOSW1Config, DOSW1Env, ControlMode


def _default_grasp_joint() -> np.ndarray:
    return np.array([-0.75, 0.0, 0.0, 1.57, 0.0, -1.57], dtype=np.float64)


def _default_lift_joint() -> np.ndarray:
    return np.array([-0.75, -0.15, 0.0, 1.57, 0.0, -1.57], dtype=np.float64)


@dataclass
class PickConfig(DOSW1Config):
    """Configuration for the DOSW1 single-arm pick task."""

    target_grasp_joint: np.ndarray = field(default_factory=_default_grasp_joint)
    target_lift_joint: np.ndarray = field(default_factory=_default_lift_joint)

    joint_reward_sharpness: float = 2.0
    grasp_threshold: float = 0.25
    lift_threshold: float = 0.25
    gripper_closed_max_width: float = 0.01
    grasp_bonus: float = 0.3

    max_joint_delta: float = 0.1

    enable_gripper_penalty: bool = False
    gripper_penalty: float = 0.05
    use_dense_reward: bool = True
    step_frequency: float = 10.0


class PickEnv(DOSW1Env):
    """Reach a grasp joint pose, close gripper, then lift to a target pose."""

    def __init__(
        self,
        override_cfg: dict,
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
    ) -> None:
        super().__init__(
            PickConfig(**override_cfg),
            worker_info,
            hardware_info,
            env_idx,
        )
        self._phase = "reach"
        self._holding_object = False
        self._task_success = False

    @property
    def task_description(self) -> str:
        return "Pick up the object with the left arm."

    def reset(
        self,
        *,
        seed=None,
        options=None,
        joint_reset: bool = False,
    ) -> tuple[dict, dict]:
        obs, info = super().reset(seed=seed, options=options, joint_reset=joint_reset)

        self._phase = "reach"
        self._holding_object = False
        self._task_success = False
        return self._get_observation(), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self._task_success or self._manual_done:
            terminated = True
        return obs, reward, terminated, truncated, info

    def _calc_step_reward(self, obs: dict, gripper_changed: bool = False) -> float:
        del obs
        if self.config.is_dummy:
            return 0.0

        cfg: PickConfig = self.config
        left_joint = self._robot_state.left_joint_positions
        left_gripper = self._robot_state.left_gripper
        grasp_joint = np.asarray(cfg.target_grasp_joint, dtype=np.float64).reshape(6)
        lift_joint = np.asarray(cfg.target_lift_joint, dtype=np.float64).reshape(6)
        sharpness = float(cfg.joint_reward_sharpness)

        if self._control_mode == ControlMode.TELEOP and self._teleop_target_left_gripper is not None:
            gripper_closed = self._teleop_target_left_gripper <= cfg.gripper_closed_max_width
        else:
            gripper_closed = left_gripper <= cfg.gripper_closed_max_width

        reward = 0.0

        if self._phase == "reach":
            dist_sq = float(np.sum(np.square(left_joint - grasp_joint)))
            if cfg.use_dense_reward:
                reward = float(np.exp(-sharpness * dist_sq))

            if gripper_closed and gripper_changed:
                self._holding_object = True
                self._phase = "lift"
                reward += cfg.grasp_bonus

        elif self._phase == "lift":
            dist_sq = float(np.sum(np.square(left_joint - lift_joint)))
            distance = float(np.sqrt(dist_sq))
            if cfg.use_dense_reward:
                reward = float(np.exp(-sharpness * dist_sq))

            if distance <= cfg.lift_threshold:
                reward = 1.0
                self._task_success = True

        if cfg.enable_gripper_penalty and gripper_changed:
            reward -= cfg.gripper_penalty

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
