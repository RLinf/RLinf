from typing import Dict, List, Optional

import numpy as np
from i2rt.robots.robot import Robot
from i2rt.robots.utils import JointMapper

# RPC Method Serialization Requirements.
ROBOT_PROTOCOL_METHODS = {
    "num_dofs": False,
    "get_joint_pos": False,
    "get_joint_state": False,
    "command_joint_pos": False,
    "command_joint_state": False,
    "get_observations": False,
    "joint_pos_spec": True,
    "joint_state_spec": True,
    "get_robot_info": True,
    "get_robot_type": True,
    "command_target_vel": False,
}


class PrintRobot(Robot):
    """A robot that prints the commanded joint state."""

    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        self._joint_state = np.zeros((num_dofs,))
        self._dont_print = dont_print

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        return self._joint_state

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert len(joint_pos) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, got {len(joint_pos)}."
        )
        self._joint_state = joint_pos
        if not self._dont_print:
            print(self._joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_pos = self.get_joint_pos()
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_pos,
        }


class ConcatenatedRobot(Robot):
    def __init__(self, robots: List[Robot], remapper: Optional[JointMapper] = None):
        self._robots = robots
        self._remapper = remapper
        self.per_robot_index = np.array([i.num_dofs() for i in self._robots]).cumsum()

    def num_dofs(self) -> int:
        return sum(robot.num_dofs() for robot in self._robots)

    def get_joint_pos(self) -> np.ndarray:
        robot_space_joint_pos = np.concatenate([robot.get_joint_pos() for robot in self._robots])
        if self._remapper is not None:
            return self._remapper.to_command_joint_pos_space(robot_space_joint_pos)
        return robot_space_joint_pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if self._remapper is not None:
            joint_pos = self._remapper.to_robot_joint_pos_space(joint_pos)
        for robot, pos in zip(self._robots, np.split(joint_pos, self.per_robot_index), strict=False):
            robot.command_joint_pos(pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        assert self._remapper is None, "Remapper is not supported for command_joint_state"
        for robot, state in zip(self._robots, np.split(joint_state, self.per_robot_index), strict=False):  # type: ignore
            robot.command_joint_state(state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = [robot.get_observations() for robot in self._robots]
        obs_dict = {}
        for o in obs:
            for k, v in o.items():
                if k in obs_dict:
                    obs_dict[k] = np.concatenate([obs_dict[k], v])
                else:
                    obs_dict[k] = v
        return obs_dict
