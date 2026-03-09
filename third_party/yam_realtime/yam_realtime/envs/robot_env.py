import time
from typing import Any, Dict, Optional, Union

import dm_env
from i2rt.robots.robot import RobotType

from yam_realtime.robots.robot import Robot
from yam_realtime.robots.utils import Rate
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.utils.portal_utils import return_futures


class RobotEnv(dm_env.Environment):
    # Abstract methods.
    """A environment with a dm_env.Environment interface for a robot arm setup."""

    def __init__(
        self,
        robot_dict: Dict[str, Robot],
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        control_rate_hz: Union[Rate, float] = 100.0,
        use_joint_state_as_action: bool = False,
    ) -> None:
        self._robot_dict = robot_dict
        if isinstance(control_rate_hz, Rate):
            self._rate = control_rate_hz
        else:
            self._rate = Rate(control_rate_hz)

        self._use_joint_state_as_action = use_joint_state_as_action
        # get camera dict
        self._camera_dict = camera_dict

    def robot(self, name: str) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot_dict[name]

    def get_all_robots(self) -> Dict[str, Robot]:
        return self._robot_dict

    def __len__(self) -> int:
        return 0

    def _apply_action(self, action_dict: Dict[str, Any]) -> None:
        with return_futures(*self._robot_dict.values()):  # type: ignore
            for name, action in action_dict.items():
                if name == "base":
                    self._robot_dict[name].command_target_vel(action)
                elif self._use_joint_state_as_action:
                    self._robot_dict[name].command_joint_state(action)
                else:
                    self._robot_dict[name].command_joint_pos(action["pos"])

    def step(self, action: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore
        """Step the environment forward.

        Args:
            action: action to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        if len(action) != 0:
            # get action at time t
            self._apply_action(action)
        self._rate.sleep()  # sleep until next timestep
        # return observation at time t+1
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        observations["timestamp"] = time.time()

        assert self._camera_dict is not None, "Camera dictionary is not set."
        clients = list(self._camera_dict.values()) + list(self._robot_dict.values())

        camera_futures = {}
        robot_futures = {}
        with return_futures(*clients):  # type: ignore
            for name, client in self._camera_dict.items():
                camera_data = client.read()
                camera_futures[name] = camera_data
            for name, robot in self._robot_dict.items():
                robot_obs = robot.get_observations()
                robot_futures[name] = robot_obs

        for name, robot_obs_future in robot_futures.items():
            robot_obs = robot_obs_future.result()
            observations[name] = robot_obs

        for name, camera_data_future in camera_futures.items():
            camera_data = camera_data_future.result()
            assert name not in observations
            observations[name] = camera_data
        observations["timestamp_end"] = time.time()

        return observations

    def reset(self) -> Dict[str, Any]:  # type: ignore
        return self.get_obs()

    def observation_spec(self):  # type: ignore
        return {}

    def action_spec(self):  # type: ignore
        spec = {}
        for name, robot in self._robot_dict.items():
            if robot.get_robot_type() == RobotType.MOBILE_BASE:
                spec[name] = robot.joint_state_spec()
            else:
                spec[name] = (
                    robot.joint_state_spec() if self._use_joint_state_as_action else {"pos": robot.joint_pos_spec()}
                )
        return spec

    def close(self) -> None:
        if self._camera_dict is not None:
            for camera_name, client in self._camera_dict.items():
                client.close()  # type: ignore
