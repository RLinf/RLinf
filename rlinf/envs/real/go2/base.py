# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Go2 high-level drive environment, without a training reward or cameras."""

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.robotics import Go2Config, Go2Robot, RobotInfo
from rlinf.robotics.parts.mobility.unitree_go2 import STATE_SHAPES
from rlinf.scheduler import WorkerInfo


@dataclass
class Go2EnvConfig:
    """Drive timing and episode bounds; motion opt-in belongs to hardware."""

    is_dummy: bool = False
    step_frequency: float = 10.0
    max_num_steps: int = 20


class Go2Env(gym.Env):
    """Drive through SportClient; watchdog covers gaps between steps.

    Successful nonterminal steps leave the command active until the next
    action or watchdog expiry. Reset, close, truncation, and errors request
    stop. Reward is zero; this is a bring-up task, not a training recipe.
    """

    TELEOP: tuple[str, ...] = ()
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: Go2EnvConfig,
        worker_info: WorkerInfo | None = None,
        robot_info: RobotInfo[Go2Config] | None = None,
        env_idx: int = 0,
    ) -> None:
        if not np.isfinite(config.step_frequency) or config.step_frequency <= 0:
            raise ValueError("step_frequency must be finite and positive.")
        if not isinstance(config.max_num_steps, int) or config.max_num_steps <= 0:
            raise ValueError("max_num_steps must be a positive integer.")
        self.config = config
        self.robot_info = robot_info
        self.hardware = get_hardware_config(
            Go2Config, robot_info, is_dummy=config.is_dummy
        )
        if 1 / config.step_frequency >= self.hardware.command_timeout:
            raise ValueError("Control period must be shorter than command_timeout.")
        self.robot: Go2Robot | None = None
        self._num_steps = 0
        self._needs_reset = True
        self._closed = False
        limit = np.asarray([0.3, 0.3, 0.5], dtype=np.float32)
        self.action_space = gym.spaces.Box(-limit, limit)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            -np.inf, np.inf, shape=shape, dtype=np.float32
                        )
                        for name, shape in STATE_SHAPES.items()
                    }
                )
            }
        )
        if not config.is_dummy:
            node_rank = self.hardware.node_rank
            if worker_info is None:
                node_rank = None
            self.robot = Go2Robot.build(
                interface=self.hardware.interface,
                node_rank=node_rank,
                worker_rank=worker_info.rank if worker_info else 0,
                env_idx=env_idx,
                enable_motion=self.hardware.enable_motion,
                command_timeout=self.hardware.command_timeout,
                state_timeout=self.hardware.state_timeout,
                rpc_timeout=self.hardware.rpc_timeout,
            )
            try:
                self.robot.connect()
            except BaseException:
                self.robot.disconnect()
                raise

    @property
    def num_steps(self) -> int:
        """Return the number of completed steps in the current episode."""
        return self._num_steps

    def _observation(self) -> dict[str, Any]:
        if self.config.is_dummy:
            return {
                "state": {
                    name: np.zeros(shape, dtype=np.float32)
                    for name, shape in STATE_SHAPES.items()
                }
            }
        return {"state": self.robot.get_observation()["base"]}

    def _stop(self) -> None:
        if self.robot is not None:
            self.robot.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Request stop and read state; never stand up or recover automatically."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed Go2 environment.")
        super().reset(seed=seed)
        self._needs_reset = True
        self._stop()
        observation = self._observation()
        self._num_steps = 0
        self._needs_reset = False
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Send one velocity action, observe state, and stop on episode expiry."""
        try:
            if self._closed or self._needs_reset:
                raise RuntimeError("Reset an open Go2 environment before stepping.")
            velocity = np.asarray(action, dtype=np.float32)
            if not self.action_space.contains(velocity):
                raise ValueError(
                    "Go2 action must be finite (3,) and within its velocity space."
                )
            started = time.monotonic()
            if self.robot is not None:
                self.robot.send_action({"base": {"velocity": velocity}})
            time.sleep(
                max(0, 1 / self.config.step_frequency - (time.monotonic() - started))
            )
            observation = self._observation()
            self._num_steps += 1
            truncated = self._num_steps >= self.config.max_num_steps
            if truncated:
                self._needs_reset = True
                self._stop()
            return observation, 0.0, False, truncated, {"requires_reset": truncated}
        except BaseException:
            self._needs_reset = True
            self._stop()
            raise

    def close(self) -> None:
        """Request stop and release the robot; repeated calls are safe."""
        self._closed = True
        self._needs_reset = True
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None
