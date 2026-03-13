# Copyright 2025 The RLinf Authors.
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

import logging
from typing import Any

from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


class AndroidReward:
    """Android World reward calculator."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize reward configuration from Hydra config."""
        self.scale: float = float(config.get("reward_scale", 1.0))
        self.device_id: str = str(config.get("device_id", "localhost:5555"))
        self.grpc_port: int = int(config.get("grpc_port", 8554))
        self.adb_path: str = str(config.get("adb_path", "adb"))

        if ":" in self.device_id:
            self.console_port: int = int(self.device_id.split(":")[1]) - 1
        else:
            self.console_port = int(self.device_id.split("-")[1]) - 1
        self._env: Any | None = None

    def get_env(self) -> Any:
        """Lazily create or return the cached Android env instance."""
        if self._env is None:
            # Delay import so that android_world path can be configured externally.
            from android_world.env import env_launcher  # type: ignore[import]

            self._env = env_launcher.load_and_setup_env(
                console_port=self.console_port,
                emulator_setup=False,
                freeze_datetime=True,
                adb_path=self.adb_path,
                grpc_port=self.grpc_port,
                device_id=self.device_id,
            )
        return self._env

    def get_reward_new(self, env: Any, result: Any, task: Any) -> float:
        """Compute reward for a finished task.

        Args:
            env: Android World environment instance.
            result: Agent execution result, expected to have a ``done`` attribute.
            task: Android World task instance, expected to implement ``is_successful``.

        Returns:
            Scaled reward in ``[0, scale]``. Returns ``0.0`` on failure or exception.
        """
        if not getattr(result, "done", False):
            return 0.0

        if not getattr(task, "initialized", False):
            # Some Android World tasks expect ``initialized`` to be set.
            task.initialized = True  # type: ignore[assignment]

        try:
            score = task.is_successful(env)
            return float(score) * self.scale
        except Exception:  # pylint: disable=broad-except
            # Reward evaluation should not crash the whole rollout.
            # Common failures include clipboard read/write constraints
            # (e.g., Clipper not foreground / permissions).
            LOGGER.exception(
                "AndroidReward.get_reward_new failed during task.is_successful; "
                "returning 0. task_name=%s params=%s class_name=%s",
                getattr(task, "task_name", None),
                getattr(task, "params", None),
                getattr(task, "class_name", None),
            )
            return 0.0
