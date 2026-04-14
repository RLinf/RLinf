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

from typing import Any

from omegaconf import DictConfig

from rlinf.algorithms.rewards.android import AndroidReward
from rlinf.scheduler import Worker


class AndroidRewardWorker(Worker):
    """Worker that reconnects Android env and computes rewards."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize AndroidRewardWorker with Hydra config."""
        super().__init__()
        self._cfg = cfg
        self._android_reward = AndroidReward(cfg.reward)

    def init_worker(self) -> None:
        """Initialize worker state (no-op for now)."""

    def _receive_env_info(
        self,
        agent_worker_group_name: str = "AndroidAgentWorkerGroup",
    ) -> tuple[Any, Any, Any]:
        """Receive env info and task from AgentWorker, then reconnect env.

        Args:
            agent_worker_group_name: Name of the agent worker group that sends
                env info and task.

        Returns:
            Tuple of (env, task, agent_result).
        """
        self.log_info(
            f"Waiting to receive env info and task from {agent_worker_group_name}[{self._rank}]..."
        )
        env_info_and_task = self.recv(
            src_group_name=agent_worker_group_name,
            src_rank=self._rank,
        )
        env_info = env_info_and_task["env_info"]

        # Delay import so that android_world path can be configured externally.
        from android_world.env import env_launcher  # type: ignore[import]

        env = env_launcher.load_and_setup_env(
            console_port=env_info["console_port"],
            emulator_setup=False,
            freeze_datetime=False,
            adb_path=env_info["adb_path"],
            grpc_port=env_info["grpc_port"],
            device_id=env_info["device_id"],
        )
        self.log_info(f"Reconnected env for device {env_info['device_id']}")
        return env, env_info_and_task["task"], env_info_and_task["agent_result"]

    def compute_reward(
        self,
        agent_worker_group_name: str = "AndroidAgentWorkerGroup",
    ) -> float:
        """Compute reward for the task and send it back to the agent worker.

        Args:
            agent_worker_group_name: Name of the agent worker group to send the
                reward to.

        Returns:
            Scalar reward for the task.
        """
        env, task, agent_result = self._receive_env_info(agent_worker_group_name)
        reward = float(self._android_reward.get_reward_new(env, agent_result, task))

        self.send(
            reward,
            dst_group_name=agent_worker_group_name,
            dst_rank=self._rank,
        )
        self.log_info(
            f"Sent reward {reward} to {agent_worker_group_name}[{self._rank}]"
        )
        return reward
