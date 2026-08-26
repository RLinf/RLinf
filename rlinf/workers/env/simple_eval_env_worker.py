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

"""Synchronous Ray worker used by the SIMPLE Isaac Sim evaluation path."""

from __future__ import annotations

from typing import Any

from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class SimpleEvalEnvWorker(EnvWorker):
    """Keep SIMPLE evaluation on Ray's worker main thread.

    Isaac Sim installs process-level signal handlers and owns an application
    event loop during startup.  Ray makes an actor asynchronous when any
    exposed method is async, which moves even ``init_worker`` to its asyncio
    thread.  SIMPLE evaluation does not use the asynchronous training methods,
    so replacing them with synchronous guards keeps this actor synchronous.
    """

    def _maybe_wait_env_delay(self, stage_id: int) -> None:
        del stage_id

    def send_rollout_trajectories(
        self, trajectory_builder: Any, channel: Channel
    ) -> None:
        del trajectory_builder, channel
        raise RuntimeError("SimpleEvalEnvWorker does not support training rollout.")

    def send_lerobot_episodes(self, episodes: list[list[dict]], channel: Channel) -> None:
        del episodes, channel
        raise RuntimeError("SimpleEvalEnvWorker does not support training rollout.")

    def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, Any]:
        del (
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield,
        )
        raise RuntimeError("SimpleEvalEnvWorker does not support training rollout.")

    def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ) -> dict[str, Any]:
        del input_channel, rollout_channel, reward_channel, actor_channel
        raise RuntimeError("SimpleEvalEnvWorker does not support training rollout.")

    def send_rollout_trajectories_pipeline(
        self, trajectory_builders: Any, actor_channel: Channel
    ) -> None:
        del trajectory_builders, actor_channel
        raise RuntimeError("SimpleEvalEnvWorker does not support training rollout.")
