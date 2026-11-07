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

"""Record online LeRobot episodes on the env worker and upload them without waiting."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.utils.logging import get_logger


class EnvEpisodeRecorder:
    """Record an env worker's own episodes and upload the finished ones.

    Recording and uploading never block the control loop. Finished episodes wait
    in an outbox, at most one asynchronous put is in flight, and the oldest
    unsent episode is dropped once more than ``max_unsent`` wait.
    """

    def __init__(self, accumulators: list, max_unsent: int):
        self._accumulators = accumulators
        self._max_unsent = max_unsent
        self._outbox: deque[list[dict[str, Any]]] = deque()
        self._inflight = None
        self._channel: Channel | None = None
        self._logger = get_logger()
        self.dropped = 0

    @classmethod
    def from_cfg(
        cls, cfg: DictConfig, *, stage_num: int, num_envs_per_stage: int
    ) -> EnvEpisodeRecorder:
        """Build one accumulator per stage from ``algorithm.dagger.online_lerobot``."""
        # Imported here so other workers do not load the trajectory module.
        from rlinf.data.schema.embodied_trajectory import LeRobotEpisodeAccumulator

        online = cfg.algorithm.dagger.online_lerobot
        accumulators = [
            LeRobotEpisodeAccumulator(
                num_envs=num_envs_per_stage,
                only_success=bool(online.get("only_success", False)),
                num_action_chunks=int(cfg.actor.model.num_action_chunks),
                action_dim=int(cfg.actor.model.action_dim),
            )
            for _ in range(stage_num)
        ]
        return cls(accumulators, int(online.get("max_unsent_episodes", 4)))

    def prepare(self, channel: Channel) -> None:
        """Open the sender path once, before the first chunk."""
        # The first send to a new peer sets up its connection on the calling
        # thread, so it must not happen in the middle of a collection.
        if self._channel is None:
            self._channel = channel
            channel.put([[]], async_op=True)

    def record(self, stage_id: int, chunk_data: dict[str, Any]) -> None:
        """Append one executed chunk and hand any finished episode to the uploader."""
        accumulator = self._accumulators[stage_id]
        accumulator.append_chunk_episode_data(policy_output=None, **chunk_data)
        self._outbox.extend(accumulator.drain_episodes())
        while len(self._outbox) > self._max_unsent:
            self._outbox.popleft()
            self.dropped += 1
            self._logger.error(
                "Dropped the oldest unsent episode: more than %d wait for upload "
                "(%d dropped so far).",
                self._max_unsent,
                self.dropped,
            )
        self._pump()

    def _pump(self) -> None:
        if self._inflight is not None and not self._inflight.done():
            return
        self._inflight = None
        if self._outbox:
            self._inflight = self._channel.put([self._outbox.popleft()], async_op=True)

    def wait_for_uploads(self, timeout_s: float = 30.0) -> None:
        """Block until queued episodes are handed to the channel; for shutdown only."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._pump()
            if self._inflight is None:
                return
            time.sleep(0.05)
        self._logger.error(
            "%d unsent episodes were lost at shutdown.", len(self._outbox)
        )
