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

"""Synchronous Ray worker for SIMPLE's Isaac Sim runtime."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


def _mask_psi0_execution_record(
    builder: Any, executed_mask: torch.Tensor
) -> None:
    """Exclude unexecuted SIMPLE actions from Psi0 policy statistics."""
    if not builder.prev_logprobs or not builder.forward_inputs:
        raise RuntimeError("Psi0 SIMPLE execution has no pending policy statistic.")
    executed_mask = executed_mask.to(dtype=torch.bool, device="cpu")
    prev_logprobs = builder.prev_logprobs[-1]
    if prev_logprobs.shape[:2] != executed_mask.shape:
        raise ValueError("Psi0 SIMPLE executed_mask does not match policy logprobs.")
    prev_logprobs.mul_(executed_mask[..., None])
    forward_inputs = builder.forward_inputs[-1]
    if "psi0_execution_mask" not in forward_inputs:
        raise RuntimeError("Psi0 SIMPLE policy statistic has no execution mask.")
    forward_inputs["psi0_execution_mask"] = executed_mask


class SimpleEnvWorker(EnvWorker):
    """Keep SIMPLE environment startup and interaction on Ray's main thread."""

    def env_interact_step(self, chunk_actions: torch.Tensor, stage_id: int) -> Any:
        """Execute one chunk and apply SIMPLE's execution mask locally."""
        result = EnvWorker.env_interact_step(self, chunk_actions, stage_id)
        env_output = result[0]
        if self.model_cfg.model_type == "psi0":
            executed_mask = (
                env_output.env_infos.get("executed_mask")
                if env_output.env_infos is not None
                else None
            )
            if executed_mask is None:
                raise RuntimeError("Psi0 SIMPLE training requires executed_mask.")
            _mask_psi0_execution_record(
                self.trajectory_builders[stage_id], executed_mask
            )
        return result

    @staticmethod
    def _finish_inline(coroutine: Any) -> Any:
        """Run a coroutine that contains no suspending operation."""
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        coroutine.close()
        raise RuntimeError("SIMPLE synchronous interaction cannot suspend.")

    def _maybe_wait_env_delay(self, stage_id: int) -> Any:
        return EnvWorker._maybe_wait_env_delay(self, stage_id)

    def send_rollout_trajectories(
        self, trajectory_builder: Any, channel: Channel
    ) -> Any:
        return EnvWorker.send_rollout_trajectories(self, trajectory_builder, channel)

    def send_lerobot_episodes(
        self, episodes: list[list[dict]], channel: Channel
    ) -> Any:
        return EnvWorker.send_lerobot_episodes(self, episodes, channel)

    def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
        *,
        cooperative_yield: bool,
    ) -> Any:
        return EnvWorker._run_interact_once(
            self,
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=cooperative_yield,
        )

    def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ) -> dict[str, Any]:
        return self._finish_inline(
            EnvWorker.interact(
                self,
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
            )
        )

    def send_rollout_trajectories_pipeline(
        self, trajectory_builders: Any, actor_channel: Channel
    ) -> Any:
        return EnvWorker.send_rollout_trajectories_pipeline(
            self, trajectory_builders, actor_channel
        )
