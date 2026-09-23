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

from dataclasses import replace
from typing import Any

import torch

from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class SimpleEnvWorker(EnvWorker):
    """Keep SIMPLE environment startup and interaction on Ray's main thread.

    Synchronous wrappers hide inherited async methods from Ray's actor detection.
    They return the original coroutines for the inline interaction loop to await.
    """

    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """Attach SIMPLE's actual execution mask to the environment transition."""
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
            env_output.transition = replace(
                env_output.transition, executed_mask=executed_mask
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

    def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> Any:
        return EnvWorker._run_interact_once(
            self,
            input_channel,
            rollout_channel,
            reward_channel,
            cooperative_yield=cooperative_yield,
        )

    def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
    ) -> dict[str, Any]:
        return self._finish_inline(
            EnvWorker.interact(
                self,
                input_channel,
                rollout_channel,
                reward_channel,
            )
        )
