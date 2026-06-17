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

"""Policy-info adapter factory for env workers."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult, RolloutResult


class NoopPolicyInfoAdapter:
    """Default adapter used when an algorithm does not emit policy_info."""

    def init_stage(self, **kwargs: Any):
        return None

    def update_stage(self, **kwargs: Any):
        return None

    def update_last_action_metadata(self, **kwargs: Any) -> None:
        return None

    def build_step_obs(self, **kwargs: Any):
        return None

    def append_step_trace(
        self,
        *,
        rollout_accumulator: EmbodiedRolloutResult,
        rollout_result: RolloutResult,
    ) -> None:
        return None

    def final_forward_inputs(self, rollout_result: RolloutResult) -> dict[str, Any]:
        return {}

    def collect_rollout_metrics(
        self,
        *,
        env_metrics: dict[str, list],
        rollout_result: RolloutResult,
    ) -> None:
        return None

    def emit_status(
        self,
        *,
        env_metrics: dict[str, torch.Tensor],
        rank: int,
        last_logged_phase: str | None,
        log_info,
    ) -> str | None:
        return last_logged_phase


def build_policy_info_adapter(cfg, train_batch_size, eval_batch_size):
    """Build an env-side policy_info adapter for the configured model."""
    model_type = str(cfg.actor.model.get("model_type", ""))
    if not model_type:
        return NoopPolicyInfoAdapter()

    module_name = f"rlinf.models.embodiment.{model_type}.env_policy_info"
    try:
        adapter_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return NoopPolicyInfoAdapter()
        raise

    adapter_builder = getattr(adapter_module, "build_policy_info_adapter", None)
    if adapter_builder is None:
        return NoopPolicyInfoAdapter()

    adapter = adapter_builder(
        cfg=cfg,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    if adapter is None:
        return NoopPolicyInfoAdapter()
    return adapter
