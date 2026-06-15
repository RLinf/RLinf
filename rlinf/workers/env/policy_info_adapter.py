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

from typing import Any


class NoopPolicyInfoAdapter:
    """Default adapter used when an algorithm does not emit policy_info."""

    def init_stage(self, **kwargs: Any):
        return None

    def update_stage(self, **kwargs: Any):
        return None

    def update_last_action_metadata(self, **kwargs: Any) -> None:
        return None


def build_policy_info_adapter(cfg, train_batch_size, eval_batch_size):
    """Build an env-side policy_info adapter for algorithms that need one."""
    intervention_cfg = cfg.algorithm.get("intervention", {})
    is_rlt_stage2 = (
        cfg.algorithm.get("loss_type", None) == "rlt_td3"
        and cfg.actor.model.get("model_type", None) == "rlt_stage2"
    )
    train_env_type = str(cfg.env.get("train", {}).get("env_type", "")).lower()
    eval_env_type = str(cfg.env.eval.get("env_type", "")).lower()
    is_maniskill = train_env_type == "maniskill" or eval_env_type == "maniskill"
    if (
        not is_rlt_stage2
        or not bool(intervention_cfg.get("enable", False))
        or str(intervention_cfg.get("mode", "local_correction")) != "local_correction"
        or not is_maniskill
    ):
        return NoopPolicyInfoAdapter()

    from rlinf.envs.maniskill.rlt_policy_info import RLTStage2PolicyInfoAdapter

    return RLTStage2PolicyInfoAdapter(
        cfg=cfg,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
