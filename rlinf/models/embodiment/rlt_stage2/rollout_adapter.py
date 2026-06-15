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

"""Rollout adapter for RLT Stage 2 TD3."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.config import SupportedModel

from .rollout_router import RLTStage2RolloutRouteConfig, route_rlt_stage2_rollout
from .training_schedule import resolve_warmup_required_updates


class RLTStage2RolloutAdapter:
    """Encapsulates RLT Stage 2 rollout-only behavior."""

    def __init__(
        self,
        *,
        cfg,
        student_model: Any,
        expert_model_getter,
        has_expert_model_config: bool,
    ) -> None:
        self.cfg = cfg
        self.student_model = student_model
        self.expert_model_getter = expert_model_getter
        self.has_expert_model_config = bool(has_expert_model_config)
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        self.intervention_mode = str(
            intervention_cfg.get("mode", "local_correction")
        )
        if (
            self.enabled()
            and bool(intervention_cfg.get("enable", False))
            and self.intervention_mode != "local_correction"
        ):
            raise ValueError(
                "RLT Stage2 ManiSkill rollout only supports "
                "algorithm.intervention.mode='local_correction', got "
                f"{self.intervention_mode!r}."
            )
        self.intervention_enabled = bool(intervention_cfg.get("enable", False)) and (
            self.intervention_mode == "local_correction"
        )

    def enabled(self) -> bool:
        return (
            self.cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.RLT_STAGE2
        )

    def ready_for_online(self, update_version: int) -> tuple[bool, int]:
        warmup_required_updates = resolve_warmup_required_updates(self.cfg)
        return update_version >= warmup_required_updates, warmup_required_updates

    def predict(
        self,
        *,
        env_obs: dict[str, Any],
        policy_info: dict[str, torch.Tensor] | None,
        model_kwargs: dict[str, Any],
        mode: str,
        allow_expert: bool,
        update_version: int,
    ):
        ready_for_online, online_gate_step = self.ready_for_online(update_version)
        return route_rlt_stage2_rollout(
            env_obs=env_obs,
            policy_info=policy_info,
            student_model=self.student_model,
            expert_model_getter=self.expert_model_getter,
            model_kwargs=model_kwargs,
            cfg=RLTStage2RolloutRouteConfig(
                ready_for_online=ready_for_online,
                online_gate_step=online_gate_step,
                intervention_enabled=self.intervention_enabled,
                allow_expert=(
                    mode == "train" and allow_expert and self.has_expert_model_config
                ),
                chunk_length=self.cfg.actor.model.num_action_chunks,
                action_dim=self.cfg.actor.model.action_dim,
            ),
        )

    def rollout_state_dict(self):
        return self.student_model.rollout_state_dict()

    def encode_step_trace(
        self,
        step_obs: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if step_obs is None or not self.enabled():
            return {}
        if not hasattr(self.student_model, "encode_obs"):
            raise RuntimeError(
                "RLT Stage2 stride replay requires hf_model.encode_obs for step features."
            )
        first_tensor = next(
            (
                value
                for key, value in step_obs.items()
                if not key.startswith("_rlt_")
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        explicit_offsets = step_obs.get("_rlt_step_offsets", None)
        if explicit_offsets is not None:
            if (
                not isinstance(explicit_offsets, torch.Tensor)
                or explicit_offsets.dim() != 2
            ):
                raise ValueError(
                    "RLT step_obs['_rlt_step_offsets'] must have shape [A, B], "
                    f"got {type(explicit_offsets)=}."
                )
            anchor_offset_tensor = explicit_offsets.to(torch.long).contiguous()
        else:
            if first_tensor is None:
                raise ValueError("RLT step_obs must contain at least one tensor field.")
            step_count = int(first_tensor.shape[0])
            batch_size = int(first_tensor.shape[1])
            anchor_offsets = self._sparse_anchor_offsets(step_count)
            anchor_offset_tensor = (
                torch.tensor(anchor_offsets, dtype=torch.long)[:, None]
                .expand(len(anchor_offsets), batch_size)
                .contiguous()
            )
            if anchor_offsets:
                step_obs = self._slice_step_obs_offsets(step_obs, anchor_offsets)

        if anchor_offset_tensor.shape[0] == 0:
            return {"anchor_offsets": anchor_offset_tensor}

        if first_tensor is None:
            raise ValueError("RLT step_obs must contain obs tensors for sparse anchors.")
        flat_obs, step_count, batch_size = self._flatten_step_obs(step_obs)
        total = step_count * batch_size
        micro_batch_size = int(
            self.cfg.actor.model.rlt_stage2.get("replay_feature_batch_size", 32)
        )
        if micro_batch_size <= 0:
            micro_batch_size = total
        encoded_x = []
        encoded_a_tilde = []
        with torch.no_grad():
            for begin in range(0, total, micro_batch_size):
                end = min(begin + micro_batch_size, total)
                obs_chunk = self._slice_flat_obs(flat_obs, begin, end)
                x, a_tilde = self.student_model.encode_obs(obs_chunk)
                encoded_x.append(x.detach().cpu())
                encoded_a_tilde.append(a_tilde.detach().cpu())
        x_all = torch.cat(encoded_x, dim=0).reshape(step_count, batch_size, -1)
        a_tilde_all = torch.cat(encoded_a_tilde, dim=0).reshape(
            step_count,
            batch_size,
            -1,
        )
        return {
            "anchor_offsets": anchor_offset_tensor,
            "x": x_all.contiguous(),
            "a_tilde": a_tilde_all.contiguous(),
        }

    def _sparse_anchor_offsets(self, step_count: int) -> list[int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        if stride <= 0 or chunk_len <= 0:
            return []

        offsets = set()
        offset = 0
        while True:
            offset = (offset + stride) % chunk_len
            if offset == 0 or offset in offsets:
                break
            if offset < step_count:
                offsets.add(offset)
        return sorted(offsets)

    @staticmethod
    def _flatten_step_obs(
        step_obs: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int]:
        first_tensor = next(
            (
                value
                for key, value in step_obs.items()
                if not key.startswith("_rlt_")
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        if first_tensor is None:
            raise ValueError("RLT step_obs must contain at least one tensor field.")
        step_count = int(first_tensor.shape[0])
        batch_size = int(first_tensor.shape[1])
        flat_obs: dict[str, Any] = {}
        for key, value in step_obs.items():
            if key.startswith("_rlt_"):
                continue
            if isinstance(value, torch.Tensor):
                flat_obs[key] = value.reshape(step_count * batch_size, *value.shape[2:])
            elif isinstance(value, list):
                flat_obs[key] = [item for step_values in value for item in step_values]
            elif value is None:
                flat_obs[key] = None
            else:
                flat_obs[key] = value
        return flat_obs, step_count, batch_size

    @staticmethod
    def _slice_step_obs_offsets(
        step_obs: dict[str, Any],
        offsets: list[int],
    ) -> dict[str, Any]:
        sliced_obs: dict[str, Any] = {}
        index_tensor = torch.as_tensor(offsets, dtype=torch.long)
        for key, value in step_obs.items():
            if key.startswith("_rlt_"):
                continue
            if isinstance(value, torch.Tensor):
                sliced_obs[key] = value.index_select(
                    0,
                    index_tensor.to(device=value.device),
                )
            elif isinstance(value, list):
                sliced_obs[key] = [value[offset] for offset in offsets]
            elif value is None:
                sliced_obs[key] = None
            else:
                sliced_obs[key] = value
        return sliced_obs

    @staticmethod
    def _slice_flat_obs(
        flat_obs: dict[str, Any],
        begin: int,
        end: int,
    ) -> dict[str, Any]:
        obs_chunk: dict[str, Any] = {}
        for key, value in flat_obs.items():
            if isinstance(value, torch.Tensor):
                obs_chunk[key] = value[begin:end]
            elif isinstance(value, list):
                obs_chunk[key] = value[begin:end]
            else:
                obs_chunk[key] = value
        return obs_chunk
