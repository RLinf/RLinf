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

"""ManiSkill local-correction controller for RLT Stage 2."""

from __future__ import annotations

from typing import Any, Literal

import torch
from omegaconf import DictConfig


class ManiSkillLocalCorrectionController:
    """State machine that requests expert takeover in peg-insertion phase."""

    REQUIRED_INFO_KEYS = (
        "consecutive_grasp_current",
        "prealigned_current",
        "partial_insert_current",
        "success_current",
        "peg_head_goal_yz_dist",
        "peg_body_goal_yz_dist",
        "peg_head_hole_x",
        "peg_head_hole_abs_y",
        "peg_head_hole_abs_z",
    )

    INSERT_PHASE = 2

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        mode: Literal["train", "eval"],
        hole_radii: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.mode = mode
        self.hole_radii = hole_radii
        self.state = self.init_state(self.batch_size, mode=mode, cfg=cfg)

    @classmethod
    def init_state(
        cls,
        batch_size: int,
        *,
        mode: Literal["train", "eval"],
        cfg: DictConfig,
    ) -> dict[str, torch.Tensor]:
        return {
            "intervention_region": torch.zeros(batch_size, dtype=torch.bool),
            "intervention_phase": torch.zeros(batch_size, dtype=torch.int64),
            "expert_takeover": torch.zeros(batch_size, dtype=torch.bool),
            "deviation": torch.zeros(batch_size, dtype=torch.bool),
            "deviation_count": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_left": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_used": torch.zeros(batch_size, dtype=torch.int64),
            "prev_yz_error": torch.full(
                (batch_size,),
                float("nan"),
                dtype=torch.float32,
            ),
            "prev_hole_x": torch.full(
                (batch_size,),
                float("nan"),
                dtype=torch.float32,
            ),
            "in_critical_phase": torch.full(
                (batch_size,),
                True,
                dtype=torch.bool,
            ),
            "record_transition": torch.full(
                (batch_size,),
                True,
                dtype=torch.bool,
            ),
        }

    @staticmethod
    def export_policy_info(
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        policy_info = {
            "expert_takeover": state["expert_takeover"][:, None],
            "deviation": state["deviation"][:, None],
            "deviation_count": state["deviation_count"].to(torch.float32)[:, None],
            "intervention_phase": state["intervention_phase"].to(torch.float32)[
                :, None
            ],
            "takeover_left": state["takeover_left"].to(torch.float32)[:, None],
            "takeover_used": state["takeover_used"].to(torch.float32)[:, None],
        }
        for key in (
            "in_critical_phase",
            "record_transition",
        ):
            if key in state:
                policy_info[key] = state[key].to(torch.bool)[:, None]
        return policy_info

    @classmethod
    def select_info_source(cls, infos: dict[str, Any]) -> dict[str, Any]:
        if all(key in infos for key in cls.REQUIRED_INFO_KEYS):
            return infos
        final_info = infos.get("final_info")
        if isinstance(final_info, dict) and all(
            key in final_info for key in cls.REQUIRED_INFO_KEYS
        ):
            return final_info
        missing = [key for key in cls.REQUIRED_INFO_KEYS if key not in infos]
        raise RuntimeError(
            "RLT intervention control is enabled, but ManiSkill info is missing "
            f"required keys {missing}. This usually means the env wrapper is not "
            "using the aligned peg-insertion info path."
        )

    def update(
        self,
        *,
        infos: dict[str, Any],
        chunk_dones: torch.Tensor,
        intervention_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        infos = self.select_info_source(infos)
        state = self.state
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        device = infos["peg_head_hole_x"].device
        for key, value in state.items():
            state[key] = value.to(device)

        done_any = chunk_dones.any(dim=1).to(device)

        success = infos["success_current"].to(torch.bool)
        grasp = infos["consecutive_grasp_current"].to(torch.bool)
        prealigned = infos["prealigned_current"].to(torch.bool)
        partial_insert = infos["partial_insert_current"].to(torch.bool)
        yz_error = torch.maximum(
            infos["peg_head_goal_yz_dist"].to(torch.float32),
            infos["peg_body_goal_yz_dist"].to(torch.float32),
        )
        hole_x = infos["peg_head_hole_x"].to(torch.float32)
        abs_y = infos["peg_head_hole_abs_y"].to(torch.float32)
        abs_z = infos["peg_head_hole_abs_z"].to(torch.float32)
        hole_radii = self._hole_radii(abs_y, intervention_cfg, device)

        no_phase = torch.zeros_like(state["intervention_phase"])
        insert_phase = torch.full_like(
            state["intervention_phase"],
            self.INSERT_PHASE,
        )
        previous_insert_region = state["intervention_region"] & (
            state["intervention_phase"] == self.INSERT_PHASE
        )

        if intervention_enabled:
            near_hole_x_min = float(intervention_cfg.get("near_hole_x_min", -0.05))
            exit_hole_x_min = float(intervention_cfg.get("exit_hole_x_min", -0.12))
            yz_margin = float(intervention_cfg.get("near_hole_yz_margin", 1.5))
            intervention_yz = (
                (yz_error <= yz_margin * hole_radii)
                & (abs_y <= yz_margin * hole_radii)
                & (abs_z <= yz_margin * hole_radii)
            ) | prealigned | partial_insert
            intervention_near_hole = hole_x >= near_hole_x_min
            insert_entry = (
                grasp & intervention_near_hole & intervention_yz & (~success)
            )
            insert_hold = (
                previous_insert_region
                & (~success)
                & (hole_x >= exit_hole_x_min)
            )
            intervention_region = insert_entry | insert_hold
        else:
            intervention_region = torch.zeros_like(success)
        insert_region = intervention_region
        region_phase = torch.where(intervention_region, insert_phase, no_phase)

        has_prev_yz = torch.isfinite(state["prev_yz_error"])
        has_prev_x = torch.isfinite(state["prev_hole_x"])
        progress_eps = float(intervention_cfg.get("progress_eps", 0.002))
        yz_error_eps = float(intervention_cfg.get("yz_error_eps", 0.002))
        safe_yz_margin = float(intervention_cfg.get("safe_yz_margin", 1.25))
        yz_worse = has_prev_yz & (yz_error > state["prev_yz_error"] + yz_error_eps)
        no_x_progress = has_prev_x & (hole_x <= state["prev_hole_x"] + progress_eps)
        safe_yz = (abs_y <= safe_yz_margin * hole_radii) & (
            abs_z <= safe_yz_margin * hole_radii
        )
        moved_away_from_hole = (
            has_prev_x
            & previous_insert_region
            & (hole_x < state["prev_hole_x"] - progress_eps)
        )
        lost_grasp = (~grasp) & previous_insert_region
        deviation = insert_region & (
            yz_worse
            | no_x_progress
            | (~safe_yz)
            | lost_grasp
            | moved_away_from_hole
        )

        patience = int(intervention_cfg.get("deviation_patience", 2))
        state["deviation_count"] = torch.where(
            deviation,
            state["deviation_count"] + 1,
            torch.zeros_like(state["deviation_count"]),
        )

        takeover_chunks = int(intervention_cfg.get("takeover_chunks", 5))
        takeover_max_chunks = int(intervention_cfg.get("takeover_max_chunks", 10))
        if takeover_chunks <= 0 or takeover_max_chunks < takeover_chunks:
            raise ValueError(
                "algorithm.intervention must satisfy "
                "0 < takeover_chunks <= takeover_max_chunks, got "
                f"{takeover_chunks=} and {takeover_max_chunks=}."
            )

        previous_takeover = state["expert_takeover"]
        previous_phase = state["intervention_phase"]
        takeover_used_after_chunk = torch.where(
            previous_takeover,
            state["takeover_used"] + 1,
            state["takeover_used"],
        )
        if self.mode == "train" and intervention_enabled:
            trigger = (
                insert_region
                & (~previous_takeover)
                & (state["deviation_count"] >= patience)
            )
        else:
            trigger = torch.zeros_like(intervention_region)
        active_phase = torch.where(previous_takeover, previous_phase, region_phase)
        takeover_phase = torch.where(trigger, insert_phase, active_phase)
        phase_min_chunks = torch.full_like(state["takeover_used"], takeover_chunks)
        phase_max_chunks = torch.full_like(state["takeover_used"], takeover_max_chunks)
        recovered = (
            (takeover_phase == self.INSERT_PHASE)
            & insert_region
            & grasp
            & safe_yz
            & (~deviation)
        )
        keep_until_min_chunks = previous_takeover & (
            takeover_used_after_chunk < phase_min_chunks
        )
        extend_until_recovered = (
            previous_takeover
            & (~recovered)
            & (takeover_used_after_chunk < phase_max_chunks)
        )
        next_takeover = (
            trigger | keep_until_min_chunks | extend_until_recovered
        ) & (~success)
        next_takeover = torch.where(
            done_any,
            torch.zeros_like(next_takeover),
            next_takeover,
        )
        released_takeover = previous_takeover & (~next_takeover)

        state["takeover_used"] = torch.where(
            trigger,
            torch.zeros_like(state["takeover_used"]),
            takeover_used_after_chunk,
        )
        state["takeover_used"] = torch.where(
            next_takeover,
            state["takeover_used"],
            torch.zeros_like(state["takeover_left"]),
        )
        remaining_to_min = torch.clamp(phase_min_chunks - state["takeover_used"], min=0)
        remaining_to_max = torch.clamp(phase_max_chunks - state["takeover_used"], min=0)
        state["takeover_left"] = torch.where(
            next_takeover,
            torch.where(
                state["takeover_used"] < phase_min_chunks,
                remaining_to_min,
                remaining_to_max,
            ),
            torch.zeros_like(state["takeover_used"]),
        )
        state["expert_takeover"] = next_takeover
        state["intervention_phase"] = torch.where(
            done_any,
            no_phase,
            torch.where(next_takeover, takeover_phase, region_phase),
        )
        state["intervention_region"] = torch.where(
            done_any,
            torch.zeros_like(intervention_region),
            intervention_region,
        )
        state["deviation"] = torch.where(
            done_any,
            torch.zeros_like(deviation),
            deviation,
        )
        state["deviation_count"] = torch.where(
            done_any | (~intervention_region) | trigger | released_takeover,
            torch.zeros_like(state["deviation_count"]),
            state["deviation_count"],
        )
        state["prev_yz_error"] = torch.where(
            insert_region,
            yz_error,
            torch.full_like(yz_error, float("nan")),
        )
        state["prev_hole_x"] = torch.where(
            insert_region,
            hole_x,
            torch.full_like(hole_x, float("nan")),
        )

        return self.export_policy_info(state)

    def _hole_radii(
        self,
        like: torch.Tensor,
        intervention_cfg: DictConfig,
        device: torch.device,
    ) -> torch.Tensor:
        if self.hole_radii is not None:
            return self.hole_radii.to(device, dtype=torch.float32)
        fallback_hole_radius = intervention_cfg.get("fallback_hole_radius", 0.035)
        return torch.full_like(like, float(fallback_hole_radius))
