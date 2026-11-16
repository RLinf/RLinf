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

"""Realworld smooth-intervene helpers for EnvWorker orchestration.

Bypasses policy inference across action-chunk boundaries while human teleop
continues. Requires ``algorithm.loss_type=embodied_dagger`` with online LeRobot
collection enabled, and a ``realworld`` env. PICO and SO-101 Leader with the
HG-DAgger keyboard wrapper are supported; SpaceMouse is not supported. The env
supplies hold actions and reports whether its controller still owns control.
This module owns external-action request construction and per-stage state.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.schema.embodied_types import PolicyInput
from rlinf.envs import SupportedEnvType
from rlinf.envs.utils import get_env_attr


def should_continue_smooth_intervene(
    intervene_flags: torch.Tensor | None, dones: torch.Tensor
) -> bool:
    if intervene_flags is None:
        return False
    return bool(intervene_flags[:, -1].any().item()) and not bool(dones.any().item())


class SmoothInterveneController:
    """Per-stage state for realworld smooth intervention."""

    def __init__(
        self,
        stage_num: int,
        num_envs_per_stage: int,
        num_action_chunks: int,
        action_dim: int,
        enabled: bool = False,
    ):
        self.enabled = bool(enabled)
        self.stage_num = int(stage_num)
        self.num_envs_per_stage = int(num_envs_per_stage)
        self.num_action_chunks = int(num_action_chunks)
        self.action_dim = int(action_dim)
        self.next_intervene_flags = [False for _ in range(self.stage_num)]
        self.last_actions: list[torch.Tensor | None] = [
            None for _ in range(self.stage_num)
        ]

    @classmethod
    def from_cfg(
        cls,
        cfg: DictConfig,
        *,
        stage_num: int,
        enable_train: bool,
        train_num_envs_per_stage: int,
    ) -> SmoothInterveneController:
        enabled = bool(
            enable_train
            and OmegaConf.select(cfg, "env.train.smooth_intervene", default=False)
        )
        if enabled:
            if OmegaConf.select(cfg, "algorithm.loss_type") != "embodied_dagger":
                raise ValueError(
                    "smooth_intervene requires algorithm.loss_type=embodied_dagger"
                )
            if not bool(
                OmegaConf.select(
                    cfg,
                    "algorithm.dagger.online_lerobot.enabled",
                    default=False,
                )
            ):
                raise ValueError(
                    "smooth_intervene requires algorithm.dagger.online_lerobot.enabled=True"
                )
            if (
                SupportedEnvType(OmegaConf.select(cfg, "env.train.env_type"))
                is not SupportedEnvType.REAL
            ):
                raise ValueError(
                    "smooth_intervene requires env.train.env_type to be 'realworld'"
                )
            if train_num_envs_per_stage != 1:
                raise ValueError(
                    "smooth_intervene requires exactly one env per EnvWorker stage"
                )
            # Imported here so a worker that never touches real hardware does
            # not pay for the realworld env package at import time.
            from rlinf.envs.real.wrappers.teleop.config import (
                resolve_teleop_devices,
            )
            from rlinf.robotics.parts.teleop import TeleopDevice

            # Every registered device, so a config naming a real one that this
            # check then rejects says why -- rather than reporting it as a
            # device that does not exist.
            devices = resolve_teleop_devices(
                OmegaConf.select(cfg, "env.train") or {},
                supported=TeleopDevice.names(),
            )
            named = [d if isinstance(d, str) else next(iter(dict(d))) for d in devices]
            episode_wrapper = OmegaConf.select(cfg, "env.train.so101_episode_wrapper")
            so101_hg_dagger = (
                named == ["so101_leader"] and episode_wrapper == "hg_dagger"
            )
            if "so101_leader" in named and episode_wrapper != "hg_dagger":
                raise ValueError(
                    "smooth_intervene with SO-101 leader requires "
                    "env.train.so101_episode_wrapper=hg_dagger"
                )
            if not named or (
                not all(name == "pico" for name in named) and not so101_hg_dagger
            ):
                raise ValueError(
                    "smooth_intervene requires PICO teleop or exactly one "
                    "SO-101 leader with the hg_dagger episode wrapper; "
                    f"got {named!r}"
                )
        model_path = "actor.model" if enable_train else "rollout.model"
        return cls(
            stage_num=stage_num,
            num_envs_per_stage=train_num_envs_per_stage,
            num_action_chunks=int(
                OmegaConf.select(cfg, f"{model_path}.num_action_chunks", default=0)
            ),
            action_dim=int(
                OmegaConf.select(cfg, f"{model_path}.action_dim", default=0)
            ),
            enabled=enabled,
        )

    def is_active(self, stage_id: int) -> bool:
        return self.enabled and self.next_intervene_flags[stage_id]

    def remember_actions(self, stage_id: int, actions: torch.Tensor | None) -> None:
        """Record the chunk a stage just executed.

        Its last step seeds ``get_hold_actions``, so intervention wrappers that
        hold at the previously commanded pose keep doing so across an external chunk
        instead of snapping back to their own default.
        """
        if not self.enabled or actions is None:
            return
        self.last_actions[stage_id] = actions.detach()

    def _fallback_actions(self, stage_id: int):
        """Return the last commanded step for a stage, if one was recorded."""
        actions = self.last_actions[stage_id]
        if actions is None:
            return None
        if actions.ndim == 2:
            actions = actions.reshape(actions.shape[0], self.num_action_chunks, -1)
        elif actions.ndim != 3:
            raise ValueError(
                "smooth_intervene expects actions with shape [B, action_dim] or "
                f"[B, T, action_dim], got {tuple(actions.shape)}"
            )
        return actions[:, -1, :].float().cpu().numpy()

    def build_external_policy_input(
        self,
        stage_id: int,
        *,
        env,
        obs: dict,
    ) -> PolicyInput:
        """Build a self-contained request for a chunk without model inference."""
        get_hold_actions = get_env_attr(env, "get_hold_actions")
        if not callable(get_hold_actions):
            raise ValueError(
                "smooth_intervene requires the env to expose get_hold_actions()"
            )
        hold_actions = torch.as_tensor(
            get_hold_actions(self._fallback_actions(stage_id)), dtype=torch.float32
        )
        expected_shape = (self.num_envs_per_stage, self.action_dim)
        if tuple(hold_actions.shape) != expected_shape:
            raise ValueError(
                f"hold_actions has shape {tuple(hold_actions.shape)}, "
                f"expected {expected_shape} for stage {stage_id}"
            )
        actions = hold_actions.unsqueeze(1).expand(-1, self.num_action_chunks, -1)
        return PolicyInput(obs=obs, external_actions=actions.contiguous())

    def on_chunk_done(
        self,
        stage_id: int,
        intervene_flags: torch.Tensor | None,
        dones: torch.Tensor,
        *,
        env=None,
    ) -> bool:
        """Update and return whether the next chunk should skip inference."""
        if not self.enabled:
            self.next_intervene_flags[stage_id] = False
            return False
        if bool(dones.any().item()):
            continue_smooth = False
        else:
            env_control_state = get_env_attr(env, "smooth_intervention_active")
            if callable(env_control_state):
                env_control_state = env_control_state()
            continue_smooth = (
                should_continue_smooth_intervene(intervene_flags, dones)
                if env_control_state is None
                else bool(env_control_state)
            )
        self.next_intervene_flags[stage_id] = continue_smooth
        return continue_smooth
