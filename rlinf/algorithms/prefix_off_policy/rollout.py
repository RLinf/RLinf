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

from typing import Any, Literal

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.prefix_off_policy.route import (
    PrefixRouteContext,
    build_prefix_route,
)
from rlinf.algorithms.prefix_off_policy.transition import (
    PREFIX_OBS_KEYS,
    PREFIX_TRANSITION_PREFIX,
)
from rlinf.data.schema.embodied_types import (
    ActionPayload,
    EnvPart,
    PolicyInput,
    TrajectoryKey,
    TrajectorySource,
    merge_batch_values,
    split_batch_value,
)
from rlinf.models.embodiment.prefix.config import (
    build_state_history_buffer,
    get_prefix_feature_model_config,
)
from rlinf.models.embodiment.prefix.contracts import (
    check_prefix_feature_dim,
    extract_prefix_obs,
)


class PrefixRolloutRuntime:
    """Own frozen features, routing, and source-indexed train/eval history."""

    def __init__(
        self,
        cfg: Any,
        policy_model: Any,
        feature_model: Any,
        *,
        gate: Any = None,
        expert_model: Any = None,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = (
            cfg.rollout.model
            if cfg.get("runner", {}).get("only_eval", False)
            else cfg.actor.model
        )
        self.policy_model = policy_model
        self.feature_model = feature_model
        self.gate = gate
        self.expert_model = expert_model
        self.route = build_prefix_route(cfg)
        self.histories = {}
        self._terminal_features = {}

    @classmethod
    def build(
        cls, cfg: Any, policy_model: Any, *, device: Any
    ) -> "PrefixRolloutRuntime | None":
        """Build an optional runtime without changing non-Prefix model configs."""
        from rlinf.algorithms.expert import build_expert_model_config
        from rlinf.models import get_model

        feature_cfg = get_prefix_feature_model_config(cfg)
        gate_cfg = OmegaConf.select(cfg, "rollout.routing_gate", default=None)
        if feature_cfg is None:
            if gate_cfg is not None and gate_cfg.get("enable", False):
                raise ValueError(
                    "rollout.routing_gate requires rollout.prefix_feature_model."
                )
            return None
        model_cfg = (
            cfg.rollout.model if cfg.runner.get("only_eval", False) else cfg.actor.model
        )
        feature_model = get_model(feature_cfg)
        feature_model.eval()
        feature_model.requires_grad_(False)
        check_prefix_feature_dim(feature_model, model_cfg.z_dim)
        gate = None
        if gate_cfg is not None and gate_cfg.get("enable", False):
            from rlinf.algorithms.rlt.routing_gate import build_routing_gate

            gate = build_routing_gate(
                gate_cfg,
                device=device,
                num_action_chunks=model_cfg.num_action_chunks,
                env_decoupled_mode=cfg.runner.get("enable_decoupled_mode", False),
            )
        expert_model = None
        if cfg.rollout.get("expert_model", None):
            expert_cfg = build_expert_model_config(
                cfg, model_cfg, prefix_feature_model_config=feature_cfg
            )
            expert_model = get_model(expert_cfg)
            if cfg.runner.get("expert_ckpt_path", None):
                expert_model.load_state_dict(torch.load(cfg.runner.expert_ckpt_path))
            expert_model.eval()
        return cls(
            cfg, policy_model, feature_model, gate=gate, expert_model=expert_model
        )

    def reset(self, mode: Literal["train", "eval"]) -> None:
        """Clear only the requested mode's routing and history state."""
        if self.gate is not None:
            self.gate.reset(mode=mode)
        for key in list(self.histories):
            if key[0] == mode:
                del self.histories[key]
        self._terminal_features.clear()

    def to(self, device: Any) -> None:
        """Move every runtime-owned model and retained history together."""
        self.feature_model.to(device)
        if self.expert_model is not None:
            self.expert_model.to(device)
        if self.gate is not None:
            self.gate.to(device)
        for history in self.histories.values():
            history.to(device)
        self._terminal_features.clear()

    def _features(
        self,
        obs: dict,
        sources: list[TrajectorySource],
        mode: str,
        *,
        commit: bool,
        dones: torch.Tensor | None = None,
    ) -> dict:
        features = extract_prefix_obs(self.feature_model, obs)
        sizes = [source.size for source in sources]
        feature_parts = split_batch_value(features, sizes)
        done_parts = split_batch_value(dones, sizes)
        for source, part, done in zip(sources, feature_parts, done_parts):
            key = (mode, source.key.env_rank, source.key.stage_id)
            history = self.histories.get(key)
            if history is None:
                history = build_state_history_buffer(self.model_cfg)
                if commit:
                    self.histories[key] = history
            rows = torch.arange(source.offset, source.offset + source.size)
            if commit and done is not None:
                mask = done.reshape(source.size, -1).any(dim=-1)
                history.reset(mask=mask, indices=rows)
            part.update(history.fuse(part, commit=commit, indices=rows))
        return merge_batch_values(feature_parts)

    @torch.no_grad()
    def prepare_transitions(self, request: PolicyInput, *, mode: str = "train") -> None:
        """Peek terminal observations before any current-episode reset or commit."""
        self._terminal_features.clear()
        if not request.env_parts:
            return
        observations = split_batch_value(request.obs, request.request_sizes)
        for part, obs in zip(request.env_parts, observations):
            if part is not None and (part.requires_inference or request.is_last):
                self._terminal_features[id(part)] = self._features(
                    part.next_obs if part.next_obs is not None else obs,
                    part.sources,
                    mode,
                    commit=False,
                )

    def transition_obs(
        self, part: EnvPart, policy_obs: dict | None, current_features: dict | None
    ) -> dict:
        """Return post-action features without advancing an environment's history."""
        cached = self._terminal_features.pop(id(part), None)
        if cached is not None:
            return cached
        if current_features is not None:
            return current_features
        obs = part.next_obs if part.next_obs is not None else policy_obs
        if obs is None:
            raise ValueError("Prefix transition requires a post-action observation.")
        return self._features(obs, part.sources, "train", commit=False)

    @torch.no_grad()
    def predict(
        self,
        request: PolicyInput,
        *,
        mode: Literal["train", "eval"] = "train",
        version: int = 0,
        stage_id: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Infer actions after resolving old-episode terminal features."""
        self.prepare_transitions(request, mode=mode)
        sources = request.sources
        if not sources:
            batch_size = next(
                value.shape[0]
                for value in request.obs.values()
                if isinstance(value, (torch.Tensor, np.ndarray))
            )
            sources = [
                TrajectorySource(
                    key=TrajectoryKey(
                        step_id=0,
                        epoch_id=0,
                        env_rank=-1,
                        stage_id=stage_id,
                        chunk_id=0,
                    ),
                    size=batch_size,
                )
            ]
        features = self._features(
            request.obs, sources, mode, commit=True, dones=request.dones
        )
        actions, result = self.policy_model.predict_action_batch(
            env_obs=features, mode=mode, return_obs=True
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actor_switch = request.rlt_switch_flags
        expert_requested = request.intervene_flags
        if self.gate is not None:
            decision = self.gate.step(
                request.obs,
                mode=mode,
                stage_id=stage_id,
                reset_mask=request.dones,
                external_actor_switch=actor_switch,
                actor_routing_enabled=self.route.actor_routing_enabled(version),
                expert_routing_enabled=self.route.expert_routing_enabled(
                    version, mode=mode, expert_model=self.expert_model
                ),
            )
            if self.gate.controls_actor_routing:
                actor_switch = decision.actor_switch
            expert_requested = decision.expert_requested
            result["trajectory_infos"] = decision.trajectory_infos
        output = self.route.route(
            PrefixRouteContext(
                env_obs=request.obs,
                rlt_obs=features,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=actor_switch,
                intervene_requested=expert_requested,
                expert_model=self.expert_model,
                version=version,
            )
        )
        for key in PREFIX_OBS_KEYS:
            output.result["forward_inputs"][f"{PREFIX_TRANSITION_PREFIX}{key}"] = (
                features[key]
            )
        return output.actions, output.result

    def action_payload(
        self, actions: torch.Tensor, diagnostics: dict | None
    ) -> ActionPayload:
        """Attach only environment-facing gate feedback to executable actions."""
        infos = {}
        if self.gate is not None:
            infos = {
                key: diagnostics[key]
                for key in self.gate.rollout_info_keys
                if diagnostics is not None and key in diagnostics
            }
        return ActionPayload(actions=actions, rollout_infos=infos)
