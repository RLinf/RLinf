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

"""SuperMimic policy: motion tracking with a learned world model.

One module holds every network so that a single FSDP unit shards them and a
single weight sync ships the motion policy to the rollout worker, which only
calls :meth:`SuperMimicPolicy.act`, :meth:`SuperMimicPolicy.fold_policy_stats`
and :meth:`predict_action_batch`. The actor worker calls :meth:`forward` with
one ``ForwardType`` per training phase.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from rlinf.algorithms.supermimic.objectives import value_ceiling
from rlinf.algorithms.supermimic.representations import make_representation
from rlinf.algorithms.supermimic.scan import make_imagination, make_world_model_rollout
from rlinf.envs.sim.g1_mimic.compile_cache import CompileCache
from rlinf.envs.sim.g1_mimic.constants import ACTOR_GOAL_DIM
from rlinf.envs.sim.g1_mimic.g1_fk import set_kinematics_compile, soft_joint_limits
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.supermimic.networks import Critic, Dynamics, MotionPolicy

# Submodules each training phase updates; every other parameter is frozen.
PHASE_MODULES = {
    ForwardType.SUPERMIMIC_WORLD_MODEL: ("dynamics",),
    ForwardType.SUPERMIMIC_ACTOR: ("actor",),
    ForwardType.SUPERMIMIC_CRITIC: ("critic",),
}


class SuperMimicPolicy(nn.Module, BasePolicy):
    """World model, motion policy and critic over one state representation.

    Args:
        representation: ``gencoord`` or ``maxcoord``.
        torch_compile: Compile the training scans and the G1 kinematics on
            large CUDA batches. The kinematics switch is process-wide.
    """

    def __init__(self, representation: str = "gencoord", torch_compile: bool = True):
        super().__init__()
        self.representation = make_representation(representation)
        enc_dim = self.representation.encoding_dim
        self.dynamics = Dynamics(enc_dim, self.representation.prediction_dim)
        self.actor = MotionPolicy(enc_dim)
        self.critic = Critic(enc_dim)
        self.register_buffer("soft_limits", soft_joint_limits(), persistent=False)
        self.autocast_dtype: torch.dtype | None = None
        self._compile_cache = CompileCache(enabled=torch_compile)
        set_kinematics_compile(torch_compile)
        self._world_model_rollout = None
        self._imagination = None

    def set_autocast(self, dtype: torch.dtype | None) -> None:
        """Autocast every linear stack to ``dtype``; ``None`` keeps float32.

        Weights, optimizer state, kinematics and losses stay float32. Float32
        matmuls use TF32 tensor cores either way; that switch is process-wide.
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        self.autocast_dtype = dtype
        for module in (self.dynamics, self.actor, self.critic):
            module.autocast_dtype = dtype

    def configure_training(self, algorithm_cfg: DictConfig) -> None:
        """Build the scans for the horizon and discount of ``algorithm_cfg``.

        Only the actor worker calls this; acting needs none of it.
        """
        horizon = int(algorithm_cfg.H)
        gamma = float(algorithm_cfg.gamma)
        name = self.representation.name
        self._world_model_rollout = make_world_model_rollout(
            self.representation,
            compile_name=f"wm-{name}",
            compile_cache=self._compile_cache,
        )
        self._imagination = make_imagination(
            self.representation,
            horizon,
            gamma,
            value_ceiling(gamma),
            compile_name=f"imagine-{name}-{horizon}-{gamma}",
            compile_cache=self._compile_cache,
        )

    def phase_parameters(self, forward_type: ForwardType) -> list[nn.Parameter]:
        """Parameters the phase of ``forward_type`` updates."""
        return [
            param
            for name in PHASE_MODULES[forward_type]
            for param in getattr(self, name).parameters()
        ]

    def set_trainable(self, forward_type: ForwardType) -> None:
        """Let gradients reach only the parameters of one training phase."""
        self.requires_grad_(False)
        for param in self.phase_parameters(forward_type):
            param.requires_grad_(True)

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def act(self, enc: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Deterministic action ``[N, 29]`` of encodings and actor goals."""
        return self.actor(enc, goal)

    @torch.no_grad()
    def fold_policy_stats(self, enc_stats: torch.Tensor, goal_stats: torch.Tensor):
        """Fold merged moments of the encodings and goals the policy just saw."""
        self.actor.obs_norm.fold(enc_stats)
        self.actor.goal_norm.fold(goal_stats)

    @torch.no_grad()
    def fold_world_model_stats(
        self, enc_stats: torch.Tensor, target_stats: torch.Tensor
    ) -> None:
        """Fold merged moments of the encodings and executed PD targets."""
        self.dynamics.state_norm.fold(enc_stats)
        self.dynamics.action_norm.fold(target_stats)

    def predict_action_batch(self, env_obs: dict, **kwargs):
        """Deterministic action chunk ``[N, 1, 29]``.

        ``env_obs["states"]`` packs the encoding and the actor goal.
        """
        device = next(self.parameters()).device
        states = env_obs["states"].to(device=device, dtype=torch.float32)
        enc_dim = self.representation.encoding_dim
        if states.shape[-1] != enc_dim + ACTOR_GOAL_DIM:
            raise ValueError(
                f"SuperMimic {self.representation.name} observations must be "
                f"{enc_dim + ACTOR_GOAL_DIM} wide (encoding, goal), "
                f"got {states.shape[-1]}"
            )
        enc, goal = torch.split(states, [enc_dim, ACTOR_GOAL_DIM], dim=-1)
        return self.act(enc, goal).unsqueeze(1), {}

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "SuperMimicPolicy trains through its SUPERMIMIC_* forward types"
        )

    # ---------------------------------------------------------------- training

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        handlers = {
            ForwardType.SUPERMIMIC_WORLD_MODEL: self._world_model_forward,
            ForwardType.SUPERMIMIC_ACTOR: self._actor_forward,
            ForwardType.SUPERMIMIC_CRITIC: self._critic_forward,
        }
        handler = handlers.get(forward_type)
        if handler is None:
            return super().forward(forward_type, **kwargs)
        return handler(**kwargs)

    def _world_model_forward(
        self, states: torch.Tensor, actions: torch.Tensor, context: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """State loss of replay windows ``[B, L, ...]`` rolled out from their start."""
        loss, group_mse = self._world_model_rollout(
            self.dynamics, states, actions, context
        )
        return {"loss": loss, "group_mse": group_mse.detach()}

    def _actor_forward(
        self,
        start: torch.Tensor,
        prev_action: torch.Tensor,
        context: torch.Tensor,
        windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        refs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Negative mean imagined return, and the detached rollout for the critic."""
        ret, rewards, term_prob, steps, terminal = self._imagination(
            self.actor,
            self.dynamics,
            self.critic,
            start,
            prev_action,
            context,
            windows,
            refs,
            self.soft_limits,
        )
        return {
            "loss": -ret.mean(),
            "return": ret.detach().mean(),
            "rewards": rewards.detach(),
            "term_prob": term_prob.detach(),
            "steps": steps.detach(),
            "terminal": terminal.detach(),
        }

    def _critic_forward(
        self, features: torch.Tensor, targets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Values of critic features, and the squared error to ``targets``."""
        values = self.critic(features)
        out = {"values": values}
        if targets is not None:
            out["loss"] = (targets - values).square().mean()
        return out

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SuperMimicPolicy:
        """Build from an ``actor.model`` / ``rollout.model`` node."""
        return cls(
            representation=str(cfg.representation),
            torch_compile=bool(cfg.torch_compile),
        )
