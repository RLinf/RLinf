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

One module holds every network so that a single weight sync ships the motion
policy to the rollout worker, which only calls :meth:`SuperMimicPolicy.act`,
:meth:`SuperMimicPolicy.fold_policy_stats` and :meth:`predict_action_batch`.
The actor worker calls :meth:`forward` with one ``ForwardType`` per training
phase. Under FSDP each network is its own unit, and a phase's forward gathers
the networks it reads once for all of its steps.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from rlinf.algorithms.supermimic.objectives import value_ceiling
from rlinf.algorithms.supermimic.scan import make_imagination, make_world_model_rollout
from rlinf.envs.sim.g1_mimic.compile_cache import CompileCache, run_compiled
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    ACTOR_GOAL_DIM,
    SIM_CONTEXT_DIM,
)
from rlinf.envs.sim.g1_mimic.g1_fk import set_kinematics_compile, soft_joint_limits
from rlinf.envs.sim.g1_mimic.representations import make_representation
from rlinf.hybrid_engines.fsdp.utils import to_local_if_dtensor
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.supermimic.networks import Critic, Dynamics, MotionPolicy
from rlinf.scheduler import Worker
from rlinf.utils.precision import torch_dtype_from_precision
from rlinf.utils.utils import collect_param_names_need_sync

# The rollout worker only acts; it never reads these networks.
ACTOR_ONLY_PREFIXES = ("dynamics.", "critic.")


def sync_param_names(module: nn.Module) -> list[str]:
    """Sorted parameter and buffer names synced from actor to rollout."""
    return sorted(
        name
        for name in collect_param_names_need_sync(module)
        if not name.startswith(ACTOR_ONLY_PREFIXES)
    )


def use_direct_weight_sync(
    cfg: DictConfig, actor_world_size: int, rollout_world_size: int
) -> bool:
    """Whether each actor rank holds full weights and pairs 1:1 with a rollout rank."""
    if actor_world_size != rollout_world_size:
        return False
    if cfg.actor.get("enable_offload", False) or cfg.rollout.get(
        "enable_offload", False
    ):
        return False
    sharding = cfg.actor.fsdp_config.get("sharding_strategy", "full_shard")
    return actor_world_size == 1 or sharding == "no_shard"


class FlatTensorPacker:
    """Pack and unpack named module parameters/buffers through one flat buffer.

    References are resolved by ``(submodule, attr)`` on each call so FSDP2
    parameter swaps between sharded and unsharded views stay valid.
    """

    def __init__(
        self,
        module: nn.Module,
        names: list[str],
        device: torch.device | str | int | None = None,
    ):
        resolved: list[tuple[nn.Module, str, torch.dtype, torch.Size, int]] = []
        offset = 0
        inferred_device: torch.device | None = None
        for name in names:
            parent_name, _, attr = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            tensor = to_local_if_dtensor(getattr(parent, attr).detach())
            if inferred_device is None:
                inferred_device = tensor.device
            align = max(tensor.element_size(), 8)
            offset = (offset + align - 1) // align * align
            resolved.append((parent, attr, tensor.dtype, tensor.shape, offset))
            offset += tensor.numel() * tensor.element_size()

        buf_device = inferred_device if device is None else device
        self.buffer = torch.empty(offset, dtype=torch.uint8, device=buf_device)
        grouped: dict[
            torch.dtype, tuple[list[tuple[nn.Module, str]], list[torch.Tensor]]
        ] = {}
        for parent, attr, dtype, shape, start in resolved:
            nbytes = shape.numel() * torch.empty((), dtype=dtype).element_size()
            view = self.buffer[start : start + nbytes].view(dtype).view(shape)
            refs, views = grouped.setdefault(dtype, ([], []))
            refs.append((parent, attr))
            views.append(view)
        self._groups = list(grouped.values())

    @torch.no_grad()
    def pack(self) -> torch.Tensor:
        """Copy current local parameters and buffers into ``self.buffer``."""
        for refs, views in self._groups:
            srcs = [
                to_local_if_dtensor(getattr(parent, attr).detach())
                for parent, attr in refs
            ]
            torch._foreach_copy_(views, srcs)
        return self.buffer

    @torch.no_grad()
    def unpack(self) -> None:
        """Copy ``self.buffer`` into the target module's parameters and buffers."""
        for refs, views in self._groups:
            dsts = [
                to_local_if_dtensor(getattr(parent, attr).detach())
                for parent, attr in refs
            ]
            torch._foreach_copy_(dsts, views)


def obs_packet_schema(
    num_envs: int, enc_dim: int
) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """Fixed tensor layout exchanged from env to rollout each control step."""
    return [
        ("enc", (num_envs, enc_dim), torch.float32),
        ("goal", (num_envs, ACTOR_GOAL_DIM), torch.float32),
        ("enc_stats", (1 + 2 * enc_dim,), torch.float64),
        ("goal_stats", (1 + 2 * ACTOR_GOAL_DIM,), torch.float64),
    ]


def transition_packet_schema(
    num_envs: int, state_dim: int, enc_dim: int
) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """Fixed tensor layout exchanged from env to actor each control step."""
    return [
        ("state", (num_envs, state_dim), torch.float32),
        ("action", (num_envs, ACTION_DIM), torch.float32),
        ("prev_action", (num_envs, ACTION_DIM), torch.float32),
        ("motion_time", (num_envs,), torch.long),
        ("terminated", (num_envs,), torch.bool),
        ("truncated", (num_envs,), torch.bool),
        ("sim_context", (num_envs, SIM_CONTEXT_DIM), torch.float32),
        ("enc_stats", (1 + 2 * enc_dim,), torch.float64),
        ("goal_stats", (1 + 2 * ACTOR_GOAL_DIM,), torch.float64),
        ("target_stats", (1 + 2 * ACTION_DIM,), torch.float64),
    ]


class FlatDictBuffer:
    """Pack a fixed-schema tensor dict into one 16-byte-aligned flat ``uint8`` buffer."""

    def __init__(
        self,
        schema: list[tuple[str, tuple[int, ...], torch.dtype]],
        device: torch.device | str | int,
    ):
        slots: list[tuple[str, tuple[int, ...], torch.dtype, int, int]] = []
        offset = 0
        for name, shape, dtype in schema:
            offset = (offset + 15) // 16 * 16
            elem_size = torch.empty((), dtype=dtype).element_size()
            numel = 1
            for dim in shape:
                numel *= int(dim)
            nbytes = numel * elem_size
            slots.append((name, shape, dtype, offset, nbytes))
            offset += nbytes
        self.buffer = torch.empty(offset, dtype=torch.uint8, device=device)
        self.views: dict[str, torch.Tensor] = {}
        grouped: dict[torch.dtype, tuple[list[str], list[torch.Tensor]]] = {}
        for name, shape, dtype, start, nbytes in slots:
            view = self.buffer[start : start + nbytes].view(dtype).view(shape)
            self.views[name] = view
            names, views = grouped.setdefault(dtype, ([], []))
            names.append(name)
            views.append(view)
        self._groups = list(grouped.values())

    @torch.no_grad()
    def pack(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Copy tensors from ``data`` into ``self.buffer``."""
        for names, views in self._groups:
            srcs = [data[name] for name in names]
            torch._foreach_copy_(views, srcs)
        return self.buffer


def _critic_step(
    critic: nn.Module, features: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = critic.forward(features)
    return values, (targets - values).square().mean()


# Submodules each training phase updates; every other parameter is frozen.
PHASE_MODULES = {
    ForwardType.SUPERMIMIC_WORLD_MODEL: ("dynamics",),
    ForwardType.SUPERMIMIC_ACTOR: ("actor",),
    ForwardType.SUPERMIMIC_CRITIC: ("critic",),
}
# Submodules each training phase reads.
PHASE_INPUTS = {
    ForwardType.SUPERMIMIC_WORLD_MODEL: ("dynamics",),
    ForwardType.SUPERMIMIC_ACTOR: ("actor", "dynamics", "critic"),
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
        self._compile_cache = CompileCache(
            enabled=torch_compile,
            dynamic=False,
        )
        set_kinematics_compile(torch_compile)
        self._world_model_rollout = None
        self._imagination = None

    def set_autocast(self, dtype: torch.dtype | None) -> None:
        """Autocast every linear stack to ``dtype``; ``None`` keeps float32.

        Weights, optimizer state, kinematics and losses stay float32. Float32
        matmuls use TF32 tensor cores on CUDA; that switch is process-wide.
        """
        if Worker.torch_device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
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
        if targets is None:
            values = run_compiled(
                "critic-values",
                self.critic.forward,
                features,
                cache=self._compile_cache,
            )
            return {"values": values}
        values, loss = run_compiled(
            "critic-step",
            _critic_step,
            self.critic,
            features,
            targets,
            cache=self._compile_cache,
        )
        return {"values": values, "loss": loss}

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SuperMimicPolicy:
        """Build from an ``actor.model`` / ``rollout.model`` node."""
        policy = cls(
            representation=str(cfg.representation),
            torch_compile=bool(cfg.torch_compile),
        )
        amp = cfg.get("amp_autocast", None)
        if amp is not None and amp.get("enabled", False):
            policy.set_autocast(torch_dtype_from_precision(amp.precision))
        return policy
