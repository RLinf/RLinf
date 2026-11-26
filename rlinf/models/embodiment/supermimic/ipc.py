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

"""Flat tensor packing and direct P2P sync helpers for SuperMimic workers."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    ACTOR_GOAL_DIM,
    SIM_CONTEXT_DIM,
)
from rlinf.hybrid_engines.fsdp.utils import to_local_if_dtensor


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
    sharding = (
        str(cfg.actor.fsdp_config.get("sharding_strategy", "full_shard"))
        .strip()
        .lower()
        .replace("-", "_")
    )
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
