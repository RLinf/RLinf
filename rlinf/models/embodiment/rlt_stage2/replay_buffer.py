# Copyright 2025 The RLinf Authors.
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

"""Replay buffer for RLT Stage 2 TD3 training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .transition import (
    COLLECTION_PHASE_UNKNOWN,
    TransitionSource,
    resolve_chunk_source,
    resolve_collection_phase_id,
)


@dataclass
class TransitionBatch:
    x: torch.Tensor
    a: torch.Tensor
    a_tilde: torch.Tensor
    action_chunk: torch.Tensor
    ref_chunk: torch.Tensor
    rewards: torch.Tensor
    next_x: torch.Tensor
    next_a_tilde: torch.Tensor
    next_ref_chunk: torch.Tensor
    dones: torch.Tensor
    intervention: torch.Tensor
    source: torch.Tensor
    source_chunk: torch.Tensor
    collection_phase_id: torch.Tensor
    success: torch.Tensor
    intervention_flag: torch.Tensor
    episode_id: torch.Tensor
    step_id: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "x": self.x,
            "a": self.a,
            "a_tilde": self.a_tilde,
            "action_chunk": self.action_chunk,
            "ref_chunk": self.ref_chunk,
            "rewards": self.rewards,
            "next_x": self.next_x,
            "next_a_tilde": self.next_a_tilde,
            "next_ref_chunk": self.next_ref_chunk,
            "dones": self.dones,
            "intervention": self.intervention,
            "source": self.source,
            "source_chunk": self.source_chunk,
            "collection_phase_id": self.collection_phase_id,
            "success": self.success,
            "intervention_flag": self.intervention_flag,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
        }


class RLTStage2ReplayBuffer:
    """Fixed-capacity circular replay buffer storing chunk-level TD3 transitions."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_chunk_dim: int,
        chunk_length: int,
        seed: int = 1234,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_chunk_dim = int(action_chunk_dim)
        self.chunk_length = int(chunk_length)

        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

        self._x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._action_chunk = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._ref_chunk = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self._next_x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_ref_chunk = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)
        self._intervention = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._source = np.zeros((capacity, 1), dtype=np.uint8)
        self._source_chunk = np.zeros((capacity, chunk_length), dtype=np.uint8)
        self._collection_phase_id = np.zeros((capacity, 1), dtype=np.uint8)
        self._success = np.zeros((capacity, 1), dtype=np.int8)
        self._intervention_flag = np.zeros((capacity, 1), dtype=np.bool_)
        self._episode_id = np.zeros((capacity, 1), dtype=np.int32)
        self._step_id = np.zeros((capacity, 1), dtype=np.int32)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        return self._size >= int(min_size)

    def add(
        self,
        *,
        x: np.ndarray,
        action_chunk: np.ndarray,
        ref_chunk: np.ndarray,
        rewards: np.ndarray,
        next_x: np.ndarray,
        next_ref_chunk: np.ndarray,
        done: float,
        intervention: float | np.ndarray = 0.0,
        source: int | None = None,
        source_chunk: int | np.ndarray | None = None,
        collection_phase: str | int | None = None,
        success: int | bool = 0,
        intervention_flag: bool | None = None,
        episode_id: int = 0,
        step_id: int = 0,
    ) -> None:
        self._x[self._ptr] = x
        self._action_chunk[self._ptr] = action_chunk
        self._ref_chunk[self._ptr] = ref_chunk
        self._rewards[self._ptr] = rewards
        self._next_x[self._ptr] = next_x
        self._next_ref_chunk[self._ptr] = next_ref_chunk
        self._dones[self._ptr] = done
        intervention_array = np.asarray(intervention, dtype=np.float32)
        if intervention_array.size == 1:
            intervention_array = np.full(
                (self.action_chunk_dim,),
                float(intervention_array.reshape(-1)[0]),
                dtype=np.float32,
            )
        elif intervention_array.size == self.chunk_length:
            action_dim = self.action_chunk_dim // self.chunk_length
            intervention_array = np.repeat(
                intervention_array.reshape(self.chunk_length, 1),
                action_dim,
                axis=1,
            )
        if intervention_array.size != self.action_chunk_dim:
            raise ValueError(
                "RLT intervention mask must be scalar, chunk_length, or action_chunk_dim, "
                f"got {intervention_array.shape=} for {self.chunk_length=} "
                f"and {self.action_chunk_dim=}."
            )
        self._intervention[self._ptr] = intervention_array.reshape(-1)

        if source_chunk is None:
            per_step_intervention = intervention_array.reshape(
                self.chunk_length,
                self.action_chunk_dim // self.chunk_length,
            ).any(axis=-1)
            source_chunk_array = np.where(
                per_step_intervention,
                int(TransitionSource.HUMAN),
                int(source if source is not None else TransitionSource.RL),
            ).astype(np.uint8)
        else:
            source_chunk_array = np.asarray(source_chunk, dtype=np.uint8)
            if source_chunk_array.size == 1:
                source_chunk_array = np.full(
                    (self.chunk_length,),
                    int(source_chunk_array.reshape(-1)[0]),
                    dtype=np.uint8,
                )
        if source_chunk_array.size != self.chunk_length:
            raise ValueError(
                "RLT source_chunk must be scalar or chunk_length, got "
                f"{source_chunk_array.shape=} for {self.chunk_length=}."
            )
        source_chunk_array = source_chunk_array.reshape(self.chunk_length)
        resolved_source = (
            int(source)
            if source is not None
            else resolve_chunk_source(source_chunk_array)
        )
        if source is not None and len(set(source_chunk_array.reshape(-1).tolist())) > 1:
            resolved_source = int(TransitionSource.MIXED)
        resolved_intervention = (
            bool(intervention_flag)
            if intervention_flag is not None
            else bool(
                intervention_array.any()
                or np.any(source_chunk_array == int(TransitionSource.HUMAN))
                or np.any(source_chunk_array == int(TransitionSource.MIXED))
            )
        )
        self._source[self._ptr] = resolved_source
        self._source_chunk[self._ptr] = source_chunk_array
        self._collection_phase_id[self._ptr] = resolve_collection_phase_id(collection_phase)
        self._success[self._ptr] = int(bool(success))
        self._intervention_flag[self._ptr] = resolved_intervention
        self._episode_id[self._ptr] = int(episode_id)
        self._step_id[self._ptr] = int(step_id)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str) -> TransitionBatch:
        indices = self._rng.integers(0, self._size, size=int(batch_size))
        return TransitionBatch(
            x=torch.as_tensor(self._x[indices], device=device),
            a=torch.as_tensor(self._action_chunk[indices], device=device),
            a_tilde=torch.as_tensor(self._ref_chunk[indices], device=device),
            action_chunk=torch.as_tensor(self._action_chunk[indices], device=device),
            ref_chunk=torch.as_tensor(self._ref_chunk[indices], device=device),
            rewards=torch.as_tensor(self._rewards[indices], device=device),
            next_x=torch.as_tensor(self._next_x[indices], device=device),
            next_a_tilde=torch.as_tensor(self._next_ref_chunk[indices], device=device),
            next_ref_chunk=torch.as_tensor(self._next_ref_chunk[indices], device=device),
            dones=torch.as_tensor(self._dones[indices], device=device),
            intervention=torch.as_tensor(self._intervention[indices], device=device),
            source=torch.as_tensor(self._source[indices], device=device),
            source_chunk=torch.as_tensor(self._source_chunk[indices], device=device),
            collection_phase_id=torch.as_tensor(
                self._collection_phase_id[indices],
                device=device,
            ),
            success=torch.as_tensor(self._success[indices], device=device),
            intervention_flag=torch.as_tensor(
                self._intervention_flag[indices],
                device=device,
            ),
            episode_id=torch.as_tensor(self._episode_id[indices], device=device),
            step_id=torch.as_tensor(self._step_id[indices], device=device),
        )

    def state_dict(self) -> dict[str, Any]:
        n = self._size
        return {
            "ptr": self._ptr,
            "size": self._size,
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_chunk_dim": self.action_chunk_dim,
            "chunk_length": self.chunk_length,
            "x": self._x[:n].copy(),
            "a": self._action_chunk[:n].copy(),
            "a_tilde": self._ref_chunk[:n].copy(),
            "action_chunk": self._action_chunk[:n].copy(),
            "ref_chunk": self._ref_chunk[:n].copy(),
            "rewards": self._rewards[:n].copy(),
            "next_x": self._next_x[:n].copy(),
            "next_a_tilde": self._next_ref_chunk[:n].copy(),
            "next_ref_chunk": self._next_ref_chunk[:n].copy(),
            "dones": self._dones[:n].copy(),
            "intervention": self._intervention[:n].copy(),
            "source": self._source[:n].copy(),
            "source_chunk": self._source_chunk[:n].copy(),
            "collection_phase_id": self._collection_phase_id[:n].copy(),
            "success": self._success[:n].copy(),
            "intervention_flag": self._intervention_flag[:n].copy(),
            "episode_id": self._episode_id[:n].copy(),
            "step_id": self._step_id[:n].copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = n
        self._x[:n] = state["x"]
        self._action_chunk[:n] = (
            state["action_chunk"] if "action_chunk" in state else state["a"]
        )
        self._ref_chunk[:n] = (
            state["ref_chunk"] if "ref_chunk" in state else state["a_tilde"]
        )
        self._rewards[:n] = state["rewards"]
        self._next_x[:n] = state["next_x"]
        self._next_ref_chunk[:n] = (
            state["next_ref_chunk"]
            if "next_ref_chunk" in state
            else state["next_a_tilde"]
        )
        self._dones[:n] = state["dones"]
        if "intervention" in state:
            intervention = np.asarray(state["intervention"], dtype=np.float32)
            if intervention.ndim == 2 and intervention.shape[1] == 1:
                intervention = np.repeat(intervention, self.action_chunk_dim, axis=1)
            self._intervention[:n] = intervention.reshape(n, self.action_chunk_dim)
        self._source[:n] = np.asarray(
            state.get(
                "source",
                np.full((n, 1), int(TransitionSource.RL), dtype=np.uint8),
            ),
            dtype=np.uint8,
        ).reshape(n, 1)
        self._source_chunk[:n] = np.asarray(
            state.get(
                "source_chunk",
                np.full((n, self.chunk_length), int(TransitionSource.RL), dtype=np.uint8),
            ),
            dtype=np.uint8,
        ).reshape(n, self.chunk_length)
        self._collection_phase_id[:n] = np.asarray(
            state.get(
                "collection_phase_id",
                np.full((n, 1), COLLECTION_PHASE_UNKNOWN, dtype=np.uint8),
            ),
            dtype=np.uint8,
        ).reshape(n, 1)
        self._success[:n] = np.asarray(
            state.get("success", np.zeros((n, 1), dtype=np.int8)),
            dtype=np.int8,
        ).reshape(n, 1)
        self._intervention_flag[:n] = np.asarray(
            state.get(
                "intervention_flag",
                self._intervention[:n].reshape(
                    n,
                    self.chunk_length,
                    self.action_chunk_dim // self.chunk_length,
                ).any(axis=(1, 2), keepdims=False).reshape(n, 1),
            ),
            dtype=np.bool_,
        ).reshape(n, 1)
        self._episode_id[:n] = np.asarray(
            state.get("episode_id", np.zeros((n, 1), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(n, 1)
        self._step_id[:n] = np.asarray(
            state.get("step_id", np.zeros((n, 1), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(n, 1)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

    def save_checkpoint(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, "buffer.pt"))

    def load_checkpoint(self, load_path: str) -> None:
        if os.path.isdir(load_path):
            load_path = os.path.join(load_path, "buffer.pt")
        state = torch.load(load_path, map_location="cpu", weights_only=False)
        if "replay_buffer" in state:
            state = state["replay_buffer"]
        self.load_state_dict(state)

    def get_stats(self) -> dict[str, float]:
        n = self._size
        intervention_rate = (
            float(self._intervention[:n].mean()) if n > 0 else 0.0
        )
        human_chunk_rate = (
            float(
                np.any(
                    np.logical_or(
                        self._source_chunk[:n] == int(TransitionSource.HUMAN),
                        self._source_chunk[:n] == int(TransitionSource.MIXED),
                    ),
                    axis=1,
                ).mean()
            )
            if n > 0
            else 0.0
        )
        return {
            "size": float(self._size),
            "capacity": float(self.capacity),
            "fill_ratio": float(self._size / max(self.capacity, 1)),
            "intervention_rate": intervention_rate,
            "human_chunk_rate": human_chunk_rate,
        }
