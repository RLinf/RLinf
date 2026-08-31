# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact episode traces for offline calibration of the RLT STEAM gate."""

import os
import time
from pathlib import Path
from typing import Any

import torch

from rlinf.algorithms.rlt.phase_head import RLT_PHASE_FEATURE_KEY
from rlinf.data.schema.embodied_types import Trajectory

RLT_GATE_TRACE_KEYS = (
    "rlt_gate_score_ready",
    "rlt_gate_score_min",
    "rlt_gate_actor_active",
    "rlt_oracle_expert_active",
    "geometry_critical_active",
    "success_current",
)


class RLTGateTraceWriter:
    """Write scalar gate diagnostics without changing replay-buffer storage."""

    def __init__(self, cfg: Any, *, rank: int) -> None:
        save_path = cfg.get("save_path", None)
        if not save_path:
            raise ValueError(
                "algorithm.rlt_gate_calibration.save_path is required when enabled"
            )
        self.output_dir = Path(os.fspath(save_path)) / f"rank_{int(rank)}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(cfg.get("chunk_size", 10))
        self.max_trace_files = int(cfg.get("max_trace_files", 0))
        self._run_id = f"{os.getpid()}_{time.time_ns()}"
        self._counter = 0

    @staticmethod
    def _step_tensor(value: torch.Tensor, *, steps: int, batch: int) -> torch.Tensor:
        value = value.detach().cpu()
        if value.shape[0] != steps or value.shape[1] != batch:
            raise ValueError(
                "Gate trace tensor must start with [T, B], got "
                f"{tuple(value.shape)} for T={steps}, B={batch}."
            )
        return value.reshape(steps, batch, -1)[:, :, -1].contiguous()

    @staticmethod
    def _aligned_done_tensor(
        value: torch.Tensor,
        *,
        steps: int,
        batch: int,
    ) -> torch.Tensor:
        value = value.detach().cpu()
        if value.shape[1] != batch:
            raise ValueError(
                f"Gate trace done tensor batch mismatch: {tuple(value.shape)}"
            )
        if value.shape[0] > steps:
            extra = int(value.shape[0] - steps)
            if extra < 1 or steps % extra != 0:
                raise ValueError(
                    "Cannot align rollout done tensor with gate trace: "
                    f"shape={tuple(value.shape)}, steps={steps}."
                )
            epoch_len = steps // extra
            value = value.reshape(extra, epoch_len + 1, batch, *value.shape[2:])
            value = value[:, 1:].reshape(steps, batch, *value.shape[3:])
        if value.shape[0] != steps:
            raise ValueError(
                f"Gate trace done tensor has {value.shape[0]} rows, expected {steps}."
            )
        return value.reshape(steps, batch, -1).to(torch.bool).any(dim=-1)

    def _build_trace(self, trajectory: Trajectory) -> dict[str, Any] | None:
        forward_inputs = trajectory.forward_inputs
        score = forward_inputs.get("rlt_gate_score_min")
        if not isinstance(score, torch.Tensor) or score.ndim < 2:
            return None

        steps, batch = int(score.shape[0]), int(score.shape[1])
        trace: dict[str, Any] = {
            "format_version": 1,
            "chunk_size": self.chunk_size,
            "model_weights_id": trajectory.model_weights_id,
            "num_steps": steps,
            "batch_size": batch,
        }
        for key in RLT_GATE_TRACE_KEYS:
            value = forward_inputs.get(key)
            if isinstance(value, torch.Tensor):
                trace[key] = self._step_tensor(value, steps=steps, batch=batch)
        phase_features = forward_inputs.get(RLT_PHASE_FEATURE_KEY)
        if isinstance(phase_features, torch.Tensor):
            if phase_features.ndim != 4 or phase_features.shape[:2] != (
                steps,
                batch,
            ):
                raise ValueError(
                    "STEAM phase features must have shape [T, B, E, D], got "
                    f"{tuple(phase_features.shape)}"
                )
            trace[RLT_PHASE_FEATURE_KEY] = (
                phase_features.detach().cpu().to(torch.float16).contiguous()
            )

        if isinstance(trajectory.rewards, torch.Tensor):
            rewards = trajectory.rewards.detach().cpu()
            if rewards.shape[:2] == (steps, batch):
                trace["reward_sum"] = rewards.reshape(steps, batch, -1).sum(dim=-1)
        for key in ("dones", "terminations", "truncations"):
            value = getattr(trajectory, key, None)
            if isinstance(value, torch.Tensor):
                trace[key] = self._aligned_done_tensor(
                    value,
                    steps=steps,
                    batch=batch,
                )
        if isinstance(trajectory.versions, torch.Tensor):
            versions = trajectory.versions
            if versions.shape[:2] == (steps - 1, batch) and steps > 1:
                versions = torch.cat([versions, versions[-1:]], dim=0)
            if versions.shape[:2] == (steps, batch):
                trace["versions"] = self._step_tensor(
                    versions,
                    steps=steps,
                    batch=batch,
                )
        return trace

    def write(self, trajectories: list[Trajectory]) -> int:
        """Write all trace-bearing trajectories and return the file count."""
        written = 0
        for trajectory in trajectories:
            if 0 < self.max_trace_files <= self._counter:
                break
            trace = self._build_trace(trajectory)
            if trace is None:
                continue
            filename = f"trace_{self._run_id}_{self._counter:08d}.pt"
            output_path = self.output_dir / filename
            temporary_path = output_path.with_suffix(".pt.tmp")
            torch.save(trace, temporary_path)
            os.replace(temporary_path, output_path)
            self._counter += 1
            written += 1
        return written


__all__ = ["RLT_GATE_TRACE_KEYS", "RLTGateTraceWriter"]
