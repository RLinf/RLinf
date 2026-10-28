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

from dataclasses import dataclass
from typing import Any

import torch

from rlinf.algorithms.rlt.rlt_steam_gate_trace import RLTGateTraceWriter
from rlinf.algorithms.rlt.rlt_steam_phase_head import RLT_PHASE_FEATURE_KEY
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    collect_trajectory_replay_metrics,
    trajectory_has_bool_tensor,
)


@dataclass(frozen=True)
class ReplayIngestResult:
    """Counts and metrics produced by one replay-ingestion batch."""

    transitions: int
    completed_episodes: int
    metrics: dict[str, float]


class RLTReplayService:
    """Convert RLT rollout trajectories into learner replay entries."""

    def __init__(self, cfg: Any, *, rank: int) -> None:
        self.cfg = cfg
        trace_cfg = cfg.algorithm.get("rlt_gate_calibration", {}) or {}
        self.trace_writer = (
            RLTGateTraceWriter(trace_cfg, rank=rank)
            if bool(trace_cfg.get("enable", False))
            else None
        )
        self.replay_buffer = None
        self.demo_buffer = None

    def bind_buffers(self, replay_buffer: Any, demo_buffer: Any | None) -> None:
        """Bind worker-owned buffers after SAC runtime initialization."""
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer

    def ingest(self, trajectories: list[Trajectory]) -> ReplayIngestResult:
        """Add rollout data to replay using the configured RLT storage mode."""
        if self.replay_buffer is None:
            raise RuntimeError("RLT replay buffers must be bound before ingestion.")
        if self.trace_writer is not None:
            self.trace_writer.write(trajectories)

        if use_simulator_transition_replay(self.cfg):
            replay_trajectories: list[Trajectory] = []
            completed_episodes = 0
            for trajectory in trajectories:
                transition_trajectories, completed = (
                    self._transition_replay_trajectories(trajectory)
                )
                replay_trajectories.extend(transition_trajectories)
                completed_episodes += completed
            metrics = {
                **self._transition_replay_metrics(replay_trajectories),
                **collect_trajectory_replay_metrics(
                    trajectories,
                    reducer=all_reduce_dict,
                ),
            }
            self.replay_buffer.add_trajectories(replay_trajectories)
            self._add_transition_demonstrations(replay_trajectories)
            return ReplayIngestResult(
                transitions=len(replay_trajectories),
                completed_episodes=completed_episodes,
                metrics=metrics,
            )

        self.replay_buffer.add_trajectories(trajectories)
        self._add_intervention_demonstrations(trajectories)
        transitions = sum(
            self._trajectory_transition_count(trajectory) for trajectory in trajectories
        )
        completed_episodes = sum(
            self._trajectory_completed_episodes(trajectory)
            for trajectory in trajectories
        )
        metrics = collect_trajectory_replay_metrics(
            trajectories,
            reducer=all_reduce_dict,
        )
        return ReplayIngestResult(transitions, completed_episodes, metrics)

    def _add_transition_demonstrations(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        if self.demo_buffer is None:
            return
        demonstrations = [
            trajectory
            for trajectory in trajectories
            if trajectory_has_bool_tensor(trajectory.intervene_flags)
        ]
        if demonstrations:
            self.demo_buffer.add_trajectories(demonstrations)

    def _add_intervention_demonstrations(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        if self.demo_buffer is None:
            return
        demonstrations = []
        for trajectory in trajectories:
            intervention_trajectories = trajectory.extract_intervene_traj()
            if intervention_trajectories is not None:
                demonstrations.extend(intervention_trajectories)
        if demonstrations:
            self.demo_buffer.add_trajectories(demonstrations)

    @staticmethod
    def _trajectory_transition_count(trajectory: Trajectory) -> int:
        if trajectory.actions is None:
            return 0
        return int(trajectory.actions.shape[0] * trajectory.actions.shape[1])

    @staticmethod
    def _trajectory_completed_episodes(trajectory: Trajectory) -> int:
        if trajectory.dones is None:
            return 0
        dones = trajectory.dones
        return int(dones.reshape(dones.shape[0], dones.shape[1], -1).any(dim=-1).sum())

    @staticmethod
    def _transition_reward_value(trajectory: Trajectory) -> float | None:
        rewards = trajectory.rewards
        if not isinstance(rewards, torch.Tensor) or rewards.numel() == 0:
            return None
        return float(rewards.detach().float().reshape(-1).sum().item())

    @staticmethod
    def _transition_done_value(trajectory: Trajectory) -> bool | None:
        dones = trajectory.dones
        if not isinstance(dones, torch.Tensor) or dones.numel() == 0:
            return None
        return bool(dones.detach().to(torch.bool).reshape(-1).any().item())

    @staticmethod
    def _row_tensor(tensor: torch.Tensor, index: int) -> torch.Tensor:
        return (
            tensor[index].detach().clone().unsqueeze(0).unsqueeze(0).cpu().contiguous()
        )

    @staticmethod
    def _step_env_tensor(
        tensor: torch.Tensor,
        step_index: int,
        env_index: int,
    ) -> torch.Tensor:
        return (
            tensor[step_index, env_index]
            .detach()
            .clone()
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
            .contiguous()
        )

    def _row_tensor_dict(
        self,
        tensor_dict: dict[str, object],
        index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            key: self._row_tensor(value, index)
            for key, value in tensor_dict.items()
            if isinstance(value, torch.Tensor) and index < value.shape[0]
        }

    def _prefix_obs_from_flat_dict(
        self,
        flat: dict[str, Any],
        dict_key: str,
        index: int,
    ) -> dict[str, torch.Tensor] | None:
        value = flat.get(dict_key)
        if not isinstance(value, dict):
            return None
        obs = self._row_tensor_dict(value, index)
        return obs if obs else None

    @staticmethod
    def _flat_record_transition(flat: dict[str, Any], index: int) -> bool:
        forward_inputs = flat.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return False
        record_transition = forward_inputs.get("record_transition")
        if not isinstance(record_transition, torch.Tensor):
            return False
        if index >= record_transition.shape[0]:
            return False
        return bool(record_transition[index].detach().to(torch.bool).reshape(-1).all())

    def _transition_replay_trajectories(
        self,
        trajectory: Trajectory,
    ) -> tuple[list[Trajectory], int]:
        if (
            trajectory.actions is None
            or trajectory.rewards is None
            or self.replay_buffer is None
        ):
            return [], 0

        flat = self.replay_buffer._flatten_trajectory(trajectory)
        actions = flat.get("actions")
        rewards = flat.get("rewards")
        if not isinstance(actions, torch.Tensor) or not isinstance(
            rewards,
            torch.Tensor,
        ):
            return [], 0

        tensor_fields = (
            "actions",
            "intervene_flags",
            "rewards",
            "terminations",
            "truncations",
            "dones",
            "prev_logprobs",
            "prev_values",
            "versions",
        )
        replay_trajectories = []
        completed_episodes = 0
        trajectory_length = int(trajectory.actions.shape[0])
        batch_size = int(trajectory.actions.shape[1])
        num_rows = int(actions.shape[0])
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_index in range(batch_size):
            for step_index in range(trajectory_length):
                index = step_index * batch_size + env_index
                if index >= num_rows:
                    break
                if not self._flat_record_transition(flat, index):
                    continue

                transition = Trajectory(
                    max_episode_length=1,
                    model_weights_id=trajectory.model_weights_id,
                )
                for field_name in tensor_fields:
                    value = flat.get(field_name)
                    if isinstance(value, torch.Tensor) and index < value.shape[0]:
                        setattr(
                            transition,
                            field_name,
                            self._row_tensor(value, index),
                        )
                forward_inputs = flat.get("forward_inputs")
                if isinstance(forward_inputs, dict):
                    row_value = self._row_tensor_dict(forward_inputs, index)
                    row_value.pop(RLT_PHASE_FEATURE_KEY, None)
                    transition.forward_inputs = row_value

                curr_obs = self._prefix_obs_from_flat_dict(flat, "curr_obs", index)
                if curr_obs is None:
                    continue
                transition.curr_obs = curr_obs

                done_index = min(
                    step_index + 1,
                    int(trajectory.dones.shape[0]) - 1
                    if isinstance(trajectory.dones, torch.Tensor)
                    else trajectory_length - 1,
                )
                for done_field in ("dones", "terminations", "truncations"):
                    done_value = getattr(trajectory, done_field, None)
                    if (
                        isinstance(done_value, torch.Tensor)
                        and done_index < done_value.shape[0]
                        and env_index < done_value.shape[1]
                    ):
                        setattr(
                            transition,
                            done_field,
                            self._step_env_tensor(
                                done_value,
                                done_index,
                                env_index,
                            ),
                        )

                is_done = bool(
                    isinstance(transition.dones, torch.Tensor)
                    and transition.dones.reshape(-1).to(torch.bool).any()
                )
                next_obs = (
                    curr_obs
                    if is_done
                    else self._prefix_obs_from_flat_dict(flat, "next_obs", index)
                )
                if next_obs is None:
                    raise ValueError(
                        "RLT transition replay requires next_obs for non-terminal "
                        f"transitions, got row index {index}."
                    )
                transition.next_obs = next_obs

                replay_trajectories.append(transition)
                if is_done:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes

    def _transition_replay_metrics(
        self,
        trajectories: list[Trajectory],
    ) -> dict[str, float]:
        metrics = {"replay/transition_count": float(len(trajectories))}
        reward_values = [
            reward
            for trajectory in trajectories
            if (reward := self._transition_reward_value(trajectory)) is not None
        ]
        if reward_values:
            metrics["replay/reward_mean"] = float(
                sum(reward_values) / len(reward_values)
            )
            metrics["replay/reward_positive_rate"] = float(
                sum(reward > 0.0 for reward in reward_values) / len(reward_values)
            )
        done_values = [
            done
            for trajectory in trajectories
            if (done := self._transition_done_value(trajectory)) is not None
        ]
        if done_values:
            metrics["replay/done_rate"] = float(
                sum(bool(done) for done in done_values) / len(done_values)
            )
        return metrics
