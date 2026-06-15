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

"""Convert RLT Stage 2 rollout trajectories into replay transitions."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory

from .replay_buffer import RLTStage2ReplayBuffer
from .transition import TransitionSource, resolve_chunk_source


class RLTStage2TrajectoryReplayAdapter:
    """Builds replay-buffer transitions from rollout trajectories."""

    def __init__(
        self,
        cfg: DictConfig,
        replay_buffer: RLTStage2ReplayBuffer | None,
    ) -> None:
        self.cfg = cfg
        self.replay_buffer = replay_buffer

    @staticmethod
    def _to_numpy_float(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.uint8, copy=False)

    def add_trajectory(self, traj: Trajectory) -> tuple[int, int]:
        """Adds trajectory transitions and returns transition/episode counts."""

        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        if stride > 0:
            if not traj.rlt_step_trace:
                raise RuntimeError(
                    "RLT Stage2 stride replay is enabled but trajectory has no "
                    "rlt_step_trace. Refusing to fall back to chunk-boundary replay."
                )
            return self._step_trace_to_transitions(traj)
        return self._chunk_trajectory_to_transitions(traj)

    def _step_trace_to_transitions(self, traj: Trajectory) -> tuple[int, int]:
        if (
            self.replay_buffer is None
            or traj.actions is None
            or traj.rewards is None
            or traj.dones is None
            or not traj.rlt_step_trace
        ):
            return 0, 0

        x_boundary = traj.forward_inputs.get("x") if traj.forward_inputs else None
        a_tilde_boundary = (
            traj.forward_inputs.get("a_tilde") if traj.forward_inputs else None
        )
        if x_boundary is None or a_tilde_boundary is None:
            raise RuntimeError(
                "RLT Stage2 stride replay requires chunk-boundary "
                "forward_inputs['x'] and forward_inputs['a_tilde']; rollout must "
                "cache policy-call features instead of forcing actor-side VLA encoding."
            )

        anchor_offsets = traj.rlt_step_trace.get("anchor_offsets")
        x_trace = traj.rlt_step_trace.get("x")
        a_tilde_trace = traj.rlt_step_trace.get("a_tilde")
        if anchor_offsets is None:
            raise RuntimeError(
                "RLT Stage2 stride replay requires sparse "
                "rlt_step_trace['anchor_offsets']; refusing to fall back to "
                "chunk-boundary replay."
            )

        chunk_steps = int(traj.actions.shape[0])
        bsz = int(traj.actions.shape[1])
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        allow_terminal_partial = bool(
            self.cfg.actor.model.rlt_stage2.get("replay_allow_terminal_partial", True)
        )
        if stride <= 0:
            return 0, 0
        if x_boundary.shape[0] < chunk_steps + 1:
            raise ValueError(
                "RLT stride replay requires one extra final chunk-boundary feature "
                f"for bootstrapping: expected at least {chunk_steps + 1}, got "
                f"{x_boundary.shape[0]}."
            )
        if x_boundary.shape[1] != bsz or a_tilde_boundary.shape[1] != bsz:
            raise ValueError(
                "RLT boundary feature batch mismatch: "
                f"{x_boundary.shape=}, {a_tilde_boundary.shape=}, expected B={bsz}."
            )
        if traj.rewards.shape[0] != chunk_steps:
            raise ValueError(
                "RLT step trace/reward length mismatch: "
                f"{chunk_steps=} but traj.rewards.shape[0]={traj.rewards.shape[0]}."
            )
        if anchor_offsets.dim() != 3:
            raise ValueError(
                "RLT sparse anchor_offsets must have shape [chunk_steps, A, B], "
                f"got {anchor_offsets.shape}."
            )
        if anchor_offsets.shape[0] != chunk_steps or anchor_offsets.shape[2] != bsz:
            raise ValueError(
                "RLT sparse anchor_offsets/action shape mismatch: "
                f"{anchor_offsets.shape=}, expected chunk_steps={chunk_steps}, B={bsz}."
            )
        feature_steps = int(anchor_offsets.shape[1])
        if feature_steps > 0:
            if x_trace is None or a_tilde_trace is None:
                raise RuntimeError(
                    "RLT sparse anchor trace has non-boundary offsets but is missing "
                    "rlt_step_trace['x'] or rlt_step_trace['a_tilde']."
                )
            if (
                x_trace.shape[0] != chunk_steps
                or x_trace.shape[1] != feature_steps
                or x_trace.shape[2] != bsz
                or a_tilde_trace.shape[0] != chunk_steps
                or a_tilde_trace.shape[1] != feature_steps
                or a_tilde_trace.shape[2] != bsz
            ):
                raise ValueError(
                    "RLT sparse anchor feature shape mismatch: "
                    f"{x_trace.shape=}, {a_tilde_trace.shape=}, "
                    f"{anchor_offsets.shape=}."
                )

        flat_actions = traj.actions.reshape(chunk_steps, bsz, chunk_len, action_dim)
        flat_rewards = traj.rewards.reshape(chunk_steps, bsz, chunk_len)
        dones_all = traj.dones
        if dones_all.shape[0] == chunk_steps + 1:
            # EmbodiedRolloutResult stores an initial bootstrap done frame.
            dones_all = dones_all[1:]
        if dones_all.shape[0] != chunk_steps:
            raise ValueError(
                "RLT step trace/done length mismatch: expected dones to have "
                f"{chunk_steps} or {chunk_steps + 1} chunk steps, got "
                f"{traj.dones.shape[0]}."
            )
        flat_dones = dones_all.reshape(chunk_steps, bsz, chunk_len)
        intervention_flags_all = traj.intervene_flags
        if intervention_flags_all is None:
            flat_interventions = torch.zeros_like(flat_rewards, dtype=torch.bool)
        else:
            if intervention_flags_all.shape[0] != chunk_steps:
                raise ValueError(
                    "RLT intervention/action length mismatch: "
                    f"expected {chunk_steps}, got {intervention_flags_all.shape[0]}."
                )
            flat_interventions = intervention_flags_all.reshape(
                chunk_steps,
                bsz,
                chunk_len,
                -1,
            ).any(dim=-1)
        source_chunk_all = (
            traj.forward_inputs.get("source_chunk") if traj.forward_inputs else None
        )
        if source_chunk_all is None:
            flat_sources = torch.where(
                flat_interventions,
                torch.full_like(
                    flat_interventions,
                    int(TransitionSource.HUMAN),
                    dtype=torch.uint8,
                ),
                torch.full_like(
                    flat_interventions,
                    int(TransitionSource.RL),
                    dtype=torch.uint8,
                ),
            )
        else:
            if source_chunk_all.shape[0] < chunk_steps:
                raise ValueError(
                    "RLT source_chunk/action length mismatch: "
                    f"expected at least {chunk_steps}, got {source_chunk_all.shape[0]}."
                )
            flat_sources = source_chunk_all[:chunk_steps].reshape(
                chunk_steps,
                bsz,
                chunk_len,
            ).to(torch.uint8)
        collection_phase_id_all = (
            traj.forward_inputs.get("collection_phase_id")
            if traj.forward_inputs
            else None
        )
        record_transition_all = (
            traj.forward_inputs.get("record_transition") if traj.forward_inputs else None
        )
        if record_transition_all is None:
            flat_record_transitions = torch.ones_like(flat_rewards, dtype=torch.bool)
        else:
            if record_transition_all.shape[0] < chunk_steps:
                raise ValueError(
                    "RLT record_transition/action length mismatch: "
                    f"expected at least {chunk_steps}, got "
                    f"{record_transition_all.shape[0]}."
                )
            record_transition_all = record_transition_all[:chunk_steps]
            if record_transition_all.dim() <= 2:
                flat_record_transitions = record_transition_all.reshape(
                    chunk_steps,
                    bsz,
                    1,
                ).expand(-1, -1, chunk_len)
            elif (
                record_transition_all.dim() == 3
                and record_transition_all.shape[2] == 1
            ):
                flat_record_transitions = record_transition_all.expand(
                    -1,
                    -1,
                    chunk_len,
                )
            elif (
                record_transition_all.dim() == 3
                and record_transition_all.shape[2] == chunk_len
            ):
                flat_record_transitions = record_transition_all
            else:
                flat_record_transitions = record_transition_all.reshape(
                    chunk_steps,
                    bsz,
                    chunk_len,
                    -1,
                ).any(dim=-1)
            flat_record_transitions = flat_record_transitions.to(torch.bool)

        added = 0
        completed_episodes = 0
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))
        total_control_steps = chunk_steps * chunk_len

        def get_feature(
            global_step: int,
            env_idx: int,
            *,
            terminal_fallback: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if global_step < 0 or global_step > total_control_steps:
                raise RuntimeError(
                    f"RLT feature lookup out of range: {global_step=} "
                    f"with {total_control_steps=}."
                )

            if global_step % chunk_len == 0:
                boundary_idx = global_step // chunk_len
                if boundary_idx < x_boundary.shape[0]:
                    return (
                        x_boundary[boundary_idx, env_idx],
                        a_tilde_boundary[boundary_idx, env_idx],
                    )
                if terminal_fallback is not None:
                    return terminal_fallback
                raise RuntimeError(
                    "Missing RLT chunk-boundary feature for non-terminal stride "
                    f"window end: {global_step=}, {boundary_idx=}."
                )

            chunk_idx = global_step // chunk_len
            offset = global_step % chunk_len
            if chunk_idx >= chunk_steps:
                if terminal_fallback is not None:
                    return terminal_fallback
                raise RuntimeError(
                    "Missing RLT sparse feature beyond rollout chunk range: "
                    f"{global_step=}, {chunk_idx=}."
                )
            if feature_steps > 0 and x_trace is not None and a_tilde_trace is not None:
                env_offsets = anchor_offsets[chunk_idx, :, env_idx].to(torch.long)
                match = torch.nonzero(env_offsets == offset, as_tuple=False).reshape(
                    -1
                )
                if match.numel() > 0:
                    pos = int(match[0].item())
                    return (
                        x_trace[chunk_idx, pos, env_idx],
                        a_tilde_trace[chunk_idx, pos, env_idx],
                    )
            if terminal_fallback is not None:
                return terminal_fallback
            raise RuntimeError(
                "Missing RLT sparse anchor feature for non-terminal stride window: "
                f"{global_step=}, {chunk_idx=}, {offset=}, {stride=}, "
                f"{anchor_offsets.shape=}."
            )

        for env_idx in range(bsz):
            env_actions = flat_actions[:, env_idx].reshape(
                total_control_steps,
                action_dim,
            )
            env_rewards = flat_rewards[:, env_idx].reshape(total_control_steps)
            env_dones = flat_dones[:, env_idx].reshape(total_control_steps)
            env_interventions = flat_interventions[:, env_idx].reshape(
                total_control_steps
            )
            env_sources = flat_sources[:, env_idx].reshape(total_control_steps)
            env_record_transitions = flat_record_transitions[:, env_idx].reshape(
                total_control_steps
            )
            done_indices = [
                int(idx.item())
                for idx in torch.nonzero(env_dones, as_tuple=False).reshape(-1)
            ]
            segment_start = 0
            stop_env = False

            def add_windows_for_segment(
                segment_start_idx: int,
                segment_end_idx: int,
                *,
                segment_terminal: bool,
            ) -> None:
                nonlocal added
                if segment_end_idx <= segment_start_idx:
                    return

                for start in range(segment_start_idx, segment_end_idx, stride):
                    end = start + chunk_len
                    valid_end = min(end, segment_end_idx)
                    terminal = bool(
                        segment_terminal and valid_end == segment_end_idx
                    )
                    is_partial = end > segment_end_idx
                    if is_partial and (not terminal or not allow_terminal_partial):
                        # Only terminal partial windows are valid; padding a live
                        # rollout boundary would fabricate future actions/rewards.
                        continue
                    if not bool(env_record_transitions[start:valid_end].all().item()):
                        continue

                    x_tensor, a_tilde_tensor = get_feature(start, env_idx)
                    next_x_tensor, next_a_tilde_tensor = get_feature(
                        valid_end,
                        env_idx,
                        terminal_fallback=(
                            (x_tensor, a_tilde_tensor) if terminal else None
                        ),
                    )

                    x = self._to_numpy_float(x_tensor)
                    a_tilde = self._to_numpy_float(a_tilde_tensor)
                    next_x = self._to_numpy_float(next_x_tensor)
                    next_a_tilde = self._to_numpy_float(next_a_tilde_tensor)

                    valid_len = valid_end - start
                    action_chunk = torch.zeros(
                        chunk_len,
                        action_dim,
                        dtype=env_actions.dtype,
                        device=env_actions.device,
                    )
                    reward_chunk = torch.zeros(
                        chunk_len,
                        dtype=env_rewards.dtype,
                        device=env_rewards.device,
                    )
                    intervention_chunk = torch.zeros(
                        chunk_len,
                        dtype=env_interventions.dtype,
                        device=env_interventions.device,
                    )
                    source_chunk = torch.full(
                        (chunk_len,),
                        int(TransitionSource.BASE),
                        dtype=torch.uint8,
                        device=env_sources.device,
                    )
                    action_chunk[:valid_len] = env_actions[start:valid_end]
                    reward_chunk[:valid_len] = env_rewards[start:valid_end]
                    intervention_chunk[:valid_len] = env_interventions[start:valid_end]
                    source_chunk[:valid_len] = env_sources[start:valid_end]

                    action_np = self._to_numpy_float(action_chunk).reshape(-1)
                    rewards_np = self._to_numpy_float(reward_chunk)
                    intervention_np = self._to_numpy_float(intervention_chunk)
                    source_chunk_np = self._to_numpy_uint8(source_chunk)
                    collection_phase_id = None
                    if collection_phase_id_all is not None:
                        phase_idx = min(
                            start // chunk_len,
                            collection_phase_id_all.shape[0] - 1,
                        )
                        collection_phase_id = int(
                            collection_phase_id_all[phase_idx, env_idx]
                            .reshape(-1)[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                    self.replay_buffer.add(
                        x=x,
                        action_chunk=action_np,
                        ref_chunk=a_tilde,
                        rewards=rewards_np,
                        next_x=next_x,
                        next_ref_chunk=next_a_tilde,
                        done=float(terminal),
                        intervention=intervention_np,
                        source=resolve_chunk_source(source_chunk_np),
                        source_chunk=source_chunk_np,
                        collection_phase=collection_phase_id,
                        intervention_flag=bool(intervention_np.any()),
                        step_id=start,
                    )
                    added += 1

                    if terminal and not auto_reset:
                        break

            for done_idx in done_indices:
                episode_end = done_idx + 1
                if episode_end <= segment_start:
                    continue
                add_windows_for_segment(
                    segment_start,
                    episode_end,
                    segment_terminal=True,
                )
                completed_episodes += 1
                if not auto_reset:
                    stop_env = True
                    break
                # A clean new episode is not available until the next action chunk
                # boundary. Do not treat the post-done tail as replay data.
                segment_start = min(
                    total_control_steps,
                    ((done_idx // chunk_len) + 1) * chunk_len,
                )

            if not stop_env and segment_start < total_control_steps:
                add_windows_for_segment(
                    segment_start,
                    total_control_steps,
                    segment_terminal=False,
                )

        return added, completed_episodes

    def _chunk_trajectory_to_transitions(self, traj: Trajectory) -> tuple[int, int]:
        if self.replay_buffer is None:
            return 0, 0
        if traj.actions is None:
            raise RuntimeError("RLT Stage2 replay requires trajectory actions.")
        if not traj.forward_inputs:
            raise RuntimeError(
                "RLT Stage2 replay requires trajectory.forward_inputs; rollout "
                "must emit the canonical RLT forward-input contract."
            )

        traj_len = traj.actions.shape[0]
        bsz = traj.actions.shape[1]
        added = 0
        completed_episodes = 0

        x_all = traj.forward_inputs.get("x")
        a_tilde_all = traj.forward_inputs.get("a_tilde")
        if x_all is None or a_tilde_all is None:
            raise RuntimeError(
                "RLT Stage2 replay requires forward_inputs['x'] and "
                "forward_inputs['a_tilde']."
            )

        dones_all = traj.dones
        rewards_all = traj.rewards
        if dones_all is None or rewards_all is None:
            raise RuntimeError("RLT Stage2 replay requires trajectory rewards and dones.")
        intervention_flags_all = traj.forward_inputs.get("intervention_flags")
        if intervention_flags_all is None:
            intervention_flags_all = traj.intervene_flags
        source_chunk_all = traj.forward_inputs.get("source_chunk")
        collection_phase_id_all = traj.forward_inputs.get("collection_phase_id")
        record_transition_all = traj.forward_inputs.get("record_transition")
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
                if record_transition_all is not None:
                    record_transition = (
                        record_transition_all[t, env_idx]
                        .detach()
                        .to(torch.bool)
                        .reshape(-1)
                    )
                    if not bool(record_transition.all().item()):
                        continue
                done_idx = min(t + 1, dones_all.shape[0] - 1)
                env_done = float(dones_all[done_idx, env_idx].any().item())
                done = float(env_done > 0.0)
                intervention_mask: float | np.ndarray = 0.0
                if intervention_flags_all is not None:
                    intervention_mask = (
                        intervention_flags_all[t, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                        .reshape(-1)
                    )
                source_chunk: np.ndarray | None = None
                source: int | None = None
                if source_chunk_all is not None:
                    source_chunk = (
                        source_chunk_all[t, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8, copy=False)
                        .reshape(-1)
                    )
                    source = resolve_chunk_source(source_chunk)
                collection_phase_id = None
                if collection_phase_id_all is not None:
                    collection_phase_id = int(
                        collection_phase_id_all[t, env_idx]
                        .reshape(-1)[0]
                        .detach()
                        .cpu()
                        .item()
                    )

                x = self._to_numpy_float(x_all[t, env_idx])
                a_tilde = self._to_numpy_float(a_tilde_all[t, env_idx])
                action = self._to_numpy_float(traj.actions[t, env_idx])
                rewards = self._to_numpy_float(rewards_all[t, env_idx])

                if done > 0.0:
                    next_x = x
                    next_a_tilde = a_tilde
                elif t + 1 < traj_len:
                    next_x = self._to_numpy_float(x_all[t + 1, env_idx])
                    next_a_tilde = self._to_numpy_float(a_tilde_all[t + 1, env_idx])
                else:
                    if x_all.shape[0] <= t + 1 or a_tilde_all.shape[0] <= t + 1:
                        raise RuntimeError(
                            "RLT Stage2 rollout boundary transition is non-terminal "
                            "but missing cached final x/a_tilde. Rollout must send "
                            "the final student forward_inputs so actor training can "
                            "bootstrap without re-encoding VLA observations."
                        )
                    next_x = self._to_numpy_float(x_all[t + 1, env_idx])
                    next_a_tilde = self._to_numpy_float(a_tilde_all[t + 1, env_idx])

                self.replay_buffer.add(
                    x=x,
                    action_chunk=action,
                    ref_chunk=a_tilde,
                    rewards=rewards,
                    next_x=next_x,
                    next_ref_chunk=next_a_tilde,
                    done=done,
                    intervention=intervention_mask,
                    source=source,
                    source_chunk=source_chunk,
                    collection_phase=collection_phase_id,
                    intervention_flag=bool(np.asarray(intervention_mask).any()),
                    step_id=t,
                )
                added += 1
                if env_done > 0.0:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return added, completed_episodes
