"""Native RLinf actor worker for RLT stage2 TD3 training."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory

from ...models.embodiment.rlt_stage2.components import actor_loss, critic_loss
from ...models.embodiment.rlt_stage2.replay_buffer import RLTStage2ReplayBuffer
from ...models.embodiment.rlt_stage2.transition import (
    TransitionSource,
    resolve_chunk_source,
)


class RLTStage2FSDPPolicyWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.version = 0

        self.replay_buffer: RLTStage2ReplayBuffer | None = None
        self.demo_buffer: RLTStage2ReplayBuffer | None = None
        self.qf_optimizer = None
        self.qf_lr_scheduler = None
        self.update_step = 0
        self.pending_update_budget = 0
        self.warmup_ready_total_transitions: int | None = None
        self.warmup_ready_total_episodes: int | None = None
        self.gradient_accumulation = 1
        self.actor_only_train_model = bool(
            cfg.algorithm.get("actor_only_train_model", True)
        )
        self._rollout_sync_key_count = 0
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0

        weight_syncer_cfg = cfg.get("weight_syncer", None)
        assert weight_syncer_cfg is not None, (
            "weight_syncer config must be provided for RLT stage2 actor worker."
        )
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
        self._sync_weight_comm_options = self.weight_syncer.comm_options
        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def _warmup_required_updates(self) -> int:
        td3_bc_cfg = self.cfg.algorithm.get("td3_bc", {})
        warmup_required_updates = int(
            td3_bc_cfg.get(
                "warmup_updates",
                self.cfg.algorithm.get("warmup_post_collect_updates", 0),
            )
        )
        if warmup_required_updates < 0:
            raise ValueError(
                "algorithm.td3_bc.warmup_updates must be >= 0, "
                f"got {warmup_required_updates}."
            )
        return warmup_required_updates

    def _resolve_actor_loss_weights(self) -> tuple[float, float, float, bool, float]:
        td3_bc_cfg = self.cfg.algorithm.get("td3_bc", {})
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        loss_warmup_updates = int(
            td3_bc_cfg.get(
                "actor_loss_warmup_updates",
                self.cfg.algorithm.get("actor_loss_warmup_updates", 0),
            )
        )
        in_warmup = self.update_step < loss_warmup_updates
        warmup_bc_weight = float(
            td3_bc_cfg.get(
                "warmup_bc_weight",
                stage2_cfg.get(
                    "warmup_bc_weight",
                    stage2_cfg.get("bc_regularizer_beta", 1.0),
                ),
            )
        )
        warmup_q_weight = float(
            td3_bc_cfg.get(
                "warmup_q_weight",
                stage2_cfg.get("warmup_q_weight", 0.1),
            )
        )
        online_bc_weight = float(
            td3_bc_cfg.get(
                "online_bc_weight",
                stage2_cfg.get(
                    "online_bc_weight",
                    stage2_cfg.get("bc_regularizer_beta", 1.0),
                ),
            )
        )
        online_q_weight = float(
            td3_bc_cfg.get(
                "online_q_weight",
                stage2_cfg.get("online_q_weight", 0.1),
            )
        )
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        else:
            ramp_updates = int(
                td3_bc_cfg.get(
                    "actor_loss_ramp_updates",
                    self.cfg.algorithm.get("actor_loss_ramp_updates", 0),
                )
            )
            if ramp_updates > 0:
                ramp_progress = min(
                    1.0,
                    max(
                        0.0,
                        float(self.update_step - loss_warmup_updates + 1)
                        / float(ramp_updates),
                    ),
                )
            else:
                ramp_progress = 1.0
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        delta_weight = float(
            td3_bc_cfg.get("delta_weight", stage2_cfg.get("delta_weight", 0.0))
        )
        return bc_weight, q_weight, delta_weight, in_warmup, ramp_progress

    def init_worker(self):
        self.setup_model_and_optimizer()
        self._init_replay_buffer()
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> torch.nn.Module:
        from rlinf.models import get_model

        model_cfg = self.cfg.actor.model
        if bool(
            self.actor_only_train_model
        ) and model_cfg.get("rlt_stage2", None) is not None:
            from copy import deepcopy
            from omegaconf import open_dict

            model_cfg = deepcopy(model_cfg)
            with open_dict(model_cfg):
                model_cfg.rlt_stage2.load_feature_backbones = False
                model_cfg.rlt_stage2.load_rl_token_model = False

        model = get_model(model_cfg)
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path, map_location="cpu")
            model.load_state_dict(model_dict)
        return model

    def setup_model_and_optimizer(self) -> None:
        module = self.model_provider_func()
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        actor_filters = {"critic": ["critic."]}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=actor_filters,
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        grad_scaler_cfg = self.cfg.actor.fsdp_config.grad_scaler
        kwargs = {}
        for key in ["init_scale", "growth_interval"]:
            value = grad_scaler_cfg.get(key, None)
            if value is not None:
                kwargs[key] = value
        self.grad_scaler = self.build_grad_scaler(
            grad_scaler_cfg.get("enabled", False),
            **kwargs,
        )

    def _init_replay_buffer(self) -> None:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        capacity = int(
            replay_cfg.get("capacity", stage2_cfg.get("buffer_capacity", 200000))
        )
        self.replay_buffer = RLTStage2ReplayBuffer(
            capacity=capacity,
            state_dim=int(self.cfg.actor.model.rlt_stage2.embedding_dim)
            + int(self.cfg.actor.model.rlt_stage2.proprio_dim),
            action_chunk_dim=int(self.cfg.actor.model.num_action_chunks)
            * int(self.cfg.actor.model.action_dim),
            chunk_length=int(self.cfg.actor.model.num_action_chunks),
            seed=int(self.cfg.actor.get("seed", 1234)) + self._rank,
        )
        demo_cfg = self.cfg.algorithm.get("demo_buffer", None)
        if demo_cfg is None:
            return
        self.demo_buffer = RLTStage2ReplayBuffer(
            capacity=int(demo_cfg.get("capacity", capacity)),
            state_dim=int(self.cfg.actor.model.rlt_stage2.embedding_dim)
            + int(self.cfg.actor.model.rlt_stage2.proprio_dim),
            action_chunk_dim=int(self.cfg.actor.model.num_action_chunks)
            * int(self.cfg.actor.model.action_dim),
            chunk_length=int(self.cfg.actor.model.num_action_chunks),
            seed=int(self.cfg.actor.get("seed", 1234)) + self._rank + 17,
        )
        if demo_cfg.get("load_path", None) is not None:
            self.demo_buffer.load_checkpoint(demo_cfg.load_path)

    def soft_update_target_model(self, tau: float | None = None) -> None:
        if tau is None:
            tau = float(self.cfg.algorithm.tau)
        self.model.update_target_networks(float(tau))

    def get_rollout_state_dict(self) -> dict:
        state_dict = self.model.filter_rollout_state_dict(
            self.get_model_state_dict(cpu_offload=False, full_state_dict=False)
        )
        self._rollout_sync_key_count = len(state_dict)
        return state_dict

    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=list(state_dict.keys()),
            )

        # Rollout uses this version as the number of completed TD3 updates.
        # Do not send runner global_step here, otherwise warmup/intervention
        # gates open before any actor/critic update has actually happened.
        await self.weight_syncer.sync(state_dict, send_func, version=self.update_step)

        if self.enable_offload:
            self.offload_param_and_grad(True)

    @staticmethod
    def _to_numpy_float(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.uint8, copy=False)

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
                torch.full_like(flat_interventions, int(TransitionSource.HUMAN), dtype=torch.uint8),
                torch.full_like(flat_interventions, int(TransitionSource.RL), dtype=torch.uint8),
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
            traj.forward_inputs.get("collection_phase_id") if traj.forward_inputs else None
        )

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
                match = torch.nonzero(env_offsets == offset, as_tuple=False).reshape(-1)
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
            env_actions = flat_actions[:, env_idx].reshape(total_control_steps, action_dim)
            env_rewards = flat_rewards[:, env_idx].reshape(total_control_steps)
            env_dones = flat_dones[:, env_idx].reshape(total_control_steps)
            env_interventions = flat_interventions[:, env_idx].reshape(total_control_steps)
            env_sources = flat_sources[:, env_idx].reshape(total_control_steps)
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
                        phase_idx = min(start // chunk_len, collection_phase_id_all.shape[0] - 1)
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
        if self.replay_buffer is None or traj.actions is None or not traj.forward_inputs:
            return 0, 0

        traj_len = traj.actions.shape[0]
        bsz = traj.actions.shape[1]
        added = 0
        completed_episodes = 0

        x_all = traj.forward_inputs.get("x")
        a_tilde_all = traj.forward_inputs.get("a_tilde")
        if x_all is None or a_tilde_all is None:
            return 0, 0

        dones_all = traj.dones
        rewards_all = traj.rewards
        if dones_all is None or rewards_all is None:
            return 0, 0
        intervention_flags_all = traj.forward_inputs.get("intervention_flags")
        if intervention_flags_all is None:
            intervention_flags_all = traj.intervene_flags
        source_chunk_all = traj.forward_inputs.get("source_chunk")
        collection_phase_id_all = traj.forward_inputs.get("collection_phase_id")
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
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

                x = x_all[t, env_idx].detach().cpu().numpy().astype(np.float32, copy=False)
                a_tilde = (
                    a_tilde_all[t, env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                action = (
                    traj.actions[t, env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                rewards = (
                    rewards_all[t, env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                if done > 0.0:
                    next_x = x
                    next_a_tilde = a_tilde
                elif t + 1 < traj_len:
                    next_x = (
                        x_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    next_a_tilde = (
                        a_tilde_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                else:
                    if x_all.shape[0] <= t + 1 or a_tilde_all.shape[0] <= t + 1:
                        raise RuntimeError(
                            "RLT Stage2 rollout boundary transition is non-terminal "
                            "but missing cached final x/a_tilde. Rollout must send "
                            "the final student forward_inputs so actor training can "
                            "bootstrap without re-encoding VLA observations."
                        )
                    next_x = (
                        x_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    next_a_tilde = (
                        a_tilde_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

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

    def _trajectory_to_transitions(self, traj: Trajectory) -> tuple[int, int]:
        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        if stride > 0:
            if not traj.rlt_step_trace:
                raise RuntimeError(
                    "RLT Stage2 stride replay is enabled but trajectory has no "
                    "rlt_step_trace. Refusing to fall back to chunk-boundary replay."
                )
            return self._step_trace_to_transitions(traj)
        return self._chunk_trajectory_to_transitions(traj)

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            added, completed_episodes = self._trajectory_to_transitions(trajectory)
            self.transitions_since_train += added
            self.episodes_since_train += completed_episodes
            self.total_transitions_added += added
            self.total_episodes_added += completed_episodes

    def _global_rollout_counters(self) -> dict[str, float]:
        return all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )

    def _global_min_replay_size(self) -> int:
        replay_size = 0 if self.replay_buffer is None else len(self.replay_buffer)
        reduced = all_reduce_dict(
            {"min_replay_size": float(replay_size)},
            op=torch.distributed.ReduceOp.MIN,
        )
        return int(reduced["min_replay_size"])

    def _global_min_demo_size(self) -> int:
        demo_size = 0 if self.demo_buffer is None else len(self.demo_buffer)
        reduced = all_reduce_dict(
            {"min_demo_size": float(demo_size)},
            op=torch.distributed.ReduceOp.MIN,
        )
        return int(reduced["min_demo_size"])

    def _sample_training_batch(
        self,
        batch_size: int,
        *,
        use_demo: bool,
    ) -> dict[str, torch.Tensor]:
        assert self.replay_buffer is not None
        if not use_demo or self.demo_buffer is None:
            return self.replay_buffer.sample(batch_size, self.device).to_dict()

        replay_batch_size = batch_size - batch_size // 2
        demo_batch_size = batch_size - replay_batch_size
        replay_batch = self.replay_buffer.sample(
            replay_batch_size,
            self.device,
        ).to_dict()
        demo_batch = self.demo_buffer.sample(demo_batch_size, self.device).to_dict()
        return {
            key: torch.cat([replay_batch[key], demo_batch[key]], dim=0)
            for key in replay_batch
        }

    def _append_training_schedule_metrics(
        self,
        metrics: dict[str, Any],
        *,
        global_counters: dict[str, float],
        global_min_replay_size: int,
        global_min_demo_size: int,
        warmup_min_size: int,
        min_demo_buffer_size: int,
        warmup_required_updates: int,
        update_ratio: int,
        train_every_transitions: int,
        train_every_episodes: int,
        should_train: bool,
        skip_reason: int,
        pending_update_budget: int,
        updates_scheduled: int = 0,
        critic_updates_run: int = 0,
        actor_updates_run: int = 0,
    ) -> None:
        append_to_dict(
            metrics,
            {
                "rlt_stage2/update_step": float(self.update_step),
                "rlt_stage2/critic_updates_run": float(critic_updates_run),
                "rlt_stage2/actor_updates_run": float(actor_updates_run),
                "rlt_stage2/should_train": float(should_train),
                "rlt_stage2/skip_reason": float(skip_reason),
                "rlt_stage2/ready_for_online": float(
                    self.update_step >= warmup_required_updates
                ),
                "rlt_stage2/global_min_replay_size": float(global_min_replay_size),
                "rlt_stage2/global_min_demo_size": float(global_min_demo_size),
                "rlt_stage2/min_replay_buffer_size": float(warmup_min_size),
                "rlt_stage2/min_demo_buffer_size": float(min_demo_buffer_size),
                "rlt_stage2/update_epoch": float(update_ratio),
                "rlt_stage2/warmup_required_updates": float(
                    warmup_required_updates
                ),
                "rlt_stage2/pending_update_budget": float(pending_update_budget),
                "rlt_stage2/updates_scheduled": float(updates_scheduled),
                "rlt_stage2/global_transitions_since_train": float(
                    global_counters["transitions_since_train"]
                ),
                "rlt_stage2/global_total_transitions_added": float(
                    global_counters["total_transitions_added"]
                ),
            },
        )

    def _append_replay_stats(self, metrics: dict[str, Any]) -> None:
        if self.replay_buffer is None:
            return
        stats = self.replay_buffer.get_stats()
        for key, value in stats.items():
            if key == "capacity":
                continue
            append_to_dict(metrics, {f"replay_buffer/{key}": value})
        if self.demo_buffer is None:
            return
        demo_stats = self.demo_buffer.get_stats()
        for key, value in demo_stats.items():
            if key == "capacity":
                continue
            append_to_dict(metrics, {f"demo_buffer/{key}": value})

    @Worker.timer("run_training")
    def run_training(self):
        if self.replay_buffer is None:
            return {}

        metrics: dict[str, Any] = {}
        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        warmup_min_size = int(
            replay_cfg.get(
                "min_buffer_size",
                self.cfg.algorithm.get("warmup_min_size", 1),
            )
        )
        warmup_required_updates = self._warmup_required_updates()
        update_ratio = int(self.cfg.algorithm.get("update_epoch", 1))
        max_updates_per_train_step = int(
            self.cfg.algorithm.get("max_updates_per_train_step", 0)
        )
        train_every_transitions = int(
            self.cfg.algorithm.get("train_every_transitions", 0)
        )
        train_every_episodes = int(self.cfg.algorithm.get("train_every_episodes", 0))
        global_counters = self._global_rollout_counters()
        global_min_replay_size = self._global_min_replay_size()
        global_min_demo_size = self._global_min_demo_size()
        min_demo_buffer_size = int(
            self.cfg.algorithm.get("demo_buffer", {}).get("min_buffer_size", 0)
        )
        if self.demo_buffer is not None:
            min_demo_buffer_size = max(min_demo_buffer_size, 1)
        demo_ready = (
            self.demo_buffer is None or global_min_demo_size >= min_demo_buffer_size
        )
        use_demo = self.demo_buffer is not None and demo_ready
        buffer_ready = global_min_replay_size >= warmup_min_size and demo_ready
        global_total_transitions_added = int(
            global_counters["total_transitions_added"]
        )
        if buffer_ready and self.warmup_ready_total_transitions is None:
            self.warmup_ready_total_transitions = global_total_transitions_added
            self.warmup_ready_total_episodes = int(
                global_counters["total_episodes_added"]
            )

        desired_total_updates = 0
        if (
            buffer_ready
            and self.warmup_ready_total_transitions is not None
            and update_ratio > 0
        ):
            online_transitions_added = max(
                global_total_transitions_added - self.warmup_ready_total_transitions,
                0,
            )
            online_episodes_added = max(
                int(global_counters["total_episodes_added"])
                - int(self.warmup_ready_total_episodes or 0),
                0,
            )
            transition_cycles = (
                online_transitions_added // train_every_transitions
                if train_every_transitions > 0
                else 0
            )
            episode_cycles = (
                online_episodes_added // train_every_episodes
                if train_every_episodes > 0
                else 0
            )
            if train_every_transitions <= 0 and train_every_episodes <= 0:
                online_update_cycles = online_transitions_added
            else:
                online_update_cycles = max(transition_cycles, episode_cycles)
            desired_total_updates = (
                warmup_required_updates + online_update_cycles * update_ratio
            )
        self.pending_update_budget = max(desired_total_updates - self.update_step, 0)
        updates_scheduled = int(self.pending_update_budget)
        should_train = buffer_ready and updates_scheduled > 0

        skip_reason = 0
        if update_ratio <= 0:
            skip_reason = 3
        elif not buffer_ready:
            skip_reason = 1
        elif not should_train:
            skip_reason = 2

        if skip_reason != 0:
            self._append_replay_stats(metrics)
            self._append_training_schedule_metrics(
                metrics,
                global_counters=global_counters,
                global_min_replay_size=global_min_replay_size,
                global_min_demo_size=global_min_demo_size,
                warmup_min_size=warmup_min_size,
                min_demo_buffer_size=min_demo_buffer_size,
                warmup_required_updates=warmup_required_updates,
                update_ratio=update_ratio,
                train_every_transitions=train_every_transitions,
                train_every_episodes=train_every_episodes,
                should_train=False,
                skip_reason=skip_reason,
                pending_update_budget=self.pending_update_budget,
                updates_scheduled=updates_scheduled,
            )
            return self._process_train_metrics(metrics)

        if self.enable_offload:
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        assert global_batch_size_per_rank % self.cfg.actor.micro_batch_size == 0, (
            "global batch per rank must be divisible by micro_batch_size"
        )
        micro_batch_cnt = global_batch_size_per_rank // self.cfg.actor.micro_batch_size
        self.gradient_accumulation = micro_batch_cnt

        self.model.train()
        critic_updates_run = 0
        actor_updates_run = 0
        updates_to_run = int(self.pending_update_budget)
        if max_updates_per_train_step > 0:
            updates_to_run = min(updates_to_run, max_updates_per_train_step)
        for _ in range(updates_to_run):
            batch_dict = self._sample_training_batch(
                global_batch_size_per_rank,
                use_demo=use_demo,
            )
            micro_batches = []
            for i in range(micro_batch_cnt):
                begin = i * self.cfg.actor.micro_batch_size
                end = begin + self.cfg.actor.micro_batch_size
                micro_batches.append({k: v[begin:end] for k, v in batch_dict.items()})
            epoch_metrics = self._update_one_epoch(
                micro_batches,
                global_min_replay_size=global_min_replay_size,
            )
            append_to_dict(metrics, epoch_metrics)
            self.update_step += 1
            critic_updates_run += 1
            if "rlt_stage2/actor_loss" in epoch_metrics:
                actor_updates_run += 1
        self.pending_update_budget = max(self.pending_update_budget - critic_updates_run, 0)

        self._append_replay_stats(metrics)
        self._append_training_schedule_metrics(
            metrics,
            global_counters=global_counters,
            global_min_replay_size=global_min_replay_size,
            global_min_demo_size=global_min_demo_size,
            warmup_min_size=warmup_min_size,
            min_demo_buffer_size=min_demo_buffer_size,
            warmup_required_updates=warmup_required_updates,
            update_ratio=update_ratio,
            train_every_transitions=train_every_transitions,
            train_every_episodes=train_every_episodes,
            should_train=True,
            skip_reason=0,
            pending_update_budget=self.pending_update_budget,
            updates_scheduled=updates_scheduled,
            critic_updates_run=critic_updates_run,
            actor_updates_run=actor_updates_run,
        )
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        return self._process_train_metrics(metrics)

    def _update_one_epoch(
        self,
        micro_batches: list[dict[str, torch.Tensor]],
        *,
        global_min_replay_size: int,
    ) -> dict[str, float]:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        critic_losses = []
        q1_values = []
        q2_values = []

        self.optimizer.zero_grad(set_to_none=True)
        self.qf_optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(micro_batches):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == len(micro_batches),
            )
            with backward_ctx:
                with self.amp_context:
                    td_target = self.model.compute_td_target_batch(
                        rewards=batch["rewards"].to(torch.float32),
                        dones=batch["dones"].to(torch.float32),
                        next_x=batch["next_x"].to(torch.float32),
                        next_a_tilde=batch["next_a_tilde"].to(torch.float32),
                    )
                    q1, q2 = self.model.critic_forward(
                        batch["x"].to(torch.float32),
                        batch["a"].to(torch.float32),
                    )
                    loss = (
                        critic_loss(q1, q2, td_target) / self.gradient_accumulation
                    )
                self.grad_scaler.scale(loss).backward()
            critic_losses.append(loss.detach().float().item() * self.gradient_accumulation)
            q1_values.append(q1.detach().float().mean().item())
            q2_values.append(q2.detach().float().mean().item())

        self.grad_scaler.unscale_(self.qf_optimizer)
        critic_grad_norm = self._strategy.clip_grad_norm_(self.model)
        self.grad_scaler.step(self.qf_optimizer)
        self.grad_scaler.update()
        self.qf_lr_scheduler.step()
        self.qf_optimizer.zero_grad(set_to_none=True)

        metrics = {
            "rlt_stage2/critic_loss": float(np.mean(critic_losses)),
            "critic/q1_mean": float(np.mean(q1_values)),
            "critic/q2_mean": float(np.mean(q2_values)),
            "critic/grad_norm": float(critic_grad_norm),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
        }

        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        warmup_min_size = int(
            replay_cfg.get(
                "min_buffer_size",
                self.cfg.algorithm.get("warmup_min_size", 1),
            )
        )
        replay_ready = global_min_replay_size >= warmup_min_size
        bc_weight, q_weight, delta_weight, in_loss_warmup, loss_ramp_progress = (
            self._resolve_actor_loss_weights()
        )
        update_actor = (
            replay_ready
            and (self.update_step + 1) % int(self.cfg.algorithm.critic_actor_ratio) == 0
        )
        if update_actor:
            actor_losses = []
            actor_q_values = []
            actor_action_ref_abs = []
            actor_bc_losses = []
            actor_bc_ref_losses = []
            actor_bc_human_losses = []
            actor_human_mask_ratios = []
            self.optimizer.zero_grad()
            self.model.set_online_critic_requires_grad(False)
            for idx, batch in enumerate(micro_batches):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == len(micro_batches),
                )
                with backward_ctx:
                    with self.amp_context:
                        actions = self.model.actor_forward(
                            batch["x"].to(torch.float32),
                            batch["a_tilde"].to(torch.float32),
                            deterministic=False,
                            apply_ref_dropout=bool(
                                stage2_cfg.get("ref_action_dropout", 0.0) > 0.0
                            ),
                            apply_action_noise=True,
                        )
                        a_tilde_flat = batch["a_tilde"].to(torch.float32)
                        action_chunk_flat = batch["action_chunk"].to(torch.float32)
                        action_ref_delta = actions - a_tilde_flat
                        chunk_len = int(self.cfg.actor.model.num_action_chunks)
                        action_dim = int(self.cfg.actor.model.action_dim)
                        actions_chunk = actions.reshape(-1, chunk_len, action_dim)
                        a_tilde_chunk = a_tilde_flat.reshape(-1, chunk_len, action_dim)
                        executed_action_chunk = action_chunk_flat.reshape(
                            -1,
                            chunk_len,
                            action_dim,
                        )
                        q_value = self.model.critic_min(
                            batch["x"].to(torch.float32),
                            actions,
                        )
                        actor_total_loss, actor_loss_metrics = actor_loss(
                            q_value=q_value,
                            a=actions_chunk,
                            a_tilde=a_tilde_chunk,
                            action_chunk=executed_action_chunk,
                            source_chunk=batch["source_chunk"].to(torch.uint8),
                            bc_weight=bc_weight,
                            q_weight=q_weight,
                            delta_weight=delta_weight,
                        )
                        loss = actor_total_loss / self.gradient_accumulation
                    self.grad_scaler.scale(loss).backward()
                actor_losses.append(loss.detach().float().item() * self.gradient_accumulation)
                actor_q_values.append(q_value.detach().float().mean().item())
                actor_action_ref_abs.append(
                    action_ref_delta.detach().float().abs().mean().item()
                )
                actor_bc_losses.append(
                    float(actor_loss_metrics["bc_loss"].float().item())
                )
                actor_bc_ref_losses.append(
                    float(actor_loss_metrics["bc_ref_loss"].float().item())
                )
                actor_bc_human_losses.append(
                    float(actor_loss_metrics["bc_human_loss"].float().item())
                )
                actor_human_mask_ratios.append(
                    float(actor_loss_metrics["human_mask_ratio"].float().item())
                )
            self.model.set_online_critic_requires_grad(True)

            self.grad_scaler.unscale_(self.optimizer)
            actor_grad_norm = self._strategy.clip_grad_norm_(self.model)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            metrics.update(
                {
                    "rlt_stage2/actor_loss": float(np.mean(actor_losses)),
                    "actor/q_mean": float(np.mean(actor_q_values)),
                    "actor/grad_norm": float(actor_grad_norm),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/action_ref_abs_mean": float(np.mean(actor_action_ref_abs)),
                    "actor/bc_weight": bc_weight,
                    "actor/q_weight": q_weight,
                    "actor/bc_loss": float(np.mean(actor_bc_losses)),
                    "actor/bc_ref_loss": float(np.mean(actor_bc_ref_losses)),
                    "actor/bc_human_loss": float(np.mean(actor_bc_human_losses)),
                    "actor/human_mask_ratio": float(np.mean(actor_human_mask_ratios)),
                }
            )
        else:
            metrics.update(
                {
                    "actor/bc_weight": bc_weight,
                    "actor/q_weight": q_weight,
                }
            )

        target_update_freq = int(self.cfg.algorithm.get("target_update_freq", 1))
        if target_update_freq <= 0:
            raise ValueError(
                f"algorithm.target_update_freq must be positive, got {target_update_freq}."
            )
        if (self.update_step + 1) % target_update_freq == 0:
            self.soft_update_target_model()
        return metrics

    def _process_train_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                mean_metric_dict[key] = float(np.mean(value))
            elif isinstance(value, torch.Tensor):
                mean_metric_dict[key] = float(value.detach().cpu().item())
            else:
                mean_metric_dict[key] = float(value)
        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    def compute_advantages_and_returns(self):
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        stage2_save_path = os.path.join(save_base_path, "rlt_stage2_components")
        os.makedirs(stage2_save_path, exist_ok=True)
        torch.save(
            {
                "update_step": self.update_step,
                "pending_update_budget": self.pending_update_budget,
                "warmup_ready_total_transitions": self.warmup_ready_total_transitions,
                "warmup_ready_total_episodes": self.warmup_ready_total_episodes,
                "version": self.version,
                "transitions_since_train": self.transitions_since_train,
                "episodes_since_train": self.episodes_since_train,
                "total_transitions_added": self.total_transitions_added,
                "total_episodes_added": self.total_episodes_added,
                "replay_buffer": self.replay_buffer.state_dict()
                if self.replay_buffer is not None
                else None,
                "demo_buffer": self.demo_buffer.state_dict()
                if self.demo_buffer is not None
                else None,
            },
            os.path.join(stage2_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        stage2_load_path = os.path.join(
            load_base_path,
            "rlt_stage2_components",
            f"checkpoint_rank_{self._rank}.pt",
        )
        if os.path.exists(stage2_load_path):
            state = torch.load(stage2_load_path, map_location="cpu", weights_only=False)
            self.update_step = int(state.get("update_step", 0))
            self.pending_update_budget = int(state.get("pending_update_budget", 0))
            warmup_ready_total_transitions = state.get(
                "warmup_ready_total_transitions", None
            )
            self.warmup_ready_total_transitions = (
                None
                if warmup_ready_total_transitions is None
                else int(warmup_ready_total_transitions)
            )
            warmup_ready_total_episodes = state.get(
                "warmup_ready_total_episodes", None
            )
            self.warmup_ready_total_episodes = (
                None
                if warmup_ready_total_episodes is None
                else int(warmup_ready_total_episodes)
            )
            self.version = int(state.get("version", self.update_step))
            self.transitions_since_train = int(
                state.get("transitions_since_train", 0)
            )
            self.episodes_since_train = int(state.get("episodes_since_train", 0))
            self.total_transitions_added = int(state.get("total_transitions_added", 0))
            self.total_episodes_added = int(state.get("total_episodes_added", 0))
            if self.replay_buffer is not None and state.get("replay_buffer") is not None:
                self.replay_buffer.load_state_dict(state["replay_buffer"])
            if self.demo_buffer is not None and state.get("demo_buffer") is not None:
                self.demo_buffer.load_state_dict(state["demo_buffer"])

    def set_global_step(self, global_step: int) -> None:
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
