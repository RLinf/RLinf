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

import queue

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    collect_trajectory_replay_metrics,
    compute_split_num,
    trajectory_has_bool_tensor,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class RLTACLossMixin:
    """RLT actor-critic losses on top of RLinf replay-buffer worker plumbing.

    Forward types follow the existing off-policy actor-critic API, while the
    RLT objective disables entropy/alpha and uses a fixed-std actor, min-Q
    critic target, Q1 actor objective, and BC regularization.
    """

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _chunk_shape(self) -> tuple[int, int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        return chunk_len, action_dim

    def get_rollout_sync_version(self) -> int:
        """Expose learner update count when RLT warmup gates actor rollout."""
        if not self.use_rlt_schedule:
            return int(self.version)
        return int(self.update_step)

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._flatten_chunk(obs["ref_chunk"])

    @staticmethod
    def _require_twin_q(all_q_values: torch.Tensor) -> None:
        if all_q_values.shape[-1] < 2:
            raise ValueError(
                "RLT Stage 2 requires at least two Q heads for twin-Q training, "
                f"got Q shape {tuple(all_q_values.shape)}."
            )

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    def _q1(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return all_q_values[..., 0:1]

    def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1)
        rewards = rewards.to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=rewards.device),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        action_chunk = self._flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
        bc_ref_chunk = self._flatten_chunk(ref_chunk).reshape(
            ref_chunk.shape[0], -1, action_dim
        )[:, :chunk_len]

        if intervene_flags is None:
            human_mask = torch.zeros(
                pi_chunk.shape[:2], dtype=torch.bool, device=pi_chunk.device
            )
        else:
            human_mask = (
                self._flatten_chunk(intervene_flags)
                .to(device=pi_chunk.device)
                .bool()
                .reshape(-1, chunk_len, action_dim)
                .any(dim=-1)
            )

        bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        bc_loss = torch.mean(bc_error)

        policy_mask = ~human_mask
        ref_error = torch.mean(torch.square(pi_chunk - bc_ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = torch.mean(human_mask.to(torch.float32)).item()
        metrics = {
            "bc_loss": bc_loss.detach().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
        }
        return bc_loss, metrics

    def _actor_objective_weights(self) -> tuple[float, float, dict[str, float]]:
        """Resolve RLT actor-objective BC/Q weights."""
        schedule_cfg = self.cfg.algorithm.get("actor_weight_schedule", {})
        schedule_enabled = bool(schedule_cfg.get("enable", False))
        if not schedule_enabled:
            bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
            q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
            return (
                bc_weight,
                q_weight,
                {
                    "bc_weight": bc_weight,
                    "q_weight": q_weight,
                    "actor_weight_schedule_enabled": 0.0,
                    "actor_weight_in_warmup": 0.0,
                    "actor_weight_ramp_progress": 1.0,
                },
            )

        weight_warmup_updates = int(schedule_cfg.get("warmup_updates", 0))
        ramp_updates = int(schedule_cfg.get("ramp_updates", 0))
        in_warmup = int(self.update_step) < weight_warmup_updates
        warmup_bc_weight = float(
            schedule_cfg.get(
                "warmup_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        warmup_q_weight = float(
            schedule_cfg.get(
                "warmup_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        online_bc_weight = float(
            schedule_cfg.get(
                "online_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        online_q_weight = float(
            schedule_cfg.get(
                "online_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        elif ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(int(self.update_step) - weight_warmup_updates + 1)
                    / float(ramp_updates),
                ),
            )
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        else:
            bc_weight = online_bc_weight
            q_weight = online_q_weight
            ramp_progress = 1.0

        metrics = {
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "actor_weight_schedule_enabled": 1.0,
            "actor_weight_in_warmup": float(in_warmup),
            "actor_weight_ramp_progress": ramp_progress,
        }
        return bc_weight, q_weight, metrics

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = batch["terminations"]
        if use_simulator_transition_replay(self.cfg):
            done_source = batch["dones"]
        done_source = done_source.to(self.torch_dtype)
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            next_actions, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
            )

            if not use_crossq:
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next_target)
            else:
                _, all_qf_next = self.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next.detach())

            reward_target = self._discounted_chunk_rewards(rewards)
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
        else:
            all_data_q_values, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        if not use_crossq:
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        num_q_values = all_qf_pi.shape[-1]
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(num_q_values)
        }
        qf_pi = self._q1(all_qf_pi)
        metrics["q_pi"] = qf_pi.mean().item()

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, rlt_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
        )
        metrics.update(rlt_metrics)

        entropy = -log_pi.mean()
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        metrics.update(weight_metrics)
        metrics["action_ref_abs_mean"] = (
            (self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk))
            .abs()
            .mean()
            .detach()
            .item()
        )
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["reference_dropout_prob"] = reference_dropout_prob

        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        del batch
        raise NotImplementedError(
            "RLT AC disables entropy/alpha training. Use "
            "algorithm.entropy_tuning.alpha_type=fixed_alpha."
        )


class RLTACReplayMixin:
    """Shared rollout-to-replay ingestion for sync and async RLT AC workers."""

    @staticmethod
    def _trajectory_transition_count(traj: Trajectory) -> int:
        if traj.actions is None:
            return 0
        return int(traj.actions.shape[0] * traj.actions.shape[1])

    @staticmethod
    def _trajectory_completed_episodes(traj: Trajectory) -> int:
        dones = traj.dones
        if dones is None:
            return 0
        return int(dones.reshape(dones.shape[0], dones.shape[1], -1).any(dim=-1).sum())

    @staticmethod
    def _transition_reward_value(traj: Trajectory) -> float | None:
        rewards = traj.rewards
        if not isinstance(rewards, torch.Tensor) or rewards.numel() == 0:
            return None
        return float(rewards.detach().float().reshape(-1).sum().item())

    @staticmethod
    def _transition_done_value(traj: Trajectory) -> bool | None:
        dones = traj.dones
        if not isinstance(dones, torch.Tensor) or dones.numel() == 0:
            return None
        return bool(dones.detach().to(torch.bool).reshape(-1).any().item())

    @staticmethod
    def _row_tensor(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return tensor[idx].detach().clone().unsqueeze(0).unsqueeze(0).cpu().contiguous()

    @staticmethod
    def _step_env_tensor(
        tensor: torch.Tensor, step_idx: int, env_idx: int
    ) -> torch.Tensor:
        return (
            tensor[step_idx, env_idx]
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
        idx: int,
    ) -> dict[str, torch.Tensor]:
        row_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                row_dict[key] = self._row_tensor(value, idx)
        return row_dict

    def _rlt_obs_from_flat_dict(
        self,
        flat: dict,
        dict_key: str,
        idx: int,
    ) -> dict[str, torch.Tensor] | None:
        value = flat.get(dict_key)
        if not isinstance(value, dict):
            return None
        obs = self._row_tensor_dict(value, idx)
        return obs if obs else None

    @staticmethod
    def _flat_record_transition(flat: dict, idx: int) -> bool:
        forward_inputs = flat.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return False
        record_transition = forward_inputs.get("record_transition")
        if not isinstance(record_transition, torch.Tensor):
            return False
        if idx >= record_transition.shape[0]:
            return False
        return bool(record_transition[idx].detach().to(torch.bool).reshape(-1).all())

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
            rewards, torch.Tensor
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
        dict_fields = ("forward_inputs",)
        replay_trajectories = []
        completed_episodes = 0
        traj_len = int(trajectory.actions.shape[0])
        bsz = int(trajectory.actions.shape[1])
        num_rows = int(actions.shape[0])
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
                idx = t * bsz + env_idx
                if idx >= num_rows:
                    break
                if not self._flat_record_transition(flat, idx):
                    continue

                transition = Trajectory(
                    max_episode_length=1,
                    model_weights_id=trajectory.model_weights_id,
                )
                for field_name in tensor_fields:
                    value = flat.get(field_name)
                    if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                        setattr(transition, field_name, self._row_tensor(value, idx))
                for field_name in dict_fields:
                    value = flat.get(field_name)
                    if isinstance(value, dict):
                        setattr(
                            transition, field_name, self._row_tensor_dict(value, idx)
                        )

                curr_obs = self._rlt_obs_from_flat_dict(flat, "curr_obs", idx)
                if curr_obs is None:
                    raise ValueError(
                        "RLT transition replay requires curr_obs. Ensure "
                        "update_rlt_transitions() populated transition obs "
                        f"before replay ingestion, got row index {idx}."
                    )
                transition.curr_obs = curr_obs

                # Dones have one extra initial slot, so transition t reads
                # terminal flags from t+1. Rewards are already action-aligned
                # by EmbodiedRolloutResult because the initial empty reward is
                # skipped and the final reward is appended after rollout.
                done_idx = min(
                    t + 1,
                    int(trajectory.dones.shape[0]) - 1
                    if isinstance(trajectory.dones, torch.Tensor)
                    else traj_len - 1,
                )
                for done_field in ("dones", "terminations", "truncations"):
                    done_value = getattr(trajectory, done_field, None)
                    if (
                        isinstance(done_value, torch.Tensor)
                        and done_idx < done_value.shape[0]
                        and env_idx < done_value.shape[1]
                    ):
                        setattr(
                            transition,
                            done_field,
                            self._step_env_tensor(done_value, done_idx, env_idx),
                        )

                is_done = (
                    isinstance(transition.dones, torch.Tensor)
                    and transition.dones.reshape(-1).to(torch.bool).any()
                )
                if is_done:
                    next_obs = curr_obs
                else:
                    next_obs = self._rlt_obs_from_flat_dict(flat, "next_obs", idx)
                if next_obs is not None:
                    transition.next_obs = next_obs
                else:
                    raise ValueError(
                        "RLT transition replay requires next_obs for non-terminal "
                        "transitions. Ensure update_rlt_transitions() populated "
                        f"transition obs before replay ingestion, got row index {idx}."
                    )

                replay_trajectories.append(transition)
                if is_done:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes

    def _transition_replay_metrics(
        self,
        replay_trajectories: list[Trajectory],
    ) -> dict[str, float]:
        metrics = {"replay/transition_count": float(len(replay_trajectories))}
        reward_values = [
            reward
            for traj in replay_trajectories
            if (reward := self._transition_reward_value(traj)) is not None
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
            for traj in replay_trajectories
            if (done := self._transition_done_value(traj)) is not None
        ]
        if done_values:
            metrics["replay/done_rate"] = float(
                sum(bool(done) for done in done_values) / len(done_values)
            )
        return metrics

    def _ingest_rollout_trajectories(
        self,
        recv_list: list[Trajectory],
    ) -> tuple[int, int]:
        self._last_replay_metrics = {}

        if use_simulator_transition_replay(self.cfg):
            replay_list = []
            completed = 0
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                transition_trajs, completed_count = (
                    self._transition_replay_trajectories(traj)
                )
                replay_list.extend(transition_trajs)
                completed += completed_count
            self._last_replay_metrics = {
                **self._transition_replay_metrics(replay_list),
                **collect_trajectory_replay_metrics(recv_list, reducer=all_reduce_dict),
            }
            self.replay_buffer.add_trajectories(replay_list)

            if self.demo_buffer is not None:
                intervene_traj_list = [
                    traj
                    for traj in replay_list
                    if trajectory_has_bool_tensor(traj.intervene_flags)
                ]
                if len(intervene_traj_list) > 0:
                    self.demo_buffer.add_trajectories(intervene_traj_list)

            return len(replay_list), completed

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        added = sum(self._trajectory_transition_count(traj) for traj in recv_list)
        completed = sum(self._trajectory_completed_episodes(traj) for traj in recv_list)
        self._last_replay_metrics = collect_trajectory_replay_metrics(
            recv_list, reducer=all_reduce_dict
        )
        return added, completed

    def _update_rollout_ingest_counters(self, added: int, completed: int) -> None:
        if not getattr(self, "use_rlt_schedule", False):
            return
        if not hasattr(self, "transitions_since_train"):
            return
        self.transitions_since_train += added
        self.episodes_since_train += completed
        self.total_transitions_added += added
        self.total_episodes_added += completed


class RLTACFSDPPolicy(RLTACLossMixin, RLTACReplayMixin, EmbodiedSACFSDPPolicy):
    """Synchronous RLT AC worker with transition replay and warmup scheduling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rlt_schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0

    def setup_sac_components(self):
        """Initialize replay components and let RLT schedule own readiness."""
        super().setup_sac_components()
        if self.use_rlt_schedule:
            self.buffer_dataset.min_replay_buffer_size = 1

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel):
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    def _global_rlt_counters(self) -> dict[str, float]:
        summed = all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )
        minimums = all_reduce_dict(
            {
                "min_replay_size": float(self.replay_buffer.total_samples),
                "min_demo_size": float(
                    0 if self.demo_buffer is None else self.demo_buffer.total_samples
                ),
            },
            op=torch.distributed.ReduceOp.MIN,
        )
        summed.update(minimums)
        return summed

    def _rlt_updates_to_run(self) -> tuple[int, dict[str, float]]:
        replay_cfg = self.cfg.algorithm.replay_buffer
        schedule_cfg = self.rlt_schedule_cfg
        min_buffer_size = int(
            schedule_cfg.get("warmup_min_size", replay_cfg.get("min_buffer_size", 1))
        )
        counters = self._global_rlt_counters()
        buffer_ready = counters["min_replay_size"] >= min_buffer_size
        warmup_required_updates = int(
            schedule_cfg.get("warmup_post_collect_updates", 0)
        )
        if buffer_ready and self._warmup_ready_total_transitions is None:
            self._warmup_ready_total_transitions = int(
                counters["total_transitions_added"]
            )
            self._warmup_ready_total_episodes = int(counters["total_episodes_added"])

        train_every_transitions = int(schedule_cfg.get("train_every_transitions", 0))
        train_every_episodes = int(schedule_cfg.get("train_every_episodes", 0))
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        max_updates = int(schedule_cfg.get("max_updates_per_train_step", 0))

        updates_to_run = 0
        skip_reason = 0
        desired_total_updates = 0
        pending_updates = 0
        updates_scheduled = 0
        if update_epoch <= 0:
            skip_reason = 3
        elif not buffer_ready:
            skip_reason = 1
        else:
            online_transitions = max(
                int(counters["total_transitions_added"])
                - int(self._warmup_ready_total_transitions or 0),
                0,
            )
            online_episodes = max(
                int(counters["total_episodes_added"])
                - int(self._warmup_ready_total_episodes or 0),
                0,
            )
            if train_every_transitions <= 0 and train_every_episodes <= 0:
                online_cycles = online_transitions
            else:
                transition_cycles = (
                    online_transitions // train_every_transitions
                    if train_every_transitions > 0
                    else 0
                )
                episode_cycles = (
                    online_episodes // train_every_episodes
                    if train_every_episodes > 0
                    else 0
                )
                online_cycles = max(transition_cycles, episode_cycles)
            desired_total_updates = (
                warmup_required_updates + online_cycles * update_epoch
            )
            pending_updates = max(desired_total_updates - int(self.update_step), 0)
            updates_scheduled = pending_updates
            updates_to_run = pending_updates
            if max_updates > 0:
                updates_to_run = min(updates_to_run, max_updates)
            if updates_to_run <= 0:
                skip_reason = 2
        self.pending_update_budget = int(pending_updates)

        metrics = {
            "rlt/update_step": float(self.update_step),
            "rlt/ready_for_online": float(
                int(self.update_step) >= warmup_required_updates
            ),
            "rlt/warmup_required_updates": float(warmup_required_updates),
            "rlt/update_epoch": float(update_epoch),
            "rlt/max_updates_per_train_step": float(max_updates),
            "rlt/train_every_transitions": float(train_every_transitions),
            "rlt/train_every_episodes": float(train_every_episodes),
            "rlt/desired_total_updates": float(desired_total_updates),
            "rlt/pending_update_budget": float(self.pending_update_budget),
            "rlt/updates_scheduled": float(updates_scheduled),
            "rlt/updates_to_run": float(updates_to_run),
            "rlt/critic_updates_run": 0.0,
            "rlt/actor_updates_run": 0.0,
            "rlt/should_train": float(updates_to_run > 0),
            "rlt/skip_reason": float(skip_reason),
            "rlt/global_min_replay_size": float(counters["min_replay_size"]),
            "rlt/min_replay_buffer_size": float(min_buffer_size),
            "rlt/global_transitions_since_train": float(
                counters["transitions_since_train"]
            ),
            "rlt/global_total_transitions_added": float(
                counters["total_transitions_added"]
            ),
        }
        metrics.update(getattr(self, "_last_replay_metrics", {}))
        return updates_to_run, metrics

    def run_training(self):
        if not self.use_rlt_schedule:
            mean_metric_dict = super().run_training()
            replay_metrics = getattr(self, "_last_replay_metrics", {})
            if replay_metrics:
                mean_metric_dict = {**mean_metric_dict, **replay_metrics}
            return mean_metric_dict

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        updates_to_run, schedule_metrics = self._rlt_updates_to_run()
        if updates_to_run <= 0:
            mean_metric_dict = self.process_train_metrics(schedule_metrics)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            return mean_metric_dict

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(updates_to_run):
            update_actor = int(self.update_step) % int(self.critic_actor_ratio) == 0
            metrics_data = self.update_one_epoch(train_actor=True)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        schedule_metrics["rlt/critic_updates_run"] = float(critic_updates_run)
        schedule_metrics["rlt/actor_updates_run"] = float(actor_updates_run)
        self.pending_update_budget = max(
            int(self.pending_update_budget) - critic_updates_run,
            0,
        )
        schedule_metrics["rlt/pending_update_budget"] = float(
            self.pending_update_budget
        )
        append_to_dict(metrics, schedule_metrics)
        mean_metric_dict = self.process_train_metrics(metrics)
        self.transitions_since_train = 0
        self.episodes_since_train = 0

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict


class AsyncRLTACFSDPPolicy(
    RLTACLossMixin, RLTACReplayMixin, AsyncEmbodiedSACFSDPPolicy
):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rlt_schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    async def run_training(self):
        mean_metric_dict = await super().run_training()
        replay_metrics = getattr(self, "_last_replay_metrics", {})
        if replay_metrics:
            mean_metric_dict = {**mean_metric_dict, **replay_metrics}
        return mean_metric_dict
