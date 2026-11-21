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

import asyncio
import math
import os
import queue

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.action_grid import ActionGrid
from rlinf.algorithms.rlt.direct_human_bc import direct_bc_loss, execution_weight
from rlinf.algorithms.rlt.human_bc import (
    KIND_ACTIVE,
    KIND_AUTO,
    build_bc_targets,
    ref_tilde,
    residual_limit_vector,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    collect_trajectory_replay_metrics,
    trajectory_has_bool_tensor,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

# Offset of the reference chunk when rollout prefetch is enabled.
_RLT_REF_OFFSET = int(os.environ.get("RLT_REF_OFFSET", "0"))


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

    def _executor_grid_enabled(self) -> bool:
        actor = (
            self.cfg.get("actor")
            if hasattr(self.cfg, "get")
            else getattr(self.cfg, "actor", None)
        )
        model = (
            actor.get("model")
            if hasattr(actor, "get")
            else getattr(actor, "model", None)
        )
        if model is None:
            return False
        return bool(
            model.get("executor_action_grid", False)
            if hasattr(model, "get")
            else getattr(model, "executor_action_grid", False)
        )

    def _action_grid(self) -> ActionGrid:
        return ActionGrid.from_config(
            self.cfg.actor.model, default_offset=_RLT_REF_OFFSET
        )

    def _target_shape(self) -> tuple[int, int]:
        return self._action_grid().target_steps, int(self.cfg.actor.model.action_dim)

    def _validate_action_grid(self, batch) -> None:
        if not self._executor_grid_enabled():
            return
        grid = self._action_grid()
        fi = batch.get("forward_inputs", {})
        metadata = fi.get("target_action_grid")
        if grid.aligned and (grid.changes_grid or metadata is not None):
            grid.validate_metadata(metadata)

    def _target_plan(self, batch):
        fi = batch["forward_inputs"]
        if not self._executor_grid_enabled():
            return fi["policy_action"]
        grid = self._action_grid()
        plan = fi.get("target_policy_action") if grid.aligned else None
        if plan is None:
            if grid.changes_grid:
                raise ValueError(
                    "executor-grid Q needs saved target_policy_action, "
                    "not interpolated measured actions"
                )
            plan = fi["policy_action"]
        return plan

    def _env_reference(self, obs):
        grid = self._action_grid()
        ref = obs["ref_chunk"].reshape(
            obs["ref_chunk"].shape[0], -1, int(self.cfg.actor.model.action_dim)
        )
        return grid.reference_env(ref)

    def _critic_joint_rel(self) -> bool:
        mcfg = self.cfg.actor.model
        val = (
            mcfg.get("critic_action_space", "critic")
            if hasattr(mcfg, "get")
            else getattr(mcfg, "critic_action_space", "critic")
        )
        return val == "joint_rel"

    def _joint_actor(self) -> bool:
        """rtj joint Stage2: 16D joint actor, critic on 20D TCP actions."""
        _, action_dim = self._chunk_shape()
        mcfg = self.cfg.actor.model
        cad = (
            mcfg.get("critic_action_dim", None)
            if hasattr(mcfg, "get")
            else getattr(mcfg, "critic_action_dim", None)
        )
        return int(cad or action_dim) != action_dim

    def _critic_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """The actor's chunk as the critic reads it: in joint mode its TCP rows
        by the differentiable native FK (Q gradients reach the joint actor).
        With critic_action_space joint_rel the critic reads the joint rows."""
        if not self._joint_actor() or self._critic_joint_rel():
            return self._flatten_chunk(actions) if self._critic_joint_rel() else actions
        from rlinf.envs.realworld.tianji.tianji_kinematics import (
            joint_chunk_rl_to_tcp_rl,
        )

        chunk_len, _ = self._target_shape()
        return joint_chunk_rl_to_tcp_rl(self._flatten_chunk(actions), chunk_len)

    @staticmethod
    def _with_vla_reference(batch, obs: dict) -> dict:
        """Schema-2 replay: always use raw VLA reference from forward_inputs.ref_chunk."""
        fi = (
            batch.get("forward_inputs")
            if isinstance(batch.get("forward_inputs"), dict)
            else {}
        )
        schema2 = isinstance(fi.get("data_schema"), torch.Tensor) and bool(
            (fi["data_schema"].reshape(-1) == 2).all()
        )
        if schema2 and "ref_chunk" in fi and "ref_chunk" in obs:
            return {**obs, "ref_chunk": fi["ref_chunk"].to(obs["ref_chunk"].device)}
        return obs

    def _with_critic_plan(self, obs: dict) -> dict:
        """Attach critic_plan in critic action space when critic_relative_action is enabled."""
        mcfg = self.cfg.actor.model
        enabled = (
            mcfg.get("critic_relative_action", False)
            if hasattr(mcfg, "get")
            else getattr(mcfg, "critic_relative_action", False)
        )
        if not enabled:
            return obs
        with torch.no_grad():
            plan = self._critic_actions(ref_tilde(self._ref_chunk(obs)))
        return {**obs, "critic_plan": plan}

    def get_rollout_sync_version(self) -> int:
        """Expose learner update count when RLT warmup gates actor rollout."""
        if not self.use_rlt_schedule:
            return int(self.version)
        return int(self.update_step)

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk_len, action_dim = self._target_shape()
        ref_chunk = self._flatten_chunk(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, action_dim
        )
        if self._action_grid().aligned:
            return (
                self._action_grid()
                .reference_targets(ref_chunk)
                .reshape(ref_chunk.shape[0], -1)
            )
        if _RLT_REF_OFFSET + chunk_len > ref_chunk.shape[1]:
            raise ValueError(
                f"RLT_REF_OFFSET={_RLT_REF_OFFSET} + chunk_len={chunk_len} "
                f"exceeds reference chunk length {ref_chunk.shape[1]}"
            )
        return ref_chunk[:, _RLT_REF_OFFSET : _RLT_REF_OFFSET + chunk_len].reshape(
            ref_chunk.shape[0], -1
        )

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

    def _discounted_chunk_rewards(
        self, rewards: torch.Tensor, gamma: float | torch.Tensor
    ) -> torch.Tensor:
        """Discount each chunk's rewards with ``gamma``, a float or ``[B, 1]``."""
        rewards = rewards.reshape(rewards.shape[0], -1)
        rewards = rewards.to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(gamma, device=rewards.device, dtype=torch.float32),
            torch.arange(chunk_len, device=rewards.device, dtype=torch.float32),
        ).to(rewards.dtype)
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

    @staticmethod
    def _record_weight(batch) -> torch.Tensor | None:
        """Per-sample loss inclusion mask ([B, 1], 1/0) based on record_transition."""
        fi = batch.get("forward_inputs") if isinstance(batch, dict) else None
        rec = fi.get("record_transition") if isinstance(fi, dict) else None
        if not isinstance(rec, torch.Tensor):
            return None
        return rec.reshape(rec.shape[0], -1).any(dim=-1, keepdim=True).float()

    @staticmethod
    def _masked_mean(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        if w is None:
            return x.mean()
        w = (
            w.to(device=x.device, dtype=x.dtype)
            .reshape(x.shape[0], *([1] * (x.dim() - 1)))
            .expand_as(x)
        )
        return (x * w).sum() / w.sum().clamp_min(1.0)

    def _execution_weight(self, batch):
        record = self._record_weight(batch)
        if self.cfg.algorithm.get("require_exec_action_validity", False):
            return execution_weight(batch, record)
        return record

    def _takeover_mask(self, batch) -> torch.Tensor | None:
        """[B, 1] chunks that began autonomous and then had a takeover step."""
        if float(self.cfg.algorithm.get("takeover_penalty", 0.0)) == 0.0:
            return None
        fi = (
            batch.get("forward_inputs")
            if isinstance(batch.get("forward_inputs"), dict)
            else {}
        )
        ivk = fi.get("iv_kind", None)
        if not isinstance(ivk, torch.Tensor):
            return None
        ivk = ivk.reshape(ivk.shape[0], -1)
        kinds = self.cfg.algorithm.get("takeover_kinds", None)
        entered = (
            ivk != 0
            if kinds is None
            else torch.isin(
                ivk,
                torch.as_tensor(list(kinds), device=ivk.device, dtype=ivk.dtype),
            )
        )
        padding = fi.get("transition_padding", None)
        if isinstance(padding, torch.Tensor):
            entered = entered & ~padding.reshape_as(ivk).to(ivk.device).bool()
        return (ivk[:, :1] == 0) & entered.any(dim=-1, keepdim=True)

    def _planned_takeovers(
        self, batch
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Takeover chunks whose Q sample is labelled with the policy's plan."""
        if not self.cfg.algorithm.get("takeover_q_uses_policy_action", False):
            return None, None
        fi = (
            batch.get("forward_inputs")
            if isinstance(batch.get("forward_inputs"), dict)
            else {}
        )
        plan = (
            self._critic_actions(self._target_plan(batch))
            if self._executor_grid_enabled()
            else fi.get("policy_action_tcp", fi.get("policy_action", None))
        )
        takeover = self._takeover_mask(batch)
        if not isinstance(plan, torch.Tensor) or takeover is None:
            return None, None
        return takeover, plan

    def _commanded_policy_labels(
        self, batch, actions: torch.Tensor
    ) -> torch.Tensor:
        """Label policy steps (iv_kind == 0) with commanded policy plan in critic space."""
        if self._executor_grid_enabled() and self._action_grid().changes_grid:
            return self._critic_actions(self._target_plan(batch)).to(actions)
        fi = (
            batch.get("forward_inputs")
            if isinstance(batch.get("forward_inputs"), dict)
            else {}
        )
        plan = fi.get("policy_action_tcp", fi.get("policy_action", None))
        ivk = fi.get("iv_kind", None)
        if not isinstance(plan, torch.Tensor) or not isinstance(ivk, torch.Tensor):
            raise ValueError(
                "critic_policy_label=commanded needs policy_action and iv_kind in the replay"
            )
        B = actions.shape[0]
        policy = ivk.reshape(B, -1).to(actions.device) == 0
        pad = fi.get("transition_padding", None)
        if isinstance(pad, torch.Tensor):
            policy = policy & ~pad.reshape(B, -1).to(actions.device).bool()
        steps = policy.shape[1]
        plan = plan.reshape(B, steps, -1).to(device=actions.device, dtype=actions.dtype)
        labels = torch.where(policy[..., None], plan, actions.reshape(B, steps, -1))
        return labels.reshape(actions.shape)

    def _q_sample_weight(
        self, batch, takeover: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Q-sample weight: executed-action validity, plus planned takeover chunks."""
        weight = self._execution_weight(batch)
        if self._executor_grid_enabled() and self._action_grid().changes_grid:
            fi = batch["forward_inputs"]
            auto = fi["iv_kind"].reshape(batch["actions"].shape[0], -1)[:, :1] == 0
            explicit = fi.get("target_exec_action_available")
            known = (
                torch.zeros_like(auto)
                if explicit is None
                else explicit.reshape(auto.shape[0], -1)
                .bool()
                .all(-1, keepdim=True)
                .to(auto.device)
            )
            if weight is not None:
                weight = weight * (auto | known).to(weight)
        if takeover is None or weight is None:
            return weight
        keep = takeover.to(device=weight.device, dtype=weight.dtype)
        record = self._record_weight(batch)
        if record is not None:
            keep = keep * record.to(device=weight.device, dtype=weight.dtype)
        return torch.where(takeover.to(weight.device), keep, weight)

    def _smooth_loss(
        self,
        pi3: torch.Tensor,
        rt3: torch.Tensor,
        dims: list,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Second-difference smoothness penalty on residual (pi - ref_tilde) along time."""
        res = (pi3 - rt3)[..., dims]
        rate = self._action_grid().rate_ratio**2 if self._executor_grid_enabled() else 1.0
        d2 = (res[:, 2:] - 2 * res[:, 1:-1] + res[:, :-2]) * rate
        return self._masked_mean(d2.square().mean(dim=(-1, -2)), sample_weight)

    def _joint_dims(self) -> list:
        mcfg = self.cfg.actor.model
        _, action_dim = self._chunk_shape()
        if bool(mcfg.get("exclude_gripper_residual", False)):
            g = {int(x) for x in mcfg.get("gripper_dims", (6, 13))}
            return [d for d in range(action_dim) if d not in g]
        return list(range(action_dim))

    def _residual_limit(self) -> torch.Tensor | None:
        if not hasattr(self, "_residual_limit_cache"):
            mcfg = self.cfg.actor.model
            scales = mcfg.get("action_scales", None)
            limits = mcfg.get("residual_limits", None)
            self._residual_limit_cache = (
                residual_limit_vector(
                    list(scales),
                    float(mcfg.get("residual_limit_rad", 0.0)),
                    tuple(mcfg.get("gripper_dims", (6, 13))),
                    limits=list(limits) if limits is not None else None,
                )
                if scales is not None
                else None
            )
        return self._residual_limit_cache

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        action_chunk = self._flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
        bc_ref_chunk = self._flatten_chunk(ref_chunk).reshape(
            ref_chunk.shape[0], chunk_len, action_dim
        )

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
        bc_loss = self._masked_mean(bc_error, sample_weight)

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

    def _stage_boundary_not_done(self, curr_obs, next_obs) -> torch.Tensor:
        """Truncate cross-stage bootstrap so each stage's Q stays decoupled."""
        mcfg = getattr(getattr(self, "cfg", None), "actor", None)
        model_cfg = getattr(mcfg, "model", None)
        n_default = os.environ.get("RLT_NUM_STAGES", "8")
        if model_cfg is not None:
            n = int(
                model_cfg.get("stage_onehot_dim", n_default)
                if hasattr(model_cfg, "get")
                else getattr(model_cfg, "stage_onehot_dim", n_default)
            )
        else:
            n = int(n_default)
        if n <= 0:
            return torch.ones(1, 1, dtype=torch.bool, device=self.device)
        try:
            a = curr_obs["proprio"]
            b = next_obs["proprio"]
        except Exception:
            return torch.ones(1, 1, dtype=torch.bool, device=self.device)
        if a.shape[-1] <= n or b.shape[-1] != a.shape[-1]:
            return torch.ones(1, 1, dtype=torch.bool, device=a.device)
        sa = a[..., -n:].reshape(-1, n)
        sb = b[..., -n:].reshape(-1, n)
        same = (sa.argmax(-1) == sb.argmax(-1)) | (sa.sum(-1) == 0) | (sb.sum(-1) == 0)
        return same.reshape(-1, 1)

    def _next_actions_for_critic_target(self, next_obs):
        next_actions, log_pi, dropped = self.model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
        )
        return self._critic_actions(next_actions), log_pi, dropped

    def _penalty_and_rewards(self, batch, gamma: float | torch.Tensor | None = None):
        """Per-chunk (discounted rewards with stage time cost, penalty, takeover)."""
        fi = batch["forward_inputs"]
        rewards = (
            batch["rewards"]
            .reshape(batch["rewards"].shape[0], -1)
            .to(self.torch_dtype)
        )
        B, C = rewards.shape
        mcfg = getattr(self.cfg.actor, "model", None)
        n_default = os.environ.get("RLT_NUM_STAGES", "8")
        n = int(
            mcfg.get("stage_onehot_dim", n_default)
            if hasattr(mcfg, "get")
            else getattr(mcfg, "stage_onehot_dim", n_default)
        )
        stage = self._flatten_chunk(batch["curr_obs"]["proprio"])[:, -n:].argmax(-1)
        T = torch.as_tensor(
            list(self.cfg.algorithm.stage_time_cost_steps),
            device=rewards.device,
            dtype=rewards.dtype,
        )
        real = ~fi["transition_padding"].reshape(B, C).to(rewards.device).bool()
        step_r = (
            rewards.clamp(min=0.0)
            - real.to(rewards.dtype) / T[stage.clamp(max=len(T) - 1)][:, None]
        )
        if gamma is None:
            gamma = (
                self._env_values("gamma", batch.get("env_index"), torch.float32)
                if hasattr(self, "_env_values")
                else self.cfg.algorithm.gamma
            )
        discounts = torch.pow(
            torch.as_tensor(gamma, device=rewards.device, dtype=torch.float32),
            torch.arange(C, device=rewards.device, dtype=torch.float32),
        ).to(rewards.dtype)
        takeover = self._takeover_mask(batch)
        takeover = (
            torch.zeros(B, 1, dtype=torch.bool, device=rewards.device)
            if takeover is None
            else takeover.to(rewards.device).bool()
        )
        failed = (rewards < 0).any(-1, keepdim=True)
        penalty = (failed | takeover).to(rewards.dtype)
        return (step_r * discounts).sum(-1, keepdim=True) - penalty, penalty, takeover

    def _forward_critic_vboot(self, batch):
        """algorithm.critic_target: v_bootstrap."""
        self._validate_action_grid(batch)
        curr_obs, next_obs, fi = (
            batch["curr_obs"],
            batch["next_obs"],
            batch["forward_inputs"],
        )
        chunk_len, _ = self._chunk_shape()
        B = batch["actions"].shape[0]
        gamma = (
            self._env_values("gamma", batch.get("env_index"), torch.float32)
            if hasattr(self, "_env_values")
            else self.cfg.algorithm.gamma
        )
        reward, penalty, takeover = self._penalty_and_rewards(batch, gamma=gamma)
        done = (
            batch["terminations"]
            .reshape(B, -1)
            .bool()
            .any(-1, keepdim=True)
            .to(reward.device)
        )
        not_done = (
            ~done
            & self._stage_boundary_not_done(curr_obs, next_obs).to(reward.device)
            & ~takeover
        )
        with torch.no_grad():
            self.target_model.eval()
            v_next = self.target_model(
                forward_type=ForwardType.SAC_Q, obs=next_obs, value=True
            )
            bootstrap_discount = (
                gamma**chunk_len
                if not isinstance(gamma, torch.Tensor)
                else (gamma**chunk_len).to(device=reward.device, dtype=reward.dtype)
            )
            target = (
                reward
                + not_done.to(reward.dtype)
                * bootstrap_discount
                * v_next.to(reward.dtype)
            )
        data_obs = curr_obs
        noise = float(self.cfg.algorithm.get("critic_z_noise_frac", 0.0))
        if noise > 0 and "z_rl" in curr_obs:
            z = curr_obs["z_rl"]
            data_obs = {
                **curr_obs,
                "z_rl": z + torch.randn_like(z) * z.std(dim=0, keepdim=True) * noise,
            }
        ivk = fi["iv_kind"].reshape(B, chunk_len).to(reward.device)
        auto = (ivk[:, :1] == 0).to(reward.dtype)
        record = self._record_weight(batch)
        record = torch.ones_like(auto) if record is None else record.to(auto)
        valid = self._execution_weight(batch)
        valid = record if valid is None else valid.to(auto)
        q_w = torch.where(takeover, record, valid) * auto
        v_w = (
            valid
            if self.cfg.algorithm.get("critic_value_samples", "all") == "all"
            else q_w
        )
        if self.cfg.algorithm.get("critic_balance_penalized", False):
            pen_m = (penalty > 0).to(q_w.dtype) * q_w
            p = (pen_m.sum() / q_w.sum().clamp_min(1.0)).clamp(0.01, 0.99)
            bt = float(self.cfg.algorithm.get("critic_balance_target", 0.5))
            q_w = q_w * torch.where(penalty > 0, bt / p, (1 - bt) / (1 - p))
        actions = (
            self._target_plan(batch)
            .reshape(B, -1)
            .to(device=reward.device, dtype=reward.dtype)
        )
        chs_mode = self.cfg.algorithm.get("critic_human_samples", "none")
        if self._executor_grid_enabled() and self._action_grid().changes_grid:
            metrics_human_q = 0.0
            hj = fi.get("target_exec_action_joint")
            available = fi.get("target_exec_action_available")
            if (
                chs_mode in ("within_cap", "all")
                and isinstance(hj, torch.Tensor)
                and isinstance(available, torch.Tensor)
            ):
                target_steps, action_dim = self._target_shape()
                hj3 = hj.reshape(B, target_steps, action_dim).to(actions)
                real = ~fi["transition_padding"].reshape(B, chunk_len).to(reward.device).bool()
                moving = (
                    (ivk == 3)
                    & fi["human_bc_available"].reshape(B, chunk_len).to(ivk.device).bool()
                    & real
                )
                frac = moving.sum(1) / real.sum(1).clamp_min(1)
                known = available.reshape(B, target_steps).to(ivk.device).bool().all(1)
                ref3 = (
                    self._ref_chunk(self._with_vla_reference(batch, curr_obs))
                    .reshape_as(hj3)
                    .to(hj3)
                )
                within = (
                    (hj3 - ref_tilde(ref3)).abs() <= self._residual_limit().to(hj3)
                ).all((1, 2))
                human_q = (ivk[:, 0] != 0) & (frac >= 0.8) & known
                if chs_mode == "within_cap":
                    human_q = human_q & within
                actions = torch.where(human_q[:, None], hj3.reshape(B, -1), actions)
                q_w = torch.where(human_q[:, None], valid, q_w)
                metrics_human_q = float(human_q.float().sum())
        elif chs_mode in ("within_cap", "all"):
            hj = fi.get("exec_action_joint", None)
            hb = fi.get("human_bc_available", None)
            limit = self._residual_limit()
            if (
                isinstance(hj, torch.Tensor)
                and isinstance(hb, torch.Tensor)
                and limit is not None
            ):
                real = ~fi["transition_padding"].reshape(B, chunk_len).to(reward.device).bool()
                moving = (ivk == 3) & hb.reshape(B, chunk_len).to(reward.device).bool() & real
                frac = moving.sum(1) / real.sum(1).clamp_min(1)
                hj3 = hj.reshape(B, chunk_len, -1).to(
                    device=reward.device, dtype=reward.dtype
                )
                ref3 = ref_tilde(
                    self._ref_chunk(self._with_vla_reference(batch, curr_obs)).reshape(
                        B, chunk_len, -1
                    )
                ).to(hj3)
                within = ((hj3 - ref3).abs() <= limit.to(hj3)).all(-1) | ~real
                human_q = (
                    (ivk[:, 0] != 0)
                    & (frac >= 0.8)
                    & (within.all(1) if chs_mode == "within_cap" else True)
                )
                actions = torch.where(human_q[:, None], hj3.reshape(B, -1), actions)
                q_w = torch.where(human_q[:, None], valid, q_w)
                metrics_human_q = float(human_q.float().sum())
            else:
                metrics_human_q = -1.0
        else:
            metrics_human_q = 0.0
        v = self.model(forward_type=ForwardType.SAC_Q, obs=data_obs, value=True)
        target = target.to(v.dtype)
        v_loss = self._masked_mean(torch.square(v - target), v_w)
        penalty_only = bool(getattr(self.model, "critic_penalty_only", False))
        if penalty_only:
            v = v.detach()
            v_loss = torch.zeros_like(v_loss)
        penalty_head = bool(self.cfg.actor.model.get("critic_penalty_head", False))
        metrics = {"human_q_samples": metrics_human_q}
        if penalty_head:
            logits = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=data_obs,
                actions=actions,
                penalty_logits=True,
            )
            prior = self.model(
                forward_type=ForwardType.SAC_Q, obs=data_obs, penalty_prior=True
            )
            label = (penalty > 0).to(logits.dtype)
            if (
                int(self.cfg.algorithm.get("critic_penalty_window", 1)) >= 2
                or float(self.cfg.algorithm.get("critic_penalty_decay", 0.0)) > 0
            ) and isinstance(fi.get("penalty_next"), torch.Tensor):
                label = torch.maximum(
                    label,
                    fi["penalty_next"]
                    .reshape(B, 1)
                    .to(device=label.device, dtype=label.dtype)
                    .clamp(0, 1),
                )
            eps = float(self.cfg.algorithm.get("critic_penalty_label_smooth", 0.0))
            if eps > 0:
                label = label * (1 - 2 * eps) + eps
            p_loss = self._masked_mean(
                F.binary_cross_entropy_with_logits(
                    logits, label.expand_as(logits), reduction="none"
                ),
                q_w,
            )
            prior_loss = self._masked_mean(
                F.binary_cross_entropy_with_logits(prior, label, reduction="none"),
                q_w,
            )
            q_loss = p_loss + prior_loss
            pen2q = getattr(self.model, "penalty_to_q", torch.sigmoid)
            with torch.no_grad():
                q = (
                    -pen2q(logits)
                    if penalty_only
                    else v + pen2q(prior) - pen2q(logits)
                )
            metrics.update(
                {"p_loss": p_loss.item(), "prior_loss": prior_loss.item()}
            )
        else:
            q = self.model(
                forward_type=ForwardType.SAC_Q, obs=data_obs, actions=actions
            )
            q_loss = self._masked_mean(torch.square(q - target.expand_as(q)), q_w)
        alpha = float(self.cfg.algorithm.get("critic_conservative_alpha", 0.0))
        cons_gap = None
        if alpha > 0:
            with torch.no_grad():
                pi, _, _ = self.model(
                    forward_type=ForwardType.SAC,
                    obs=self._with_vla_reference(batch, curr_obs),
                    deterministic=True,
                )
            pi = self._critic_actions(pi)
            has = (q_w > 0).to(q_w.dtype)
            if penalty_head:
                p_pi = pen2q(
                    self.model(
                        forward_type=ForwardType.SAC_Q,
                        obs=data_obs,
                        actions=pi,
                        penalty_logits=True,
                    )
                )
                gap = pen2q(logits).detach() - p_pi
            else:
                gap = (
                    self.model(
                        forward_type=ForwardType.SAC_Q, obs=data_obs, actions=pi
                    )
                    - q.detach()
                )
            q_loss = q_loss + alpha * self._masked_mean(F.relu(gap), has)
            cons_gap = float(self._masked_mean(gap.detach(), has))
        with torch.no_grad():
            adv = q[:, :1] - v
            pen = (penalty > 0) & (q_w > 0)
            ok = (penalty == 0) & (q_w > 0)
            metrics.update(
                {
                    "q_data": float(self._masked_mean(q[:, :1], q_w)),
                    "v_data": float(self._masked_mean(v, v_w)),
                    "td_target": float(self._masked_mean(target, q_w)),
                    "q_loss": float(q_loss),
                    "v_loss": float(v_loss),
                    "penalized_ratio": float(pen.float().sum() / q_w.sum().clamp_min(1.0)),
                    "adv_penalized": float(adv[pen].mean()) if pen.any() else 0.0,
                    "adv_ok": float(adv[ok].mean()) if ok.any() else 0.0,
                }
            )
            if penalty_head:
                phat = torch.sigmoid(logits).amax(-1, keepdim=True)
                metrics["phat_penalized"] = float(phat[pen].mean()) if pen.any() else 0.0
                metrics["phat_ok"] = float(phat[ok].mean()) if ok.any() else 0.0
            if cons_gap is not None:
                metrics["q_pi_minus_q_data"] = cons_gap
        if self.cfg.algorithm.get("correction_metrics", False):
            metrics.update(self._correction_metrics(batch, curr_obs))
        return q_loss + v_loss, metrics

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        self._validate_action_grid(batch)
        if self.cfg.algorithm.get("critic_target", "q_bootstrap") == "v_bootstrap":
            return self._forward_critic_vboot(batch)
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        critic_obs = self._with_critic_plan(self._with_vla_reference(batch, curr_obs))
        next_critic_obs = self._with_critic_plan(next_obs)
        actions = (
            self._critic_actions(self._target_plan(batch))
            if self._executor_grid_enabled() and self._action_grid().changes_grid
            else batch["actions"]
        )
        planned, plan = self._planned_takeovers(batch)
        if planned is not None:
            plan = plan.reshape(actions.shape).to(
                device=actions.device, dtype=actions.dtype
            )
            pick = planned.to(actions.device).reshape(-1, *([1] * (actions.dim() - 1)))
            actions = torch.where(pick, plan, actions)
        if self.cfg.algorithm.get("critic_policy_label", "measured") == "commanded":
            actions = self._commanded_policy_labels(batch, actions)
        rewards = batch["rewards"]
        done_source = batch["terminations"]
        if use_simulator_transition_replay(self.cfg):
            done_source = batch["dones"]
        done_source = done_source.to(self.torch_dtype)
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )
        not_done = not_done & self._stage_boundary_not_done(curr_obs, next_obs).to(
            not_done.device
        )

        with torch.no_grad():
            next_actions, _, _ = self._next_actions_for_critic_target(next_obs)

            if not use_crossq:
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_critic_obs,
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

            gamma = (
                self._env_values("gamma", batch.get("env_index"), torch.float32)
                if hasattr(self, "_env_values")
                else self.cfg.algorithm.gamma
            )
            reward_target = self._discounted_chunk_rewards(rewards, gamma)
            tp = float(self.cfg.algorithm.get("takeover_penalty", 0.0))
            takeover = self._takeover_mask(batch)
            if takeover is not None:
                takeover = takeover.to(reward_target.device)
                reward_target = reward_target + tp * takeover.to(reward_target.dtype)
                not_done = not_done & ~takeover.to(not_done.device)
                self._last_takeover_ratio = float(takeover.float().mean())
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = gamma**reward_horizon
            if isinstance(bootstrap_discount, torch.Tensor):
                bootstrap_discount = bootstrap_discount.to(self.torch_dtype)
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")
            q_target_min = self.cfg.algorithm.get("q_target_min", None)
            q_target_max = self.cfg.algorithm.get("q_target_max", None)
            if q_target_min is not None or q_target_max is not None:
                target_q_values = torch.clamp(
                    target_q_values,
                    min=None if q_target_min is None else float(q_target_min),
                    max=None if q_target_max is None else float(q_target_max),
                )

        if not use_crossq:
            data_obs = critic_obs
            critic_z_noise = float(self.cfg.algorithm.get("critic_z_noise_frac", 0.0))
            if critic_z_noise > 0 and "z_rl" in critic_obs:
                z = critic_obs["z_rl"]
                data_obs = {
                    **critic_obs,
                    "z_rl": z
                    + torch.randn_like(z) * z.std(dim=0, keepdim=True) * critic_z_noise,
                }
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=data_obs,
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
        sample_weight = self._q_sample_weight(batch, planned)
        critic_loss = self._masked_mean(
            torch.square(
                all_data_q_values - target_q_values.expand_as(all_data_q_values)
            ),
            sample_weight,
        )
        metrics = {"q_data": all_data_q_values.mean().item()}
        if (
            self.cfg.algorithm.get("require_exec_action_validity", False)
            and sample_weight is not None
        ):
            metrics["critic_valid_ratio"] = sample_weight.mean().item()
        if takeover is not None:
            metrics["takeover_ratio"] = self._last_takeover_ratio
        if self.cfg.algorithm.get("correction_metrics", False) and not use_crossq:
            metrics.update(self._correction_metrics(batch, critic_obs))
        return critic_loss, metrics

    @torch.no_grad()
    def _correction_metrics(self, batch, obs: dict) -> dict:
        if not self._joint_actor():
            return {}
        if self._executor_grid_enabled() and self._action_grid().changes_grid:
            return {"correction_metrics_requires_target_clock_labels": 1.0}
        fi = (
            batch.get("forward_inputs")
            if isinstance(batch.get("forward_inputs"), dict)
            else {}
        )
        keys = (
            "iv_kind",
            "transition_padding",
            "human_bc_available",
            "exec_action_joint",
            "ref_chunk",
        )
        if any(not isinstance(fi.get(k), torch.Tensor) for k in keys):
            return {}
        chunk_len, action_dim = self._chunk_shape()
        B = batch["actions"].shape[0]
        dev = batch["actions"].device
        ivk = fi["iv_kind"].reshape(B, chunk_len).to(dev)
        real = ~fi["transition_padding"].reshape(B, chunk_len).to(dev).bool()
        moving = (
            (ivk == 3) & fi["human_bc_available"].reshape(B, chunk_len).to(dev).bool() & real
        )
        frac = moving.sum(1) / real.sum(1).clamp_min(1)
        idx = torch.nonzero((ivk[:, 0] != 0) & (frac >= 0.8)).reshape(-1)
        n = int(idx.numel())
        if n < 2:
            return {}
        sub = {
            k: v[idx] if torch.is_tensor(v) and v.shape[0] == B else v
            for k, v in self._with_vla_reference(batch, obs).items()
        }
        plan = ref_tilde(
            self._ref_chunk(sub).reshape(n, chunk_len, action_dim).to(torch.float32)
        )
        human = fi["exec_action_joint"].reshape(B, chunk_len, action_dim)[idx].to(
            dev, torch.float32
        )

        def q(joints):
            a = self._critic_actions(joints.reshape(n, -1))
            return self._min_twin_q(
                self.model(forward_type=ForwardType.SAC_Q, obs=sub, actions=a)
            )[:, 0]

        grippers = set(int(g) for g in self.cfg.actor.model.get("gripper_dims", ()))
        arm = torch.tensor([d not in grippers for d in range(action_dim)], device=dev)
        d = (human - plan) * arm
        g = torch.Generator(device=dev).manual_seed(0)
        u = torch.randn(n, 1, action_dim, device=dev, generator=g) * arm
        rnd = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9) * d.norm(
            dim=-1, keepdim=True
        )
        q_plan, q_human = q(plan), q(human)
        q_flip = q((2 * plan - human).clamp(-1, 1))
        q_rand = q((plan + rnd).clamp(-1, 1))
        out = {
            "hil/corr_n": float(n),
            "hil/corr_q_gain": float((q_human - q_plan).mean()),
            "hil/corr_q_gain_pos": float((q_human > q_plan).float().mean()),
            "hil/corr_q_vs_flip": float((q_human - q_flip).mean()),
            "hil/corr_q_vs_flip_pos": float((q_human > q_flip).float().mean()),
            "hil/corr_q_vs_rand": float((q_human - q_rand).mean()),
            "hil/corr_q_vs_rand_pos": float((q_human > q_rand).float().mean()),
        }
        scale = self.cfg.actor.model.get("action_scales", None)
        bias = self.cfg.actor.model.get("action_bias", None)
        if scale is not None and bias is not None:
            cur = (
                self._flatten_chunk(sub["proprio"])[:, :action_dim].to(torch.float32)
                - torch.tensor(list(bias), device=dev)
            ) / torch.tensor(list(scale), device=dev)
            q_stop = q(torch.where(arm, cur.clamp(-1, 1)[:, None, :], plan))
            out["hil/corr_q_vs_stop"] = float((q_human - q_stop).mean())
            out["hil/corr_q_vs_stop_pos"] = float((q_human > q_stop).float().mean())
        limit = self._residual_limit()
        if limit is not None:
            cap = limit.to(d)[arm]

            def capped(x):
                peak = x[..., arm].abs().amax(1).clamp_min(1e-9)
                return x * (cap / peak).amin(1).clamp(max=1.0)[:, None, None]

            q_step = q((plan + capped(d)).clamp(-1, 1))
            q_step_rand = q((plan + capped(rnd)).clamp(-1, 1))
            out["hil/corr_q_step"] = float((q_step - q_plan).mean())
            out["hil/corr_q_step_pos"] = float((q_step > q_plan).float().mean())
            out["hil/corr_q_step_vs_rand"] = float((q_step - q_step_rand).mean())
            out["hil/corr_q_step_vs_rand_pos"] = float(
                (q_step > q_step_rand).float().mean()
            )
        return out

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        self._validate_action_grid(batch)
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"

        curr_obs = batch["curr_obs"]
        fi = (
            batch.get("forward_inputs")
            if isinstance(batch.get("forward_inputs"), dict)
            else {}
        )
        chunk_len, action_dim = self._target_shape()
        B = batch["actions"].shape[0]
        schema2 = isinstance(fi.get("data_schema"), torch.Tensor) and bool(
            (fi["data_schema"].reshape(-1) == 2).all()
        )
        curr_obs = self._with_vla_reference(batch, curr_obs)
        actor_obs = curr_obs
        z_noise = float(self.cfg.algorithm.get("z_noise_frac", 0.0))
        if z_noise > 0 and "z_rl" in curr_obs:
            z = curr_obs["z_rl"]
            actor_obs = {
                **curr_obs,
                "z_rl": z + torch.randn_like(z) * z.std(dim=0, keepdim=True) * z_noise,
            }
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, dropped = self.model(
            forward_type=ForwardType.SAC,
            obs=actor_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        pi_critic = self._critic_actions(pi)
        if not use_crossq:
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=self._with_critic_plan(curr_obs),
                actions=pi_critic,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi_critic,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        num_q_values = all_qf_pi.shape[-1]
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(num_q_values)
        }
        qf_pi = (
            self._min_twin_q(all_qf_pi)
            if bool(self.cfg.algorithm.get("actor_uses_min_q", True))
            else self._q1(all_qf_pi)
        )
        metrics["q_pi"] = qf_pi.mean().item()

        rec_w = self._record_weight(batch)
        ref_chunk = self._ref_chunk(curr_obs)
        ref3 = self._flatten_chunk(ref_chunk).reshape(B, chunk_len, action_dim)
        pi3 = self._flatten_chunk(pi).reshape(B, chunk_len, action_dim)
        dims = self._joint_dims()
        rt3 = ref_tilde(ref3)

        if (
            self._joint_actor()
            and self.cfg.algorithm.get("human_bc_target", "fitted") != "direct"
        ):
            raise ValueError("a joint actor needs algorithm.human_bc_target=direct")
        if self.cfg.algorithm.get("human_bc_target", "fitted") == "direct":
            penalized = None
            if self.cfg.algorithm.get("bc_skip_penalized_chunks", False):
                failed = (batch["rewards"].reshape(B, -1) < 0).any(-1, keepdim=True)
                taken = self._takeover_mask(batch)
                taken = (
                    torch.zeros_like(failed)
                    if taken is None
                    else taken.to(failed.device).bool().reshape(B, 1)
                )
                penalized = failed | taken
                metrics["bc_penalized_ratio"] = penalized.float().mean().item()
                metrics["bc_takeover_skip_ratio"] = taken.float().mean().item()
            bc_loss, direct_metrics = direct_bc_loss(
                self._action_grid().to_env(pi3),
                self._env_reference(curr_obs),
                fi,
                rec_w,
                human_weight=float(self.cfg.algorithm.get("bc_human_weight_mult", 1.0)),
                skip_auto=penalized,
                human_within=(
                    self._residual_limit()
                    if self.cfg.algorithm.get("human_bc_within_cap", False)
                    else None
                ),
            )
            metrics.update(direct_metrics)
        elif schema2 and isinstance(fi.get("iv_kind"), torch.Tensor):
            act3 = self._flatten_chunk(batch["actions"]).reshape(
                B, chunk_len, action_dim
            )
            iv_kind = fi["iv_kind"].reshape(B, -1).to(pi.device)
            sw = fi.get("switch_step", None)
            sw = (
                sw.reshape(B, -1).to(pi.device)
                if isinstance(sw, torch.Tensor)
                else None
            )
            oh_dim = int(self.cfg.actor.model.get("stage_onehot_dim", 8))
            st_c = self._flatten_chunk(curr_obs["proprio"])[:, -oh_dim:].argmax(-1)
            st_n = (
                self._flatten_chunk(batch["next_obs"]["proprio"])[:, -oh_dim:]
                .argmax(-1)
                .to(st_c.device)
            )
            target, step_w, hinfo = build_bc_targets(
                ref3,
                act3,
                iv_kind,
                sw,
                st_c != st_n,
                self._residual_limit(),
                human_weight=float(self.cfg.algorithm.get("bc_human_weight_mult", 1.0)),
                clip_frac=float(self.cfg.algorithm.get("human_target_clip_frac", 0.8)),
                min_run=int(self.cfg.algorithm.get("human_fit_min_run", 5)),
                guard_steps=int(self.cfg.algorithm.get("human_guard_steps", 5)),
                trim_still=bool(self.cfg.algorithm.get("human_trim_still", False)),
                still_eps=float(self.cfg.algorithm.get("human_still_eps", 1e-3)),
                still_edge_min=int(self.cfg.algorithm.get("human_still_edge_min", 8)),
            )
            bc_err = (pi3 - target)[..., dims].square().mean(-1)
            w = step_w * (rec_w.to(step_w.dtype) if rec_w is not None else 1.0)
            bc_loss = (bc_err * w).sum() / w.sum().clamp_min(1.0)
            active = (iv_kind == KIND_ACTIVE) & (w > 0)
            auto = (iv_kind == KIND_AUTO) & (w > 0)
            metrics.update(
                {
                    "bc_loss": bc_loss.detach().item(),
                    "bc_ref_loss": (
                        bc_err[auto].mean() if auto.any() else bc_err.new_zeros(())
                    )
                    .detach()
                    .item(),
                    "bc_human_loss": (
                        bc_err[active].mean() if active.any() else bc_err.new_zeros(())
                    )
                    .detach()
                    .item(),
                    "bc_human_vla_baseline": (
                        (rt3 - target)[..., dims].square().mean(-1)[active].mean()
                        if active.any()
                        else bc_err.new_zeros(())
                    )
                    .detach()
                    .item(),
                    "human_mask_ratio": hinfo["active_ratio"],
                    "hold_ratio": hinfo["hold_ratio"],
                    "ramp_ratio": hinfo["ramp_ratio"],
                    "still_trim_ratio": hinfo["still_trim_ratio"],
                    "human_bc_steps": float(hinfo["human_bc_steps"]),
                    "policy_mask_ratio": 1.0
                    - hinfo["active_ratio"]
                    - hinfo["hold_ratio"]
                    - hinfo["ramp_ratio"],
                }
            )
        else:
            bc_loss, rlt_metrics = self._bc_metrics(
                pi=pi,
                actions=batch["actions"],
                ref_chunk=ref_chunk,
                intervene_flags=batch.get("intervene_flags", None),
                sample_weight=rec_w,
            )
            metrics.update(rlt_metrics)

        entropy = -log_pi.mean()
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        q_w = self._q_sample_weight(batch, self._planned_takeovers(batch)[0])
        actor_loss = -q_weight * self._masked_mean(qf_pi, q_w) + bc_weight * bc_loss
        smooth_weight = float(self.cfg.algorithm.get("smooth_weight", 0.0))
        if smooth_weight > 0:
            sm_w = torch.ones(B, 1, device=pi.device, dtype=pi.dtype)
            if q_w is not None:
                sm_w = sm_w * q_w.to(pi.device, pi.dtype)
            if dropped is not None:
                sm_w = sm_w * (~dropped).to(pi.dtype).reshape(B, 1)
            smooth_loss = self._smooth_loss(pi3, rt3, dims, sm_w)
            actor_loss = actor_loss + smooth_weight * smooth_loss
            metrics["smooth_loss"] = smooth_loss.detach().item()
            metrics["weighted_smooth"] = (smooth_weight * smooth_loss).detach().item()
        metrics.update(weight_metrics)
        with torch.no_grad():
            res_joint = (pi3 - rt3)[..., dims].abs()
            if dropped is not None and (~dropped).any():
                res_joint = res_joint[~dropped]
            metrics["action_ref_abs_mean"] = res_joint.mean().item()
            metrics["residual_joint_p99"] = (
                torch.quantile(res_joint.reshape(-1).float(), 0.99).item()
                if res_joint.numel() > 0
                else 0.0
            )
        metrics["weighted_q"] = (q_weight * self._masked_mean(qf_pi, q_w)).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["reference_dropout_prob"] = reference_dropout_prob
        metrics["schema2"] = float(schema2)

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
        env_cfg = self._rollout_geometry.layout.spec(trajectory.env_name).cfg
        auto_reset = bool(env_cfg.get("auto_reset", False))

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
                    env_name=trajectory.env_name,
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
                        "TrajectoryCollector populated transition observations "
                        f"before replay ingestion, got row index {idx}."
                    )
                transition.curr_obs = curr_obs

                # Dones have one extra initial slot, so transition t reads
                # terminal flags from t+1. Rewards are already action-aligned
                # by the trajectory accumulator because the initial empty reward
                # is skipped and the final reward is appended after rollout.
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
                        "transitions. Ensure TrajectoryCollector populated "
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

    def _penalty_window_reach(self) -> int:
        """Chunks beyond the current one that the critic's penalty label looks at (0: none)."""
        cfg = getattr(self, "cfg", None)
        algo = getattr(cfg, "algorithm", None) if cfg is not None else None
        if algo is None:
            return 0
        K = int(algo.get("critic_penalty_window", 1))
        KF = int(algo.get("critic_failure_window", K))
        decay = float(algo.get("critic_penalty_decay", 0.0))
        if decay > 0:
            return int(math.ceil(math.log(0.02) / math.log(decay)))
        return max(K, KF) - 1

    def _penalty_events(self, traj: Trajectory):
        """Per chunk [T, B]: takeover (began autonomous, then H / Pico), step failure (-1), episode end."""
        fi = getattr(traj, "forward_inputs", None)
        if (
            not isinstance(fi, dict)
            or not isinstance(fi.get("iv_kind"), torch.Tensor)
            or not isinstance(traj.rewards, torch.Tensor)
        ):
            return None
        T, B = int(traj.rewards.shape[0]), int(traj.rewards.shape[1])
        ivk = fi["iv_kind"][:T].reshape(T, B, -1)
        kinds = self.cfg.algorithm.get("takeover_kinds", None)
        entered = (
            ivk != 0
            if kinds is None
            else torch.isin(
                ivk, torch.as_tensor(list(kinds), device=ivk.device, dtype=ivk.dtype)
            )
        )
        pad = fi.get("transition_padding", None)
        if isinstance(pad, torch.Tensor):
            entered = entered & ~pad[:T].reshape(T, B, -1).to(ivk.device).bool()
        take = (ivk[..., 0] == 0) & entered.any(-1)
        fail = (traj.rewards.reshape(T, B, -1) < 0).any(-1).to(take.device)
        dones = getattr(traj, "dones", None)
        if isinstance(dones, torch.Tensor) and int(dones.shape[0]) >= T + 1:
            ends = (
                dones.reshape(int(dones.shape[0]), B, -1)
                .bool()
                .any(-1)[1 : T + 1]
                .to(take.device)
            )
        else:
            ends = torch.zeros(T, B, dtype=torch.bool, device=take.device)
        return take, fail, ends

    def _attach_penalty_window(
        self, traj: Trajectory, lookahead: Trajectory | None = None
    ) -> None:
        ev = self._penalty_events(traj)
        reach = self._penalty_window_reach()
        if ev is None or reach <= 0:
            return
        take, fail, ends = ev
        T, B = take.shape
        la = self._penalty_events(lookahead) if lookahead is not None else None
        if la is not None and la[0].shape[1] == B:
            take = torch.cat([take, la[0][:reach]])
            fail = torch.cat([fail, la[1][:reach]])
            ends = torch.cat([ends, la[2][:reach]])
        K = int(self.cfg.algorithm.get("critic_penalty_window", 1))
        KF = int(self.cfg.algorithm.get("critic_failure_window", K))
        decay = float(self.cfg.algorithm.get("critic_penalty_decay", 0.0))
        if decay > 0:
            K = KF = reach + 1
        nxt = torch.zeros(T, B, dtype=torch.float32, device=take.device)
        for t in range(T):
            alive = ~ends[t]
            for u in range(t + 1, min(int(take.shape[0]), t + reach + 1)):
                hit = torch.zeros_like(alive)
                if u - t < K:
                    hit |= take[u]
                if u - t < KF:
                    hit |= fail[u]
                w = decay ** (u - t) if decay > 0 else 1.0
                nxt[t] = torch.maximum(nxt[t], (alive & hit).float() * w)
                alive = alive & ~ends[u]
        out = nxt.to(traj.rewards.dtype).unsqueeze(-1)
        extra = int(traj.forward_inputs["iv_kind"].shape[0]) - T
        if extra > 0:
            out = torch.cat(
                [out, torch.zeros(extra, B, 1, dtype=out.dtype, device=out.device)]
            )
        traj.forward_inputs["penalty_next"] = out

    def _wrap_replay_flatten(self, buffer) -> None:
        if (
            buffer is None
            or getattr(buffer, "_rlt_penalty_wrapped", False)
            or not hasattr(buffer, "_flatten_trajectory")
        ):
            return
        orig = buffer._flatten_trajectory

        def flatten(trajectory, *args, **kwargs):
            fi = getattr(trajectory, "forward_inputs", None)
            if (
                isinstance(fi, dict)
                and "penalty_next" not in fi
                and self._penalty_window_reach() > 0
            ):
                self._attach_penalty_window(trajectory)
            return orig(trajectory, *args, **kwargs)

        buffer._flatten_trajectory = flatten
        buffer._rlt_penalty_wrapped = True

    def _label_and_delay_rollouts(
        self, recv_list: list[Trajectory]
    ) -> list[Trajectory]:
        if self._penalty_window_reach() <= 0 or not recv_list:
            return recv_list
        trajs = [t for t in recv_list if isinstance(t, Trajectory)]
        single_env = all(
            isinstance(t.rewards, torch.Tensor) and int(t.rewards.shape[1]) == 1
            for t in trajs
        )
        if not single_env or len(trajs) != len(recv_list):
            for t in trajs:
                self._attach_penalty_window(t)
            return recv_list
        pending = getattr(self, "_pending_rollout", None)
        chain = ([pending] if pending is not None else []) + trajs
        for i in range(len(chain) - 1):
            self._attach_penalty_window(chain[i], lookahead=chain[i + 1])
        self._pending_rollout = chain[-1]
        return chain[:-1]

    def _ingest_rollout_trajectories(
        self,
        recv_list: list[Trajectory],
    ) -> tuple[int, int]:
        self._last_replay_metrics = {}
        recv_list = self._label_and_delay_rollouts(recv_list)
        if not recv_list:
            return 0, 0

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
        self._wrap_replay_flatten(getattr(self, "replay_buffer", None))

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel):
        clear_memory(sync=False)

        split_num = self._rollout_geometry.shards_per_actor()

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
        replay_size = self.replay_buffer.min_env_samples
        minimums = all_reduce_dict(
            {
                "min_replay_size": float(replay_size),
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
        buffer_ready = counters["min_replay_size"] >= max(min_buffer_size, 1)
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
            metrics_data = self.update_one_epoch(train_actor=update_actor)
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
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0
        self._utd_ratio = float(cfg.algorithm.get("utd_ratio", 0.0) or 0.0)
        self._warmup_transitions = int(cfg.algorithm.get("warmup_transitions", 0) or 0)
        self._ingested_transitions = 0
        self._utd_updates_done = 0
        self._staleness_discarded_trajectories = 0
        self._staleness_discarded_transitions = 0
        self._staleness_missing_versions = 0

    @staticmethod
    def _trajectory_transition_count_for_staleness(traj: Trajectory) -> int:
        actions = getattr(traj, "actions", None)
        if not isinstance(actions, torch.Tensor) or actions.ndim < 2:
            return 0
        return int(actions.shape[0] * actions.shape[1])

    def _filter_stale_rollout_trajectories(
        self, recv_list: list[Trajectory]
    ) -> tuple[list[Trajectory], dict[str, float]]:
        """Drop rollout batches older than the configured actor version window."""
        threshold_cfg = self.cfg.algorithm.get("staleness_threshold", None)
        current_version = int(self.get_rollout_sync_version())
        metrics = {
            "rlt/staleness_current_version": float(current_version),
            "rlt/staleness_threshold": (
                float(threshold_cfg) if threshold_cfg is not None else -1.0
            ),
        }
        if threshold_cfg is None:
            return recv_list, metrics

        threshold = int(threshold_cfg)
        if threshold < 0:
            raise ValueError(
                f"algorithm.staleness_threshold must be >= 0, got {threshold}"
            )

        accepted: list[Trajectory] = []
        discarded = 0
        discarded_transitions = 0
        missing_versions = 0
        accepted_min_version: float | None = None
        for traj in recv_list:
            versions = getattr(traj, "versions", None)
            if not isinstance(versions, torch.Tensor) or versions.numel() == 0:
                missing_versions += 1
                discarded += 1
                discarded_transitions += (
                    self._trajectory_transition_count_for_staleness(traj)
                )
                continue
            min_version = float(versions.detach().float().min().item())
            if current_version - min_version > threshold:
                discarded += 1
                discarded_transitions += (
                    self._trajectory_transition_count_for_staleness(traj)
                )
                continue
            accepted.append(traj)
            accepted_min_version = (
                min_version
                if accepted_min_version is None
                else min(accepted_min_version, min_version)
            )

        self._staleness_discarded_trajectories += discarded
        self._staleness_discarded_transitions += discarded_transitions
        self._staleness_missing_versions += missing_versions
        metrics.update(
            {
                "rlt/staleness_discarded_trajectories": float(
                    self._staleness_discarded_trajectories
                ),
                "rlt/staleness_discarded_transitions": float(
                    self._staleness_discarded_transitions
                ),
                "rlt/staleness_missing_versions": float(
                    self._staleness_missing_versions
                ),
                "rlt/staleness_accepted_min_version": (
                    -1.0 if accepted_min_version is None else accepted_min_version
                ),
            }
        )
        return accepted, metrics

    def setup_sac_components(self):
        """Initialize replay components and let RLT schedule own readiness."""
        super().setup_sac_components()
        if self.use_rlt_schedule:
            self.buffer_dataset.min_replay_buffer_size = 1
        self._preloaded_transitions = int(
            getattr(self.replay_buffer, "total_samples", 0) or 0
        )
        self._wrap_replay_flatten(getattr(self, "replay_buffer", None))

    def _global_rlt_counters(self) -> dict[str, float]:
        """Reuse the shared sync/async RLT counter reduction contract."""
        return RLTACFSDPPolicy._global_rlt_counters(self)

    def _rlt_updates_to_run(self) -> tuple[int, dict[str, float]]:
        """Reuse the shared sync/async RLT update-budget calculation."""
        return RLTACFSDPPolicy._rlt_updates_to_run(self)

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

        accepted, staleness_metrics = self._filter_stale_rollout_trajectories(
            recv_list
        )
        if not accepted:
            self._last_replay_metrics = {
                **getattr(self, "_last_replay_metrics", {}),
                **staleness_metrics,
            }
            return

        added, completed = self._ingest_rollout_trajectories(accepted)
        self._update_rollout_ingest_counters(added, completed)
        self._ingested_transitions += int(added)
        self._last_replay_metrics = {
            **getattr(self, "_last_replay_metrics", {}),
            **staleness_metrics,
        }

    async def run_training(self):
        if self._utd_ratio > 0:
            update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
            while True:
                self._drain_received_trajectories(
                    max_trajectories=self.cfg.actor.get(
                        "recv_drain_max_trajectories", 1024
                    )
                )
                preloaded = getattr(self, "_preloaded_transitions", 0)
                total = self._ingested_transitions + preloaded
                if total >= self._warmup_transitions:
                    preload_utd = float(
                        self.cfg.algorithm.get("preload_utd_ratio", self._utd_ratio)
                    )
                    budget = (
                        self._utd_ratio * self._ingested_transitions
                        + preload_utd * preloaded
                        - self._utd_updates_done
                    )
                    if budget >= update_epoch:
                        break
                await asyncio.sleep(0.5)
            self._utd_updates_done += update_epoch

        if self.use_rlt_schedule and self._utd_ratio <= 0:
            if self.cfg.actor.get("enable_offload", False):
                self.load_param_and_grad(self.device)
                self.load_optimizer(self.device)

            while True:
                self._drain_received_trajectories(
                    max_trajectories=self.cfg.actor.get(
                        "recv_drain_max_trajectories", 1024
                    )
                )
                updates_to_run, schedule_metrics = self._rlt_updates_to_run()
                if updates_to_run > 0:
                    break
                await asyncio.sleep(0.5)

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

            torch.distributed.barrier()
            self.model.train()
            metrics = {}
            critic_updates_run = 0
            actor_updates_run = 0
            for _ in range(updates_to_run):
                await asyncio.sleep(0)
                update_actor = (
                    int(self.update_step) % int(self.critic_actor_ratio) == 0
                )
                metrics_data = self.update_one_epoch(train_actor=update_actor)
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

            Worker.torch_platform.synchronize()
            torch.distributed.barrier()
            Worker.torch_platform.empty_cache()
            replay_metrics = getattr(self, "_last_replay_metrics", {})
            if replay_metrics:
                mean_metric_dict = {**mean_metric_dict, **replay_metrics}
            return mean_metric_dict

        mean_metric_dict = await super().run_training()
        if self._utd_ratio > 0:
            mean_metric_dict = {
                **mean_metric_dict,
                "rlt/ingested_transitions": float(self._ingested_transitions),
                "rlt/preloaded_transitions": float(
                    getattr(self, "_preloaded_transitions", 0)
                ),
                "rlt/utd_updates_done": float(self._utd_updates_done),
            }
        replay_metrics = getattr(self, "_last_replay_metrics", {})
        if replay_metrics:
            mean_metric_dict = {**mean_metric_dict, **replay_metrics}
        return mean_metric_dict
