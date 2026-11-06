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

"""SuperMimic policy: motion and velocity tracking with a learned world model.

One module holds every network so that a single FSDP unit shards them and a
single weight sync ships the policy, critics and input norms to the rollout
worker. The rollout worker only calls :meth:`SuperMimicPolicy.act`,
:meth:`SuperMimicPolicy.critic_value` and :meth:`predict_action_batch`. The
actor worker calls :meth:`forward` with one ``ForwardType`` per training phase.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from rlinf.algorithms.supermimic.objectives import (
    actor_objective,
    state_prediction_loss,
    termination_bce,
)
from rlinf.algorithms.supermimic.scan import (
    CompileCache,
    make_dynamics_window,
    make_imagination,
)
from rlinf.envs.sim.g1_mimic.constants import (
    ANCHOR_DIM,
    GOAL_REF_DIM,
    HISTORY_LEN,
    OBS_DIM,
    critic_input_dim,
)
from rlinf.envs.sim.g1_mimic.features import integrate_residual
from rlinf.envs.sim.g1_mimic.g1_fk import set_kinematics_compile
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_failure_prob,
    analytic_reward,
    compose_goal,
    differentiable_fk,
    failure_signals,
    rollout_critic_feature,
    termination_head_input,
)
from rlinf.envs.sim.g1_mimic.velocity import (
    ALIGN_WEIGHT,
    normalize_command,
    velocity_fall_soft,
    velocity_reward,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.supermimic.networks import (
    CriticNet,
    DynamicsNet,
    MotionDecoder,
    MotionEncoder,
    TerminationHead,
    VelocityCritic,
    VelocityEncoder,
    running_state_loss,
)

GOAL730_DIM = 730

# Submodules each training phase updates; every other parameter is frozen.
PHASE_MODULES = {
    ForwardType.SUPERMIMIC_WORLD_MODEL: ("dynamics", "termination"),
    ForwardType.SUPERMIMIC_ACTOR: ("encoder", "decoder"),
    ForwardType.SUPERMIMIC_CRITIC: ("critic",),
    ForwardType.SUPERMIMIC_VELOCITY_ACTOR: ("velocity_encoder",),
    ForwardType.SUPERMIMIC_VELOCITY_CRITIC: ("velocity_critic",),
}


def goal_width(goal_encoding: str, goal_anchor: bool) -> int:
    """Width of the policy goal for one encoding."""
    if goal_encoding == "goal730":
        return GOAL730_DIM
    return GOAL_REF_DIM + (ANCHOR_DIM if goal_anchor else 0)


class SuperMimicPolicy(nn.Module, BasePolicy):
    """World model, motion policy, critics and the optional velocity interface.

    Args:
        goal_encoding: ``paper`` (680-d, 689-d with the anchor) or ``goal730``.
        goal_anchor: Append the 9-d torso anchor to the paper goal.
        obs_dim: Width of one observation frame (93, or 99 with the reference
            orientation).
        velocity_tracking: Build the velocity encoder and its critic.
        critic_features: ``root4`` or ``body_error`` motion-critic inputs.
        input_norm: Running normalization of policy, critic and world-model
            inputs.
        wm_state_loss: ``paper``, ``scaled`` or ``running``. ``running``
            also keeps the running mean and std of the world-model residual.
        termination_bias_init: Initial termination-head bias; ``None`` keeps
            the default initialization.
        torch_compile: Compile the training scans and the G1 kinematics on
            large CUDA batches. The kinematics switch is process-wide.
    """

    def __init__(
        self,
        goal_encoding: str = "paper",
        goal_anchor: bool = False,
        obs_dim: int = OBS_DIM,
        velocity_tracking: bool = False,
        critic_features: str = "root4",
        input_norm: bool = False,
        wm_state_loss: str = "paper",
        termination_bias_init: float | None = None,
        torch_compile: bool = True,
    ):
        super().__init__()
        self.goal_encoding = str(goal_encoding)
        self.goal_anchor = bool(goal_anchor)
        self.obs_dim = int(obs_dim)
        self.goal_dim = goal_width(self.goal_encoding, self.goal_anchor)
        self.velocity_tracking = bool(velocity_tracking)
        self.critic_features = str(critic_features)
        self.scale_xi = bool(input_norm)
        self.include_ref_ori = self.obs_dim == 99
        self.wm_state_loss = str(wm_state_loss)

        self.dynamics = DynamicsNet(input_norm, self.wm_state_loss == "running")
        self.encoder = MotionEncoder(self.goal_dim, input_norm)
        self.decoder = MotionDecoder(self.obs_dim, input_norm)
        self.critic = CriticNet(
            critic_input_dim(self.goal_dim, self.critic_features), input_norm
        )
        self.termination = TerminationHead(termination_bias_init)
        shared_obs_norm = self.decoder.obs_norm
        self.velocity_encoder = (
            VelocityEncoder(self.obs_dim, shared_obs_norm)
            if self.velocity_tracking
            else None
        )
        self.velocity_critic = (
            VelocityCritic(input_norm) if self.velocity_tracking else None
        )
        self.autocast_dtype: torch.dtype | None = None
        self._training: DictConfig | None = None
        self._compile_cache = CompileCache(enabled=torch_compile)
        set_kinematics_compile(torch_compile)
        self._wm_window = None
        self._imagination = None

    def set_autocast(self, dtype: torch.dtype | None) -> None:
        """Autocast every linear stack to ``dtype``; ``None`` keeps float32.

        Weights, optimizer state, kinematics and losses stay float32. Float32
        matmuls use TF32 tensor cores either way; that switch is process-wide.
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        self.autocast_dtype = dtype
        for module in self.children():
            module.autocast_dtype = dtype

    def configure_training(self, algorithm_cfg: DictConfig) -> None:
        """Bind the loss settings of ``algorithm_cfg`` and build the scans.

        Only the actor worker calls this; acting needs none of it.
        """
        self._training = algorithm_cfg
        self.k = int(algorithm_cfg.k)
        self.horizon = int(algorithm_cfg.H)
        self.gamma = float(algorithm_cfg.gamma)
        self.residual_frame = str(algorithm_cfg.residual_frame)
        self.learned_termination = str(algorithm_cfg.termination_head) == "learned"
        self.termination_body = str(algorithm_cfg.termination_body)
        self.reward_anchor = str(algorithm_cfg.reward_anchor)
        self.bce_weight = float(algorithm_cfg.bce_weight)
        wm_name = (
            f"wm-{self.k}-{self.wm_state_loss}-{self.residual_frame}-"
            f"{int(self.learned_termination)}-{self.termination_body}"
        )
        self._wm_window = make_dynamics_window(
            self.k,
            self.learned_termination,
            self.termination_body,
            residual_frame=self.residual_frame,
            loss_mode=self.wm_state_loss,
            compile_name=wm_name,
            compile_cache=self._compile_cache,
        )
        imagine_name = (
            f"imagine-{self.horizon}-{int(self.learned_termination)}-"
            f"{self.termination_body}-{self.critic_features}-"
            f"{int(self.include_ref_ori)}-{self.obs_dim}-{self.reward_anchor}-"
            f"{self.residual_frame}-{int(self.scale_xi)}-{self.gamma}"
        )
        self._imagination = make_imagination(
            self.horizon,
            self.learned_termination,
            self.termination_body,
            self.critic_features,
            self.include_ref_ori,
            self.obs_dim,
            self.gamma,
            self.reward_anchor,
            self.residual_frame,
            self.scale_xi,
            compile_name=imagine_name,
            compile_cache=self._compile_cache,
        )

    def phase_parameters(self, forward_type: ForwardType) -> list[nn.Parameter]:
        """Parameters the phase of ``forward_type`` updates."""
        params: list[nn.Parameter] = []
        for name in PHASE_MODULES[forward_type]:
            module = getattr(self, name)
            if module is not None:
                params.extend(module.parameters())
        return params

    def set_trainable(self, forward_type: ForwardType) -> None:
        """Let gradients reach only the parameters of one training phase."""
        self.requires_grad_(False)
        for param in self.phase_parameters(forward_type):
            param.requires_grad_(True)

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def act(
        self,
        history: torch.Tensor,
        goal: torch.Tensor,
        command: torch.Tensor | None = None,
        task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Clean action ``[N, 29]`` of every env.

        ``task`` is 1 on velocity-tracking rows, which read the normalized
        ``command``; the other rows read ``goal``.
        """
        if self.velocity_encoder is None or task is None:
            return self.decoder(self.encoder(goal), history)
        velocity_rows = task.bool()
        motion_rows = ~velocity_rows
        action = history.new_zeros(history.shape[0], self.decoder.net[-1].out_features)
        if bool(motion_rows.any()):
            action[motion_rows] = self.decoder(
                self.encoder(goal[motion_rows]), history[motion_rows]
            )
        if bool(velocity_rows.any()):
            rows_history = history[velocity_rows]
            action[velocity_rows] = self.decoder(
                self.velocity_encoder(command[velocity_rows], rows_history),
                rows_history,
            )
        return action

    @torch.no_grad()
    def critic_value(self, features: torch.Tensor) -> torch.Tensor:
        """Motion-critic value ``[N]`` of raw critic features."""
        return self.critic(features)

    @torch.no_grad()
    def fold_policy_stats(
        self,
        obs_stats: torch.Tensor,
        obs_count: float,
        goal_stats: torch.Tensor | None,
        goal_count: float,
    ) -> None:
        """Fold merged statistics of the frame and goal the policy is about to see."""
        if self.decoder.obs_norm is not None:
            self.decoder.obs_norm.update_from_packed(obs_stats, obs_count)
        if self.encoder.goal_norm is not None and goal_stats is not None:
            self.encoder.goal_norm.update_from_packed(goal_stats, goal_count)

    def predict_action_batch(self, env_obs: dict, **kwargs):
        """Deterministic action chunk ``[N, 1, 29]`` for evaluation.

        ``env_obs["states"]`` packs the history, the goal and, with velocity
        tracking, the normalized command and the velocity-task flag.
        """
        device = next(self.parameters()).device
        states = env_obs["states"].to(device=device, dtype=torch.float32)
        widths = [HISTORY_LEN * self.obs_dim, self.goal_dim]
        if self.velocity_encoder is not None:
            widths += [3, 1]
        if states.shape[-1] != sum(widths):
            raise ValueError(
                f"SuperMimic observations must be {sum(widths)} wide "
                f"(history, goal{', command, task' if len(widths) > 2 else ''}), "
                f"got {states.shape[-1]}"
            )
        parts = torch.split(states, widths, dim=-1)
        command, task = (
            (parts[2], parts[3][:, 0] > 0.5) if len(parts) > 2 else (None, None)
        )
        action = self.act(parts[0], parts[1], command, task)
        return action.unsqueeze(1), {}

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "SuperMimicPolicy trains through its SUPERMIMIC_* forward types"
        )

    # ---------------------------------------------------------------- training

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """Run one training phase.

        ``empty=True`` is the call of a rank without rows. It still enters the
        FSDP root forward, so parameter gathers and gradient reductions match
        on every rank, and its loss is a zero that touches every trainable
        parameter.
        """
        handlers = {
            ForwardType.SUPERMIMIC_WORLD_MODEL: self._world_model_forward,
            ForwardType.SUPERMIMIC_ACTOR: self._actor_forward,
            ForwardType.SUPERMIMIC_CRITIC: self._critic_forward,
            ForwardType.SUPERMIMIC_VELOCITY_ACTOR: self._velocity_actor_forward,
            ForwardType.SUPERMIMIC_VELOCITY_CRITIC: self._velocity_critic_forward,
        }
        handler = handlers.get(forward_type)
        if handler is None:
            return super().forward(forward_type, **kwargs)
        empty = bool(kwargs.pop("empty", False))
        out = {} if empty else handler(**kwargs)
        if torch.is_grad_enabled():
            trainable = [p for p in self.parameters() if p.requires_grad]
            if trainable:
                touch = torch.stack([p.sum() for p in trainable]).sum() * 0.0
                out["loss"] = touch if empty else out["loss"] + touch
        return out

    def _termination_prob(self, height, tilt) -> torch.Tensor:
        if self.learned_termination:
            return torch.sigmoid(self.termination(termination_head_input(height, tilt)))
        return analytic_failure_prob(height, tilt)

    def _state_loss(self, pred, true) -> torch.Tensor:
        if self.wm_state_loss == "running":
            return running_state_loss(pred, true, self.dynamics, self.residual_frame)
        return state_prediction_loss(pred, true, scaled=self.wm_state_loss == "scaled")

    def _world_model_forward(
        self,
        state0: torch.Tensor,
        actions: torch.Tensor,
        xi: torch.Tensor,
        targets: torch.Tensor,
        refs: torch.Tensor,
        end_kind: torch.Tensor,
        task: torch.Tensor | None = None,
        bce_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """k-step world-model loss on replay windows.

        ``refs`` are the reference states after every step; the analytic
        termination reads ``targets`` instead. With velocity rows (``task``
        given) the termination BCE averages over motion rows only, which the
        compiled window cannot express, so that mix stays an open loop.
        """
        if task is None:
            loss, one, last = self._wm_window(
                self.dynamics,
                self.termination,
                state0,
                actions,
                xi,
                targets,
                refs,
                end_kind,
                self.bce_weight,
            )
        else:
            loss, one, last = self._world_model_open_loop(
                state0, actions, xi, targets, refs, end_kind, task, bce_scale
            )
        return {
            "loss": loss,
            "one_step_loss": one.detach(),
            "k_step_loss": last.detach(),
        }

    def _world_model_open_loop(
        self, state0, actions, xi, targets, refs, end_kind, task, bce_scale
    ):
        pred = state0
        loss = pred.new_zeros(())
        one = last = loss
        motion_rows = task == 0
        has_motion = self.learned_termination and bool(motion_rows.any())
        for t in range(self.k):
            pred = integrate_residual(
                pred, self.dynamics(pred, actions[:, t], xi), self.residual_frame
            )
            step = self._state_loss(pred, targets[:, t])
            if t == 0:
                one = step
            last = step
            loss = loss + step
            if has_motion:
                height, tilt, _ = failure_signals(
                    pred[motion_rows],
                    refs[motion_rows, t],
                    termination_body=self.termination_body,
                )
                logit = self.termination(termination_head_input(height, tilt))
                loss = loss + self.bce_weight * bce_scale * termination_bce(
                    logit, end_kind[motion_rows, t]
                )
        return loss, one, last

    def _actor_forward(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        xi: torch.Tensor,
        history: torch.Tensor,
        refs: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_rot: torch.Tensor,
        ref_lin: torch.Tensor,
        ref_ang: torch.Tensor,
        q0: torch.Tensor,
        joint_low: torch.Tensor,
        joint_high: torch.Tensor,
        goals: torch.Tensor | None = None,
        goal_refs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """H-step imagination of the motion policy through the world model.

        ``refs``, ``goals`` and the reference body poses cover times
        ``t .. t+H``. Goals that do not depend on the imagined state come
        precomputed and the rollout runs as compiled scans. The anchor goal
        reads the imagined torso, so it is composed from ``goal_refs`` step by
        step on an open loop.
        """
        ref_kin = (ref_pos, ref_rot, ref_lin, ref_ang)
        if not self.goal_anchor:
            _, _, features, rewards, d_hat, objective = self._imagination(
                self.encoder,
                self.decoder,
                self.dynamics,
                self.termination,
                self.critic,
                state,
                prev_action,
                xi,
                history,
                refs,
                goals,
                *ref_kin,
                q0,
                joint_low,
                joint_high,
            )
        else:
            features, rewards, d_hat, objective = self._anchored_imagination(
                state,
                prev_action,
                xi,
                history,
                refs,
                goal_refs,
                ref_kin,
                q0,
                joint_low,
                joint_high,
            )
        return {
            "loss": -objective,
            "objective": objective.detach(),
            "features": features.detach(),
            "rewards": rewards.detach(),
            "d": d_hat.detach(),
        }

    def _anchored_imagination(
        self, state, prev, xi, history, refs, goal_refs, ref_kin, q0, low, high
    ):
        def ref_fk_at(t):
            return tuple(part[:, t] for part in ref_kin)

        def goal_at(t, current):
            return compose_goal(goal_refs[:, t], current, refs[:, t], True)

        features, rewards, d_hats = [], [], []
        state_fk = differentiable_fk(state)
        for t in range(self.horizon):
            goal = goal_at(t, state)
            features.append(
                rollout_critic_feature(
                    state,
                    goal,
                    xi,
                    refs[:, t],
                    self.critic_features,
                    state_fk,
                    ref_fk_at(t),
                    scale_xi=self.scale_xi,
                )
            )
            action = self.decoder(self.encoder(goal), history)
            state = integrate_residual(
                state, self.dynamics(state, action, xi), self.residual_frame
            )
            ref_next = refs[:, t + 1]
            state_fk = differentiable_fk(state)
            rewards.append(
                analytic_reward(
                    state,
                    ref_next,
                    action,
                    prev,
                    low,
                    high,
                    state_fk,
                    ref_fk_at(t + 1),
                    reward_anchor=self.reward_anchor,
                )["reward"]
            )
            height, tilt, _ = failure_signals(
                state, ref_next, termination_body=self.termination_body
            )
            d_hats.append(self._termination_prob(height, tilt))
            ref_quat = ref_next[..., 3:7] if self.include_ref_ori else None
            obs_next = add_sensor_noise(
                analytic_obs(state, action, xi, q0, ref_quat=ref_quat), True
            )
            frames = history.reshape(-1, HISTORY_LEN, self.obs_dim)
            history = torch.cat((frames[:, 1:], obs_next.unsqueeze(1)), dim=1).reshape(
                history.shape
            )
            prev = action
        terminal = rollout_critic_feature(
            state,
            goal_at(self.horizon, state),
            xi,
            refs[:, self.horizon],
            self.critic_features,
            state_fk,
            ref_fk_at(self.horizon),
            scale_xi=self.scale_xi,
            keep_grad=True,
        )
        features.append(terminal.detach())
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        objective = actor_objective(rewards_t, d_t, self.critic(terminal), self.gamma)
        return torch.stack(features, dim=1), rewards_t, d_t, objective

    def _critic_forward(
        self, features: torch.Tensor, targets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Values of normalized critic features, and the regression loss."""
        values = self.critic.value_mlp(features)
        out = {"values": values}
        if targets is not None:
            out["loss"] = (values - targets).pow(2).mean()
        return out

    def _velocity_actor_forward(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        xi: torch.Tensor,
        history: torch.Tensor,
        command: torch.Tensor,
        from_ref: torch.Tensor,
        ref_commands: torch.Tensor,
        ref_goals: torch.Tensor,
        q0: torch.Tensor,
        joint_low: torch.Tensor,
        joint_high: torch.Tensor,
        rl_share: float,
        align_share: float,
    ) -> dict[str, torch.Tensor]:
        """H-step imagination of the velocity policy.

        Rows whose command follows the reference clip (``from_ref``) read
        ``ref_commands`` ``[B, H+1, 3]`` and pull the velocity latent toward
        the frozen motion latent of ``ref_goals`` ``[R, H, goal_dim]``, one
        goal row per ``from_ref`` row. ``rl_share`` and ``align_share`` are
        this rank's shares of all rows and of all ``from_ref`` rows.
        """
        has_reference_rows = ref_goals.shape[0] > 0
        states, commands, rewards, d_hats, align_terms = [], [], [], [], []
        ref_rows = from_ref.unsqueeze(-1)
        for t in range(self.horizon):
            command = torch.where(ref_rows, ref_commands[:, t], command)
            states.append(state)
            commands.append(command)
            latent = self.velocity_encoder(normalize_command(command), history)
            if has_reference_rows:
                motion_latent = self.encoder(ref_goals[:, t]).detach()
                align_terms.append((latent[from_ref] - motion_latent).pow(2).mean())
            action = self.decoder(latent, history)
            state = integrate_residual(
                state, self.dynamics(state, action, xi), self.residual_frame
            )
            rewards.append(
                velocity_reward(
                    state, command, action, prev_action, joint_low, joint_high
                )
            )
            d_hats.append(velocity_fall_soft(state))
            obs_next = add_sensor_noise(analytic_obs(state, action, xi, q0), True)
            frames = history.reshape(-1, HISTORY_LEN, self.obs_dim)
            history = torch.cat((frames[:, 1:], obs_next.unsqueeze(1)), dim=1).reshape(
                history.shape
            )
            prev_action = action
        command = torch.where(ref_rows, ref_commands[:, self.horizon], command)
        states.append(state)
        commands.append(command)
        states_t = torch.stack(states, dim=1)
        commands_t = torch.stack(commands, dim=1)
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        v_last = self.velocity_critic(states_t[:, -1], commands_t[:, -1], xi)
        objective = actor_objective(rewards_t, d_t, v_last, self.gamma)
        align = (
            torch.stack(align_terms).mean() if align_terms else objective.new_zeros(())
        )
        return {
            "loss": -rl_share * objective + ALIGN_WEIGHT * align_share * align,
            "objective": objective.detach(),
            "align": align.detach(),
            "states": states_t.detach(),
            "commands": commands_t.detach(),
            "rewards": rewards_t.detach(),
            "d": d_t.detach(),
        }

    def _velocity_critic_forward(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        xi: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Velocity-critic values, and the regression loss."""
        values = self.velocity_critic(state, command, xi)
        out = {"values": values}
        if targets is not None:
            out["loss"] = (values - targets).pow(2).mean()
        return out

    @classmethod
    def from_config(cls, cfg: DictConfig) -> SuperMimicPolicy:
        """Build from an ``actor.model`` / ``rollout.model`` node."""
        bias = cfg.get("termination_bias_init", None)
        return cls(
            goal_encoding=str(cfg.goal_encoding),
            goal_anchor=bool(cfg.goal_anchor),
            obs_dim=int(cfg.obs_dim),
            velocity_tracking=bool(cfg.velocity_tracking),
            critic_features=str(cfg.critic_features),
            input_norm=str(cfg.input_norm) == "running",
            wm_state_loss=str(cfg.wm_state_loss),
            termination_bias_init=None if bias is None else float(bias),
            torch_compile=bool(cfg.torch_compile),
        )
