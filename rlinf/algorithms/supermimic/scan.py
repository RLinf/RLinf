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

"""World-model and imagination scans.

The arithmetic is the eager loop. Each scan has two compiled pieces: one
recurrent step, called from Python once per step, and one batched score
over every step at once. The step holds only what feeds the next state:
policy, dynamics and the new observation. Kinematics, reward, termination,
losses and critic features run once on ``[B * steps]`` rows after the loop.
A step-sized graph compiles in minutes; an unrolled horizon took over an
hour.

On a CUDA batch of at least 256 rows each piece is compiled once. A
network's ``autocast_dtype`` autocasts that network's linear stack only;
the scans do not cast the state, the kinematics or the loss. A cache built
with ``enabled=False`` keeps every piece eager.
"""

from __future__ import annotations

import torch

from rlinf.algorithms.supermimic.objectives import (
    actor_objective,
    running_state_loss,
    state_prediction_loss,
    termination_bce,
)
from rlinf.envs.sim.g1_mimic.compile_cache import CompileCache, run_compiled
from rlinf.envs.sim.g1_mimic.constants import HISTORY_LEN
from rlinf.envs.sim.g1_mimic.features import integrate_residual
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_failure_prob,
    analytic_reward,
    differentiable_fk,
    failure_signals,
    rollout_critic_feature,
    termination_head_input,
)
from rlinf.models.embodiment.supermimic.networks import normalize_history


def _caller(compile_name: str | None, cache: CompileCache | None):
    """``run_compiled`` under ``compile_name``, or a plain call without one."""
    if compile_name is None:
        return lambda _piece, fn, *args: fn(*args)
    if cache is None:
        cache = CompileCache()
    return lambda piece, fn, *args: run_compiled(
        f"{compile_name}-{piece}", fn, *args, cache=cache
    )


def _rows(x: torch.Tensor) -> torch.Tensor:
    """Merge the leading batch and step axes."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def make_dynamics_window(
    k: int,
    learned: bool,
    termination_body: str,
    residual_frame: str = "heading",
    loss_mode: str = "paper",
    compile_name: str | None = None,
    compile_cache: CompileCache | None = None,
):
    """k-step world-model loss. ``k`` and the flags are fixed for one run.

    ``loss_mode`` is ``paper``, ``scaled`` or ``running``.
    ``residual_frame`` is ``heading`` or ``body``. With ``compile_name`` the
    step and the loss are compiled under that name.
    ``compile_cache`` belongs to the worker; if omitted, this scan owns its cache.

    Every per-step loss is a mean over the same batch, so ``k`` times the
    mean over all ``B * k`` rows is the sum of the per-step losses.
    """
    call = _caller(compile_name, compile_cache)

    def state_loss(pred, target, dynamics):
        if loss_mode == "running":
            return running_state_loss(
                pred, target, dynamics.out_norm.std(), residual_frame
            )
        return state_prediction_loss(pred, target, scaled=loss_mode == "scaled")

    def step(dynamics, pred, action, xi):
        return integrate_residual(pred, dynamics(pred, action, xi), residual_frame)

    def score(dynamics, termination, preds, targets, refs, end_kind, bce_weight):
        loss = k * state_loss(_rows(preds), _rows(targets), dynamics)
        one = state_loss(preds[:, 0].detach(), targets[:, 0], dynamics)
        last = state_loss(preds[:, -1].detach(), targets[:, -1], dynamics)
        if learned:
            height, tilt, _ = failure_signals(
                _rows(preds), _rows(refs), termination_body=termination_body
            )
            logit = termination(termination_head_input(height, tilt))
            bce = termination_bce(logit, _rows(end_kind))
            loss = loss + float(bce_weight) * k * bce
        return loss, one, last

    def run(
        dynamics,
        termination,
        state0,
        actions,
        xi,
        targets,
        refs,
        end_kind,
        bce_weight,
    ):
        pred = state0
        preds = []
        for t in range(k):
            pred = call("step", step, dynamics, pred, actions[:, t], xi)
            preds.append(pred)
        return call(
            "score",
            score,
            dynamics,
            termination,
            torch.stack(preds, dim=1),
            targets,
            refs,
            end_kind,
            bce_weight,
        )

    return run


def make_imagination(
    horizon: int,
    learned: bool,
    termination_body: str,
    critic_features: str,
    include_ref_ori: bool,
    obs_dim: int,
    gamma: float,
    reward_anchor: str = "torso",
    residual_frame: str = "heading",
    scale_xi: bool = False,
    compile_name: str | None = None,
    compile_cache: CompileCache | None = None,
):
    """H-step imagination. Goals and references are already gathered.

    The recurrent step carries policy, dynamics and the next observation.
    Reward, termination and critic features never feed the next state, so
    they run once on all ``B * H`` imagined rows after the loop. With
    ``compile_name`` the step and the score are compiled under that name.
    ``compile_cache`` belongs to the worker; if omitted, this scan owns its cache.
    """
    call = _caller(compile_name, compile_cache)

    def step(decoder, obs_norm, dynamics, state, latent, history, xi, q0, ref_quat):
        action = decoder(latent, normalize_history(obs_norm, history, obs_dim))
        state = integrate_residual(state, dynamics(state, action, xi), residual_frame)
        obs_next = add_sensor_noise(
            analytic_obs(state, action, xi, q0, ref_quat=ref_quat), True
        )
        hist2d = history.reshape(-1, HISTORY_LEN, obs_dim)
        history = torch.cat((hist2d[:, 1:], obs_next.unsqueeze(1)), dim=1).reshape(
            history.shape
        )
        return action, state, history

    def score(
        termination,
        critic,
        states,
        actions,
        prev,
        xi,
        refs,
        goals,
        ref_pos,
        ref_rot,
        ref_lin,
        ref_ang,
        joint_low,
        joint_high,
    ):
        batch = states.shape[0]
        # The predicted states walk the tree with grad. Reference body poses
        # come from the clip cache.
        state_fk = tuple(
            x.reshape(batch, horizon + 1, *x.shape[1:])
            for x in differentiable_fk(_rows(states))
        )
        ref_fk = (ref_pos, ref_rot, ref_lin, ref_ang)

        def after(x):
            return _rows(x[:, 1:])

        prevs = torch.cat((prev.unsqueeze(1), actions[:, :-1]), dim=1)
        rewards = analytic_reward(
            after(states),
            after(refs),
            _rows(actions),
            _rows(prevs),
            joint_low,
            joint_high,
            tuple(after(x) for x in state_fk),
            tuple(after(x) for x in ref_fk),
            reward_anchor=reward_anchor,
        )["reward"].reshape(batch, horizon)
        height, tilt, _ = failure_signals(
            after(states), after(refs), termination_body=termination_body
        )
        if learned:
            d_hat = torch.sigmoid(termination(termination_head_input(height, tilt)))
        else:
            d_hat = analytic_failure_prob(height, tilt)
        d_hat = d_hat.reshape(batch, horizon)

        # The actor reads only the terminal value. Every other feature goes
        # back detached for the critic fit.
        xi_rows = xi.unsqueeze(1).expand(-1, horizon + 1, -1)
        features = rollout_critic_feature(
            _rows(states),
            _rows(goals[:, : horizon + 1]),
            _rows(xi_rows),
            _rows(refs),
            critic_features,
            tuple(_rows(x) for x in state_fk),
            tuple(_rows(x) for x in ref_fk),
            scale_xi=scale_xi,
        ).reshape(batch, horizon + 1, -1)
        terminal = rollout_critic_feature(
            states[:, -1],
            goals[:, horizon],
            xi,
            refs[:, horizon],
            critic_features,
            tuple(x[:, -1] for x in state_fk),
            tuple(x[:, horizon] for x in ref_fk),
            scale_xi=scale_xi,
            keep_grad=True,
        )
        objective = actor_objective(rewards, d_hat, critic(terminal), gamma)
        return features, rewards, d_hat, objective

    def run(
        encoder,
        decoder,
        obs_norm,
        dynamics,
        termination,
        critic,
        state,
        prev,
        xi,
        history,
        refs,
        goals,
        ref_pos,
        ref_rot,
        ref_lin,
        ref_ang,
        q0,
        joint_low,
        joint_high,
    ):
        # Goals ignore the predicted state, so the encoder runs once for every
        # future frame.
        step_goals = goals[:, :horizon]
        latents = encoder(step_goals.reshape(-1, step_goals.shape[-1]))
        latents = latents.reshape(step_goals.shape[0], horizon, -1)
        states = [state]
        actions = []
        for t in range(horizon):
            ref_quat = refs[:, t + 1, 3:7] if include_ref_ori else None
            action, state, history = call(
                "step",
                step,
                decoder,
                obs_norm,
                dynamics,
                state,
                latents[:, t],
                history,
                xi,
                q0,
                ref_quat,
            )
            states.append(state)
            actions.append(action)
        states_t = torch.stack(states, dim=1)
        feat_t, rewards_t, d_t, objective = call(
            "score",
            score,
            termination,
            critic,
            states_t,
            torch.stack(actions, dim=1),
            prev,
            xi,
            refs,
            goals,
            ref_pos,
            ref_rot,
            ref_lin,
            ref_ang,
            joint_low,
            joint_high,
        )
        return states_t, goals, feat_t, rewards_t, d_t, objective

    return run
