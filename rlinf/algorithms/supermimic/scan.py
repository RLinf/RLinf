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

"""Fused world-model and imagination scans.

The arithmetic is the eager loop. On a CUDA batch of at least 256 rows the
scan is compiled once so the 2048-wide layers are not launched from Python
one step at a time. A network's ``bf16_matmul`` flag autocasts that
network's linear stack only; this scan does not cast the state, the
kinematics or the loss. ``SUPERMIMIC_SCAN_EAGER=1`` keeps the Python loop.
"""

from __future__ import annotations

import os
import warnings

import torch

from rlinf.algorithms.supermimic.objectives import (
    actor_objective,
    state_prediction_loss,
    termination_bce,
)
from rlinf.envs.sim.g1_mimic.constants import HISTORY_LEN
from rlinf.envs.sim.g1_mimic.features import integrate_residual
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_failure_prob,
    analytic_reward,
    differentiable_fk,
    failure_signals,
    motion_critic_input,
    termination_head_input,
)
from rlinf.models.embodiment.supermimic import running_state_loss

_COMPILED: dict = {}
_FAILED: set[str] = set()
_ANNOUNCED: set[str] = set()


def _leading_batch(args: tuple) -> torch.Tensor | None:
    for value in args:
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] > 0:
            return value
    return None


def run_compiled(name: str, fn, *args):
    """Run ``fn`` eagerly, or the compiled scan on a large CUDA batch."""
    probe = _leading_batch(args)
    eager = (
        os.environ.get("SUPERMIMIC_SCAN_EAGER") == "1"
        or name in _FAILED
        or probe is None
        or (not probe.is_cuda)
        or int(probe.shape[0]) < 256
    )
    if eager:
        return fn(*args)
    compiled = _COMPILED.get(name)
    if compiled is None:
        if name not in _ANNOUNCED:
            warnings.warn(
                "Compiling a SuperMimic training scan (TF32, same math). "
                "The first large batch waits; later batches reuse it.",
                RuntimeWarning,
                stacklevel=2,
            )
            _ANNOUNCED.add(name)
        compiled = torch.compile(fn, dynamic=True, fullgraph=False)
        _COMPILED[name] = compiled
    try:
        return compiled(*args)
    except Exception as exc:
        module = type(exc).__module__
        if module.startswith("torch._dynamo") or "Compiler" in type(exc).__name__:
            _FAILED.add(name)
            _COMPILED.pop(name, None)
            warnings.warn(
                f"SuperMimic scan {name} stayed eager ({type(exc).__name__}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return fn(*args)
        raise


def make_dynamics_window(
    k: int,
    scaled: bool,
    learned: bool,
    termination_body: str,
    residual_frame: str = "heading",
    loss_mode: str | None = None,
):
    """k-step world-model loss. ``k`` and the flags are fixed for one run.

    ``loss_mode`` is ``paper``, ``scaled`` or ``running``. When it is omitted
    the older ``scaled`` flag chooses ``scaled`` or ``paper``.
    ``residual_frame`` is ``heading`` or ``body``.
    """
    if loss_mode is None:
        mode = "scaled" if scaled else "paper"
    else:
        mode = loss_mode

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
        loss = pred.new_zeros(())
        one = pred.new_zeros(())
        last = pred.new_zeros(())
        for t in range(k):
            pred = integrate_residual(
                pred, dynamics(pred, actions[:, t], xi), residual_frame
            )
            if mode == "running":
                step = running_state_loss(pred, targets[:, t], dynamics, residual_frame)
            else:
                step = state_prediction_loss(
                    pred, targets[:, t], scaled=mode == "scaled"
                )
            if t == 0:
                one = step
            last = step
            loss = loss + step
            if learned:
                height, tilt, _ = failure_signals(
                    pred, refs[:, t], termination_body=termination_body
                )
                logit = termination(termination_head_input(height, tilt))
                loss = loss + float(bce_weight) * termination_bce(logit, end_kind[:, t])
        return loss, one, last

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
):
    """H-step imagination. Goals and references are already gathered."""

    def run(
        encoder,
        decoder,
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
        states = []
        features = []
        rewards = []
        d_hats = []
        # Goals ignore the predicted state, so the encoder runs once for every
        # future frame. Reference body poses are gathered from the clip cache.
        # The predicted state still walks the tree, and that walk keeps grad.
        step_goals = goals[:, :horizon]
        latents = encoder(step_goals.reshape(-1, step_goals.shape[-1]))
        latents = latents.reshape(step_goals.shape[0], horizon, -1)
        state_fk = differentiable_fk(state)
        ref_fk = (ref_pos[:, 0], ref_rot[:, 0], ref_lin[:, 0], ref_ang[:, 0])
        for t in range(horizon):
            ref = refs[:, t]
            goal = goals[:, t]
            states.append(state)
            features.append(
                motion_critic_input(
                    state,
                    goal,
                    xi,
                    ref,
                    critic_features,
                    state_fk,
                    ref_fk,
                    scale_xi=scale_xi,
                )
            )
            action = decoder(latents[:, t], history)
            state = integrate_residual(
                state, dynamics(state, action, xi), residual_frame
            )
            ref_next = refs[:, t + 1]
            state_fk = differentiable_fk(state)
            ref_fk = (
                ref_pos[:, t + 1],
                ref_rot[:, t + 1],
                ref_lin[:, t + 1],
                ref_ang[:, t + 1],
            )
            reward = analytic_reward(
                state,
                ref_next,
                action,
                prev,
                joint_low,
                joint_high,
                state_fk,
                ref_fk,
                reward_anchor=reward_anchor,
            )["reward"]
            height, tilt, _ = failure_signals(
                state, ref_next, termination_body=termination_body
            )
            if learned:
                d_hat = torch.sigmoid(termination(termination_head_input(height, tilt)))
            else:
                d_hat = analytic_failure_prob(height, tilt)
            rewards.append(reward)
            d_hats.append(d_hat)
            ref_quat = ref_next[..., 3:7] if include_ref_ori else None
            obs_next = add_sensor_noise(
                analytic_obs(state, action, xi, q0, ref_quat=ref_quat), True
            )
            hist2d = history.reshape(-1, HISTORY_LEN, obs_dim)
            history = torch.cat((hist2d[:, 1:], obs_next.unsqueeze(1)), dim=1).reshape(
                history.shape
            )
            prev = action
        states.append(state)
        features.append(
            motion_critic_input(
                state,
                goals[:, horizon],
                xi,
                refs[:, horizon],
                critic_features,
                state_fk,
                ref_fk,
                scale_xi=scale_xi,
            )
        )
        states_t = torch.stack(states, dim=1)
        feat_t = torch.stack(features, dim=1)
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        objective = actor_objective(rewards_t, d_t, critic(feat_t[:, -1]), gamma)
        return states_t, goals, feat_t, rewards_t, d_t, objective

    return run
