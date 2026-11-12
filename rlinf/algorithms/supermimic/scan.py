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

Each scan has two compiled pieces: one recurrent step, called from Python
once per step, and one batched score over every step at once. The step holds
only what feeds the next state; kinematics, reward, termination and losses
run once on all steps after the loop. A step-sized graph compiles in
minutes; an unrolled horizon took over an hour.

On a CUDA batch of at least 256 rows each piece is compiled once. A cache
built with ``enabled=False`` keeps every piece eager.
"""

from __future__ import annotations

import torch

from rlinf.algorithms.supermimic.objectives import imagined_return, world_model_loss
from rlinf.envs.sim.g1_mimic.compile_cache import CompileCache, run_compiled
from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.g1_fk import PELVIS_BODY
from rlinf.envs.sim.g1_mimic.motion import assemble_goals
from rlinf.envs.sim.g1_mimic.reward import termination_probability, tracking_reward


def _caller(compile_name: str | None, cache: CompileCache | None):
    """``run_compiled`` under ``compile_name``, or a plain call without one."""
    if compile_name is None:
        return lambda _piece, fn, *args: fn(*args)
    if cache is None:
        cache = CompileCache()
    return lambda piece, fn, *args: run_compiled(
        f"{compile_name}-{piece}", fn, *args, cache=cache
    )


def root_goals(
    root: torch.Tensor,
    content: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-model and actor goals of the goal window seen from ``root``."""
    return assemble_goals(
        content, root[..., :3], quat_to_rotmat(root[..., 3:7]), target_pos, target_rot
    )


def make_world_model_rollout(
    representation,
    compile_name: str | None = None,
    compile_cache: CompileCache | None = None,
):
    """Open-loop world-model rollout over replay windows and its state loss.

    The rollout starts from the window's first state and replays its actions;
    the prediction after the last action is never scored, so a window of
    ``L`` states yields ``L - 1`` predictions.
    """
    call = _caller(compile_name, compile_cache)

    def step(dynamics, root, enc, action, context):
        return representation.advance(root, enc, dynamics(enc, action, context))

    def score(roots, encs, true_roots, true_encs):
        return world_model_loss(
            representation.loss_blocks(roots, encs),
            representation.loss_blocks(true_roots, true_encs),
        )

    def run(dynamics, states, actions, context):
        encs = representation.encode(states)
        roots = representation.root(states)
        root, enc = roots[:, 0], encs[:, 0]
        pred_roots, pred_encs = [], []
        for t in range(states.shape[1] - 1):
            root, enc = call("step", step, dynamics, root, enc, actions[:, t], context)
            pred_roots.append(root)
            pred_encs.append(enc)
        return call(
            "score",
            score,
            torch.stack(pred_roots, dim=1),
            torch.stack(pred_encs, dim=1),
            roots[:, 1:],
            encs[:, 1:],
        )

    return run


def make_imagination(
    representation,
    horizon: int,
    gamma: float,
    value_max: float,
    compile_name: str | None = None,
    compile_cache: CompileCache | None = None,
):
    """H-step imagination of the motion policy through the world model.

    Step ``k`` builds the goals from the imagined root and the window at
    reference time ``t_k``, acts, and predicts the next state, which is
    scored against the reference bodies at ``t_k``. The critic value of the
    state after ``H`` steps, capped at ``value_max``, closes the return.
    """
    call = _caller(compile_name, compile_cache)

    def step(actor, dynamics, root, enc, context, content, target_pos, target_rot):
        world_goal, actor_goal = root_goals(root, content, target_pos, target_rot)
        action = actor(enc, actor_goal)
        root, enc = representation.advance(root, enc, dynamics(enc, action, context))
        return action, root, enc, world_goal

    def score(
        critic,
        roots,
        encs,
        actions,
        prev_action,
        world_goals,
        context,
        refs,
        soft_limits,
        last_window,
    ):
        after_roots, after_encs = roots[:, 1:], encs[:, 1:]
        root_rot = quat_to_rotmat(after_roots[..., 3:7])
        joint_pos, _ = representation.joints(after_encs)
        previous = torch.cat((prev_action.unsqueeze(1), actions[:, :-1]), dim=1)
        rewards = tracking_reward(
            after_roots[..., :3],
            root_rot,
            representation.bodies(after_roots, after_encs),
            refs,
            joint_pos,
            actions,
            previous,
            soft_limits,
        )
        ref_pos, ref_rot = refs[0], refs[1]
        term_prob = termination_probability(
            after_roots[..., :3],
            root_rot,
            ref_pos[..., PELVIS_BODY, :],
            ref_rot[..., PELVIS_BODY, :, :],
        )
        last_goal, _ = root_goals(roots[:, -1], *last_window)
        terminal = torch.cat((encs[:, -1], last_goal, context), dim=-1)
        value = critic(terminal).clamp(max=value_max)
        ret = imagined_return(rewards, term_prob, value, gamma)
        steps = torch.cat(
            (
                encs[:, :-1],
                world_goals,
                context.unsqueeze(1).expand(-1, horizon, -1),
            ),
            dim=-1,
        )
        return ret, rewards, term_prob, steps, terminal

    def run(
        actor,
        dynamics,
        critic,
        start,
        prev_action,
        context,
        windows,
        refs,
        soft_limits,
    ):
        """``windows`` holds the goal content and root targets at ``t_0 .. t_H``
        and ``refs`` the reference bodies at ``t_0 .. t_{H-1}``."""
        content, target_pos, target_rot = windows
        root, enc = representation.root(start), representation.encode(start)
        roots, encs, actions, world_goals = [root], [enc], [], []
        for k in range(horizon):
            action, root, enc, world_goal = call(
                "step",
                step,
                actor,
                dynamics,
                root,
                enc,
                context,
                content[:, k],
                target_pos[:, k],
                target_rot[:, k],
            )
            roots.append(root)
            encs.append(enc)
            actions.append(action)
            world_goals.append(world_goal)
        return call(
            "score",
            score,
            critic,
            torch.stack(roots, dim=1),
            torch.stack(encs, dim=1),
            torch.stack(actions, dim=1),
            prev_action,
            torch.stack(world_goals, dim=1),
            context,
            refs,
            soft_limits,
            (content[:, horizon], target_pos[:, horizon], target_rot[:, horizon]),
        )

    return run
