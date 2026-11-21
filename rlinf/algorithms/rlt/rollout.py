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

import os
from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import (
    RLT_OBS_KEYS,
    RLT_OPTIONAL_OBS_KEYS,
    RLT_TRANSITION_PREFIX,
)
from rlinf.tianji_shadow import transport as shadow


def _resolve_rlt_num_stages(policy_model: Any = None) -> int:
    """Resolve stage one-hot width without disturbing non-stage RLT policies."""
    env_val = os.environ.get("RLT_NUM_STAGES")
    if env_val is not None:
        return int(env_val)
    if (
        policy_model is not None
        and int(getattr(policy_model, "num_stage_actors", 0) or 0) > 0
    ):
        return int(getattr(policy_model, "stage_onehot_dim", 8) or 8)
    return 0


def _append_tactile(rlt_obs: dict, env_obs: dict) -> dict:
    """Bypass frozen VLA processing; retain independent Rectify replay tensors."""
    from rlinf.utils.xense_tactile import TACTILE_KEYS

    present = [key in env_obs for key in TACTILE_KEYS]
    if any(present) and not all(present):
        raise ValueError("incomplete tactile observation fields")
    if all(present):
        device = rlt_obs["proprio"].device
        rlt_obs = {
            **rlt_obs,
            **{key: torch.as_tensor(env_obs[key], device=device) for key in TACTILE_KEYS},
        }
    return rlt_obs


def _append_qvel(rlt_obs: dict, env_obs: dict) -> dict:
    """Append joint velocities right after proprio and before stage one-hot."""
    if (
        not isinstance(env_obs, dict)
        or "rlt_qvel" not in env_obs
        or "proprio" not in rlt_obs
    ):
        return rlt_obs
    proprio = rlt_obs["proprio"]
    qvel = torch.as_tensor(
        env_obs["rlt_qvel"], device=proprio.device, dtype=proprio.dtype
    )
    rlt_obs = dict(rlt_obs)
    rlt_obs["proprio"] = torch.cat(
        [proprio, qvel.reshape(proprio.shape[0], -1)], dim=-1
    )
    return rlt_obs


def _append_stage_onehot(
    rlt_obs: dict, rlt_switch_flags: Any, num_stages: int = 0
) -> dict:
    """Append stage one-hot tail to ``proprio`` (and ``critic_proprio`` if present)."""
    if num_stages <= 0 or "proprio" not in rlt_obs:
        return rlt_obs
    proprio = rlt_obs["proprio"]
    if rlt_switch_flags is None:
        flags = torch.zeros(proprio.shape[0], device=proprio.device)
    else:
        flags = torch.as_tensor(rlt_switch_flags, device=proprio.device).reshape(-1)
    if flags.shape[0] != proprio.shape[0]:
        flags = (
            flags[: proprio.shape[0]]
            if flags.shape[0] > proprio.shape[0]
            else flags.new_full(
                (proprio.shape[0],), int(flags[0]) if flags.numel() else 0
            )
        )
    idx = flags.long().clamp_(0, num_stages)
    onehot = torch.zeros(
        (proprio.shape[0], num_stages), device=proprio.device, dtype=proprio.dtype
    )
    active = idx > 0
    if active.any():
        onehot[active, idx[active] - 1] = 1.0
    rlt_obs = dict(rlt_obs)
    rlt_obs["proprio"] = torch.cat([proprio, onehot], dim=-1)
    if "critic_proprio" in rlt_obs:
        rlt_obs["critic_proprio"] = torch.cat(
            [rlt_obs["critic_proprio"], onehot], dim=-1
        )
    return rlt_obs


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
    rlt_switch_flags: Any = None,
    num_stages: int = 0,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = _append_stage_onehot(
            _append_tactile(_append_qvel(feature_model.extract_rlt_obs(final_obs), final_obs), final_obs),
            rlt_switch_flags,
            num_stages=num_stages,
        )
    for key in RLT_OBS_KEYS + tuple(
        k for k in RLT_OPTIONAL_OBS_KEYS if k in transition_obs
    ):
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]


def split_actor_critic_actions(forward_inputs: dict[str, Any], chunk_len: int) -> None:
    """rtj joint Stage2: actor plans 16D joints while critic reads 20D TCP via FK."""
    from rlinf.envs.realworld.tianji.tianji_kinematics import joint_chunk_rl_to_tcp_rl

    plan = forward_inputs["action"].reshape(forward_inputs["action"].shape[0], -1)
    plan_tcp = joint_chunk_rl_to_tcp_rl(plan.double(), chunk_len).to(plan.dtype)
    forward_inputs["policy_action"] = plan.clone()
    forward_inputs["model_action"] = plan.clone()
    forward_inputs["policy_action_tcp"] = plan_tcp
    forward_inputs["action"] = plan_tcp.clone()
    if "target_policy_action" in forward_inputs:
        target = forward_inputs["target_policy_action"]
        target_steps = target.shape[-1] // 16
        forward_inputs["target_policy_action_tcp"] = joint_chunk_rl_to_tcp_rl(
            target.double(), target_steps
        ).to(target.dtype)


def predict_rlt_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        num_stages = _resolve_rlt_num_stages(policy_model)
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        shadow.tap(env_obs, rlt_obs)
        rlt_obs = _append_stage_onehot(
            _append_tactile(_append_qvel(rlt_obs, env_obs), env_obs),
            rlt_switch_flags,
            num_stages=num_stages,
        )
        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=expert_model,
                version=version,
            )
        )
        actions = route_output.actions
        result = route_output.result

        num_heads = int(getattr(policy_model, "num_stage_actors", 0) or 0)
        if num_heads > 0 and num_stages > 0 and "proprio" in rlt_obs:
            outs = []
            target_outs = []
            for k in range(1, num_heads + 1):
                pro = rlt_obs["proprio"].clone()
                pro[..., -num_stages:] = 0.0
                pro[..., -num_stages + k - 1] = 1.0
                a_k, result_k = policy_model.predict_action_batch(
                    env_obs={**rlt_obs, "proprio": pro},
                    mode=mode,
                    return_obs=False,
                )
                a_k = torch.as_tensor(a_k)
                outs.append(a_k.reshape(a_k.shape[0], -1))
                if "target_policy_action" in result_k["forward_inputs"]:
                    target_outs.append(
                        result_k["forward_inputs"]["target_policy_action"]
                    )
            result["forward_inputs"]["actions_by_stage"] = torch.stack(outs, dim=1)
            if target_outs:
                result["forward_inputs"]["target_actions_by_stage"] = torch.stack(
                    target_outs, dim=1
                )

        if "critic_proprio" in rlt_obs:
            split_actor_critic_actions(result["forward_inputs"], int(actions.shape[1]))

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
            rlt_switch_flags=rlt_switch_flags,
            num_stages=num_stages,
        )

    return actions, result
