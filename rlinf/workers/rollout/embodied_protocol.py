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

"""Publish the policy and environment halves of embodied rollout requests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from rlinf.data.schema.embodied_types import (
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    split_batch_value,
)

if TYPE_CHECKING:
    from rlinf.scheduler import Channel


def publish_policy_part(
    policy_input: PolicyInput,
    actions: torch.Tensor,
    policy_output: PolicyOutput | None,
    actor_channel: Channel,
    *,
    collect_raw_transition_obs: bool,
) -> None:
    """Publish the policy data for a request that produced actions.

    Args:
        policy_input: Routed request whose sources identify the action rows.
        actions: Actions returned to the environment, including external actions.
        policy_output: Inference statistics; absent for external actions.
        actor_channel: Channel consumed by the training trajectory collector.
        collect_raw_transition_obs: Whether the collector needs raw observations.
    """
    transition_obs = policy_input.obs if collect_raw_transition_obs else {}
    if policy_input.requires_inference:
        policy_part = PolicyPart(
            sources=policy_input.sources,
            obs=transition_obs,
            output=policy_output,
        )
    else:
        policy_part = PolicyPart(
            sources=policy_input.sources,
            obs=transition_obs,
            external_actions=actions,
        )
    actor_channel.put(policy_part, async_op=True)


def publish_env_parts(
    policy_input: PolicyInput,
    actor_channel: Channel,
    *,
    collect_raw_transition_obs: bool,
    infer_terminal: Callable[
        [dict[str, Any], dict[str, Any] | None],
        tuple[torch.Tensor | None, dict[str, Any] | None],
    ],
    get_next_rlt_obs: Callable[[], dict[str, Any] | None] | None = None,
) -> None:
    """Complete and publish environment data carried by a policy request.

    Args:
        policy_input: Request carrying previous chunks and producer batch sizes.
        actor_channel: Channel consumed by the training trajectory collector.
        collect_raw_transition_obs: Whether to retain post-action observations.
        infer_terminal: Backend callback returning terminal values and features
            from the effective observation and the current next-state features.
        get_next_rlt_obs: Backend callback providing batched next-state features
            after the attached-part boundaries have been validated.

    Raises:
        ValueError: If part boundaries disagree or terminal observations are absent.
    """
    if not policy_input.env_parts:
        return
    if len(policy_input.env_parts) != len(policy_input.request_sizes):
        raise ValueError("Environment parts and request sizes must align.")

    part_count = len(policy_input.env_parts)
    next_rlt_obs = get_next_rlt_obs() if get_next_rlt_obs is not None else None
    next_rlt_observations = (
        split_batch_value(next_rlt_obs, policy_input.request_sizes)
        if next_rlt_obs is not None
        else [None] * part_count
    )
    needs_policy_obs = any(
        part is not None
        and (
            collect_raw_transition_obs
            or (part.requires_inference and part.next_obs is None)
        )
        for part in policy_input.env_parts
    )
    policy_observations = (
        split_batch_value(policy_input.obs, policy_input.request_sizes)
        if needs_policy_obs
        else [None] * part_count
    )
    for env_part, policy_obs, next_features in zip(
        policy_input.env_parts,
        policy_observations,
        next_rlt_observations,
    ):
        if env_part is None:
            continue
        next_obs = env_part.next_obs if env_part.next_obs is not None else policy_obs
        final_prev_values = None
        if env_part.requires_inference:
            if next_obs is None:
                raise ValueError(
                    "Terminal inference requires a post-action observation."
                )
            values, next_features = infer_terminal(next_obs, next_features)
            if values is not None:
                final_prev_values = values.cpu().contiguous()
        actor_channel.put(
            env_part.complete(
                next_obs=next_obs if collect_raw_transition_obs else None,
                next_rlt_obs=next_features,
                final_prev_values=final_prev_values,
            ),
            async_op=True,
        )
