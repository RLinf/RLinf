# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming Flow Policy trajectory sampler.

Ordinary :meth:`~rlinf.models.embodiment.openpi.pi0.Pi0.sample_actions`
denoises a whole action chunk from ``t=1`` to ``t=0``. SFP instead integrates
the learned trajectory velocity from the current action state, matching
StreamingVLA ``SVLAPytorch.sample_actions``: one suffix token per step,
``dt = 1 / action_horizon``, ``a_t = v_t * dt``.
"""

from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi.modules.model import (
    Observation,
    preprocess_observation,
)


@torch.no_grad()
def sample_sfp_actions(pi0_model, observation: Observation) -> torch.Tensor:
    """Integrate the SFP velocity field into one action chunk.

    Args:
        pi0_model: A ``Pi0`` (or eval subclass) with prefix/suffix helpers.
        observation: Already-transformed observation whose ``action_states``
            is the normalized, padded cumulative action state ``(B, action_dim)``.

    Returns:
        Model-space action deltas ``(B, action_horizon, action_dim)`` in float32.
        Callers run the output transform to recover env-space actions.
    """
    observation = preprocess_observation(observation, train=False)
    if observation.action_states is None:
        raise ValueError(
            "SFP sampling requires observation.action_states. Pi0Eval must "
            "inject the env-space accumulator before the input transform."
        )

    batch_size = observation.state.shape[0]
    device = observation.state.device
    horizon = int(pi0_model.action_horizon)
    action_dim = int(pi0_model.action_dim)

    action_states = observation.action_states.to(device=device, dtype=torch.float32)
    if action_states.ndim != 2 or action_states.shape[0] != batch_size:
        raise ValueError(
            "action_states must have shape (batch, action_dim); got "
            f"{tuple(action_states.shape)}."
        )
    if action_states.shape[-1] < action_dim:
        action_states = torch.nn.functional.pad(
            action_states, (0, action_dim - action_states.shape[-1])
        )
    elif action_states.shape[-1] > action_dim:
        action_states = action_states[..., :action_dim]

    _, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)

    x_t = action_states.unsqueeze(1).contiguous()
    dt = 1.0 / horizon
    time = torch.zeros(batch_size, device=device, dtype=torch.float32)
    deltas = []
    for _ in range(horizon):
        suffix_out = pi0_model.run_suffix(observation, x_t, time, kv_cache, prefix_mask)
        v_t = pi0_model.velocity_from_suffix(suffix_out[:, -1:])
        time = time + dt
        x_t = x_t + dt * v_t
        deltas.append(dt * v_t)

    return torch.cat(deltas, dim=1)
