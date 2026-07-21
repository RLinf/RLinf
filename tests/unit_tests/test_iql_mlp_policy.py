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

import torch

from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import (
    IQLMLPPolicy,
    IQLTwinCritic,
)
from rlinf.workers.actor.fsdp_iql_policy_worker import EmbodiedIQLFSDPPolicy


def _build_critic(obs_dim: int, action_dim: int) -> IQLMLPPolicy:
    critic = IQLMLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_action_chunks=1,
        add_value_head=False,
        add_q_head=False,
    )
    critic.configure_iql(
        {
            "type": "critic",
            "hidden_dims": (8, 8),
        }
    )
    return critic


def test_iql_twin_critic_uses_root_forward() -> None:
    """The twin container should jointly forward q1/q2 without child indexing."""
    obs_dim = 3
    action_dim = 2
    critic = IQLTwinCritic(
        q1=_build_critic(obs_dim, action_dim),
        q2=_build_critic(obs_dim, action_dim),
    )
    observations = torch.randn(4, obs_dim)
    actions = torch.randn(4, action_dim)

    q1, q2 = critic(observations=observations, actions=actions)

    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert any(name.startswith("q1.") for name in critic.state_dict())
    assert any(name.startswith("q2.") for name in critic.state_dict())


def test_iql_worker_calls_critic_root_forward() -> None:
    """The worker should call the wrapped root instead of indexing its children."""

    class RootOnlyCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(
            self, observations: torch.Tensor, actions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.called = True
            return observations[:, 0], actions[:, 0]

        def __getitem__(self, name: str) -> None:
            raise AssertionError(f"unexpected child access: {name}")

    critic = RootOnlyCritic()
    observations = torch.randn(4, 3)
    actions = torch.randn(4, 2)

    q1, q2 = EmbodiedIQLFSDPPolicy.forward_critic_module(critic, observations, actions)

    assert critic.called
    assert torch.equal(q1, observations[:, 0])
    assert torch.equal(q2, actions[:, 0])
