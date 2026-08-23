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

from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute


class _ConstantExpert:
    def predict_action_batch(self, **_):
        return torch.full((3, 2, 2), 2.0), {}


class _TwoItemExpert:
    def predict_action_batch(self, **_):
        return torch.full((2, 2, 2), 2.0), {}


class _FakePolicy:
    def predict_action_batch(self, env_obs, **_):
        return torch.ones((2, 2, 2)), {"forward_inputs": dict(env_obs)}


class _FakeFeatureModel:
    def extract_rlt_obs(self, _):
        return {
            "z_rl": torch.zeros((2, 1)),
            "proprio": torch.zeros((2, 1)),
            "ref_chunk": torch.zeros((2, 4)),
        }


class _HybridGate:
    controls_actor_routing = False

    def step(self, _env_obs, *, external_actor_switch, **_):
        assert torch.equal(
            external_actor_switch,
            torch.tensor([[False], [True]]),
        )
        return type(
            "Decision",
            (),
            {
                "actor_switch": torch.tensor([[True], [False]]),
                "expert_requested": torch.tensor([[False], [True]]),
                "diagnostics": {},
            },
        )()


def test_simulator_route_records_mutually_exclusive_action_sources():
    student_actions = torch.ones((3, 2, 2))
    base_actions = torch.zeros((3, 2, 2))
    result = {
        "forward_inputs": {"ref_chunk": base_actions.reshape(3, -1)},
    }
    route = SimulatorRLTRoute(use_schedule=False, warmup_updates=0)

    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": base_actions.reshape(3, -1)},
            student_actions=student_actions,
            result=result,
            mode="train",
            rlt_switch_flags=torch.tensor([[False], [True], [True]]),
            intervene_requested=torch.tensor([[False], [False], [True]]),
            expert_model=_ConstantExpert(),
        )
    )

    assert torch.equal(output.actions[0], base_actions[0])
    assert torch.equal(output.actions[1], student_actions[1])
    assert torch.equal(output.actions[2], torch.full((2, 2), 2.0))
    route_flags = torch.cat(
        [
            output.result["forward_inputs"][key]
            for key in (
                "actual_base_action",
                "actual_actor_action",
                "actual_expert_action",
            )
        ],
        dim=1,
    )
    assert torch.equal(
        route_flags,
        torch.eye(3, dtype=torch.bool),
    )


def test_simulator_route_records_critical_transition_during_warmup():
    student_actions = torch.ones((2, 2, 2))
    base_actions = torch.zeros((2, 2, 2))
    result = {
        "forward_inputs": {"ref_chunk": base_actions.reshape(2, -1)},
    }
    route = SimulatorRLTRoute(use_schedule=True, warmup_updates=30_000)

    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": base_actions.reshape(2, -1)},
            student_actions=student_actions,
            result=result,
            mode="train",
            rlt_switch_flags=torch.tensor([[False], [True]]),
            version=0,
        )
    )

    assert torch.equal(output.actions, base_actions)
    assert torch.equal(
        output.result["forward_inputs"]["record_transition"],
        torch.tensor([[False], [True]]),
    )
    assert torch.equal(
        output.result["forward_inputs"]["actual_base_action"],
        torch.ones((2, 1), dtype=torch.bool),
    )


def test_rollout_preserves_geometry_actor_switch_with_steam_expert_gate():
    geometry_actor = torch.tensor([[False], [True]])

    actions, result = predict_rlt_actions(
        policy_model=_FakePolicy(),
        feature_model=_FakeFeatureModel(),
        rlt_route=SimulatorRLTRoute(use_schedule=False, warmup_updates=0),
        env_obs={},
        final_obs=None,
        mode="train",
        rlt_switch_flags=geometry_actor,
        expert_model=_TwoItemExpert(),
        critical_phase_gate=_HybridGate(),
    )

    assert torch.equal(actions[0], torch.zeros((2, 2)))
    assert torch.equal(actions[1], torch.full((2, 2), 2.0))
    assert torch.equal(result["forward_inputs"]["record_transition"], geometry_actor)
    assert torch.equal(
        result["forward_inputs"]["intervention_requested"],
        torch.tensor([[False], [True]]),
    )
