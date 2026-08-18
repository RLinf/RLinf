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

from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute


class _ConstantExpert:
    def predict_action_batch(self, **_):
        return torch.full((3, 2, 2), 2.0), {}


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
