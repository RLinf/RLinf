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

from types import SimpleNamespace

import numpy as np
import torch

from rlinf.algorithms.rlt.critical_phase_gate import SteamCriticalPhaseGate


def _make_hybrid_gate() -> SteamCriticalPhaseGate:
    gate = object.__new__(SteamCriticalPhaseGate)
    gate.model = torch.nn.Linear(1, 1)
    gate.actor_switch_enabled = False
    gate.actor_mode = "active"
    gate.mode = "active"
    gate.chunk_size = 10
    gate.lookback_chunks = 1
    gate.patience_chunks = 1
    gate.enter_threshold = -0.1
    gate.latch_until_done = True
    gate.expert_takeover_enabled = True
    gate.expert_mode = "active"
    gate.expert_enter_threshold = -0.2
    gate.expert_warmup_chunks = 1
    gate.expert_patience_chunks = 1
    gate.expert_latch_until_done = True
    gate._states = {}
    gate._extract_images = lambda _: {"image": np.zeros((2, 2, 2, 3), dtype=np.uint8)}
    gate._extract_prompts = lambda _obs, batch_size: ["task"] * batch_size
    gate._predict_pair = lambda _state, _prompts: SimpleNamespace(
        score_min=torch.full((2,), -0.5),
        score_mean=torch.full((2,), -0.5),
        prediction_variance=torch.zeros(2),
    )
    return gate


def test_geometry_actor_phase_drives_independent_steam_expert_gate():
    gate = _make_hybrid_gate()
    geometry_actor = torch.tensor([[False], [True]])

    first = gate.step(
        {},
        mode="train",
        stage_id=0,
        external_actor_switch=geometry_actor,
    )
    second = gate.step(
        {},
        mode="train",
        stage_id=0,
        external_actor_switch=geometry_actor,
    )

    assert not gate.controls_actor_routing
    assert torch.equal(first.actor_switch, geometry_actor)
    assert torch.equal(second.actor_switch, geometry_actor)
    assert torch.equal(
        second.diagnostics["rlt_gate_steam_critical_active"],
        torch.tensor([[True], [True]]),
    )
    assert torch.equal(
        second.diagnostics["rlt_gate_actor_active"],
        geometry_actor,
    )
    assert torch.equal(
        second.expert_requested,
        torch.tensor([[False], [True]]),
    )


def test_enabled_steam_actor_switch_owns_critical_phase():
    gate = _make_hybrid_gate()
    gate.actor_switch_enabled = True
    geometry_actor = torch.zeros((2, 1), dtype=torch.bool)

    gate.step(
        {},
        mode="train",
        stage_id=0,
        external_actor_switch=geometry_actor,
    )
    decision = gate.step(
        {},
        mode="train",
        stage_id=0,
        external_actor_switch=geometry_actor,
    )

    assert gate.controls_actor_routing
    assert torch.equal(
        decision.actor_switch,
        torch.ones((2, 1), dtype=torch.bool),
    )
