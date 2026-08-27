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


import math

import pytest
import torch

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.utils.metric_utils import (
    collect_trajectory_replay_metrics,
    compute_evaluate_metrics,
)


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


def test_collect_trajectory_replay_metrics_reports_explicit_route_sources():
    trajectory = Trajectory(
        intervene_flags=torch.tensor([[[False]], [[False]], [[True]]]),
        forward_inputs={
            "record_transition": torch.tensor([[[False]], [[True]], [[True]]]),
            "actor_switch": torch.tensor([[[False]], [[True]], [[False]]]),
            "rlt_gate_actor_active": torch.tensor([[[False]], [[True]], [[True]]]),
            "rlt_gate_steam_critical_active": torch.tensor(
                [[[True]], [[False]], [[False]]]
            ),
            "geometry_critical_active": torch.tensor([[[False]], [[False]], [[True]]]),
            "rlt_gate_phase_probability": torch.tensor([[[0.1]], [[0.5]], [[0.9]]]),
            "rlt_gate_phase_prediction_variance": torch.tensor(
                [[[0.01]], [[0.02]], [[0.03]]]
            ),
            "actual_base_action": torch.tensor([[[True]], [[False]], [[False]]]),
            "actual_actor_action": torch.tensor([[[False]], [[True]], [[False]]]),
            "actual_expert_action": torch.tensor([[[False]], [[False]], [[True]]]),
            "intervention_requested": torch.tensor([[[False]], [[False]], [[True]]]),
        }
    )

    metrics = collect_trajectory_replay_metrics([trajectory])

    assert metrics["replay/actual_base_action_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/actual_actor_action_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/actual_expert_action_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/steam_critical_active_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/geometry_critical_active_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/steam_phase_probability_mean"] == pytest.approx(0.5)
    assert metrics["replay/steam_phase_variance_mean"] == pytest.approx(0.02)
    assert metrics["replay/actor_switch_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/intervention_requested_rate"] == pytest.approx(1 / 3)
    assert metrics["replay/intervention_rate"] == pytest.approx(1 / 3)
