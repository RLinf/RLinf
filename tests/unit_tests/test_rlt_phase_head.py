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

import torch

from rlinf.algorithms.rlt.gate_calibration import RLTGateTraceWriter
from rlinf.algorithms.rlt.phase_head import RLT_PHASE_FEATURE_KEY, SteamPhaseHead
from rlinf.data.schema.embodied_types import Trajectory
from toolkits.rlt.train_steam_phase_head import (
    PhaseEpisode,
    calibrate_probability_gate,
)


def test_phase_head_checkpoint_round_trip(tmp_path):
    model = SteamPhaseHead(
        input_dim=6,
        hidden_dim=4,
        ensemble_size=2,
        dropout=0.0,
    ).eval()
    features = torch.randn(2, 3, 6)
    expected = model.predict(features)
    checkpoint = tmp_path / "phase_head.pt"

    model.save_checkpoint(
        checkpoint,
        metadata={"recommended_enter_threshold": 0.6},
    )
    restored, metadata = SteamPhaseHead.from_checkpoint(checkpoint)
    actual = restored.eval().predict(features)

    assert torch.allclose(actual.probability, expected.probability)
    assert torch.allclose(
        actual.prediction_variance,
        expected.prediction_variance,
    )
    assert metadata["recommended_enter_threshold"] == 0.6


def test_trace_writer_preserves_fp16_phase_features(tmp_path):
    writer = RLTGateTraceWriter(
        {"save_path": str(tmp_path), "chunk_size": 10},
        rank=0,
    )
    features = torch.randn(3, 2, 2, 6)
    trajectory = Trajectory(
        forward_inputs={
            "rlt_gate_score_min": torch.zeros(3, 2, 1),
            RLT_PHASE_FEATURE_KEY: features,
        },
        dones=torch.tensor(
            [
                [False, False],
                [False, False],
                [True, True],
            ]
        ),
    )

    assert writer.write([trajectory]) == 1
    trace_path = next(tmp_path.rglob("trace_*.pt"))
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)

    assert trace[RLT_PHASE_FEATURE_KEY].shape == (3, 2, 2, 6)
    assert trace[RLT_PHASE_FEATURE_KEY].dtype == torch.float16
    assert torch.allclose(
        trace[RLT_PHASE_FEATURE_KEY].float(),
        features,
        atol=1.0e-3,
    )


def test_phase_gate_calibration_prioritizes_active_timing_match():
    episode = PhaseEpisode(
        episode_id="episode",
        chunk_size=10,
        features=torch.zeros(5, 1, 2),
        ready=torch.ones(5, dtype=torch.bool),
        geometry_active=torch.tensor([False, False, True, True, True]),
    )
    probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.9])

    selected, _ = calibrate_probability_gate(
        [(episode, probabilities)],
        thresholds=[0.15, 0.5],
        patience_values=[1],
    )

    assert selected["threshold"] == 0.5
    assert selected["active_disagreement_rate"] == 0.0
    assert selected["within_one_chunk_coverage"] == 1.0
