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

from toolkits.rlt.calibrate_steam_critical_gate import (
    EpisodeTrace,
    evaluate_parameters,
    load_episodes,
    replay_critical_gate,
    select_recommended,
)


def _episode(
    episode_id: str,
    *,
    scores: list[float],
    geometry: list[bool],
) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id=episode_id,
        chunk_size=10,
        score=torch.tensor(scores),
        score_ready=torch.tensor([False] + [True] * (len(scores) - 1)),
        geometry_active=torch.tensor(geometry),
        versions=None,
        complete=True,
    )


def test_replay_critical_gate_requires_consecutive_low_scores():
    episode = _episode(
        "episode",
        scores=[0.0, -0.2, -0.3, 0.1],
        geometry=[False, False, True, True],
    )

    entry, active = replay_critical_gate(
        episode,
        threshold=-0.1,
        patience_chunks=2,
    )

    assert entry == 2
    assert torch.equal(active, torch.tensor([False, False, True, True]))


def test_load_episodes_keeps_geometry_negative_episodes(tmp_path):
    trace = {
        "format_version": 1,
        "chunk_size": 10,
        "rlt_gate_score_min": torch.tensor(
            [
                [0.0, 0.0],
                [-0.2, 0.2],
                [-0.3, 0.2],
            ]
        ),
        "rlt_gate_score_ready": torch.tensor(
            [
                [False, False],
                [True, True],
                [True, True],
            ]
        ),
        "geometry_critical_active": torch.tensor(
            [
                [False, False],
                [False, False],
                [True, False],
            ]
        ),
        "dones": torch.tensor(
            [
                [False, False],
                [False, False],
                [True, True],
            ]
        ),
    }
    torch.save(trace, tmp_path / "trace_test.pt")

    episodes = load_episodes(tmp_path, min_version=None, max_version=None)

    assert len(episodes) == 2
    assert sum(bool(episode.geometry_active.any()) for episode in episodes) == 1


def test_select_recommended_prefers_geometry_aligned_profile():
    episodes = [
        _episode(
            "positive",
            scores=[0.0, -0.2, -0.3, 0.1],
            geometry=[False, False, True, True],
        ),
        _episode(
            "negative",
            scores=[0.0, -0.2, 0.2, 0.2],
            geometry=[False, False, False, False],
        ),
    ]
    rows = [
        evaluate_parameters(
            episodes,
            split="calibration",
            threshold=-0.1,
            patience_chunks=patience,
        )
        for patience in (1, 2)
    ]

    selected = select_recommended(
        rows,
        split="calibration",
        min_entry_recall=0.95,
        max_false_entry_rate=0.05,
    )

    assert selected["patience_chunks"] == 2
    assert selected["median_absolute_entry_delta_steps"] == 0
    assert selected["false_entry_rate"] == 0
    assert selected["feasible"]
