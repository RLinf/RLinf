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

import json

import numpy as np
import pandas as pd
from PIL import Image

from examples.offline_rl.analysis.plot_aloha_sandwich_per_episode_curves import (
    generate_per_episode_plots,
    intervention_mask,
)


def test_intervention_mask_uses_half_open_intervals() -> None:
    frames = np.arange(6)

    mask = intervention_mask(frames, [[1, 3], [4, 5]])

    np.testing.assert_array_equal(mask, [False, True, True, False, True, False])


def test_generate_per_episode_plots_marks_hitl_frames(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "hil_segments.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "episode_name": "episode_0.hdf5",
                        "num_frames": 4,
                        "raw_reward": 1.0,
                        "is_success": True,
                        "teleop_segments": [[1, 3]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    advantages_path = meta_dir / "advantages_test.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            "value_current": [-1.0, -0.5, -0.25, 0.0],
            "advantage_continuous": [-0.2, 0.1, 0.3, 0.2],
            "advantage": [False, True, True, True],
        }
    ).to_parquet(advantages_path, index=False)
    output_dir = tmp_path / "plots"

    summary = generate_per_episode_plots(
        raw_dir=tmp_path / "unused_raw",
        dataset_dir=dataset_dir,
        advantages_path=advantages_path,
        output_dir=output_dir,
        threshold=0.0,
        dpi=40,
    )

    assert len(summary) == 1
    assert summary.iloc[0]["teleop_frames"] == 2
    assert summary.iloc[0]["advantage_positive_frames"] == 3
    plot_path = output_dir / "episode_000_hitl_success.png"
    with Image.open(plot_path) as image:
        assert image.size == (480, 200)
        image.verify()
    assert (output_dir / "per_episode_value_advantage_summary.csv").exists()
    assert (output_dir / "per_episode_value_advantage_summary.json").exists()
