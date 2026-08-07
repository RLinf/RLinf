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

from examples.offline_rl.analysis.plot_aloha_sandwich_advantage_curves import (
    category_resampled,
    generate_advantage_plots,
    resolve_threshold,
)
from rlinf.data.process.mixture_config import write_mixture_config_tag


def test_resolve_threshold_prefers_explicit_value(tmp_path) -> None:
    write_mixture_config_tag(
        tmp_path,
        "test",
        {"unified_threshold": 0.25},
    )

    assert resolve_threshold(tmp_path, "test", 0.5) == 0.5
    assert resolve_threshold(tmp_path, "test", None) == 0.25


def test_category_resampled_sorts_frames() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [1, 0, 1, 0],
            "category": ["hitl_success"] * 4,
            "advantage_continuous": [1.0, 0.0, 3.0, 2.0],
        }
    )

    curves = category_resampled(df, "hitl_success", num_points=3)

    np.testing.assert_allclose(curves, [[0.0, 0.5, 1.0], [2.0, 2.5, 3.0]])


def test_generate_advantage_plots_writes_expected_outputs(tmp_path) -> None:
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
                        "num_frames": 3,
                        "raw_reward": 1.0,
                        "is_success": True,
                        "teleop_segments": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    advantages_path = meta_dir / "advantages_test.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "value_current": [-1.0, -0.5, 0.0],
            "advantage_continuous": [-0.2, 0.1, 0.3],
            "advantage": [False, True, True],
        }
    ).to_parquet(advantages_path, index=False)
    output_dir = tmp_path / "plots"

    generated = generate_advantage_plots(
        raw_dir=tmp_path / "unused_raw",
        dataset_dir=dataset_dir,
        advantages_path=advantages_path,
        output_dir=output_dir,
        num_points=10,
        threshold=0.0,
    )

    assert len(generated) == 3
    expected = {
        "advantage_curves_by_category.png",
        "advantage_curves_category_means.png",
        "advantage_distribution.png",
        "advantage_curves_episode_summary.csv",
        "advantage_curves_summary.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    summary = json.loads(
        (output_dir / "advantage_curves_summary.json").read_text(encoding="utf-8")
    )
    assert summary["num_frames"] == 3
    assert summary["num_episodes"] == 1
    assert summary["positive_frames"] == 2
