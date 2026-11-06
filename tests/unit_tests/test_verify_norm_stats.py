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

"""Unit tests for toolkits/lerobot/verify_norm_stats.py.

These cover the CPU helpers used to verify that RLinf's OpenPI SFT
``norm_stats.json`` are consistent with the data pipeline that produced them
(in particular the extra ``DeltaActions`` transform enabled by
``extra_delta_transform=True`` in the ``pi0_libero`` config).
"""

import numpy as np
import pytest

from toolkits.lerobot.verify_norm_stats import (
    apply_delta_actions,
    compare_stats,
    compute_episode_stats,
    extract_reference_stats,
)


class TestApplyDeltaActions:
    def test_matches_openpi_delta_actions_semantics(self):
        # openpi DeltaActions: actions[..., :dims] -= state broadcast over
        # the chunk; the gripper dim stays absolute.
        horizon = 4
        actions = np.tile(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7]]), (8, 1))
        state = np.tile(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]]), (8, 1))
        out = apply_delta_actions(actions, state, horizon=horizon)

        # First chunk uses state[0]; second chunk uses state[horizon].
        np.testing.assert_allclose(
            out[:horizon, :6], actions[:horizon, :6] - state[0, :6]
        )
        np.testing.assert_allclose(
            out[horizon:, :6], actions[horizon:, :6] - state[horizon, :6]
        )
        # Gripper dim is untouched (absolute).
        np.testing.assert_allclose(out[:, 6], 0.7)
        # Input is not mutated.
        np.testing.assert_allclose(actions[:, 0], 1.0)

    def test_independent_per_chunk(self):
        horizon = 2
        actions = np.zeros((4, 7))
        state = np.array([[1.0], [2.0], [3.0], [4.0]]).repeat(7, axis=1)
        out = apply_delta_actions(actions, state, horizon=horizon)
        # Chunks start at frames 0 and 2, whose states are 1.0 and 3.0.
        np.testing.assert_allclose(out[:, 0], [-1.0, -1.0, -3.0, -3.0])

    def test_custom_delta_dims(self):
        actions = np.ones((3, 7))
        state = np.full((3, 8), 0.5)
        out = apply_delta_actions(actions, state, num_delta_dims=7)
        np.testing.assert_allclose(out, np.full((3, 7), 0.5))


class TestComputeEpisodeStats:
    def test_known_values(self):
        state = np.array([[0.0], [2.0]])
        actions = np.array([[1.0, -1.0], [3.0, 1.0]])
        stats = compute_episode_stats(state, actions)
        assert stats["state"]["mean"] == pytest.approx([1.0])
        assert stats["state"]["std"] == pytest.approx([1.0])
        assert stats["actions"]["mean"] == pytest.approx([2.0, 0.0])
        assert stats["actions"]["q01"][0] == pytest.approx(
            float(np.quantile([1.0, 3.0], 0.01))
        )
        assert stats["actions"]["q99"][0] == pytest.approx(
            float(np.quantile([1.0, 3.0], 0.99))
        )


class TestExtractReferenceStats:
    def test_lerobot_layout(self):
        ref = {"state": {"mean": [1.0, 2.0]}, "actions": {"std": [0.5]}}
        np.testing.assert_allclose(
            extract_reference_stats(ref, "state")["mean"], [1.0, 2.0]
        )

    def test_openpi_padded_layout(self):
        ref = {"norm_stats": {"state": {"mean": [1.0] + [0.0] * 31}}}
        out = extract_reference_stats(ref, "state")["mean"]
        assert out.shape == (32,)


class TestCompareStats:
    def _computed(self):
        return {
            "state": {"mean": [1.0, 2.0], "std": [0.1, 0.2]},
            "actions": {"mean": [0.0], "std": [1.0]},
        }

    def test_identical_stats_report_zero_diff(self):
        ref = {
            "norm_stats": {
                "state": {"mean": [1.0, 2.0, 0.0], "std": [0.1, 0.2, 0.0]},
                "actions": {"mean": [0.0, 0.0], "std": [1.0, 0.0]},
            }
        }
        report = compare_stats(self._computed(), ref)
        for stat in ("mean", "std"):
            assert report["state"][stat]["max_abs_diff"] == pytest.approx(0.0)
            # Only the overlapping (unpadded) dims are compared.
            assert report["state"][stat]["n_dims"] == 2

    def test_detects_mismatch(self):
        ref = {
            "norm_stats": {
                "state": {"mean": [9.0, 2.0]},
                "actions": {"mean": [0.0]},
            }
        }
        report = compare_stats(self._computed(), ref)
        assert report["state"]["mean"]["max_abs_diff"] == pytest.approx(8.0)

    def test_delta_transform_moves_actions_toward_checkpoint_stats(self):
        # Regression-style check for the issue #1042 scenario: raw dataset
        # action stats do NOT match the released pi0_libero norm_stats, but
        # applying the extra DeltaActions transform does (within the
        # tolerance of a single episode).
        rng = np.random.default_rng(0)
        horizon = 50
        # state[joint dims] has a large offset; delta actions are small.
        state = np.tile(
            np.array([[0.0, 0.0, 0.8, 3.0, -0.2, -0.1, 0.0, -0.03]]), (100, 1)
        )
        raw_actions = rng.normal(0.0, 0.05, size=(100, 7))

        ref_raw = {
            "norm_stats": {
                "state": {"mean": state.mean(0).tolist()},
                "actions": {
                    "mean": raw_actions.mean(0).tolist(),
                },
            }
        }
        computed_raw = compute_episode_stats(state, raw_actions)
        report_raw = compare_stats(computed_raw, ref_raw)
        assert report_raw["actions"]["mean"]["max_abs_diff"] < 0.05

        # A checkpoint norm_stats generated with extra_delta_transform=True
        # describes (actions - state) on the first 6 dims.
        delta_actions = apply_delta_actions(raw_actions, state, horizon)
        ref_delta = {
            "norm_stats": {"actions": {"mean": delta_actions.mean(0).tolist()}}
        }
        computed_delta = compute_episode_stats(state, delta_actions)
        report_delta = compare_stats(computed_delta, ref_delta, keys=("actions",))
        assert report_delta["actions"]["mean"]["max_abs_diff"] < 0.05

        # And the raw comparison against the pipeline stats is far off,
        # which is exactly the discrepancy reported in issue #1042.
        raw_vs_delta = np.abs(
            np.asarray(computed_raw["actions"]["mean"])
            - np.asarray(ref_delta["norm_stats"]["actions"]["mean"])
        ).max()
        assert raw_vs_delta > 0.5
