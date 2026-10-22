# Copyright 2025 The RLinf Authors.
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

"""Unit tests for rlinf.utils.eval_stats (success-rate noise statistics)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Evaluation success-rate statistics (eval_stats)
# ---------------------------------------------------------------------------


class TestEvalStats:
    def test_std_matches_closed_form(self):
        import math

        from rlinf.utils.eval_stats import success_rate_std

        assert success_rate_std(0.55, 16) == pytest.approx(math.sqrt(0.55 * 0.45 / 16))

    def test_std_zero_for_degenerate_rates(self):
        from rlinf.utils.eval_stats import success_rate_std

        assert success_rate_std(0.0, 10) == 0.0
        assert success_rate_std(1.0, 10) == 0.0

    def test_wilson_interval_stays_in_unit_range_for_extremes(self):
        from rlinf.utils.eval_stats import wilson_interval

        for rate, n in [(0.0, 4), (1.0, 4), (1.0, 1), (0.0, 1)]:
            low, high = wilson_interval(rate, n)
            assert 0.0 <= low <= high <= 1.0

    def test_wilson_interval_narrows_with_more_episodes(self):
        from rlinf.utils.eval_stats import wilson_interval

        low_n, _ = wilson_interval(0.55, 16)
        _, high_n = wilson_interval(0.55, 1600)
        assert high_n - 0.55 < 0.55 - low_n

    def test_normal_interval_is_centered(self):
        from rlinf.utils.eval_stats import normal_interval

        rate, n = 0.55, 400
        low, high = normal_interval(rate, n)
        assert low + high == pytest.approx(2 * rate)
        assert low < rate < high

    def test_episoded_needed_for_width_grows_as_width_shrinks(self):
        from rlinf.utils.eval_stats import episodes_needed_for_width

        coarse = episodes_needed_for_width(0.2)
        fine = episodes_needed_for_width(0.02)
        assert coarse == pytest.approx(25, abs=2)
        assert fine > coarse

    def test_two_proportion_z_test_flags_large_gap_with_many_episodes(self):
        from rlinf.utils.eval_stats import is_within_noise

        assert not is_within_noise(0.30, 1000, 0.60, 1000)
        # The exact 0.5 vs 0.6 oscillation from the issue, at a small eval
        # batch, must be indistinguishable from noise.
        assert is_within_noise(0.50, 16, 0.60, 16)

    def test_two_proportion_z_test_values(self):
        from rlinf.utils.eval_stats import two_proportion_z_test

        z, p = two_proportion_z_test(0.5, 100, 0.5, 100)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_input_validation(self):
        from rlinf.utils.eval_stats import success_rate_std

        with pytest.raises(ValueError):
            success_rate_std(1.5, 10)
        with pytest.raises(ValueError):
            success_rate_std(0.5, 0)
        with pytest.raises(ValueError):
            success_rate_std(-0.1, 10)
