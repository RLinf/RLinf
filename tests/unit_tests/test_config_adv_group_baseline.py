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

"""Which advantage estimators require ``algorithm.group_size > 1``.

The group-baseline guard in ``validate_cfg`` used to test for a
``reinpp_baseline`` ``adv_type`` that is never produced (ReinForce++ registers as
``reinpp`` and toggles its baseline through the separate ``use_reinpp_baseline``
flag), so the check silently never fired for the ReinForce++ baseline.
"""

import pytest

from rlinf.config import adv_requires_group_baseline


@pytest.mark.parametrize("adv_type", ["grpo", "grpo_dynamic"])
def test_grpo_variants_always_require_a_group(adv_type):
    # These normalize every reward against its group regardless of any flag.
    assert adv_requires_group_baseline(adv_type)
    assert adv_requires_group_baseline(adv_type, use_reinpp_baseline=False)


@pytest.mark.parametrize("adv_type", ["GRPO", "Grpo_Dynamic", "ReinPP"])
def test_guard_matches_case_insensitive_dispatch(adv_type):
    # get_adv_and_returns dispatches on adv_type.lower(); the guard must too, or
    # a differently-cased name skips the check yet still runs the estimator.
    assert adv_requires_group_baseline(adv_type, use_reinpp_baseline=True)


def test_reinpp_requires_a_group_only_in_baseline_mode():
    # The bug: this case was never guarded, so reinpp + baseline + group_size=1
    # silently subtracted each reward from itself and zeroed every advantage.
    assert adv_requires_group_baseline("reinpp", use_reinpp_baseline=True)
    # Plain ReinForce++ normalizes over the whole batch; a group of one is fine.
    assert not adv_requires_group_baseline("reinpp", use_reinpp_baseline=False)
    assert not adv_requires_group_baseline("reinpp")


@pytest.mark.parametrize("adv_type", ["gae", "raw", "opd"])
def test_non_group_adv_types_are_exempt(adv_type):
    # Exempt even if a stray baseline flag is set: the flag only applies to reinpp.
    # (grpo_video is deliberately not asserted here: its group requirement is
    # advantage_mode-dependent and left to a separate change.)
    assert not adv_requires_group_baseline(adv_type)
    assert not adv_requires_group_baseline(adv_type, use_reinpp_baseline=True)
