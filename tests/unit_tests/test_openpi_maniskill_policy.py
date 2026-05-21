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

import numpy as np
import pytest

pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.policies.maniskill_policy import ManiSkillOutputs


def test_maniskill_outputs_respects_configured_action_dim():
    transform = ManiSkillOutputs(output_action_dim=4)

    sample_outputs = transform({"actions": np.zeros((8, 7), dtype=np.float32)})
    batch_outputs = transform({"actions": np.zeros((2, 8, 7), dtype=np.float32)})

    assert sample_outputs["actions"].shape == (8, 4)
    assert batch_outputs["actions"].shape == (2, 8, 4)
