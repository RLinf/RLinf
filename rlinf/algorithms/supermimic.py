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

"""Re-export. Prefer ``rlinf.envs.sim.g1_mimic.objectives`` in tests."""

from rlinf.envs.sim.g1_mimic.objectives import (  # noqa: F401
    actor_objective,
    collect_bootstrap_mask,
    gae_targets,
    is_episode_boundary,
    state_prediction_loss,
    survival_weights,
    termination_bce,
    wm_termination_target,
)
