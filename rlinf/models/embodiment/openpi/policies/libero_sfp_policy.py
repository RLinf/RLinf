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

import dataclasses

from rlinf.models.embodiment.openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class LiberoSfpInputs(libero_policy.LiberoInputs):
    """LIBERO inputs that also carry the cumulative action state.

    Streaming Flow Policy starts each chunk's trajectory at the action state the
    episode has reached so far, so that field has to survive alongside the
    images, state, and action deltas the flow-matching pipeline uses.
    """

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        if "observation/action_states" not in data:
            raise KeyError(
                "SFP training needs 'observation/action_states'. Convert the "
                "dataset with toolkits/lerobot/convert_libero_data_to_lerobot.py, "
                "which records the cumulative action state per frame."
            )
        inputs["action_states"] = data["observation/action_states"]
        return inputs
