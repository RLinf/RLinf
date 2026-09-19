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

"""Transforms shared by every Streaming Flow Policy data config."""

import dataclasses

import numpy as np
from openpi import transforms


class SfpNormalize(transforms.Normalize):
    """Quantile normalization by scale alone, without recentering.

    SFP reconstructs an action trajectory by summing ``action_states`` and the
    action deltas, so the two must stay on one common scale. Dividing by
    ``max(|q01|, |q99|)`` is linear and therefore commutes with that sum:
    accumulating normalized deltas gives the same trajectory as normalizing the
    accumulated one. The affine map :class:`openpi.transforms.Normalize` applies
    by default shifts each value by ``q01`` and breaks that identity, which is
    why SFP replaces it rather than reusing it.
    """

    def _normalize_quantile(
        self, x: np.ndarray, stats: transforms.NormStats
    ) -> np.ndarray:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("SFP normalization needs q01 and q99 statistics.")
        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return x / (np.maximum(np.abs(q01), np.abs(q99)) + 1e-6)


@dataclasses.dataclass(frozen=True)
class PadSfpActionStates(transforms.DataTransformFn):
    """Zero-pad ``action_states`` to the model action dimension.

    Runs after :class:`openpi.transforms.PadStatesAndActions`, which pads
    ``state`` and ``actions`` but not the SFP-only field.
    """

    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        if "action_states" not in data:
            raise KeyError(
                "SFP training needs an 'action_states' field in every sample. "
                "Convert the dataset with "
                "toolkits/lerobot/convert_libero_data_to_lerobot.py, which "
                "records the cumulative action state per frame."
            )
        data["action_states"] = transforms.pad_to_dim(
            data["action_states"], self.model_action_dim, axis=-1
        )
        return data
