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
# See the License for the Specific language governing permissions and
# limitations under the License.

import torch

from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import OpenPiPytorchRLTConfig


class _PrefixPoolStub(OpenPiPytorchActionModel):
    def __init__(self, rlt_cfg: OpenPiPytorchRLTConfig):
        torch.nn.Module.__init__(self)
        self.rlt_cfg = rlt_cfg


def test_encode_vlm_prefix_flat_mean_pool_with_mask():
    model = _PrefixPoolStub(
        OpenPiPytorchRLTConfig(stage2_z_source="vlm_prefix", rlt_use_mask=True)
    )
    prefix = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0], [8.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    pooled = model._encode_vlm_prefix_flat(prefix, mask)
    torch.testing.assert_close(pooled[0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([2.0, 0.0]))


def test_encode_vlm_prefix_flat_mean_pool_without_mask():
    model = _PrefixPoolStub(
        OpenPiPytorchRLTConfig(stage2_z_source="vlm_prefix", rlt_use_mask=False)
    )
    prefix = torch.ones(2, 4, 3)
    pooled = model._encode_vlm_prefix_flat(prefix, torch.ones(2, 4, dtype=torch.bool))
    torch.testing.assert_close(pooled, torch.ones(2, 3))


def test_encode_vlm_prefix_flat_uses_shared_pool_prefix():
    from rlinf.models.embodiment.prefix_ft.pool import pool_prefix

    model = _PrefixPoolStub(
        OpenPiPytorchRLTConfig(prefix_pool="last", stage2_z_source="vlm_prefix")
    )
    prefix = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    mask = torch.tensor([[True, False], [True, True]])
    pooled = model._encode_vlm_prefix_flat(prefix, mask)
    torch.testing.assert_close(pooled, pool_prefix(prefix, mask, mode="last"))
