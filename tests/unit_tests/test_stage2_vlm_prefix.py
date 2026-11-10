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

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.prefix import OpenPiPrefixMixin
from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig


class _PrefixPoolStub(OpenPiPrefixMixin, Pi0):
    def __init__(self, rlt_cfg: OpenPiPytorchRLTConfig):
        torch.nn.Module.__init__(self)
        self.rlt_cfg = rlt_cfg


def test_encode_vlm_prefix_flat_mean_pool_with_mask():
    model = _PrefixPoolStub(OpenPiPytorchRLTConfig(prefix_pool="masked_mean"))
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
    model = _PrefixPoolStub(OpenPiPytorchRLTConfig(prefix_pool="mean"))
    prefix = torch.ones(2, 4, 3)
    pooled = model._encode_vlm_prefix_flat(prefix, torch.ones(2, 4, dtype=torch.bool))
    torch.testing.assert_close(pooled, torch.ones(2, 3))


def test_encode_vlm_prefix_flat_uses_shared_pool_prefix():
    from rlinf.models.embodiment.prefix.pool import pool_prefix

    model = _PrefixPoolStub(OpenPiPytorchRLTConfig(prefix_pool="last"))
    prefix = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    mask = torch.tensor([[True, False], [True, True]])
    pooled = model._encode_vlm_prefix_flat(prefix, mask)
    torch.testing.assert_close(pooled, pool_prefix(prefix, mask, mode="last"))


def test_prefix_pooling_requires_rlt_only_for_rlt_tokens():
    for pool, use_rlt, required in (
        ("masked_mean", False, False),
        ("rlt_token", True, True),
    ):
        model = _PrefixPoolStub(
            OpenPiPytorchRLTConfig(use_rlt=use_rlt, prefix_pool=pool)
        )
        assert model._stage2_requires_rlt() is required


@pytest.mark.parametrize("image_only,expected", [(True, 2.0), (False, 51.0)])
def test_stage2_pooling_respects_explicit_image_only_setting(image_only, expected):
    from rlinf.models.embodiment.openpi.rlt_config import build_rlt_config

    cfg = OmegaConf.create(
        {
            "prefix": {"pool": "masked_mean", "image_only": image_only},
            "openpi": {"use_rlt": False},
        }
    )
    model = _PrefixPoolStub(build_rlt_config(cfg.openpi, parent_cfg=cfg))
    hidden = torch.tensor([[[2.0], [4.0], [100.0]]])
    mask = torch.tensor([[True, False, True]])
    language = torch.ones(1, 1, dtype=torch.long)
    pooled = model._encode_stage2_z(hidden, mask, language)
    torch.testing.assert_close(pooled, torch.tensor([[expected]]))
