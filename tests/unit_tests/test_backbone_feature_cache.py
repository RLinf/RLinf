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

import pytest
import torch
from torch import nn

from rlinf.utils.backbone_cache import (
    ROLLOUT_BACKBONE_FEATURE_KEY,
    ROLLOUT_BACKBONE_INPUT_KEYS,
    ROLLOUT_BACKBONE_MASK_KEY,
    ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE,
    filter_rollout_backbone_transport,
    rollout_backbone_producer_rank,
    validate_frozen_backbone,
    validate_rollout_backbone_sample_ids,
)
from rlinf.utils.pinned_rollout_cache import PinnedRolloutBackboneCache


def test_sample_ids_encode_one_rollout_producer():
    base = 3 * ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
    sample_ids = torch.tensor([[base + 2, base], [base + 1, base + 3]])

    assert rollout_backbone_producer_rank(sample_ids) == 3


@pytest.mark.parametrize(
    "sample_ids",
    [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([-1]),
        torch.tensor([0, ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE]),
    ],
)
def test_sample_ids_reject_invalid_producer_namespaces(sample_ids):
    with pytest.raises(ValueError):
        rollout_backbone_producer_rank(sample_ids)


def test_sample_ids_form_complete_permuted_cache_namespace():
    base = 2 * ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
    validate_rollout_backbone_sample_ids(
        torch.tensor([base + 2, base, base + 3, base + 1]),
        producer_rank=2,
        total_samples=4,
    )


@pytest.mark.parametrize(
    "sample_ids",
    [
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 1, 3]),
        torch.tensor([0, 1, 2, ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE]),
    ],
)
def test_sample_ids_reject_incomplete_or_mixed_cache(sample_ids):
    with pytest.raises(ValueError, match="do not form"):
        validate_rollout_backbone_sample_ids(
            sample_ids,
            producer_rank=0,
            total_samples=4,
        )


def test_frozen_eval_backbone_is_required():
    backbone = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="trainable"):
        validate_frozen_backbone(backbone)

    backbone.requires_grad_(False)
    with pytest.raises(ValueError, match="eval mode"):
        validate_frozen_backbone(backbone)

    backbone.eval()
    validate_frozen_backbone(backbone)


def _rollout_forward_inputs():
    return {
        **{key: torch.ones(1) for key in ROLLOUT_BACKBONE_INPUT_KEYS},
        ROLLOUT_BACKBONE_FEATURE_KEY: torch.ones(1),
        ROLLOUT_BACKBONE_MASK_KEY: torch.ones(1),
    }


def test_transport_disabled_drops_feature_and_keeps_raw_inputs():
    forward_inputs = _rollout_forward_inputs()
    assert not filter_rollout_backbone_transport(forward_inputs, reuse_enabled=False)
    assert ROLLOUT_BACKBONE_FEATURE_KEY not in forward_inputs
    assert ROLLOUT_BACKBONE_MASK_KEY not in forward_inputs
    assert all(key in forward_inputs for key in ROLLOUT_BACKBONE_INPUT_KEYS)


def test_complete_feature_drops_raw_backbone_inputs():
    forward_inputs = _rollout_forward_inputs()
    assert filter_rollout_backbone_transport(
        forward_inputs,
        reuse_enabled=True,
    )
    assert ROLLOUT_BACKBONE_FEATURE_KEY in forward_inputs
    assert ROLLOUT_BACKBONE_MASK_KEY in forward_inputs
    assert all(key not in forward_inputs for key in ROLLOUT_BACKBONE_INPUT_KEYS)


def test_incomplete_feature_keeps_raw_inputs_for_fallback():
    forward_inputs = _rollout_forward_inputs()
    forward_inputs.pop(ROLLOUT_BACKBONE_MASK_KEY)
    assert not filter_rollout_backbone_transport(
        forward_inputs,
        reuse_enabled=True,
    )
    assert ROLLOUT_BACKBONE_FEATURE_KEY not in forward_inputs
    assert ROLLOUT_BACKBONE_MASK_KEY not in forward_inputs
    assert all(key in forward_inputs for key in ROLLOUT_BACKBONE_INPUT_KEYS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streamed_pinned_cache_round_trip_and_stats():
    device = torch.device("cuda", torch.cuda.current_device())
    features = torch.arange(5 * 3 * 2, dtype=torch.float32, device=device).reshape(
        5, 3, 2
    )
    masks = torch.arange(5 * 3, dtype=torch.int64, device=device).reshape(5, 3)
    cache = PinnedRolloutBackboneCache(
        total_samples=5,
        feature_example=features[:2],
        mask_example=masks[:2],
        device=device,
    )

    cache.store_block(offset=0, feature=features[:2], mask=masks[:2])
    cache.store_blocks(
        offset=2,
        blocks=[
            (features[2:4], masks[2:4]),
            (features[4:], masks[4:]),
        ],
    )
    cache.finalize()

    first_ids = torch.tensor([4, 0, 2], dtype=torch.int64)
    second_ids = torch.tensor([1, 3], dtype=torch.int64)
    first = cache.load(first_ids)
    second = cache.load(second_ids)
    torch.testing.assert_close(
        first["backbone_features"], features.index_select(0, first_ids.to(device))
    )
    torch.testing.assert_close(
        first["backbone_attention_mask"], masks.index_select(0, first_ids.to(device))
    )
    torch.testing.assert_close(
        second["backbone_features"], features.index_select(0, second_ids.to(device))
    )
    torch.testing.assert_close(
        second["backbone_attention_mask"], masks.index_select(0, second_ids.to(device))
    )

    stats = cache.stats()
    assert stats.samples == 5
    assert stats.stored_samples == 5
    assert stats.loads == 2
    assert stats.loaded_samples == 5
    assert stats.staging_bytes > 0

    cache.clear()
    with pytest.raises(RuntimeError, match="cleared"):
        cache.load(torch.tensor([0]))
