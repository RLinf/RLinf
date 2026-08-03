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

from __future__ import annotations

import torch
from torch import nn

ROLLOUT_BACKBONE_FEATURE_KEY = "rollout_backbone_features"
ROLLOUT_BACKBONE_MASK_KEY = "rollout_backbone_attention_mask"
ROLLOUT_BACKBONE_SAMPLE_IDS_KEY = "_rlinf_rollout_backbone_sample_ids"
ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE = 1 << 40
ROLLOUT_BACKBONE_TRANSPORT_KEY = "rollout_backbone_feature_transport"
ROLLOUT_BACKBONE_BORROWED_IPC_PINNED = "borrowed_ipc_pinned"


def is_rollout_backbone_ipc_transport(transport: str | None) -> bool:
    return transport == ROLLOUT_BACKBONE_BORROWED_IPC_PINNED


ROLLOUT_BACKBONE_INPUT_KEYS = (
    "eagle_input_ids",
    "eagle_attention_mask",
    "eagle_pixel_values",
    "eagle_image_sizes",
)

BackboneOutput = dict[str, torch.Tensor]


def rollout_backbone_channel_key(actor_rank: int) -> str:
    return f"rollout_backbone_actor_rank_{int(actor_rank)}"


def rollout_backbone_producer_rank(sample_ids: torch.Tensor) -> int:
    """Return the single Rollout rank encoded by a sample-ID tensor."""
    flat_ids = sample_ids.reshape(-1).to(dtype=torch.int64, device="cpu")
    if flat_ids.numel() == 0 or int(flat_ids.min().item()) < 0:
        raise ValueError(
            "rollout backbone sample IDs must be non-empty and non-negative"
        )
    producer_ranks = torch.div(
        flat_ids,
        ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE,
        rounding_mode="floor",
    )
    unique_ranks, counts = producer_ranks.unique(return_counts=True)
    if unique_ranks.numel() != 1:
        rank_counts = {
            int(rank): int(count)
            for rank, count in zip(unique_ranks.tolist(), counts.tolist())
        }
        raise ValueError(
            "one trajectory spans multiple Rollout feature producers: "
            f"counts={rank_counts}"
        )
    return int(unique_ranks.item())


def validate_rollout_backbone_sample_ids(
    sample_ids: torch.Tensor,
    *,
    producer_rank: int,
    total_samples: int,
) -> None:
    """Require exactly one occurrence of every sample in a producer namespace."""
    if producer_rank < 0 or total_samples <= 0:
        raise ValueError("invalid rollout backbone sample-ID contract")
    flat_ids = sample_ids.reshape(-1).to(dtype=torch.int64, device="cpu")
    sample_id_base = producer_rank * ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
    expected_ids = torch.arange(
        sample_id_base,
        sample_id_base + total_samples,
        dtype=torch.int64,
    )
    if flat_ids.numel() != total_samples or not torch.equal(
        flat_ids.sort().values, expected_ids
    ):
        raise ValueError(
            "trajectory sample IDs do not form the pinned backbone cache"
        )


def filter_rollout_backbone_transport(
    forward_inputs: dict[str, torch.Tensor],
    *,
    reuse_enabled: bool,
) -> bool:
    """Keep either raw Eagle inputs or a complete reusable backbone output."""
    if not reuse_enabled:
        forward_inputs.pop(ROLLOUT_BACKBONE_FEATURE_KEY, None)
        forward_inputs.pop(ROLLOUT_BACKBONE_MASK_KEY, None)
        return False

    has_complete_feature = all(
        key in forward_inputs
        for key in (ROLLOUT_BACKBONE_FEATURE_KEY, ROLLOUT_BACKBONE_MASK_KEY)
    )
    if not has_complete_feature:
        forward_inputs.pop(ROLLOUT_BACKBONE_FEATURE_KEY, None)
        forward_inputs.pop(ROLLOUT_BACKBONE_MASK_KEY, None)
        return False

    for key in ROLLOUT_BACKBONE_INPUT_KEYS:
        forward_inputs.pop(key, None)
    return True


def validate_frozen_backbone(backbone: nn.Module) -> None:
    parameters = list(backbone.parameters())
    if not parameters:
        raise ValueError("backbone cache requires a backbone with parameters")
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    if trainable:
        raise ValueError(
            "backbone cache requires every backbone parameter to be frozen; "
            f"found {trainable} trainable parameters"
        )
    if backbone.training:
        raise ValueError("backbone cache requires the backbone to be in eval mode")
