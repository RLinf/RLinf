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

"""RolloutResult metadata updates for RLT Stage 2 interventions."""

from __future__ import annotations

import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult

from .rollout_schema import resolve_source_from_source_chunk
from .transition import TransitionSource


def update_last_rlt_action_metadata(
    rollout_result: EmbodiedRolloutResult,
    intervene_flags: torch.Tensor,
) -> None:
    """Update RLT source metadata after an env-side action override."""
    if not rollout_result.forward_inputs:
        return

    last_forward_inputs = rollout_result.forward_inputs[-1]
    if "source_chunk" not in last_forward_inputs:
        return

    if intervene_flags.dim() == 1:
        intervene_flags = intervene_flags[:, None]
    intervene_flags = intervene_flags.to(torch.bool)

    batch_size, num_action_chunks = intervene_flags.shape[:2]
    source_chunk = last_forward_inputs["source_chunk"].clone()
    source_chunk = source_chunk.reshape(batch_size, num_action_chunks)
    source_chunk[intervene_flags.to(source_chunk.device)] = int(TransitionSource.HUMAN)
    last_forward_inputs["source_chunk"] = source_chunk.cpu().contiguous()
    last_forward_inputs["source"] = (
        resolve_source_from_source_chunk(source_chunk).cpu().contiguous()
    )
    last_forward_inputs["intervention_flag"] = (
        intervene_flags.any(dim=1, keepdim=True).cpu().contiguous()
    )
    if "intervention_flags" in last_forward_inputs:
        last_forward_inputs["intervention_flags"] = (
            intervene_flags.cpu().contiguous()
        )
    last_forward_inputs.pop("model_action", None)
