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

from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.distributed as dist


class SFTPromptCache:
    """Cache frozen T5 outputs before DreamZero applies its batch-wide mask."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: OrderedDict[tuple[bytes, bytes], torch.Tensor] = OrderedDict()

    def encode(
        self,
        encoder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        keys: list[tuple[bytes, bytes]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse raw embeddings and apply the mask for the current batch."""
        miss = any(key not in self.entries for key in keys)
        if dist.is_initialized() and dist.get_world_size() > 1:
            # T5 is sharded. All ranks must agree whether to call it.
            any_miss = torch.tensor(int(miss), device=input_ids.device)
            dist.all_reduce(any_miss, op=dist.ReduceOp.MAX)
            miss = bool(any_miss.item())

        if miss:
            embeddings = encoder(input_ids, attention_mask).to(torch.bfloat16)
            for key, embedding in zip(keys, embeddings.detach().cpu(), strict=True):
                self.entries[key] = embedding.clone()
                self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        else:
            embeddings = torch.stack([self.entries[key] for key in keys]).to(
                input_ids.device
            )
            for key in keys:
                self.entries.move_to_end(key)

        # Upstream truncates every row at the shortest valid length in the batch.
        length = attention_mask.gt(0).sum(dim=1).min()
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        return embeddings.masked_fill((positions >= length)[None, :, None], 0)
