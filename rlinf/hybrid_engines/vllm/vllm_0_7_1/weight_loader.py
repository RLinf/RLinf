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

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

original_vocab_loader = VocabParallelEmbedding.weight_loader


def vocab_loader(
    self: VocabParallelEmbedding, param: Parameter, loaded_weight: torch.Tensor
):
    output_dim = getattr(param, "output_dim", 0)

    partition_size = self.num_embeddings_per_partition
    loaded_vocab_size = loaded_weight.shape[output_dim]
    if loaded_vocab_size == partition_size:
        assert param.data.shape[output_dim] == partition_size, (
            f"Parameter shard size mismatch. Expected {partition_size}, got {param.data.shape[output_dim]}"
        )
        param.data.copy_(loaded_weight)

    else:
        original_vocab_loader(self, param, loaded_weight)


VocabParallelEmbedding.weight_loader = vocab_loader
ParallelLMHead.weight_loader = vocab_loader
