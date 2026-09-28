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

from dataclasses import dataclass
from typing import Callable, Optional

from megatron.core.transformer import TransformerConfig

from rlinf.utils.convertor.utils import get_mg2hf_convertor

from .utils import (
    get_tp_gather_fn,
    get_tpe_gather_fn,
)


@dataclass
class ReshardConfig:
    model_type: str
    """Supported model type, valid options are `qwen2.5` and `llama2`."""

    model_config: TransformerConfig

    reshard_weights_format: str = "sglang"
    """Resharding weights format, support sglang, mcore (megatron core)."""

    reshard_tp_size: int = 1
    """Resharding tp size."""

    reshard_pp_size: int = 1
    """Resharding pp size."""

    mg_ep_size: int = 1
    """Megatron expert model parallel size."""

    mg_tpe_size: int = 1
    """Megatron expert tensor parallel size."""

    moe_grouped_gemm: Optional[str] = None
    """Resharding moe_grouped_gemm. avail in [None, 'te']"""

    bucket_capacity: int = 128 * 1024 * 1024
    """sync weight the Bucket capacity size. Now set the bucket capacity to 128MB."""

    convert_fn: Callable = None
    """Function to convert the model weights from megatron format to HuggingFace format."""

    tp_gather_fn: Callable = None
    """Gather-half function for the TP reshard: all_gathers each parameter to
    the full tensor and tags it with a narrow spec (see MegatronCoreWeightReshard
    .gather_full_model)."""

    tpe_gather_fn: Callable = None
    """Gather-half function for the expert-tensor-parallel reshard, from
    expert_tensor_parallel_size to the full expert weight."""

    rollout_ep_size: int = 1
    """Rollout expert-model-parallel size >1 means the rollout engine shards MoE
    experts across its TP ranks via EP. Experts are then tagged with the
    expert_filter spec and narrowed to each target's subset."""

    rollout_moe_dense_tp_size: Optional[int] = None
    """Rollout dense-MLP TP size (sglang moe_dense_tp_size). Controls how the
    first N non-MoE FFN layers are parallelized on the rollout side. None =
    inherit reshard_tp_size (attn_tp, same as attention). 1 = replicated
    (dense MLP not sharded), requiring a full TP all_gather during reshard."""

    rollout_full_tp_size: int = 1
    """Rollout engine full TP size (= sglang tensor_parallel_size). Used for
    shared_experts which are sharded by the full engine TP, not attn_tp. When
    > moe_dense_merge_factor, shared_experts need gather-to-full + slice."""

    rollout_lm_head_tp_size: int = 1
    """Rollout lm_head TP size. sglang lm_head uses the full engine TP group when
    enable_dp_lm_head=False, else the attn_tp group. lm_head is ColumnParallel on
    the actor (sharded by actor TP) but may be sharded by a different TP on the
    rollout, so it needs gather-to-full + slice to (dst_lm_head_rank, this)."""

    enable_dp_attention: bool = False
    """Whether sglang rollout uses DP-attention (enable_dp_attention). When True,
    attn TP (reshard_tp_size) != full engine TP (rollout_full_tp_size), requiring
    different gather+slice strategies per weight type (DPA three/four-way split)."""

    @property
    def split_fc1(self) -> bool:
        """Whether tp_gather_fn should split fused fc1 into gate_proj/up_proj.

        True for sglang/vllm rollout (HF-format receiver expects gate/up);
        False for mcore-format inference reshard (mcore receiver expects the
        fused linear_fc1 key).
        """
        return self.reshard_weights_format != "mcore"

    def __post_init__(self):
        # No TP-ratio assertion: the gather/narrow pair handles any ratio
        # (actor < rollout via all_gather full + slice; actor > rollout via
        # subgroup all_gather or slice; actor == rollout via no-op slice).
        if self.model_type is None:
            raise ValueError(
                "Please specify the model_type, valid options are `qwen2.5` and `llama2`."
            )

        if self.convert_fn is None and self.reshard_weights_format != "mcore":
            self._convertor = get_mg2hf_convertor(self.model_type, self, strict=True)
            self.convert_fn = self._convertor.convert

        if self.tp_gather_fn is None:
            self.tp_gather_fn = get_tp_gather_fn(self.model_type)

        # tpe_gather_fn only use in moe model parallel
        if self.model_config.num_moe_experts is not None and self.tpe_gather_fn is None:
            self.tpe_gather_fn = get_tpe_gather_fn(self.model_type)
