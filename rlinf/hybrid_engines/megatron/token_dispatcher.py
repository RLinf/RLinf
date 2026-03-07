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

import os
from typing import NamedTuple, Optional

import torch
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.moe_utils import ModelCommProcessGroups
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig

from rlinf.hybrid_engines.megatron.megatron_model_manager import HAVE_FUSCO, fusco_lib

if HAVE_FUSCO:
    import idxtools
    from fusco import FUSCO
else:
    raise RuntimeError(
        "FUSCO is not available. This module should not be loaded when FUSCO support is disabled."
    )


def gather_along_first_dim(input_, group):
    world_size = torch.distributed.get_world_size(group=group)

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


class FuscoInfo(NamedTuple):
    # cross-node
    global_fusco: FUSCO
    sendindices_s1: torch.Tensor
    recvindices_s1: torch.Tensor
    backindices_s1: torch.Tensor
    send_splits_s1: torch.Tensor
    recv_splits_s1: torch.Tensor
    send_tokens_s1: int
    recv_tokens_s1: int
    seq_len: int
    # intra-node
    intra_fusco: FUSCO = None
    sendindices_s2: torch.Tensor = None
    recvindices_s2: torch.Tensor = None
    backindices_s2: torch.Tensor = None
    send_splits_s2: torch.Tensor = None
    recv_splits_s2: torch.Tensor = None
    send_tokens_s2: int = 0
    recv_tokens_s2: int = 0
    intragroup_expanded_size: torch.Tensor = None
    intergroup_expanded_size: torch.Tensor = None


def dispatch_raw(
    hidden_states: torch.Tensor,
    info: FuscoInfo,
):
    assert hidden_states.dim() == 2, "hidden_states must be a 2D tensor"
    hidden_dim = hidden_states.shape[1]
    curr_device = hidden_states.device
    curr_dtype = hidden_states.dtype
    curr_stream = torch.cuda.current_stream()

    is_2dmode = info.intra_fusco is not None

    s1_out = hidden_states.new_empty(
        (info.recv_tokens_s1, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if not is_2dmode:
        info.global_fusco.all_to_all(
            s1_out,
            hidden_states,
            recvindices=info.recvindices_s1,
            sendindices=info.sendindices_s1,
            recv_splits=info.recv_splits_s1,
            send_splits=info.send_splits_s1,
            stream=curr_stream,
        )
        return s1_out
    else:
        s2_out = hidden_states.new_empty(
            (info.recv_tokens_s2, hidden_dim),
            dtype=curr_dtype,
            device=curr_device,
        )

        info.global_fusco.all_to_all(
            s1_out,
            hidden_states,
            recvindices=info.recvindices_s1,
            sendindices=info.sendindices_s1,
            recv_splits=info.recv_splits_s1,
            send_splits=info.send_splits_s1,
            stream=curr_stream,
        )

        info.intra_fusco.all_to_all(
            s2_out,
            s1_out,
            recvindices=info.recvindices_s2,
            sendindices=info.sendindices_s2,
            recv_splits=info.recv_splits_s2,
            send_splits=info.send_splits_s2,
            stream=curr_stream,
        )

        return s2_out


def combine_raw(hidden_states: torch.Tensor, info: FuscoInfo):
    assert hidden_states.dim() == 2, "hidden_states must be a 2D tensor"
    hidden_dim = hidden_states.shape[1]
    curr_device = hidden_states.device
    curr_dtype = hidden_states.dtype
    curr_stream = torch.cuda.current_stream()

    is_2dmode = info.intra_fusco is not None

    s1_out = hidden_states.new_empty(
        (info.send_tokens_s1, hidden_dim),
        dtype=curr_dtype,
        device=curr_device,
    )

    if not is_2dmode:
        info.global_fusco.all_to_all(
            s1_out,
            hidden_states,
            recvindices=info.backindices_s1,
            sendindices=info.recvindices_s1,
            recv_splits=info.send_splits_s1,
            send_splits=info.recv_splits_s1,
            stream=curr_stream,
        )

        s1_out = s1_out.reshape((info.seq_len, -1, hidden_dim))
        s1_out = s1_out.sum(dim=1)

        return s1_out
    else:
        s2_out = hidden_states.new_empty(
            (info.send_tokens_s2, hidden_dim),
            dtype=curr_dtype,
            device=curr_device,
        )

        info.intra_fusco.all_to_all(
            s2_out,
            hidden_states,
            recvindices=info.backindices_s2,
            sendindices=info.recvindices_s2,
            recv_splits=info.send_splits_s2,
            send_splits=info.recv_splits_s2,
            stream=curr_stream,
        )

        if info.send_tokens_s2 > 0:
            s2_out = torch.segment_reduce(
                s2_out, lengths=info.intragroup_expanded_size, reduce="sum"
            )

        info.global_fusco.all_to_all(
            s1_out,
            s2_out,
            recvindices=info.backindices_s1,
            sendindices=info.recvindices_s1,
            recv_splits=info.send_splits_s1,
            send_splits=info.recv_splits_s1,
            stream=torch.cuda.current_stream(),
        )

        if info.send_tokens_s1 != info.intergroup_expanded_size.numel():
            s1_out = torch.segment_reduce(
                s1_out, lengths=info.intergroup_expanded_size, reduce="sum"
            )

        return s1_out


class FuscoDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        info: FuscoInfo,
    ):
        ctx.info = info
        return dispatch_raw(hidden_states, info)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return combine_raw(grad_output, ctx.info), None


class FuscoCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        info: FuscoInfo,
    ):
        ctx.info = info
        return combine_raw(hidden_states, info)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return dispatch_raw(grad_output, ctx.info), None


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        config: TransformerConfig,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)

        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        self.local_expert_indices = local_expert_indices
        self.topk = config.moe_router_topk
        assert self.ep_size > 1, "Fusco token dispatcher requires EP size > 1"
        assert self.tp_size == 1, "Fusco token dispatcher only supports TP size == 1"
        assert self.tp_ep_group.size() == self.ep_size, (
            "Fusco token dispatcher only supports EP-only"
        )
        assert HAVE_FUSCO is True, (
            "Fusco is not available. Please install Fusco to use MoEFuscoTokenDispatcher."
        )
        assert self.config.moe_pad_expert_input_to_capacity is False, (
            "Fusco token dispatcher does not support --moe-pad-expert-input-to-capacity"
        )
        assert self.config.moe_router_padding_for_fp8 is False, (
            "Fusco token dispatcher does not support --moe-router-padding-for-fp8"
        )

        node_local_world_size_str = os.getenv("NODE_LOCAL_WORLD_SIZE")
        if node_local_world_size_str is not None:
            self.num_local_ranks = int(node_local_world_size_str)
        else:
            if torch.cuda.is_available():
                self.num_local_ranks = torch.cuda.device_count()
            else:
                raise RuntimeError(
                    "NODE_LOCAL_WORLD_SIZE is not set and CUDA is not available to infer the "
                    "local world size. Please set NODE_LOCAL_WORLD_SIZE to the number of "
                    "ranks per node."
                )
        if self.num_local_ranks <= 0:
            raise RuntimeError(
                f"Computed NODE_LOCAL_WORLD_SIZE={self.num_local_ranks}, but a positive "
                "integer is required."
            )
        self.is_2dmode = self.topk > 1 and self.ep_size > self.num_local_ranks

        self.global_fusco = FUSCO(nccl_ep_group=self.ep_group, shared_lib=fusco_lib)

        if self.is_2dmode:
            intra_node_ranks = [
                list(range(start, start + self.num_local_ranks))
                for start in range(
                    0,
                    torch.distributed.get_world_size(self.ep_group),
                    self.num_local_ranks,
                )
            ]
            intra_group = None
            nccl_options = torch.distributed.ProcessGroupNCCL.Options()
            nccl_options.config.cga_cluster_size = 8
            nccl_options.config.max_ctas = 32
            nccl_options.config.min_ctas = 32
            for ranks in intra_node_ranks:
                if torch.distributed.get_rank() in ranks:
                    intra_group = torch.distributed.new_group(
                        ranks, backend="nccl", pg_options=nccl_options
                    )
            self.intra_fusco = FUSCO(nccl_ep_group=intra_group, shared_lib=fusco_lib)

    def preprocess_1d(self, indices: torch.Tensor) -> torch.Tensor:
        seqlen, topk = indices.shape
        num_local_tokens_per_expert = torch.bincount(
            indices.view(-1), minlength=self.num_experts
        )

        num_local_tokens_per_rank = num_local_tokens_per_expert.view(
            self.ep_size, self.num_local_experts
        ).sum(dim=1)

        topk = indices.size(1)
        flatten_indices = indices.view(-1)

        sendindices_unique = torch.argsort(flatten_indices, stable=True).contiguous()
        sendindices_with_duplicates = (sendindices_unique // topk).contiguous()
        send_splits = num_local_tokens_per_rank.to(torch.device("cpu"))

        num_global_tokens_per_expert = gather_along_first_dim(
            num_local_tokens_per_expert, self.ep_group
        ).reshape(self.ep_size, self.num_experts)

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        num_tokens_per_ep = num_global_tokens_per_local_expert.sum(dim=1)

        num_ep_tokens = num_global_tokens_per_local_expert.sum()

        if num_ep_tokens > 0:
            recvindices = idxtools.indices_gen(
                num_global_tokens_per_local_expert,
                num_tokens_per_local_expert,
                num_tokens_per_ep,
                num_ep_tokens.item(),
            )
        else:
            recvindices = torch.empty(
                0, dtype=torch.int64, device=torch.cuda.current_device()
            )
        recv_splits = num_tokens_per_ep.to(torch.device("cpu"))

        self.info = FuscoInfo(
            global_fusco=self.global_fusco,
            sendindices_s1=sendindices_with_duplicates,
            recvindices_s1=recvindices,
            backindices_s1=sendindices_unique,
            send_splits_s1=send_splits,
            recv_splits_s1=recv_splits,
            send_tokens_s1=indices.numel(),
            recv_tokens_s1=num_ep_tokens,
            seq_len=seqlen,
        )

        return num_tokens_per_local_expert

    def preprocess_2d(self, indices: torch.Tensor) -> torch.Tensor:
        seqlen, topk = indices.shape
        assert topk > 1, (
            "2D Dispatcher is efficient only when topk > 1, please use FuscoMoEDispatcher instead"
        )
        ranks_per_node = self.num_local_ranks
        my_rank = torch.distributed.get_rank(group=self.ep_group)
        local_rank = my_rank % ranks_per_node
        my_node = my_rank // ranks_per_node
        self.nnodes = self.ep_size // ranks_per_node
        num_experts_per_node = self.num_experts // self.nnodes
        node_expert_begin, node_expert_end = (
            num_experts_per_node * my_node,
            num_experts_per_node * (my_node + 1),
        )
        device = torch.cuda.current_device()

        # [ep_size * seqlen, topk]
        global_indices = gather_along_first_dim(indices, self.ep_group).reshape(
            self.ep_size, seqlen, topk
        )

        # ========================= stage 1 =========================
        intergroup = torch.arange(
            local_rank, self.ep_size, ranks_per_node, device=device
        )
        # [nNodes, seqlen, topk]
        intragroup_indices = global_indices[intergroup]
        intergroup_indices = (
            intragroup_indices // num_experts_per_node * ranks_per_node
            + (my_rank % ranks_per_node)
        )

        # [nNodes, seqlen, ep_size]
        intergroup_mapping = torch.zeros(
            size=[self.nnodes, seqlen, self.ep_size], dtype=torch.int64, device=device
        )
        intergroup_mapping.scatter_(2, intergroup_indices, 1)
        # [seqlen, ep_size]
        intergroup_send_mapping = intergroup_mapping[my_node]

        # [ep_size]
        send_splits_s1 = intergroup_send_mapping.sum(dim=0).to(torch.device("cpu"))
        # int
        send_tokens_s1 = send_splits_s1.sum()
        # [seqlen]
        intergroup_expanded_size = intergroup_send_mapping.sum(dim=1)
        # [nNodes]
        recv_tokens_per_rank_s1 = intergroup_mapping[:, :, my_rank].sum(dim=1)
        # int
        recv_tokens_s1 = recv_tokens_per_rank_s1.sum()

        recvindices_s1 = torch.arange(recv_tokens_s1, dtype=torch.int64, device=device)
        idx = torch.arange(self.ep_size, dtype=torch.int64, device=device)
        recv_splits_s1 = torch.where(
            idx % ranks_per_node == local_rank,
            recv_tokens_per_rank_s1[idx // ranks_per_node],
            0,
        ).to(torch.device("cpu"))

        ranks = torch.arange(self.ep_size, dtype=torch.int64, device=device)
        row_mask, col_mask = intergroup_send_mapping.nonzero(as_tuple=True)
        intergroup_send_mapping[row_mask, col_mask] = ranks[col_mask]
        intergroup_send_indices = intergroup_send_mapping[row_mask, col_mask]
        backindices_s1 = torch.argsort(intergroup_send_indices, stable=True)
        pos_mapping_s1 = torch.repeat_interleave(
            torch.arange(seqlen, device=device), intergroup_expanded_size
        )
        sendindices_s1 = pos_mapping_s1[backindices_s1]

        # ========================= stage 2 =========================
        # [ep_size, seqlen, num_experts]
        self.global_mapping = torch.zeros(
            size=[self.ep_size, seqlen, self.num_experts],
            dtype=torch.bool,
            device=device,
        )
        self.global_mapping.scatter_(2, global_indices, True)
        # [ep_size, num_experts]
        num_global_tokens_per_expert = self.global_mapping.sum(dim=1)

        # [nNodes * seqlen, topk]
        intragroup_indices = intragroup_indices.reshape(-1, topk)

        intragroup_mask = (intragroup_indices >= node_expert_begin) & (
            intragroup_indices < node_expert_end
        )
        intragroup_expanded_size = intragroup_mask.sum(dim=1)
        intragroup_expanded_size = intragroup_expanded_size[
            intragroup_expanded_size != 0
        ]
        mask_indices = intragroup_indices[intragroup_mask]

        # [8, num_experts]
        group_send_tokens_per_expert_s2 = num_global_tokens_per_expert.reshape(
            self.nnodes, ranks_per_node, -1
        ).sum(dim=0)
        # [num_node_experts]
        send_tokens_per_expert_s2 = group_send_tokens_per_expert_s2[
            local_rank, node_expert_begin:node_expert_end
        ]
        # [8]
        send_splits_s2 = (
            send_tokens_per_expert_s2.view(ranks_per_node, self.num_local_experts)
            .sum(dim=1)
            .to(torch.device("cpu"))
        )
        # int
        send_tokens_s2 = send_splits_s2.sum()

        backindices_s2 = torch.argsort(mask_indices, stable=True)
        pos_mapping_s2 = torch.repeat_interleave(
            torch.arange(intragroup_expanded_size.numel(), device=device),
            intragroup_expanded_size,
        )
        sendindices_s2 = pos_mapping_s2[backindices_s2]

        # [8, num_local_experts]
        group_recv_tokens_per_expert_s2 = group_send_tokens_per_expert_s2[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # [num_local_experts]
        recv_tokens_per_expert_s2 = group_recv_tokens_per_expert_s2.sum(dim=0)
        # [8]
        recv_tokens_per_rank_s2 = group_recv_tokens_per_expert_s2.sum(dim=1)
        recv_splits_s2 = recv_tokens_per_rank_s2.to(torch.device("cpu"))
        # int
        recv_tokens_s2 = recv_tokens_per_rank_s2.sum()
        if recv_tokens_s2 > 0:
            recvindices_s2 = idxtools.indices_gen(
                group_recv_tokens_per_expert_s2,
                recv_tokens_per_expert_s2,
                recv_tokens_per_rank_s2,
                recv_tokens_s2.item(),
            )
        else:
            recvindices_s2 = torch.empty(0, dtype=torch.int64, device=device)

        self.info = FuscoInfo(
            global_fusco=self.global_fusco,
            sendindices_s1=sendindices_s1,
            recvindices_s1=recvindices_s1,
            backindices_s1=backindices_s1,
            send_splits_s1=send_splits_s1,
            recv_splits_s1=recv_splits_s1,
            send_tokens_s1=send_tokens_s1,
            recv_tokens_s1=recv_tokens_s1,
            seq_len=seqlen,
            intra_fusco=self.intra_fusco,
            sendindices_s2=sendindices_s2,
            recvindices_s2=recvindices_s2,
            backindices_s2=backindices_s2,
            send_splits_s2=send_splits_s2,
            recv_splits_s2=recv_splits_s2,
            send_tokens_s2=send_tokens_s2,
            recv_tokens_s2=recv_tokens_s2,
            intragroup_expanded_size=intragroup_expanded_size,
            intergroup_expanded_size=intergroup_expanded_size,
        )

        return recv_tokens_per_expert_s2

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.num_tokens = hidden_states.shape[0]

        _, indices = torch.topk(probs, self.topk, dim=-1)

        if self.is_2dmode:
            self.tokens_per_expert = self.preprocess_2d(indices)
        else:
            self.tokens_per_expert = self.preprocess_1d(indices)

        return hidden_states, probs

    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        tokens_by_expert = FuscoDispatch.apply(hidden_states, self.info)
        if self.is_2dmode is False:
            self.global_mapping = gather_from_sequence_parallel_region(
                self.routing_map, group=self.ep_group
            )
        probs = gather_from_sequence_parallel_region(probs, group=self.ep_group)
        return tokens_by_expert, probs

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        if self.is_2dmode is False:
            local_probs = probs[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()

            local_mapping = self.global_mapping[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()

            permuted_local_probs = local_probs.T.contiguous().masked_select(
                local_mapping.T.contiguous()
            )
        else:
            local_probs = (
                probs.reshape(
                    self.nnodes, self.num_local_ranks, self.num_tokens, self.num_experts
                )
                .transpose(0, 1)
                .reshape(-1, self.num_experts)[
                    :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
                ]
                .contiguous()
            )

            local_mapping = (
                self.global_mapping.reshape(
                    self.nnodes, self.num_local_ranks, self.num_tokens, self.num_experts
                )
                .transpose(0, 1)
                .reshape(-1, self.num_experts)[
                    :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
                ]
                .contiguous()
            )

            permuted_local_probs = local_probs.T.contiguous().masked_select(
                local_mapping.T.contiguous()
            )
        return hidden_states, self.tokens_per_expert, permuted_local_probs

    def combine_preprocess(self, hidden_states):
        return hidden_states

    def token_combine(self, hidden_states):
        return FuscoCombine.apply(hidden_states, self.info)

    def combine_postprocess(self, hidden_states):
        return hidden_states.view(self.hidden_shape)
