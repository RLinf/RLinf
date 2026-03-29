# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Simplified wrapper around TransformerEngine's C++ pytorch backend.
This supports torch.compile(fullgraph=True).
Lowers to cudnn ultimately.
Only bf16 / fp16 is supported.
Only BSHD layout is supported.
Currently, tensors are made contiguous -- packed th2d, th3d not supported yet.
"""

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import transformer_engine

_TE_VER = tuple(int(x) for x in transformer_engine.__version__.split(".")[:2])


try:
    # transformer_engine >= 2.8.0
    import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
except ImportError:
    # transformer_engine < 2.8.0
    import transformer_engine.pytorch.dot_product_attention.utils as dpa_utils

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import (
    TE_DType,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    QKVLayout,
)

if _TE_VER >= (2, 8):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import SoftmaxType

from transformer_engine.pytorch.utils import get_cudnn_version

__all__ = ["DotProductAttention"]


class DotProductAttention(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "bshd",
        attn_mask_type: str = "no_mask",
        window_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        assert qkv_format == "bshd", "Only bshd layout is supported."

        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type

        self.softmax_scale = 1.0 / math.sqrt(kv_channels if isinstance(kv_channels, int) else kv_channels[0])
        self.attention_dropout = attention_dropout
        self.softmax_type = "vanilla"
        self.window_size = dpa_utils.check_set_window_size(attn_mask_type)

        self.fused_attention = FusedAttention(
            self.softmax_scale,
            deterministic=False,
            attention_dropout=self.attention_dropout,
            softmax_type=self.softmax_type,
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.
        """

        qkv_layout = "bshd_bshd_bshd"
        batch_size = query_layer.shape[0]
        device = query_layer.device

        def _get_cu_seqlens(max_seqlen: int) -> torch.Tensor:
            return torch.arange(
                0,
                (batch_size + 1) * max_seqlen,
                step=max_seqlen,
                dtype=torch.int32,
                device=device,
            )

        max_seqlen_q = query_layer.shape[1]
        max_seqlen_kv = key_layer.shape[1]
        cu_seqlens_q = _get_cu_seqlens(max_seqlen_q)
        cu_seqlens_kv = _get_cu_seqlens(max_seqlen_kv)

        return self.fused_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_layout=qkv_layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q,
            cu_seqlens_kv_padded=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask_type=self.attn_mask_type,
            window_size=self.window_size,
            fused_attention_backend=FusedAttnBackend["F16_arbitrary_seqlen"],
        )


def prepare_for_saving(*tensors) -> Tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Prepare tensors for saving. Needed because save_for_backward accepts only
    torch.Tensor/torch.nn.Parameter types, while we want to be able to save
    the internal TensorBase types too."""

    tensor_list, tensor_objects_list = [], []
    for tensor in tensors:
        if tensor is None or isinstance(tensor, torch.Tensor):
            tensor_list.append(tensor)
            tensor_objects_list.append(None)
        else:
            t, t_obj = tensor.prepare_for_saving()
            tensor_list.extend(t)
            tensor_objects_list.append(t_obj)
    return tensor_list, tensor_objects_list


def restore_from_saved(tensors, saved_tensors) -> Tuple[Any, ...]:
    """Recombine the tensor data and metadata during backward pass."""
    tensor_objects = []
    for tensor in tensors:
        if tensor is None or isinstance(tensor, torch.Tensor):
            tensor_objects.append(saved_tensors[0])
            saved_tensors = saved_tensors[1:]
        else:
            saved_tensors = tensor.restore_from_saved(saved_tensors)
            tensor_objects.append(tensor)

    return tuple(tensor_objects)


class FusedAttention(torch.nn.Module):
    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        deterministic: bool = False,
        softmax_type: str = "vanilla",
    ) -> None:
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.deterministic = deterministic
        self.softmax_type = softmax_type

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_q_padded: torch.Tensor,
        cu_seqlens_kv_padded: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        fused_attention_backend: tex.NVTE_Fused_Attn_Backend = tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        quantizers=None,
        pad_between_seqs: bool = False,
        softmax_offset: torch.Tensor | None = None,
        softmax_scale: float = None,
        attention_dropout: float = 0.0,
        deterministic: bool = False,
        softmax_type: str = "vanilla",
    ) -> torch.Tensor:
        """fused attention fprop"""

        cu_seqlens_q_padded = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_kv

        out_nominal_dtype = query_layer.dtype
        output_tensors = fused_attn(
            self.training,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            query_layer,
            key_layer,
            value_layer,
            out_nominal_dtype,
            window_size,
            core_attention_bias,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            None,  # page_table_k,
            None,  # page_table_v,
            self.softmax_scale,
            self.attention_dropout if self.training else 0.0,
            fast_zero_fill,
            qkv_layout,
            core_attention_bias_type,
            attn_mask_type,
            self.softmax_type,
            self.deterministic,
            softmax_offset,
        )
        return output_tensors[0]


BACKEND_F16arb_ELTS_PER_THREADS = 16


@torch.library.custom_op("groot::fused_attn", mutates_args=())
def fused_attn(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fake_dtype: torch.dtype,
    window_size: List[int],
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    page_table_k: torch.Tensor = None,
    page_table_v: torch.Tensor = None,
    attn_scale: Optional[float] = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    deterministic: bool = False,
    softmax_offset: torch.Tensor = None,
) -> List[torch.Tensor]:
    assert deterministic is not None

    rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS
    s_quantizer = None
    o_quantizer = None
    rng_gen = None

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    args = (
        max_seqlen_q,
        max_seqlen_kv,
        is_training,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
    )

    if _TE_VER >= (2, 8):
        args += (SoftmaxType[softmax_type],)

    args += (
        tuple(window_size),
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        fake_dtype,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        page_table_k,
        page_table_v,
        s_quantizer,
        o_quantizer,
        attn_bias,
    )

    if _TE_VER >= (2, 8):
        args += (softmax_offset,)

    args += (
        rng_gen,
        rng_elts_per_thread,
    )

    if _TE_VER >= (2, 9):
        # return_max_logit
        args += (False,)

    if _TE_VER >= (2, 10):
        # is_cuda_graph
        args += (False,)

    output_tensors = tex.fused_attn_fwd(*args)
    return output_tensors


@fused_attn.register_fake
def _(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fake_dtype: torch.dtype,
    window_size: List[int],
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    page_table_k: torch.Tensor = None,
    page_table_v: torch.Tensor = None,
    attn_scale: Optional[float] = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    deterministic: bool = False,
    softmax_offset: torch.Tensor = None,
) -> List[torch.Tensor]:
    return [
        q.new_empty(tuple(q.shape[:-1]) + (v.shape[-1],)),
        q.new_empty(
            q.shape[0], q.shape[2], q.shape[1], 1, dtype=torch.float32
        ),  # these are the softmax outputs from cudnn; will always be float32
        q.new_empty((2,), dtype=torch.int64),
    ]


def fused_attn_bwd_setup_context(ctx, inputs, output) -> None:
    (
        _,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        _,
        window_size,
        _,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        _,
        _,
        attn_scale,
        dropout,
        fast_zero_fill,
        qkv_layout,
        attn_bias_type,
        attn_mask_type,
        softmax_type,
        deterministic,
        _,
    ) = inputs

    out = output[0]
    aux_ctx_tensors = output[1:]
    qkvo_tensors = (q, k, v, out)

    # assume fwd and bwd always use the same high precision, i.e. torch.float16 or torch.bfloat16
    # used when some tensors are base tensors and loose the "dtype" attribute
    ctx.nominal_dtype = q.dtype

    tensors_to_save, tensor_objects = prepare_for_saving(
        *qkvo_tensors,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        *aux_ctx_tensors,
    )
    ctx.save_for_backward(*tensors_to_save)
    ctx.tensor_objects = tensor_objects

    ctx.QKV_quantizer = None
    ctx.O_quantizer = None
    ctx.dQKV_quantizer = None
    ctx.dO_quantizer = None
    ctx.dP_quantizer = None
    ctx.S_quantizer = None

    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_kv = max_seqlen_kv
    ctx.attn_scale = attn_scale
    ctx.dropout_p = dropout
    ctx.fast_zero_fill = fast_zero_fill
    ctx.qkv_layout = qkv_layout
    ctx.attn_bias_type = attn_bias_type
    ctx.attn_mask_type = attn_mask_type
    ctx.softmax_type = softmax_type
    ctx.window_size = window_size
    ctx.fused_attention_backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    ctx.deterministic = deterministic


@torch.library.custom_op("groot::fused_attn_bwd_op", mutates_args=())
def fused_attn_bwd_op(
    max_seqlen_q: int,
    max_seqlen_kv: int,
    attn_scale: float,
    dropout: float,
    fast_zero_fill: bool,
    qkv_layout: str,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: List[int],
    deterministic: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    d_out: torch.Tensor,
    dqkv_nominal_dtype: torch.dtype,
    dqkv_te_dtype: torch.dtype,
    aux_ctx_tensors: List[torch.Tensor],
    cu_seqlens_q_padded: torch.Tensor,
    cu_seqlens_kv_padded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    args = (
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
    )

    if _TE_VER >= (2, 8):
        args += (SoftmaxType[softmax_type],)

    args += (
        window_size,
        deterministic,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        out,
        d_out,
        dqkv_nominal_dtype,
        TE_DType[dqkv_te_dtype],
        aux_ctx_tensors,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        None,  # s_quantizer,
        None,  # dp_quantizer,
        None,  # dqkv_quantizer,
    )

    if _TE_VER >= (2, 10):
        # is_cuda_graph
        args += (False,)

    dq, dk, dv, *rest = tex.fused_attn_bwd(*args)
    return dq, dk, dv


@fused_attn_bwd_op.register_fake
def _(
    max_seqlen_q: int,
    max_seqlen_kv: int,
    attn_scale: float,
    dropout: float,
    fast_zero_fill: bool,
    qkv_layout: str,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: List[int],
    deterministic: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    d_out: torch.Tensor,
    dqkv_nominal_dtype: torch.dtype,
    dqkv_te_dtype: torch.dtype,
    aux_ctx_tensors: List[torch.Tensor],
    cu_seqlens_q_padded: torch.Tensor,
    cu_seqlens_kv_padded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def fused_attn_bwd_impl(ctx, grad):
    d_out, _, _ = grad
    d_out = d_out.contiguous()

    (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        *other_tensors,
    ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

    aux_ctx_tensors = other_tensors

    if not aux_ctx_tensors[0].is_contiguous():
        aux_ctx_tensors[0] = aux_ctx_tensors[0].contiguous()

    with torch.cuda.nvtx.range("FusedAttnFunc.backward"):
        assert ctx.fused_attention_backend != FusedAttnBackend["No_Backend"], (
            "Fused attention does not support this input combination."
        )

        # get nominal data type of dq, dk, dv
        # FP16/BF16 attention: torch.float16 or torch.bfloat16
        dqkv_nominal_dtype = ctx.nominal_dtype

        # q, k, v, out, d_out, dq, dk, dv: torch.Tensor; torch.float16 or torch.bfloat16
        dq, dk, dv = fused_attn_bwd_op(
            ctx.max_seqlen_q,
            ctx.max_seqlen_kv,
            ctx.attn_scale,
            ctx.dropout_p,
            ctx.fast_zero_fill,
            ctx.qkv_layout,
            ctx.attn_bias_type,
            ctx.attn_mask_type,
            ctx.softmax_type,
            ctx.window_size,
            ctx.deterministic,
            cu_seqlens_q,
            cu_seqlens_kv,
            q,
            k,
            v,
            out,
            d_out,
            dqkv_nominal_dtype,
            d_out.dtype,
            aux_ctx_tensors,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        )

        output = (
            None,  # is_training
            None,  # max_seqlen_q
            None,  # max_seqlen_kv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_kv
            dq,
            dk,
            dv,
            None,  # fake_dtype
            None,  # window_size
            None,  # d_bias,  # attn_bias
            None,  # cu_seqlens_q_padded
            None,  # cu_seqlens_kv_padded
            None,  # page_table_k
            None,  # page_table_v
            None,  # attn_scale
            None,  # dropout
            None,  # fast_zero_fill
            None,  # qkv_layout
            None,  # attn_bias_type
            None,  # attn_mask_type
            None,  # softmax_type
            None,  # deterministic
            None,  # d_softmax_offset,  # softmax_offset
        )
        return output


fused_attn.register_autograd(fused_attn_bwd_impl, setup_context=fused_attn_bwd_setup_context)
