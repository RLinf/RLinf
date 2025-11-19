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

import logging
import math
from inspect import isfunction
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from packaging import version
from torch import Tensor

logpy = logging.getLogger(__name__)

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except Exception:
    XFORMERS_IS_AVAILABLE = False
    logpy.warning("no module 'xformers'. Processing without...")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        logpy.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = (
            t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous()
            for t in (q, k, v)
        )

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = []
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-5, affine=True
    )


class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, up_time=False, causal=False):
        super().__init__()
        self.with_conv = with_conv
        assert with_conv is True
        self.up_time = 2 if up_time else 1
        self.causal = causal
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels,
                in_channels * self.up_time,
                kernel_size=3,
                stride=1,
                padding=0,
            )

    def forward(self, x):
        ori_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float16)
        x = torch.nn.functional.interpolate(
            x, scale_factor=(1.0, 2.0, 2.0), mode="nearest"
        )

        if self.with_conv:
            x = torch.nn.functional.pad(x, (1, 1, 1, 1, 0, 0), mode="constant", value=0)
            if not self.causal:
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
            else:
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 2, 0), mode="replicate")
            x = x.to(ori_dtype)
            x = self.conv(x)
            x = rearrange(x, "b (n c) t h w -> b c (t n) h w", n=self.up_time)
            if self.up_time != 1:
                x = x[:, :, 1:, :, :]
        else:
            x = x.to(ori_dtype)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, down_time=False):
        super().__init__()
        self.with_conv = with_conv
        assert with_conv is True
        self.down_time = down_time
        self.stride = 2 if self.down_time else (1, 2, 2)
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=self.stride, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            x = torch.nn.functional.pad(x, (0, 1, 0, 1, 0, 0), mode="constant", value=0)
            if x.dtype == torch.bfloat16:
                x = torch.nn.functional.pad(
                    x.to(torch.float16), (0, 0, 0, 0, 2, 0), mode="replicate"
                ).to(torch.bfloat16)
            else:
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 2, 0), mode="replicate")
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=self.stride, stride=2)
        return x


class CausalConv3d(torch.nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "replicate",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
            "zeros",
            device,
            dtype,
        )
        assert isinstance(padding, int)
        self.custom_padding = padding
        self.time_padding_mode = padding_mode

    def forward(self, input: Tensor) -> Tensor:
        ori_dtype = input.dtype
        input = input.to(dtype=torch.float32)
        input = torch.nn.functional.pad(
            input,
            (
                self.custom_padding,
                self.custom_padding,
                self.custom_padding,
                self.custom_padding,
                0,
                0,
            ),
            mode="constant",
            value=0,
        )

        if input.dtype == torch.bfloat16:
            input = torch.nn.functional.pad(
                input.to(torch.float16),
                (0, 0, 0, 0, 2 * self.custom_padding, 0),
                mode=self.time_padding_mode,
            ).to(torch.bfloat16)
        else:
            input = torch.nn.functional.pad(
                input,
                (0, 0, 0, 0, 2 * self.custom_padding, 0),
                mode=self.time_padding_mode,
            )
        input = input.to(ori_dtype)
        return super().forward(input)


class Conv2dWithExtraDim(torch.nn.Conv2d):
    def forward(self, input: Tensor) -> Tensor:
        if input.dim() == 5:
            b, c, t, h, w = input.shape
            input = rearrange(input, "b c t h w -> (b t) c h w")
            output = super().forward(input)
            output = rearrange(output, "(b t) c h w -> b c t h w", b=b, t=t)
        else:
            output = super().forward(input)
        return output


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        use_3d_conv=True,
        half_3d=True,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        conv_cls = CausalConv3d if causal else nn.Conv3d
        conv_cls = conv_cls if use_3d_conv else Conv2dWithExtraDim

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_cls(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        if half_3d:
            self.conv2 = Conv2dWithExtraDim(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv2 = conv_cls(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_cls(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = conv_cls(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = (
            rearrange(x, "b c h w -> b 1 (h w) c").contiguous() for x in (q, k, v)
        )
        h_ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        b, c, t, h, w = h_.shape
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", b=b, t=t)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = (rearrange(x, "b c h w -> b (h w) c") for x in (q, k, v))

        q, k, v = (
            t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous()
            for t in (q, k, v)
        )
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        b, c, t, h, w = h_.shape
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", b=b, t=t)
        return x + h_


class MemoryEfficientAttnVideoBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

        self.q_t = torch.nn.Linear(in_channels, in_channels)
        self.k_t = torch.nn.Linear(in_channels, in_channels)
        self.v_t = torch.nn.Linear(in_channels, in_channels)
        self.proj_out_t = torch.nn.Linear(in_channels, in_channels)
        self.norm_t = torch.nn.LayerNorm(in_channels)

    def attention_t(self, h_: torch.Tensor) -> torch.Tensor:
        # (b h w) t c
        h_ = self.norm_t(h_)
        # compute attention
        q = self.q_t(h_).contiguous()
        k = self.k_t(h_).contiguous()
        v = self.v_t(h_).contiguous()

        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = self.proj_out_t(out)

        return out

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = (rearrange(x, "b c h w -> b (h w) c") for x in (q, k, v))

        q, k, v = (
            t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous()
            for t in (q, k, v)
        )
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        b, c, t, h, w = h_.shape
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        # h_ = rearrange(h_, "(b t) c h w -> b c t h w", b=b, t=t)
        h_ = rearrange(h_, "(b t) c h w -> (b h w) t c", b=b, t=t)
        h_ = self.attention_t(h_)
        h_ = rearrange(h_, "(b h w) t c -> b c t h w", b=b, h=h, w=w)
        return x + h_


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "spatial-temporal-xformer",
        "memory-efficient-cross-attn",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if (
        version.parse(torch.__version__) < version.parse("2.0.0")
        and attn_type != "none"
    ):
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        attn_type = "vanilla-xformers"
    logpy.info(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        logpy.info(
            f"building MemoryEfficientAttnBlock with {in_channels} in_channels..."
        )
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == "spatial-temporal-xformer":
        return MemoryEfficientAttnVideoBlock(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        z_channels,
        resolution=256,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        use_3d_conv=True,
        half_3d=True,
        causal=True,
        half_t_mult=True,
        gradient_checkpointing=True,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing

        conv_cls = CausalConv3d if causal else nn.Conv3d
        conv_cls = conv_cls if use_3d_conv else Conv2dWithExtraDim

        # downsampling
        self.conv_in = conv_cls(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        use_3d_conv=use_3d_conv,
                        half_3d=half_3d,
                        causal=causal,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if half_t_mult:
                    down_time = i_level % 2 == 0
                else:
                    down_time = True
                down.downsample = Downsample3D(block_in, resamp_with_conv, down_time)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_3d_conv=use_3d_conv,
            half_3d=half_3d,
            causal=causal,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_3d_conv=use_3d_conv,
            half_3d=half_3d,
            causal=causal,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_cls(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        self.down[i_level].block[i_block].__call__,
                        hs[-1],
                        temb,
                        use_reentrant=False,
                    )
                else:
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        z_channels,
        resolution=256,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        use_3d_conv=True,
        half_3d=True,
        causal=True,
        half_t_mult=True,
        gradient_checkpointing=True,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.gradient_checkpointing = gradient_checkpointing

        # compute in_ch_mult, block_in and curr_res at lowest res
        (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logpy.info(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )
        self.causal = causal
        self.use_3d_conv = use_3d_conv

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        self.conv_in = make_conv_cls(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_3d_conv=use_3d_conv,
            half_3d=half_3d,
            causal=causal,
        )
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_3d_conv=use_3d_conv,
            half_3d=half_3d,
            causal=causal,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        use_3d_conv=use_3d_conv,
                        half_3d=half_3d,
                        causal=causal,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if half_t_mult:
                    up_time = i_level % 2 == 1
                else:
                    up_time = True
                up.upsample = Upsample3D(block_in, resamp_with_conv, up_time)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetBlock3D

    def _make_conv(self) -> Callable:
        conv_cls = CausalConv3d if self.causal else nn.Conv3d
        conv_cls = conv_cls if self.use_3d_conv else Conv2dWithExtraDim
        return conv_cls

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        self.up[i_level].block[i_block].__call__,
                        h,
                        temb,
                        use_reentrant=False,
                        **kwargs,
                    )
                else:
                    h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
