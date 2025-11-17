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

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except Exception:
    raise RuntimeError("xformers is not installed correctly.")
    XFORMERS_IS_AVAILBLE = False

try:
    from rlinf.envs.worldmodel.evac.lvdm.basics import zero_module
    from rlinf.envs.worldmodel.evac.lvdm.common import (
        checkpoint,
        default,
        exists,
    )
except Exception:
    from rlinf.envs.worldmodel.evac.lvdm.basics import zero_module
    from rlinf.envs.worldmodel.evac.lvdm.common import (
        checkpoint,
        default,
        exists,
    )


class RelativePosition(nn.Module):
    """https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py"""

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


# MAX_XFORMERS_BATCH_SIZE = 2^31 - 1
# https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Dimensions
# https://github.com/facebookresearch/xformers/issues/998
def xformers_attn(q, k, v, MAX_XFORMERS_BATCH_SIZE=65535):
    if q.shape[0] <= MAX_XFORMERS_BATCH_SIZE:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
    else:
        outs = []
        n_batch = int(math.ceil(float(q.shape[0]) / MAX_XFORMERS_BATCH_SIZE))
        for _i in range(n_batch):
            sidx = _i * MAX_XFORMERS_BATCH_SIZE
            eidx = min(q.shape[0], (_i + 1) * MAX_XFORMERS_BATCH_SIZE)
            outs.append(
                xformers.ops.memory_efficient_attention(
                    q[sidx:eidx], k[sidx:eidx], v[sidx:eidx], attn_bias=None, op=None
                )
            )
        out = torch.cat(outs, dim=0)
    return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        relative_position=False,
        temporal_length=None,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        image_cross_attention_scale_learnable=False,
        text_context_len=77,
        image_context_len=256,
        traj_cross_attention=False,
        traj_cross_attention_scale_learnable=False,
        traj_context_len=16,
        traj_cross_attention_scale=1,
        block_idx=-1,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.block_idx = block_idx
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

        self.relative_position = relative_position
        if self.relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length
            )
            self.relative_position_v = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length
            )
        else:
            temporal_length = None
            # only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                # print('NOTE: Using XFORMERS')
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = (
            image_cross_attention_scale_learnable
        )
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            if image_cross_attention_scale_learnable:
                self.register_parameter("alpha", nn.Parameter(torch.tensor(0.0)))

        self.traj_cross_attention = traj_cross_attention
        self.traj_cross_attention_scale = traj_cross_attention_scale
        self.traj_cross_attention_scale_learnable = traj_cross_attention_scale_learnable
        if self.traj_cross_attention:
            self.text_context_len = 0
            self.to_k_tp = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_tp = nn.Linear(context_dim, inner_dim, bias=False)
            if traj_cross_attention_scale_learnable:
                self.register_parameter("beta", nn.Parameter(torch.tensor(0.0)))

        self.image_context_len = image_context_len
        self.traj_context_len = traj_context_len

        self.infer_block_batchsize = 1

    def forward(self, x, context=None, mask=None, seq_length=None):
        spatial_self_attn = context is None
        k_ip, v_ip, out_ip = None, None, None

        k_vpre, v_vpre, out_vpre = None, None, None

        h = self.heads
        q = self.to_q(x)
        print(f"q shape: {q.shape}")
        context = default(context, x)
        print(f"context shape: {context.shape}")
        print(
            f"image_cross_attention: {self.image_cross_attention}, traj_cross_attention: {self.traj_cross_attention}, spatial_self_attn: {spatial_self_attn}"
        )
        if (
            self.image_cross_attention
            and self.traj_cross_attention
            and not spatial_self_attn
        ):
            num_v = context.shape[1] // (
                self.text_context_len + self.image_context_len + self.traj_context_len
            )
            context = rearrange(context, "b (v l) c -> b v l c", v=num_v)

            if self.block_idx < 0:
                context_image, context_traj = (
                    context[
                        :,
                        :,
                        self.text_context_len : (
                            self.text_context_len + self.image_context_len
                        ),
                        :,
                    ],
                    context[
                        :,
                        :,
                        self.text_context_len + self.image_context_len : (
                            self.text_context_len
                            + self.image_context_len
                            + self.traj_context_len
                        ),
                        :,
                    ],
                )
                # context = rearrange(context, 'b v l c -> b (v l) c')[:,:self.text_context_len]
                # k = self.to_k(context)
                # v = self.to_v(context)
                context_image = rearrange(context_image, "b v l c -> b (v l) c")
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
                context_traj = rearrange(context_traj, "b v l c -> b (v l) c")
                self.to_k_tp(context_traj)
                self.to_v_tp(context_traj)

            else:
                if self.block_idx % 2 == 0:
                    context = context[
                        :,
                        :,
                        (self.text_context_len + self.image_context_len) : (
                            self.text_context_len
                            + self.image_context_len
                            + self.traj_context_len
                        ),
                        :,
                    ]
                    context = rearrange(context, "b v l c -> b (v l) c")[
                        :,
                        (self.text_context_len + self.image_context_len) : (
                            self.text_context_len
                            + self.image_context_len
                            + self.traj_context_len
                        ),
                    ]
                    self.to_k_tp(context)
                    self.to_v_tp(context)
                else:
                    context_image = context[
                        :,
                        :,
                        self.text_context_len : (
                            self.text_context_len + self.image_context_len
                        ),
                        :,
                    ]
                    context_image = rearrange(context_image, "b v l c -> b (v l) c")
                    k_ip = self.to_k_ip(context_image)
                    v_ip = self.to_v_ip(context_image)

        elif self.image_cross_attention and not spatial_self_attn:
            context, context_image = (
                context[:, : self.text_context_len, :],
                context[
                    :,
                    self.text_context_len : (
                        self.text_context_len + self.image_context_len
                    ),
                    :,
                ],
            )
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)

        else:
            if not spatial_self_attn:
                if not seq_length:
                    context = context[:, : self.text_context_len, :]
                else:
                    context = context[:, :seq_length, :]
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        print(f"relative_position: {self.relative_position}")
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum("b t d, t s d -> b t s", q, k2) * self.scale  # TODO check
            sim += sim2
        del k

        if exists(mask):
            # feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            sim.masked_fill_(~(mask > 0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", sim, v)

        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum("b t s, t s d -> b t d", sim, v2)  # TODO check
            out += out2
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        print(
            f"k_ip is not None: {k_ip is not None}, k_vpre is not None: {k_vpre is not None}"
        )
        # for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = (
                rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (k_ip, v_ip)
            )
            sim_ip = torch.einsum("b i d, b j d -> b i j", q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum("b i j, b j d -> b i d", sim_ip, v_ip)
            out_ip = rearrange(out_ip, "(b h) n d -> b n (h d)", h=h)

        print(
            f"out_ip is not None: {out_ip is not None}, out_vpre is not None: {out_vpre is not None}"
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (
                    torch.tanh(self.alpha) + 1
                )
            else:
                out = out + self.image_cross_attention_scale * out_ip

        # for vpre cross-attention
        if k_vpre is not None:
            k_vpre, v_vpre = (
                rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (k_vpre, v_vpre)
            )
            sim_vpre = torch.einsum("b i d, b j d -> b i j", q, k_vpre) * self.scale
            del k_vpre
            sim_vpre = sim_vpre.softmax(dim=-1)
            out_vpre = torch.einsum("b i j, b j d -> b i d", sim_vpre, v_vpre)
            out_vpre = rearrange(out_vpre, "(b h) n d -> b n (h d)", h=h)

        if out_vpre is not None:
            out = out + out_vpre

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None, seq_length=None):
        spatial_self_attn = context is None
        k, v, out = None, None, None
        k_ip, v_ip, out_ip = None, None, None
        k_tp, v_tp, out_tp = None, None, None
        k_vpre, v_vpre, out_vpre = None, None, None
        # [2560, 20, 512]
        # print(f'x shape: {x.shape}')
        # print(f'context shape: {context.shape}' if context is not None else 'context is None')
        q = self.to_q(x)
        # print(f'q shape: {q.shape}')
        context = default(context, x)
        # print(f'context shape: {context.shape}')
        # print(f'image_cross_attention: {self.image_cross_attention}, traj_cross_attention: {self.traj_cross_attention}, spatial_self_attn: {spatial_self_attn}')
        if (
            self.image_cross_attention
            and self.traj_cross_attention
            and not spatial_self_attn
        ):
            num_v = context.shape[1] // (
                self.text_context_len + self.image_context_len + self.traj_context_len
            )
            # 总的token数/每个视角的token数 = 视角数
            context = rearrange(context, "b (v l) c -> b v l c", v=num_v)
            if self.block_idx < 0:
                context_image, context_traj = (
                    context[
                        :,
                        :,
                        self.text_context_len : (
                            self.text_context_len + self.image_context_len
                        ),
                        :,
                    ],
                    context[
                        :,
                        :,
                        self.text_context_len + self.image_context_len : (
                            self.text_context_len
                            + self.image_context_len
                            + self.traj_context_len
                        ),
                        :,
                    ],
                )
                # context = rearrange(context, 'b v l c -> b (v l) c')[:,:self.text_context_len]
                # k = self.to_k(context)
                # v = self.to_v(context)
                # 切分出图像提示和轨迹提示
                context_image = rearrange(context_image, "b v l c -> b (v l) c")
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
                context_traj = rearrange(context_traj, "b v l c -> b (v l) c")
                k_tp = self.to_k_tp(context_traj)
                v_tp = self.to_v_tp(context_traj)
            else:
                if self.block_idx % 2 == 0:
                    context = rearrange(context, "b v l c -> b (v l) c")[
                        :,
                        (self.text_context_len + self.image_context_len) : (
                            self.text_context_len
                            + self.image_context_len
                            + self.traj_context_len
                        ),
                    ]
                    k_tp = self.to_k_tp(context)
                    v_tp = self.to_v_tp(context)
                else:
                    context_image = context[
                        :,
                        :,
                        self.text_context_len : (
                            self.text_context_len + self.image_context_len
                        ),
                        :,
                    ]
                    context_image = rearrange(context_image, "b v l c -> b (v l) c")
                    k_ip = self.to_k_ip(context_image)
                    v_ip = self.to_v_ip(context_image)

        elif self.image_cross_attention and not spatial_self_attn:
            context, context_image = (
                context[:, : self.text_context_len, :],
                context[
                    :,
                    self.text_context_len : (
                        self.text_context_len + self.image_context_len
                    ),
                    :,
                ],
            )
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)

        else:
            if not spatial_self_attn:
                if not seq_length:
                    context = context[:, : self.text_context_len, :]
                else:
                    context = context[:, :seq_length, :]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        # [2560, 20, 512]

        if k is not None:
            q, k, v = (
                t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                # [2560, 20, 8, 64]
                # H * W 拆成了 8 个头，每个头64维度，通道数2560
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous()
                for t in (q, k, v)
            )
            # [20480, 20, 64]
            # (b * heads, n, dim_head)
            # actually compute the attention, what we cannot get enough of
            # out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
            out = xformers_attn(q, k, v)

            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            # (b, n, heads * dim_head)
            # print(f'k out shape: {out.shape}')
        # for image cross-attention
        if k_ip is not None:
            if k is not None:
                k_ip, v_ip = (
                    t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous()
                    for t in (k_ip, v_ip)
                )
            else:
                q, k_ip, v_ip = (
                    t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous()
                    for t in (q, k_ip, v_ip)
                )
            # out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = xformers_attn(q, k_ip, v_ip)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out_ip.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out_ip.shape[1], self.heads * self.dim_head)
            )

        # for vpre cross-attention
        if k_vpre is not None:
            k_vpre, v_vpre = (
                t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous()
                for t in (k_vpre, v_vpre)
            )
            # (batch*heads, token数, dim_head)
            # out_vpre = xformers.ops.memory_efficient_attention(q, k_vpre, v_vpre, attn_bias=None, op=None)
            out_vpre = xformers_attn(q, k_vpre, v_vpre)
            out_vpre = (
                out_vpre.unsqueeze(0)
                .reshape(b, self.heads, out_vpre.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out_vpre.shape[1], self.heads * self.dim_head)
            )

        # for tp cross-attention
        if k_tp is not None:
            if k is not None or k_ip is not None:
                k_tp, v_tp = (
                    t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous()
                    for t in (k_tp, v_tp)
                )
            else:
                q, k_tp, v_tp = (
                    t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous()
                    for t in (q, k_tp, v_tp)
                )
            # out_tp = xformers.ops.memory_efficient_attention(q, k_tp, v_tp, attn_bias=None, op=None)
            out_tp = xformers_attn(q, k_tp, v_tp)
            out_tp = (
                out_tp.unsqueeze(0)
                .reshape(b, self.heads, out_tp.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out_tp.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        if out_ip is not None:
            if out is not None:
                if self.image_cross_attention_scale_learnable:
                    out = out + self.image_cross_attention_scale * out_ip * (
                        torch.tanh(self.alpha) + 1
                    )
                else:
                    out = out + self.image_cross_attention_scale * out_ip
            else:
                out = out_ip
        if out_vpre is not None:
            out = out + out_vpre
        if out_tp is not None:
            if out is not None:
                if self.traj_cross_attention_scale_learnable:
                    out = out + self.traj_cross_attention_scale * out_tp * (
                        torch.tanh(self.beta) + 1
                    )
                else:
                    out = out + self.traj_cross_attention_scale * out_tp
            else:
                out = out_tp
        # input('x')
        # [20, 2560, 320]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attention_cls=None,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        image_cross_attention_scale_learnable=False,
        text_context_len=77,
        image_context_len=256,
        traj_cross_attention=False,
        traj_cross_attention_scale_learnable=False,
        traj_context_len=16,
        block_idx=-1,
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            video_length=video_length,
            image_cross_attention=image_cross_attention,
            image_cross_attention_scale=image_cross_attention_scale,
            image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
            text_context_len=text_context_len,
            image_context_len=image_context_len,
            traj_cross_attention=traj_cross_attention,
            traj_cross_attention_scale_learnable=traj_cross_attention_scale_learnable,
            traj_context_len=traj_context_len,
            block_idx=block_idx,
        )

        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, **kwargs):
        # implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  # should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, input_tuple, self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None, mask=None):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask,
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SigleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attention_cls=None,
        image_cross_attention=False,
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, seq_length=None, **kwargs):
        # implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  # should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if seq_length is not None:
            forward_part = partial(self._forward, seq_length=seq_length)
            return checkpoint(
                forward_part, input_tuple, self.parameters(), self.checkpoint
            )
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, input_tuple, self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None, mask=None, seq_length=None):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask,
                seq_length=seq_length,
            )
            + x
        )
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        disable_self_attn=False,
        use_linear=False,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        chunk=8,
        parallel=False,
        use_block_idx=False,
        block_idx=-1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.parallel = parallel
        self.use_block_idx = use_block_idx
        if self.use_block_idx:
            assert block_idx >= 0
        else:
            block_idx = -1
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    attention_cls=attention_cls,
                    video_length=video_length,
                    image_cross_attention=image_cross_attention,
                    image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                    block_idx=block_idx,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        self.use_cache = False
        self.video_length = video_length
        self.chunk = chunk

    def init_cache(self, denoise_step=50, chunk=1):
        self.use_cache = True
        self.denoise_step = 0

        self.denoise_oneloop_step = denoise_step
        self.cache_pool = [[] for _ in range(self.denoise_oneloop_step)]

    def clean_cache(self, denoise_step=50):
        self.use_cache = False
        self.denoise_step = 0
        self.denoise_oneloop_step = denoise_step
        self.cache_pool = None

    def forward(self, x, context=None, **kwargs):
        if self.parallel:
            x_skip = x

        if self.use_cache and self.denoise_step // self.denoise_oneloop_step > 0:
            # x_reuse = x[:,:,:-1].clone()

            x = rearrange(x, "(b t) c h w -> b c t h w", t=self.video_length)
            x = x[:, :, -(2 * self.chunk) :]
            x = rearrange(x, "b c t h w -> (b t) c h w")

            context = rearrange(context, "(b t) l c -> b t l c", t=self.video_length)
            context = context[:, -(2 * self.chunk) :]
            context = rearrange(context, "b t l c -> (b t) l c")

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        res = x + x_in

        if self.use_cache:
            if self.denoise_step // self.denoise_oneloop_step > 0:
                res_reuse = self.cache_pool[
                    self.denoise_step % self.denoise_oneloop_step
                ].pop(0)  # b c t h w

                res = rearrange(res, "(b t) c h w -> b c t h w", t=2 * self.chunk)

                res = torch.cat([res_reuse, res], dim=2)  # b c t h w
                self.cache_pool[self.denoise_step % self.denoise_oneloop_step].append(
                    res[:, :, self.chunk : -self.chunk]
                )
                res = rearrange(res, "b c t h w -> (b t) c h w")
            else:
                res = rearrange(res, "(b t) c h w -> b c t h w", t=self.video_length)
                self.cache_pool[self.denoise_step % self.denoise_oneloop_step].append(
                    res[:, :, self.chunk : -self.chunk]
                )
                res = rearrange(res, "b c t h w -> (b t) c h w")

            self.denoise_step += 1

        if self.parallel:
            return (res, x_skip)
        return res


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        only_self_att=True,
        causal_attention=False,
        causal_block_size=1,
        relative_position=False,
        temporal_length=None,
        chunk=4,
        temporal_batch_size=-1,
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size
        self.chunk = chunk
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Conv1d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert temporal_length is not None
            attention_cls = partial(
                CrossAttention, relative_position=True, temporal_length=temporal_length
            )
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert temporal_length is not None
            # self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

            idx = []
            for i in range(temporal_length // self.chunk):
                idx += [i] * self.chunk
            idx = torch.tensor(idx)

            mask = (
                torch.arange(0, temporal_length, 1)
                .unsqueeze(1)
                .repeat(1, temporal_length)
            )
            mask = (mask >= idx).unsqueeze(0).float()
            self.mask = mask

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        self.temporal_batch_size = temporal_batch_size

    def forward(self, x, context=None):
        b, c, v, t, h, w = x.shape

        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c v t h w -> (b v h w) c t").contiguous()

        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "bvhw c t -> bvhw t c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # slice the from mask map
            temp_mask = self.mask[:, :t, :t].to(x.device)

        if self.temporal_batch_size <= 0 and temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, "l i j -> (l bvhw) i j", bvhw=b * v * h * w)
        elif temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, "l i j -> (l k) i j", k=self.temporal_batch_size)
        else:
            mask = None

        if self.only_self_att:
            # note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                if self.temporal_batch_size > 0:
                    n_batch = int(
                        math.ceil(float(x.shape[0]) / self.temporal_batch_size)
                    )
                    for _i in range(n_batch):
                        sidx = _i * self.temporal_batch_size
                        eidx = min(x.shape[0], (_i + 1) * self.temporal_batch_size)
                        x[sidx:eidx] = block(
                            x[sidx:eidx],
                            mask=mask[: (eidx - sidx)] if mask is not None else None,
                        )
                else:
                    x = block(x, mask=mask)
            x = rearrange(
                x, "(b v h w) t c -> (b v) (h w) t c", b=b, v=v, h=h
            ).contiguous()
        else:
            x = rearrange(
                x, "(b v h w) t c -> (b v) (h w) t c", b=b, v=v, h=h
            ).contiguous()
            context = rearrange(
                context, "(b v t) l con -> (b v) t l con", t=t, b=b
            ).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(context.shape[0]):
                    context_j = repeat(
                        context[j], "t l con -> (t r) l con", r=(h * w) // t, t=t
                    ).contiguous()
                    # note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(
                x, "(b v) (h w) t c -> b c v t h w", h=h, w=w, v=v
            ).contiguous()
        if not self.use_linear:
            x = rearrange(
                x, "(b v) (h w) t c -> (b v h w) c t", b=b, v=v, h=h, w=w
            ).contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, "(b v h w) c t -> b c v t h w", b=b, h=h, w=w, v=v
            ).contiguous()

        return x + x_in


class MultiViewTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        only_self_att=True,
        relative_position=False,
        temporal_length=None,
        chunk=8,
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.chunk = chunk
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Conv1d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert temporal_length is not None
            attention_cls = partial(
                CrossAttention, relative_position=True, temporal_length=temporal_length
            )
        else:
            # attention_cls = partial(CrossAttention, temporal_length=temporal_length)
            attention_cls = partial(CrossAttention)

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList(
            [
                SigleTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, t=None):
        b, c, v, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c v t h w -> (b t) c (v h w)").contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(
            x, "(b t) c (v h w) -> (b t) (v h w) c", b=b, v=v, h=h, w=w
        ).contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        """
        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            x = rearrange(x, '(b t) (v h w) c -> b t (v h w) c', b=b,v=v,h=h,w=w).contiguous()
        else:
            x = rearrange(x, '(b t) (v h w) c -> b t (v h w) c', b=b,v=v,h=h,w=w).contiguous()
            # TODO: the context length is l or accordingly, vl ?
            context = rearrange(context, '(b v t) l con -> b t v l con', t=t, b=b, v=v).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = context[j]
                    x[j] = block(x[j], context=context_j[:,0])
        """

        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
        x = rearrange(
            x, "(b t) (v h w) c -> b t (v h w) c", b=b, v=v, h=h, w=w
        ).contiguous()

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, "b t (v h w) c -> b c v t h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, "b t (v h w) c -> (b t) c (v h w)", h=h, w=w).contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, "(b t) c (v h w) -> b c v t h w", b=b, h=h, w=w
            ).contiguous()

        return x + x_in


class CrossMultiViewTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        only_self_att=False,
        relative_position=False,
        temporal_length=None,
        chunk=8,
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.chunk = chunk
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Conv1d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert temporal_length is not None
            attention_cls = partial(
                CrossAttention, relative_position=True, temporal_length=temporal_length
            )
        else:
            # attention_cls = partial(CrossAttention, temporal_length=temporal_length)
            attention_cls = partial(CrossAttention)

        if self.only_self_att:
            context_dim = None
        # TODO: we force the content dim to be the same as inner_dim
        context_dim = inner_dim
        self.transformer_blocks = nn.ModuleList(
            [
                SigleTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                    disable_self_attn=True,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, t=None):
        b, c, v, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c v t h w -> (b t) c (v h w)").contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(
            x, "(b t) c (v h w) -> (b t) (v h w) c", b=b, v=v, h=h, w=w
        ).contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        x = rearrange(x, "(b t) (v h w) c -> (b t) v (h w) c", b=b, v=v, h=h, w=w)

        """
        x_q = torch.repeat_interleave(x, repeats=v-1, dim=1)
        x_kv = []
        for i in range(v):
            if i == 0:
                x_kv.append(x[:, 1:])
            elif i == v - 1:
                x_kv.append(x[:, :-1])
            else:
                x_kv.append(torch.cat((x[:, :i], x[:, i+1:]), dim=1))
        x_kv = torch.cat(x_kv, dim=1)
        x_q = rearrange(x_q, '(b t) v (h w) c -> (b t v) (h w) c', b=b,t=t,h=h,w=w)
        x_kv = rearrange(x_kv, '(b t) v (h w) c -> (b t v) (h w) c', b=b,t=t,h=h,w=w)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x_q, context=x_kv, seq_length=h*w)
        x = rearrange(x, '(b t v) (h w) c -> (b t) v (h w) c', b=b,t=t,h=h,w=w).contiguous()
        x = rearrange(x, 'bt (v k) hw c -> bt v k hw c', v=v).contiguous()
        # TODO: how to combine the res of cross-atten?
        x = torch.sum(x, dim=2)
        x = rearrange(x, '(b t) v (h w) c -> b t (v h w) c', b=b,v=v,h=h,w=w)
        """

        for _, block in enumerate(self.transformer_blocks):
            middle = []
            for i in range(v):
                if i == 0:
                    middle.append(
                        block(
                            x[:, i],
                            context=rearrange(x[:, 1:], "bt v hw c -> bt (v hw) c"),
                            seq_length=(v - 1) * h * w,
                        )
                    )
                elif i == v - 1:
                    middle.append(
                        block(
                            x[:, i],
                            context=rearrange(x[:, :-1], "bt v hw c -> bt (v hw) c"),
                            seq_length=(v - 1) * h * w,
                        )
                    )
                else:
                    middle.append(
                        block(
                            x[:, i],
                            context=rearrange(
                                torch.cat((x[:, :i], x[:, i + 1 :]), dim=1),
                                "bt v hw c -> bt (v hw) c",
                            ),
                            seq_length=(v - 1) * h * w,
                        )
                    )
            x = torch.stack(middle, dim=1)

        x = rearrange(x, "(b t) v (h w) c -> b t (v h w) c", b=b, v=v, h=h, w=w)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, "b t (v h w) c -> b c v t h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, "b t (v h w) c -> (b t) c (v h w)", h=h, w=w).contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, "(b t) c (v h w) -> b c v t h w", b=b, h=h, w=w
            ).contiguous()

        return x + x_in


class S2MVTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        disable_self_attn=False,
        use_linear=False,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        traj_cross_attention=False,
        traj_cross_attention_scale_learnable=False,
        chunk=8,
        parallel=False,
        use_block_idx=False,
        block_idx=-1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.parallel = parallel
        self.use_block_idx = use_block_idx
        if self.use_block_idx:
            assert block_idx >= 0
        else:
            block_idx = -1
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    attention_cls=attention_cls,
                    video_length=video_length,
                    image_cross_attention=image_cross_attention,
                    image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                    traj_cross_attention=traj_cross_attention,
                    traj_cross_attention_scale_learnable=traj_cross_attention_scale_learnable,
                    block_idx=block_idx,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        self.use_cache = False
        self.video_length = video_length
        self.chunk = chunk

    def init_cache(self, denoise_step=50, chunk=1):
        self.use_cache = True
        self.denoise_step = 0

        self.denoise_oneloop_step = denoise_step
        self.cache_pool = [[] for _ in range(self.denoise_oneloop_step)]

    def clean_cache(self, denoise_step=50):
        self.use_cache = False
        self.denoise_step = 0
        self.denoise_oneloop_step = denoise_step
        self.cache_pool = None

    def forward(self, x, context=None, **kwargs):
        if self.use_cache and self.denoise_step // self.denoise_oneloop_step > 0:
            # x_reuse = x[:,:,:-1].clone()

            x = rearrange(x, "(b t) c h w -> b c t h w", t=self.video_length)
            x = x[:, :, -(2 * self.chunk) :]
            x = rearrange(x, "b c t h w -> (b t) c h w")

            context = rearrange(context, "(b t) l c -> b t l c", t=self.video_length)
            context = context[:, -(2 * self.chunk) :]
            context = rearrange(context, "b t l c -> (b t) l c")

        b, c, h, w = x.shape
        # print(f'S2MV Input x shape: {x.shape}')
        x_in = x
        # 先标准化
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        # [20, 512, 320]
        # print(f'use_linear: {self.use_linear}')T
        if self.use_linear:
            x = self.proj_in(x)
            # 线性层
        # print(f'x shape after proj_in: {x.shape}')
        # [20, 2560, 320]
        for i, block in enumerate(self.transformer_blocks):
            # print('S2MV block:', i)
            x = block(x, context=context, **kwargs)
            # print(f'x shape after block {i}: {x.shape}')
        if self.use_linear:
            x = self.proj_out(x)
        # print(f'x shape after proj_out: {x.shape}')
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        res = x + x_in

        if self.use_cache:
            if self.denoise_step // self.denoise_oneloop_step > 0:
                res_reuse = self.cache_pool[
                    self.denoise_step % self.denoise_oneloop_step
                ].pop(0)  # b c t h w

                res = rearrange(res, "(b t) c h w -> b c t h w", t=2 * self.chunk)

                res = torch.cat([res_reuse, res], dim=2)  # b c t h w
                self.cache_pool[self.denoise_step % self.denoise_oneloop_step].append(
                    res[:, :, self.chunk : -self.chunk]
                )
                res = rearrange(res, "b c t h w -> (b t) c h w")
            else:
                res = rearrange(res, "(b t) c h w -> b c t h w", t=self.video_length)
                self.cache_pool[self.denoise_step % self.denoise_oneloop_step].append(
                    res[:, :, self.chunk : -self.chunk]
                )
                res = rearrange(res, "b c t h w -> (b t) c h w")

            self.denoise_step += 1

        return res


class Transformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        disable_self_attn=False,
        use_linear=False,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        vpre_cross_attention=False,
        chunk=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    attention_cls=attention_cls,
                    video_length=video_length,
                    image_cross_attention=image_cross_attention,
                    image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                    vpre_cross_attention=vpre_cross_attention,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        self.use_cache = False
        self.video_length = video_length
        self.chunk = chunk

    def init_cache(self, denoise_step=50, chunk=1):
        self.use_cache = True
        self.denoise_step = 0

        self.denoise_oneloop_step = denoise_step
        self.cache_pool = [[] for _ in range(self.denoise_oneloop_step)]

    def clean_cache(self, denoise_step=50):
        self.use_cache = False
        self.denoise_step = 0
        self.denoise_oneloop_step = denoise_step
        self.cache_pool = None

    def forward(self, x, context=None, **kwargs):
        if self.use_cache and self.denoise_step // self.denoise_oneloop_step > 0:
            # x_reuse = x[:,:,:-1].clone()

            x = rearrange(x, "(b t) c h w -> b c t h w", t=self.video_length)
            x = x[:, :, -(2 * self.chunk) :]
            x = rearrange(x, "b c t h w -> (b t) c h w")

            context = rearrange(context, "(b t) l c -> b t l c", t=self.video_length)
            context = context[:, -(2 * self.chunk) :]
            context = rearrange(context, "b t l c -> (b t) l c")

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        res = x + x_in

        if self.use_cache:
            if self.denoise_step // self.denoise_oneloop_step > 0:
                res_reuse = self.cache_pool[
                    self.denoise_step % self.denoise_oneloop_step
                ].pop(0)  # b c t h w

                res = rearrange(res, "(b t) c h w -> b c t h w", t=2 * self.chunk)

                res = torch.cat([res_reuse, res], dim=2)  # b c t h w
                self.cache_pool[self.denoise_step % self.denoise_oneloop_step].append(
                    res[:, :, self.chunk : -self.chunk]
                )
                res = rearrange(res, "b c t h w -> (b t) c h w")
            else:
                res = rearrange(res, "(b t) c h w -> b c t h w", t=self.video_length)
                self.cache_pool[self.denoise_step % self.denoise_oneloop_step].append(
                    res[:, :, self.chunk : -self.chunk]
                )
                res = rearrange(res, "b c t h w -> (b t) c h w")

            self.denoise_step += 1

        return res


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


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


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
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

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_
