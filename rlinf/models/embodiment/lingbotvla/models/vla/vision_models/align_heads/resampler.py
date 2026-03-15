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

# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(seq_len + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_all_tokens=False):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x.permute(1, 0, 2)  # (N(HW)C) => (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        if return_all_tokens:
            return x
        else:
            return x[0]


class Resampler(nn.Module):
    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(1, num_queries, dim_in) / dim_mid**0.5)

        self.proj_in = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim_mid, dim_head=dim_head, heads=num_heads
                        ),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        queries = self.queries.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class TaskTokenResampler(nn.Module):
    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim_mid, dim_head=dim_head, heads=num_heads
                        ),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class ResamplerXL(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output1_dim=768,
        output2_dim=1280,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        # self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(dim)

        self.in_dim = dim
        self.out_dim = output1_dim + output2_dim

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.unet_proj_1 = nn.Linear(self.in_dim, output1_dim)
        self.unet_proj_2 = nn.Linear(self.in_dim, output2_dim)
        self.unet_attnpool = AttentionPool2d(
            num_queries, self.in_dim, heads, output2_dim
        )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        hidden_embeds = self.norm_out(latents)

        encoder_hidden_1 = self.unet_proj_1(hidden_embeds)  # [bs, 256, 768]
        encoder_hidden_2 = self.unet_proj_2(hidden_embeds)  # [bs, 256, 1280]
        prompt_embeds = torch.cat(
            [encoder_hidden_1, encoder_hidden_2], dim=-1
        )  # [bs, 256, 2048]
        pooled_prompt_embeds = self.unet_attnpool(hidden_embeds)  # [bs, 1280]

        return prompt_embeds, pooled_prompt_embeds


class ResamplerXLV2(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output1_dim=768,
        output2_dim=1280,
        ff_mult=4,
        normalize=True,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.normalize = normalize
        self.proj_in = nn.Linear(embedding_dim, dim)

        # self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(dim)

        self.in_dim = dim
        self.out_dim = output1_dim + output2_dim

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.unet_proj_1 = nn.Linear(self.in_dim, output1_dim)
        self.unet_proj_2 = nn.Linear(self.in_dim, output2_dim)
        self.unet_attnpool = AttentionPool2d(
            num_queries, self.in_dim, heads, output2_dim
        )

    def forward(self, x, pooled_text_embeds=None):
        latents = self.latents.repeat(x.size(0), 1, 1)

        if self.normalize:
            x = F.normalize(x)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        hidden_embeds = self.norm_out(latents)

        encoder_hidden_1 = self.unet_proj_1(hidden_embeds)  # [bs, 256, 768]
        encoder_hidden_2 = self.unet_proj_2(hidden_embeds)  # [bs, 256, 1280]
        prompt_embeds = torch.cat(
            [encoder_hidden_1, encoder_hidden_2], dim=-1
        )  # [bs, 256, 2048]
        pooled_prompt_embeds = self.unet_attnpool(hidden_embeds)  # [bs, 1280]

        return prompt_embeds, pooled_prompt_embeds


class ResamplerXLIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, pooled_text_embeds=None):
        return x, pooled_text_embeds


if __name__ == "__main__":
    image_proj_model = Resampler(
        dim=1024,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=1024,
        embedding_dim=1024,
        output_dim=1024,
        ff_mult=4,
    )
    numel = 0
    for name, param in image_proj_model.named_parameters():
        numel += param.numel()

    print(f"Total params: {numel}")
