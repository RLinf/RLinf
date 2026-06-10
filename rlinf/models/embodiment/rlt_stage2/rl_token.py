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

"""RL token module used by RLT Stage 2.

The implementation is intentionally kept close to the original Stage 1/2 code:
Stage 2 only needs the encoder path at inference time, but we keep the full
module so the checkpoint structure stays compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RLTokenEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.e_rl = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        batch_size = z.shape[0]
        e_rl = self.e_rl.expand(batch_size, -1, -1)
        tokens = torch.cat([z, e_rl], dim=1)
        rl_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=z.device)
        extended_pad_mask = torch.cat([pad_mask, rl_mask], dim=1)
        ignore_mask = ~extended_pad_mask
        output = self.transformer(tokens, src_key_padding_mask=ignore_mask)
        return output[:, -1, :]


class RLTokenDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.h_phi = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_rl: Tensor, z: Tensor, pad_mask: Tensor) -> Tensor:
        tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)
        seq_len = tgt.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tgt.device
        )
        memory = z_rl.unsqueeze(1)
        tgt_key_padding_mask = ~pad_mask
        output = self.transformer(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.h_phi(output)


class RLTokenModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = RLTokenEncoder(embedding_dim, encoder_layers, encoder_heads)
        self.decoder = RLTokenDecoder(embedding_dim, decoder_layers, decoder_heads)

    def forward(self, z: Tensor, pad_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = z.detach()
        z_rl = self.encoder(z, pad_mask)
        z_hat = self.decoder(z_rl, z, pad_mask)
        mse = (z_hat - z).pow(2).mean(dim=-1)
        masked_mse = mse * pad_mask.float()
        num_valid = pad_mask.float().sum()
        loss = masked_mse.sum() / num_valid.clamp(min=1.0)
        return loss, z_rl, z_hat

    @torch.no_grad()
    def encode(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        return self.encoder(z, pad_mask)
