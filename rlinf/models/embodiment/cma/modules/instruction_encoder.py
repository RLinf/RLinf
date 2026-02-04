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

import gzip
import json
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class InstructionEncoder(nn.Module):
    """An encoder that uses RNN to encode an instruction."""

    def __init__(self, config: dict[str, Any]) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
                sensor_uuid: "instruction" or "rxr_instruction"
                vocab_size: vocabulary size (if not using pretrained embeddings)
                use_pretrained_embeddings: whether to use pretrained embeddings
                embedding_file: path to pretrained embeddings file
                fine_tune_embeddings: whether to fine-tune embeddings
                bidirectional: whether to use bidirectional RNN
        """
        super().__init__()

        self.config = config
        self.sensor_uuid = config.get("sensor_uuid", "instruction")
        self.rnn_type = config.get("rnn_type", "LSTM")
        self.bidirectional = config.get("bidirectional", False)
        self.final_state_only = config.get("final_state_only", False)

        rnn = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
            bidirectional=self.bidirectional,
        )

        if self.sensor_uuid == "instruction":
            if config.get("use_pretrained_embeddings", False):
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(config["embedding_file"]),
                    freeze=not config.get("fine_tune_embeddings", False),
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config["vocab_size"],
                    embedding_dim=config["embedding_size"],
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config["hidden_size"] * (1 + int(self.bidirectional))

    def _load_embeddings(self, embedding_file: str) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size] or [batch_size x seq_length x hidden_size]
        """
        if self.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)
        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.rnn_type == "LSTM":
            final_state = final_state[0]

        if self.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)
