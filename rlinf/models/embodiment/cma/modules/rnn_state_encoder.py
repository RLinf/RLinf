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
import torch.nn as nn


class RNNStateEncoder(nn.Module):
    """RNN State Encoder for processing sequential state information."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    @property
    def num_recurrent_layers(self) -> int:
        return self.num_layers

    def forward(
        self, x: torch.Tensor, rnn_states: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, input_size]
            rnn_states: [batch_size, num_layers, hidden_size] for GRU
                       [batch_size, num_layers, hidden_size] for LSTM (only h)
            masks: [batch_size] - 1 for valid, 0 for invalid
        Returns:
            (torch.Tensor, torch.Tensor): A tuple containing:
                - output: [batch_size, hidden_size]
                - new_rnn_states: [batch_size, num_layers, hidden_size]
        """
        # Reshape for RNN: [batch_size, 1, input_size]
        x = x.unsqueeze(1)

        # Reshape rnn_states: [num_layers, batch_size, hidden_size]
        rnn_states = rnn_states.permute(1, 0, 2)

        # Apply masks to reset hidden states
        masks = masks.view(1, -1, 1).float()
        rnn_states = rnn_states * masks

        if self.rnn_type == "LSTM":
            # For LSTM, we need both h and c
            # Assume c is initialized as zeros if not provided
            # In practice, we might need to track c separately, but for now use zeros
            c = torch.zeros_like(rnn_states)
            rnn_states_tuple = (rnn_states, c)
        else:
            rnn_states_tuple = rnn_states

        output, new_rnn_states = self.rnn(x, rnn_states_tuple)

        if self.rnn_type == "LSTM":
            new_rnn_states = new_rnn_states[0]  # Only return h

        # Reshape back: [batch_size, num_layers, hidden_size]
        new_rnn_states = new_rnn_states.permute(1, 0, 2)

        # Extract output: [batch_size, hidden_size]
        output = output.squeeze(1)

        return output, new_rnn_states


def build_rnn_state_encoder(
    input_size: int,
    hidden_size: int,
    rnn_type: str = "GRU",
    num_layers: int = 1,
) -> RNNStateEncoder:
    """Build an RNN state encoder."""
    return RNNStateEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
    )
