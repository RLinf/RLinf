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
from .utils import make_mlp

class QHead(nn.Module):
    """
    Q-value head for SAC critic networks.
    Processes state and action separately before fusion to handle dimension imbalance.
    
    Architecture:
        - State pathway: projects from hidden_size to 256
        - Action pathway: projects from action_dim to 256
        - Fusion: concatenate [256, 256] -> 512 -> 256 -> 128 -> 1
    """
    def __init__(self, hidden_size, action_dim, hidden_dims, output_dim=1, use_mix_embedding_input=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.use_mix_embedding_input = use_mix_embedding_input
        
        if use_mix_embedding_input:
            raise NotImplementedError
        else:
            self.net = make_mlp(
                in_channels=hidden_size+action_dim, 
                mlp_channels=hidden_dims+[output_dim, ], 
                act_builder=nn.ReLU, 
                last_act=False
            )

        # self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        if self.use_mix_embedding_input:
            # Initialize state and action projection layers
            for module in [self.state_proj, self.action_proj]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                        nn.init.zeros_(layer.bias)
            
            # Initialize fusion layers
            for layer in [self.fusion_l1, self.fusion_l2, self.fusion_l3]:
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            
            # Final layer with smaller initialization for stability
            nn.init.uniform_(self.fusion_l4.weight, -3e-3, 3e-3)
        else:
            # Original initialization
            for layer in [self.head_l1, self.head_l2, self.head_l3]:
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            
            nn.init.uniform_(self.head_l4.weight, -3e-3, 3e-3)

    def forward(self, state_features, action_features):
        """
        Forward pass for Q-value computation.
        
        Args:
            state_features (torch.Tensor): State representation [batch_size, hidden_size]
            action_features (torch.Tensor): Action representation [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Q-values [batch_size, output_dim]
        """
        if self.use_mix_embedding_input:
            raise NotImplementedError
        else:
            # Original simple concatenation
            x = torch.cat([state_features, action_features], dim=-1)
            q_values = self.net(x)
        
        return q_values


class MultiQHead(nn.Module):
    """
    Double Q-network for SAC to reduce overestimation bias.
    """
    def __init__(
            self, 
            hidden_size, action_dim, 
            hidden_dims, 
            num_q_heads=2, 
            output_dim=1, 
            use_mix_embedding_input=True, 
        ):
        super().__init__()

        self.num_q_heads = num_q_heads
        qs = []
        for q_id in range(self.num_q_heads):
            qs.append(
                QHead(hidden_size, action_dim, hidden_dims, output_dim, use_mix_embedding_input)
            )
        self.qs = nn.ModuleList(qs)

    def forward(self, state_features, action_features):
        """
        Forward pass for both Q-networks.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q1 and Q2 values
        """
        q_vs = []
        for qf in self.qs: 
            q_vs.append(qf(state_features, action_features))
        return torch.cat(q_vs, dim=-1)
    
    def q_id_forward(self, q_id, state_features, action_features):
        """Forward pass for Q1 network only"""
        return self.qs[q_id](state_features, action_features)
    
