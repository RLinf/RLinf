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


class QValueHead(nn.Module):
    """
    Q-value head for SAC critic networks.
    Processes state and action separately before fusion to handle dimension imbalance.
    
    Architecture:
        - State pathway: projects from hidden_size to 256
        - Action pathway: projects from action_dim to 256
        - Fusion: concatenate [256, 256] -> 512 -> 256 -> 128 -> 1
    """
    def __init__(self, hidden_size, action_dim, output_dim=1, use_separate_processing=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.use_separate_processing = use_separate_processing
        
        if use_separate_processing:
            # Separate processing pathways for state and action
            # State pathway: compress high-dimensional state
            self.state_proj = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU()
            )
            
            # Action pathway: expand low-dimensional action
            self.action_proj = nn.Sequential(
                nn.Linear(action_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU()
            )
            
            # Fusion layers: combine processed state and action
            self.fusion_l1 = nn.Linear(256 + 256, 512)
            self.fusion_act1 = nn.GELU()
            self.fusion_l2 = nn.Linear(512, 256)
            self.fusion_act2 = nn.GELU()
            self.fusion_l3 = nn.Linear(256, 128)
            self.fusion_act3 = nn.GELU()
            self.fusion_l4 = nn.Linear(128, output_dim, bias=False)
        else:
            # Original simple concatenation approach
            self.head_l1 = nn.Linear(hidden_size + action_dim, 512)
            self.head_act1 = nn.GELU()
            self.head_l2 = nn.Linear(512, 256)
            self.head_act2 = nn.GELU()
            self.head_l3 = nn.Linear(256, 128)
            self.head_act3 = nn.GELU()
            self.head_l4 = nn.Linear(128, output_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        if self.use_separate_processing:
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
        if self.use_separate_processing:
            # Process state and action separately
            state_embed = self.state_proj(state_features)  # [B, 256]
            action_embed = self.action_proj(action_features)  # [B, 256]
            
            # Concatenate processed features (now balanced: 256 + 256)
            x = torch.cat([state_embed, action_embed], dim=-1)  # [B, 512]
            
            # Fusion network
            x = self.fusion_act1(self.fusion_l1(x))
            x = self.fusion_act2(self.fusion_l2(x))
            x = self.fusion_act3(self.fusion_l3(x))
            q_values = self.fusion_l4(x)
        else:
            # Original simple concatenation
            x = torch.cat([state_features, action_features], dim=-1)
            x = self.head_act1(self.head_l1(x))
            x = self.head_act2(self.head_l2(x))
            x = self.head_act3(self.head_l3(x))
            q_values = self.head_l4(x)
        
        return q_values


class DoubleQValueHead(nn.Module):
    """
    Double Q-network for SAC to reduce overestimation bias.
    """
    def __init__(self, hidden_size, action_dim, output_dim=1, use_separate_processing=True):
        super().__init__()
        self.q1 = QValueHead(hidden_size, action_dim, output_dim, use_separate_processing)
        self.q2 = QValueHead(hidden_size, action_dim, output_dim, use_separate_processing)
    
    def forward(self, state_features, action_features):
        """
        Forward pass for both Q-networks.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q1 and Q2 values
        """
        q1_values = self.q1(state_features, action_features)
        q2_values = self.q2(state_features, action_features)
        return q1_values, q2_values
    
    def q1_forward(self, state_features, action_features):
        """Forward pass for Q1 network only"""
        return self.q1(state_features, action_features)
    
    def q2_forward(self, state_features, action_features):
        """Forward pass for Q2 network only"""
        return self.q2(state_features, action_features)
