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
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch import Tensor

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.cma.modules import (
    CMABasePolicy,
    InstructionEncoder,
    TorchVisionResNet18,
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    build_rnn_state_encoder,
)
from rlinf.models.embodiment.modules.value_head import ValueHead


@dataclass
class CMAConfig:
    """Configuration for CMA Policy."""

    # Image and action configs
    image_size: list[int] = field(default_factory=lambda: [256, 256, 3])
    action_dim: int = 1
    num_action_classes: int = 4
    state_dim: int = 0  # Not used in CMA, but kept for compatibility
    num_action_chunks: int = 1
    model_path: Optional[str] = None

    # Encoder configs
    instruction_encoder_config: dict[str, Any] = field(default_factory=dict)
    depth_encoder_config: dict[str, Any] = field(default_factory=dict)
    rgb_encoder_config: dict[str, Any] = field(default_factory=dict)
    state_encoder_config: dict[str, Any] = field(default_factory=dict)

    # Model configs
    hidden_size: int = 512
    ablate_instruction: bool = False
    ablate_depth: bool = False
    ablate_rgb: bool = False
    normalize_rgb: bool = False

    # RL head configs
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"
    num_q_heads: int = 2

    # Action distribution configs
    independent_std: bool = True
    action_scale = None
    final_tanh = False
    std_range = None
    logstd_range = None

    # Progress monitor
    use_progress_monitor: bool = False
    progress_monitor_alpha: float = 1.0

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)

        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            self.independent_std = False
            if self.action_scale is None:
                self.action_scale = (-1, 1)
            self.final_tanh = True
            self.std_range = (1e-5, 5)

        # Validate model path if needed
        if self.model_path is not None:
            assert os.path.exists(self.model_path), (
                f"Model path does not exist: {self.model_path}"
            )

    def build_observation_space(self) -> spaces.Space:
        # Default depth sensor config (matching VLN-CE/vlnce_r2r.yaml)
        depth_width = 256
        depth_height = 256
        min_depth = 0.0
        max_depth = 10.0

        # Build observation_space similar to VLN-CE
        observation_space = spaces.Dict(
            {
                "depth": spaces.Box(
                    low=min_depth,
                    high=max_depth,
                    shape=(depth_height, depth_width, 1),
                    dtype=np.float32,
                )
            }
        )

        return observation_space


class CMANet(nn.Module):
    """Cross-Modal Attention Network backbone.

    An implementation of the cross-modal attention (CMA) network from
    https://arxiv.org/abs/2004.02857
    """

    def __init__(
        self, cfg: CMAConfig, observation_space: Optional[spaces.Space] = None
    ):
        super().__init__()
        self.cfg = cfg

        # Init the instruction encoder
        instruction_cfg = cfg.instruction_encoder_config.copy()
        instruction_cfg["final_state_only"] = False  # CMA needs full sequence
        self.instruction_encoder = InstructionEncoder(instruction_cfg)

        # Build observation_space from config if not provided (like VLN-CE)
        if observation_space is None:
            observation_space = cfg.build_observation_space()
            print("Built observation_space from CMAConfig defaults for depth encoder")

        # Init the depth encoder
        depth_cfg = cfg.depth_encoder_config.copy()
        depth_cnn_type = depth_cfg.pop("cnn_type", "VlnResnetDepthEncoder")
        assert depth_cnn_type == "VlnResnetDepthEncoder"

        # Save output_size before passing to encoder
        depth_output_size = depth_cfg.pop("output_size", 128)
        depth_checkpoint = depth_cfg.pop("ddppo_checkpoint", "NONE")
        depth_backbone = depth_cfg.pop("backbone", "resnet50")
        depth_trainable = depth_cfg.pop("trainable", False)
        depth_width = depth_cfg.pop("depth_width", 256)
        depth_height = depth_cfg.pop("depth_height", 256)
        min_depth = depth_cfg.pop("min_depth", 0.0)
        max_depth = depth_cfg.pop("max_depth", 10.0)

        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space=observation_space,
            output_size=depth_output_size,
            checkpoint=depth_checkpoint,
            backbone=depth_backbone,
            trainable=depth_trainable,
            spatial_output=True,
            depth_width=depth_width,
            depth_height=depth_height,
            min_depth=min_depth,
            max_depth=max_depth,
            **depth_cfg,
        )

        # Init the RGB visual encoder
        rgb_cfg = cfg.rgb_encoder_config.copy()
        rgb_cnn_type = rgb_cfg.pop("cnn_type", "TorchVisionResNet50")
        assert rgb_cnn_type in ["TorchVisionResNet18", "TorchVisionResNet50"]

        # Save output_size before passing to encoder
        rgb_output_size = rgb_cfg.pop("output_size", 256)
        rgb_trainable = rgb_cfg.pop("trainable", False)

        rgb_encoder_class = (
            TorchVisionResNet50
            if rgb_cnn_type == "TorchVisionResNet50"
            else TorchVisionResNet18
        )
        self.rgb_encoder = rgb_encoder_class(
            output_size=rgb_output_size,
            normalize_visual_inputs=cfg.normalize_rgb,
            trainable=rgb_trainable,
            spatial_output=True,
        )

        # Action embedding (for prev_action)
        self.prev_action_embedding = nn.Embedding(cfg.num_action_classes + 1, 32)

        hidden_size = cfg.hidden_size
        self._hidden_size = hidden_size

        # RGB and depth linear projections
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.rgb_encoder.output_shape[0], rgb_output_size),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.depth_encoder.output_shape), depth_output_size),
            nn.ReLU(True),
        )

        # Init the RNN state encoder
        state_cfg = cfg.state_encoder_config.copy()
        rnn_input_size = depth_output_size
        rnn_input_size += rgb_output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        state_rnn_type = state_cfg.pop("rnn_type", "GRU")
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=state_rnn_type,
            num_layers=1,
        )

        self._output_size = (
            hidden_size
            + rgb_output_size
            + depth_output_size
            + self.instruction_encoder.output_size
        )

        # Attention layers
        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + rgb_output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + depth_output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(self.instruction_encoder.output_size, hidden_size // 2)

        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        # Second state encoder
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=state_rnn_type,  # Use the same RNN type as first state encoder
            num_layers=1,
        )
        self._output_size = hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)
        self._init_progress_monitor()

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    def _init_progress_monitor(self) -> None:
        if hasattr(self, "progress_monitor"):
            nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Cross-modal attention mechanism."""
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self,
        observations: dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through CMA network.

        Args:
            observations: dict with keys:
                - "instruction": [batch_size, seq_length] or [batch_size, seq_length, hidden_size]
                - "rgb": [batch_size, H, W, 3] or [batch_size, 3, H, W]
                - "depth": [batch_size, H, W, 1] or [batch_size, 1, H, W]
            rnn_states: [batch_size, num_layers, hidden_size]
            prev_actions: [batch_size] - previous action indices
            masks: [batch_size] - 1 for valid, 0 for invalid
        Returns:
            (torch.Tensor, torch.Tensor): A tuple containing:
                - x: [batch_size, hidden_size]
                - rnn_states_out: [batch_size, num_layers, hidden_size]
        """
        # Encode instruction
        instruction_embedding = self.instruction_encoder(observations)

        # Encode depth
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        # Encode RGB
        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        # Encode previous action
        prev_actions_emb = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        # Ablation (for debugging)
        if self.cfg.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.cfg.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.cfg.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        # Project RGB and depth
        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        # First RNN state encoder
        state_in = torch.cat([rgb_in, depth_in, prev_actions_emb], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        # Cross-modal attention: state -> instruction
        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        # Cross-modal attention: instruction -> RGB and depth
        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        # Second RNN state encoder
        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions_emb,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )

        # TODO: add progress monitor when develop training

        return x, rnn_states_out


class CMAPolicy(nn.Module, BasePolicy):
    """CMA Policy for embodied navigation."""

    def __init__(self, cfg: CMAConfig, observation_space: spaces.Space = None):
        super(CMAPolicy, self).__init__()
        self.cfg = cfg

        self.rnn_states = None
        self.prev_actions = None
        self.prev_episode_id = None
        self.action_map = {
            0: "stop",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
        }
        assert len(self.action_map) == self.cfg.num_action_classes, (
            "CMA action_map size must match num_action_classes"
        )

        self.policy = CMABasePolicy(
            net=CMANet(cfg, observation_space), dim_actions=cfg.num_action_classes
        )

        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=self.policy.net.output_size,
                hidden_sizes=(256, 256, 256),
                activation="relu",
            )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        policy_state_keys = set(self.policy.state_dict().keys())
        value_head_state_keys = set()
        if hasattr(self, "value_head"):
            value_head_state_keys = set(self.value_head.state_dict().keys())

        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("policy.") or key.startswith("value_head."):
                normalized_state_dict[key] = value
            elif key in policy_state_keys:
                normalized_state_dict[f"policy.{key}"] = value
            elif key in value_head_state_keys:
                normalized_state_dict[f"value_head.{key}"] = value
            else:
                normalized_state_dict[key] = value

        return super().load_state_dict(
            normalized_state_dict, strict=False, assign=assign
        )

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        """Preprocess environment observations.
        Input env_obs:
        - "main_images": [B, C, H, W] tensor (rgb)
        - "extra_view_images": [B, 1, C, H, W] tensor (depth)
        - "wrist_images": [B] tensor (instruction tokens)
        - "task_descriptions": [B] tensor (instruction text)
        - "states": [B] tensor (episode_ids)
        Output processed_env_obs:
        - "rgb": [B, C, H, W] tensor
        - "depth": [B, 1, 1, H, W] tensor
        - "instruction": [B] tensor
        - "states": [B] tensor
        We need to handle both the wrapped obs (from HabitatEnv._wrap_obs)
        and raw obs (from HabitatRLEnv) which may contain instruction.
        """
        device = next(self.parameters()).device
        processed_env_obs = {}
        # Process main images to rgb
        assert "main_images" in env_obs and env_obs["main_images"] is not None
        processed_env_obs["rgb"] = env_obs["main_images"].clone().to(device)
        # Process depth images
        assert (
            "extra_view_images" in env_obs and env_obs["extra_view_images"] is not None
        )
        processed_env_obs["depth"] = (
            env_obs["extra_view_images"][:, 0][:, :, :, 0]
            .unsqueeze(3)
            .clone()
            .to(device)
            .float()
        )
        # Process instruction
        assert "wrist_images" in env_obs and env_obs["wrist_images"] is not None
        processed_env_obs["instruction"] = torch.tensor(
            env_obs["wrist_images"], device=device
        )
        # Process states (if needed)
        if "states" in env_obs and env_obs["states"] is not None:
            processed_env_obs["states"] = env_obs["states"].clone().to(device)

        return processed_env_obs

    def predict_action_batch(
        self,
        env_obs,
        mode="train",
        **kwargs,
    ):
        """Predict actions for a batch of observations."""
        env_obs = self.preprocess_env_obs(env_obs=env_obs)
        batch_size = env_obs["rgb"].shape[0]
        device = env_obs["rgb"].device
        current_episode_ids = env_obs["states"]

        is_first_step = self.prev_episode_id is None
        if is_first_step:
            self.prev_episode_id = current_episode_ids.clone()
            self.rnn_states = torch.zeros(
                batch_size,
                self.policy.net.state_encoder.num_recurrent_layers
                + self.policy.net.second_state_encoder.num_recurrent_layers,
                self.cfg.hidden_size,
                device=device,
            )
            self.prev_actions = torch.zeros(
                batch_size, 1, device=device, dtype=torch.long
            )
            step_masks = torch.zeros_like(self.prev_actions, dtype=torch.uint8)
        else:
            assert (
                self.prev_actions is not None
                and self.rnn_states is not None
                and self.prev_episode_id is not None
            ), "CMA recurrent state must be initialized."
            step_masks = torch.ones_like(self.prev_actions, dtype=torch.uint8)
            reset_mask = current_episode_ids != self.prev_episode_id
            if reset_mask.any():
                self.rnn_states[reset_mask] = 0
                self.prev_actions[reset_mask] = 0
                step_masks[reset_mask] = 0
                self.prev_episode_id[reset_mask] = current_episode_ids[reset_mask]

        pre_rnn_states = self.rnn_states.clone()
        prev_actions = self.prev_actions.clone()
        step_masks = step_masks.clone()

        features, rnn_states = self.policy.net(
            env_obs, pre_rnn_states, prev_actions, step_masks
        )
        distribution = self.policy.action_distribution(features)
        if mode == "train":
            action = distribution.sample()
        elif mode == "eval":
            action = distribution.mode()
        else:
            raise NotImplementedError(f"{mode=}")
        prev_logprobs = distribution.log_probs(action)
        self.rnn_states, self.prev_actions = rnn_states, action

        chunk_actions = []
        for i in range(batch_size):
            chunk_actions.append(
                [self.action_map[a.item()] for a in action[i].cpu().numpy()]
            )
        chunk_actions = np.array(chunk_actions)

        if hasattr(self, "value_head"):
            chunk_values = self.value_head(features)
        else:
            chunk_values = torch.zeros_like(prev_logprobs[..., :1])

        forward_inputs = {
            "action": action.clone(),
            "pre_rnn_states": pre_rnn_states,
            "prev_actions": prev_actions,
            "masks": step_masks,
            "env_obs_rgb": env_obs["rgb"].clone(),
            "env_obs_depth": env_obs["depth"].clone(),
            "env_obs_instruction": env_obs["instruction"].clone(),
            "env_obs_states": env_obs["states"].clone(),
        }

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def forward(self, forward_type="default_forward", **kwargs):
        """Forward pass dispatcher."""
        if forward_type == "sac_forward":
            return self.sac_forward(**kwargs)
        elif forward_type == "sac_q_forward":
            return self.sac_q_forward(**kwargs)
        elif forward_type == "crossq_forward":
            return self.crossq_forward(**kwargs)
        elif forward_type == "crossq_q_forward":
            return self.crossq_q_forward(**kwargs)
        elif forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        obs = {
            "rgb": forward_inputs["env_obs_rgb"],
            "depth": forward_inputs["env_obs_depth"],
            "instruction": forward_inputs["env_obs_instruction"],
            "states": forward_inputs["env_obs_states"],
        }
        action = forward_inputs["action"].long()
        if action.ndim == 1:
            action = action.unsqueeze(-1)

        pre_rnn_states = forward_inputs["pre_rnn_states"].clone()
        prev_actions = forward_inputs["prev_actions"].long().clone()
        masks = forward_inputs["masks"].clone()

        features, _ = self.policy.net(obs, pre_rnn_states, prev_actions, masks)
        distribution = self.policy.action_distribution(features)

        output_dict = {}
        if compute_logprobs:
            logprobs = distribution.log_probs(action)
            output_dict.update(logprobs=logprobs)
        if compute_entropy:
            entropy = distribution.entropy()
            if entropy.ndim == 1:
                entropy = entropy.unsqueeze(-1)
            output_dict.update(entropy=entropy)
        if compute_values:
            if hasattr(self, "value_head"):
                values = self.value_head(features)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def sac_forward(self, obs, **kwargs):
        raise NotImplementedError

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        raise NotImplementedError

    def crossq_forward(self, obs, **kwargs):
        raise NotImplementedError

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        raise NotImplementedError
