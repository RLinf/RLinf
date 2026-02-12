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

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.flow_actor import FlowTActor, JaxFlowTActor
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder
from rlinf.models.embodiment.modules.utils import init_mlp_weights, layer_init, make_mlp
from rlinf.models.embodiment.modules.value_head import ValueHead


@dataclass
class FlowConfig:
    image_size: list[int] = field(default_factory=list)
    image_num: int = 1
    action_dim: int = 4
    state_dim: int = 29
    num_action_chunks: int = 1
    backbone: str = "resnet"
    model_path: Optional[str] = None  # used as dir actually!
    encoder_config: dict[str, Any] = field(
        default_factory=dict
    )  # 'extra_config' rename to 'encoder_config'
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"  # same as cnn_policy.py

    state_latent_dim: int = 64
    action_scale = None
    final_tanh = True
    std_range = None  # same as cnn_policy.py
    logstd_range = None  # same as cnn_policy.py

    num_q_heads: int = 2  # same as cnn_policy.py

    # -- Flow Matching specific parameters --##
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "JaxFlowTActor"  # "FlowTActor" or "JaxFlowTActor"
    # Whether to use a separate head to predict noise_std
    noise_std_head: bool = False
    # Min/Max log std for training (if using noise_std_head)
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    # Min/Max log std for rollout (if using noise_std_head)
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    # Fixed noise std for training (if not using noise_std_head)
    noise_std_train: float = 0.3
    # Fixed noise std for rollout (if not using noise_std_head)
    noise_std_rollout: float = 0.02

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True
            if self.backbone == "resnet":
                self.std_range = (1e-5, 5)

        assert self.model_path is not None, "Please specify the model_path."
        assert "ckpt_name" in self.encoder_config, (
            "Please specify the ckpt_name in encoder_config to load pretrained encoder weights."
        )
        ckpt_path = os.path.join(self.model_path, self.encoder_config["ckpt_name"])
        assert os.path.exists(ckpt_path), (
            f"Pretrained encoder weights not found at {ckpt_path} with model path {self.model_path} and encoder ckpt name {self.encoder_config['ckpt_name']}"
        )
        self.encoder_config["ckpt_path"] = ckpt_path


class FlowPolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.image_size[0]

        # Step1: Init Image encoders (same as CNNPolicy)
        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        if self.cfg.backbone == "resnet":
            sample_x = torch.randn(1, *self.cfg.image_size)
            for img_id in range(self.cfg.image_num):
                self.encoders.append(
                    ResNetEncoder(
                        sample_x, out_dim=256, encoder_cfg=self.cfg.encoder_config
                    )
                )
                encoder_out_dim += self.encoders[img_id].out_dim
        else:
            raise NotImplementedError

        if self.cfg.backbone == "resnet":
            self.state_proj = nn.Sequential(
                *make_mlp(
                    in_channels=self.cfg.state_dim,
                    mlp_channels=[
                        self.cfg.state_latent_dim,
                    ],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.state_proj, nonlinearity="tanh")
            self.mix_proj = nn.Sequential(
                *make_mlp(
                    in_channels=encoder_out_dim + self.cfg.state_latent_dim,
                    mlp_channels=[256, 256],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.mix_proj, nonlinearity="tanh")

        # --- Step2: Create flow actor --- #
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256

        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)

        if self.cfg.flow_actor_type == "FlowTActor":
            self.flow_actor = FlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
            )
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_head=self.cfg.noise_std_head,
                log_std_min_train=self.cfg.log_std_min_train,
                log_std_max_train=self.cfg.log_std_max_train,
                log_std_min_rollout=self.cfg.log_std_min_rollout,
                log_std_max_rollout=self.cfg.log_std_max_rollout,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # --- Step3: Create Q-head for SAC --- #
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            if self.cfg.backbone == "resnet":  # Now only "resnet" backbone is supported
                hidden_size = encoder_out_dim + self.cfg.state_latent_dim
                hidden_dims = [256, 256, 256]
            if self.cfg.q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=hidden_size,
                    hidden_dims=hidden_dims,
                    num_q_heads=self.cfg.num_q_heads,  # pass from actor.model.num_q_heads
                    action_feature_dim=self.cfg.action_dim,
                )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed_env_obs = {}
        processed_env_obs["states"] = env_obs["states"].clone().to(device)
        processed_env_obs["main_images"] = (
            env_obs["main_images"].clone().to(device).float() / 255.0
        )
        if env_obs.get("extra_view_images", None) is not None:
            processed_env_obs["extra_view_images"] = (
                env_obs["extra_view_images"].clone().to(device).float() / 255.0
            )
        return processed_env_obs

    def get_feature(self, obs):
        """Extract features from observations (images + states)"""
        visual_features = []
        # from image_keys to image_num
        for img_id in range(self.cfg.image_num):
            if img_id == 0:
                images = obs["main_images"]
            else:
                images = obs["extra_view_images"][:, img_id - 1]
            if images.shape[3] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
            visual_features.append(self.encoders[img_id](images))
        visual_feature = torch.cat(visual_features, dim=-1)

        state_feature = self.state_proj(obs["states"])
        full_feature = torch.cat([visual_feature, state_feature], dim=-1)

        return full_feature, visual_feature

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)

        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(mix_feature, train=True, log_grad=False)

        return action, log_prob, full_feature

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    # use get_q_values() as sac_q_forward()
    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    def default_forward(
        self,
        forward_inputs,
        compute_entropy=False,
        compute_values=False,
        **kwargs,
    ):
        """Default forward pass"""

        obs = {
            "main_images": forward_inputs["main_images"],
            "states": forward_inputs["states"],
        }
        if "extra_view_images" in forward_inputs:
            obs["extra_view_images"] = forward_inputs["extra_view_images"]
        obs = self.preprocess_env_obs(obs)

        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)

        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)

        output_dict = {
            "action": action,
            "log_prob": log_prob,  # key 'log_prob' or 'logprobs' as used in both cnn_policy.py??
        }

        if compute_entropy:
            # For flow matching, entropy is computed from log_prob
            # Approximate entropy as negative log_prob (this is a simplification)
            entropy = -log_prob
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(mix_feature)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        return_shared_feature=False,
        **kwargs,
    ):
        """Predict actions in batch"""
        env_obs = self.preprocess_env_obs(env_obs)

        full_feature, visual_feature = self.get_feature(env_obs)
        mix_feature = self.mix_proj(full_feature)

        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)

        # chunk_actions is always numpy array
        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )
        chunk_actions = chunk_actions.cpu().numpy()

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            # x1. image indexing logic changed
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["states"] = env_obs["states"]
            if "extra_view_images" in env_obs:
                forward_inputs["extra_view_images"] = env_obs["extra_view_images"]

        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result


@dataclass
class FlowStateConfig:
    action_dim: int = 4
    obs_dim: int = 29
    num_action_chunks: int = 1
    encoder_config: dict[str, Any] = field(default_factory=dict)
    add_value_head: bool = False  # No visual_feature -> No mix_feature -> No value_head -> add_value_head must be false !
    add_q_head: bool = False
    q_head_type: str = "default"
    num_q_heads: int = 2

    action_scale = None
    final_tanh = True

    # Flow Matching specific parameters
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "JaxFlowTActor"  # "FlowTActor" or "JaxFlowTActor"
    # Whether to use a separate head to predict noise_std
    noise_std_head: bool = False
    # Min/Max log std for training (if using noise_std_head)
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    # Min log std for rollout (if using noise_std_head)
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    # Fixed noise std for training (if not using noise_std_head)
    noise_std_train: float = 0.3
    # Fixed noise std for rollout (if not using noise_std_head)
    noise_std_rollout: float = 0.02

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True


class FlowStatePolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: FlowStateConfig):
        super().__init__()
        self.cfg = cfg

        # 3 layer MLP encoder for obs
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.cfg.obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        # Create flow actor
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256

        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)

        if self.cfg.flow_actor_type == "FlowTActor":
            self.flow_actor = FlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
            )
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_head=self.cfg.noise_std_head,
                log_std_min_train=self.cfg.log_std_min_train,
                log_std_max_train=self.cfg.log_std_max_train,
                log_std_min_rollout=self.cfg.log_std_min_rollout,
                log_std_max_rollout=self.cfg.log_std_max_rollout,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # Q-head for SAC
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.cfg.obs_dim,
                hidden_dims=[256, 256, 256],
                num_q_heads=self.cfg.num_q_heads,
                action_feature_dim=self.cfg.action_dim,
            )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None

    # added num_action_chunks property
    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        return {"states": env_obs["states"].to(device)}

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        feat = self.backbone(obs["states"])

        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(feat, train=True, log_grad=False)

        return action, log_prob, None

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        return self.q_head(obs["states"], actions)

    # use get_q_values() as sac_q_forward()
    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        return self.q_head(obs["states"], actions)

    # 10. add unified forward()
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)  # originally exists
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)  # use get_q_values()
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)  # NOT USED (NO get_feature)
        else:
            raise NotImplementedError

    def default_forward(
        self, obs, compute_entropy=False, compute_values=False, **kwargs
    ):
        """
        Default forward pass for FlowStatePolicy.

        This method is not supported for FlowStatePolicy because it relies on features
        (e.g., get_feature, mix_proj) that are not defined for this class.
        It should not be used; kept only for compatibility.
        """
        raise NotImplementedError(
            "FlowStatePolicy.default_forward is not supported. "
            "Use FlowStatePolicy.forward with the appropriate forward_type instead."
        )

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,  # NOT USED, unlike FlowPolicy
        return_obs=True,
        return_shared_feature=False,  # NOT USED, unlike FlowPolicy
        **kwargs,
    ):
        """
        Predict actions in batch.
        Called by MultiStepRolloutWorker for rollout
        """
        env_obs = self.preprocess_env_obs(env_obs)

        feat = self.backbone(env_obs["states"])  # encode obs using the 3 layer MLP

        # Use flow actor
        action, log_prob = self.flow_actor(feat, train=False, log_grad=False)

        # chunk_actions is always numpy array
        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )
        chunk_actions = chunk_actions.cpu().numpy()

        chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            forward_inputs["states"] = env_obs[
                "states"
            ]  # add 'states' to forward_inputs instead of 'obs/{key}'

        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
