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
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    # ===== DSRL-specific parameters =====
    use_dsrl: bool = False  # Enable DSRL algorithm
    dsrl_state_dim: int = 8  # Raw state dimension for DSRL encoders
    dsrl_action_noise_dim: int = 32  # Noise dimension output by GaussianPolicy
    dsrl_num_q_heads: int = 10  # Number of Q-networks
    dsrl_agg_q: str = "mean"  # Q aggregation method: 'mean' | 'min'
    dsrl_image_latent_dim: int = 64  # Latent dim for lightweight image encoder
    dsrl_state_latent_dim: int = 64  # Hidden dim for state encoder
    dsrl_hidden_dims: tuple = field(
        default_factory=lambda: (128, 128, 128)
    )  # Hidden dims for Q-head and GaussianPolicy


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.logger = get_logger()
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name in ["pi05_maniskill", "pi05_libero"]:
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )

        # ===== DSRL components initialization =====
        if self.config.use_dsrl:
            from rlinf.models.embodiment.modules.compact_encoders import (
                CompactMultiQHead,
                CompactStateEncoder,
                LightweightImageEncoder64,
            )
            from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

            # Use explicit bfloat16 to match the backbone dtype that will be
            # loaded from the checkpoint later.  At __init__ time the backbone
            # parameters are still float32 (weights are loaded afterwards by
            # safetensors.torch.load_model), so next(self.parameters()).dtype
            # would incorrectly return float32.  Hardcoding bfloat16 here
            # ensures all parameters share a single dtype when FSDP creates
            # its FlatParameter, avoiding the writeback shape-mismatch error.
            _dsrl_dtype = torch.bfloat16

            dsrl_input_dim = (
                self.config.dsrl_state_latent_dim + self.config.dsrl_image_latent_dim
            )  # e.g. 64 + 64 = 128

            self.dsrl_action_noise_net = GaussianPolicy(
                input_dim=dsrl_input_dim,
                output_dim=self.config.dsrl_action_noise_dim,
                hidden_dims=self.config.dsrl_hidden_dims,
                low=None,
                high=None,
                action_horizon=self.config.action_horizon,
            ).to(dtype=_dsrl_dtype)

            self.actor_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=self.config.dsrl_image_latent_dim,
                image_size=64,
            ).to(dtype=_dsrl_dtype)
            self.actor_state_encoder = CompactStateEncoder(
                state_dim=self.config.dsrl_state_dim,
                hidden_dim=self.config.dsrl_state_latent_dim,
            ).to(dtype=_dsrl_dtype)
            self.critic_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=self.config.dsrl_image_latent_dim,
                image_size=64,
            ).to(dtype=_dsrl_dtype)
            self.critic_state_encoder = CompactStateEncoder(
                state_dim=self.config.dsrl_state_dim,
                hidden_dim=self.config.dsrl_state_latent_dim,
            ).to(dtype=_dsrl_dtype)
            self.q_head = CompactMultiQHead(
                state_dim=self.config.dsrl_state_latent_dim,
                image_dim=self.config.dsrl_image_latent_dim,
                action_dim=self.config.dsrl_action_noise_dim,
                hidden_dims=self.config.dsrl_hidden_dims,
                num_q_heads=self.config.dsrl_num_q_heads,
                output_dim=1,
            ).to(dtype=_dsrl_dtype)

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        # input transform
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)

        is_dsrl_train = self.config.use_dsrl and mode == "train"
        if is_dsrl_train:
            # DSRL mode and train mode

            # Step 1: SAC agent outputs noise
            # No data augmentation during rollout, train=False

            # Convert env_obs to the format expected by sac_forward
            # env_obs has: main_images, wrist_images, states
            # sac_forward expects: images (list), states
            dsrl_obs = {"images": [env_obs["main_images"]], "states": env_obs["states"]}

            noise_actions, noise_logprob, _ = self.sac_forward(
                dsrl_obs, train=False, mode=mode
            )

            # Step 2: Use noise to sample actual actions from diffusion model
            outputs = self.sample_actions(
                observation,
                noise=noise_actions,
                mode="eval",
                compute_values=compute_values,
            )

            # Step 3: Extract actual actions for environment interaction
            real_actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"].numpy()

            # Return actual actions to environment, but forward_inputs stores noise.
            actions = real_actions
            prev_logprobs = noise_logprob  # SAC noise logprob
            prev_values = outputs.get("prev_values")
            forward_action = noise_actions  # Used for SAC training

        else:
            # Non-DSRL or eval mode
            outputs = self.sample_actions(
                observation, mode=mode, compute_values=compute_values
            )
            actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"].numpy()
            prev_logprobs = outputs["prev_logprobs"]
            prev_values = outputs["prev_values"]
            forward_action = None

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if forward_action is not None:
            forward_inputs["action"] = forward_action

        # Clone observations to avoid cross-step reference issues.
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in to_process_obs.items() if k != "prompt"}
        )
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            # DSRL: SAC provides noise, convert dtype to match action_in_proj
            noise = noise.to(self.action_in_proj.weight.dtype)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(suffix_out)  # [bs,n_action_steps,max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if not self.use_vlm_value:
                chains_values.append(value_t)
        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            # Base freeze: paligemma (SigLIP vision encoder + Gemma)
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

            # ========== DSRL additional freezing ==========
            if self.config.use_dsrl:
                self.logger.info(
                    "[FREEZE_VLM] DSRL mode: freezing gemma_expert parameters"
                )
                self.paligemma_with_expert.gemma_expert.eval()
                for params in self.paligemma_with_expert.gemma_expert.parameters():
                    params.requires_grad = False

                # Freeze projection layers (used in rollout/eval but not optimized).
                # These are from PI0Pytorch parent class: action_in_proj, action_out_proj, state_proj, action_time_mlp
                self.logger.info(
                    "[FREEZE_VLM] DSRL mode: freezing projection layers (used in rollout/eval but not optimized)"
                )
                projection_names = [
                    "action_in_proj",
                    "action_out_proj",
                    "state_proj",
                    "action_time_mlp",
                ]
                frozen_count = 0
                for name, param in self.named_parameters():
                    if any(proj_name in name for proj_name in projection_names):
                        param.requires_grad = False
                        frozen_count += 1
                        if frozen_count <= 10:  # Print first 10 for brevity
                            self.logger.info(f"  Froze: {name}")
                if frozen_count > 10:
                    self.logger.info(
                        f"  ... and {frozen_count - 10} more projection layer parameters"
                    )

                # Freeze reinflow_explore_noise_net (only used in reinflow diffuser sampling)
                if hasattr(self, "reinflow_explore_noise_net"):
                    self.logger.info(
                        "[FREEZE_VLM] DSRL mode: freezing reinflow_explore_noise_net (used in non-DSRL rollout but not optimized)"
                    )
                    self.reinflow_explore_noise_net.eval()
                    noise_net_params = 0
                    for params in self.reinflow_explore_noise_net.parameters():
                        params.requires_grad = False
                        noise_net_params += params.numel()
                    self.logger.info(
                        f"  Froze {noise_net_params:,} parameters in reinflow_explore_noise_net"
                    )

    # ===== DSRL-specific methods =====

    def sac_forward(
        self, obs=None, data=None, train=False, return_dist_params=False, **kwargs
    ):
        """SAC forward pass for DSRL.

        Args:
            obs: Observation dict (preferred, matches sac_dsrl).
                 Supports two formats:
                   1. {"images": list of tensors, "states": tensor} - internal format
                   2. {"main_images": tensor, "wrist_images": tensor, "states": tensor} - env format
            data: Dictionary containing observations (legacy, for backward compatibility).
            train: Whether to use data augmentation.
            return_dist_params: Whether to return distribution parameters for logging.

        Returns:
            actions: [B, action_horizon, output_dim] - noise or actual actions
            logprobs: [B] - log probabilities
            dist_params: (mean, std) or None - distribution parameters for logging
        """
        if not self.config.use_dsrl:
            raise ValueError("sac_forward called but use_dsrl=False")

        # Support both call styles: obs (new, from sac_dsrl) or data (legacy)
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})

        # Handle two obs formats:
        # Format 1 (internal): {"images": [...], "states": ...}
        # Format 2 (env): {"main_images": ..., "wrist_images": ..., "states": ...}
        if "images" not in obs:
            # Convert env format to internal format
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(
                    f"Invalid obs format: {obs.keys()}. Expected 'images' or 'main_images' key."
                )

        # Preprocess images: resize to 64x64, use only agentview camera
        # Returns [B, 1, C, 64, 64] in [-1, 1] range (float32)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])

        # Move to the same device as actor encoders, convert to bfloat16
        device = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)

        # Extract features (using actor's independent encoder)
        image_features = self.actor_image_encoder(images)  # [B, 64]
        state_features = self.actor_state_encoder(states)  # [B, 64]
        features = torch.cat([state_features, image_features], dim=-1)  # [B, 128]

        # Sample from GaussianPolicy
        mode = kwargs.get("mode", "train")
        deterministic = mode == "eval"

        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )

        # Optional: return distribution parameters for logging
        dist_params = None
        if return_dist_params:
            dist = self.dsrl_action_noise_net.forward(features)
            dist_params = (dist.mean, dist.stddev)

        return action_noise, logprobs, dist_params

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        """Q-value forward pass for DSRL.

        Args:
            obs: Observation dict (preferred, matches sac_dsrl).
                 Supports two formats:
                   1. {"images": list of tensors, "states": tensor} - internal format
                   2. {"main_images": tensor, "wrist_images": tensor, "states": tensor} - env format
            data: Dictionary containing observations (legacy, for backward compatibility).
            actions: [B, action_dim] or [B, action_horizon, action_dim]
            detach_encoder: Whether to detach encoder gradients.
            train: Whether to use data augmentation.

        Returns:
            q_values: [B, num_q_heads] - Q-values from all Q-networks.
        """
        if not self.config.use_dsrl:
            raise ValueError("sac_q_forward called but use_dsrl=False")

        # Support both call styles: obs (new, from sac_dsrl) or data (legacy)
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")

        # Handle two obs formats:
        # Format 1 (internal): {"images": [...], "states": ...}
        # Format 2 (env): {"main_images": ..., "wrist_images": ..., "states": ...}
        if "images" not in obs:
            # Convert env format to internal format
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(
                    f"Invalid obs format: {obs.keys()}. Expected 'images' or 'main_images' key."
                )

        # Preprocess images: resize to 64x64, use only agentview camera
        # Returns [B, 1, C, 64, 64] in [-1, 1] range (float32)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])

        # Move to the same device as critic encoders, convert to bfloat16
        device = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        actions = actions.to(device=device, dtype=torch.bfloat16)

        # Extract features (using critic's independent encoder)
        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)

        # Optionally detach encoder
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()

        # Process actions (DSRL: should be noise, already flattened)
        if actions.dim() == 3:
            actions = actions[:, 0, :]  # [B, action_horizon, dim] -> [B, dim]

        # Compute Q values
        q_values = self.q_head(state_features, image_features, actions)

        return q_values

    def _preprocess_dsrl_images(self, images, train=False):
        """Preprocess images for DSRL: resize to 64x64, use only agentview camera.

        Args:
            images: List of tensors.
                Can be [B, H, W, C] (NHWC) from environment or
                [B, C, H, W] (NCHW) from processed data.
                For Libero: images[0] is agentview, images[1] is wrist.
            train: Whether to use data augmentation (placeholder for now).

        Returns:
            Tensor of shape [B, 1, C, 64, 64] - only agentview, resized, in [-1, 1].
        """
        import torch.nn.functional as F

        # Extract only agentview camera (first image in the list)
        if isinstance(images, list):
            agentview_img = images[0]
        else:
            # Assume it's already a tensor
            agentview_img = images

        # Detect and convert NHWC -> NCHW (environment outputs NHWC)
        if agentview_img.shape[-1] == 3:
            # NHWC format: [B, H, W, C] -> [B, C, H, W]
            agentview_img = agentview_img.permute(0, 3, 1, 2)

        B, C, H, W = agentview_img.shape
        target_size = 64

        # ===== UNIFIED VALUE RANGE HANDLING =====
        # Convert to float32 and normalize to [0, 1] for PyTorch resize
        if agentview_img.dtype == torch.uint8:
            # [0, 255] -> [0, 1]
            agentview_img = agentview_img.float() / 255.0
        else:
            # Check if in [-1, 1] range
            if agentview_img.min() < 0:
                # [-1, 1] -> [0, 1]
                agentview_img = (agentview_img + 1.0) / 2.0
            # else: already in [0, 1] range, assume correctly normalized
        # ===========================================

        # Clamp to ensure valid range
        agentview_img = agentview_img.clamp(0.0, 1.0)

        # ===== GPU-ACCELERATED RESIZE (aligned with PIL behavior) =====
        # PyTorch bilinear with align_corners=False approximates PIL's behavior
        resized_img = F.interpolate(
            agentview_img,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        # =============================================================

        # Convert back to [-1, 1] range (to match PIL-based pipeline)
        resized_img = resized_img * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Add num_images dimension: [B, C, 64, 64] -> [B, 1, C, 64, 64]
        resized_img = resized_img.unsqueeze(1)

        return resized_img

    def _preprocess_states(self, states):
        """
        Preprocess states: flatten to 2D and convert to bfloat16.

        Args:
            states: [B, ...] any shape

        Returns:
            states: [B, state_dim] flattened states as bfloat16
        """
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        # Convert to bfloat16 to match encoder's dtype
        if states.dtype != torch.bfloat16:
            states = states.to(torch.bfloat16)
        return states
