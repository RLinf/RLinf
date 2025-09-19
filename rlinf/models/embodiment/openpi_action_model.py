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

import random
from collections.abc import Sequence
from typing import Any, Dict, Literal

import jax
import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch import Tensor


class OpenPi0ForRLActionPrediction(PI0Pytorch):
    """Pi0 model for reinforcement learning action prediction.

    This is a template class that defines the interfaces needed for RL training.
    You need to implement all the methods marked with 'TODO: Implement'.
    """

    @property
    def _no_split_modules(self) -> list[str]:
        # Currently, PaliGemmaForConditionalGeneration only support DDP, as many of it's modules are called without forward
        return [
            "PaliGemmaForConditionalGeneration",
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaForCausalLM",
            "GemmaRotaryEmbedding",
        ]

    def __init__(
        self,
        config: Pi0Config,
    ):
        super().__init__(config)

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
            inputs = {
                value: inputs[value]
                for value in inputs.keys()
                if value
                in ["observation/image", "observation/wrist_image", "observation/state"]
            }
        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            # convert from [3,256,256] -> [256,256,3]
            if transpose:
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

    def forward(
        self,
        data: dict[str, Tensor],
        mode: Literal["train", "eval", "compute_logprob"] = "predict",
    ) -> Dict[str, Any]:
        """
        Unified forward function for Pi0 model that handles both prediction and logprob computation.

        Args:
            batch: Input batch dictionary containing observations (predict mode) or training data (compute_logprob mode)
            mode: "predict" for action prediction, "compute_logprob" for logprob computation
            sample_mode: Sampling mode for prediction ("ode" or "sde") - not used in current predict implementation
            rollout_stage: Whether in rollout stage - not used in current predict implementation
            output_lang_tokens: Whether to output language tokens in prediction mode
            requires_grad: Whether to enable gradients (auto-detected if None)

        Returns:
            Dictionary containing:
            - For predict mode: actions, chains, log_probs, values, (optionally lang_tokens, lang_masks)
            - For compute_logprob mode: token_level_entropy, token_level_log_probs, action_level_logprobs
        """
        if mode in ["train", "eval"]:
            data = _model.Observation.from_dict(data)
            outputs = self.sample_actions(data, mode=mode)
            outputs["actions"] = self.output_transform(
                {
                    "actions": outputs["actions"],
                    "state": data.state,  # TODO: for openpi interface, state function?
                }
            )["actions"].numpy()
            return outputs
        elif mode == "compute_logprob":
            # chains
            length = data["chains"].shape[0]
            denoise_inds = data["denoise_inds"]
            chains_pre = data["chains"][
                torch.arange(length), denoise_inds
            ]  # [length, act_steps, action_dim]
            chains_next = data["chains"][
                torch.arange(length), denoise_inds + 1
            ]  # [length, act_steps, action_dim]
            # input transform
            observation = self.input_transform(data)
            observation = _model.Observation.from_dict(observation)
            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )
            # transfer to device
            device = chains_pre.device
            images = [img.to(device) for img in images]
            img_masks = [img_mask.to(device) for img_mask in img_masks]
            state = state.to(device)
            # get log prob
            log_probs, value_t = self.get_log_prob_value(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                chains_pre,
                chains_next,
                denoise_inds,
            )
            log_probs = log_probs[
                :, : self.config.action_chunk, : self.config.action_env_dim
            ]
            return {
                "logprobs": log_probs,
                "values": value_t,
                "entropy": None,
            }

    @torch.no_grad()
    def sample_actions(
        self, observation: _model.Observation, noise=None, mode="train"
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

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

        _, past_key_values = self.paligemma_with_expert.forward(
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

        # denoise index
        if mode == "train":
            denoise_inds = torch.tensor([random.randint(0, num_steps - 1)] * bsize)
        else:
            denoise_inds = torch.tensor([-1] * bsize)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0]:
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    x_t,
                    idx,
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    "train",
                    num_steps,
                )
            else:
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    x_t,
                    idx,
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    "eval",
                    num_steps,
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
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        values = torch.stack(values, dim=1)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self, x_t, idx, state, prefix_pad_masks, past_key_values, mode, denoise_steps
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
        if hasattr(self.config, "noise_level"):
            noise_level = torch.tensor(self.config.noise_level).to(device)
        else:
            noise_level = torch.tensor(0).to(device)
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
        if hasattr(self.config, "adv_method") and self.config.adv_method == "gae":
            suffix_out = torch.mean(suffix_out, dim=1, keepdim=False)
            value_t = self.value_proj(suffix_out)[:, 0]
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
        elif mode in ["train", "compute_logprob"]:
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

    def get_logprob_norm(self, sample, mu, sigma):
        if torch.sum(torch.abs(sigma)) == 0:
            return torch.zeros_like(sample)
        sigma_safe = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        log_prob = torch.where(sigma == 0, torch.zeros_like(log_prob), log_prob)
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
        chains_pre,
        chains_next,
        denoise_inds,
    ):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
            chains_pre,
            denoise_inds,
            state,
            prefix_pad_masks,
            past_key_values,
            "compute_logprob",
            self.config.num_steps,
        )
        log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
        return log_probs, value_t

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

    # Pi0 do not need this function, so we leave it empty, parameters are useless here
    def setup_params(self, model_config, cfg):
        self.action_dim = cfg.actor.model.action_dim
        self.max_prompt_length = cfg.runner.max_prompt_length
