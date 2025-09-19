import contextlib
from typing import Any, Dict, Literal, Tuple
import torch
from torch import Tensor
from typing_extensions import override

import math
import pickle
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.models.pi0_config import Pi0Config
from collections.abc import Sequence
import jax
import jax.numpy as jnp
import numpy as np
import time
from collections import namedtuple
import random
import torch.nn as nn
import torch
from torch import nn
from typing import List

from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
class OpenPi0ForRLActionPrediction(PI0Pytorch):
    """Pi0 model for reinforcement learning action prediction.
    
    This is a template class that defines the interfaces needed for RL training.
    You need to implement all the methods marked with 'TODO: Implement'.
    """
    @property
    def _no_split_modules(self) -> list[str]:
        # Currently, PaliGemmaForConditionalGeneration only support DDP, as many of it's modules are called without forward
        return ['PaliGemmaForConditionalGeneration', 'GemmaDecoderLayer', 'SiglipVisionEmbeddings', 'GemmaRMSNorm', 'GemmaForCausalLM', 'GemmaRotaryEmbedding', 'ValueProj']
        
    def __init__(
        self,
        config: Pi0Config,
    ):
        super().__init__(config)
        proj_width = 1024
        self.global_step = 0
        if self.config.adv_method == "ppo":
            self.value_proj = ValueProj(proj_width)
            # self.value_proj = nn.Linear(width, 1)
        if self.config.noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=proj_width, 
                out_dim=self.config.action_dim,
                hidden_dims=[128,64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn"
            )
    
    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
    
    def input_transform(self,obs: dict, transpose = True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {value: inputs[value] for value in inputs.keys() if value in ["observation/image","observation/wrist_image","observation/state"]} 
        # tensor -> numpy
        inputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, 'shape'))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(
                lambda x: x[i],
                inputs
            )
            # convert from [3,256,256] -> [256,256,3]
            if transpose:
                sample = jax.tree.map(lambda x: x.transpose(1,2,0) if len(x.shape) == 3 and transpose else x, sample)
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
            *transformed_samples
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if first_process == False:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs
    
    def output_transform(self,outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()),outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()), 
            *transformed_samples
        )
        outputs["actions"] = outputs["actions"][:,:self.config.action_chunk]
        return outputs
    
    def forward(
        self, 
        data: dict[str, Tensor], 
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
        chains = data['chains']
        denoise_inds = data['denoise_inds']
        # input transform
        observation = self.input_transform(data)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t = self.get_log_prob_value(images, img_masks, lang_tokens, lang_masks, state, chains, denoise_inds)
        log_probs = log_probs[:,:, :self.config.action_chunk,:self.config.action_env_dim]
        # post process
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = data["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:,0]
            prev_logprobs = data["prev_logprobs"]
            prev_logprobs = prev_logprobs[torch.arange(bsize),denoise_inds[:,0], :self.config.action_chunk,:self.config.action_env_dim]
        value_t = value_t.mean(dim=-1,keepdim=False)
        return {
            "logprobs": log_probs,
            "prev_logprobs": prev_logprobs,
            "values": value_t,
            "entropy": None,
        }
    def _process_obs_from_env(self, env_processed_obs):
        to_process_obs = {
            "observation/image": env_processed_obs["image"],
            "observation/wrist_image": env_processed_obs["wrist_image"],
            "observation/state": env_processed_obs["state"],
            "prompt": env_processed_obs["task_descriptions"],
        }
        processed_obs = self.input_transform(to_process_obs)
        processed_obs.update(
            {
                "observation/image": env_processed_obs["image"],
                "observation/wrist_image": env_processed_obs["wrist_image"],
                "observation/state": env_processed_obs["state"],
            }
        )
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous() if torch.is_tensor(item) else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            # todo: patch for openpi
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(device=device).contiguous()
        return processed_obs
    
    def predict_action_batch(self, env_processed_obs, mode: Literal["train", "eval"] = "train") -> Tensor:
        processed_obs = self._process_obs_from_env(env_processed_obs)
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(observation, mode = mode) 
        outputs["actions"] = self.output_transform(
            {
                "actions": outputs["actions"],
                "state": observation.state 
            }
        )["actions"].numpy()
        return outputs, processed_obs

    @torch.no_grad()
    def sample_actions(self, observation: _model.Observation, noise=None, mode="train") -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
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
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(x_t, torch.zeros_like(noise), torch.ones_like(noise))
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.noise_method == "flow_sde":
                    if self.config.ignore_last:
                        denoise_inds =  torch.tensor([random.randint(0,num_steps-2)] * num_steps) 
                    else:
                        denoise_inds =  torch.tensor([random.randint(0,num_steps-1)] * num_steps) 
                elif self.config.noise_method == "flow_cps":
                    # the last denoising step of the flow-cps is deterministic
                    denoise_inds =  torch.tensor([random.randint(0,num_steps-1)] * num_steps) 
                elif self.config.noise_method == "reinflow":
                    denoise_inds =  torch.tensor([random.randint(0,num_steps-1)] * num_steps) 
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                x_t_mean,x_t_std,value_t = self.sample_mean_var_val(x_t,idx,state,prefix_pad_masks,past_key_values,"train",num_steps)
            else:
                x_t_mean,x_t_std,value_t = self.sample_mean_var_val(x_t,idx,state,prefix_pad_masks,past_key_values,"eval",num_steps)
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t,x_t_mean,x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains,dim=1)
        log_probs = torch.stack(log_probs, dim=1)[:,:,:self.config.action_chunk,:self.config.action_env_dim]
        values = torch.stack(values, dim=1).mean(dim = -1, keepdim = True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds
        }    

    def sample_mean_var_val(self,x_t,idx,state,prefix_pad_masks,past_key_values,mode,denoise_steps):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx,int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = noise_start + (noise_end - noise_start) * min(self.global_step, anneal_steps) / anneal_steps
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
        v_t = self.action_out_proj(suffix_out) # [bs,n_action_steps,max_action_dim]
        # value prediction 
        breakpoint()
        if self.config.adv_method == "ppo":
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(suffix_out[:,:self.config.action_chunk],dim = 1,keepdim=False)
            else:
                suffix_out_value = torch.mean(suffix_out,dim = 1,keepdim=False)
            # detach critic input 
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_proj(suffix_out_value)[:,0]
        else:
            value_t = torch.zeros((bsize),device=device)
        # ode sde mix sampling
        delta = delta[:,None,None].expand_as(x_t)
        t_input = t_input[:,None,None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode in ["train","compute_logprob"]:
            if self.config.noise_method == "flow_sde":
                sigmas = noise_level * torch.sqrt(
                    timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                )[:-1]
                sigma_i = sigmas[idx][:,None,None].expand_as(x_t)
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
            elif self.config.noise_method == "reinflow":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.reinflow_explore_noise_net(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean,x_t_std,value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

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
    def get_logprob_norm(self,sample,mu,sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self, images, img_masks, lang_tokens, lang_masks, state, 
        chains, denoise_inds
    ):
        bsize = state.shape[0]
        device = state.device
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
        chains_log_probs = []
        chains_values = []
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(chains[:,0], torch.zeros_like(chains[:,0]), torch.ones_like(chains[:,0]))
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:,idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean,x_t_std,value_t = self.sample_mean_var_val(chains_pre,denoise_ind,state,prefix_pad_masks,past_key_values,"compute_logprob",self.config.num_steps)
            log_probs = self.get_logprob_norm(chains_next,x_t_mean,x_t_std)
            chains_log_probs.append(log_probs)
            chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        return chains_log_probs, chains_values

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
        
    # Pi0 do not need this function, so we leave it empty, parameters are useless here
    def setup_params(self, model_config, cfg):
        self.action_dim = cfg.actor.model.action_dim
        self.max_prompt_length = cfg.runner.max_prompt_length   

class ExploreNoiseNet(nn.Module):
    '''
    Neural network to generate learnable exploration noise, conditioned on time embeddings and or state embeddings. 
    \sigma(s,t) or \sigma(s)
    '''
    def __init__(self,
                 in_dim:int,
                 out_dim:int,
                 hidden_dims:List[int], 
                 activation_type:str,
                 noise_logvar_range:list, #[min_std, max_std]
                 noise_scheduler_type: str
                 ):
        super().__init__()
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims +[out_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
        )
        self.noise_scheduler_type=noise_scheduler_type
        self.set_noise_range(noise_logvar_range)
    
    def set_noise_range(self, noise_logvar_range:list):
        self.noise_logvar_range=noise_logvar_range
        noise_logvar_min = self.noise_logvar_range[0]
        noise_logvar_max = self.noise_logvar_range[1]
        self.register_buffer('logvar_min', torch.log(torch.tensor(noise_logvar_min**2, dtype=torch.float32)).unsqueeze(0))
        self.register_buffer('logvar_max', torch.log(torch.tensor(noise_logvar_max**2, dtype=torch.float32)).unsqueeze(0))
        
    def forward(self, noise_feature:torch.Tensor):
        if 'const' in self.noise_scheduler_type: # const or const_schedule_itr
            # pick the lowest noise level when we use constant noise schedulers. 
            noise_std     = torch.exp(0.5 * self.logvar_min)
        else:
            # use learnable noise level.
            noise_logvar  = self.mlp_logvar(noise_feature)
            noise_std     = self.post_process(noise_logvar)
        return noise_std

    def post_process(self, noise_logvar):
        """
        input:
            torch.Tensor([B, Ta , Da])   log \sigma^2 
        output:
            torch.Tensor([B, Ta, Da]),   sigma, floating point values, bounded in [noise_logvar_min, noise_logvar_max]
        """
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1)/2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std

class ValueProj(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(width, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.value_proj(x)



activation_dict = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
        "silu": nn.SiLU(),
    }
)

class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="tanh",
        out_activation_type="identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        out_bias_init=None,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Ensure append_layers is always a list to avoid TypeError
        self.append_layers = append_layers if append_layers is not None else []

        # Construct module list
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))   # type: ignore
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))   # type: ignore

            # Add activation function
            act = (
                activation_dict[activation_type.lower()]
                if idx != num_layer - 1
                else activation_dict[out_activation_type.lower()]
            )
            layers.append(("act_1", act))   # type: ignore

            # Re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][0]  # Linear layer is first in the last Sequential # type: ignore
            nn.init.constant_(final_linear.bias, out_bias_init)
            logger.info(f"Initialized the bias of the final linear layer to {out_bias_init}")
    
    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x