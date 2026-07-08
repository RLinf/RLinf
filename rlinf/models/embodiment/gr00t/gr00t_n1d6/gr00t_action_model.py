# Copyright 2026 The RLinf Authors.
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

import json
import logging
import random
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from torch import nn
from torch.distributions import Normal
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION_N1D6,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


class SimulationContent:
    """Lightweight container for a single timestep's worth of GR00T processor inputs."""

    def __init__(self, embodiment, states, actions, images, text, masks=None):
        self.embodiment = embodiment
        self.states = states
        self.actions = actions
        self.images = images
        self.text = text
        self.masks = masks


class FlowMatchingActionHeadForRLActionPrediction(nn.Module):
    def __init__(
        self,
        config: Any,  # Gr00tN1d6Config
        rl_head_config: dict[str, Any],
        output_action_chunks: int = 1,
        parent_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Keep a non-registered reference to the GR00T parent so this RL wrapper
        # can call the pretrained action modules without duplicating them.
        self.__dict__["_parent_model"] = parent_model
        self.config = config
        self.rl_config = rl_head_config
        # Only set defaults if not already specified in config
        if "noise_method" not in self.rl_config:
            self.rl_config["noise_method"] = "flow_sde"
        if "noise_level" not in self.rl_config:
            self.rl_config["noise_level"] = 0.5
        if "noise_anneal" not in self.rl_config:
            self.rl_config["noise_anneal"] = False
        self.padding_value = rl_head_config.get("padding_value", 0)
        self.output_action_chunks = output_action_chunks
        # self.valid_action_dim = getattr(self, "valid_action_dim", config.get("action_dim", 7))
        self.valid_action_dim = getattr(config, "max_action_dim", 7)
        self.action_chunk = output_action_chunks
        self.action_dim = getattr(
            config, "action_dim", getattr(config, "max_action_dim", 7)
        )
        self.hidden_size = getattr(
            config, "hidden_size", getattr(self, "hidden_size", 1024)
        )
        self.action_horizon = getattr(
            config, "action_horizon", getattr(self, "action_horizon", 16)
        )
        self.num_timestep_buckets = getattr(
            config, "num_timestep_buckets", getattr(self, "num_timestep_buckets", 1000)
        )
        self.num_inference_timesteps = getattr(
            config,
            "num_inference_timesteps",
            getattr(self, "num_inference_timesteps", 4),
        )

        vlm_width = getattr(config, "backbone_embedding_dim", 2048)
        state_width = getattr(config, "input_embedding_dim", 1536)
        if self.rl_config.get("use_vlm_value", False):
            proj_width = vlm_width
        else:
            proj_width = vlm_width + state_width

        if self.rl_config.get("add_value_head", False):
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.get("noise_method") == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                # out_dim=config.get("action_dim", 7),
                out_dim=getattr(config, "max_action_dim", 7),
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    def _get_component(self, name: str):
        component = getattr(self, name, None)
        if component is not None:
            return component
        parent_model = self.__dict__.get("_parent_model")
        if parent_model is None:
            return None
        return getattr(parent_model, name, None)

    def _process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        vlln = self._get_component("vlln")
        if vlln is not None and hasattr(backbone_output, "backbone_features"):
            backbone_output.backbone_features = vlln(backbone_output.backbone_features)
        return backbone_output

    def prepare_input(self, inputs: dict) -> BatchFeature:
        from transformers.feature_extraction_utils import BatchFeature

        action_inputs = {}
        for k in ["state", "action", "action_mask", "embodiment_id"]:
            if k in inputs:
                action_inputs[k] = inputs[k]

        return BatchFeature(data=action_inputs)

    def get_logprob_norm(self, sample, mu, sigma):
        if self.rl_config.get("safe_get_logprob", False):
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
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

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        backbone_output: Optional[BatchFeature] = None,
    ):
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        if self.rl_config.get("noise_anneal"):
            noise_start, noise_end, anneal_steps = self.rl_config.get(
                "noise_params", (0.0, 0.0, 100)
            )
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(getattr(self, "global_step", 0), anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.rl_config.get("noise_level", 0.5)).to(
                device
            )

        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_encoder = self._get_component("action_encoder")
        action_features = (
            action_encoder(x_t, timesteps_tensor, embodiment_id)
            if action_encoder is not None
            else x_t
        )
        position_embedding = self._get_component("position_embedding")
        if (
            getattr(self.config, "add_pos_embed", False)
            and position_embedding is not None
        ):
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(bsize, -1, -1) if hasattr(self, "future_tokens") else torch.zeros(bsize, 1, self.hidden_size, device=device)
        # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        sa_embs = torch.cat((state_features, action_features), dim=1)

        denoising_model = self._get_component("model")
        if denoising_model is not None:
            if (
                getattr(self.config, "use_alternate_vl_dit", False)
                and backbone_output is not None
            ):
                model_output = denoising_model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = denoising_model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
        else:
            model_output = sa_embs
        model_output = model_output[:, -self.action_horizon :]

        action_decoder = self._get_component("action_decoder")
        v_t = (
            action_decoder(model_output, embodiment_id)
            if action_decoder is not None
            else torch.zeros_like(model_output)
        )

        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)

        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta
            x_t_std = torch.zeros_like(t_input)
        else:
            if self.rl_config.get("noise_method") == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            else:
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = torch.zeros_like(t_input)

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = (
            backbone_output.backbone_features
            if hasattr(backbone_output, "backbone_features")
            else backbone_output
        )
        embodiment_id = (
            action_input.embodiment_id if hasattr(action_input, "embodiment_id") else 0
        )
        state_encoder = self._get_component("state_encoder")
        state_features = (
            state_encoder(action_input.state, embodiment_id)
            if state_encoder is not None
            else action_input.state
        )
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.action_horizon, self.valid_action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.get("joint_logprob"):
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        if mode == "train":
            if self.rl_config.get("joint_logprob"):
                denoise_inds = torch.arange(num_steps)
            else:
                denoise_inds = torch.tensor(
                    [random.randint(0, num_steps - 1)] * num_steps
                )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="train",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                    backbone_output=backbone_output,
                )
            else:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="eval",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                    backbone_output=backbone_output,
                )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)

        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.action_dim
        ]
        if compute_values:
            values = self.get_value(vl_embs, state_features)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(data={"action_pred": x_0}), {
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = (
            backbone_output.backbone_features
            if hasattr(backbone_output, "backbone_features")
            else backbone_output
        )
        embodiment_id = (
            action_input.embodiment_id if hasattr(action_input, "embodiment_id") else 0
        )
        state_encoder = self._get_component("state_encoder")
        state_features = (
            state_encoder(action_input.state, embodiment_id)
            if state_encoder is not None
            else action_input.state
        )
        batch_size = vl_embs.shape[0]

        chains_log_probs = []
        if self.rl_config.get("joint_logprob"):
            num_steps = getattr(self.config, "num_steps", 1)
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
                backbone_output=backbone_output,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )

        return chains_log_probs, chains_values

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.bfloat16, device=device
        )

    def get_value(self, vl_embs, state_features):
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.get("value_vlm_mode") == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.get("value_vlm_mode") == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.get("value_vlm_mode") == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :].mean(dim=1, keepdim=False)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.get("use_vlm_value", False):
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)

        values_vlm = self.value_head(value_embs)[:, 0]

        return values_vlm


class GR00T_N1_6_ForRLActionPrediction(Gr00tN1d6, BasePolicy):
    """
    GR00T N1.6 model for reinforcement learning action prediction.
    """

    _no_split_modules = [
        "Qwen3DecoderLayer",
        "Siglip2EncoderLayer",
    ]

    def __init__(
        self,
        config: Gr00tN1d6Config,
        rl_head_config: dict[str, Any],
        embodiment_tag: Union[str, EmbodimentTag],
        local_model_path: str,
        modality_config: Optional[Any] = None,
        modality_transform: Optional[Any] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
        **kwargs,
    ):
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag
        processor_path = kwargs.pop("processor_path", None)
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        super().__init__(
            config, transformers_loading_kwargs=transformers_loading_kwargs, **kwargs
        )

        self.padding_value = rl_head_config.get("padding_value", 0)
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks
        self.action_dim = getattr(
            config, "action_dim", getattr(config, "max_action_dim", 7)
        )

        if modality_config is None or modality_transform is None:
            from transformers import AutoProcessor

            logging.info("loading Processor...")
            if processor_path is not None:
                processor_path = Path(processor_path)
                with open(processor_path / "processor_config.json", "r") as f:
                    processor_cfg = json.load(f)["processor_kwargs"]
                with open(processor_path / "statistics.json", "r") as f:
                    processor_cfg["statistics"] = json.load(f)
                with open(processor_path / "embodiment_id.json", "r") as f:
                    processor_cfg["embodiment_id_mapping"] = json.load(f)
                modality_transform = Gr00tN1d6Processor(**processor_cfg)
                modality_config = getattr(modality_transform, "modality_configs", None)
            else:
                processor = AutoProcessor.from_pretrained(
                    str(local_model_path), trust_remote_code=True
                )
                modality_transform = processor
                modality_config = getattr(processor, "modality_config", None)

            logging.info("Processor loaded safely. No model weights were touched.")

            # GR00T16_RLINF_COMPAT: ``Eagle3_VLImageProcessorFast`` (shipped
            # inside the SFT checkpoint as a ``trust_remote_code`` module) was
            # written against transformers <4.55 where ``BaseImageProcessorFast``
            # exposed ``_prepare_input_images``. From 4.55+ the helper is
            # renamed to ``_prepare_image_like_inputs``. Alias the new method
            # under the old name on the processor's class so the workshop
            # SO-101 SFT checkpoint loads on transformers 4.57.x.
            try:
                _eagle_image_proc = getattr(
                    getattr(modality_transform, "processor", None),
                    "image_processor",
                    None,
                )
                if _eagle_image_proc is not None:
                    _cls = type(_eagle_image_proc)
                    if not hasattr(_cls, "_prepare_input_images") and hasattr(
                        _cls, "_prepare_image_like_inputs"
                    ):
                        _cls._prepare_input_images = _cls._prepare_image_like_inputs
                        logging.info(
                            "[GR00T16_RLINF_COMPAT] aliased "
                            f"{_cls.__name__}._prepare_input_images -> "
                            "_prepare_image_like_inputs (transformers >=4.55)"
                        )
            except Exception as _e:  # pragma: no cover
                logging.info(
                    "[GR00T16_RLINF_COMPAT] could not alias "
                    f"_prepare_input_images on Eagle3 image processor: {_e}"
                )

        self._modality_config = modality_config
        self._modality_transform = modality_transform

        pretrained_action_head = self.action_head
        self.action_head = FlowMatchingActionHeadForRLActionPrediction(
            config, rl_head_config, output_action_chunks
        )
        for attr_name in (
            "model",
            "state_encoder",
            "action_encoder",
            "action_decoder",
            "position_embedding",
            "vlln",
            "mask_token",
        ):
            if hasattr(pretrained_action_head, attr_name):
                setattr(
                    self.action_head,
                    attr_name,
                    getattr(pretrained_action_head, attr_name),
                )

        if denoising_steps is not None and hasattr(
            self.action_head, "num_inference_timesteps"
        ):
            self.action_head.num_inference_timesteps = denoising_steps

        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION_N1D6[obs_converter_type]
        exp_cfg_path = self.model_path / "experiment_cfg"
        self._load_metadata(exp_cfg_path)

        self._no_split_modules = self.__class__._no_split_modules

        if hasattr(self, "config"):
            self.config.no_split_modules = self._no_split_modules
            self.config._no_split_modules = self._no_split_modules
        logging.info(
            f" Forced FSDP _no_split_modules into config: {self.config.no_split_modules}"
        )

    def eval(self):
        self._modality_transform.eval()
        super().eval()

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:
                return False
        return True

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        state_tensor = forward_inputs["state"]
        state_mask = forward_inputs.get("state_mask", None)
        if state_mask is None:
            state_mask = torch.ones(
                state_tensor.shape[:-1], dtype=torch.bool, device=state_tensor.device
            )
        bsize = state_tensor.shape[0]
        self.image_nums = forward_inputs["eagle_pixel_values"].shape[1]
        normalized_input = {
            "state": forward_inputs["state"],
            # "state_mask": forward_inputs["state_mask"],
            "state_mask": forward_inputs.get("state_mask", None),
            "eagle_input_ids": forward_inputs["eagle_input_ids"],
            "eagle_attention_mask": forward_inputs["eagle_attention_mask"],
            "eagle_pixel_values": forward_inputs["eagle_pixel_values"].reshape(
                bsize, self.image_nums, *forward_inputs["eagle_pixel_values"].shape[-3:]
            ),
            "eagle_image_sizes": forward_inputs["eagle_image_sizes"].reshape(
                bsize, self.image_nums, *forward_inputs["eagle_image_sizes"].shape[-1:]
            ),
            "embodiment_id": forward_inputs["embodiment_id"],
        }

        if "eagle_input_ids" in normalized_input:
            normalized_input["input_ids"] = normalized_input["eagle_input_ids"]
            normalized_input["attention_mask"] = normalized_input[
                "eagle_attention_mask"
            ]
            normalized_input["pixel_values"] = normalized_input["eagle_pixel_values"]
            if "eagle_image_sizes" in normalized_input:
                normalized_input["image_sizes"] = normalized_input["eagle_image_sizes"]

        normalized_input = {k: v for k, v in normalized_input.items() if v is not None}

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        if self.action_head.rl_config.get("joint_logprob"):
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = kwargs["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = kwargs["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        log_probs = log_probs[..., : self.action_dim]
        prev_logprobs = prev_logprobs[..., : self.action_dim]

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": None,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ):
        # Here we have a source causing tiny inference-training inconsistency,
        # force convert the state to bf16 then back to float32 to reproduce the info loss in training.
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()

        observations = self.obs_convert_fn(env_obs)

        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        # GR00T16_RLINF_COMPAT: GR00T N1.6 SO-101 action configs use
        # ActionRepresentation.RELATIVE. Decoding relative model actions back
        # into absolute IsaacLab joint targets requires the current raw state as
        # the reference frame. The rollout obs converter publishes state keys as
        # ``state.<joint_group>`` while StateActionProcessor expects bare group
        # names such as ``single_arm``.
        decode_state = {}
        for k, v in obs_copy.items():
            if k.startswith("state."):
                decode_state[k.split(".", 1)[1]] = np.asarray(v, dtype=np.float32)
        if not decode_state and "state" in obs_copy:
            decode_state["single_arm"] = np.asarray(obs_copy["state"], dtype=np.float32)
        if not decode_state:
            decode_state = None

        normalized_input = self.apply_transforms(obs_copy)

        for key in normalized_input:
            if isinstance(normalized_input[key], torch.Tensor):
                if normalized_input[key].dtype == torch.float32:
                    normalized_input[key] = normalized_input[key].to(torch.bfloat16)

        if "input_ids" in normalized_input:
            normalized_input["input_ids"] = torch.nn.functional.pad(
                normalized_input["input_ids"],
                pad=(0, self.padding_value - normalized_input["input_ids"].shape[-1]),
                mode="constant",
                value=0,
            )
            normalized_input["eagle_input_ids"] = normalized_input["input_ids"]

        if "attention_mask" in normalized_input:
            normalized_input["attention_mask"] = torch.nn.functional.pad(
                normalized_input["attention_mask"],
                pad=(
                    0,
                    self.padding_value - normalized_input["attention_mask"].shape[-1],
                ),
                mode="constant",
                value=0,
            )
            normalized_input["eagle_attention_mask"] = normalized_input[
                "attention_mask"
            ]

        normalized_action, result = self._get_rl_action(normalized_input, mode=mode)
        unnormalized_action = self._get_unnormalized_action(
            normalized_action, state=decode_state
        )

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )

        if mode == "train":
            noise_scale = self.action_head.rl_config.get("action_noise_scale", 0.1)
            if noise_scale > 0:
                is_numpy = isinstance(raw_action, np.ndarray)
                raw_tensor = torch.from_numpy(raw_action) if is_numpy else raw_action
                noise = torch.randn_like(raw_tensor) * noise_scale
                raw_tensor = (raw_tensor + noise).clamp(-1.0, 1.0)
                raw_action = raw_tensor.numpy() if is_numpy else raw_tensor

        return raw_action, result

    # ------------------------------------------------------------------
    # Observation transform helpers (refactored from the monolithic
    # apply_transforms method).
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_value(embodiment_tag) -> str:
        return (
            embodiment_tag.value
            if hasattr(embodiment_tag, "value")
            else str(embodiment_tag)
        )

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _split_libero_state(value):
        value = GR00T_N1_6_ForRLActionPrediction._to_numpy(value).reshape(-1)
        if value.shape[0] < 7:
            raise ValueError(
                f"LIBERO state should have at least 7 dims, got {value.shape}"
            )
        return {
            "x": value[0:1][None, :],
            "y": value[1:2][None, :],
            "z": value[2:3][None, :],
            "roll": value[3:4][None, :],
            "pitch": value[4:5][None, :],
            "yaw": value[5:6][None, :],
            "gripper": value[6:7][None, :],
        }

    @staticmethod
    def _classify_obs_keys(obs: dict):
        text_key = None
        image_keys = []
        state_keys = []
        for k in obs.keys():
            k_lower = k.lower()
            if "task" in k_lower or "lang" in k_lower or "instruction" in k_lower:
                text_key = k
            elif (
                "image" in k_lower
                or "rgb" in k_lower
                or "cam" in k_lower
                or k_lower.startswith("video.")
            ):
                image_keys.append(k)
            else:
                state_keys.append(k)
        return text_key, image_keys, state_keys

    @staticmethod
    def _find_state_key(obs: dict):
        for candidate in ("state", "observation.state"):
            if candidate in obs:
                return candidate
        return None

    @classmethod
    def _extract_obs_text(cls, obs: dict, text_key, i: int) -> str:
        if text_key is None:
            return ""
        text = obs[text_key][i]
        if isinstance(text, (list, np.ndarray)):
            return str(text[0]) if len(text) > 0 else ""
        return str(text)

    @classmethod
    def _process_obs_states(cls, obs: dict, state_keys, tag_val: str, i: int) -> dict:
        state_key = cls._find_state_key(obs)
        if tag_val == "libero_panda" and state_key is not None:
            return cls._split_libero_state(obs[state_key][i])
        if tag_val == "libero_panda":
            states_dict = {}
            for k in state_keys:
                if k.startswith("state."):
                    states_dict[k.split(".", 1)[1]] = cls._to_numpy(obs[k][i])
            if not states_dict:
                raise KeyError(
                    f"No LIBERO state keys found in observation: {list(obs.keys())}"
                )
            return states_dict

        states_dict = {}
        for k in state_keys:
            v = obs[k][i]
            if isinstance(v, torch.Tensor):
                v = v.cpu().float().numpy()
            bare_key = k.split(".", 1)[1] if k.startswith("state.") else k
            states_dict[bare_key] = np.array(v)

        ref_T = next(iter(states_dict.values())).shape[0] if states_dict else 1
        if tag_val == "robocasa_panda_omron":
            robocasa_requirements = {
                "end_effector_position_relative": 3,
                "end_effector_rotation_relative": 4,
                "gripper_qpos": 2,
                "base_position": 3,
                "base_rotation": 4,
            }
            for req_k, req_dim in robocasa_requirements.items():
                if req_k not in states_dict:
                    states_dict[req_k] = np.zeros((ref_T, req_dim), dtype=np.float32)
        return states_dict

    def _process_obs_images(self, obs: dict, image_keys, tag_val: str, i: int) -> dict:
        import numpy as np
        from PIL import Image

        raw_images_list = []
        for img_k in image_keys:
            img_data = obs[img_k][i]
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.cpu().numpy()

            frames = []
            if img_data.ndim == 3:
                img_data = np.expand_dims(img_data, 0)

            for t in range(img_data.shape[0]):
                frame = img_data[t]
                if frame.shape[0] in [1, 3] and frame.shape[2] > 3:
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                frames.append(Image.fromarray(frame))
            raw_images_list.append(frames)

        try:
            req_img_keys = self._modality_transform.modality_configs[tag_val][
                "video"
            ].modality_keys
        except Exception:
            req_img_keys = [
                "res256_image_side_0",
                "res256_image_wrist_0",
                "res256_image_front_0",
            ]

        self.image_nums = len(req_img_keys)

        images_dict = {}
        if tag_val == "libero_panda":
            raw_by_key = dict(zip(image_keys, raw_images_list))
            source_for_req = {
                "image": (
                    raw_by_key.get("base_0_rgb")
                    or raw_by_key.get("observation.images.image")
                    or raw_by_key.get("video.image")
                ),
                "wrist_image": (
                    raw_by_key.get("right_wrist_0_rgb")
                    or raw_by_key.get("observation.images.wrist_image")
                    or raw_by_key.get("video.wrist_image")
                ),
            }
            for r_key in req_img_keys:
                images_dict[r_key] = source_for_req.get(r_key) or (
                    raw_images_list[-1] if raw_images_list else []
                )
        else:
            for idx, r_key in enumerate(req_img_keys):
                if idx < len(raw_images_list):
                    images_dict[r_key] = raw_images_list[idx]
                else:
                    images_dict[r_key] = raw_images_list[-1] if raw_images_list else []
        return images_dict

    def _collate_and_rename(self, processed_outputs: list) -> dict:
        collated_batch = self._modality_transform.collator(processed_outputs)

        if hasattr(collated_batch, "data") and "inputs" in collated_batch.data:
            batched_out = collated_batch.data["inputs"]
        elif "inputs" in collated_batch:
            batched_out = collated_batch["inputs"]
        else:
            batched_out = dict(collated_batch)

        for src, dst in [
            ("input_ids", "eagle_input_ids"),
            ("attention_mask", "eagle_attention_mask"),
            ("pixel_values", "eagle_pixel_values"),
            ("image_sizes", "eagle_image_sizes"),
        ]:
            if src in batched_out:
                batched_out[dst] = batched_out[src]

        if "eagle_pixel_values" not in batched_out and "images" in batched_out:
            batched_out["eagle_pixel_values"] = batched_out["images"]
            batched_out["pixel_values"] = batched_out["images"]

        return batched_out

    def apply_transforms(self, obs: dict) -> dict:
        text_key, image_keys, state_keys = self._classify_obs_keys(obs)
        tag_val = self._tag_value(self.embodiment_tag)
        batch_size = len(next(iter(obs.values())))
        processed_outputs = []

        for i in range(batch_size):
            text = self._extract_obs_text(obs, text_key, i)
            states_dict = self._process_obs_states(obs, state_keys, tag_val, i)
            images_dict = self._process_obs_images(obs, image_keys, tag_val, i)

            content = SimulationContent(
                embodiment=self.embodiment_tag,
                states=states_dict,
                actions=None,
                images=images_dict,
                text=text,
            )
            messages = [{"role": "user", "content": content}]
            processed_outputs.append(self._modality_transform(messages=messages))

        return self._collate_and_rename(processed_outputs)

    def unapply_transforms(
        self, action: dict[str, Any], state: dict[str, np.ndarray] | None = None
    ) -> dict[str, Any]:
        raw_action_tensor = action["action"]

        if isinstance(raw_action_tensor, torch.Tensor):
            raw_action_tensor = raw_action_tensor.detach().cpu().numpy()

        decoded = self._modality_transform.decode_action(
            action=raw_action_tensor, embodiment_tag=self.embodiment_tag, state=state
        )
        return decoded

    def _get_rl_action(
        self,
        normalized_input: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ):
        if "eagle_input_ids" in normalized_input:
            normalized_input["input_ids"] = normalized_input["eagle_input_ids"]
            normalized_input["attention_mask"] = normalized_input[
                "eagle_attention_mask"
            ]
            normalized_input["pixel_values"] = normalized_input["eagle_pixel_values"]
            if "eagle_image_sizes" in normalized_input:
                normalized_input["image_sizes"] = normalized_input["eagle_image_sizes"]

        if normalized_input.get("state_mask") is None and "state" in normalized_input:
            normalized_input["state_mask"] = torch.ones(
                normalized_input["state"].shape[:-1],
                dtype=torch.bool,
                device=normalized_input["state"].device,
            )
        normalized_input = {k: v for k, v in normalized_input.items() if v is not None}

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
            backbone_outputs, action_inputs, mode=mode
        )
        actions = rlinf_outputs["actions"]
        self.validate_data(
            action_head_outputs, backbone_outputs, is_training=False
        ) if hasattr(self, "validate_data") else None
        actions = actions.float()

        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        bsize = normalized_input["state"].shape[0]
        device = normalized_input["state"].device

        for k in [
            "eagle_pixel_values",
            "eagle_image_sizes",
            "pixel_values",
            "image_sizes",
        ]:
            if k in normalized_input and isinstance(normalized_input[k], list):
                if len(normalized_input[k]) > 0 and isinstance(
                    normalized_input[k][0], torch.Tensor
                ):
                    normalized_input[k] = torch.stack(normalized_input[k]).to(device)
                else:
                    normalized_input[k] = torch.tensor(
                        normalized_input[k], device=device
                    )

        if "eagle_pixel_values" in normalized_input:
            forward_inputs["eagle_pixel_values"] = normalized_input[
                "eagle_pixel_values"
            ].reshape(
                bsize,
                self.image_nums,
                *normalized_input["eagle_pixel_values"].shape[-3:],
            )
        if "eagle_image_sizes" in normalized_input:
            forward_inputs["eagle_image_sizes"] = normalized_input[
                "eagle_image_sizes"
            ].reshape(
                bsize,
                self.image_nums,
                *normalized_input["eagle_image_sizes"].shape[-1:],
            )
        if "pixel_values" in normalized_input:
            forward_inputs["pixel_values"] = normalized_input["pixel_values"].reshape(
                bsize, self.image_nums, *normalized_input["pixel_values"].shape[-3:]
            )
        if "image_sizes" in normalized_input:
            forward_inputs["image_sizes"] = normalized_input["image_sizes"].reshape(
                bsize, self.image_nums, *normalized_input["image_sizes"].shape[-1:]
            )
        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        for key in ["input_ids", "attention_mask", "pixel_values", "image_sizes"]:
            if key in result["forward_inputs"]:
                val = result["forward_inputs"][key]
                if isinstance(val, list):
                    result["forward_inputs"][key] = (
                        torch.stack(val)
                        if len(val) > 0 and torch.is_tensor(val[0])
                        else torch.tensor(val)
                    )

            eagle_key = f"eagle_{key}"
            if eagle_key in result["forward_inputs"]:
                val = result["forward_inputs"][eagle_key]
                if isinstance(val, list):
                    result["forward_inputs"][eagle_key] = (
                        torch.stack(val)
                        if len(val) > 0 and torch.is_tensor(val[0])
                        else torch.tensor(val)
                    )
        return actions, result

    def _get_unnormalized_action(
        self,
        normalized_action: torch.Tensor,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()}, state=state)

    def _load_metadata(self, exp_cfg_dir: Path):
        metadata_path = exp_cfg_dir / "metadata.json"
        if not metadata_path.exists():
            logging.info(
                f"Metadata file not found at {metadata_path}. "
                "Inferring from modality_config..."
            )

            tag_value = self.embodiment_tag.value
            if self._modality_config is None or tag_value not in self._modality_config:
                logging.info(
                    "Modality config is missing or does not contain tag "
                    f"{self.embodiment_tag.value}. Attempting to infer "
                    "valid_action_dim and image_nums from config attributes."
                )
                self.valid_action_dim = getattr(
                    self.config, "max_action_dim", getattr(self.config, "action_dim", 7)
                )
                self.image_nums = getattr(self.config, "image_nums", 1)
                logging.info(
                    "Inferred fallback: "
                    f"valid_action_dim={self.valid_action_dim}, "
                    f"image_nums={self.image_nums}"
                )
                return

            current_modality = self._modality_config[tag_value]

            valid_action_dim = 0
            if "action" in current_modality:
                action_modality_cfg = current_modality["action"]
                if (
                    hasattr(action_modality_cfg, "dim_map")
                    and action_modality_cfg.dim_map
                ):
                    for dim_val in action_modality_cfg.dim_map.values():
                        valid_action_dim += dim_val
                elif (
                    hasattr(action_modality_cfg, "modality_keys")
                    and action_modality_cfg.modality_keys
                ):
                    valid_action_dim = getattr(self.config, "max_action_dim", 29)
                elif isinstance(action_modality_cfg, dict):
                    if action_modality_cfg.get("dim_map"):
                        for dim_val in action_modality_cfg["dim_map"].values():
                            valid_action_dim += dim_val
                    elif "dim" in action_modality_cfg:
                        valid_action_dim = action_modality_cfg["dim"]
                    else:
                        valid_action_dim = getattr(self.config, "max_action_dim", 29)
                else:
                    valid_action_dim = getattr(self.config, "max_action_dim", 29)

            self.valid_action_dim = valid_action_dim

            if "video" in current_modality:
                video_modality_cfg = current_modality["video"]
                if (
                    hasattr(video_modality_cfg, "modality_keys")
                    and video_modality_cfg.modality_keys
                ):
                    self.image_nums = len(video_modality_cfg.modality_keys)
                elif (
                    isinstance(video_modality_cfg, dict)
                    and "modality_keys" in video_modality_cfg
                ):
                    self.image_nums = len(video_modality_cfg["modality_keys"])
                else:
                    self.image_nums = 1
            else:
                self.image_nums = 1

            logging.info(
                "Inferred from modality_config: "
                f"valid_action_dim={self.valid_action_dim}, "
                f"image_nums={self.image_nums}"
            )
            return

        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}"
            )

        self.metadata = metadata_dict
        if hasattr(self._modality_transform, "set_metadata"):
            self._modality_transform.set_metadata(metadata_dict)

        valid_action_dim = 0
        action_mods = self.metadata.get("modalities", {}).get("action", {})
        for v in action_mods.values():
            shape = v.get("shape", [0]) if isinstance(v, dict) else [0]
            valid_action_dim += shape[0] if len(shape) > 0 else 0
        self.valid_action_dim = valid_action_dim

        video_mods = self.metadata.get("modalities", {}).get("video", {})
        self.image_nums = len(video_mods)
