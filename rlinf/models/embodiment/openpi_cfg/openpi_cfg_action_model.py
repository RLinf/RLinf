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
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import jax
import numpy as np
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.shared.array_typing as at
import torch
import torch.nn.functional as F
from flax import struct
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.model import Observation as Obs
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


@at.typecheck
@struct.dataclass
class Observation(Obs[ArrayT]):
    tokenized_positive_guidance_prompt: ArrayT | None = None  # noqa: F722
    tokenized_positive_guidance_prompt_mask: ArrayT | None = None  # noqa: F722
    tokenized_negative_guidance_prompt: ArrayT | None = None  # noqa: F722
    tokenized_negative_guidance_prompt_mask: ArrayT | None = None  # noqa: F722

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )
        if ("tokenized_positive_guidance_prompt" in data) != (
            "tokenized_positive_guidance_prompt_mask" in data
        ):
            raise ValueError(
                "tokenized_positive_guidance_prompt and tokenized_positive_guidance_prompt_mask must be provided together."
            )
        if ("tokenized_negative_guidance_prompt" in data) != (
            "tokenized_negative_guidance_prompt_mask" in data
        ):
            raise ValueError(
                "tokenized_negative_guidance_prompt and tokenized_negative_guidance_prompt_mask must be provided together."
            )

        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = (
                    data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                )
            elif (
                hasattr(data["image"][key], "dtype")
                and data["image"][key].dtype == torch.uint8
            ):
                data["image"][key] = (
                    data["image"][key].to(torch.float32).permute(0, 3, 1, 2)
                    / 255.0
                    * 2.0
                    - 1.0
                )

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            tokenized_positive_guidance_prompt=data.get(
                "tokenized_positive_guidance_prompt"
            ),
            tokenized_positive_guidance_prompt_mask=data.get(
                "tokenized_positive_guidance_prompt_mask"
            ),
            tokenized_negative_guidance_prompt=data.get(
                "tokenized_negative_guidance_prompt"
            ),
            tokenized_negative_guidance_prompt_mask=data.get(
                "tokenized_negative_guidance_prompt_mask"
            ),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = (
        "pi0_libero"  # pi0_libero, pi05_libero, pi0_metaworld, pi05_metaworld
    )
    num_images_in_input: int = 2  # number of images in input
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False

    cfgrl_guidance_scale: float = 1.0
    unconditional_prob: float = 0.3
    guidance_type: str = "positive"  # "positive", "negative", "no_guide"


class OpenPi0ForCFGActionPrediction(BasePolicy, PI0Pytorch):
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
        self.global_step = 0
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
            inputs.pop("positive_guidance_prompt")
            inputs.pop("negative_guidance_prompt")
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
                positive_guidance_prompt = obs["positive_guidance_prompt"][i]
                # TODO: This is written roughly, refine it later
                positive_guidance_dict = self._input_transform.transforms[5](
                    {"prompt": positive_guidance_prompt}
                )
                negative_guidance_prompt = obs["negative_guidance_prompt"][i]
                negative_guidance_dict = self._input_transform.transforms[5](
                    {"prompt": negative_guidance_prompt}
                )
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            if first_process:
                transformed_sample.update(
                    {
                        "tokenized_positive_guidance_prompt": positive_guidance_dict[
                            "tokenized_prompt"
                        ],
                        "tokenized_positive_guidance_prompt_mask": positive_guidance_dict[
                            "tokenized_prompt_mask"
                        ],
                        "tokenized_negative_guidance_prompt": negative_guidance_dict[
                            "tokenized_prompt"
                        ],
                        "tokenized_negative_guidance_prompt_mask": negative_guidance_dict[
                            "tokenized_prompt_mask"
                        ],
                    }
                )
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
            inputs["tokenized_positive_guidance_prompt"] = obs[
                "tokenized_positive_guidance_prompt"
            ]
            inputs["tokenized_positive_guidance_prompt_mask"] = obs[
                "tokenized_positive_guidance_prompt_mask"
            ]
            inputs["tokenized_negative_guidance_prompt"] = obs[
                "tokenized_negative_guidance_prompt"
            ]
            inputs["tokenized_negative_guidance_prompt_mask"] = obs[
                "tokenized_negative_guidance_prompt_mask"
            ]
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
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """CFGRL forward - unified for both SFT and CFGRL training.

        Supports two data formats:
        1. SFT mode: data contains "observation" and "actions"
        2. CFGRL mode: data contains "dones", "advantages", "raw_actions", etc.
        """
        # Determine whether this is SFT mode or CFGRL mode
        is_sft_mode = "observation" in data and "actions" in data

        # ========== 1. Get observation and actions ==========
        if is_sft_mode:
            observation = data["observation"]  # Already a CFGObservation object
            actions = data["actions"]
            device = actions.device
        else:
            device = data["dones"].device
            observation = self.input_transform(data, transpose=False)
            observation = Observation.from_dict(observation)
            actions = data["raw_actions"]

        # ========== 2. Unified observation preprocessing ==========
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            positive_guidance_lang_tokens,
            positive_guidance_lang_masks,
            negative_guidance_lang_tokens,
            negative_guidance_lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True)

        # ========== 3. Select guidance tokens ==========
        positive_guidance_count = 0
        negative_guidance_count = 0
        if is_sft_mode:
            # CFG mode: select positive or negative guidance based on advantage
            if "advantage" not in data:
                raise ValueError(
                    "Missing 'advantage' field in data. "
                    "Please run compute_advantages.py first to generate "
                    "meta/advantages.parquet for your dataset."
                )

            advantage = data["advantage"].to(device)
            advantage_expanded = advantage.unsqueeze(-1)  # [batch_size, 1]
            guidance_lang_tokens = torch.where(
                advantage_expanded,
                positive_guidance_lang_tokens,
                negative_guidance_lang_tokens,
            )
            guidance_lang_masks = torch.where(
                advantage_expanded,
                positive_guidance_lang_masks,
                negative_guidance_lang_masks,
            )
            positive_guidance_count = advantage.sum().item()
            negative_guidance_count = advantage.numel() - positive_guidance_count

        else:
            # CFGRL mode: select positive or negative guidance based on advantages
            advantages = data["advantages"].to(device)
            advantages_expanded = advantages.unsqueeze(-1)  # [batch_size, 1]
            guidance_lang_tokens = torch.where(
                advantages_expanded,
                positive_guidance_lang_tokens,
                negative_guidance_lang_tokens,
            )
            guidance_lang_masks = torch.where(
                advantages_expanded,
                positive_guidance_lang_masks,
                negative_guidance_lang_masks,
            )
            # Count positive/negative guidance usage
            positive_guidance_count = advantages.sum().item()
            negative_guidance_count = advantages.numel() - positive_guidance_count

        # ========== 4. Decide whether to use guidance based on unconditional_prob ==========
        use_guidance = random.random() > self.config.unconditional_prob

        # ========== 5. Align device ==========
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        actions = actions.to(device, dtype=torch.float32)
        if kwargs.get("time", None) is not None:
            time = kwargs.get("time")
        else:
            time = self.sample_time(actions.shape[0], device)

        if kwargs.get("noise", None) is not None:
            noise = kwargs.get("noise")
        else:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        if use_guidance:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, guidance_lang_tokens, guidance_lang_masks
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )
        # [i.sum() for i in img_masks]  [i.sum() for i in images]
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        flow_loss = F.mse_loss(u_t, v_t, reduction="none")
        # flow_loss = flow_loss[:, : self.config.action_chunk, : self.config.action_env_dim].mean()
        flow_loss = flow_loss.mean()
        flow_loss_value = flow_loss.detach().item()
        if use_guidance:
            conditional_count = 1
            unconditional_count = 0
            conditional_loss = flow_loss_value
            unconditional_loss = 0.0
        else:
            conditional_count = 0
            unconditional_count = 1
            conditional_loss = 0.0
            unconditional_loss = flow_loss_value

        return flow_loss, {
            "conditional_count": conditional_count,
            "unconditional_count": unconditional_count,
            "conditional_loss": conditional_loss,
            "unconditional_loss": unconditional_loss,
            "positive_guidance_count": positive_guidance_count,
            "negative_guidance_count": negative_guidance_count,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        positive_guidance_prompt = [
            f"[POSITIVE][POSITIVE]\nTask: {desc}"
            for desc in env_obs["task_descriptions"]
        ]
        negative_guidance_prompt = [
            f"[NEGATIVE][NEGATIVE]\nTask: {desc}"
            for desc in env_obs["task_descriptions"]
        ]
        processed_obs["positive_guidance_prompt"] = positive_guidance_prompt
        processed_obs["negative_guidance_prompt"] = negative_guidance_prompt
        # state observation
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
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
        return_obs=True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        # observation = _model.Observation.from_dict(processed_obs)
        observation = Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation,
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()
        forward_inputs = {
            "raw_actions": outputs[
                "actions"
            ],  # raw_actions refers to actions not yet processed by output_transform
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            "tokenized_positive_guidance_prompt": processed_obs[
                "tokenized_positive_guidance_prompt"
            ],
            "tokenized_positive_guidance_prompt_mask": processed_obs[
                "tokenized_positive_guidance_prompt_mask"
            ],
            "tokenized_negative_guidance_prompt": processed_obs[
                "tokenized_negative_guidance_prompt"
            ],
            "tokenized_negative_guidance_prompt_mask": processed_obs[
                "tokenized_negative_guidance_prompt_mask"
            ],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        forward_inputs.pop("positive_guidance_prompt", None)
        forward_inputs.pop("negative_guidance_prompt", None)
        result = {
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
    ) -> dict:
        """
        v = (1 - w) * v_uncond + w * v_cond (when guidance_type != "no_guide")
        v = v_uncond (when guidance_type == "no_guide")
        """
        # breakpoint()
        guidance_type = self.config.guidance_type
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if (
            noise is None
        ):  # action horizon limits the maximum step count. To exceed 10 (default), SFT must be re-run
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            positive_guidance_lang_tokens,
            positive_guidance_lang_masks,
            negative_guidance_lang_tokens,
            negative_guidance_lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs_uncond, prefix_pad_masks_uncond, prefix_att_masks_uncond = (
            self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        )
        prefix_att_2d_masks_uncond = make_att_2d_masks(
            prefix_pad_masks_uncond, prefix_att_masks_uncond
        )
        prefix_position_ids_uncond = torch.cumsum(prefix_pad_masks_uncond, dim=1) - 1
        prefix_att_2d_masks_4d_uncond = self._prepare_attention_masks_4d(
            prefix_att_2d_masks_uncond
        )

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values_uncond = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d_uncond,
            position_ids=prefix_position_ids_uncond,
            past_key_values=None,
            inputs_embeds=[prefix_embs_uncond, None],
            use_cache=True,
        )

        # Only compute conditional when guidance_type is not "no_guide"
        if guidance_type != "no_guide":
            if guidance_type == "positive":
                guidance_lang_tokens = positive_guidance_lang_tokens
                guidance_lang_masks = positive_guidance_lang_masks
            elif guidance_type == "negative":
                guidance_lang_tokens = negative_guidance_lang_tokens
                guidance_lang_masks = negative_guidance_lang_masks
            else:
                raise ValueError(f"Unknown guidance_type: {guidance_type}")

            prefix_embs_cond, prefix_pad_masks_cond, prefix_att_masks_cond = (
                self.embed_prefix(
                    images, img_masks, guidance_lang_tokens, guidance_lang_masks
                )
            )

            prefix_att_2d_masks_cond = make_att_2d_masks(
                prefix_pad_masks_cond, prefix_att_masks_cond
            )
            prefix_position_ids_cond = torch.cumsum(prefix_pad_masks_cond, dim=1) - 1
            prefix_att_2d_masks_4d_cond = self._prepare_attention_masks_4d(
                prefix_att_2d_masks_cond
            )

            _, past_key_values_cond = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d_cond,
                position_ids=prefix_position_ids_cond,
                past_key_values=None,
                inputs_embeds=[prefix_embs_cond, None],
                use_cache=True,
            )
        else:
            # When guidance_type == "no_guide", no need to compute conditional
            prefix_pad_masks_cond = None
            past_key_values_cond = None

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t_uncond = self.denoise_step(
                state,
                prefix_pad_masks_uncond,
                past_key_values_uncond,
                x_t,
                expanded_time,
            )

            if guidance_type == "no_guide":
                # Use unconditional only
                v_t = v_t_uncond
            else:
                # Use conditional guidance
                v_t_cond = self.denoise_step(
                    state,
                    prefix_pad_masks_cond,
                    past_key_values_cond,
                    x_t,
                    expanded_time,
                )
                v_t = (
                    1 - self.config.cfgrl_guidance_scale
                ) * v_t_uncond + self.config.cfgrl_guidance_scale * v_t_cond

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        return {"actions": x_t}

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

    def preprocess_for_train(self, data):
        return data

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        part_observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=train
        )
        setattr(
            part_observation,
            "tokenized_positive_guidance_prompt",
            observation.tokenized_positive_guidance_prompt,
        )
        setattr(
            part_observation,
            "tokenized_positive_guidance_prompt_mask",
            observation.tokenized_positive_guidance_prompt_mask,
        )
        setattr(
            part_observation,
            "tokenized_negative_guidance_prompt",
            observation.tokenized_negative_guidance_prompt,
        )
        setattr(
            part_observation,
            "tokenized_negative_guidance_prompt_mask",
            observation.tokenized_negative_guidance_prompt_mask,
        )
        # Return tuple consistent with parent class interface, but including positive and negative guidance tokens
        return (
            list(part_observation.images.values()),
            list(part_observation.image_masks.values()),
            part_observation.tokenized_prompt,
            part_observation.tokenized_prompt_mask,
            part_observation.tokenized_positive_guidance_prompt,
            part_observation.tokenized_positive_guidance_prompt_mask,
            part_observation.tokenized_negative_guidance_prompt,
            part_observation.tokenized_negative_guidance_prompt_mask,
            part_observation.state,
        )
