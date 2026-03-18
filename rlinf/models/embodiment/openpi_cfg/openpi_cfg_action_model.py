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


_VALID_GUIDANCE_TYPES = ("positive", "negative", "no_guide")


def compute_cfg_routing_masks(
    advantage: torch.Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    positive_unconditional_prob: float,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sample routing masks for CFG training.

    Args:
        advantage: Boolean tensor where True marks positive samples.
        positive_only_conditional: Route only positive samples to the
            conditional branch when True.
        unconditional_prob: Dropout probability for the legacy CFG training
            mode that routes both positive and negative samples conditionally.
        positive_unconditional_prob: Dropout probability applied only to
            positive samples when ``positive_only_conditional`` is enabled.
        random_values: Optional pre-sampled uniform noise in ``[0, 1)`` used to
            make routing deterministic in tests.

    Returns:
        Dictionary of boolean masks describing how the batch is routed.
    """
    advantage = advantage.to(dtype=torch.bool)
    batch_size = advantage.shape[0]
    device = advantage.device

    if random_values is None:
        random_values = torch.rand(batch_size, device=device)
    else:
        random_values = random_values.to(device=device)

    positive_mask = advantage
    negative_mask = ~positive_mask

    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (
            random_values > positive_unconditional_prob
        )
        negative_conditional_mask = torch.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    positive_unconditional_mask = positive_mask & ~positive_conditional_mask
    negative_unconditional_mask = negative_mask & ~negative_conditional_mask

    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_unconditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_unconditional_mask,
    }


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
    positive_only_conditional: bool = False
    positive_unconditional_prob: float = 0.0
    positive_probe_diagnostics: bool = False

    def __post_init__(self):
        if self.guidance_type not in _VALID_GUIDANCE_TYPES:
            raise ValueError(
                f"guidance_type must be one of {_VALID_GUIDANCE_TYPES}, "
                f"got '{self.guidance_type}'"
            )
        if not 0.0 <= self.unconditional_prob <= 1.0:
            raise ValueError(
                f"unconditional_prob must be in [0, 1], got {self.unconditional_prob}"
            )
        if not 0.0 <= self.positive_unconditional_prob <= 1.0:
            raise ValueError(
                "positive_unconditional_prob must be in [0, 1], "
                f"got {self.positive_unconditional_prob}"
            )
        if not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValueError(
                f"num_steps must be a positive integer, got {self.num_steps}"
            )


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
        if getattr(self.config, "noise_method", None) == "flow_noise":
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

        # Store reference to tokenizer transform for guidance prompt processing
        # instead of relying on a brittle hardcoded index
        self._tokenize_transform = None
        for t in self._input_transform.transforms:
            if type(t).__name__ in ("TokenizePrompt", "TokenizePromptWithGuidance"):
                self._tokenize_transform = t
                break
        if self._tokenize_transform is None:
            raise ValueError(
                "Cannot find TokenizePrompt or TokenizePromptWithGuidance "
                "in input transforms"
            )

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
                    lambda x: x.transpose(1, 2, 0) if len(x.shape) == 3 else x,
                    sample,
                )
            if first_process:
                sample["prompt"] = obs["prompt"][i]
                positive_guidance_prompt = obs["positive_guidance_prompt"][i]
                positive_guidance_dict = self._tokenize_transform(
                    {"prompt": positive_guidance_prompt}
                )
                negative_guidance_prompt = obs["negative_guidance_prompt"][i]
                negative_guidance_dict = self._tokenize_transform(
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

    def default_forward(self, **kwargs):
        """Default forward — delegates to forward()."""
        return self.forward(**kwargs)

    def _compute_flow_losses(
        self,
        images,
        img_masks,
        state,
        actions,
        lang_tokens,
        lang_masks,
        device,
        time=None,
        noise=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute flow loss and detached per-sample loss for a language route."""
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        actions = actions.to(device, dtype=torch.float32)

        if time is None:
            time = self.sample_time(actions.shape[0], device)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )
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
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

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

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        per_element_loss = F.mse_loss(u_t, v_t, reduction="none")
        flow_loss = per_element_loss.mean()
        per_sample_loss = per_element_loss.detach().mean(dim=(-1, -2))
        return flow_loss, per_sample_loss

    @staticmethod
    def _masked_loss_sum(per_sample_loss: torch.Tensor, mask: torch.Tensor) -> float:
        """Return the summed loss over a boolean mask."""
        if mask.numel() == 0 or not torch.any(mask):
            return 0.0
        return (per_sample_loss * mask.float()).sum().item()

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
        if "advantage" in data:
            advantage = data["advantage"].to(device)
        elif "advantages" in data:
            advantage = data["advantages"].to(device)
        else:
            raise ValueError(
                "Missing 'advantage' field in data. "
                "Please run compute_advantages.py first to generate "
                "meta/advantages.parquet for your dataset."
            )
        advantage = advantage.to(dtype=torch.bool)
        routing = compute_cfg_routing_masks(
            advantage,
            positive_only_conditional=self.config.positive_only_conditional,
            unconditional_prob=self.config.unconditional_prob,
            positive_unconditional_prob=self.config.positive_unconditional_prob,
        )
        positive_mask = routing["positive_mask"]
        negative_mask = routing["negative_mask"]
        conditional_mask = routing["conditional_mask"]
        positive_conditional_mask = routing["positive_conditional_mask"]
        positive_unconditional_mask = routing["positive_unconditional_mask"]
        negative_conditional_mask = routing["negative_conditional_mask"]
        negative_unconditional_mask = routing["negative_unconditional_mask"]

        if self.config.positive_only_conditional:
            final_lang_tokens = torch.where(
                positive_conditional_mask.unsqueeze(-1),
                positive_guidance_lang_tokens,
                lang_tokens,
            )
            final_lang_masks = torch.where(
                positive_conditional_mask.unsqueeze(-1),
                positive_guidance_lang_masks,
                lang_masks,
            )
        else:
            guidance_lang_tokens = torch.where(
                positive_mask.unsqueeze(-1),
                positive_guidance_lang_tokens,
                negative_guidance_lang_tokens,
            )
            guidance_lang_masks = torch.where(
                positive_mask.unsqueeze(-1),
                positive_guidance_lang_masks,
                negative_guidance_lang_masks,
            )
            final_lang_tokens = torch.where(
                conditional_mask.unsqueeze(-1),
                guidance_lang_tokens,
                lang_tokens,
            )
            final_lang_masks = torch.where(
                conditional_mask.unsqueeze(-1),
                guidance_lang_masks,
                lang_masks,
            )

        # ========== 4. Compute selected-route loss ==========
        actions = actions.to(device, dtype=torch.float32)
        if kwargs.get("time", None) is not None:
            time = kwargs.get("time")
        else:
            time = self.sample_time(actions.shape[0], device)

        if kwargs.get("noise", None) is not None:
            noise = kwargs.get("noise")
        else:
            noise = self.sample_noise(actions.shape, device)
        flow_loss, per_sample_loss = self._compute_flow_losses(
            images=images,
            img_masks=img_masks,
            state=state,
            actions=actions,
            lang_tokens=final_lang_tokens,
            lang_masks=final_lang_masks,
            device=device,
            time=time,
            noise=noise,
        )

        conditional_count = conditional_mask.sum().item()
        unconditional_count = (~conditional_mask).sum().item()
        positive_label_count = positive_mask.sum().item()
        negative_label_count = negative_mask.sum().item()

        metrics = {
            "conditional_count": conditional_count,
            "unconditional_count": unconditional_count,
            "conditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, conditional_mask
            ),
            "unconditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, ~conditional_mask
            ),
            "positive_label_count": positive_label_count,
            "negative_label_count": negative_label_count,
            "positive_conditional_count": positive_conditional_mask.sum().item(),
            "positive_unconditional_count": positive_unconditional_mask.sum().item(),
            "negative_conditional_count": negative_conditional_mask.sum().item(),
            "negative_unconditional_count": negative_unconditional_mask.sum().item(),
            "positive_conditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, positive_conditional_mask
            ),
            "positive_unconditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, positive_unconditional_mask
            ),
            "negative_conditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, negative_conditional_mask
            ),
            "negative_unconditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, negative_unconditional_mask
            ),
        }
        # TODO: zhihao: remove this future
        if (
            self.config.positive_only_conditional
            and self.config.positive_probe_diagnostics
            and positive_label_count > 0
        ):
            probe_images = [img[positive_mask] for img in images]
            probe_img_masks = [img_mask[positive_mask] for img_mask in img_masks]
            probe_state = state[positive_mask]
            probe_actions = actions[positive_mask]
            probe_time = time[positive_mask]
            probe_noise = noise[positive_mask]
            probe_positive_tokens = positive_guidance_lang_tokens[positive_mask]
            probe_positive_masks = positive_guidance_lang_masks[positive_mask]
            probe_uncond_tokens = lang_tokens[positive_mask]
            probe_uncond_masks = lang_masks[positive_mask]

            checkpointing_enabled = self.gradient_checkpointing_enabled
            self.gradient_checkpointing_enabled = False
            try:
                with torch.no_grad():
                    _, probe_positive_cond_loss = self._compute_flow_losses(
                        images=probe_images,
                        img_masks=probe_img_masks,
                        state=probe_state,
                        actions=probe_actions,
                        lang_tokens=probe_positive_tokens,
                        lang_masks=probe_positive_masks,
                        device=device,
                        time=probe_time,
                        noise=probe_noise,
                    )
                    _, probe_positive_uncond_loss = self._compute_flow_losses(
                        images=probe_images,
                        img_masks=probe_img_masks,
                        state=probe_state,
                        actions=probe_actions,
                        lang_tokens=probe_uncond_tokens,
                        lang_masks=probe_uncond_masks,
                        device=device,
                        time=probe_time,
                        noise=probe_noise,
                    )
            finally:
                self.gradient_checkpointing_enabled = checkpointing_enabled

            metrics.update(
                {
                    "positive_probe_count": positive_label_count,
                    "positive_probe_conditional_loss_sum": probe_positive_cond_loss.sum().item(),
                    "positive_probe_unconditional_loss_sum": probe_positive_uncond_loss.sum().item(),
                }
            )

        return flow_loss, metrics

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        positive_guidance_prompt = [
            f"{desc}\nAdvantage: positive" for desc in env_obs["task_descriptions"]
        ]
        negative_guidance_prompt = [
            f"{desc}\nAdvantage: negative" for desc in env_obs["task_descriptions"]
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
        guidance_type = self.config.guidance_type
        if self.config.positive_only_conditional and guidance_type == "negative":
            raise ValueError(
                "guidance_type='negative' is incompatible with "
                "positive_only_conditional training."
            )
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
