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

"""LingBot-VLA 2.0 flow-SDE PPO with an explicit latent trajectory likelihood."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger


def flow_sde_transition(
    x: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    delta: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP32 Gaussian transition used by sampling and PPO replay.

    At t=1 the diffusion denominator uses the next time, as in RLinf's
    LingBot-VLA/OpenPI flow-SDE. Every step has positive variance, including
    the final step. Scores are densities of the complete latent path.
    """
    x, velocity = x.float(), velocity.float()
    denominator = torch.where(time == 1, delta, 1 - time)
    sigma_squared = noise_level**2 * time / denominator
    mean = (
        x
        - delta * velocity
        - sigma_squared * delta / (2 * time) * (x + (1 - time) * velocity)
    )
    std = torch.sqrt(delta * sigma_squared)
    return mean, std


def gaussian_logprob(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Score a detached continuous sample without reducing latent dimensions."""
    return torch.distributions.Normal(mean.float(), std.float()).log_prob(
        sample.float()
    )


class LingbotVLAV2ActionModel(nn.Module, BasePolicy):
    """Adapt the installed V2 policy and FeatureTransform to RLinf PPO.

    Training uses a stochastic transition at every denoising step. PPO sums
    their log densities over all model dimensions; the deterministic robot
    transform then selects the 14 RoboTwin controls. Evaluation uses the ODE.
    """

    _no_split_modules = ["Qwen2DecoderLayer", "Qwen3VLTextDecoderLayer"]
    _no_split_names = [
        "visual",
        "embed_tokens",
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "time_mlp_in",
        "time_mlp_out",
        "expert_final_norm",
        "value_head",
    ]

    def __init__(self, config: DictConfig, torch_dtype: torch.dtype):
        super().__init__()
        from lingbotvla.checkpoint_metadata import (
            build_checkpoint_config,
            read_checkpoint_config,
        )
        from lingbotvla.data.vla_data.utils import FeatureTransform
        from lingbotvla.distributed.parallel_state import (
            get_parallel_state,
            init_parallel_state,
        )
        from lingbotvla.models import build_processor
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
            LingbotVlaV2Policy,
        )
        from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
            apply_lingbot_qwen2_patch,
        )
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
            apply_lingbot_qwen3_vl_patch,
        )
        from safetensors.torch import load_model
        from transformers.modeling_utils import load_sharded_checkpoint

        self.config = config
        self.torch_dtype = torch_dtype
        self.num_steps = int(config.num_steps)
        self.forward_micro_batch_size = int(config.forward_micro_batch_size)
        if self.forward_micro_batch_size < 1:
            raise ValueError("forward_micro_batch_size must be positive")
        self.noise_level = float(config.noise_level)
        if self.num_steps < 2 or not self.noise_level > 0:
            raise ValueError("V2 flow-SDE requires num_steps >= 2 and noise_level > 0")
        if config.rl_trainable_scope != "action_expert":
            raise ValueError("V2 currently supports rl_trainable_scope=action_expert")
        if config.noise_method != "flow_sde" or not config.joint_logprob:
            raise ValueError("V2 PPO requires flow_sde and joint_logprob=true")
        if config.is_lora:
            raise ValueError("V2 LoRA is not supported")
        if config.lingbotvlav2.data_parallel_backend != "fsdp2":
            raise ValueError("V2 currently requires the FSDP2 training backend")
        if torch.distributed.is_initialized():
            # RLinf owns the process group and parameter sharding. The native
            # MoE runtime still needs a matching topology to disable EP/TP.
            world_size = torch.distributed.get_world_size()
            init_parallel_state(
                dp_size=world_size,
                dp_shard_size=world_size,
                dp_mode=config.lingbotvlav2.data_parallel_backend,
                device_type="cuda",
            )
            state = get_parallel_state()
            if (
                state.dp_size != world_size
                or state.dp_shard_size != world_size
                or state.ep_enabled
                or state.tp_enabled
                or state.pp_enabled
                or state.sp_enabled
            ):
                raise ValueError("V2 parallel state must match RLinf data parallelism")

        weights = Path(config.model_path).expanduser()
        model_config = build_checkpoint_config(
            read_checkpoint_config(weights), config.tokenizer_path
        )
        model_config.attention_implementation = (
            config.lingbotvlav2.attention_implementation
        )
        model_config.use_cache = True
        # Fixed-grid caches are batch-size dependent; actor micro-batches and
        # rollout batches need not have the same size.
        model_config.precompute_grid_thw = False
        model_config.use_qwen3_fixed_grid_cache = False
        model_config.bias_update_speed = 0.0
        model_config.train_expert_only = True
        model_config.freeze_vision_encoder = True
        self.horizon = int(model_config.n_action_steps)
        self.latent_dim = int(model_config.max_action_dim)
        if config.num_action_chunks != self.horizon:
            raise ValueError("num_action_chunks must match the V2 checkpoint horizon")
        if config.logprob_dim != self.latent_dim:
            raise ValueError("logprob_dim must cover every V2 latent action dimension")
        if config.action_dim != 14 or model_config.max_state_dim != 55:
            raise ValueError(
                "This V2 adapter requires RoboTwin 14D controls and 55D state"
            )

        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        self.vla_model = LingbotVlaV2Policy(model_config, eval=True)
        if (weights / "model.safetensors.index.json").is_file():
            load_sharded_checkpoint(
                self.vla_model, str(weights), strict=True, prefer_safe=True
            )
        elif (weights / "model.safetensors").is_file():
            load_model(self.vla_model, str(weights / "model.safetensors"), strict=True)
        else:
            raise FileNotFoundError(f"No safetensors checkpoint in {weights}")
        self.vla_model.to(dtype=torch_dtype)
        # No-grad inference kernels can have different numerical reductions.
        # The source package exposes a common differentiable MoE backend.
        self.vla_model.set_moe_forward_backend("differentiable")
        self.vla_model.requires_grad_(False)
        for name, param in self.vla_model.named_parameters():
            if any(
                part in name
                for part in (
                    ".qwen_expert.",
                    ".state_proj.",
                    ".action_in_proj.",
                    ".action_out_proj.",
                    ".action_time_mlp_",
                    ".time_mlp_",
                )
            ):
                param.requires_grad_(True)
        self.value_head = None
        if config.add_value_head:
            self.value_head = ValueHead(
                input_dim=model_config.hidden_size + model_config.max_state_dim,
                hidden_sizes=list(config.value_head.hidden_sizes),
                activation=config.value_head.activation,
                output_dim=1,
                bias_last=True,
            ).to(dtype=torch_dtype)

        with (
            Path(config.lingbotvlav2.training_config_path).expanduser().open() as source
        ):
            data_config = dict(yaml.safe_load(source)["data"])
        for key in ("joints", "norm_type"):
            data_config[key] = [
                str(item) if isinstance(item, dict) else item
                for item in data_config[key]
            ]
        self.image_size = int(data_config["img_size"])
        self.processor = build_processor(config.tokenizer_path)
        self.transform = FeatureTransform(
            config.lingbotvlav2.robot_config_path,
            SimpleNamespace(**data_config),
            model_config,
            self.processor,
            chunk_size=self.horizon,
            norm_stats_path=config.lingbotvlav2.stats_path,
        )
        if self.transform.actions_convert_from_state:
            raise ValueError("RoboTwin V2 needs direct qpos action commands")
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.rsplit(".", 1)[-1]
        self.vla_model.model.qwenvl_with_expert.qwen_expert.model.norm._fsdp_wrap_name = "expert_final_norm"
        get_logger().info(
            "LingBot-VLA V2 loaded: trainable=%d total=%d horizon=%d latent_dim=%d",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            sum(p.numel() for p in self.parameters()),
            self.horizon,
            self.latent_dim,
        )

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Recompute experts inside each decoder's FSDP ownership boundary."""
        # A checkpoint around the entire velocity call re-enters shared FSDP
        # modules while their backward hooks own sharded parameters.
        self.vla_model.set_moe_activation_checkpointing(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vla_model.model.qwenvl_with_expert.qwenvl.eval()
        return self

    def prepare_observations(self, env_obs: dict) -> dict[str, torch.Tensor]:
        """Use V2's training transform once, retaining batch-major replay inputs."""
        from torchvision.transforms.v2 import Resize

        states = env_obs["states"]
        wrists = env_obs["wrist_images"]
        if states.ndim != 2 or states.shape[-1] != 14:
            raise ValueError("RoboTwin states must be [batch, 14]")
        if wrists.shape[1] != 2:
            raise ValueError("V2 RoboTwin requires both wrist cameras")
        resize = Resize((self.image_size, self.image_size))
        prepared = []
        for index in range(len(states)):
            raw = {
                "task": env_obs["task_descriptions"][index],
                "observation.state": states[index].detach().float().cpu(),
            }
            for key, frame in (
                ("observation.images.cam_high", env_obs["main_images"][index]),
                ("observation.images.cam_left_wrist", wrists[index, 0]),
                ("observation.images.cam_right_wrist", wrists[index, 1]),
            ):
                frame = torch.as_tensor(frame).detach().cpu()
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("RoboTwin images must be HWC RGB")
                raw[key] = resize(frame.permute(2, 0, 1).float())
            prepared.append(self.transform.apply(raw, policy_eval=True))
        keys = (
            "images",
            "img_masks",
            "lang_tokens",
            "lang_masks",
            "state",
            "image_grid_thw",
            "state_joint_mask",
            "action_joint_mask",
        )
        return {key: torch.stack([item[key] for item in prepared]) for key in keys}

    def _device_inputs(self, inputs: dict) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        return {
            key: value.to(
                device=device,
                dtype=self.torch_dtype if key in {"images", "state"} else value.dtype,
            )
            for key, value in inputs.items()
        }

    def _prefix(self, inputs: dict):
        flow = self.vla_model.model
        # All prefix weights are frozen in action_expert scope, including
        # depth/video queries. Keep those queries in the attention context.
        with torch.no_grad():
            prefix, padding, attention, positions, visual_mask, deepstack = (
                flow.embed_prefix(
                    inputs["images"],
                    inputs["img_masks"],
                    inputs["lang_tokens"],
                    inputs["lang_masks"],
                    image_grid_thw=inputs["image_grid_thw"],
                )
            )
            groups = attention.cumsum(dim=1)
            mask = (
                (groups[:, None, :] <= groups[:, :, None])
                & padding[:, None, :]
                & padding[:, :, None]
            )
            outputs, cache, _ = flow.qwenvl_with_expert.forward(
                attention_mask=mask,
                position_ids=positions,
                vlm_position_ids=positions,
                inputs_embeds=[prefix, None],
                past_key_values=None,
                use_cache=True,
                fill_kv_cache=True,
                visual_pos_masks=visual_mask,
                deepstack_visual_embeds=deepstack,
            )
            valid = padding.to(outputs[0].dtype).unsqueeze(-1)
            pooled = (outputs[0] * valid).sum(1) / valid.sum(1).clamp_min(1)
            features = torch.cat((pooled, inputs["state"].to(pooled.dtype)), dim=-1)
        return padding, positions, cache, features

    def _velocity(self, x: torch.Tensor, time: torch.Tensor, inputs: dict, context):
        padding, positions, cache, _ = context
        return self.vla_model.model.predict_velocity(
            inputs["state"],
            padding,
            cache,
            x.to(self.torch_dtype),
            time.expand(len(x)),
            prefix_position_ids=positions,
        ).float()

    def _values(self, features: torch.Tensor) -> torch.Tensor:
        if self.value_head is None:
            return features.new_zeros((len(features), 1), dtype=torch.float32)
        return self.value_head(features.to(self.torch_dtype)).float()

    def _times(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(
            1, 0, self.num_steps + 1, device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "train", **kwargs: Any):
        """Return physical actions plus the detached normalized trajectory for PPO."""
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unknown V2 rollout mode {mode!r}")
        prepared = self.prepare_observations(env_obs)
        parts = [
            self._sample_actions(batch, mode) for batch in self._micro_batches(prepared)
        ]
        return torch.cat([part[0] for part in parts]), {
            "prev_logprobs": torch.cat([part[1]["prev_logprobs"] for part in parts]),
            "prev_values": torch.cat([part[1]["prev_values"] for part in parts]),
            "forward_inputs": {
                key: torch.cat([part[1]["forward_inputs"][key] for part in parts])
                for key in parts[0][1]["forward_inputs"]
            },
        }

    def _micro_batches(self, inputs: dict):
        # BF16 MoE routing can change with GEMM batch shape. Use the same
        # bounded shape for rollout and replay, regardless of env/actor batching.
        for start in range(0, len(inputs["state"]), self.forward_micro_batch_size):
            yield {
                key: value[start : start + self.forward_micro_batch_size]
                for key, value in inputs.items()
            }

    def _sample_actions(self, prepared: dict, mode: str):
        inputs = self._device_inputs(prepared)
        context = self._prefix(inputs)
        shape = (len(inputs["state"]), self.horizon, self.latent_dim)
        x = torch.randn(shape, device=inputs["state"].device, dtype=torch.float32)
        chain = [x]
        logprobs = torch.zeros_like(x)
        times = self._times(x.device)
        for step in range(self.num_steps):
            time, delta = times[step], times[step] - times[step + 1]
            velocity = self._velocity(x, time, inputs, context)
            if mode == "train":
                mean, std = flow_sde_transition(
                    x, velocity, time, delta, self.noise_level
                )
                x = mean + std * torch.randn_like(mean)
                logprobs += gaussian_logprob(x, mean, std)
            else:
                x = x - delta * velocity
            chain.append(x)
        physical = []
        for index in range(len(x)):
            item = {key: value[index].cpu() for key, value in prepared.items()}
            item["actions"] = x[index].float().cpu()
            physical.append(self.transform.unapply(item)["action"])
        actions = torch.stack(physical).float()
        if actions.shape != (shape[0], self.horizon, self.config.action_dim):
            raise ValueError(f"Unexpected V2 robot action shape {actions.shape}")
        if not torch.isfinite(actions).all():
            raise ValueError("V2 generated non-finite robot actions")
        forward_inputs = {
            **prepared,
            "chains": torch.stack(chain, dim=1),
            "actions": actions,
        }
        return actions, {
            "prev_logprobs": logprobs.flatten(1),
            "prev_values": self._values(context[-1]),
            "forward_inputs": forward_inputs,
        }

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs: Any):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                "The V2 adapter currently implements PPO/GRPO only"
            )
        return self.default_forward(**kwargs)

    def default_forward(
        self, forward_inputs: dict, compute_values: bool = True, **kwargs: Any
    ):
        """Re-score recorded transitions; never draw noise during actor training."""
        if forward_inputs["chains"].dtype != torch.float32:
            raise ValueError(
                "V2 SDE samples must stay FP32; set FSDP cast_forward_inputs=false"
            )
        parts = [
            self._score_actions(batch, compute_values)
            for batch in self._micro_batches(forward_inputs)
        ]
        return {
            key: torch.cat([part[key] for part in parts])
            if parts[0][key] is not None
            else None
            for key in parts[0]
        }

    def _score_actions(self, forward_inputs: dict, compute_values: bool):
        inputs = self._device_inputs(forward_inputs)
        chain = inputs["chains"].detach().float()
        expected = (
            len(inputs["state"]),
            self.num_steps + 1,
            self.horizon,
            self.latent_dim,
        )
        if tuple(chain.shape) != expected:
            raise ValueError(
                f"Expected latent trajectory {expected}, got {tuple(chain.shape)}"
            )
        context = self._prefix(inputs)
        times = self._times(chain.device)
        logprobs = torch.zeros_like(chain[:, 0])
        entropy = torch.zeros_like(logprobs)
        for step in range(self.num_steps):
            time, delta = times[step], times[step] - times[step + 1]
            velocity = self._velocity(chain[:, step], time, inputs, context)
            mean, std = flow_sde_transition(
                chain[:, step], velocity, time, delta, self.noise_level
            )
            logprobs = logprobs + gaussian_logprob(chain[:, step + 1], mean, std)
            entropy = entropy + torch.distributions.Normal(mean, std).entropy()
        return {
            "logprobs": logprobs.flatten(1),
            "values": self._values(context[-1]) if compute_values else None,
            "entropy": entropy.flatten(1),
        }
