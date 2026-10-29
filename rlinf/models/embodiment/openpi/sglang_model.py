# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang's native Pi0.5 network under RLinf's flow-policy RL interface."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig


class SGLangPi05RL(Pi0RL):
    """Train and sample the same SGLang model with RLinf's stochastic flow sampler.

    RLinf owns the optimizer, rollout channels, transforms and probability
    reduction. SGLang owns the frozen VLM and differentiable action expert.
    State dictionaries use the same fused layout on Actor and rollout workers,
    so both bucket and patch synchronization can update the tensors directly.
    """

    def __init__(self, config: Pi0Config, *, rl_cfg: Pi0RLConfig, **kwargs):
        if not config.pi05 or config.pcd:
            raise ValueError("SGLang RL supports Pi0.5 without point-cloud inputs")
        if config.dtype not in ("bfloat16", "float32"):
            raise ValueError("SGLang Pi0.5 RL supports bf16 or fp32 precision")
        if not rl_cfg.train_expert_only:
            raise ValueError("SGLang Pi0.5 RL requires train_expert_only=True")
        if rl_cfg.is_nft:
            raise ValueError("SGLang Pi0.5 RL currently supports PPO/GRPO, not NFT")
        if rl_cfg.noise_method not in ("flow_sde", "flow_cps"):
            raise ValueError("SGLang Pi0.5 RL requires flow_sde or flow_cps noise")
        if not 0 < rl_cfg.noise_level <= 1:
            raise ValueError("SGLang Pi0.5 RL requires 0 < noise_level <= 1")
        if kwargs.get("num_steps", 10) < 2:
            raise ValueError("Stochastic Pi0.5 training requires num_steps >= 2")
        if rl_cfg.noise_method == "flow_cps" and not rl_cfg.ignore_last:
            raise ValueError(
                "flow_cps requires ignore_last=True (last step has zero noise)"
            )
        super().__init__(config, rl_cfg=rl_cfg, **kwargs)

    def _build_network(self, config: Pi0Config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.pi05 import (
            Pi05PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.models.vlas.pi05_core import Pi05CoreModel
        from sglang.srt.runtime_context import get_parallel, publish
        from sglang.srt.server_args import ServerArgs

        if not hasattr(Pi05CoreModel, "enable_action_training"):
            raise ImportError("Install the SGLang Pi0.5 action-training branch first")
        # SigLIP's replicated SRT layers read logical TP metadata. FSDP owns
        # distributed training; SGLang must not initialize a second TP group.
        publish(ServerArgs(model_path="dummy", tp_size=1), role="diffusion_gpu_worker")
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)
        pipeline_cfg = Pi05PipelineConfig(
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
            materialize_dtype=config.dtype,
            enable_prefix_cuda_graph=False,
            enable_action_cuda_graph=False,
        )
        self.action_dim = config.action_dim
        self.core_model = Pi05CoreModel(pipeline_cfg, action_training=True)
        self._gradient_checkpointing = False

    def build_prefix_cache(self, observation: Observation) -> tuple:
        # The prefix is frozen in both rollout and replay; never keep its graph.
        with torch.no_grad():
            images = [
                image.permute(0, 3, 1, 2).contiguous()
                for image in observation.images.values()
            ]
            cache, mask, _, hidden = self.core_model.encode_prefix(
                images,
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                tokens_trimmed=True,
                return_hidden_states=True,
            )
        return hidden, mask, cache

    def run_suffix(
        self,
        observation: Observation,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        kv_cache: tuple,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        del observation
        if self._gradient_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                self.core_model.forward_suffix,
                prefix_mask,
                kv_cache,
                x_t,
                t_tensor,
                use_reentrant=False,
            ).float()
        return self.core_model.forward_suffix(
            prefix_mask, kv_cache, x_t, t_tensor
        ).float()

    def velocity_from_suffix(self, suffix_out_act: torch.Tensor) -> torch.Tensor:
        projection = self.core_model.action_out_proj
        return projection(suffix_out_act.to(projection.weight.dtype)).float()

    def freeze_vlm(self, freeze_action_expert: bool = False) -> int:
        if freeze_action_expert:
            raise ValueError("SGLang Pi0.5 RL trains the action expert")
        prefix = self.core_model.paligemma_with_expert.paligemma
        prefix.requires_grad_(False)
        return sum(1 for _ in prefix.parameters())

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        self.core_model.paligemma_with_expert.to_selected_dtype(precision)

    @property
    def _no_split_modules(self) -> list[str]:
        # RMSNorm and vision embeddings retain fp32; FSDP1 wraps them separately.
        return [
            "PiGemmaDecoderLayer",
            "PiGemmaRMSNorm",
            "SiglipEncoderLayer",
            "SiglipVisionEmbeddings",
            "ValueHead",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "embed_tokens",
            "multi_modal_projector",
            "post_layernorm",
        ]

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None, **kwargs
    ) -> None:
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self._gradient_checkpointing = False
