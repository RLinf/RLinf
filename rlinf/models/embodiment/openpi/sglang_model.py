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

from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
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
        self._rollout_graph = None
        self._rollout_graph_enabled = False

    def capture_cuda_graph(self, train_batch_size: int, eval_batch_size: int):
        """Enable lazy capture on the first observation after loading weights."""
        del train_batch_size, eval_batch_size
        from sglang.multimodal_gen.runtime.vla.cuda_graph import VLATensorGraphRunner

        self._rollout_graph_enabled = True
        if self._rollout_graph is None:
            self._rollout_graph = VLATensorGraphRunner(
                "pi05_rl_rollout", enabled=True, max_entries=4
            )

    def release_cuda_graph(self):
        graph = getattr(self, "_rollout_graph", None)
        if graph is not None:
            graph.clear()
        self._rollout_graph = None

    def _apply(self, fn, recurse=True):
        # Offload/dtype changes can replace the storage referenced by a graph.
        self.release_cuda_graph()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        if mode:
            self.release_cuda_graph()
        return super().train(mode)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.release_cuda_graph()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_global_step(self, global_step: int) -> None:
        if global_step != self.global_step:
            self.release_cuda_graph()
        super().set_global_step(global_step)

    def _predict_train(self, observation, *, noise, rng, compute_values):
        if (
            not self._rollout_graph_enabled
            or self.training
            or torch.is_grad_enabled()
            or observation.state.device.type != "cuda"
        ):
            return super()._predict_train(
                observation, noise=noise, rng=rng, compute_values=compute_values
            )
        self.capture_cuda_graph(0, 0)
        observation = preprocess_observation(observation, train=False)
        batch, device = observation.state.shape[0], observation.state.device
        if noise is None:
            noise = torch.randn(
                batch,
                self.action_horizon,
                self.model_action_dim,
                device=device,
                dtype=torch.float32,
                generator=rng,
            )
        indices = self._sample_denoise_indices(batch, device)
        # Match the eager RNG calls, including the unused noise on ODE steps.
        step_noise = tuple(
            torch.randn(noise.shape, device=device, dtype=torch.float32, generator=rng)
            for _ in range(self.num_steps)
        )
        cameras = tuple(observation.images)
        full_attention = self._has_full_attention(observation)
        inputs = (
            observation.state,
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            *(observation.images[name] for name in cameras),
            *(observation.image_masks[name] for name in cameras),
            noise,
            indices,
            *step_noise,
        )

        def sample(*tensors):
            state, tokens, masks = tensors[:3]
            count = len(cameras)
            obs = Observation(
                state=state,
                tokenized_prompt=tokens,
                tokenized_prompt_mask=masks,
                images=dict(zip(cameras, tensors[3 : 3 + count], strict=True)),
                image_masks=dict(
                    zip(cameras, tensors[3 + count : 3 + 2 * count], strict=True)
                ),
            )
            x_t, selected, *noises = tensors[3 + 2 * count :]
            sampled = self._sample_train_chain(
                obs,
                x_t,
                selected,
                self.build_prefix_cache(obs, full_attention=full_attention),
                compute_values=compute_values,
                step_noises=tuple(noises),
            )
            return sampled[:4]

        output = self._rollout_graph.run(
            sample,
            inputs,
            key=(cameras, self.num_steps, self.rl_cfg, compute_values, full_attention),
        )
        # Replays reuse graph storage; each trajectory must own its tensors.
        sampled = (*(tensor.clone() for tensor in output), None)
        return self._pack_train_output(observation, indices, sampled)

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

    def build_prefix_cache(
        self, observation: Observation, *, full_attention: bool | None = None
    ) -> tuple:
        """Encode the frozen prefix; graph callers supply the mask decision."""
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
                prefix_full_attention_hint=full_attention,
                tokens_trimmed=True,
                return_hidden_states=True,
            )
        return hidden, mask, cache

    @staticmethod
    def _has_full_attention(observation: Observation) -> bool:
        """Resolve the mask-dependent graph branch before CUDA capture."""
        masks = [observation.tokenized_prompt_mask, *observation.image_masks.values()]
        return bool(torch.cat([mask.flatten() for mask in masks]).all().item())

    def prepare_actor_graph_inputs(self, forward_inputs: dict) -> tuple[dict, bool]:
        """Run host transforms before capturing the actor network."""
        observation = self._observation_from_forward_inputs(forward_inputs)
        inputs = {
            "chains": forward_inputs["chains"],
            "denoise_inds": forward_inputs["denoise_inds"],
            "state": observation.state,
            "tokens": observation.tokenized_prompt,
            "token_mask": observation.tokenized_prompt_mask,
        }
        for name, tensor in observation.images.items():
            inputs["image/" + name] = tensor
            inputs["mask/" + name] = observation.image_masks[name]
        return inputs, self._has_full_attention(observation)

    def default_forward(self, forward_inputs: dict, **kwargs):
        if not kwargs.get("prepared_actor_inputs", False):
            return super().default_forward(forward_inputs, **kwargs)
        observation = Observation(
            state=forward_inputs["state"],
            tokenized_prompt=forward_inputs["tokens"],
            tokenized_prompt_mask=forward_inputs["token_mask"],
            images={name: forward_inputs[f"image/{name}"] for name in IMAGE_KEYS},
            image_masks={name: forward_inputs[f"mask/{name}"] for name in IMAGE_KEYS},
        )
        prefix = self.build_prefix_cache(
            observation, full_attention=kwargs["prefix_full_attention"]
        )
        return self._recompute_from_prefix(
            observation,
            forward_inputs["chains"],
            forward_inputs["denoise_inds"],
            *prefix,
            compute_values=kwargs.get("compute_values", True),
        )

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
        self.release_cuda_graph()
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
