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

from __future__ import annotations

import dataclasses
import random
from typing import Any, Literal, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi_rlinf.env_io import EnvIO
from rlinf.models.embodiment.openpi_rlinf.modules import gemma
from rlinf.models.embodiment.openpi_rlinf.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.sampling import rl_sampler
from rlinf.models.embodiment.openpi_rlinf.transforms.env import repack_env_obs


@dataclasses.dataclass(frozen=True)
class Pi0RLConfig:
    """Static RL knobs read from ``actor.model.openpi`` in YAML."""

    add_value_head: bool = False
    noise_method: str = "flow_ode"
    noise_level: float = 0.0
    noise_logvar_range: tuple[float, float] = (0.08, 0.16)
    joint_logprob: bool = False
    ignore_last: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    train_expert_only: bool = False
    is_nft: bool = False
    config_name: str = ""


class Pi0RL(EnvIO, Pi0):
    """On-policy flow RL (PPO / GRPO / NFT) inheriting vendored Pi0."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        num_steps: int = 10,
        action_chunk: int | None = None,
        action_env_dim: int | None = None,
        rl_cfg: Pi0RLConfig,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name or rl_cfg.config_name,
            state_indices=state_indices,
        )
        self.model_action_dim = self.action_dim
        self.rl_cfg = rl_cfg
        self.global_step = 0
        paligemma_width = gemma.get_config(config.paligemma_variant).width
        action_expert_width = gemma.get_config(config.action_expert_variant).width

        if rl_cfg.add_value_head:
            input_dim = (
                paligemma_width if rl_cfg.value_after_vlm else action_expert_width
            )
            hidden = (1024, 512, 256) if rl_cfg.value_after_vlm else (512, 256, 128)
            if any(
                name in rl_cfg.config_name
                for name in ("pi05_maniskill", "pi05_libero", "pi05_droid_polaris")
            ):
                hidden = (1024, 512, 256)
            # Stay fp32 like OpenPI: action/value heads sit outside
            # paligemma_with_expert and are never converted to bf16.
            self.value_head = ValueHead(
                input_dim=input_dim,
                hidden_sizes=hidden,
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if rl_cfg.noise_method == "flow_noise":
            from rlinf.models.embodiment.modules.explore_noise_net import (
                ExploreNoiseNet,
            )

            self.noise_head = ExploreNoiseNet(
                in_dim=action_expert_width,
                out_dim=self.model_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=list(rl_cfg.noise_logvar_range),
                noise_scheduler_type="learn",
            )

        self._mark_fsdp_wrap_names()

    @property
    def _no_split_modules(self) -> list[str] | None:
        names = list(super()._no_split_modules or [])
        if self.rl_cfg.add_value_head:
            names.append("ValueHead")
        if self.rl_cfg.noise_method == "flow_noise":
            names.append("ExploreNoiseNet")
        return names

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def _build_prefix_cache_for_actor(self, observation: Observation):
        """Prefix KV for actor recompute.

        ``train_expert_only`` freezes the VLM, so prefix must not record
        autograd. OpenPI gets this for free: paligemma is a separate module
        with ``requires_grad=False``. Here expert-0/1 share an FSDP Block, so
        a training prefix forward would keep 18 layers of fp32 attention
        (~2.5GiB/layer at micro_batch=128) and OOM. When the VLM is trained,
        prefix stays in the graph.
        """
        if self.rl_cfg.train_expert_only:
            with torch.no_grad():
                return self.build_prefix_cache(observation)
        return self.build_prefix_cache(observation)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = True,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        repacked = repack_env_obs(
            self.config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        observation = self._observation_dict_to_device(
            self.input_transform(repacked, transpose=False)
        )
        if mode == "eval":
            return self._predict_eval(observation, noise=noise, rng=rng)
        actions, result = self._predict_train(
            observation, noise=noise, rng=rng, compute_values=compute_values
        )
        # Match OpenPI: stash raw env obs so the actor can re-transform.
        result["forward_inputs"].update(_clone_replay_obs(repacked))
        return actions, result

    def _predict_train(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
        compute_values: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        rl_cfg = self.rl_cfg
        device = self.device
        observation = preprocess_observation(observation, train=False)
        B = observation.state.shape[0]
        num_steps = self.num_steps

        if noise is None:
            noise = torch.randn(
                B,
                self.action_horizon,
                self.model_action_dim,
                device=device,
                dtype=torch.float32,
                generator=rng,
            )

        prefix_out, prefix_mask, kv_cache = self.build_prefix_cache(observation)
        vlm_value = None
        if rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            vlm_value = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
            )

        collect_nft = rl_cfg.is_nft
        if rl_cfg.joint_logprob or collect_nft:
            chosen_per_step = torch.arange(num_steps, device=device)
            denoise_inds = chosen_per_step[None].expand(B, -1).contiguous()
        else:
            hi = num_steps - 2 if rl_cfg.ignore_last else num_steps - 1
            chosen = random.randint(0, max(hi, 0))
            denoise_inds = torch.full(
                (B, num_steps), chosen, device=device, dtype=torch.long
            )

        nft_state = self._init_nft_state(collect_nft, noise, num_steps, device)
        timesteps = rl_sampler.get_timesteps(num_steps, device)
        idx_step = torch.empty(B, device=device, dtype=torch.long)
        chains = [noise]
        log_probs: list[torch.Tensor] = []
        suffix_values: list[torch.Tensor] = []
        x_t = noise

        if rl_cfg.joint_logprob:
            log_probs.append(
                rl_sampler.gaussian_logprob(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
            )

        for idx in range(num_steps):
            method = (
                rl_cfg.noise_method
                if idx == int(denoise_inds[0, idx].item())
                else "flow_ode"
            )
            t_val = float(timesteps[idx].item())
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
            suffix_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)
            noise_std = None
            if method == "flow_noise":
                noise_std = self.noise_head(suffix_act).to(torch.float32)
            idx_step.fill_(idx)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                x_t.to(torch.float32),
                v_t,
                idx_step,
                noise_method=method,
                noise_level=rl_cfg.noise_level,
                num_steps=num_steps,
                noise_std=noise_std,
            )
            x_t_prev = x_t
            step_noise = torch.randn(
                x_t.shape, device=device, dtype=torch.float32, generator=rng
            )
            x_t = x_t_mean + step_noise * x_t_std
            self._update_nft_state(nft_state, idx, x_t_prev, v_t, x_t, method)
            log_probs.append(rl_sampler.gaussian_logprob(x_t, x_t_mean, x_t_std))
            chains.append(x_t)
            if compute_values and rl_cfg.add_value_head and not rl_cfg.value_after_vlm:
                suffix_values.append(
                    rl_sampler.value_from_suffix(
                        self.value_head,
                        suffix_act,
                        action_chunk=self.action_chunk,
                        chunk_critic_input=rl_cfg.chunk_critic_input,
                        detach_critic_input=rl_cfg.detach_critic_input,
                    )
                )

        x_0 = x_t
        chains_tensor = torch.stack(chains, dim=1).contiguous()
        log_probs_stacked = torch.stack(log_probs, dim=1)
        log_probs_stacked = log_probs_stacked[
            :, :, : self.action_chunk, : self.action_env_dim
        ]
        if rl_cfg.joint_logprob:
            log_probs_picked = log_probs_stacked.mean(dim=1)
        else:
            log_probs_picked = log_probs_stacked[
                torch.arange(B, device=device), denoise_inds[:, 0]
            ]
        log_probs_picked = log_probs_picked.float().contiguous()

        if vlm_value is not None:
            prev_values = vlm_value[:, None].float().contiguous()
        elif suffix_values:
            prev_values = torch.stack(suffix_values, dim=1).mean(dim=1, keepdim=True)
        else:
            prev_values = torch.zeros((B, 1), device=device, dtype=torch.float32)

        actions = self.decode_actions(x_0, observation.state)

        forward_inputs: dict[str, torch.Tensor] = {
            "chains": chains_tensor,
            "denoise_inds": denoise_inds,
            "tokenized_prompt": observation.tokenized_prompt.contiguous(),
            "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
            "action": actions.reshape(B, -1).contiguous(),
            "model_action": x_0.reshape(B, -1).contiguous(),
        }
        if nft_state is not None:
            forward_inputs.update(nft_state)
            forward_inputs["nft_x0"] = x_0.detach()

        return actions, {
            "prev_logprobs": log_probs_picked,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
            "model_actions": x_0,
        }

    def _init_nft_state(
        self,
        collect: bool,
        x_t: torch.Tensor,
        num_steps: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if not collect:
            return None
        return {
            "nft_step_index": torch.randint(
                0, num_steps, (x_t.shape[0],), device=device
            ),
            "nft_xcur": torch.zeros_like(x_t),
            "nft_v": torch.zeros_like(x_t),
            "nft_xnext": torch.zeros_like(x_t),
            "nft_noise_level": torch.zeros(
                x_t.shape[0], device=device, dtype=x_t.dtype
            ),
        }

    def _update_nft_state(
        self,
        nft_state: dict[str, torch.Tensor] | None,
        idx: int,
        x_t_prev: torch.Tensor,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        sample_method: str,
    ) -> None:
        if nft_state is None:
            return
        mask = nft_state["nft_step_index"] == idx
        if not mask.any():
            return
        mask_bc = mask[:, None, None]
        nft_state["nft_xcur"] = torch.where(
            mask_bc, x_t_prev.detach(), nft_state["nft_xcur"]
        )
        nft_state["nft_v"] = torch.where(mask_bc, v_t.detach(), nft_state["nft_v"])
        nft_state["nft_xnext"] = torch.where(
            mask_bc, x_t.detach(), nft_state["nft_xnext"]
        )
        level = 0.0 if sample_method == "flow_ode" else float(self.rl_cfg.noise_level)
        nft_state["nft_noise_level"] = torch.where(
            mask,
            torch.full_like(nft_state["nft_noise_level"], level),
            nft_state["nft_noise_level"],
        )

    def _observation_from_forward_inputs(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> Observation:
        """Rebuild the rollout observation so actor logprobs match sampling.

        Prefer raw ``observation/*`` tensors (OpenPI path): re-run
        ``input_transform`` then ``preprocess_observation``. Fall back to the
        older ``obs_image__*`` layout, assembling cameras in ``IMAGE_KEYS``
        order rather than ``dict`` iteration order.
        """
        has_raw_obs = any(key.startswith("observation/") for key in forward_inputs)
        if has_raw_obs:
            processed = self.input_transform(forward_inputs, transpose=False)
            observation = self._observation_dict_to_device(processed)
        else:
            images: dict[str, torch.Tensor] = {}
            image_masks: dict[str, torch.Tensor] = {}
            for name in IMAGE_KEYS:
                image_key = f"obs_image__{name}"
                if image_key in forward_inputs:
                    images[name] = forward_inputs[image_key]
                mask_key = f"obs_image_mask__{name}"
                if mask_key in forward_inputs:
                    image_masks[name] = forward_inputs[mask_key]
            for key, value in forward_inputs.items():
                if key.startswith("obs_image__"):
                    name = key[len("obs_image__") :]
                    images.setdefault(name, value)
                elif key.startswith("obs_image_mask__"):
                    name = key[len("obs_image_mask__") :]
                    image_masks.setdefault(name, value)
            observation = Observation(
                images=images,
                image_masks=image_masks,
                state=forward_inputs["obs_state"],
                tokenized_prompt=forward_inputs["tokenized_prompt"],
                tokenized_prompt_mask=forward_inputs["tokenized_prompt_mask"],
            )
        return preprocess_observation(observation, train=False)

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            return super().sft_forward(**kwargs)
        raise NotImplementedError(
            f"{type(self).__name__} does not support forward_type={forward_type!r}."
        )

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """PPO/GRPO recompute of logprobs and values from the stored chain."""
        rl_cfg = self.rl_cfg
        compute_values = kwargs.get("compute_values", True)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        B = chains.shape[0]
        device = chains.device
        observation = self._observation_from_forward_inputs(forward_inputs)

        prefix_out, prefix_mask, kv_cache = self._build_prefix_cache_for_actor(
            observation
        )

        timesteps = rl_sampler.get_timesteps(self.num_steps, device)
        if rl_cfg.joint_logprob:
            num_recompute = self.num_steps
            initial = rl_sampler.gaussian_logprob(
                chains[:, 0].to(torch.float32),
                torch.zeros_like(chains[:, 0], dtype=torch.float32),
                torch.ones_like(chains[:, 0], dtype=torch.float32),
            )
            step_logprobs = [initial]
            step_entropy = [
                rl_sampler.gaussian_entropy(
                    torch.ones_like(chains[:, 0], dtype=torch.float32)
                )
            ]
        else:
            num_recompute = 1
            step_logprobs = []
            step_entropy = []

        suffix_values = []
        arange_B = torch.arange(B, device=device)
        for idx in range(num_recompute):
            denoise_ind = denoise_inds[:, idx].to(torch.long)
            chains_pre = chains[arange_B, denoise_ind]
            chains_next = chains[arange_B, denoise_ind + 1]
            t_input = timesteps[denoise_ind].to(torch.float32)
            suffix_act = self.run_suffix(
                observation, chains_pre, t_input, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)
            noise_std = None
            if rl_cfg.noise_method == "flow_noise":
                noise_std = self.noise_head(suffix_act).to(torch.float32)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                chains_pre.to(torch.float32),
                v_t,
                denoise_ind,
                noise_method=rl_cfg.noise_method,
                noise_level=rl_cfg.noise_level,
                num_steps=self.num_steps,
                noise_std=noise_std,
            )
            step_logprobs.append(
                rl_sampler.gaussian_logprob(
                    chains_next.to(torch.float32), x_t_mean, x_t_std
                )
            )
            step_entropy.append(rl_sampler.gaussian_entropy(x_t_std))
            if compute_values and rl_cfg.add_value_head and not rl_cfg.value_after_vlm:
                suffix_values.append(
                    rl_sampler.value_from_suffix(
                        self.value_head,
                        suffix_act,
                        action_chunk=self.action_chunk,
                        chunk_critic_input=rl_cfg.chunk_critic_input,
                        detach_critic_input=rl_cfg.detach_critic_input,
                    )
                )

        log_probs = torch.stack(step_logprobs, dim=1)
        log_probs = log_probs[:, :, : self.action_chunk, : self.action_env_dim].float()
        if rl_cfg.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[:, 0]

        if compute_values and rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            values = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
            )
        elif suffix_values:
            values = torch.stack(suffix_values, dim=1).mean(dim=1)
        else:
            values = torch.zeros(B, device=device, dtype=torch.float32)

        entropy = torch.stack(step_entropy, dim=1)
        entropy = entropy[:, :, : self.action_chunk, : self.action_env_dim]
        if rl_cfg.noise_method == "flow_noise":
            entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        else:
            entropy = torch.zeros((B, 1), device=device, dtype=torch.float32)
        return {
            "logprobs": log_probs.contiguous(),
            "values": values.float(),
            "entropy": entropy.float(),
        }

    def nft_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """Compute velocity v_theta at explicit (x_t, timesteps) for NFT loss."""
        nft_inputs = kwargs["nft_inputs"]
        observation = self._observation_from_forward_inputs(forward_inputs)
        device = self.device
        x_t = nft_inputs["x_t"].to(device)
        t = nft_inputs["timesteps"].to(device)
        if t.dim() > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        prefix_out, prefix_mask, kv_cache = self._build_prefix_cache_for_actor(
            observation
        )
        suffix_act = self.run_suffix(observation, x_t, t, kv_cache, prefix_mask)
        v_theta = self.velocity_from_suffix(suffix_act)
        v_theta = v_theta[:, : self.action_chunk, :]
        result: dict[str, Any] = {"v_theta": v_theta, "x_t": x_t, "timesteps": t}
        if kwargs.get("compute_values") and self.rl_cfg.add_value_head:
            if self.rl_cfg.value_after_vlm:
                result["values"] = rl_sampler.value_from_prefix(
                    self.value_head,
                    prefix_out,
                    prefix_mask,
                    mode=self.rl_cfg.value_vlm_mode,
                )[:, None]
            else:
                result["values"] = rl_sampler.value_from_suffix(
                    self.value_head, suffix_act, action_chunk=self.action_chunk
                )[:, None]
        return result


def _clone_replay_obs(repacked: dict[str, Any]) -> dict[str, Any]:
    """Clone env-repacked tensors for ``forward_inputs`` (skip the prompt)."""
    cloned: dict[str, Any] = {}
    for key, value in repacked.items():
        if key == "prompt":
            continue
        if torch.is_tensor(value):
            cloned[key] = value.detach().contiguous()
        elif isinstance(value, np.ndarray):
            cloned[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            cloned[key] = value
    return cloned
