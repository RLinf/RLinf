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

"""RLinf embodied-policy wrapper for the Evo-1 (evo1-flash) VLA model.

Evo-1 = InternVL3-1B embedder + flow-matching (DiT) action head. This wrapper
implements the RLinf ``BasePolicy`` surface:

    * ``predict_action_batch`` -- eval/rollout inference on LIBERO-style envs.
      ``mode="train"`` runs the SDE-replay sampler used for RL rollouts (stores
      the denoising ``chains`` / ``denoise_inds`` into ``forward_inputs``);
      ``mode="eval"`` runs the deterministic ODE integrator (unchanged).
    * ``sft_forward``          -- native flow-matching MSE training loss.
    * ``default_forward``      -- RL path; replays the stored denoising chains
      under the current policy weights to recompute per-step log-probs
      (flow-matching SDE-replay, GRPO-first). See ``evo1_rl_head.py``.

The underlying ``EVO1`` module is imported from the externally-installed Evo-1
repo (see ``requirements/install.sh::install_evo1_model``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.evo1 import evo1_rl_head as rlh
from rlinf.models.embodiment.evo1.utils.data_pipeline import build_evo1_inputs
from rlinf.models.embodiment.evo1.utils.normalizer import Normalizer
from rlinf.utils.logging import get_logger

logger = get_logger()


class Evo1ForRLActionPrediction(nn.Module, BasePolicy):
    """Wrap an ``EVO1`` model into RLinf's embodied policy interface."""

    # Submodule class names FSDP should keep intact (not split across shards).
    # InternVL3 vision/LLM blocks + the DiT action-head block.
    _no_split_modules = [
        "InternVisionEncoderLayer",
        "Qwen2DecoderLayer",
        "LlamaDecoderLayer",
        "BasicTransformerBlock",
    ]

    def __init__(
        self,
        evo1_model: nn.Module,
        normalizer: Normalizer,
        *,
        action_dim: int,
        num_action_chunks: int,
        arm_key: str,
        dataset_key: str,
        policy_setup: str | None = "libero",
        image_size: int = 448,
        num_view_slots: int = 3,
        binarize_gripper: bool = True,
        gripper_open_value: float = 1.0,
        gripper_close_value: float = -1.0,
        rl_head_config: Any = None,
        rl_denoising_steps: int | None = None,
        valid_action_dim: int | None = None,
        prompt_max_len: int = 512,
    ):
        super().__init__()
        self.evo1 = evo1_model
        self.normalizer = normalizer

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.arm_key = arm_key
        self.dataset_key = dataset_key
        self.policy_setup = (policy_setup or "").strip().lower()
        self.image_size = image_size
        self.num_view_slots = num_view_slots
        self.binarize_gripper = binarize_gripper
        self.gripper_open_value = gripper_open_value
        self.gripper_close_value = gripper_close_value

        # Evo-1 model geometry.
        self.horizon = getattr(evo1_model, "horizon", None)
        self.per_action_dim = getattr(evo1_model, "per_action_dim", None)

        # RL (SDE-replay) knobs. ``valid_action_dim`` is the env action dim used
        # for the log-prob slice; defaults to ``action_dim``. ``rl_denoising_steps``
        # is the (small) number of SDE steps used for RL rollout/replay; both must
        # use the *same* value. Falls back to the head's inference step count.
        self.rl_head_config = rl_head_config
        self.valid_action_dim = valid_action_dim or action_dim
        self.prompt_max_len = int(prompt_max_len)
        head_steps = getattr(
            getattr(evo1_model, "action_head", None), "num_inference_timesteps", None
        )
        self.rl_denoising_steps = int(
            rl_denoising_steps
            or rlh.cfg_get(rl_head_config, "denoising_steps", None)
            or head_steps
            or 32
        )

    # ------------------------------------------------------------------ #
    # FSDP auto-wrap: transformer layer classes to shard/wrap.
    # InternVL3-1B embedder = InternViT (InternVisionEncoderLayer) + Qwen2 LLM
    # (Qwen2DecoderLayer); the flow-matching action head uses BasicTransformerBlock.
    # ------------------------------------------------------------------ #
    _no_split_modules = [
        "InternVisionEncoderLayer",
        "Qwen2DecoderLayer",
        "BasicTransformerBlock",
    ]

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            f"Evo-1 does not support forward_type={forward_type}."
        )

    # ------------------------------------------------------------------ #
    # Shared input encoding
    # ------------------------------------------------------------------ #
    def _encode_env_obs(self, env_obs: dict[str, Any]):
        """Turn ``env_obs`` into Evo-1 inference tensors.

        Returns ``(images, image_mask, prompts, norm_state, batch_size, device)``
        where ``images`` is a list[B] of list[V] of CHW float tensors, and
        ``norm_state`` is the normalized proprio state ``[B, state_dim]`` (fp32).
        """
        device = next(self.evo1.parameters()).device
        images, image_mask, prompts, states = build_evo1_inputs(
            env_obs,
            image_size=self.image_size,
            num_view_slots=self.num_view_slots,
        )
        images = [
            [v.to(device) if torch.is_tensor(v) else v for v in view_list]
            for view_list in images
        ]
        batch_size = len(images)
        image_mask = image_mask.to(device)

        if states is None:
            raise ValueError("Evo-1 requires 'states' in env_obs.")
        states = states.to(device)
        norm_state = self.normalizer.normalize_state(
            states, self.arm_key, self.dataset_key
        ).to(dtype=torch.float32)
        return images, image_mask, prompts, norm_state, batch_size, device

    def _action_mask_3d(self, batch_size: int, device) -> torch.Tensor:
        """Per-step action mask ``[B, H, per_dim]`` with valid dims = 1.

        Evo-1's flow head requires a mask that zeros padded action dims across
        the whole denoising trajectory. LIBERO uses the first ``action_dim`` of
        ``per_action_dim`` slots.
        """
        per_dim = self.per_action_dim or self.normalizer.target_dim
        mask = torch.zeros(
            batch_size, self.horizon, per_dim, dtype=torch.float32, device=device
        )
        mask[:, :, : self.action_dim] = 1.0
        return mask

    def _finalize_env_action(self, norm_actions: torch.Tensor) -> np.ndarray:
        """Denormalize ``[B, H, per_dim]`` -> env chunk actions ``[B, chunk, dim]``."""
        actions = self.normalizer.denormalize_action(
            norm_actions, self.arm_key, self.dataset_key
        )  # -> [..., per_dim]
        actions = actions[:, : self.num_action_chunks, : self.action_dim]
        env_chunk_actions = actions.detach().cpu().numpy().astype(np.float32)
        return self._apply_gripper_mapping(env_chunk_actions)

    # ------------------------------------------------------------------ #
    # Inference / rollout
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        calculate_logprobs: bool = False,
        calculate_values: bool = False,
        return_obs: bool = True,
        mode: str = "eval",
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict env-space action chunks for a batch of observations.

        ``mode="eval"`` (default): deterministic ODE integration, no RL bookkeeping
        (``prev_logprobs``/``prev_values`` are ``None`` and ``forward_inputs`` is
        empty). ``mode="train"``: SDE-replay sampler that also returns the RL
        rollout data required by ``default_forward``.
        """
        del return_obs, calculate_logprobs, calculate_values, kwargs

        if mode == "train":
            return self._predict_action_batch_train(env_obs)

        images, image_mask, prompts, norm_state, batch_size, device = (
            self._encode_env_obs(env_obs)
        )

        per_dim = self.per_action_dim or self.normalizer.target_dim
        action_mask = torch.zeros(
            batch_size, per_dim, dtype=torch.float32, device=device
        )
        action_mask[:, : self.action_dim] = 1.0

        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            fused = self.evo1.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=prompts,
                return_cls_only=False,
            )
            state_t = self.evo1.prepare_state(norm_state)
            flat_actions = self.evo1.predict_action(
                fused, state_t, actions_gt=None, action_mask=action_mask
            )
        flat_actions = flat_actions.float()

        actions = flat_actions.reshape(batch_size, -1, per_dim)
        env_chunk_actions = self._finalize_env_action(actions)

        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {},
        }
        return env_chunk_actions, result

    @torch.no_grad()
    def _predict_action_batch_train(
        self, env_obs: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """SDE-replay rollout: sample actions and record the denoising chain."""
        images, image_mask, prompts, norm_state, batch_size, device = (
            self._encode_env_obs(env_obs)
        )
        action_mask_3d = self._action_mask_3d(batch_size, device)
        embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            fused = self.evo1.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=prompts,
                return_cls_only=False,
            )
            state_t = self.evo1.prepare_state(norm_state)
            x_0, chains, denoise_inds, log_probs = rlh.rl_denoise_sample(
                self.evo1.action_head,
                fused_tokens=fused,
                state=state_t,
                embodiment_id=embodiment_id,
                action_mask_3d=action_mask_3d,
                denoise_steps=self.rl_denoising_steps,
                rl_cfg=self.rl_head_config,
                mode="train",
            )

        # Reduce per-step log-probs to the executed-chunk / valid-dim form,
        # matching what ``default_forward`` returns (so ratios line up 1:1).
        prev_logprobs = self._reduce_logprobs(log_probs, denoise_inds)

        # Env actions from the final (normalized) sample.
        env_chunk_actions = self._finalize_env_action(x_0.float())

        forward_inputs = self._pack_forward_inputs(
            chains=chains,
            denoise_inds=denoise_inds,
            images=images,
            image_mask=image_mask,
            prompts=prompts,
            norm_state=norm_state,
            action_mask_3d=action_mask_3d,
            x_0=x_0,
            device=device,
        )

        result = {
            "prev_logprobs": prev_logprobs.float().cpu(),
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }
        return env_chunk_actions, result

    def _reduce_logprobs(
        self, log_probs: torch.Tensor, denoise_inds: torch.Tensor
    ) -> torch.Tensor:
        """Reduce ``[B, num_steps(+1), H, per_dim]`` -> ``[B, chunk, valid_dim]``.

        Non-joint: pick the single stochastic step (``denoise_inds[:, 0]``).
        Joint: mean over all recorded steps.
        """
        joint = bool(rlh.cfg_get(self.rl_head_config, "joint_logprob", False))
        sliced = log_probs[:, :, : self.num_action_chunks, : self.valid_action_dim]
        if joint:
            return sliced.mean(dim=1)
        # Non-joint: pick the single stochastic transition. Rollout keeps all N
        # steps (index by denoise_ind); replay stores only that step (dim size 1).
        bsz, num_steps = sliced.shape[0], sliced.shape[1]
        if num_steps == 1:
            return sliced[:, 0]
        return sliced[torch.arange(bsz, device=sliced.device), denoise_inds[:, 0]]

    def _pack_forward_inputs(
        self,
        *,
        chains,
        denoise_inds,
        images,
        image_mask,
        prompts,
        norm_state,
        action_mask_3d,
        x_0,
        device,
    ) -> dict[str, torch.Tensor]:
        """Assemble the all-tensor ``forward_inputs`` dict for replay.

        RLinf's rollout->train merge pipeline requires every value to be a
        tensor. Images are stacked to ``[B, V, 3, H, W]`` and prompts are encoded
        as a fixed-width UTF-8 byte tensor (losslessly reconstructed at replay).
        """
        pixel_values = torch.stack(
            [torch.stack(view_list, dim=0) for view_list in images], dim=0
        )
        prompt_bytes, prompt_len = rlh.bytes_from_prompts(
            prompts, max_len=self.prompt_max_len, device=device
        )
        bsz = pixel_values.shape[0]
        return {
            "chains": chains.detach().float(),
            "denoise_inds": denoise_inds.detach().long(),
            "pixel_values": pixel_values.detach().float(),
            "image_mask": image_mask.detach().long(),
            "prompt_bytes": prompt_bytes.detach().long(),
            "prompt_len": prompt_len.detach().long(),
            "state": norm_state.detach().float(),
            "action_mask": action_mask_3d.detach().float(),
            "action": x_0.detach().reshape(bsz, -1).float(),
        }

    def _apply_gripper_mapping(self, actions: np.ndarray) -> np.ndarray:
        """Map Evo-1's gripper output to the LIBERO env convention.

        Evo-1's LIBERO client thresholds the gripper channel (index 6) at 0.5:
        ``>0.5 -> close(-1)``, ``else -> open(+1)`` (see libero_client_4tasks.py).
        """
        if not self.binarize_gripper:
            return actions
        if self.policy_setup != "libero" or actions.shape[-1] < 7:
            return actions
        out = actions.astype(np.float32, copy=True)
        close = out[..., 6] > 0.5
        out[..., 6] = np.where(
            close, self.gripper_close_value, self.gripper_open_value
        ).astype(np.float32)
        return out

    # ------------------------------------------------------------------ #
    # SFT training -- native flow-matching MSE.
    # ------------------------------------------------------------------ #
    def sft_forward(self, data: dict[str, Any], **kwargs: Any):
        """Compute Evo-1's flow-matching MSE loss on an SFT batch.

        ``data`` is the batch dict produced by Evo-1's ``custom_collate_fn``:
        keys ``prompts``, ``images``, ``states``, ``actions``, ``action_mask``,
        ``state_mask``, ``image_masks``, ``embodiment_ids``.
        """
        device = next(self.evo1.parameters()).device
        prompts = data["prompts"]
        # images: list (len B) of [num_views, 3, H, W] float tensors -> to device.
        images = [
            img.to(device) if torch.is_tensor(img) else img
            for img in data["images"]
        ]
        image_masks = data["image_masks"].to(device)
        # Match the reference train loop: states/actions in float32.
        states = data["states"].to(device=device, dtype=torch.float32)
        actions_gt = data["actions"].to(device=device, dtype=torch.float32)
        action_mask = data["action_mask"].to(device)
        embodiment_ids = data.get("embodiment_ids")
        if embodiment_ids is not None:
            embodiment_ids = embodiment_ids.to(device)

        fused = self.evo1.get_vl_embeddings(
            images=images,
            image_mask=image_masks,
            prompt=prompts,
            return_cls_only=False,
        )
        pred_velocity, noise = self.evo1(
            fused,
            state=states,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_ids=embodiment_ids,
        )

        bsz = actions_gt.shape[0]
        target_velocity = (actions_gt - noise).view(bsz, -1)
        pred_velocity = pred_velocity.view(bsz, -1)
        mask_flat = action_mask.view(bsz, -1).to(dtype=pred_velocity.dtype)

        pred_v = (pred_velocity * mask_flat).float()
        target_v = (target_velocity * mask_flat).float()
        loss = torch.nn.functional.mse_loss(pred_v, target_v)
        loss = loss * (mask_flat.numel() / (mask_flat.sum() + 1e-8))
        return loss

    # ------------------------------------------------------------------ #
    # RL path -- flow-matching SDE-replay (GRPO-first).
    # ------------------------------------------------------------------ #
    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Replay the stored denoising chains and recompute per-step log-probs.

        Reconstructs the VLM inputs from ``forward_inputs`` (byte prompts + stacked
        pixel values), recomputes the fused embeddings under the *current* policy
        weights, then re-evaluates the Gaussian transition at the stored
        ``chains`` / ``denoise_inds`` (see ``evo1_rl_head.rl_replay_logprob``).
        Returns the RLinf actor contract ``{logprobs, values, entropy}``.
        """
        del compute_logprobs, use_cache, kwargs

        device = next(self.evo1.parameters()).device
        chains = forward_inputs["chains"].to(device)
        denoise_inds = forward_inputs["denoise_inds"].to(device)
        image_mask = forward_inputs["image_mask"].to(device)
        action_mask_3d = forward_inputs["action_mask"].to(device)
        norm_state = forward_inputs["state"].to(device=device, dtype=torch.float32)
        pixel_values = forward_inputs["pixel_values"].to(device)
        prompts = rlh.prompts_from_bytes(
            forward_inputs["prompt_bytes"], forward_inputs["prompt_len"]
        )

        batch_size = pixel_values.shape[0]
        num_views = pixel_values.shape[1]
        images = [
            [pixel_values[b, v] for v in range(num_views)] for b in range(batch_size)
        ]
        embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            fused = self.evo1.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=prompts,
                return_cls_only=False,
            )
            state_t = self.evo1.prepare_state(norm_state)
            chains_log_probs = rlh.rl_replay_logprob(
                self.evo1.action_head,
                fused_tokens=fused,
                state=state_t,
                embodiment_id=embodiment_id,
                action_mask_3d=action_mask_3d,
                chains=chains,
                denoise_inds=denoise_inds,
                denoise_steps=self.rl_denoising_steps,
                rl_cfg=self.rl_head_config,
            )

        log_probs = self._reduce_logprobs(chains_log_probs, denoise_inds)

        values = None
        if compute_values and rlh.cfg_get(self.rl_head_config, "add_value_head", False):
            raise NotImplementedError(
                "Evo-1 value head (PPO/GAE) is not implemented yet (GRPO-first). "
                "Use adv_type=grpo / loss_type=actor for now."
            )

        return {
            "logprobs": log_probs.float(),
            "values": values,
            "entropy": None,
        }
