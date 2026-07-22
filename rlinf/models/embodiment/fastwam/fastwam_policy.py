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

"""RLinf policy wrapper around the upstream FastWAM world-action model.

FastWAM (``Fast-WAM: Do World Action Models Need Test-time Future Imagination?``)
is a Wan2.2 video-diffusion world model with a flow-matching action expert. This
module adapts its self-contained inference / training API to RLinf's
:class:`~rlinf.models.embodiment.base_policy.BasePolicy` contract so the model can
be evaluated (LIBERO / LIBERO-Plus) and SFT-trained through RLinf workers.

The wrapper deliberately *reuses* upstream FastWAM code (``infer_action`` for
rollout, ``training_loss`` for SFT, ``FastWAMProcessor`` for normalization) rather
than re-implementing the model, so behaviour matches the official repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from fastwam.models.wan22.fastwam import FastWAM

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

# Upstream FastWAM. The prompt template and gripper helper are vendored here as
# small constants to avoid importing the (non-packaged) ``experiments`` scripts.
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip the gripper open/close sign (matches FastWAM ``invert_gripper_action``)."""
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


@dataclass
class FastWAMPolicyConfig:
    """RLinf-facing FastWAM policy configuration."""

    action_horizon: int = 32
    num_action_chunks: int = 8
    num_inference_steps: int = 10
    binarize_gripper: bool = True
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    sigma_shift: Optional[float] = None
    seed: Optional[int] = None
    rand_device: str = "cpu"
    tiled: bool = False
    concat_multi_camera: str = "horizontal"
    visualize_future_video: bool = False
    future_video_dir: Optional[str] = None
    num_video_frames: int = 9
    max_video_saves: int = 12


class FastWAMPolicy(FastWAM, BasePolicy):
    """Adapter exposing FastWAM through RLinf's embodied policy interface.

    Args:
        processor: A ``FastWAMProcessor`` whose normalizer is already populated
            from ``dataset_stats.json`` (action/state min-max stats).
        policy_cfg: Resolved RLinf policy configuration.
    """

    def configure_rlinf(
        self,
        processor: Any,
        policy_cfg: FastWAMPolicyConfig,
    ) -> "FastWAMPolicy":
        """Attach RLinf preprocessing and rollout configuration."""
        self.processor = processor
        self.policy_cfg = policy_cfg

        # Cache the (single) merged action/state keys used by the processor.
        self._state_key = processor.shape_meta["state"][0]["key"]
        self._action_key = processor.shape_meta["action"][0]["key"]
        # Per-camera target HxW from shape_meta (e.g. [3,224,224] -> 224,224).
        self._cam_hw = [
            (int(m["shape"][1]), int(m["shape"][2]))
            for m in processor.shape_meta["images"]
        ]
        self._num_cameras = int(processor.num_output_cameras)
        self._video_saves = 0  # count of predicted-future clips saved so far
        return self

    def _save_future_video(self, frames) -> None:
        """Save a predicted future-video clip (list of PIL frames) as an MP4."""
        import os

        from fastwam.utils.video_io import save_mp4

        out_dir = self.policy_cfg.future_video_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"future_{self._video_saves:04d}.mp4")
        save_mp4(frames, path, fps=8)
        self._video_saves += 1

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _center_crop_resize_batch(
        images: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize and center-crop a ``[B,H,W,C]`` image batch."""
        images = images.permute(0, 3, 1, 2).float()
        src_height, src_width = images.shape[-2:]
        scale = max(width / src_width, height / src_height)
        resized_height = round(src_height * scale)
        resized_width = round(src_width * scale)
        images = F.interpolate(
            images,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        top = max((resized_height - height) // 2, 0)
        left = max((resized_width - width) // 2, 0)
        return images[:, :, top : top + height, left : left + width]

    def _build_input_images(self, main_images, wrist_images) -> torch.Tensor:
        """Build a batched FastWAM input frame tensor in ``[-1, 1]``."""
        main_images = torch.as_tensor(main_images)
        if self._num_cameras == 1:
            height, width = self._cam_hw[0]
            images = self._center_crop_resize_batch(main_images, height, width)
        else:
            if wrist_images is None:
                raise ValueError(
                    "FastWAM requires wrist_images for a two-camera policy."
                )
            height_0, width_0 = self._cam_hw[0]
            height_1, width_1 = self._cam_hw[1]
            primary = self._center_crop_resize_batch(
                main_images,
                height_0,
                width_0,
            )
            wrist = self._center_crop_resize_batch(
                torch.as_tensor(wrist_images),
                height_1,
                width_1,
            )
            concat_dim = 2 if self.policy_cfg.concat_multi_camera == "vertical" else 3
            images = torch.cat([primary, wrist], dim=concat_dim)
        images = images.to(device=self.device, dtype=self.torch_dtype)
        return images * (2.0 / 255.0) - 1.0

    def _normalize_proprio(self, states) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=torch.float32).cpu()
        if states.ndim == 1:
            states = states.unsqueeze(0)
        batch = {"state": {self._state_key: states}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self._state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self._action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str | Sequence[str]],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Run FastWAM action denoising for a full observation batch."""
        del negative_prompt, text_cfg_scale
        self.eval()
        if (
            str(getattr(self.video_expert, "video_attention_mask_mode", ""))
            != "first_frame_causal"
        ):
            raise ValueError(
                "`infer_action` requires "
                "`video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                "`input_image` must have shape [B,3,H,W] or [3,H,W], "
                f"got {tuple(input_image.shape)}"
            )
        batch_size, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "`input_image` must be resized before inference; expected "
                f"multiples of 16 but got HxW=({height},{width})."
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None`, so the "
                    "proprio encoder is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (
                batch_size,
                self.proprio_dim,
            ):
                raise ValueError(
                    "`proprio` must have shape "
                    f"[{batch_size},{self.proprio_dim}], got {tuple(proprio.shape)}."
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_video_latents(
            input_image.unsqueeze(2),
            tiled=tiled,
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are exclusive.")
        if not use_prompt and not use_context:
            raise ValueError(
                "Provide either `prompt` or both `context` and `context_mask`."
            )

        if use_prompt:
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} prompts, received {len(prompt)}."
                )
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if (
                context.ndim != 3
                or context_mask.ndim != 2
                or context.shape[0] != batch_size
                or context_mask.shape[0] != batch_size
            ):
                raise ValueError(
                    "`context/context_mask` must have shape [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}."
                )
            context = context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            context_mask = context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )

        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (batch_size,),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps, infer_deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.to(
                dtype=latents_action.dtype,
                device=self.device,
            ).expand(batch_size)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action,
                step_delta,
                latents_action,
            )

        return {
            "action": latents_action.detach().to(
                device="cpu",
                dtype=torch.float32,
            )
        }

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "eval", **kwargs):
        """Predict an action chunk for a batch of LIBERO observations.

        Args:
            env_obs: dict from :class:`~rlinf.envs.libero.libero_env.LiberoEnv`
                with ``main_images`` / ``wrist_images`` ``[B,H,W,3]`` uint8,
                ``states`` ``[B,8]`` and ``task_descriptions`` list[str].

        Returns:
            (actions, result): ``actions`` is ``np.ndarray`` of shape
            ``[B, num_action_chunks, action_dim]`` ready for ``env.chunk_step``;
            ``result`` is an (empty) metadata dict for the eval path.
        """
        cfg = self.policy_cfg
        main_images = env_obs["main_images"]
        wrist_images = env_obs.get("wrist_images")
        states = env_obs["states"]
        task_descriptions = env_obs["task_descriptions"]
        batch_size = int(main_images.shape[0])

        images = self._build_input_images(main_images, wrist_images)
        proprio = self._normalize_proprio(states)
        prompts = [
            DEFAULT_PROMPT.format(task=str(task_descriptions[index]))
            for index in range(batch_size)
        ]
        common = {
            "prompt": prompts,
            "input_image": images,
            "action_horizon": cfg.action_horizon,
            "proprio": proprio,
            "negative_prompt": cfg.negative_prompt,
            "text_cfg_scale": cfg.text_cfg_scale,
            "num_inference_steps": cfg.num_inference_steps,
            "sigma_shift": cfg.sigma_shift,
            "seed": cfg.seed,
            "rand_device": cfg.rand_device,
            "tiled": cfg.tiled,
        }
        pred = self.infer_action(**common)
        actions = self._denormalize_action(pred["action"])

        do_viz = (
            cfg.visualize_future_video
            and cfg.future_video_dir is not None
            and self._video_saves < cfg.max_video_saves
        )
        if do_viz:
            video_pred = super().infer_joint(
                prompt=prompts[0],
                input_image=images[:1],
                action_horizon=cfg.action_horizon,
                proprio=proprio[:1],
                negative_prompt=cfg.negative_prompt,
                text_cfg_scale=cfg.text_cfg_scale,
                num_inference_steps=cfg.num_inference_steps,
                sigma_shift=cfg.sigma_shift,
                seed=cfg.seed,
                rand_device=cfg.rand_device,
                tiled=cfg.tiled,
                num_video_frames=cfg.num_video_frames,
                test_action_with_infer_action=False,
            )
            self._save_future_video(video_pred["video"])

        # FastWAM normalizes gripper as 0=close, 1=open. Convert it back to
        # LIBERO's -1=open, +1=close convention before execution.
        actions[..., -1] = actions[..., -1] * 2 - 1
        actions = _invert_gripper_action(actions)
        if cfg.binarize_gripper:
            actions[..., -1] = np.sign(actions[..., -1])

        actions = actions[:, : cfg.num_action_chunks].astype(np.float32)
        return actions, {}

    # ------------------------------------------------------------------ SFT
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"FastWAMPolicy does not support {forward_type}.")

    def sft_forward(self, data: Any = None, **kwargs):
        """Supervised flow-matching loss over a FastWAM sample batch.

        ``data`` is a dict produced by FastWAM's ``RobotVideoDataset`` collate
        (``video``, ``action``, ``context``, ``context_mask`` and optional
        ``proprio`` / ``*_is_pad``). Returns a dict with at least ``loss``.
        """
        if data is None:
            raise ValueError("FastWAMPolicy.sft_forward requires `data`.")
        loss, metrics = self.training_loss(data)
        out = {"loss": loss}
        # Surface video / action sub-losses for logging (worker reads .item()).
        if isinstance(metrics, dict):
            if "loss_video" in metrics:
                out["dynamics_loss"] = torch.as_tensor(metrics["loss_video"])
            if "loss_action" in metrics:
                out["action_loss"] = torch.as_tensor(metrics["loss_action"])
        return out

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "FastWAMPolicy.default_forward is unused; rollout uses predict_action_batch "
            "and SFT uses sft_forward."
        )

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enable checkpointing through RLinf's FSDP model manager."""
        del gradient_checkpointing_kwargs
        for expert in (self.video_expert, self.action_expert):
            expert.use_gradient_checkpointing = True
        self.mot.mot_checkpoint_mixed_attn = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable checkpointing on both FastWAM experts."""
        for expert in (self.video_expert, self.action_expert):
            expert.use_gradient_checkpointing = False
        self.mot.mot_checkpoint_mixed_attn = False
