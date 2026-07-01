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

from typing import Any

import numpy as np
import torch
import torch.nn as nn

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


def _center_crop_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize + center crop to (height, width).

    Mirrors ``experiments/libero/eval_libero_single.py::_center_crop_resize`` so
    the pixels handed to the model match the official LIBERO evaluation exactly.
    """
    from PIL import Image

    pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    resized = pil.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


class FastWAMPolicy(nn.Module, BasePolicy):
    """Adapter exposing FastWAM through RLinf's embodied policy interface.

    Args:
        model: An instantiated upstream ``fastwam.models.wan22.fastwam.FastWAM``.
        processor: A ``FastWAMProcessor`` whose normalizer is already populated
            from ``dataset_stats.json`` (action/state min-max stats).
        infer_cfg: Resolved inference hyper-parameters (see :class:`FastWAMInferConfig`).
    """

    def __init__(self, model: nn.Module, processor: Any, infer_cfg: "FastWAMInferConfig"):
        nn.Module.__init__(self)
        self.model = model
        self.processor = processor
        self.infer_cfg = infer_cfg

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

    def _save_future_video(self, frames) -> None:
        """Save a predicted future-video clip (list of PIL frames) as an MP4."""
        import os

        from fastwam.utils.video_io import save_mp4

        out_dir = self.infer_cfg.future_video_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"future_{self._video_saves:04d}.mp4")
        save_mp4(frames, path, fps=8)
        self._video_saves += 1

    # ------------------------------------------------------------------ utils
    @property
    def device(self):
        return getattr(self.model, "device", next(self.model.parameters()).device)

    @property
    def torch_dtype(self):
        return getattr(self.model, "torch_dtype", torch.bfloat16)

    def _build_input_image(self, main_img, wrist_img) -> torch.Tensor:
        """Build the FastWAM input frame: per-cam crop/resize, h-concat, [-1, 1]."""
        main_np = main_img.detach().cpu().numpy() if torch.is_tensor(main_img) else np.asarray(main_img)
        if self._num_cameras == 1:
            h, w = self._cam_hw[0]
            rgb = _center_crop_resize(main_np, width=w, height=h)
        else:
            h0, w0 = self._cam_hw[0]
            h1, w1 = self._cam_hw[1]
            primary = _center_crop_resize(main_np, width=w0, height=h0)
            wrist_np = (
                wrist_img.detach().cpu().numpy()
                if torch.is_tensor(wrist_img)
                else np.asarray(wrist_img)
            )
            wrist = _center_crop_resize(wrist_np, width=w1, height=h1)
            if self.infer_cfg.concat_multi_camera == "vertical":
                rgb = np.concatenate([primary, wrist], axis=0)
            else:
                rgb = np.concatenate([primary, wrist], axis=1)
        x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=self.device, dtype=self.torch_dtype)
        x = x * (2.0 / 255.0) - 1.0
        return x

    def _normalize_proprio(self, state_vec) -> torch.Tensor:
        state = state_vec.detach().cpu() if torch.is_tensor(state_vec) else torch.as_tensor(state_vec)
        state = state.to(dtype=torch.float32).reshape(-1)
        batch = {"state": {self._state_key: state.unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self._state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self._action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

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
        cfg = self.infer_cfg
        main_images = env_obs["main_images"]
        wrist_images = env_obs.get("wrist_images")
        states = env_obs["states"]
        task_descriptions = env_obs["task_descriptions"]
        batch_size = int(main_images.shape[0])

        chunks = []
        for i in range(batch_size):
            image = self._build_input_image(
                main_images[i], None if wrist_images is None else wrist_images[i]
            )
            proprio = self._normalize_proprio(states[i])
            prompt = DEFAULT_PROMPT.format(task=str(task_descriptions[i]))

            common = dict(
                prompt=prompt,
                input_image=image,
                action_horizon=cfg.action_horizon,
                proprio=proprio,
                negative_prompt=cfg.negative_prompt,
                text_cfg_scale=cfg.text_cfg_scale,
                num_inference_steps=cfg.num_inference_steps,
                sigma_shift=cfg.sigma_shift,
                seed=cfg.seed,
                rand_device=cfg.rand_device,
                tiled=cfg.tiled,
            )
            # World-model imagination: also decode the predicted future video for
            # env 0 (capped) when enabled. infer_joint denoises video latents + VAE
            # decode (slower); the fast action-only path is used otherwise.
            do_viz = (
                cfg.visualize_future_video
                and cfg.future_video_dir is not None
                and i == 0
                and self._video_saves < cfg.max_video_saves
            )
            if do_viz:
                pred = self.model.infer_joint(
                    num_video_frames=cfg.num_video_frames,
                    test_action_with_infer_action=False,
                    **common,
                )
                self._save_future_video(pred["video"])
            else:
                pred = self.model.infer_action(**common)
            action = self._denormalize_action(pred["action"])[0]  # [T, D]

            # FastWAM dataloader maps gripper to (0=close, 1=open); undo that and
            # restore LIBERO's (-1=open, +1=close) convention before execution.
            action[..., -1] = action[..., -1] * 2 - 1
            action = _invert_gripper_action(action)
            if cfg.binarize_gripper:
                action[..., -1] = np.sign(action[..., -1])

            chunks.append(action[: cfg.num_action_chunks])

        actions = np.stack(chunks, axis=0).astype(np.float32)
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
        loss, metrics = self.model.training_loss(data)
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

    def gradient_checkpointing_enable(self, **kwargs):
        # FastWAM toggles gradient checkpointing through its DiT configs
        # (``use_gradient_checkpointing`` / ``mot_checkpoint_mixed_attn``); this is a
        # no-op so RLinf's FSDP manager can call it uniformly.
        return
