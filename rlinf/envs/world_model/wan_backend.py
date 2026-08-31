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

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from PIL import Image

from rlinf.envs.world_model.backend import FrameQueue, autocast

__all__ = ["WanBackend"]


class WanBackend:
    """In-process backend holding diffsynth's action-conditioned ``WanVideoPipeline``.

    A session holds the trajectory's reference frame, its last generated frames and the
    actions that produced them.
    """

    def __init__(self, cfg, device: torch.device, device_str: str):
        self.cfg = cfg
        self.device = device
        self.num_inference_steps = cfg.num_inference_steps
        self.num_frames = cfg.num_frames
        self.condition_frame_length = cfg.condition_frame_length
        self.chunk = cfg.chunk
        self.image_size = tuple(cfg.image_size)
        self.retain_action = cfg.get("retain_action", True)
        if self.num_frames != self.condition_frame_length + self.chunk:
            raise ValueError(
                f"num_frames must be condition_frame_length + chunk; got {self.num_frames} != "
                f"{self.condition_frame_length} + {self.chunk}"
            )
        self._sessions: dict[int, dict[str, Any]] = {}
        self._pipe = self._build_pipeline(device_str)

    def _build_pipeline(self, device_str: str) -> WanVideoPipeline:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,
            model_configs=[
                ModelConfig(path=self.cfg.model_path, offload_device="cpu"),
                ModelConfig(path=self.cfg.VAE_path, offload_device="cpu"),
            ],
        )
        pipe.dit.to(self.device)
        pipe.vae.to(self.device)
        return pipe

    @staticmethod
    def _to_pil(frame: torch.Tensor) -> Image.Image:
        """``[C, 1, H, W]`` in ``[-1, 1]`` or ``[0, 1]`` to the pipeline's uint8 image."""
        img = np.transpose(frame[:, 0].cpu().numpy(), (1, 2, 0))
        if img.max() <= 1.2:
            img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255)
        return Image.fromarray(img.astype(np.uint8))

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        task_ids: Sequence[Any],
        seeds: Sequence[int],
    ) -> None:
        for row, (env_id, task_id, seed) in enumerate(zip(env_ids, task_ids, seeds)):
            self._sessions[int(env_id)] = {
                "task_id": task_id,
                "seed": int(seed),
                "frames": [self._to_pil(f) for f in init_frames[row]],
                "actions": init_actions[row].clone(),
            }

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    def _batch_seed(self, env_ids: Sequence[int]) -> int:
        missing = [int(i) for i in env_ids if int(i) not in self._sessions]
        if missing:
            raise RuntimeError(
                f"generate called for env slots without a session: {missing}"
            )
        seeds = {self._sessions[int(i)]["seed"] for i in env_ids}
        if len(seeds) > 1:
            raise NotImplementedError(
                f"WanVideoPipeline draws one noise tensor per batch, so the batch needs a single "
                f"seed; got {sorted(seeds)}"
            )
        return seeds.pop()

    def _window_actions(
        self, env_ids: Sequence[int], actions: torch.Tensor
    ) -> torch.Tensor:
        """Prepend each session's action history to its chunk, then roll it forward."""
        history = torch.stack(
            [self._sessions[int(i)]["actions"] for i in env_ids], dim=0
        ).to(device=actions.device, dtype=actions.dtype)

        if self.retain_action:
            actions = torch.cat([history, actions], dim=1)

        tail = actions[:, -(self.condition_frame_length - 1) :, :]
        for row, env_id in enumerate(env_ids):
            session_actions = self._sessions[int(env_id)]["actions"]
            session_actions[1 : self.condition_frame_length] = tail[row].to(
                device=session_actions.device, dtype=session_actions.dtype
            )
        return actions

    def _pipe_kwargs(
        self, env_ids: Sequence[int], actions: torch.Tensor
    ) -> dict[str, Any]:
        windows = [self._sessions[int(i)]["frames"] for i in env_ids]

        return {
            "seed": self._batch_seed(env_ids),
            "tiled": False,
            "input_image": [frames[0] for frames in windows],
            "input_image4": [frames[-4:] for frames in windows],
            "action": self._window_actions(env_ids, actions),
            "height": 256,
            "width": 256,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "cfg_scale": 1.0,
            "progress_bar_cmd": lambda x: x,
            "batch_size": len(windows),
        }

    def generate(
        self,
        env_ids: Sequence[int],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(env_ids)
        if actions.shape[0] != batch_size:
            raise ValueError(
                f"env_ids and actions must describe the same batch rows; got "
                f"{batch_size}, {actions.shape[0]}"
            )
        actions = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        )
        with autocast(self.device, torch.bfloat16):
            output = self._pipe(**self._pipe_kwargs(env_ids, actions))

        videos = []
        for env_idx, env_id in enumerate(env_ids):
            # Keep the pipeline's own frames as-is; a [-1, 1] round trip costs a gray level.
            window = self._sessions[int(env_id)]["frames"]
            window[1:] = output[env_idx][-(self.condition_frame_length - 1) :]

            frames = []
            # The pipeline regenerates the conditioned frames; only the new ones leave here.
            for img in output[env_idx][self.condition_frame_length :]:
                # Keep frame tensors in fp32 to avoid silent fp64 promotion
                # that can significantly increase GPU memory usage.
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = arr * 2.0 - 1.0
                frames.append(arr)

            video = np.stack(frames, axis=0)  # [T, H, W, 3]
            video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
            video = torch.from_numpy(video)
            videos.append(video.transpose(0, 1))  # [3, T, H, W]

        return torch.stack(videos, dim=0)

    def offload(self) -> None:
        self._pipe.vae = self._pipe.vae.to("cpu")
        self._pipe.dit = self._pipe.dit.to("cpu")

    def onload(self) -> None:
        self._pipe.dit = self._pipe.dit.to(self.device)
        self._pipe.vae = self._pipe.vae.to(self.device)
