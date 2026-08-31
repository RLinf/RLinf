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

import json
import os
from collections import deque
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype

from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.backend import FrameQueue, autocast

__all__ = ["OpenSoraBackend"]


class OpenSoraBackend:
    """In-process backend holding the action-conditioned OpenSora STDiT and its VAE.

    A session holds the trajectory's condition window as a queue of latents, so pixels are
    only ever decoded on the way out.
    """

    def __init__(self, cfg, device: torch.device, device_str: str):
        self.cfg = cfg
        self.wm_cfg = cfg.world_model_cfg
        self.device = device
        self.inference_dtype = to_torch_dtype(self.wm_cfg.get("dtype", "bf16"))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.chunk = self.wm_cfg.chunk  # Ta
        self.condition_frame_length = self.wm_cfg.condition_frame_length  # To
        self.num_frames = self.chunk + self.condition_frame_length
        self.image_size = tuple(self.wm_cfg.image_size)

        self.vae = self._load_vae().eval().to(self.device, self.inference_dtype)
        self.model = self._load_model().eval().to(self.device, self.inference_dtype)
        self.scheduler = self._load_scheduler()

        # Determine VAE type for frame calculations
        vae_type = self.wm_cfg.vae.get("type", "OpenSoraVAE_V1_2")
        self.is_vae_v1_2 = vae_type == "OpenSoraVAE_V1_2"
        self.z_mask_frame_num = int(self.chunk / 4 if self.is_vae_v1_2 else self.chunk)
        self.z_condition_frame_length = int(
            self.condition_frame_length / 4
            if self.is_vae_v1_2
            else self.condition_frame_length
        )

        self.action_stats = self._load_action_stats()

        self.fps = self.wm_cfg.get("fps", 3.0)
        self.multi_resolution = self.wm_cfg.get("multi_resolution", "STDiT2")
        self.model_args = prepare_multi_resolution_info(
            self.multi_resolution,
            1,
            self.image_size,
            self.num_frames,
            self.fps,
            self.device,
            self.inference_dtype,
        )

        self._sessions: dict[int, dict[str, Any]] = {}

    def _load_vae(self):
        vae_cfg = OmegaConf.to_container(self.wm_cfg.vae, resolve=True)
        return build_module(vae_cfg, MODELS)

    def _load_model(self):
        latent_size = self.vae.get_latent_size((self.num_frames, *self.image_size))
        model_cfg = OmegaConf.to_container(self.wm_cfg.model, resolve=True)
        return build_module(
            model_cfg,
            MODELS,
            input_size=latent_size,
            in_channels=self.vae.out_channels,
            enable_sequence_parallelism=False,
        )

    def _load_scheduler(self):
        scheduler_cfg = OmegaConf.to_container(self.wm_cfg.scheduler, resolve=True)
        return build_module(scheduler_cfg, SCHEDULERS)

    def _load_action_stats(self) -> dict[str, np.ndarray]:
        stats_path = self.wm_cfg.get("stats_path", None)
        if stats_path is None or not os.path.exists(stats_path):
            raise ValueError(f"Action stats path {stats_path} does not exist")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return {
            "q01": np.asarray(stats["action"]["q01"], np.float32),
            "q99": np.asarray(stats["action"]["q99"], np.float32),
        }

    def _normalize_action(self, actions: np.ndarray) -> np.ndarray:
        """Map actions into ``[-1, 1]`` with the dataset's q01 / q99 statistics."""
        q01 = self.action_stats["q01"]
        q99 = self.action_stats["q99"]
        return 2 * ((actions - q01) / (q99 - q01)) - 1

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        task_ids: Sequence[Any],
        seeds: Sequence[int],
    ) -> None:
        """Encode the initial condition frames into each session's latent queue.

        ``init_actions`` is unused: OpenSora conditions on the action chunk only.
        """
        windows = torch.stack(
            [torch.cat(list(frames), dim=1) for frames in init_frames], dim=0
        )  # [B, C, T, H, W]
        batch_size, channels, window_len, height, width = windows.shape
        flat = (
            windows.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * window_len, channels, height, width)
            .to(self.device, self.inference_dtype)
        )

        with torch.no_grad():
            encoded = self.vae.encode(flat.unsqueeze(2))  # [B * T, C', 1, H', W']

        encoded = encoded.squeeze(2)
        encoded = encoded.reshape(batch_size, window_len, *encoded.shape[1:])
        encoded = encoded.permute(0, 2, 1, 3, 4)  # [B, C', T, H', W']

        for row, (env_id, task_id, seed) in enumerate(zip(env_ids, task_ids, seeds)):
            queue = deque(maxlen=self.z_condition_frame_length)
            for t_idx in range(window_len):
                queue.append(encoded[row : row + 1, :, t_idx : t_idx + 1, :, :])
            self._sessions[int(env_id)] = {
                "task_id": task_id,
                "seed": int(seed),
                "latents": queue,
            }

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    def _session_latents(self, env_ids: Sequence[int]) -> torch.Tensor:
        missing = [int(i) for i in env_ids if int(i) not in self._sessions]
        if missing:
            raise RuntimeError(
                f"generate called for env slots without a session: {missing}"
            )
        return torch.stack(
            [
                torch.concat(list(self._sessions[int(i)]["latents"]), dim=2).squeeze(0)
                for i in env_ids
            ],
            dim=0,
        )  # [B, C', T_cond, H', W']

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

        actions_np = (
            actions if isinstance(actions, np.ndarray) else actions.cpu().numpy()
        )
        actions_batch = (
            torch.from_numpy(self._normalize_action(actions_np))
            .to(self.device, self.inference_dtype)
            .reshape(batch_size, -1, actions.shape[-1])
        )

        latent_size = self.vae.get_latent_size((self.num_frames, *self.image_size))
        condition = self._session_latents(env_ids)

        with autocast(self.device, self.inference_dtype):
            # The scheduler draws noise from the global RNG, so session seeds go unused.
            z = torch.randn(
                batch_size,
                self.vae.out_channels,
                self.z_mask_frame_num,
                *latent_size[1:],
                device=self.device,
                dtype=self.inference_dtype,
            )
            masks = torch.tensor(
                [[0] * self.z_condition_frame_length + [1] * self.z_mask_frame_num]
                * batch_size,
                device=self.device,
                dtype=self.inference_dtype,
            )
            samples = self.scheduler.sample(
                self.model,
                z=torch.concat([condition, z], dim=2),
                y=actions_batch,
                device=self.device,
                additional_args=self.model_args,
                progress=False,
                mask=masks,
            )
            pred_latents = samples[:, :, -self.z_mask_frame_num :, :, :].to(
                self.inference_dtype
            )

            # Roll latents, not pixels, into the windows; the next chunk conditions on them.
            frames_per_env = self.z_mask_frame_num if self.is_vae_v1_2 else self.chunk
            for row, env_id in enumerate(env_ids):
                queue = self._sessions[int(env_id)]["latents"]
                for latent in (
                    pred_latents[row : row + 1].clone().chunk(frames_per_env, dim=2)
                ):
                    queue.append(latent)

            if self.is_vae_v1_2:
                return self.vae.decode(pred_latents, num_frames=12)
            return self.vae.decode(pred_latents)

    def offload(self) -> None:
        self.vae = self.vae.to("cpu")
        self.model = self.model.to("cpu")
        self._move_sessions("cpu")

    def onload(self) -> None:
        self.vae = self.vae.to(self.device, self.inference_dtype)
        self.model = self.model.to(self.device, self.inference_dtype)
        self._move_sessions(self.device)

    def _move_sessions(self, device) -> None:
        for session in self._sessions.values():
            session["latents"] = deque(
                [recursive_to_device(latent, device) for latent in session["latents"]],
                maxlen=self.z_condition_frame_length,
            )
