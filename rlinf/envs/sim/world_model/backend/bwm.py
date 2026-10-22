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

from typing import Any, Optional, Sequence

import numpy as np
import torch

from rlinf.envs.sim.world_model.registry import register_backend

from . import FrameQueue
from .robotwin_action_bridge import RobotwinActionBridge

__all__ = ["BwmBackend"]

# BWM's latent grid takes 4k + 1 frames: 9 condition frames + 52 future frames. A chunk
# fills 50 of them; the last command is held for the other 2, which are discarded.
_HISTORY = 9
_CHUNK = 50
_FUTURE = 52
# Training noises history latents as (1 - s) z + s eps; BWM's inference applies it at
# sigma = 1 for 50 steps, which replaces the history with noise.
_HISTORY_SIGMA = 0.2083


def _history_condition_latents(
    pipe, inputs_shared: dict, *, use_history_condition_noise_in_inference: bool
):
    first = inputs_shared.get("first_frame_latents")
    latents = inputs_shared.get("latents")
    if first is None or latents is None:
        return None, 0
    if first.ndim == 4:
        first = first.unsqueeze(0)
    history_t = min(int(first.shape[2]), int(latents.shape[2]))
    if history_t <= 0:
        return None, 0
    cond = first[:, :, :history_t].clone()
    if use_history_condition_noise_in_inference and history_t > 1:
        # inputs_shared["noise"] aliases latents, whose history slots already hold z.
        generator = torch.Generator("cpu").manual_seed(
            int(inputs_shared.get("seed") or 0) + 7919
        )
        eps = torch.randn(cond[:, :, 1:].shape, generator=generator)
        eps = eps.to(cond.device, cond.dtype)
        cond[:, :, 1:] = (1 - _HISTORY_SIGMA) * cond[:, :, 1:] + _HISTORY_SIGMA * eps
    latents[:, :, :history_t] = cond
    return cond, history_t


@register_backend("bwm")
class BwmBackend:
    """In-process backend for the Boundless World Model on RoboTwin's Aloha.

    The policy commands joint angles; BWM was trained on normalised absolute end-effector
    poses, so every action is converted by :class:`RobotwinActionBridge` before it reaches
    the pipeline. A session holds the reset frame, the last 8 generated frames and the
    poses that produced them.
    """

    supports_kir = False

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.chunk = int(cfg.chunk)
        self.condition_frame_length = int(cfg.condition_frame_length)
        self.image_size = tuple(cfg.image_size)
        if (self.condition_frame_length, self.chunk) != (_HISTORY, _CHUNK):
            raise ValueError(
                f"BWM generates windows of {_HISTORY} condition + {_CHUNK} new frames; "
                f"got {self.condition_frame_length} + {self.chunk}"
            )
        self.num_inference_steps = int(cfg.num_inference_steps)
        bridge = cfg.action_bridge
        self._bridge = RobotwinActionBridge(
            urdf_path=bridge.urdf_path,
            srdf_path=bridge.srdf_path,
            stat_path=bridge.stat_path,
            robot_pose=bridge.robot_pose,
        )
        rest = self._bridge(np.asarray(cfg.rest_action, dtype=np.float32))
        self._rest_history = torch.from_numpy(rest)[None].repeat(_HISTORY, 1)
        self._sessions: dict[int, dict[str, Any]] = {}
        self._pipe = self._build_pipeline()

    @staticmethod
    def load_reward_model(cfg):
        from rlinf.models.embodiment.reward.robotwin_t5_reward_model import (
            RoboTwinT5RewardModel,
        )

        return RoboTwinT5RewardModel(cfg.reward_model)

    def reward_instructions(self, env) -> Optional[list[str]]:
        instructions = []
        for env_idx in range(env.num_envs):
            instructions.extend([env.task_descriptions[env_idx]] * self.chunk)
        return instructions

    def _build_pipeline(self):
        from diffsynth.core import ModelConfig
        from wan_video_action.pipelines import wan_video_action
        from wan_video_action.utils import resolve_model_path

        wan_video_action._prepare_history_condition_latents = _history_condition_latents
        pipe = wan_video_action.build_wan_video_action_pipeline(
            torch_dtype=torch.bfloat16,
            device=str(self.device),
            model_configs=[
                ModelConfig(path=resolve_model_path(self.cfg.model_path)),
                ModelConfig(path=resolve_model_path(self.cfg.VAE_path)),
            ],
            tokenizer_config=None,
            ckpt_path=self.cfg.checkpoint_path,
            action_dim=int(self.cfg.action_dim),
            action_mode="adaln",
        )
        pipe.use_gradient_checkpointing = False
        pipe.use_gradient_checkpointing_offload = False
        pipe.eval()
        return pipe

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        seeds: Sequence[int],
    ) -> None:
        # The history poses come from the configured rest pose, not init_actions: the env
        # builds those in joint space, and a zero pose is mid-workspace after the bridge.
        for row, (env_id, seed) in enumerate(zip(env_ids, seeds)):
            self._sessions[int(env_id)] = {
                "seed": int(seed),
                "frames": [f[:, 0].detach().float().cpu() for f in init_frames[row]],
                "actions": self._rest_history.clone(),
                "step_id": 0,
            }

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    def _generate_window(
        self, frames: list[torch.Tensor], actions: torch.Tensor, seed: int
    ) -> torch.Tensor:
        """9 condition frames and 59 poses to the 50 new frames, ``[3, 50, H, W]``."""
        held = actions[-1:].repeat(_FUTURE - _CHUNK, 1)
        actions = torch.cat([actions, held], dim=0)[None]
        video = torch.stack(frames, dim=1)[None]
        height, width = self.image_size
        out = self._pipe(
            input_video=video.to(self.device, self._pipe.torch_dtype),
            action=actions.to(self.device, self._pipe.torch_dtype),
            seed=seed,
            rand_device="cpu",
            tiled=False,
            height=height,
            width=width,
            num_frames=_HISTORY + _FUTURE,
            num_history_frames=_HISTORY,
            cfg_scale=1.0,
            num_inference_steps=self.num_inference_steps,
            use_history_condition_noise_in_inference=True,
            progress_bar_cmd=lambda it, *args, **kwargs: it,
            output_type="floatpoint",
        )
        return out[0, :, _HISTORY : _HISTORY + _CHUNK].detach().float().cpu()

    @torch.no_grad()
    def generate(self, env_ids: Sequence[int], actions: torch.Tensor) -> torch.Tensor:
        missing = [int(i) for i in env_ids if int(i) not in self._sessions]
        if missing:
            raise RuntimeError(
                f"generate called for env slots without a session: {missing}"
            )
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        poses = torch.from_numpy(self._bridge(actions))
        keep = _HISTORY - 1
        videos = []
        for row, env_id in enumerate(env_ids):
            session = self._sessions[int(env_id)]
            window = torch.cat([session["actions"], poses[row]], dim=0)
            new = self._generate_window(
                session["frames"], window, session["seed"] + session["step_id"]
            )
            session["frames"][1:] = [new[:, i] for i in range(_CHUNK - keep, _CHUNK)]
            session["actions"][1:] = window[-keep:]
            session["step_id"] += 1
            videos.append(new)
        return torch.stack(videos, dim=0)

    def offload(self) -> None:
        for name in ("dit", "vae", "action_encoder"):
            getattr(self._pipe, name).to("cpu")

    def onload(self) -> None:
        for name in ("dit", "vae", "action_encoder"):
            getattr(self._pipe, name).to(self.device)
