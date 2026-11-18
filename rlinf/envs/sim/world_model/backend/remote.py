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

"""Backend that advances frames on a world model served over HTTP.

The env-facing verbs are fixed (``open_session`` / ``generate`` / ``close_session``). The
request body is owned by an adapter looked up by ``remote.adapter`` (see
:func:`register_wm_adapter`), and the round trip, multipart encoding and retries by
:class:`~rlinf.utils.http_client.InferenceHTTPClient`. Server addresses follow the SGLang
embodied worker: the driver hands ``remote.server_urls`` to the env workers, each of which
picks one by ``rank % len(urls)`` and passes it in as ``server_url``.

A full-sequence diffusion world model keeps no history across calls, so the condition window
stays on this side and rides along with every request. That also makes a retry free: the
window only advances after a response arrives, so re-sending a step cannot double-advance the
trajectory.
"""

from __future__ import annotations

import io
import json
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from rlinf.envs.sim.world_model.registry import get_backend, register_backend
from rlinf.utils.http_client import InferenceHTTPClient

from . import FrameQueue
from .robotwin_action_bridge import RobotwinActionBridge

__all__ = ["RemoteWorldModelBackend", "get_wm_adapter_cls", "register_wm_adapter"]

_WM_ADAPTER_REGISTRY: dict[str, Callable[[], type]] = {}


def register_wm_adapter(name: str, builder: Callable[[], type], force: bool = False):
    """Register a lazy world-model adapter builder for ``remote.adapter``.

    ``builder`` is a zero-arg callable returning the adapter class; keep heavy imports
    inside it. An adapter is built as ``cls(cfg, remote_cfg)`` and provides ``path``,
    ``build_request(cond_frames, actions, seed) -> (fields, files, state)`` and
    ``parse_response(resp_bytes, state) -> [T, H, W, 3] uint8``, one frame per action row.
    """
    if not name:
        raise ValueError("adapter name must be a non-empty string.")
    key = name.lower()
    if not force and key in _WM_ADAPTER_REGISTRY:
        raise ValueError(
            f"world model adapter `{key}` is already registered. "
            "Set force=True to override it."
        )
    _WM_ADAPTER_REGISTRY[key] = builder


def get_wm_adapter_cls(name: str) -> type:
    """Return the world-model adapter class registered for ``name``."""
    builder = _WM_ADAPTER_REGISTRY.get(str(name).lower())
    if builder is None:
        raise ValueError(
            f"unknown world model adapter {name!r}; "
            f"available: {sorted(_WM_ADAPTER_REGISTRY)}"
        )
    return builder()


@register_backend("remote")
class RemoteWorldModelBackend:
    """Advances frames on a served world model.

    The benchmark's conventions stay on this side, as they do for the local backends: with
    ``action_bridge`` set, actions and the rest-pose history are converted the way
    :class:`~rlinf.envs.sim.world_model.backend.bwm.BwmBackend` converts them, and
    ``remote.reward_backend`` names the local backend whose scorer to use.

    Args:
        cfg: The env config. Generation geometry comes from the usual keys; everything about
            the served model lives under ``world_model.remote``.
        device: Where returned frames land.
        server_url: Base URL of the serving process this env worker talks to.
    """

    # The served model conditions on the frames it is sent, not on KIR keyframes.
    supports_kir = False

    def __init__(self, cfg, device: torch.device, server_url: str):
        self.cfg = cfg
        self.device = device
        self.chunk = cfg.chunk
        self.condition_frame_length = cfg.condition_frame_length
        self.num_frames = cfg.num_frames
        self.image_size = tuple(cfg.image_size)
        self.retain_action = cfg.get("retain_action", True)
        if self.num_frames != self.condition_frame_length + self.chunk:
            raise ValueError(
                f"num_frames must be condition_frame_length + chunk; got {self.num_frames} != "
                f"{self.condition_frame_length} + {self.chunk}"
            )

        self.remote_cfg = cfg.remote
        self.server_url = server_url
        self.request_timeout_s = self.remote_cfg.get("request_timeout_s", 300)
        self.warmup_timeout_s = self.remote_cfg.get("warmup_timeout_s", 1800)
        self.max_retries = self.remote_cfg.get("max_retries", 2)
        self.retry_backoff_s = self.remote_cfg.get("retry_backoff_s", 1.0)
        adapter_cls = get_wm_adapter_cls(self.remote_cfg.get("adapter", "videos_sync"))
        self._adapter = adapter_cls(self.cfg, self.remote_cfg)
        self._client = InferenceHTTPClient(self.server_url)
        self._warm = False
        self._sessions: dict[int, dict[str, Any]] = {}

        self._bridge = None
        self._rest_history = None
        bridge = cfg.get("action_bridge", None)
        if bridge is not None:
            self._bridge = RobotwinActionBridge(
                urdf_path=bridge.urdf_path,
                srdf_path=bridge.srdf_path,
                stat_path=bridge.stat_path,
                robot_pose=bridge.robot_pose,
            )
            rest = self._bridge(np.asarray(cfg.rest_action, dtype=np.float32))
            self._rest_history = torch.from_numpy(rest)[None].repeat(
                self.condition_frame_length, 1
            )

    @staticmethod
    def load_reward_model(cfg):
        """The scorer runs here, not on the server: the one of ``remote.reward_backend``."""
        return get_backend(cfg.remote.reward_backend).load_reward_model(cfg)

    def reward_instructions(self, env) -> Optional[list[str]]:
        reward_backend = get_backend(self.cfg.remote.reward_backend)
        return reward_backend.reward_instructions(self, env)

    @staticmethod
    def _to_uint8(frame: torch.Tensor) -> np.ndarray:
        """``[C, 1, H, W]`` in ``[-1, 1]`` to ``[H, W, C]`` uint8."""
        img = np.transpose(frame[:, 0].detach().cpu().numpy(), (1, 2, 0))
        return ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        seeds: Sequence[int],
    ) -> None:
        for row, (env_id, seed) in enumerate(zip(env_ids, seeds)):
            if self._rest_history is not None:
                # init_actions are joint-space; a zero pose is mid-workspace after the bridge.
                history = self._rest_history.clone()
            else:
                history = init_actions[row].detach().cpu().clone()
            self._sessions[int(env_id)] = {
                "seed": int(seed),
                "frames": [self._to_uint8(f) for f in init_frames[row]],
                "actions": history,
                "step_id": 0,
            }

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    def _window_actions(
        self, env_ids: Sequence[int], actions: torch.Tensor
    ) -> torch.Tensor:
        """Prepend each session's action history to its chunk. Does not mutate the history."""
        history = torch.stack(
            [self._sessions[int(i)]["actions"] for i in env_ids], dim=0
        ).to(dtype=actions.dtype)
        if not self.retain_action:
            return actions
        return torch.cat([history, actions], dim=1)

    def _roll_forward(
        self, env_id: int, windowed: torch.Tensor, frames: np.ndarray
    ) -> None:
        """Advance a session's window and action history. Only called after a success."""
        session = self._sessions[int(env_id)]
        keep = self.condition_frame_length - 1
        session["frames"][1:] = list(frames[-keep:])
        tail = windowed[-keep:, :]
        session["actions"][1 : self.condition_frame_length] = tail.to(
            dtype=session["actions"].dtype
        )
        session["step_id"] += 1

    def _step(self, session: dict[str, Any], actions: np.ndarray) -> np.ndarray:
        fields, files, state = self._adapter.build_request(
            np.stack(session["frames"]), actions, session["seed"]
        )
        # The first request may be building kernels server-side, so it gets the long timeout.
        resp = self._client.post(
            self._adapter.path,
            fields,
            files=files,
            raw_response=True,
            timeout_s=self.request_timeout_s if self._warm else self.warmup_timeout_s,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )
        self._warm = True
        return self._adapter.parse_response(resp, state)

    def generate(
        self,
        env_ids: Sequence[int],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        missing = [int(i) for i in env_ids if int(i) not in self._sessions]
        if missing:
            raise RuntimeError(
                f"generate called for env slots without a session: {missing}"
            )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.detach().cpu().float()
        if self._bridge is not None:
            actions = torch.from_numpy(self._bridge(actions.numpy())).float()
        if actions.shape[0] != len(env_ids):
            raise ValueError(
                f"env_ids and actions must describe the same batch rows; got "
                f"{len(env_ids)}, {actions.shape[0]}"
            )

        windowed = self._window_actions(env_ids, actions)

        # One request per slot: a generation endpoint carries one trajectory's step, and the
        # engine serialises them anyway (max_num_seqs=1). Batching across slots is a server
        # concern, not something the client can fake.
        videos = []
        for row, env_id in enumerate(env_ids):
            session = self._sessions[int(env_id)]
            frames = self._step(session, windowed[row].numpy())
            if frames.ndim != 4 or frames.shape[0] < self.chunk:
                raise RuntimeError(
                    f"served {frames.shape} for env {int(env_id)}, expected at least "
                    f"{self.chunk} frames"
                )
            new_frames = frames[-self.chunk :]
            self._roll_forward(env_id, windowed[row], frames)

            video = new_frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # [T, 3, H, W]
            videos.append(video.transpose(0, 1))  # [3, T, H, W]

        return torch.stack(videos, dim=0).to(self.device)

    def offload(self) -> None:
        """No-op: the weights live in the serving process, not this one."""

    def onload(self) -> None:
        """No-op: see :meth:`offload`."""


class VideosSyncAdapter:
    """A step as ``POST /v1/videos/sync``, the one-shot generation endpoint.

    The condition window rides as an mp4 in ``input_reference`` and the actions in
    ``extra_params``, which the server merges into the sampling params.
    """

    path = "/v1/videos/sync"

    def __init__(self, cfg, remote_cfg):
        self.model = remote_cfg.model
        self.num_inference_steps = remote_cfg.get("num_inference_steps", 50)
        self.fps = remote_cfg.get("fps", 24)
        self.prompt = remote_cfg.get("prompt", "")
        self.image_size = tuple(cfg.image_size)

    def build_request(
        self, cond_frames: np.ndarray, actions: np.ndarray, seed: int
    ) -> tuple[dict[str, Any], dict[str, tuple[str, bytes, str]], int]:
        import imageio.v2 as imageio

        # The latent grid is 4n+1 pixel frames. A window off that grid is truncated server-side
        # without an error, so round up and carry the last action across the padding.
        length = actions.shape[0]
        num_frames = 1 + ((length - 1 + 3) // 4) * 4
        if length < num_frames:
            pad = np.repeat(actions[-1:], num_frames - length, axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        history = io.BytesIO()
        writer = imageio.get_writer(history, format="mp4", fps=self.fps, quality=8)
        for frame in cond_frames:
            writer.append_data(frame)
        writer.close()

        h, w = self.image_size
        fields = {
            "model": self.model,
            "prompt": self.prompt,
            "size": f"{w}x{h}",
            "num_frames": num_frames,
            "fps": self.fps,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": 1.0,
            "extra_params": json.dumps(
                {"action": [[float(x) for x in row] for row in actions]}
            ),
            "seed": seed,
        }
        files = {"input_reference": ("history.mp4", history.getvalue(), "video/mp4")}
        return fields, files, length

    def parse_response(self, resp: bytes, state: int) -> np.ndarray:
        """mp4 bytes to ``[T, H, W, 3]`` uint8, dropping the frames of padded actions."""
        import imageio.v3 as iio

        frames = np.asarray(
            iio.imread(io.BytesIO(resp), extension=".mp4", plugin="FFMPEG")
        )
        return frames[:state]


register_wm_adapter("videos_sync", lambda: VideosSyncAdapter)
