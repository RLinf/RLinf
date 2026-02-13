# Copyright 2025 The RLinf Authors.
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

import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from rlinf.envs.utils import put_info_on_image, tile_images


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    Features:
    1. Manage render_images and video_cnt
    2. Append frames after reset/step/chunk_step
    3. Provide flush_video to save videos
    4. Support multiple obs image formats and batched/sequence inputs
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        self.render_images: list[np.ndarray] = []
        self.video_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _get_image_from_dict(self, obs: dict) -> Optional[Any]:
        """Pick the best image field from an observation dict."""
        for key in ("main_images", "images", "rgb", "full_image", "main_image"):
            if key in obs and obs[key] is not None:
                return obs[key]
        return None

    def _extract_frame_batches(self, obs: Any) -> list[list[np.ndarray]]:
        """Extract a list of per-step image batches from obs."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            image_src = self._get_image_from_dict(obs)
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    image_src = self._get_image_from_dict(item)
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if torch is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _build_info_item(
        self,
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["termination"] = bool(value) if value is not None else value

        return info_item

    def _append_frame(
        self,
        images: list[np.ndarray],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
    ) -> None:
        """Overlay info (optional) and append a tiled frame."""
        if not images:
            return
        if self.video_cfg.get("info_on_video", True):
            images = [
                put_info_on_image(
                    img,
                    self._build_info_item(rewards, terminations, env_id, time_idx),
                )
                for env_id, img in enumerate(images)
            ]
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
            self.render_images.append(full_image)
        else:
            self.render_images.append(images[0])

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
    ):
        """Extract frames from obs and append to the buffer."""
        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
            return

        if isinstance(infos, (list, tuple)):
            for time_idx, images in enumerate(frames):
                self._append_frame(images, rewards, terminations, time_idx)
            return

        for time_idx, images in enumerate(frames):
            self._append_frame(images, rewards, terminations, time_idx)

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        obs, info = self.env.reset(*args, **kwargs)
        self.add_new_frames(obs, info)
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self.add_new_frames(obs, info, reward, terminations)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]
            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self.add_new_frames(reset_obs, None)
            else:
                self.add_new_frames(obs_list, infos_list, rewards, terminations)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to an MP4 file (async)."""
        if not self.render_images:
            return

        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        frames = list(self.render_images)
        self.render_images = []
        self.video_cnt += 1
        self._submit_save(frames, mp4_path)

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Submit a background job to save the video."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background)."""
        video_writer = None
        try:
            video_writer = imageio.get_writer(mp4_path, fps=self._fps)
            for img in frames:
                video_writer.append_data(img)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def close(self):
        """Wait for pending video writes before closing."""
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
