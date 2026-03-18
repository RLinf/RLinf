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

import atexit
import copy
import os
import pickle
import signal
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from rlinf.data.lerobot_writer import LeRobotDatasetWriter

_VALID_FORMATS = ("pickle", "lerobot")


class CollectEpisode(gym.Wrapper):
    """Wrapper for collecting rollout data episode by episode.

    Records observations, actions, rewards, termination flags, and info dicts
    at each step. Completed episodes are asynchronously saved to disk in either
    pickle or LeRobot format.

    Supports both single and vectorized environments. When used with
    auto-resetting environments (those that embed ``final_observation`` in
    ``info``), the pre-reset observation is correctly attributed to the finished
    episode, and the post-reset observation is carried over to the next episode.

    Args:
        env: The gymnasium environment to wrap.
        save_dir: Directory for saving collected episode data.
        rank: Worker rank for file naming in distributed settings. Defaults to 0.
        num_envs: Number of parallel environments. Defaults to 1.
        show_goal_site: Whether to show goal visualization in renders.
            Defaults to True.
        export_format: Episode export format, ``"pickle"`` or ``"lerobot"``.
            Defaults to ``"pickle"``.
        robot_type: Robot type for LeRobot metadata. Defaults to ``"panda"``.
        fps: FPS for LeRobot metadata. Defaults to 10.
        only_success: Whether to save only successful episodes. Defaults to False.
        stats_sample_ratio: Sampling ratio for incremental image statistics
            (lerobot only). Defaults to 0.1.
        finalize_interval: Call ``writer.finalize()`` every this many completed
            episodes to flush ``info.json`` and ``stats.json`` as a checkpoint.
            ``0`` disables periodic flushing (lerobot only). Defaults to 100.
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: str,
        rank: int = 0,
        num_envs: int = 1,
        show_goal_site: bool = True,
        export_format: str = "pickle",
        robot_type: str = "panda",
        fps: int = 10,
        only_success: bool = False,
        stats_sample_ratio: float = 0.1,
        finalize_interval: int = 100,
    ):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if export_format not in _VALID_FORMATS:
            raise ValueError(
                f"Unsupported export_format={export_format!r}, "
                f"expected one of {_VALID_FORMATS}"
            )

        self.save_dir = save_dir
        self.rank = rank
        self.num_envs = num_envs
        self.show_goal_site = show_goal_site
        self.export_format = export_format
        self.robot_type = robot_type
        self.fps = fps
        self.only_success = only_success
        self.stats_sample_ratio = stats_sample_ratio
        self.finalize_interval = finalize_interval

        # LeRobot writer is created lazily on the first completed episode.
        self._lerobot_writer: Optional[LeRobotDatasetWriter] = None
        self._lerobot_lock = Lock()
        self._episodes_written = 0  # guarded by _lerobot_lock

        # Single-worker executor keeps write ordering deterministic.
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"collect_episode_export_rank_{self.rank}",
        )
        self._futures: list[Future] = []

        # Per-environment episode state.
        self._episode_ids = [0] * num_envs
        self._episode_success = [False] * num_envs
        # Holds the post-reset obs for auto-reset envs to prepend to next episode.
        self._pending_obs: list[Any] = [None] * num_envs
        self._buffers: list[dict[str, list]] = [
            self._new_buffer() for _ in range(num_envs)
        ]

        self._closed = False

        os.makedirs(self.save_dir, exist_ok=True)
        atexit.register(self._finalize_on_exit)
        signal.signal(signal.SIGTERM, self._handle_signal)

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    # ─────────────────────────────────────────── gymnasium interface ──────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """Reset the environment and initialise episode buffers.

        Args:
            seed: Optional random seed for environment reset.
            options: Optional dictionary of reset options.

        Returns:
            Tuple of (observation, info) from the underlying environment.
        """
        self._buffers = [self._new_buffer() for _ in range(self.num_envs)]
        self._episode_success = [False] * self.num_envs
        self._pending_obs = [None] * self.num_envs

        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs, info = self.env.reset()

        self._show_goal_site_visual()
        self._record_reset_obs(obs)
        return obs, info

    def step(self, action, **kwargs):
        """Execute a step and record the transition.

        Args:
            action: Action to execute in the environment.
            **kwargs: Additional arguments forwarded to the underlying step.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        self._record_step(action, obs, reward, terminated, truncated, info)
        self._maybe_flush(terminated, truncated)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions, recording each sub-step individually.

        Both pickle and lerobot formats receive step-level records for maximum
        data fidelity.

        Args:
            chunk_actions: Action chunk, typically a tensor of shape
                ``[num_envs, chunk_size, action_dim]``.

        Returns:
            Tuple of (obs_list, rewards, terminations, truncations, infos_list).
        """
        obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
            chunk_actions
        )

        chunk_size = len(obs_list) if isinstance(obs_list, (list, tuple)) else 1
        for step_idx in range(chunk_size):
            step_action = (
                chunk_actions[:, step_idx]
                if isinstance(chunk_actions, (torch.Tensor, np.ndarray))
                and chunk_actions.ndim > 1
                else chunk_actions
            )
            step_obs = (
                obs_list[step_idx] if isinstance(obs_list, (list, tuple)) else obs_list
            )
            step_reward = (
                rewards[:, step_idx] if getattr(rewards, "ndim", 1) > 1 else rewards
            )
            step_term = (
                terminations[:, step_idx]
                if getattr(terminations, "ndim", 1) > 1
                else terminations
            )
            step_trunc = (
                truncations[:, step_idx]
                if getattr(truncations, "ndim", 1) > 1
                else truncations
            )
            step_info = (
                infos_list[step_idx]
                if isinstance(infos_list, (list, tuple))
                else infos_list
            )
            self._record_step(
                step_action, step_obs, step_reward, step_term, step_trunc, step_info
            )
            self._maybe_flush(step_term, step_trunc)

        return obs_list, rewards, terminations, truncations, infos_list

    def close(self):
        if self._closed:
            return None
        self._closed = True
        self._finalize_lerobot()
        self._wait_futures()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    # ─────────────────────────────────────────── buffer management ────────────

    def _new_buffer(self) -> dict[str, list]:
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }

    def _record_reset_obs(self, obs) -> None:
        """Record the initial observation from reset into every env's buffer."""
        for env_idx in range(self.num_envs):
            self._buffers[env_idx]["observations"].append(
                self._slice_copy(obs, env_idx)
            )

    def _record_step(self, action, obs, reward, terminated, truncated, info) -> None:
        """Record one transition into every env's buffer."""
        for env_idx in range(self.num_envs):
            # Auto-reset envs store the pre-reset obs in info["final_observation"];
            # the current `obs` is the post-reset obs for the *next* episode.
            if isinstance(info, dict) and "final_observation" in info:
                env_obs = self._slice_copy(info["final_observation"], env_idx)
                self._pending_obs[env_idx] = self._slice_copy(obs, env_idx)
            else:
                env_obs = self._slice_copy(obs, env_idx)

            buf = self._buffers[env_idx]
            buf["observations"].append(env_obs)
            buf["actions"].append(self._slice_copy(action, env_idx))
            buf["rewards"].append(self._slice_copy(reward, env_idx))
            buf["terminated"].append(self._slice_copy(terminated, env_idx))
            buf["truncated"].append(self._slice_copy(truncated, env_idx))
            buf["infos"].append(self._slice_copy(info, env_idx))

            # Update per-env success using already-sliced info (no extra copy).
            self._update_success(env_idx, self._slice_data(info, env_idx))

    def _reset_env_buffer(self, env_idx: int) -> None:
        """Advance episode counter, clear the buffer, and carry over pending obs."""
        self._episode_ids[env_idx] += 1
        self._buffers[env_idx] = self._new_buffer()
        self._episode_success[env_idx] = False

        if self._pending_obs[env_idx] is not None:
            self._buffers[env_idx]["observations"].append(self._pending_obs[env_idx])
            self._pending_obs[env_idx] = None

    # ─────────────────────────────────────────── episode flushing ─────────────

    def _maybe_flush(self, terminated, truncated) -> None:
        """Save finished episodes and reset their buffers."""
        for env_idx in range(self.num_envs):
            if self._scalar_flag(terminated, env_idx) or self._scalar_flag(
                truncated, env_idx
            ):
                is_success = self._get_episode_success(self._buffers[env_idx], env_idx)
                if not self.only_success or is_success:
                    self._flush_episode(env_idx, is_success)
                self._reset_env_buffer(env_idx)

    def _flush_episode(self, env_idx: int, is_success: bool) -> None:
        """Dispatch a completed episode to the appropriate format writer."""
        buf = self._buffers[env_idx]
        if not buf["actions"]:
            return

        if self.export_format == "lerobot":
            ep_data = self._buffer_to_lerobot_ep(buf, env_idx, is_success)
            if ep_data is not None:
                self._submit(self._write_lerobot_episode, ep_data)
        else:
            episode_data = self._copy(
                {
                    "rank": self.rank,
                    "env_idx": env_idx,
                    "episode_id": self._episode_ids[env_idx],
                    "success": is_success,
                    "observations": buf["observations"],
                    "actions": buf["actions"],
                    "rewards": buf["rewards"],
                    "terminated": buf["terminated"],
                    "truncated": buf["truncated"],
                    "infos": buf["infos"],
                }
            )
            label = "success" if is_success else "fail"
            filename = (
                f"rank_{self.rank}_env_{env_idx}_"
                f"episode_{self._episode_ids[env_idx]}_{label}.pkl"
            )
            self._submit(
                self._write_pickle, os.path.join(self.save_dir, filename), episode_data
            )

    # ─────────────────────────────────────────── lerobot helpers ──────────────

    def _buffer_to_lerobot_ep(
        self, buf: dict, env_idx: int, is_success: bool
    ) -> Optional[dict[str, Any]]:
        """Convert a raw episode buffer into a LeRobot-compatible episode dict.

        The observations list contains one extra entry prepended at reset time,
        so it is aligned to the actions list by taking the trailing N entries.
        Steps where any required field (image, state, action) is missing are
        silently skipped.
        """
        actions = buf["actions"]
        terminated = buf["terminated"]
        obs_steps = buf["observations"]

        if not actions:
            return None

        if len(obs_steps) > len(actions):
            obs_steps = obs_steps[: len(actions)]

        task_desc = self._extract_task_description(buf, env_idx)
        images: list[np.ndarray] = []
        wrist_images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        np_actions: list[np.ndarray] = []
        dones: list[bool] = []
        first_term_step: Optional[int] = None

        for i, action in enumerate(actions):
            obs = obs_steps[i] if i < len(obs_steps) else None
            image, wrist_image, state = self._extract_obs_image_state(obs)

            # overwrite action with intervene action
            if "final_info" in buf["infos"][i]:
                info_with_intervene = copy.deepcopy(buf["infos"][i]["final_info"])
            else:
                info_with_intervene = copy.deepcopy(buf["infos"][i])
            print("info_with_intervene.keys()", info_with_intervene.keys())
            if (
                "intervene_flags" in info_with_intervene.keys()
                and "intervene_action" in info_with_intervene.keys()
            ):
                # overwrite action with
                print(
                    "intervene_flags.shape:",
                    info_with_intervene["intervene_flags"].shape,
                    ", intervene_action.shape:",
                    info_with_intervene["intervene_action"].shape,
                )

            np_action = self._to_numpy(action)

            if image is None or state is None or np_action is None:
                continue

            images.append(self._to_uint8(np.asarray(image)))
            if wrist_image is not None:
                wrist_images.append(self._to_uint8(np.asarray(wrist_image)))
            states.append(np.asarray(state).astype(np.float32))
            np_actions.append(np.asarray(np_action).astype(np.float32))
            dones.append(False)

            if bool(terminated[i]) and first_term_step is None:
                first_term_step = len(np_actions)

        if not images:
            return None

        end = first_term_step if first_term_step is not None else len(images)
        dones_out = dones[:end]
        if dones_out:
            dones_out[-1] = True

        return {
            "images": images[:end],
            "wrist_images": wrist_images[:end] if wrist_images else None,
            "states": states[:end],
            "actions": np_actions[:end],
            "dones": dones_out,
            "task": task_desc,
            "is_success": is_success,
        }

    def _ensure_lerobot_writer(self, ep_data: dict) -> LeRobotDatasetWriter:
        """Create the LeRobot writer on first use. Must be called inside the lock."""
        if self._lerobot_writer is None:
            self._lerobot_writer = LeRobotDatasetWriter(
                root_dir=self.save_dir,
                robot_type=self.robot_type,
                fps=self.fps,
                image_shape=ep_data["images"][0].shape,
                state_dim=ep_data["states"][0].shape[-1],
                action_dim=ep_data["actions"][0].shape[-1],
                use_incremental_stats=True,
                stats_sample_ratio=self.stats_sample_ratio,
            )
        return self._lerobot_writer

    def _write_lerobot_episode(self, ep_data: dict) -> None:
        with self._lerobot_lock:
            writer = self._ensure_lerobot_writer(ep_data)
            wrist_images = ep_data["wrist_images"]
            writer.add_episode(
                images=np.stack(ep_data["images"]),
                wrist_images=np.stack(wrist_images)
                if wrist_images is not None
                else None,
                states=np.stack(ep_data["states"]),
                actions=np.stack(ep_data["actions"]),
                task=ep_data["task"],
                is_success=ep_data["is_success"],
                dones=np.array(ep_data["dones"], dtype=bool),
            )
            self._episodes_written += 1
            if (
                self.finalize_interval > 0
                and self._episodes_written % self.finalize_interval == 0
            ):
                writer.finalize()

    def _finalize_lerobot(self) -> None:
        """Drain pending futures then write the LeRobot dataset metadata."""
        if self.export_format != "lerobot":
            return
        self._wait_futures()
        with self._lerobot_lock:
            if self._lerobot_writer is not None:
                self._lerobot_writer.finalize()
                self._lerobot_writer = None

    # ─────────────────────────────────────────── pickle helpers ───────────────

    def _write_pickle(self, save_path: str, episode_data: dict) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(episode_data, f)

    # ─────────────────────────────────────────── async I/O ────────────────────

    def _submit(self, fn, *args) -> None:
        if self._executor is None:
            return
        self._futures.append(self._executor.submit(fn, *args))
        self._drain_futures()

    def _drain_futures(self) -> None:
        remaining = []
        for f in self._futures:
            if f.done():
                f.result()
            else:
                remaining.append(f)
        self._futures = remaining

    def _wait_futures(self) -> None:
        for f in self._futures:
            f.result()
        self._futures = []

    def _handle_signal(self, signum, frame) -> None:
        self.close()
        raise SystemExit(0)

    def _finalize_on_exit(self) -> None:
        self.close()

    # ─────────────────────────────────────────── success tracking ─────────────

    def _update_success(self, env_idx: int, env_info) -> None:
        """Update the per-env success flag from a single-env info dict."""
        if not isinstance(env_info, dict):
            return

        ep = env_info.get("episode") or {}
        success = ep.get("success_once") if isinstance(ep, dict) else None
        if success is None and isinstance(ep, dict):
            success = ep.get("success_at_end")
        if success is None:
            success = env_info.get("success")

        if success is not None:
            val = success.item() if isinstance(success, torch.Tensor) else success
            self._episode_success[env_idx] = bool(val)

    def _get_episode_success(self, buf: dict, env_idx: int) -> bool:
        """Determine final episode success by scanning recorded info dicts.

        Checks (in priority order): ``final_info``, ``episode``, and the root
        info dict, looking for ``success_once``, ``success_at_end``, and
        ``success`` keys. Falls back to the incrementally-updated
        ``_episode_success`` flag.
        """
        for info in reversed(buf["infos"]):
            if not isinstance(info, dict):
                continue
            for src in (
                info.get("final_info", info).get("episode"),
                info.get("final_info"),
                info.get("episode"),
                info,
            ):
                if not isinstance(src, dict):
                    continue
                for key in ("success_once", "success_at_end", "success"):
                    val = src.get(key)
                    if val is not None:
                        return bool(
                            val.item() if isinstance(val, torch.Tensor) else val
                        )
        return self._episode_success[env_idx]

    # ─────────────────────────────────────────── data utilities ───────────────

    def _extract_task_description(self, buf: dict, env_idx: int) -> str:
        for obs in reversed(buf["observations"]):
            if not isinstance(obs, dict) or "task_descriptions" not in obs:
                continue
            desc = obs["task_descriptions"]
            if isinstance(desc, (list, tuple)):
                return str(desc[env_idx] if len(desc) == self.num_envs else desc[0])
            return str(desc)
        return "unknown task"

    def _extract_obs_image_state(self, obs):
        """Return ``(image, wrist_image, state)`` numpy arrays from an obs dict."""
        if not isinstance(obs, dict):
            return None, None, None
        image = obs.get("main_images", obs.get("image", obs.get("full_image")))
        wrist_image = obs.get("wrist_images", obs.get("wrist_image"))
        state = obs.get("states", obs.get("state"))
        return self._to_numpy(image), self._to_numpy(wrist_image), self._to_numpy(state)

    def _slice_data(self, data, env_idx: int):
        """Slice batched data for a single env without copying."""
        if isinstance(data, torch.Tensor):
            return (
                data[env_idx]
                if data.dim() > 0 and data.shape[0] == self.num_envs
                else data
            )
        if isinstance(data, np.ndarray):
            return (
                data[env_idx]
                if data.ndim > 0 and data.shape[0] == self.num_envs
                else data
            )
        if isinstance(data, dict):
            return {k: self._slice_data(v, env_idx) for k, v in data.items()}
        if isinstance(data, list):
            return data[env_idx] if len(data) == self.num_envs else data
        return data

    def _slice_copy(self, data, env_idx: int):
        """Slice batched data for a single env and deep-copy the result."""
        return self._copy(self._slice_data(data, env_idx))

    def _scalar_flag(self, flags, env_idx: int) -> bool:
        """Extract a boolean flag for ``env_idx`` from a batched flag."""
        if isinstance(flags, torch.Tensor):
            if flags.dim() > 1:
                return bool(flags[env_idx, -1].item())
            if flags.dim() == 1 and flags.shape[0] == self.num_envs:
                return bool(flags[env_idx].item())
            return bool(flags.item())
        if isinstance(flags, np.ndarray):
            if flags.ndim > 0 and flags.shape[0] == self.num_envs:
                return bool(flags[env_idx])
            return bool(flags)
        return bool(flags)

    @staticmethod
    def _to_numpy(data) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        return (
            (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        )

    def _copy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().clone()
        if isinstance(data, np.ndarray):
            return data.copy()
        if isinstance(data, dict):
            return {k: self._copy(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._copy(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self._copy(item) for item in data)
        return data

    # ─────────────────────────────────────── goal site visualization ──────────

    def _show_goal_site_visual(self) -> None:
        """Unhide the goal site in environments that support it."""
        if not self.show_goal_site:
            return

        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped

        if not hasattr(unwrapped, "goal_site"):
            return

        goal_site = unwrapped.goal_site
        if hasattr(unwrapped, "_hidden_objects"):
            while goal_site in unwrapped._hidden_objects:
                unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()
