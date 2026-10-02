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

import atexit
import copy
import json
import os
import pickle
import re
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from rlinf.data.schema.embodied_types import LeRobotFrame
from rlinf.utils.logging import get_logger

_VALID_FORMATS = ("pickle", "lerobot")

# Bound the queue while covering the measured multi-second v2 episode save.
# 240 three-view 640x480 RGB frames occupy about 633 MiB, not a whole episode.
_MAX_PENDING_FUTURES = 240


_ID_DIR_RE = re.compile(r"^id_(\d+)$")


def _scan_existing_lerobot_shards(save_dir: str, rank: int) -> tuple[int, int]:
    """Return ``(total_episodes, next_shard_id)`` for resume.

    ``next_shard_id`` is ``max(existing_id_numbers) + 1`` over every
    ``id_<int>/`` directory (regardless of whether ``meta/info.json`` is
    finalized) so a crashed session's partial shard is never overwritten.
    Shards with unparseable ``info.json`` contribute 0 to ``total_episodes``.
    """
    rank_dir = os.path.join(save_dir, f"rank_{rank}")
    if not os.path.isdir(rank_dir):
        return 0, 0

    total = 0
    max_id = -1
    for entry in os.listdir(rank_dir):
        m = _ID_DIR_RE.match(entry)
        if m is None:
            continue
        if not os.path.isdir(os.path.join(rank_dir, entry)):
            continue
        max_id = max(max_id, int(m.group(1)))

        info_path = os.path.join(rank_dir, entry, "meta", "info.json")
        try:
            with open(info_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        count = meta.get("total_episodes")
        if isinstance(count, int) and count > 0:
            total += count
    next_shard_id = max_id + 1 if max_id >= 0 else 0
    return total, next_shard_id


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
        finalize_interval: Call ``writer.finalize()`` every this many completed
            non-streaming LeRobot episodes to flush ``info.json`` and
            ``stats.json`` as a checkpoint. Streaming LeRobot collection writes
            each kept episode to its own shard and finalizes that shard after
            the episode save finishes. ``0`` disables periodic non-streaming
            flushing. Defaults to 100.
        resume: If True and ``export_format == "lerobot"``, reuse ``save_dir``
            across sessions — new episodes land in a fresh ``id_{N}`` shard
            (N = max existing shard id + 1) so the in-progress write never
            touches previously-finalized or partially-written data. Ignored
            for pickle. Defaults to False.
        streaming: If True and ``export_format == "lerobot"``, write each
            recorded frame to the LeRobot dataset immediately instead of
            buffering the whole episode in memory. Images land on disk
            per-frame via the dataset's async image writer, so RAM usage stays
            flat regardless of episode length. Every recorded episode is saved
            — successful or not — with the episode-level ``is_success`` flag
            stamped at episode end, so ``only_success`` filtering does not
            apply in this mode. Defaults to False.
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
        finalize_interval: int = 100,
        resume: bool = False,
        streaming: bool = False,
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
        if streaming and export_format != "lerobot":
            raise ValueError("streaming=True requires export_format='lerobot'")

        self.save_dir = save_dir
        self.rank = rank
        self.num_envs = num_envs
        self.show_goal_site = show_goal_site
        self.export_format = export_format
        self.robot_type = robot_type
        self.fps = fps
        self.only_success = only_success
        self.finalize_interval = finalize_interval
        self.streaming = streaming

        # Streaming-mode per-env episode state. Only one previous observation
        # (the pre-action frame) is kept per env; everything else is written
        # to disk as it is recorded.
        self._stream_prev_obs: list[Any] = [None] * num_envs
        self._stream_frames: list[int] = [0] * num_envs
        self._stream_invalid: list[bool] = [False] * num_envs
        self._stream_success_marks: list[list[Optional[bool]]] = [
            [] for _ in range(num_envs)
        ]
        self._stream_task: list[Optional[str]] = [None] * num_envs
        # Per-shard JSONL sidecar holding per-frame state/actions (streaming).
        self._stream_sidecar = None
        self._stream_sidecar_episode_start = 0
        self._stream_sidecar_root: Optional[str] = None

        self._preexisting_episode_count = 0
        self._next_shard_id = 0
        self._save_futures_lock = Lock()
        if export_format == "lerobot":
            self._lerobot_writer: Optional[Any] = None
            self._lerobot_lock = Lock()
            if resume:
                (
                    self._preexisting_episode_count,
                    self._next_shard_id,
                ) = _scan_existing_lerobot_shards(save_dir, rank)
            self._episodes_written = (
                self._preexisting_episode_count
            )  # guarded by _save_futures_lock

        # Single-worker executor keeps write ordering deterministic.
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"collect_episode_export_rank_{self.rank}",
        )
        self._futures: list[Future] = []
        self._save_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"collect_episode_save_rank_{self.rank}",
            )
            if export_format == "lerobot" and streaming
            else None
        )
        self._save_futures: list[Future] = []

        # Per-environment episode state.
        self._episode_ids = [0] * num_envs
        self._episode_success = [False] * num_envs
        self._episode_discarded = [False] * num_envs
        self._segment_ids: list[int] = [0] * num_envs
        self._global_step = 0
        # Holds the post-reset obs for auto-reset envs to prepend to next episode.
        self._pending_obs: list[Any] = [None] * num_envs
        self._pending_info: list[Any] = [None] * num_envs
        self._buffers: list[dict[str, list]] = [
            self._new_buffer() for _ in range(num_envs)
        ]

        self._closed = False
        self.logger = get_logger()

        os.makedirs(self.save_dir, exist_ok=True)
        atexit.register(self._finalize_on_exit)

    @property
    def preexisting_episode_count(self) -> int:
        """Number of episodes on disk at construction time (resume mode only)."""
        return self._preexisting_episode_count

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

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
        if self.streaming:
            self._drain_save_futures()
            # Finish an in-progress recording before the next episode begins.
            for env_idx in range(self.num_envs):
                self._stream_end_episode(env_idx)

        self._buffers = [self._new_buffer() for _ in range(self.num_envs)]
        self._episode_success = [False] * self.num_envs
        self._pending_obs = [None] * self.num_envs
        self._pending_info = [None] * self.num_envs

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
        if self.streaming:
            self._drain_save_futures()
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        self._record_step(action, obs, reward, terminated, truncated, info)
        if self.streaming:
            info["recording_invalid"] = np.array(self._stream_invalid, dtype=bool)
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
        if self.streaming:
            self._drain_save_futures()
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
                copy.deepcopy(infos_list[step_idx])
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
        result = None
        primary_error = None
        env_close_attempted = False
        try:
            # Release robot hardware before potentially lengthy dataset finalization.
            try:
                if hasattr(self.env, "close"):
                    env_close_attempted = True
                    result = self.env.close()
            except BaseException as error:
                primary_error = error
            self._finalize_lerobot()
        except BaseException as error:
            if primary_error is None:
                primary_error = error
        try:
            self._wait_futures()
        except BaseException as error:
            if primary_error is None:
                primary_error = error
        try:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                    self._executor = None
            finally:
                try:
                    if not env_close_attempted and hasattr(self.env, "close"):
                        result = self.env.close()
                finally:
                    try:
                        self._wait_save_futures()
                    except BaseException as error:
                        if primary_error is None:
                            primary_error = error
                    try:
                        if self._save_executor is not None:
                            self._save_executor.shutdown(wait=True)
                            self._save_executor = None
                    except BaseException as error:
                        if primary_error is None:
                            primary_error = error
        finally:
            if primary_error is not None:
                raise primary_error
        self._closed = True
        return result

    def _new_buffer(self) -> dict[str, list]:
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
            "segment_ids": [],
        }

    def _seed_reset_frame(self, env_idx: int, env_obs: Any) -> None:
        """Seed a fresh buffer with the post-reset state-aligned entry.

        State-aligned fields (observations / rewards / terminated / truncated /
        infos) get a leading reset entry; action-aligned fields (actions,
        segment_ids) stay empty and fill on the first regular step.
        """
        if self.streaming:
            self._stream_prev_obs[env_idx] = env_obs
            self._stream_update_task(env_idx, env_obs)
            return
        buf = self._buffers[env_idx]
        buf["observations"].append(env_obs)
        buf["rewards"].append(0.0)
        buf["terminated"].append(False)
        buf["truncated"].append(False)
        buf["infos"].append({})

    def _record_reset_obs(self, obs) -> None:
        """Record the initial observation from reset into every env's buffer."""
        for env_idx in range(self.num_envs):
            self._seed_reset_frame(env_idx, self._slice_copy(obs, env_idx))

    @staticmethod
    def _bool_from_env_info(env_info: Any, key: str) -> bool:
        """Read a per-env bool flag (scalar / 0-d / size-1 array all OK)."""
        if not isinstance(env_info, dict) or env_info.get(key) is None:
            return False
        return bool(np.asarray(env_info[key]).any())

    def _record_step(self, action, obs, reward, terminated, truncated, info) -> None:
        """Record one transition into every env's buffer."""
        self._global_step += 1

        has_final_obs = isinstance(info, dict) and "final_observation" in info
        if has_final_obs:
            final_observation = info["final_observation"]
            final_info_batch = info["final_info"]
            info_no_reset = copy.deepcopy(info)
            info_no_reset.pop("final_observation")
            info_no_reset.pop("final_info")

        for env_idx in range(self.num_envs):
            # Auto-reset envs store the pre-reset obs in info["final_observation"];
            # the current `obs` is the post-reset obs for the *next* episode.
            # Only use final_observation for envs that are actually done this step.
            env_done = self._scalar_flag(terminated, env_idx) or self._scalar_flag(
                truncated, env_idx
            )
            if has_final_obs and env_done:
                env_obs = self._slice_copy(final_observation, env_idx)
                env_info = self._slice_copy(final_info_batch, env_idx)
                self._pending_obs[env_idx] = self._slice_copy(obs, env_idx)
                self._pending_info[env_idx] = self._slice_copy(info_no_reset, env_idx)
                if "intervene_action" in env_info:
                    env_info["intervene_action"] = env_info["intervene_action"][-1]
                    env_info["intervene_flag"] = env_info["intervene_flag"][-1]
            else:
                env_obs = self._slice_copy(obs, env_idx)
                env_info = self._slice_copy(info, env_idx)
                if "final_observation" in env_info:
                    env_info.pop("final_observation")
                    env_info.pop("final_info")

            record_reset = self._bool_from_env_info(env_info, "record_reset")
            pre_record = self._bool_from_env_info(env_info, "pre_record")

            self._episode_discarded[env_idx] = self._bool_from_env_info(
                env_info, "episode_discarded"
            )
            self._update_success(env_idx, env_info)

            if record_reset:
                if self.streaming:
                    # Write-all: the recording being replaced is still saved.
                    self._stream_end_episode(env_idx)
                else:
                    self._buffers[env_idx] = self._new_buffer()
                self._episode_success[env_idx] = False
                self._segment_ids[env_idx] = 0
                self._seed_reset_frame(env_idx, env_obs)
                continue

            if pre_record:
                continue

            if self._bool_from_env_info(env_info, "segment_advance"):
                self._segment_ids[env_idx] += 1

            if self.streaming:
                self._stream_record_frame(env_idx, action, env_obs, env_info)
            else:
                buf = self._buffers[env_idx]
                buf["observations"].append(env_obs)
                buf["actions"].append(self._slice_copy(action, env_idx))
                buf["rewards"].append(self._slice_copy(reward, env_idx))
                buf["terminated"].append(self._slice_copy(terminated, env_idx))
                buf["truncated"].append(self._slice_copy(truncated, env_idx))
                buf["infos"].append(env_info)
                buf["segment_ids"].append(int(self._segment_ids[env_idx]))

    def _reset_env_buffer(self, env_idx: int) -> None:
        """Advance episode counter, clear the buffer, and carry over pending obs."""
        self._episode_ids[env_idx] += 1
        self._buffers[env_idx] = self._new_buffer()
        self._episode_success[env_idx] = False
        self._segment_ids[env_idx] = 0

        if self._pending_obs[env_idx] is not None:
            self._buffers[env_idx]["observations"].append(self._pending_obs[env_idx])
            self._pending_obs[env_idx] = None

            if self._pending_info[env_idx] is not None:
                self._buffers[env_idx]["infos"].append(self._pending_info[env_idx])
                self._pending_info[env_idx] = None
            else:
                self._buffers[env_idx]["infos"].append({})

            self._buffers[env_idx]["rewards"].append(0.0)
            self._buffers[env_idx]["terminated"].append(False)
            self._buffers[env_idx]["truncated"].append(False)

    def _maybe_flush(self, terminated, truncated) -> None:
        """Save finished episodes and reset their buffers."""
        for env_idx in range(self.num_envs):
            if self.streaming:
                if self._scalar_flag(terminated, env_idx) or self._scalar_flag(
                    truncated, env_idx
                ):
                    self._stream_end_episode(env_idx)
                continue
            if self._episode_discarded[env_idx]:
                self._reset_env_buffer(env_idx)
                continue
            is_success = self._get_episode_success(self._buffers[env_idx], env_idx)
            done_by_term = self._scalar_flag(terminated, env_idx)
            done_by_trunc = self._scalar_flag(truncated, env_idx)
            if self.only_success:
                if is_success and done_by_term:
                    self._flush_episode(env_idx, is_success)
                    self._reset_env_buffer(env_idx)
                else:
                    if done_by_trunc:
                        self._reset_env_buffer(env_idx)
            else:
                if done_by_term or done_by_trunc:
                    self._flush_episode(env_idx, is_success)
                    self._reset_env_buffer(env_idx)

    def _flush_episode(self, env_idx: int, is_success: bool) -> None:
        """Dispatch a completed episode to the appropriate format writer."""
        self.logger.info(f"Flush env {env_idx}")
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
                    "step": self._global_step,
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
                f"episode_{self._episode_ids[env_idx]}_"
                f"step_{self._global_step}_"
                f"{label}.pkl"
            )
            self._submit(
                self._write_pickle, os.path.join(self.save_dir, filename), episode_data
            )

    # ------------------------------------------------------------------
    # Streaming mode (lerobot only)
    # ------------------------------------------------------------------

    def _stream_update_task(self, env_idx: int, env_obs: Any) -> None:
        """Track the latest task description carried by an observation."""
        if not isinstance(env_obs, dict) or "task_descriptions" not in env_obs:
            return
        desc = env_obs["task_descriptions"]
        if isinstance(desc, (list, tuple)):
            desc = desc[env_idx] if len(desc) == self.num_envs else desc[0]
        self._stream_task[env_idx] = str(desc)

    def _stream_record_frame(
        self, env_idx: int, action: Any, env_obs: Any, env_info: Any
    ) -> None:
        """Convert one recorded step into a LeRobot frame and write it out.

        Mirrors the per-step transformation of ``_buffer_to_lerobot_ep``:
        the frame pairs the *pre-action* observation with this step's action,
        overwrites the action with ``intervene_action`` on fully-intervened
        steps, and skips steps missing a state or action.
        """
        if self._stream_invalid[env_idx]:
            return
        self._drain_futures()
        if len(self._futures) >= _MAX_PENDING_FUTURES:
            self._stream_invalid[env_idx] = True
            self.logger.error(
                "Recording queue full (%s/%s); episode %s recording stopped. "
                "Teleoperation continues. Its recorded prefix will be isolated "
                "outside the training dataset. End this recording and start a new one "
                "after the writer catches up.",
                len(self._futures),
                _MAX_PENDING_FUTURES,
                self._episode_ids[env_idx],
            )
            return
        prev_obs = self._stream_prev_obs[env_idx]
        self._stream_prev_obs[env_idx] = env_obs
        self._stream_update_task(env_idx, env_obs)
        self._stream_success_marks[env_idx].append(
            self._extract_success_from_info(env_info)
        )

        np_action = self._to_numpy(self._slice_data(action, env_idx))
        if (
            isinstance(env_info, dict)
            and "intervene_flag" in env_info
            and "intervene_action" in env_info
            and np.asarray(env_info["intervene_flag"]).all()
        ):
            np_action = self._to_numpy(env_info["intervene_action"])

        frame = LeRobotFrame.from_values(
            observation=prev_obs,
            action=np_action,
            task=self._stream_task[env_idx] or "unknown task",
            intervene_flag=self._intervene_flag_from_info(env_info),
            segment_id=int(self._segment_ids[env_idx]),
        )
        if frame is None:
            return

        # ``is_success``/``done`` are stamped episode-level by ``save_episode``
        # once the outcome is known.
        self._submit(
            self._stream_add_frame, frame.to_dict(episode_success=False, done=False)
        )
        self._stream_frames[env_idx] += 1

    def _stream_episode_success(self, env_idx: int) -> bool:
        """Episode success from per-step marks plus the sticky flag.

        Same precedence as ``_get_episode_success`` without the info buffer.
        """
        if self._episode_success[env_idx]:
            return True
        found_any = False
        is_success = False
        for mark in self._stream_success_marks[env_idx]:
            if mark is not None:
                found_any = True
                is_success = is_success or mark
        if found_any:
            return is_success
        return self._episode_success[env_idx]

    def _stream_end_episode(self, env_idx: int) -> None:
        """Close the in-progress streaming episode and start a fresh one."""
        if self._stream_frames[env_idx] > 0:
            if self._episode_discarded[env_idx]:
                self._submit(self._stream_discard_episode)
            else:
                self._submit(
                    self._stream_finish_episode,
                    self._stream_episode_success(env_idx)
                    and not self._stream_invalid[env_idx],
                    self._stream_invalid[env_idx],
                )
        self._episode_discarded[env_idx] = False
        self._episode_ids[env_idx] += 1
        self._episode_success[env_idx] = False
        self._segment_ids[env_idx] = 0
        self._stream_frames[env_idx] = 0
        self._stream_invalid[env_idx] = False
        self._stream_success_marks[env_idx] = []
        # Auto-reset envs carry the post-reset obs over as the next episode's
        # first frame, mirroring ``_reset_env_buffer``.
        self._stream_prev_obs[env_idx] = self._pending_obs[env_idx]
        self._pending_obs[env_idx] = None
        self._pending_info[env_idx] = None

    def _stream_discard_episode(self) -> None:
        """Discard only the current unpublished episode, after queued frames."""
        with self._lerobot_lock:
            if self._lerobot_writer is None or self._lerobot_writer.dataset is None:
                return
            dataset = self._lerobot_writer.dataset
            # LeRobot v2 only removes images when its image_writer is enabled;
            # our streaming writer writes PNGs directly and disables that queue.
            for key in dataset.meta.camera_keys:
                paths = dataset.episode_buffer[key]
                if paths:
                    shutil.rmtree(Path(paths[0]).parent)
            dataset.clear_episode_buffer()
            if self._stream_sidecar is not None:
                self._stream_sidecar.seek(self._stream_sidecar_episode_start)
                self._stream_sidecar.truncate()
                self._stream_sidecar.flush()
            self.logger.info("Discarded the current unpublished episode")

    def _stream_add_frame(self, frame: dict[str, Any]) -> None:
        """Writer-thread entry: create the dataset lazily, then add one frame."""
        with self._lerobot_lock:
            writer = self._ensure_lerobot_writer(frame)
            writer.add_frame(frame)
            self._write_stream_sidecar(writer, frame)

    def _write_stream_sidecar(self, writer: Any, frame: dict[str, Any]) -> None:
        """Append the frame's small fields to a per-shard JSONL sidecar.

        Called under ``_lerobot_lock`` right after ``add_frame``. Images are
        already durable per-frame, but ``state``/``actions`` only reach disk
        when ``save_episode`` finishes writing the (image-embedding) parquet,
        which can take minutes for long episodes; the sidecar makes the joint
        data crash-safe and cheap to read for tooling such as hand-eye
        calibration.
        """
        dataset = writer.dataset
        if dataset is None:
            return
        root = getattr(dataset, "root", None)
        if root is None:
            return
        if self._stream_sidecar_root != str(root):
            self._stream_sidecar_close()
            self._stream_sidecar = open(
                os.path.join(str(root), "stream_frames.jsonl"), "a"
            )
            self._stream_sidecar_root = str(root)
        episode_index = int(dataset.meta.total_episodes)
        frame_index = int(dataset.episode_buffer["size"]) - 1
        line = {
            "episode": episode_index,
            "frame": frame_index,
            "state": np.asarray(frame["state"]).tolist(),
            "actions": np.asarray(frame["actions"]).tolist(),
        }
        if frame_index == 0:
            self._stream_sidecar_episode_start = self._stream_sidecar.tell()
        self._stream_sidecar.write(json.dumps(line) + "\n")
        self._stream_sidecar.flush()

    def _stream_sidecar_close(self) -> None:
        if self._stream_sidecar is not None:
            self._stream_sidecar.close()
            self._stream_sidecar = None
            self._stream_sidecar_root = None

    def _stream_finish_episode(
        self, is_success: bool, recording_invalid: bool = False
    ) -> None:
        """Writer-thread entry: detach kept episodes for background saving."""
        with self._lerobot_lock:
            if self._lerobot_writer is None or self._lerobot_writer.dataset is None:
                return
            if recording_invalid:
                self._stream_isolate_invalid_episode()
                return
            writer = self._lerobot_writer
            self._stream_sidecar_close()
            self._lerobot_writer = None
        self._submit_save(self._save_detached_stream_episode, writer, is_success)

    def _stream_isolate_invalid_episode(self) -> None:
        """Move the active invalid episode out of the training shard."""
        dataset = self._lerobot_writer.dataset
        invalid_root = dataset.root / "invalid_episodes"
        invalid_root.mkdir(exist_ok=True)
        archive = tempfile.mkdtemp(prefix="overflow_", dir=invalid_root)
        buffer = dataset.episode_buffer
        # Preserve the contiguous recorded prefix for diagnosis, but never
        # publish it as a regular LeRobot training episode.
        for key in dataset.meta.camera_keys:
            if not buffer[key]:
                continue
            image_dir = os.path.dirname(buffer[key][0])
            destination = os.path.join(archive, key)
            shutil.move(image_dir, destination)
            buffer[key] = [
                os.path.join(destination, os.path.basename(path))
                for path in buffer[key]
            ]
        with open(os.path.join(archive, "frames.pkl"), "wb") as frames:
            pickle.dump(buffer, frames)
        with open(dataset.root / "recording_errors.jsonl", "a") as errors:
            errors.write(
                json.dumps(
                    {
                        "episode_index": dataset.meta.total_episodes,
                        "reason": "recording_queue_overflow",
                        "saved_frames": buffer["size"],
                        "archive": archive,
                    }
                )
                + "\n"
            )
        if self._stream_sidecar is not None:
            self._stream_sidecar.seek(self._stream_sidecar_episode_start)
            self._stream_sidecar.truncate()
            self._stream_sidecar.flush()
        dataset.clear_episode_buffer()
        self.logger.error("Isolated an incomplete episode at %s", archive)

    def _save_detached_stream_episode(self, writer: Any, is_success: bool) -> None:
        """Save and finalize one detached streaming shard."""
        primary_error = None
        try:
            writer.save_episode(is_success=is_success)
            with self._save_futures_lock:
                self._episodes_written += 1
        except BaseException as error:
            primary_error = error
        try:
            if writer.dataset is not None:
                writer.finalize()
        except BaseException as error:
            if primary_error is None:
                primary_error = error
        if primary_error is not None:
            raise primary_error

    def _buffer_to_lerobot_ep(
        self, buf: dict, env_idx: int, is_success: bool
    ) -> Optional[list[dict[str, Any]]]:
        """Convert a raw episode buffer into a list of per-step frame dicts.
        Produces the format expected by ``LeRobotDatasetWriter.add_episode``:
        a ``list[dict]`` where every dict represents one step and carries the
        fields ``image``, ``state``, ``actions``, ``task``, ``is_success``,
        ``done``, ``intervene_flag``, and optionally ``wrist_image`` /
        ``extra_view_image``.
        The observations list contains one extra entry prepended at reset time,
        so it is aligned to the actions list by taking the leading N entries.
        Steps where any required field (image, state, action) is missing are
        silently skipped.
        Args:
            buf: Raw episode buffer produced by ``_new_buffer``.
            env_idx: Index of the parallel environment this buffer belongs to.
            is_success: Whether the episode was successful.
        Returns:
            A list of per-step frame dicts, or ``None`` if no valid frames
            could be extracted.
        """
        actions = buf["actions"]
        terminated = buf["terminated"]
        obs_steps = buf["observations"]
        seg_ids = buf.get("segment_ids", [])
        if not actions:
            return None
        if len(obs_steps) > len(actions):
            obs_steps = obs_steps[: len(actions)]
        task_desc = self._extract_task_description(buf, env_idx)
        steps: list[dict[str, Any]] = []
        first_term_step: Optional[int] = None
        for i, action in enumerate(actions):
            obs = obs_steps[i] if i < len(obs_steps) else None
            # Overwrite action with intervene action if present.
            np_action = self._to_numpy(action)
            raw_info = buf["infos"][i + 1]
            if isinstance(raw_info, dict) and "final_info" in raw_info:
                # _record_step normally pops final_info before storing; if it
                # leaked through (e.g. nested wrapper that also auto-resets)
                # drop the frame instead of crashing the writer.
                self.logger.warning(
                    "collect_episode: dropping frame %d because info still "
                    "carries final_info; check upstream auto-reset wrappers",
                    i,
                )
                continue
            info_with_intervene = copy.deepcopy(raw_info)

            if (
                "intervene_flag" in info_with_intervene
                and "intervene_action" in info_with_intervene
            ):
                if info_with_intervene["intervene_flag"].all():
                    np_action = self._to_numpy(info_with_intervene["intervene_action"])
            intervene_flag = self._intervene_flag_from_info(info_with_intervene)
            seg_id = int(seg_ids[i]) if i < len(seg_ids) else 0
            frame = LeRobotFrame.from_values(
                observation=obs,
                action=np_action,
                task=task_desc,
                intervene_flag=intervene_flag,
                segment_id=seg_id,
            )
            if frame is None:
                continue
            steps.append(frame.to_dict(episode_success=is_success, done=False))
            if bool(terminated[i]) and first_term_step is None:
                first_term_step = len(steps)
        if not steps:
            return None
        end = first_term_step if first_term_step is not None else len(steps)
        steps = steps[:end]
        steps[-1]["done"] = np.array([True], dtype=bool)
        return steps

    def _ensure_lerobot_writer(self, first_frame: dict):
        """Get-or-create the LeRobot writer. Must be called under ``_lerobot_lock``.

        Args:
            first_frame: The episode's first frame dict, used to derive the
                dataset schema when the dataset does not exist yet.
        """
        from rlinf.data.storage.lerobot import LeRobotDatasetWriter

        if self._lerobot_writer is None:
            self._lerobot_writer = LeRobotDatasetWriter()
        shard_id = self._next_shard_id

        if self._lerobot_writer.dataset is None:
            first = first_frame
            wrist_image_keys = self._collect_image_keys(first, "wrist_image")
            extra_view_image_keys = self._collect_image_keys(first, "extra_view_image")
            self._lerobot_writer.create(
                repo_id=os.path.join(
                    self.save_dir, f"rank_{self.rank}", f"id_{shard_id}"
                ),
                robot_type=self.robot_type,
                fps=self.fps,
                # Image writes run in the existing export thread, never in
                # forked workers inheriting live camera/CAN descriptors.
                image_writer_processes=0,
                image_writer_threads=0,
                image_shape=first["image"].shape if "image" in first else None,
                state_dim=int(first["state"].shape[-1]),
                action_dim=int(first["actions"].shape[-1]),
                has_image="image" in first,
                wrist_image_keys=wrist_image_keys,
                extra_view_image_keys=extra_view_image_keys,
                has_intervene_flag="intervene_flag" in first,
                has_segment_id="segment_id" in first,
            )
            self._next_shard_id = shard_id + 1
        return self._lerobot_writer

    @staticmethod
    def _collect_image_keys(
        frame: dict[str, Any],
        prefix: str,
    ) -> dict[str, tuple[int, ...]]:
        """Return ``{key: (H, W, C)}`` for all frame keys matching *prefix*.

        Matches both the bare ``prefix`` (e.g. ``wrist_image``) and indexed
        variants (``wrist_image-0``, ``wrist_image-1``, …). The separator is a
        hyphen, not ``/``: lerobot >= 0.3 rejects feature names containing
        ``/`` in ``LeRobotDatasetMetadata.create``.
        """
        return {
            k: tuple(frame[k].shape)
            for k in frame
            if (k == prefix or k.startswith(f"{prefix}-"))
            and isinstance(frame[k], np.ndarray)
            and frame[k].ndim == 3
        }

    def _write_lerobot_episode(self, ep_data: dict) -> None:
        with self._lerobot_lock:
            writer = self._ensure_lerobot_writer(ep_data[0])
            writer.add_episode(ep_data)
            with self._save_futures_lock:
                self._episodes_written += 1
                count = self._episodes_written
            if (
                not self.streaming
                and self.finalize_interval > 0
                and count % self.finalize_interval == 0
            ):
                writer.finalize()

    def _finalize_lerobot(self) -> None:
        """Drain pending futures then write the LeRobot dataset metadata."""
        if self.export_format != "lerobot":
            return
        primary_error = None
        try:
            self._wait_futures()
        except BaseException as error:
            primary_error = error
        writer = None
        try:
            with self._lerobot_lock:
                self._stream_sidecar_close()
                if self._lerobot_writer is not None:
                    writer = self._lerobot_writer
                    self._lerobot_writer = None
        except BaseException as error:
            if primary_error is None:
                primary_error = error
        if writer is not None:
            try:
                writer.finalize()
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
        if primary_error is not None:
            raise primary_error

    def _write_pickle(self, save_path: str, episode_data: dict) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(episode_data, f)

    def _submit(self, fn, *args) -> None:
        if self._executor is None:
            return
        self._futures.append(self._executor.submit(fn, *args))
        self.logger.debug(f"Futures queue length: {len(self._futures)}")
        self._drain_futures()
        # Backpressure: cap in-flight write tasks so a slow disk turns into
        # loop latency instead of unbounded memory growth.
        # Streaming rejects new frames before submission when full; episode
        # closing tasks contain no images and must retain their FIFO order.
        while not self.streaming and len(self._futures) > _MAX_PENDING_FUTURES:
            self._futures.pop(0).result()

    def _submit_save(self, fn, *args) -> None:
        if self._save_executor is None:
            raise RuntimeError("streaming save executor is not available")
        future = self._save_executor.submit(fn, *args)
        with self._save_futures_lock:
            self._save_futures.append(future)
            queue_len = len(self._save_futures)
        self.logger.debug(f"Save futures queue length: {queue_len}")

    def _drain_futures(self) -> None:
        done = []
        remaining = []
        for f in self._futures:
            if f.done():
                done.append(f)
            else:
                remaining.append(f)
        self._futures = remaining
        first_error = self._settle_futures(done)
        if first_error is not None:
            raise first_error

    def _wait_futures(self) -> None:
        futures = self._futures
        self._futures = []
        first_error = self._settle_futures(futures)
        if first_error is not None:
            raise first_error

    def _drain_save_futures(self) -> None:
        with self._save_futures_lock:
            done = []
            remaining = []
            for f in self._save_futures:
                if f.done():
                    done.append(f)
                else:
                    remaining.append(f)
            self._save_futures = remaining
        first_error = self._settle_futures(done)
        if first_error is not None:
            raise first_error

    def _wait_save_futures(self) -> None:
        first_error = None
        while True:
            with self._save_futures_lock:
                futures = self._save_futures
                self._save_futures = []
            if not futures:
                break
            first_error = self._settle_futures(futures, first_error)
        if first_error is not None:
            raise first_error

    @staticmethod
    def _settle_futures(
        futures: list[Future], first_error: Optional[BaseException] = None
    ) -> Optional[BaseException]:
        for f in futures:
            try:
                f.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        return first_error

    def _finalize_on_exit(self) -> None:
        self.close()

    def _update_success(self, env_idx: int, env_info) -> None:
        """Update the per-env success flag from a single-env info dict."""
        if not isinstance(env_info, dict):
            return

        success = self._extract_success_from_info(env_info)
        if success is not None:
            # Keep success sticky during an episode.
            self._episode_success[env_idx] = self._episode_success[env_idx] or success

    def _get_episode_success(self, buf: dict, env_idx: int) -> bool:
        """Determine final episode success by scanning recorded info dicts.

        Checks (in priority order): ``final_info``, ``episode``, and the root
        info dict, looking for ``success_once``, ``success_at_end``, and
        ``success`` keys. Falls back to the incrementally-updated
        ``_episode_success`` flag.
        """
        if self._episode_success[env_idx]:
            return True

        found_any = False
        is_success = False

        for info in buf["infos"]:
            if not isinstance(info, dict):
                continue
            success = self._extract_success_from_info(info)
            if success is not None:
                found_any = True
                is_success = is_success or success

        if found_any:
            return is_success
        return self._episode_success[env_idx]

    @staticmethod
    def _intervene_flag_from_info(info: Any) -> bool:
        """Whether this timestep used human / expert intervention (per-env info)."""
        if not isinstance(info, dict):
            return False
        val = info.get("intervene_flag")
        if val is None:
            return False
        arr = CollectEpisode._to_numpy(val)
        if arr is None:
            return False
        return bool(np.asarray(arr, dtype=bool).reshape(-1).any())

    @staticmethod
    def _to_bool_scalar(val) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                return None
            return bool(val.item())
        if isinstance(val, np.ndarray):
            if val.size != 1:
                return None
            return bool(val.reshape(-1)[0])
        return bool(val)

    def _extract_success_from_source(self, src) -> Optional[bool]:
        if not isinstance(src, dict):
            return None
        for key in ("success_once", "success_at_end", "success"):
            val = self._to_bool_scalar(src.get(key))
            if val is not None:
                return val
        return None

    def _extract_success_from_info(self, info: dict) -> Optional[bool]:
        """Extract success with episode-level fields taking priority."""
        episode_values: list[bool] = []

        final_info = info.get("final_info", None)
        if isinstance(final_info, dict):
            final_info_success = self._extract_success_from_source(final_info)
            if final_info_success is not None:
                episode_values.append(final_info_success)
            final_episode_success = self._extract_success_from_source(
                final_info.get("episode")
            )
            if final_episode_success is not None:
                episode_values.append(final_episode_success)

        current_episode_success = self._extract_success_from_source(info.get("episode"))
        if current_episode_success is not None:
            episode_values.append(current_episode_success)

        if episode_values:
            return any(episode_values)

        return self._extract_success_from_source(info)

    def _extract_task_description(self, buf: dict, env_idx: int) -> str:
        for obs in reversed(buf["observations"]):
            if not isinstance(obs, dict) or "task_descriptions" not in obs:
                continue
            desc = obs["task_descriptions"]
            if isinstance(desc, (list, tuple)):
                return str(desc[env_idx] if len(desc) == self.num_envs else desc[0])
            return str(desc)
        return "unknown task"

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

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
