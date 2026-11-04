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

"""Official RLinf adapter for an externally launched GenieSim server."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from rlinf.envs.sim.geniesim.resolvers import (
    _client_kwargs,
    _override_dict,
    resolve_geniesim_address,
    resolve_geniesim_randomization,
    resolve_geniesim_render_width,
    resolve_geniesim_slot_overrides,
    resolve_geniesim_task_names,
    resolve_geniesim_work_item_shard,
)
from rlinf.envs.sim.geniesim.work_items import (
    GenieSimWorkItem,
    build_geniesim_reset_vectors,
    build_geniesim_work_items,
    parse_geniesim_task_metadata,
)
from rlinf.envs.utils import to_tensor
from rlinf.utils.logging import get_logger


def _nonfinite_state_error_types() -> tuple[type[BaseException], ...]:
    """Sim-side non-finite error classes, imported lazily.

    Returns an empty tuple when ``rlinf_geniesim`` is unavailable, which
    disables reset retries instead of breaking this module's import.
    """
    try:
        from rlinf_geniesim.envs.vec_env import SimStateNonFiniteError
    except Exception:  # pragma: no cover - depends on the deployment
        return ()
    return (SimStateNonFiniteError,)


# Raw simulator camera key -> the name used by the embodied observation.
_CAM_MAP = {
    "main_images": "main",
    "wrist_images": "wrist_left",
    "wrist_images_right": "wrist_right",
}
# Camera keys in the raw simulator observation.
_IMAGE_KEYS = frozenset(_CAM_MAP)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _make_geniesim_client(backend: str, **kwargs: Any) -> Any:
    from rlinf_geniesim.comm import make_client

    return make_client(backend, **kwargs)


def _make_geniesim_venv(client: Any, **kwargs: Any) -> Any:
    from rlinf_geniesim.envs.vec_env import GenieSimVecEnv

    return GenieSimVecEnv(client, **kwargs)


def _task_descriptions(raw_obs: dict[str, Any], num_envs: int) -> list[str]:
    raw = raw_obs.get("task_descriptions")
    if isinstance(raw, str) or not isinstance(raw, list | tuple):
        raise ValueError("GenieSim task_descriptions must be a list of strings")
    if len(raw) != num_envs:
        raise ValueError(
            f"GenieSim task_descriptions must have length {num_envs}, got {len(raw)}"
        )
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ValueError("GenieSim task_descriptions must contain non-empty strings")
    return [value.strip() for value in raw]


def _frame_batch(value: Any) -> np.ndarray:
    frames = _as_numpy(value)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected an NHWC image batch, got shape {frames.shape}")
    return np.ascontiguousarray(frames.astype(np.uint8, copy=False))


def wrap_geniesim_observation(
    raw_obs: dict[str, Any],
    *,
    main_image_key: str = "main",
    extra_image_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Convert one vector observation returned by ``GenieSimVecEnv``.

    Wrist cameras are stacked into ``extra_view_images`` in ``extra_image_keys``
    order. Task descriptions stay a list of strings.
    """
    states = _as_numpy(raw_obs["states"]).astype(np.float32, copy=False)
    if states.ndim != 2:
        raise ValueError(f"Expected a 2-D GenieSim state batch, got {states.shape}")
    num_envs = states.shape[0]
    if num_envs == 0:
        raise ValueError(
            "wrap_geniesim_observation received an empty observation batch"
        )

    descriptions = _task_descriptions(raw_obs, num_envs)
    frames = {
        dst: _frame_batch(raw_obs[src])
        for src, dst in _CAM_MAP.items()
        if src in raw_obs
    }
    if main_image_key not in frames:
        raise KeyError(
            f"GenieSim observation is missing main camera {main_image_key!r}"
        )
    if any(frame.shape[0] != num_envs for frame in frames.values()):
        raise ValueError("GenieSim image batches must match the state batch size")

    out: dict[str, Any] = {
        "states": states,
        "main_images": frames[main_image_key],
    }
    extra_keys = (
        list(extra_image_keys)
        if extra_image_keys is not None
        else ["wrist_left", "wrist_right"]
    )
    extra_keys = extra_keys or sorted(key for key in frames if key != main_image_key)
    if extra_keys:
        missing = [key for key in extra_keys if key not in frames]
        if missing:
            raise KeyError(f"GenieSim observation is missing cameras: {missing}")
        out["extra_view_images"] = np.stack([frames[key] for key in extra_keys], axis=1)
    tensor_out = to_tensor(out)
    tensor_out["task_descriptions"] = descriptions
    return tensor_out


class GenieSimEnv:
    """Thin env wrapper matching official EnvWorker's reset/chunk_step contract."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info=None,
    ) -> None:
        self.cfg = cfg
        self._num_envs = int(num_envs)
        # Logical slot identity includes the worker rank and pipeline stage.
        self._seed_offset = int(seed_offset)
        self._total_num_processes = int(total_num_processes)
        # The physical worker rank is deliberately not kept: it aliases every
        # pipeline stage of a process onto one slot.
        del worker_info
        self._auto_reset = bool(cfg.get("auto_reset", False))
        self._ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self._max_episode_steps = int(cfg.get("max_episode_steps", 200))

        override = _override_dict(cfg)
        # Per-logical-slot config, if any. Applied before every other key is
        # read, so a slot can override anything the shared block sets (task,
        # instances, episode budget, physics backend, ...).
        self._slot_overrides = resolve_geniesim_slot_overrides(
            override, world_size=self._total_num_processes
        )
        if self._slot_overrides is not None:
            override = self._slot_overrides[self._seed_offset]
        # Retry only reset failures; a lost step has no valid transition to return.
        self._nonfinite_state_recovery = bool(
            override.get("nonfinite_state_recovery", True)
        )
        self._nonfinite_state_reset_retries = int(
            override.get("nonfinite_state_reset_retries", 3)
        )
        if self._nonfinite_state_reset_retries < 1:
            raise ValueError(
                f"GenieSim nonfinite_state_reset_retries must be >= 1, "
                f"got {self._nonfinite_state_reset_retries}"
            )
        self._address = resolve_geniesim_address(
            override,
            rank=self._seed_offset,
            world_size=self._total_num_processes,
        )
        self._task_names = resolve_geniesim_task_names(override)
        self._sub_task_name = str(override.get("sub_task_name", "")).strip()
        if not self._task_names or not self._sub_task_name:
            raise ValueError("GenieSim task_name and sub_task_name are required")

        backend, client_kwargs = _client_kwargs(override, address=self._address)
        self._client = _make_geniesim_client(backend, **client_kwargs)
        backend_cfg = dict(override.get("backend_cfg") or {})
        if backend_cfg:
            self._client.set_backends(backend_cfg)
        self._venv = _make_geniesim_venv(
            self._client,
            to_torch=bool(override.get("venv_to_torch", True)),
            # Keep per-step boundaries; this adapter owns auto-reset.
            auto_reset=False,
            # The client truncates on its own counter, so it needs the same
            # limit; its default is 200 and would cut a longer episode short.
            max_episode_steps=self._max_episode_steps,
        )
        # Settings for the task init and the work-item catalogue.
        self._render_width = resolve_geniesim_render_width(override)
        self._randomization = resolve_geniesim_randomization(override)
        self._eval_interval = int(override.get("eval_interval", 1))
        self._work_item_spec = {
            "instance_ids": override.get("instance_ids"),
            "instruction_ids_by_instance": override.get("instruction_ids_by_instance"),
            "num_episodes": int(override.get("num_episodes", 1)),
            "budget": int(override.get("work_item_budget", 0) or 0),
        }
        self._need_images = bool(override.get("need_images", True))
        # Render only the last action substep when enabled.
        self._images_at_chunk_end_only = bool(
            override.get("images_at_chunk_end_only", False)
        )
        # Video recording requires a frame for every substep.
        self._save_video = bool((cfg.get("video_cfg") or {}).get("save_video", False))
        self._base_seed = int(cfg.get("seed", 0))
        self._cycle_work_items = bool(override.get("cycle_work_items", True))

        self._main_image_key = str(cfg.get("main_image_key", "main"))
        self._extra_image_keys = (
            list(cfg.extra_image_keys) if "extra_image_keys" in cfg else None
        )
        self._elapsed_steps = np.zeros(self._num_envs, dtype=np.int32)
        # Shard work by task catalogue and GRPO group; group members share an item and seed.
        (
            self._work_item_shard_rank,
            self._work_item_shard_world_size,
        ) = resolve_geniesim_work_item_shard(
            override,
            self._slot_overrides,
            rank=self._seed_offset,
            world_size=self._total_num_processes,
        )
        self._group_size = int(cfg.get("group_size", 1) or 1)
        if self._group_size < 1:
            raise ValueError(
                f"GenieSim group_size must be >= 1, got {self._group_size}"
            )
        if self._num_envs % self._group_size != 0:
            # ``config.py`` does not check this pairing, so without the raise a
            # trailing partial group would quietly get its own work item and the
            # GRPO baseline would be a mean over *different* initial states --
            # which looks exactly like "the run just does not learn".
            raise ValueError(
                f"GenieSim group_size {self._group_size} must divide the per-process "
                f"env count {self._num_envs}"
            )
        if self._group_size > 1 and self._auto_reset:
            # Auto-reset hands only the finished envs to reset(). That advances
            # the whole group's work item while the other members are still in
            # the previous episode, so a group-relative baseline compares
            # different tasks. GRPO keeps one fixed horizon and resets the
            # group together at the epoch boundary.
            raise ValueError(
                "GenieSim group_size > 1 requires auto_reset: False. "
                "An automatic reset only includes the envs that finished, which "
                "advances the whole group's work item while the other members "
                "are still in the previous episode."
            )
        self._num_group = self._num_envs // self._group_size
        self._work_stride = self._work_item_shard_world_size * self._num_group

        self._metadata = parse_geniesim_task_metadata(
            self._client.task_metadata(sub_task_name=self._sub_task_name),
            task_names=self._task_names,
            sub_task_name=self._sub_task_name,
        )
        self._bind_sub_task()
        self.is_start = True
        # Track the first reset independently of the worker's is_start flag.
        self._first_reset = True
        # One-shot flag for the group init-state agreement check below.
        self._group_state_checked = False
        # Reuse camera frames when intermediate substeps skip rendering.
        self._last_raw_obs: dict[str, Any] | None = None
        self._last_is_success = np.zeros(self._num_envs, dtype=bool)
        # PPO uses the change in episode progress as reward.
        self._prev_progress = np.zeros(self._num_envs, dtype=np.float32)
        self._episode_return = np.zeros(self._num_envs, dtype=np.float32)
        self._success_once = np.zeros(self._num_envs, dtype=bool)
        self._terminated = np.zeros(self._num_envs, dtype=bool)
        self._truncated = np.zeros(self._num_envs, dtype=bool)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def seed(self) -> int:
        """Effective seed of this env instance.

        ``rlinf.envs.wrappers.RecordVideo`` requires this attribute (it refuses
        to wrap an env without one) and names its output directory after it.
        """
        # -1 disables reset randomization on every server. Do not turn it into
        # a nonnegative seed on slots with a positive offset.
        if self._base_seed == -1:
            return -1
        return self._base_seed + self._seed_offset

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _init_config(self) -> dict[str, Any] | None:
        """Collect optional task-init sections without changing server defaults."""
        config: dict[str, Any] = {}
        if self._render_width is not None:
            config["observation"] = {"render_width": self._render_width}
        if self._randomization is not None:
            config["randomization"] = copy.deepcopy(self._randomization)
        return config or None

    def _bind_sub_task(self) -> None:
        """Initialize the server task and lay out this slot's work-item cursor."""
        # The sub-task can select any background from the configured list.
        self._task_name = self._metadata.task_name
        self._venv.init(
            {
                "task_name": self._task_name,
                "sub_task_name": self._sub_task_name,
                "num_envs": self._num_envs,
                "eval_interval": self._eval_interval,
            },
            config=self._init_config(),
        )
        self._action_dim = int(self._venv.action_dim)

        work_items = build_geniesim_work_items(
            self._metadata,
            instance_ids=self._work_item_spec["instance_ids"],
            instruction_ids_by_instance=self._work_item_spec[
                "instruction_ids_by_instance"
            ],
            num_episodes=self._work_item_spec["num_episodes"],
        )
        budget = self._work_item_spec["budget"]
        if budget > 0:
            work_items = work_items[:budget]
        # Slots with distinct catalogues start at their own catalogue offset.
        base = self._work_item_shard_rank * self._num_group
        self._work_items = work_items
        self._work_cursor = [
            (base + env_id // self._group_size) % len(work_items)
            for env_id in range(self._num_envs)
        ]

        get_logger().info(
            f"[geniesim] slot={self._seed_offset}/{self._total_num_processes} "
            f"address={self._address} "
            f"task={self._task_name} sub_task={self._sub_task_name} "
            f"num_envs={self._num_envs} server action_dim={self._action_dim} "
            f"work_items={len(self._work_items)} "
            f"shard={self._work_item_shard_rank}/{self._work_item_shard_world_size} "
            f"group_size={self._group_size} num_group={self._num_group} "
            f"cursor={self._work_cursor} stride={self._work_stride}"
        )

    @property
    def _success_metric_key(self) -> str:
        """Task-specific metric key for the currently bound task."""
        return f"success/{self._sub_task_name}"

    def _current_work_items(self) -> list[GenieSimWorkItem]:
        return [self._work_items[cursor] for cursor in self._work_cursor]

    def _advance_work(self, group_id: int) -> None:
        """Move one whole group to its next work item, in lockstep.

        The cursor stays a per-env list so that every reader keeps indexing by
        env id (``_current_work_items``, the reset log); it is
        the *value* that a group shares, so it is read from the group's first env
        and written back to all of them. At ``group_size == 1`` that is the old
        single-element update, unchanged.
        """
        first = group_id * self._group_size
        next_cursor = self._work_cursor[first] + self._work_stride
        if next_cursor >= len(self._work_items):
            next_cursor = (
                next_cursor % len(self._work_items)
                if self._cycle_work_items
                else self._work_cursor[first]
            )
        for env_id in range(first, first + self._group_size):
            self._work_cursor[env_id] = next_cursor

    def _venv_reset_with_retry(
        self,
        ids: np.ndarray,
        seeds: np.ndarray,
        instance_ids: np.ndarray,
        instruction_ids: np.ndarray,
    ) -> tuple[dict, dict]:
        """Reset, retrying when the reset itself returns non-finite state.

        Retries only when recovery is enabled. A bad frame here means the
        fresh episode diverged before its first step, so there is no last-good
        obs to fall back on -- retrying the reset is the only recovery
        available.
        """
        attempts = (
            self._nonfinite_state_reset_retries if self._nonfinite_state_recovery else 1
        )
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return self._venv.reset(
                    env_ids=ids.tolist(),
                    seeds=seeds,
                    instance_ids=instance_ids,
                    instruction_ids=instruction_ids,
                )
            except _nonfinite_state_error_types() as exc:
                if not self._nonfinite_state_recovery:
                    raise
                last_exc = exc
                get_logger().error(
                    "[geniesim] reset returned non-finite state (attempt %d/%d), "
                    "envs=%s: %s",
                    attempt + 1,
                    attempts,
                    ids.tolist(),
                    exc,
                )
        raise RuntimeError(
            f"GenieSim reset returned non-finite state on all {attempts} attempts, "
            f"envs={ids.tolist()}"
        ) from last_exc

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return wrap_geniesim_observation(
            raw_obs,
            main_image_key=self._main_image_key,
            extra_image_keys=self._extra_image_keys,
        )

    def reset(
        self, *, seed: int | None = None, options=None, env_idx=None
    ) -> tuple[dict, dict]:
        del options
        if seed is not None:
            self._base_seed = int(seed)
        ids = (
            np.arange(self._num_envs, dtype=np.int64)
            if env_idx is None
            else np.asarray(env_idx, dtype=np.int64).reshape(-1)
        )
        # The first reset consumes each env's initial work item; later resets
        # advance each group once to keep its work item and seed aligned.
        if not self._first_reset:
            groups = sorted({env_id // self._group_size for env_id in ids.tolist()})
            if self._group_size > 1 and int(ids.size) != self._group_size * len(groups):
                # Partial group resets break GRPO initial-state alignment.
                get_logger().warning(
                    "[geniesim] partial group reset with group_size=%d: "
                    "env_idx=%s covers %d group(s) but only %d env(s); group "
                    "members will stop sharing a work item",
                    self._group_size,
                    ids.tolist(),
                    len(groups),
                    int(ids.size),
                )
            for group_id in groups:
                self._advance_work(group_id)

        seeds, instance_ids, instruction_ids = build_geniesim_reset_vectors(
            self._num_envs,
            ids,
            self._current_work_items(),
            self.seed,
        )
        raw_obs, infos = self._venv_reset_with_retry(
            ids, seeds, instance_ids, instruction_ids
        )
        self._last_raw_obs = raw_obs
        self._prev_progress[ids] = 0.0
        self._last_is_success[ids] = False
        self._elapsed_steps[ids] = 0
        self._episode_return[ids] = 0.0
        self._success_once[ids] = False
        self._terminated[ids] = False
        self._truncated[ids] = False
        self.is_start = False
        self._first_reset = False
        obs = self._wrap_obs(raw_obs)
        # Log the resolved prompt and work item to diagnose server-side instruction fallback.
        get_logger().info(
            f"[geniesim] instruction[0]={obs['task_descriptions'][0]!r} "
            f"(instance={instance_ids[0]} instruction_id={instruction_ids[0]} "
            f"work_item={self._work_cursor[int(ids[0])] + 1}/{len(self._work_items)} "
            f"group_size={self._group_size} "
            f"groups={sorted({c + 1 for c in self._work_cursor})})"
        )
        self._log_group_state_agreement(obs)
        return obs, dict(infos)

    def _log_group_state_agreement(self, obs: dict) -> None:
        """Once per run, measure whether a group really shares its initial state.

        Sharing a work item gives the group identical reset *arguments*; that the
        resulting worlds are identical is the server's side of the contract, and
        nothing else in the run would ever check it. A GRPO baseline built over
        group members that actually started from different states is not an
        error anywhere -- it just makes the advantage noise, i.e. a run that
        trains and does not improve. So compare the group members' proprio state
        vectors on the first reset and print the worst disagreement.

        No-op at ``group_size == 1`` (nothing to compare) and after the first
        reset, so PPO configs never reach past the first line.
        """
        if self._group_size < 2 or self._group_state_checked:
            return
        self._group_state_checked = True
        raw_states = obs.get("states")
        if raw_states is None:
            return
        states = _as_numpy(raw_states)
        if states.ndim != 2 or states.shape[0] != self._num_envs:
            return
        worst_group, worst_dev = -1, 0.0
        for group_id in range(self._num_group):
            first = group_id * self._group_size
            block = states[first : first + self._group_size]
            dev = float(np.max(np.abs(block - block[0]))) if block.size else 0.0
            if dev >= worst_dev:
                worst_group, worst_dev = group_id, dev
        log = get_logger().info if worst_dev == 0.0 else get_logger().warning
        log(
            "[geniesim] group init-state check: group_size=%d num_group=%d "
            "max |state - state_of_group_member_0| = %.3e (worst group %d); "
            "0 means the group really does share its initial state, which is "
            "what a group-relative (GRPO) baseline needs",
            self._group_size,
            self._num_group,
            worst_dev,
            worst_group,
        )

    @property
    def supports_chunk_render_skip(self) -> bool:
        """Whether configured chunk rendering can skip intermediate frames."""
        return self._need_images and self._images_at_chunk_end_only

    def _fill_unrendered_frames(self, raw_obs: dict[str, Any]) -> bool:
        """Fill missing camera frames from the last rendered observation."""
        missing = [
            key
            for key, value in raw_obs.items()
            if key in _IMAGE_KEYS and value is None
        ]
        if not missing:
            return False
        if not self.supports_chunk_render_skip:
            # Never train on a frozen reset frame when images are disabled.
            raise RuntimeError(
                f"need_images=false leaves the camera keys {missing} empty on every "
                "step, but the observation adapter requires camera frames. Backfilling "
                "them would freeze the image at the reset frame and train on it. Set "
                "need_images=true, and use images_at_chunk_end_only to cut render cost."
            )
        cached = self._last_raw_obs
        if cached is None or any(cached.get(key) is None for key in missing):
            raise RuntimeError(
                "chunk-end-only rendering has no cached frame to reuse for "
                f"{missing}: the first observation of an episode must be a real "
                "frame. This means reset returned no image, which should not happen."
            )
        for key in missing:
            raw_obs[key] = cached[key]
        return True

    def step(
        self,
        action: np.ndarray,
        *,
        need_images: bool | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Render every substep while recording video.
        if self._save_video and need_images is False:
            need_images = True
        actions = np.ascontiguousarray(_as_numpy(action), dtype=np.float32)
        if actions.ndim == 1 and self._num_envs == 1:
            actions = actions[None, :]
        expected = (self._num_envs, self._action_dim)
        if actions.shape != expected:
            raise ValueError(
                f"GenieSim actions must have shape {expected}, got {actions.shape}"
            )
        # The environment-level flag controls all image rendering.
        step_images = self._need_images and (
            True if need_images is None else need_images
        )
        raw_obs, rewards, terminations, truncations, infos = self._venv.step(
            actions[:, None, :],
            auto_reset=False,
            need_images=step_images,
            return_per_step_obs=False,
        )
        # Fill skipped frames before replacing the cached observation.
        self._fill_unrendered_frames(raw_obs)
        self._last_raw_obs = raw_obs
        obs = self._wrap_obs(raw_obs)
        # Use progress at the end of the action chunk.
        rewards = _as_numpy(rewards).astype(np.float32)[:, -1]
        terminations = _as_numpy(terminations).astype(bool)[:, 0]
        truncations = _as_numpy(truncations).astype(bool)[:, 0]
        success = _as_numpy(infos.get("is_success", np.zeros(self._num_envs))).astype(
            bool
        )
        return self._record_step(
            obs, rewards, success, terminations, truncations, infos
        )

    def _episode_info(self, info: dict) -> dict:
        """Attach adapter-owned metrics, measured until the first episode boundary."""
        info = dict(info)
        success = to_tensor(self._last_is_success.astype(np.float32))
        info["episode"] = {
            "success": success,
            self._success_metric_key: success.clone(),
            "success_once": to_tensor(self._success_once.astype(np.float32)),
            "success_at_end": to_tensor(self._last_is_success.copy()),
            "return": to_tensor(self._episode_return.copy()),
            "length": to_tensor(self._elapsed_steps.astype(np.float32)),
        }
        return info

    def _record_step(self, obs, progress, success, terminated, truncated, info):
        """Credit active episodes once; latch boundaries until an explicit reset."""
        active = ~(self._terminated | self._truncated)
        delta = np.zeros(self._num_envs, dtype=np.float32)
        delta[active] = progress[active] - self._prev_progress[active]
        self._prev_progress[active] = progress[active]
        self._last_is_success[active] = success[active]
        self._success_once |= success & active
        self._episode_return += delta
        self._elapsed_steps[active] += 1
        if not self._ignore_terminations:
            self._terminated |= (terminated | success) & active
        self._truncated |= (
            truncated | (self._elapsed_steps >= self._max_episode_steps)
        ) & active
        info = self._episode_info(info)
        info["success"] = self._last_is_success.copy()
        info["geniesim_progress_delta"] = delta.copy()
        if (self._terminated | self._truncated).any():
            info["final_observation"] = obs
        return (
            obs,
            to_tensor(delta),
            to_tensor(self._terminated.copy()),
            to_tensor(self._truncated.copy()),
            info,
        )

    def _finish_chunk(self, observations, rewards, terminated, truncated, infos):
        """Reset once at the chunk boundary and preserve terminal metrics."""
        done = _as_numpy(terminated[-1] | truncated[-1]).astype(bool)
        if done.any() and self._auto_reset:
            final_info = infos[-1]
            final_observation = observations[-1]
            observations[-1], infos[-1] = self.reset(env_idx=np.flatnonzero(done))
            infos[-1]["final_info"] = final_info
            infos[-1]["final_observation"] = final_observation
            infos[-1]["_final_info"] = done.copy()
            infos[-1]["_final_observation"] = done.copy()
        return (
            observations,
            torch.stack(rewards, dim=1),
            torch.stack(terminated, dim=1),
            torch.stack(truncated, dim=1),
            infos,
        )

    def chunk_step(
        self, actions: np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        action_chunk = np.ascontiguousarray(_as_numpy(actions), dtype=np.float32)
        if action_chunk.ndim == 2 and self._num_envs == 1:
            action_chunk = action_chunk[None, :, :]
        expected_prefix = (self._num_envs,)
        if (
            action_chunk.ndim != 3
            or action_chunk.shape[:1] != expected_prefix
            or action_chunk.shape[2] != self._action_dim
        ):
            raise ValueError(
                "GenieSim action chunks must have shape "
                f"({self._num_envs}, chunk, {self._action_dim}), got {action_chunk.shape}"
            )

        if self._can_batch_chunk(action_chunk.shape[1]):
            return self._chunk_step_batched(action_chunk)

        obs_list = []
        rewards = []
        terminations = []
        truncations = []
        infos_list = []
        # Intermediate substeps reuse the last rendered frame when enabled.
        num_substeps = action_chunk.shape[1]
        render_every_substep = not self.supports_chunk_render_skip
        for chunk_idx in range(num_substeps):
            obs, reward, terminated, truncated, info = self.step(
                action_chunk[:, chunk_idx],
                need_images=(
                    True if render_every_substep else chunk_idx == num_substeps - 1
                ),
            )
            obs_list.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos_list.append(info)
        return self._finish_chunk(
            obs_list, rewards, terminations, truncations, infos_list
        )

    def _can_batch_chunk(self, num_substeps: int) -> bool:
        """Whether the whole chunk can go to the simulator in one call.

        ``GenieSimVecEnv.step`` takes an ``(N, chunk, A)`` chunk and decides by
        itself whether the server can run it in one round trip. It only returns
        the chunk's last frame, so a profile that needs every frame keeps
        stepping substep by substep.
        """
        return (
            num_substeps > 1
            and self.supports_chunk_render_skip
            and not self._save_video
        )

    def _chunk_step_batched(self, action_chunk: np.ndarray):
        """Send one action chunk and reconstruct RLinf's per-step tensors.

        The simulator returns per-substep progress, termination, truncation and
        success arrays, plus the chunk's last frame. Episode resets stay here
        because only this adapter owns the work-item cursor and the reset seed.
        Intermediate frames do not exist, so the last real frame stands in for
        them.
        """
        n, horizon, _ = action_chunk.shape
        raw_obs, c_rew, c_term, c_trunc, venv_info = self._venv.step(
            action_chunk,
            auto_reset=False,
            need_images=self._need_images,
            return_per_step_obs=False,
        )
        if "chunk_success" not in venv_info:
            raise RuntimeError(
                "GenieSim step() returned no 'chunk_success'; per-substep "
                "success is required to credit a chunk. Expected the key from "
                "a chunk-aware client, got keys: "
                f"{sorted(venv_info)[:8]}"
            )
        progress = _as_numpy(c_rew).astype(np.float32).reshape(n, horizon)
        terms = _as_numpy(c_term).astype(bool).reshape(n, horizon)
        truncs = _as_numpy(c_trunc).astype(bool).reshape(n, horizon)
        successes = (
            _as_numpy(venv_info["chunk_success"]).astype(bool).reshape(n, horizon)
        )
        obs = self._wrap_obs(raw_obs)
        self._last_raw_obs = raw_obs

        rewards_out = []
        terms_out = []
        truncs_out = []
        infos_out = []
        for index in range(horizon):
            info = dict(venv_info) if index == horizon - 1 else {}
            info.pop("chunk_success", None)
            _, reward, term, trunc, info = self._record_step(
                obs,
                progress[:, index],
                successes[:, index],
                terms[:, index],
                truncs[:, index],
                info,
            )
            rewards_out.append(reward)
            terms_out.append(term)
            truncs_out.append(trunc)
            infos_out.append(info)
        return self._finish_chunk(
            [obs] * horizon, rewards_out, terms_out, truncs_out, infos_out
        )

    def update_reset_state_ids(self, *args, **kwargs) -> None:
        del args, kwargs

    def flush_video(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        for resource in (getattr(self, "_venv", None), getattr(self, "_client", None)):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
