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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from rlinf.envs.behavior.instance_loader import parse_activity_instance_ids


@dataclass(frozen=True)
class ReplayEpisode:
    episode_index: int
    instance_id: int
    parquet_path: Path
    annotation_path: Path | None
    orchestrator_path: Path | None


@dataclass(frozen=True)
class ReplayPlan:
    episode_index: int
    instance_id: int
    actions: np.ndarray
    replay_steps: int
    target_step: int
    stage_prompts: tuple[str, ...] = ()


class BehaviorReplayInitializer:
    """Replay BEHAVIOR demonstration prefixes before policy rollout starts."""

    def __init__(self, cfg: DictConfig, seed_offset: int = 0):
        replay_cfg = OmegaConf.select(cfg, "replay_init")
        self.enabled = bool(
            replay_cfg is not None
            and OmegaConf.select(replay_cfg, "enabled", default=False)
        )
        if not self.enabled:
            return

        self.use_subtask_prompt = bool(
            OmegaConf.select(cfg, "use_subtask_prompt", default=False)
        )
        self.dataset_root = Path(
            OmegaConf.select(replay_cfg, "dataset_root", default="")
        ).expanduser()
        if not self.dataset_root.is_dir():
            raise ValueError(
                f"env.replay_init.dataset_root must be an existing directory, got {self.dataset_root}"
            )

        self.task_id = int(OmegaConf.select(replay_cfg, "task_id", default=0))
        self.task_dir = f"task-{self.task_id:04d}"
        self.action_column = str(
            OmegaConf.select(replay_cfg, "action_column", default="action")
        )
        self.stage_index = OmegaConf.select(replay_cfg, "stage_index", default=None)
        if self.stage_index is not None:
            self.stage_index = int(self.stage_index)
        self.stage_boundary = str(
            OmegaConf.select(replay_cfg, "stage_boundary", default="start")
        ).lower()
        if self.stage_boundary not in ("start", "end"):
            raise ValueError("env.replay_init.stage_boundary must be 'start' or 'end'.")
        self.target_step = OmegaConf.select(replay_cfg, "target_step", default=None)
        if self.target_step is not None:
            self.target_step = int(self.target_step)

        self.replay_ratio = float(
            OmegaConf.select(replay_cfg, "replay_ratio", default=1.0)
        )
        if self.replay_ratio < 0:
            raise ValueError("env.replay_init.replay_ratio must be non-negative.")
        self.min_replay_steps = int(
            OmegaConf.select(replay_cfg, "min_replay_steps", default=0)
        )
        self.max_replay_steps = OmegaConf.select(
            replay_cfg, "max_replay_steps", default=None
        )
        if self.max_replay_steps is not None:
            self.max_replay_steps = int(self.max_replay_steps)
        self.jitter_steps = int(OmegaConf.select(replay_cfg, "jitter_steps", default=0))

        try:
            seed_value = OmegaConf.select(replay_cfg, "seed", default=None)
        except Exception:
            seed_value = None
        if seed_value is None:
            seed_value = cfg.get("seed", 0)
        seed = int(seed_value) + int(seed_offset)
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.action_noise_std = float(
            OmegaConf.select(replay_cfg, "action_noise_std", default=0.0)
        )
        if self.action_noise_std < 0:
            raise ValueError("env.replay_init.action_noise_std must be non-negative.")
        self.action_noise_indices = self._parse_int_sequence(
            OmegaConf.select(replay_cfg, "action_noise_indices", default=None)
        )
        self.action_noise_stage_indices = self._parse_int_sequence(
            OmegaConf.select(replay_cfg, "action_noise_stage_indices", default=None)
        )
        self._action_cache: dict[int, np.ndarray] = {}
        self.allowed_instance_ids = self._resolve_allowed_instance_ids(cfg)

        requested_episode_ids = OmegaConf.select(
            replay_cfg, "episode_ids", default=None
        )
        if isinstance(requested_episode_ids, ListConfig):
            requested_episode_ids = OmegaConf.to_container(requested_episode_ids)
        if requested_episode_ids is not None:
            requested_episode_ids = [int(x) for x in requested_episode_ids]

        self.episodes = self._discover_episodes(requested_episode_ids)
        if not self.episodes:
            raise ValueError(
                f"No replay episodes found under {self.dataset_root / 'data' / self.task_dir}."
            )
        self._episodes_by_instance: dict[int, list[ReplayEpisode]] = {}
        for episode in self.episodes:
            self._episodes_by_instance.setdefault(episode.instance_id, []).append(
                episode
            )

    def sample_plan_for_instance(self, instance_id: int) -> ReplayPlan:
        episodes = self._episodes_by_instance.get(int(instance_id), [])
        if not episodes:
            raise ValueError(
                f"No replay episodes found for task_id={self.task_id} "
                f"and instance_id={instance_id}."
            )
        return self._build_plan(self._py_rng.choice(episodes))

    def sample_plans(self, num_envs: int) -> list[ReplayPlan]:
        episodes = [self._py_rng.choice(self.episodes) for _ in range(num_envs)]
        return [self._build_plan(episode) for episode in episodes]

    def sample_grouped_plans(self, num_envs: int, group_size: int) -> list[ReplayPlan]:
        if group_size <= 1:
            return self.sample_plans(num_envs)
        if num_envs % group_size != 0:
            raise ValueError(
                f"num_envs={num_envs} must be divisible by group_size={group_size} "
                "for grouped BEHAVIOR replay initialization."
            )
        group_plans = self.sample_plans(num_envs // group_size)
        plans: list[ReplayPlan] = []
        for plan in group_plans:
            plans.extend([plan] * group_size)
        return plans

    def _discover_episodes(
        self, requested_episode_ids: list[int] | None
    ) -> list[ReplayEpisode]:
        data_dir = self.dataset_root / "data" / self.task_dir
        annotation_dir = self.dataset_root / "annotations" / self.task_dir
        orchestrator_dir = self.dataset_root / "orchestrators" / self.task_dir
        episodes = []
        requested = set(requested_episode_ids) if requested_episode_ids else None
        for parquet_path in sorted(data_dir.glob("episode_*.parquet")):
            episode_index = self._parse_episode_index(parquet_path.stem)
            if requested is not None and episode_index not in requested:
                continue
            instance_id = self._episode_index_to_instance_id(episode_index)
            if (
                self.allowed_instance_ids is not None
                and instance_id not in self.allowed_instance_ids
            ):
                continue
            annotation_path = annotation_dir / f"episode_{episode_index:08d}.json"
            orchestrator_path = (
                orchestrator_dir
                / f"episode_{episode_index:08d}"
                / "task_annotated.json"
            )
            episodes.append(
                ReplayEpisode(
                    episode_index=episode_index,
                    instance_id=instance_id,
                    parquet_path=parquet_path,
                    annotation_path=annotation_path
                    if annotation_path.is_file()
                    else None,
                    orchestrator_path=orchestrator_path
                    if orchestrator_path.is_file()
                    else None,
                )
            )
        return episodes

    @staticmethod
    def _resolve_allowed_instance_ids(cfg: DictConfig) -> set[int] | None:
        instance_ids = OmegaConf.select(
            cfg, "omni_config.task.activity_instance_id", default=None
        )
        parsed_instance_ids = parse_activity_instance_ids(instance_ids)
        return set(parsed_instance_ids) if parsed_instance_ids is not None else None

    def _build_plan(self, episode: ReplayEpisode) -> ReplayPlan:
        actions = self._load_actions(episode)
        target_step = self._resolve_target_step(episode, len(actions))
        replay_steps = int(round(target_step * self.replay_ratio))
        if self.jitter_steps > 0:
            replay_steps += self._py_rng.randint(-self.jitter_steps, self.jitter_steps)
        replay_steps = max(self.min_replay_steps, replay_steps)
        if self.max_replay_steps is not None:
            replay_steps = min(self.max_replay_steps, replay_steps)
        replay_steps = min(max(replay_steps, 0), len(actions))

        replay_actions = actions[:replay_steps].copy()
        self._apply_action_noise(replay_actions, episode)
        return ReplayPlan(
            episode_index=episode.episode_index,
            instance_id=episode.instance_id,
            actions=replay_actions,
            replay_steps=replay_steps,
            target_step=target_step,
            stage_prompts=(
                self._load_stage_prompts(episode) if self.use_subtask_prompt else ()
            ),
        )

    @staticmethod
    def _parse_int_sequence(value) -> tuple[int, ...] | None:
        parsed = parse_activity_instance_ids(value)
        return tuple(parsed) if parsed is not None else None

    def _apply_action_noise(self, actions: np.ndarray, episode: ReplayEpisode) -> None:
        if self.action_noise_std <= 0 or actions.size == 0:
            return

        noise_steps = self._resolve_action_noise_steps(episode, len(actions))
        if noise_steps.size == 0:
            return

        if self.action_noise_indices is None:
            action_indices = np.arange(actions.shape[-1], dtype=np.int64)
        else:
            action_indices = np.asarray(self.action_noise_indices, dtype=np.int64)
            if np.any(action_indices < 0) or np.any(
                action_indices >= actions.shape[-1]
            ):
                raise ValueError(
                    "env.replay_init.action_noise_indices must be within replay "
                    f"action dimension [0, {actions.shape[-1]}), got "
                    f"{self.action_noise_indices}."
                )
        if action_indices.size == 0:
            return

        noise = self._np_rng.normal(
            loc=0.0,
            scale=self.action_noise_std,
            size=(noise_steps.size, action_indices.size),
        ).astype(actions.dtype, copy=False)
        actions[np.ix_(noise_steps, action_indices)] += noise

    def _resolve_action_noise_steps(
        self, episode: ReplayEpisode, replay_steps: int
    ) -> np.ndarray:
        if replay_steps <= 0:
            return np.asarray([], dtype=np.int64)
        if self.action_noise_stage_indices is None:
            return np.arange(replay_steps, dtype=np.int64)

        intervals = self._load_stage_intervals(episode)
        selected_steps: list[int] = []
        for stage_idx in self.action_noise_stage_indices:
            idx = int(stage_idx)
            if idx < 0:
                idx = len(intervals) + idx
            if idx < 0 or idx >= len(intervals):
                raise ValueError(
                    f"action_noise_stage_indices contains {stage_idx}, but "
                    f"{episode.annotation_path} has {len(intervals)} stages."
                )
            start, end = intervals[idx]
            start = max(int(start), 0)
            end = min(int(end), replay_steps)
            if end > start:
                selected_steps.extend(range(start, end))
        return np.asarray(sorted(set(selected_steps)), dtype=np.int64)

    def _load_stage_intervals(self, episode: ReplayEpisode) -> list[tuple[int, int]]:
        if episode.annotation_path is None:
            raise ValueError(
                "action_noise_stage_indices was set but no annotation file exists "
                f"for episode {episode.episode_index}."
            )
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not skills:
            raise ValueError(
                f"No skill_annotation entries in {episode.annotation_path}."
            )

        intervals = []
        for skill in skills:
            duration = skill.get("frame_duration")
            if not isinstance(duration, list) or len(duration) != 2:
                raise ValueError(
                    f"Invalid frame_duration in {episode.annotation_path}."
                )
            intervals.append((int(duration[0]), int(duration[1])))
        return intervals

    def _load_actions(self, episode: ReplayEpisode) -> np.ndarray:
        cached = self._action_cache.get(episode.episode_index)
        if cached is not None:
            return cached

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "env.replay_init requires pyarrow to read BEHAVIOR parquet files."
            ) from exc

        table = pq.read_table(episode.parquet_path, columns=[self.action_column])
        values = table[self.action_column].to_pylist()
        actions = np.asarray(values, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Replay action column {self.action_column!r} in {episode.parquet_path} "
                f"must be 2-D after loading, got shape {actions.shape}."
            )
        self._action_cache[episode.episode_index] = actions
        return actions

    def _resolve_target_step(self, episode: ReplayEpisode, action_count: int) -> int:
        if self.target_step is not None:
            return min(max(self.target_step, 0), action_count)
        if self.stage_index is None:
            return action_count
        return min(max(self._load_stage_step(episode), 0), action_count)

    def _load_stage_step(self, episode: ReplayEpisode) -> int:
        if episode.annotation_path is None:
            raise ValueError(
                f"stage_index was set but no annotation file exists for episode {episode.episode_index}."
            )
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not skills:
            raise ValueError(
                f"No skill_annotation entries in {episode.annotation_path}."
            )
        idx = self.stage_index
        if idx < 0:
            idx = len(skills) + idx
        if idx < 0 or idx >= len(skills):
            raise ValueError(
                f"stage_index={self.stage_index} is out of range for "
                f"{episode.annotation_path} with {len(skills)} stages."
            )
        duration = skills[idx].get("frame_duration")
        if not isinstance(duration, list) or len(duration) != 2:
            raise ValueError(
                f"Invalid frame_duration for stage {self.stage_index} in {episode.annotation_path}."
            )
        return int(duration[0] if self.stage_boundary == "start" else duration[1])

    @staticmethod
    def _load_stage_prompts(episode: ReplayEpisode) -> tuple[str, ...]:
        if episode.orchestrator_path is None:
            return ()
        with episode.orchestrator_path.open("r", encoding="utf-8") as f:
            orchestrator = json.load(f)
        prompts = orchestrator.get("cot_subtask_description_list", [])
        if not isinstance(prompts, list):
            return ()
        return tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())

    def _episode_index_to_instance_id(self, episode_index: int) -> int:
        task_offset = self.task_id * 10000
        within_task_index = episode_index - task_offset
        if within_task_index <= 0:
            raise ValueError(
                f"episode_index={episode_index} does not belong to task_id={self.task_id}."
            )
        if within_task_index % 10 == 0:
            return within_task_index // 10
        return within_task_index

    @staticmethod
    def _parse_episode_index(stem: str) -> int:
        prefix = "episode_"
        if not stem.startswith(prefix):
            raise ValueError(f"Invalid BEHAVIOR episode filename stem: {stem}")
        return int(stem[len(prefix) :])


def maybe_make_replay_initializer(
    cfg: DictConfig, seed_offset: int = 0
) -> BehaviorReplayInitializer | None:
    initializer = BehaviorReplayInitializer(cfg, seed_offset=seed_offset)
    return initializer if initializer.enabled else None


def replay_plans_to_infos(plans: list[ReplayPlan]) -> list[dict[str, Any]]:
    return [
        {
            "replay_episode_index": plan.episode_index,
            "replay_instance_id": plan.instance_id,
            "replay_steps": plan.replay_steps,
            "replay_target_step": plan.target_step,
            "replay_stage_prompts": list(plan.stage_prompts),
        }
        for plan in plans
    ]
