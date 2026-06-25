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

"""RoboCasa365 environment wrapper for RLinf."""

import copy
import re
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.robocasa.venv import RobocasaSubprocEnv
from rlinf.envs.robocasa365.eval_schedule import (
    resolve_robocasa365_episode_horizons,
    validate_robocasa365_eval_horizons,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

_LEGACY_TASK_DESC_MAP = {
    "OpenSingleDoor": "open cabinet or microwave door",
    "CloseSingleDoor": "close cabinet or microwave door",
    "OpenDoubleDoor": "open double cabinet doors",
    "CloseDoubleDoor": "close double cabinet doors",
    "OpenDrawer": "open drawer",
    "CloseDrawer": "close drawer",
    "PnPCounterToCab": "pick and place from counter to cabinet",
    "PnPCabToCounter": "pick and place from cabinet to counter",
    "PnPCounterToSink": "pick and place from counter to sink",
    "PnPSinkToCounter": "pick and place from sink to counter",
    "PnPCounterToStove": "pick and place from counter to stove",
    "PnPStoveToCounter": "pick and place from stove to counter",
    "PnPCounterToMicrowave": "pick and place from counter to microwave",
    "PnPMicrowaveToCounter": "pick and place from microwave to counter",
    "TurnOnMicrowave": "turn on microwave",
    "TurnOffMicrowave": "turn off microwave",
    "TurnOnSinkFaucet": "turn on sink faucet",
    "TurnOffSinkFaucet": "turn off sink faucet",
    "TurnSinkSpout": "turn sink spout",
    "TurnOnStove": "turn on stove",
    "TurnOffStove": "turn off stove",
    "CoffeeSetupMug": "setup mug for coffee",
    "CoffeeServeMug": "serve coffee into mug",
    "CoffeePressButton": "press coffee machine button",
}

_OFFICIAL_PROMPT_INFO_KEY = "ep_meta"
_OFFICIAL_PROMPT_LANG_KEY = "lang"


def _cfg_to_python(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _split_camel_case(name: str) -> str:
    name = name.split("/", 1)[-1]
    name = name.removeprefix("Kitchen")
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)
    return " ".join(token.lower() for token in tokens) if tokens else name.lower()


def _official_prompt_from_info(info_single: dict[str, Any]) -> str:
    value = info_single[_OFFICIAL_PROMPT_INFO_KEY][_OFFICIAL_PROMPT_LANG_KEY]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar_value = value.item()
        if isinstance(scalar_value, str):
            return scalar_value.strip()
    raise TypeError(
        "RoboCasa365 official prompt info "
        f"'{_OFFICIAL_PROMPT_INFO_KEY}.{_OFFICIAL_PROMPT_LANG_KEY}' must contain "
        f"a string, got {type(value).__name__}."
    )


def _normalize_task_filter(task_filter: Any) -> dict[str, list[str]]:
    if task_filter is None:
        return {"include": [], "exclude": []}
    if isinstance(task_filter, dict):
        include = _ensure_list(task_filter.get("include"))
        exclude = _ensure_list(task_filter.get("exclude"))
    else:
        include = _ensure_list(task_filter)
        exclude = []
    return {
        "include": [str(pattern) for pattern in include if pattern],
        "exclude": [str(pattern) for pattern in exclude if pattern],
    }


def _pattern_matches(pattern: str, text: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, flags=re.IGNORECASE) is not None
    return pattern.lower() in text.lower()


def _task_matches_filter(
    task_spec: dict[str, Any], task_filter: dict[str, list[str]]
) -> bool:
    haystack = " | ".join(
        str(
            task_spec.get(key, "")
            if key != "metadata"
            else task_spec.get("metadata_view", {})
        )
        for key in ("task_name", "env_name", "task_description", "benchmark_selection")
    )

    include = task_filter["include"]
    if include and not any(_pattern_matches(pattern, haystack) for pattern in include):
        return False

    exclude = task_filter["exclude"]
    if exclude and any(_pattern_matches(pattern, haystack) for pattern in exclude):
        return False

    return True


def _guess_task_mode(
    task_name: str, task_soup: Optional[str], metadata: dict[str, Any]
) -> Optional[str]:
    mode = metadata.get("task_mode") or metadata.get("mode")
    if isinstance(mode, str) and mode:
        return mode

    if task_soup:
        lowered = task_soup.lower()
        if "atomic" in lowered:
            return "atomic"
        if "composite" in lowered:
            return "composite"

    lowered_name = task_name.lower()
    if "atomic" in lowered_name:
        return "atomic"
    if "composite" in lowered_name:
        return "composite"
    return None


def _build_benchmark_selection(
    task_source: str,
    split: Optional[str],
    task_soups: list[str],
    dataset_source: Optional[str],
) -> str:
    if task_source == "legacy":
        return "legacy"

    parts = [part for part in [dataset_source, split] if part]
    if task_soups:
        parts.append("+".join(task_soups))
    return "/".join(parts) if parts else "registry"


def _get_task_horizon(task_name: str, fallback_horizon: int) -> int:
    try:
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        return int(get_task_horizon(task_name))
    except Exception:
        return fallback_horizon


def load_robocasa365_task_specs(cfg: Any) -> list[dict[str, Any]]:
    """Load and filter RoboCasa365 task specs from an environment config."""

    task_source = str(cfg.get("task_source", "dataset_registry"))
    if task_source != "dataset_registry":
        raise ValueError(
            "RoboCasa365 task loading currently requires "
            f"task_source='dataset_registry', got {task_source!r}."
        )

    dataset_source = cfg.get("dataset_source", None)
    if dataset_source is not None:
        dataset_source = str(dataset_source)
    split = cfg.get("split", None)
    if split is not None:
        split = str(split)
    task_soups = [
        str(soup) for soup in _ensure_list(_cfg_to_python(cfg.get("task_soup", None)))
    ]
    task_mode = cfg.get("task_mode", None)
    if task_mode is not None:
        task_mode = str(task_mode)
    task_filter = _normalize_task_filter(_cfg_to_python(cfg.get("task_filter", None)))
    benchmark_selection = cfg.get(
        "benchmark_selection",
        _build_benchmark_selection(
            task_source=task_source,
            split=split,
            task_soups=task_soups,
            dataset_source=dataset_source,
        ),
    )

    try:
        from robocasa.utils.dataset_registry_utils import get_ds_meta, get_ds_soup
    except ImportError as exc:
        raise ImportError(
            "RoboCasa365 benchmark selection requires "
            "robocasa.utils.dataset_registry_utils. Install a RoboCasa version "
            "that includes the benchmark dataset registry."
        ) from exc

    task_names: list[str] = []
    if task_soups:
        if split is None:
            raise ValueError(
                "split must be provided when task_soup is set for RoboCasa365."
            )
        for task_soup in task_soups:
            entries = get_ds_soup(
                task_set=task_soup,
                split=split,
                source=dataset_source or "human",
            )
            task_names.extend(str(task_entry["task"]) for task_entry in entries)
    else:
        task_names = [
            str(task)
            for task in _ensure_list(_cfg_to_python(cfg.get("task_names", None)))
        ]

    fallback_horizon = int(cfg.get("max_episode_steps", 300))
    task_specs = []
    for task_name in _dedupe_preserve_order(task_names):
        try:
            metadata = (
                get_ds_meta(
                    task_name,
                    split=split,
                    source=dataset_source or "human",
                )
                or {}
            )
        except Exception:
            metadata = {}

        task_label = task_name.split("/", 1)[-1]
        task_soup = metadata.get("task_soup")
        if not task_soup and task_soups:
            task_soup = task_soups[0] if len(task_soups) == 1 else "+".join(task_soups)
        selected_task_mode = _guess_task_mode(task_label, task_soup, metadata)
        metadata_view = {
            "task_name": task_label,
            "env_name": f"robocasa/{task_label}",
            "task_source": "dataset_registry",
            "dataset_source": dataset_source or "human",
            "split": split,
            "task_soup": task_soup,
            "benchmark_selection": benchmark_selection,
            "task_mode": selected_task_mode,
        }
        task_spec = {
            "task_name": task_label,
            "env_name": f"robocasa/{task_label}",
            "task_description": _split_camel_case(task_label),
            "horizon": _get_task_horizon(task_label, fallback_horizon),
            "metadata_view": {
                key: value for key, value in metadata_view.items() if value is not None
            },
            "task_mode": selected_task_mode,
            "benchmark_selection": benchmark_selection,
        }
        if task_mode and selected_task_mode and selected_task_mode != task_mode:
            continue
        if not _task_matches_filter(task_spec, task_filter):
            continue
        task_specs.append(task_spec)

    if not task_specs:
        raise ValueError(
            "No RoboCasa365 tasks were selected. "
            "Check split/task_soup/task_filter/task_mode."
        )
    return task_specs


class Robocasa365Env(gym.Env):
    """Vectorized RLinf wrapper for the RoboCasa365 benchmark.

    This wrapper keeps the legacy ``robocasa`` integration untouched and exposes
    a separate benchmark-native path that selects tasks via RoboCasa's official
    dataset registry. It supports split- and task-soup-based selection, prompt
    extraction from task metadata, and configurable observation / action
    adapters for mobile manipulation recipes.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        """Initialize the RoboCasa365 wrapper.

        Args:
            cfg: Environment configuration.
            num_envs: Number of vectorized environments on this worker.
            seed_offset: Worker-local seed offset.
            total_num_processes: Total number of env processes across all workers.
            worker_info: Distributed worker metadata.

        Raises:
            ValueError: If ``group_size`` is invalid for the requested
                ``num_envs``.
        """
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        if self.group_size <= 0:
            raise ValueError(
                f"RoboCasa365 group_size must be positive, got {self.group_size}."
            )
        if self.num_envs % self.group_size != 0:
            raise ValueError(
                "RoboCasa365 requires num_envs to be divisible by group_size, "
                f"got num_envs={self.num_envs} and group_size={self.group_size}."
            )
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", False)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.seed_strategy = str(cfg.get("seed_strategy", "worker_offset"))
        self._generator = np.random.default_rng(seed=self.seed)

        self.task_source = str(cfg.get("task_source", "dataset_registry"))
        self.dataset_source = cfg.get("dataset_source", None)
        if self.dataset_source is not None:
            self.dataset_source = str(self.dataset_source)
        self.split = cfg.get("split", None)
        if self.split is not None:
            self.split = str(self.split)
        self.task_soups = [
            str(soup)
            for soup in _ensure_list(_cfg_to_python(cfg.get("task_soup", None)))
        ]
        self.task_mode = cfg.get("task_mode", None)
        if self.task_mode is not None:
            self.task_mode = str(self.task_mode)
        self.task_filter = _normalize_task_filter(
            _cfg_to_python(cfg.get("task_filter", None))
        )
        self.observation_cfg = _cfg_to_python(cfg.get("observation", {})) or {}
        self.action_space_cfg = _cfg_to_python(cfg.get("action_space", {})) or {}
        self.task_sampling_strategy = str(
            cfg.get(
                "task_sampling_strategy",
                "ordered" if bool(cfg.get("is_eval", False)) else "random",
            )
        )
        self.rotate_tasks_on_auto_reset = bool(
            cfg.get("rotate_tasks_on_auto_reset", True)
        )
        self.benchmark_selection = cfg.get(
            "benchmark_selection",
            _build_benchmark_selection(
                task_source=self.task_source,
                split=self.split,
                task_soups=self.task_soups,
                dataset_source=self.dataset_source,
            ),
        )
        self.task_specs = self._load_task_specs()
        self.num_tasks = len(self.task_specs)
        self.episode_horizon_source = str(
            self.cfg.get("episode_horizon_source", "max_episode_steps")
        )
        selected_episode_horizons = resolve_robocasa365_episode_horizons(
            task_horizons=(task_spec["horizon"] for task_spec in self.task_specs),
            max_episode_steps=int(self.cfg.get("max_episode_steps", 300)),
            episode_horizon_source=self.episode_horizon_source,
        )
        if bool(self.cfg.get("is_eval", False)):
            validate_robocasa365_eval_horizons(
                episode_horizons=selected_episode_horizons,
                max_steps_per_rollout_epoch=int(
                    self.cfg.get("max_steps_per_rollout_epoch", 300)
                ),
            )
        self._ordered_task_cursor = (
            (self.seed_offset * self.num_group) % self.num_tasks
            if self.num_tasks
            else 0
        )
        self._init_reset_state_ids()
        self.update_reset_state_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_raw_obs = None
        self.current_info_list = None

        self.video_cfg = cfg.video_cfg

    def _load_task_specs(self) -> list[dict[str, Any]]:
        return load_robocasa365_task_specs(self.cfg)

    def _task_mode_matches(self, task_spec: dict[str, Any]) -> bool:
        if not self.task_mode:
            return True
        if not task_spec.get("task_mode"):
            return True
        return task_spec["task_mode"] == self.task_mode

    def _init_reset_state_ids(self):
        cfg_seed = int(self.cfg.seed)
        if self.seed_strategy in ("worker_offset", "legacy"):
            base_seed = self.seed
            self.env_seeds = [base_seed + i for i in range(self.num_envs)]
        elif self.seed_strategy in ("same", "openpi"):
            base_seed = cfg_seed
            self.env_seeds = [base_seed for _ in range(self.num_envs)]
        elif self.seed_strategy == "global_unique":
            base_seed = cfg_seed
            global_env_start = self.seed_offset * self.num_envs
            self.env_seeds = [
                base_seed + global_env_start + i for i in range(self.num_envs)
            ]
        else:
            raise ValueError(
                "RoboCasa365 seed_strategy must be one of "
                "{worker_offset, legacy, same, openpi, global_unique}, "
                f"got {self.seed_strategy!r}."
            )

    def update_reset_state_ids(self):
        self._set_next_task_ids()

    def _sample_task_ids(self, num_groups: int) -> np.ndarray:
        use_ordered = (
            self.task_sampling_strategy == "ordered"
            or bool(self.cfg.get("is_eval", False))
            or bool(self.cfg.get("use_ordered_reset_state_ids", False))
        )
        if use_ordered:
            group_task_ids = (
                np.arange(
                    self._ordered_task_cursor,
                    self._ordered_task_cursor + num_groups,
                )
                % self.num_tasks
            )
            self._ordered_task_cursor = (
                self._ordered_task_cursor + num_groups * self.total_num_processes
            ) % self.num_tasks
            return group_task_ids.astype(np.int32, copy=False)

        return self._generator.integers(
            low=0,
            high=self.num_tasks,
            size=num_groups,
            dtype=np.int32,
        )

    def _set_next_task_ids(
        self, env_idx: Optional[Union[int, list[int], np.ndarray]] = None
    ) -> np.ndarray:
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        elif isinstance(env_idx, int):
            env_idx = np.asarray([env_idx], dtype=np.int32)
        else:
            env_idx = np.asarray(env_idx, dtype=np.int32)

        if env_idx.size == 0:
            return env_idx

        if self.use_fixed_reset_state_ids and hasattr(self, "task_ids"):
            return np.empty(0, dtype=np.int32)

        group_ids = np.unique(env_idx // self.group_size)
        group_task_ids = self._sample_task_ids(len(group_ids))
        affected_env_ids = []
        new_task_ids = getattr(
            self,
            "task_ids",
            np.zeros(self.num_envs, dtype=np.int32),
        ).copy()

        for group_id, task_id in zip(group_ids, group_task_ids):
            group_start = int(group_id) * self.group_size
            group_end = min(group_start + self.group_size, self.num_envs)
            group_env_ids = np.arange(group_start, group_end, dtype=np.int32)
            new_task_ids[group_env_ids] = int(task_id)
            affected_env_ids.extend(group_env_ids.tolist())

        affected_env_ids = np.asarray(affected_env_ids, dtype=np.int32)
        old_task_ids = getattr(self, "task_ids", None)
        self.task_ids = new_task_ids
        self._refresh_task_context()

        if old_task_ids is not None and hasattr(self, "env"):
            changed_env_ids = affected_env_ids[
                old_task_ids[affected_env_ids] != self.task_ids[affected_env_ids]
            ]
            if changed_env_ids.size:
                env_fns = self.get_env_fns(env_idx=changed_env_ids)
                self.env.reconfigure_env_fns(env_fns, changed_env_ids)

        return affected_env_ids

    def _init_env(self):
        self._refresh_task_context()

        env_fns = self.get_env_fns()
        self.env = RobocasaSubprocEnv(env_fns)

    def _refresh_task_context(self):
        self.task_descriptions = [
            self.task_specs[task_id]["task_description"] for task_id in self.task_ids
        ]
        self.task_metadata = [
            copy.deepcopy(self.task_specs[task_id]["metadata_view"])
            for task_id in self.task_ids
        ]
        fallback_horizon = int(self.cfg.get("max_episode_steps", 300))
        registry_horizons = [
            int(self.task_specs[task_id].get("horizon", fallback_horizon))
            for task_id in self.task_ids
        ]
        self.task_horizons = np.asarray(
            resolve_robocasa365_episode_horizons(
                task_horizons=registry_horizons,
                max_episode_steps=fallback_horizon,
                episode_horizon_source=self.episode_horizon_source,
            ),
            dtype=np.int32,
        )

    def _get_camera_names(self) -> list[str]:
        camera_names = _ensure_list(_cfg_to_python(self.cfg.camera_names))
        return [str(camera_name) for camera_name in camera_names]

    def get_env_fns(self, env_idx: Optional[Union[list[int], np.ndarray]] = None):
        env_fns = []

        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        else:
            env_idx = np.asarray(env_idx, dtype=np.int32)

        camera_names = self._get_camera_names()
        camera_widths = self.cfg.init_params.camera_widths
        camera_heights = self.cfg.init_params.camera_heights
        render_camera = self.cfg.get("render_camera", None) or (
            camera_names[0] if camera_names else None
        )
        robot_name = self.cfg.robot_name
        env_split = self.split
        has_renderer = bool(self.cfg.get("has_renderer", False))
        env_kwargs = {
            "camera_depths": bool(self.cfg.get("camera_depths", False)),
            "translucent_robot": bool(self.cfg.get("translucent_robot", False)),
        }
        if self.cfg.get("generative_textures", None) is not None:
            env_kwargs["generative_textures"] = self.cfg.generative_textures
        if self.cfg.get("randomize_cameras", None) is not None:
            env_kwargs["randomize_cameras"] = bool(self.cfg.randomize_cameras)

        for env_id in env_idx:
            env_id = int(env_id)
            task_spec = self.task_specs[self.task_ids[env_id]]
            env_seed = self.env_seeds[env_id]

            def env_fn(
                spec=task_spec,
                seed=env_seed,
                cameras=camera_names,
                width=camera_widths,
                height=camera_heights,
                robot=robot_name,
                render_camera_name=render_camera,
                render_onscreen=has_renderer,
                extra_env_kwargs=env_kwargs,
                split_name=env_split,
            ):
                import robocasa  # noqa: F401 Robocasa must be imported to register envs
                from robocasa.utils.env_utils import create_env

                common_kwargs = {
                    "env_name": spec["task_name"],
                    "robots": robot,
                    "camera_names": cameras,
                    "camera_widths": width,
                    "camera_heights": height,
                    "seed": seed,
                    "render_onscreen": render_onscreen,
                    "split": split_name,
                    **extra_env_kwargs,
                }
                if render_camera_name:
                    common_kwargs["render_camera"] = render_camera_name

                env = create_env(**common_kwargs)

                return env

            env_fns.append(env_fn)

        return env_fns

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_info["episode_len"], 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _get_obs_key(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.observation_cfg.get(key, default)

    def _extract_state_vector(self, obs_single: dict[str, Any]) -> np.ndarray:
        state_layout = self.observation_cfg.get("state_layout", [])
        key_map = self.observation_cfg.get("state_key_map", {}) or {}

        state_components: list[np.ndarray] = []
        for component in state_layout:
            if component.startswith("zeros:"):
                zeros_dim = int(component.split(":", 1)[1])
                state_components.append(np.zeros(zeros_dim, dtype=np.float32))
                continue

            obs_key = key_map.get(component, component)
            state_components.append(
                np.asarray(obs_single[obs_key], dtype=np.float32).reshape(-1)
            )

        if not state_components:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(state_components).astype(np.float32, copy=False)

    def _extract_image_and_state(self, obs, info_list):
        base_images = []
        wrist_images = []
        extra_view_images = []
        states = []
        task_descriptions = []

        main_camera_key = self._get_obs_key(
            "main_camera_key", "robot0_agentview_left_image"
        )
        wrist_camera_key = self._get_obs_key(
            "wrist_camera_key", "robot0_eye_in_hand_image"
        )
        extra_camera_keys = [
            str(key) for key in _ensure_list(self._get_obs_key("extra_camera_keys", []))
        ]
        flip_images_vertical = bool(
            self.observation_cfg.get("flip_images_vertical", True)
        )

        for env_id in range(len(obs)):
            obs_single = obs[env_id]
            base_img = obs_single.get(main_camera_key)
            if base_img is None:
                raise KeyError(
                    f"RoboCasa365 observation key '{main_camera_key}' was not found. "
                    "Update env.observation.main_camera_key to match your RoboCasa camera config."
                )
            base_img = np.asarray(base_img)
            if flip_images_vertical:
                base_img = base_img[::-1]
            base_images.append(base_img)

            if wrist_camera_key:
                wrist_img = obs_single.get(wrist_camera_key)
                if wrist_img is None:
                    raise KeyError(
                        f"RoboCasa365 observation key '{wrist_camera_key}' was not found. "
                        "Update env.observation.wrist_camera_key to match your RoboCasa camera config."
                    )
                wrist_img = np.asarray(wrist_img)
                if flip_images_vertical:
                    wrist_img = wrist_img[::-1]
                wrist_images.append(wrist_img)

            if extra_camera_keys:
                env_extra_views = []
                for camera_key in extra_camera_keys:
                    extra_img = obs_single.get(camera_key)
                    if extra_img is None:
                        raise KeyError(
                            f"RoboCasa365 observation key '{camera_key}' was not found. "
                            "Update env.observation.extra_camera_keys to match your RoboCasa camera config."
                        )
                    extra_img = np.asarray(extra_img)
                    if flip_images_vertical:
                        extra_img = extra_img[::-1]
                    env_extra_views.append(extra_img)
                extra_view_images.append(env_extra_views)

            states.append(self._extract_state_vector(obs_single))
            task_descriptions.append(_official_prompt_from_info(info_list[env_id]))

        return {
            "base_image": np.asarray(base_images),
            "wrist_image": np.asarray(wrist_images) if wrist_images else None,
            "extra_view_images": (
                np.asarray(extra_view_images) if extra_view_images else None
            ),
            "state": np.asarray(states),
            "task_descriptions": task_descriptions,
        }

    def _wrap_obs(self, obs_list, info_list):
        extracted = self._extract_image_and_state(obs_list, info_list)
        self._refresh_task_context()
        for env_id, task_description in enumerate(extracted["task_descriptions"]):
            self.task_descriptions[env_id] = task_description
            self.task_metadata[env_id]["task_description"] = task_description

        obs = {
            "main_images": torch.from_numpy(extracted["base_image"]),
            "wrist_images": (
                torch.from_numpy(extracted["wrist_image"])
                if extracted["wrist_image"] is not None
                else None
            ),
            "states": torch.from_numpy(extracted["state"]),
            "task_descriptions": list(self.task_descriptions),
            "task_metadata": copy.deepcopy(self.task_metadata),
        }
        if extracted["extra_view_images"] is not None:
            obs["extra_view_images"] = torch.from_numpy(extracted["extra_view_images"])
        return obs

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all or selected environments.

        Args:
            env_idx: Optional indices to reset. ``None`` resets all envs.
            options: Optional reset options. Reserved for future use.

        Returns:
            A tuple of batched RLinf observations and an info dictionary.
        """
        del options
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            self._is_start = False

        if isinstance(env_idx, int):
            env_idx = [env_idx]

        raw_obs, info_list = self.env.reset(id=env_idx)

        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        if self.current_info_list is None:
            self.current_info_list = [None] * self.num_envs
        for raw_obs_id, target_env_id in enumerate(env_idx):
            self.current_raw_obs[int(target_env_id)] = raw_obs[raw_obs_id]
            self.current_info_list[int(target_env_id)] = info_list[raw_obs_id]

        obs = self._wrap_obs(self.current_raw_obs, self.current_info_list)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(
        self,
        actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Step the vectorized RoboCasa365 environments once.

        Args:
            actions: Batched actions for every environment.
            auto_reset: Whether to auto-reset completed environments.

        Returns:
            A tuple of observations, rewards, terminations, truncations, and
            info dictionaries in RLinf format.
        """
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            rewards = np.zeros(self.num_envs, dtype=np.float32)

            return (
                obs,
                to_tensor(rewards),
                to_tensor(terminations),
                to_tensor(truncations),
                infos,
            )

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1

        raw_obs, rewards, dones, info_lists = self.env.step(actions)
        del rewards, dones
        self.current_raw_obs = raw_obs
        self.current_info_list = info_lists

        terminations = np.array(
            [info.get("success", False) for info in info_lists]
        ).astype(bool)
        truncations = self._elapsed_steps >= self.task_horizons
        obs = self._wrap_obs(raw_obs, info_lists)

        step_reward = self._calc_step_reward(terminations)

        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        done_mask = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if done_mask.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(done_mask, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(
        self, chunk_actions: np.ndarray
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Step a chunk of actions through the vectorized environments.

        Args:
            chunk_actions: Batched chunk actions with shape
                ``[num_envs, chunk_size, action_dim]``.

        Returns:
            Per-step observations, rewards, terminations, truncations, and infos.
        """
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if (
            self.auto_reset
            or self.ignore_terminations
            or bool(self.cfg.get("is_eval", False))
        ):
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(
        self, dones: np.ndarray, _final_obs: dict[str, Any], infos: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.rotate_tasks_on_auto_reset and self.group_size == 1:
            self._set_next_task_ids(env_idx=env_idx)
        obs, infos = self.reset(env_idx=env_idx)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations: np.ndarray) -> np.ndarray:
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        return reward

    def close(self) -> None:
        """Close the vectorized RoboCasa365 environments."""
        if hasattr(self, "env"):
            self.env.close()
