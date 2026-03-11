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

"""RoboCasa365 environment wrapper for RLinf."""

import copy
import re
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401 Robocasa must be imported to register envs
import torch
from omegaconf import OmegaConf

from rlinf.envs.robocasa.venv import RobocasaSubprocEnv
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

_PROMPT_CANDIDATE_KEYS = (
    "prompt",
    "language",
    "lang",
    "instruction",
    "task_description",
    "description",
    "task",
)
_NESTED_PROMPT_KEYS = ("ep_meta", "metadata", "task_metadata")


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


def _prompt_from_metadata(metadata: Any) -> Optional[str]:
    if isinstance(metadata, str):
        prompt = metadata.strip()
        return prompt or None
    if not isinstance(metadata, dict):
        return None

    for key in _PROMPT_CANDIDATE_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in _NESTED_PROMPT_KEYS:
        nested_prompt = _prompt_from_metadata(metadata.get(key))
        if nested_prompt:
            return nested_prompt

    return None


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

        self._init_reset_state_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg

    def _load_task_specs(self) -> list[dict[str, Any]]:
        task_specs = self._load_dataset_registry_task_specs()
        task_specs = [
            task_spec
            for task_spec in task_specs
            if self._task_mode_matches(task_spec)
            and _task_matches_filter(task_spec, self.task_filter)
        ]

        if not task_specs:
            raise ValueError(
                "No RoboCasa365 tasks were selected. Check split/task_soup/task_filter/task_mode."
            )
        return task_specs

    def _load_dataset_registry_task_specs(self) -> list[dict[str, Any]]:
        try:
            from robocasa.utils.dataset_registry import get_ds_meta, get_ds_soup
        except ImportError as exc:
            raise ImportError(
                "RoboCasa365 benchmark selection requires robocasa.utils.dataset_registry. "
                "Install a RoboCasa version that includes the benchmark dataset registry."
            ) from exc

        task_names: list[str] = []
        if self.task_soups:
            if self.split is None:
                raise ValueError(
                    "split must be provided when task_soup is set for RoboCasa365."
                )
            for task_soup in self.task_soups:
                task_names.extend(
                    list(
                        get_ds_soup(
                            task_soup=task_soup,
                            split=self.split,
                            source=self.dataset_source or "human",
                        )
                    )
                )
        else:
            task_names = [
                str(task)
                for task in _ensure_list(_cfg_to_python(self.cfg.get("task_names", None)))
            ]

        task_specs = []
        for task_name in _dedupe_preserve_order(task_names):
            metadata = {}
            try:
                metadata = get_ds_meta(task_name, source=self.dataset_source or "human") or {}
            except TypeError:
                metadata = get_ds_meta(task_name) or {}
            except Exception:
                metadata = {}

            task_label = task_name.split("/", 1)[-1]
            prompt = _prompt_from_metadata(metadata)
            if prompt is None:
                prompt = _LEGACY_TASK_DESC_MAP.get(
                    task_label.removeprefix("Kitchen"),
                    _split_camel_case(task_label),
                )

            task_soup = metadata.get("task_soup")
            if not task_soup and self.task_soups:
                task_soup = (
                    self.task_soups[0]
                    if len(self.task_soups) == 1
                    else "+".join(self.task_soups)
                )

            task_mode = _guess_task_mode(task_label, task_soup, metadata)
            metadata_view = {
                "task_name": task_label,
                "env_name": f"robocasa/{task_label}",
                "task_source": "dataset_registry",
                "dataset_source": self.dataset_source or "human",
                "split": self.split,
                "task_soup": task_soup,
                "benchmark_selection": self.benchmark_selection,
                "task_mode": task_mode,
            }
            task_specs.append(
                {
                    "task_name": task_label,
                    "env_name": f"robocasa/{task_label}",
                    "task_description": prompt,
                    "metadata_view": {
                        k: v for k, v in metadata_view.items() if v is not None
                    },
                    "task_mode": task_mode,
                    "benchmark_selection": self.benchmark_selection,
                }
            )

        return task_specs

    def _task_mode_matches(self, task_spec: dict[str, Any]) -> bool:
        if not self.task_mode:
            return True
        if not task_spec.get("task_mode"):
            return True
        return task_spec["task_mode"] == self.task_mode

    def _init_reset_state_ids(self):
        base_seed = self.seed
        self.env_seeds = [base_seed + i for i in range(self.num_envs)]

    def update_reset_state_ids(self):
        pass

    def _init_env(self):
        self.task_ids = np.array([env_id % self.num_tasks for env_id in range(self.num_envs)])
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

    def _get_camera_names(self) -> list[str]:
        camera_names = _ensure_list(_cfg_to_python(self.cfg.camera_names))
        return [str(camera_name) for camera_name in camera_names]

    def get_env_fns(self):
        env_fns = []

        camera_names = self._get_camera_names()
        camera_widths = self.cfg.init_params.camera_widths
        camera_heights = self.cfg.init_params.camera_heights
        render_camera = self.cfg.get("render_camera", None) or (
            camera_names[0] if camera_names else None
        )
        robot_name = self.cfg.robot_name
        env_split = self.split

        for env_id in range(self.num_envs):
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
                split_name=env_split,
            ):
                from robosuite.controllers import load_composite_controller_config

                controller_config = load_composite_controller_config(
                    controller=None,
                    robot=robot,
                )

                common_kwargs = {
                    "robots": robot,
                    "controller_configs": controller_config,
                    "camera_names": cameras,
                    "camera_widths": width,
                    "camera_heights": height,
                    "has_renderer": False,
                    "has_offscreen_renderer": True,
                    "ignore_done": True,
                    "use_object_obs": True,
                    "use_camera_obs": True,
                    "camera_depths": False,
                    "seed": seed,
                    "translucent_robot": False,
                }
                if render_camera_name:
                    common_kwargs["render_camera"] = render_camera_name
                if split_name is not None:
                    common_kwargs["split"] = split_name

                return gym.make(spec["env_name"], **common_kwargs)

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
        state_layout = self.observation_cfg.get(
            "state_layout",
            [
                "base_xy",
                "base_pad",
                "base_to_eef_quat",
                "base_to_eef_pos",
                "gripper_qvel",
                "gripper_qpos",
            ],
        )
        key_map = self.observation_cfg.get(
            "state_key_map",
            {
                "base_pos": "robot0_base_pos",
                "base_to_eef_quat": "robot0_base_to_eef_quat",
                "base_to_eef_pos": "robot0_base_to_eef_pos",
                "gripper_qvel": "robot0_gripper_qvel",
                "gripper_qpos": "robot0_gripper_qpos",
            },
        )

        state_components: list[np.ndarray] = []
        for component in state_layout:
            if component == "base_xy":
                base_pos = np.asarray(obs_single[key_map["base_pos"]], dtype=np.float32)
                state_components.append(base_pos[:2])
            elif component == "base_pad":
                state_components.append(np.zeros(3, dtype=np.float32))
            elif component == "base_to_eef_quat":
                state_components.append(
                    np.asarray(obs_single[key_map["base_to_eef_quat"]], dtype=np.float32)
                )
            elif component == "base_to_eef_pos":
                state_components.append(
                    np.asarray(obs_single[key_map["base_to_eef_pos"]], dtype=np.float32)
                )
            elif component == "gripper_qvel":
                state_components.append(
                    np.asarray(obs_single[key_map["gripper_qvel"]], dtype=np.float32)
                )
            elif component == "gripper_qpos":
                state_components.append(
                    np.asarray(obs_single[key_map["gripper_qpos"]], dtype=np.float32)
                )
            elif component.startswith("zeros:"):
                zeros_dim = int(component.split(":", 1)[1])
                state_components.append(np.zeros(zeros_dim, dtype=np.float32))
            else:
                obs_key = key_map.get(component, component)
                state_components.append(np.asarray(obs_single[obs_key], dtype=np.float32))

        if not state_components:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(state_components, dtype=np.float32)

    def _extract_image_and_state(self, obs):
        base_images = []
        wrist_images = []
        extra_view_images = []
        states = []

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

        return {
            "base_image": np.asarray(base_images),
            "wrist_image": np.asarray(wrist_images) if wrist_images else None,
            "extra_view_images": (
                np.asarray(extra_view_images) if extra_view_images else None
            ),
            "state": np.asarray(states),
        }

    def _wrap_obs(self, obs_list):
        extracted = self._extract_image_and_state(obs_list)
        self._refresh_task_context()

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
            obs["extra_view_images"] = torch.from_numpy(
                extracted["extra_view_images"]
            )
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

        if isinstance(env_idx, int):
            env_idx = [env_idx]

        raw_obs = self.env.reset(id=env_idx)

        obs = self._wrap_obs(raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(
        self,
        actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        auto_reset: bool = True,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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

        terminations = np.array(
            [info.get("success", False) for info in info_lists]
        ).astype(bool)
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

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

        if self.auto_reset or self.ignore_terminations:
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
