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

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch


def _resolve_gym_config_path(gym_config_path: str) -> Path:
    raw = Path(gym_config_path).expanduser()
    checked_paths: list[Path] = []

    def _try_resolve(candidate: Path) -> Optional[Path]:
        checked_paths.append(candidate)
        if candidate.is_file():
            return candidate.resolve()
        return None

    if raw.is_absolute():
        resolved = _try_resolve(raw)
        if resolved is not None:
            return resolved
        rel_path: Optional[Path] = None
    else:
        rel_path = raw

    if rel_path is not None and rel_path.parts[:1] == ("rlinf",):
        package_root = Path(__file__).resolve().parents[3]
        candidate = (package_root / Path(*rel_path.parts[1:])).resolve()
        if not candidate.is_relative_to(package_root):
            raise ValueError("RLinf gym config must stay within the package.")
        resolved = _try_resolve(candidate)
        if resolved is None:
            raise FileNotFoundError(f"RLinf gym config not found: {candidate}")
        return resolved

    if rel_path is not None and rel_path.parts[:2] == (
        "embodichain_tasks",
        "configs",
    ):
        from embodichain.utils.config_paths import resolve_config_path

        candidate = Path(resolve_config_path(rel_path))
        resolved = _try_resolve(candidate)
        if resolved is not None:
            return resolved

    root = os.environ.get("EMBODICHAIN_PATH")
    if root and rel_path is not None:
        resolved = _try_resolve(Path(root).expanduser().resolve() / rel_path)
        if resolved is not None:
            return resolved

    import_error: Optional[ImportError] = None
    try:
        import embodichain
    except ImportError as exc:
        import_error = exc
    else:
        pkg = Path(embodichain.__file__).resolve().parent
        if rel_path is not None:
            for base in (pkg, pkg.parent):
                resolved = _try_resolve(base / rel_path)
                if resolved is not None:
                    return resolved

    checked = ", ".join(str(path) for path in checked_paths)
    import_hint = ""
    if import_error is not None:
        import_hint = f" Importing `embodichain` also failed: {import_error}."

    raise FileNotFoundError(
        f"EmbodiChain gym config not found: {gym_config_path!r}. "
        f"Checked: {checked}. Set EMBODICHAIN_PATH or install embodichain with "
        f"configs next to the package.{import_hint}"
    )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _text_from_value(value: Any, *, _depth: int = 0) -> Optional[str]:
    """Extract a task instruction from common EmbodiChain metadata shapes."""
    if _depth > 8 or value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, Mapping):
        for key in (
            "lang",
            "task_prompt",
            "instruction",
            "task_description",
            "dataset",
            "env",
            "lerobot",
            "params",
            "metadata",
            "text",
        ):
            prompt = _text_from_value(value.get(key), _depth=_depth + 1)
            if prompt:
                return prompt
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            prompt = _text_from_value(item, _depth=_depth + 1)
            if prompt:
                return prompt
    return None


def _env_task_prompt(env: Any) -> Optional[str]:
    """Read the dataset instruction exposed by an EmbodiChain environment."""
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("task_prompt", "instruction", "task_description"):
            prompt = _text_from_value(getattr(current, attr, None))
            if prompt:
                return prompt
        prompt = _text_from_value(getattr(current, "metadata", None))
        if prompt:
            return prompt
        current = getattr(current, "unwrapped", None)
    return None


def _normalise_task_descriptions(value: Any, num_envs: int) -> list[str]:
    """Return one string instruction per vectorized environment row."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == num_envs:
        return [str(item) if item is not None else "" for item in value]
    prompt = _text_from_value(value) or ""
    return [prompt] * num_envs


def _config_task_prompt(
    config: Mapping[str, Any], config_path: Path, load_config
) -> Optional[str]:
    """Extract an instruction from a Gym config or its environment component."""
    prompt = _text_from_value(config)
    if prompt:
        return prompt
    environment = config.get("environment")
    if not isinstance(environment, Mapping) or "component" not in environment:
        return None
    from embodichain.utils.config_paths import resolve_config_path

    component_path = resolve_config_path(
        str(environment["component"]), base_dir=config_path.parent
    )
    if not component_path.is_file():
        return None
    return _text_from_value(load_config(component_path))


def _resolve_sim_device_and_gpu_id(
    cfg: Any, worker_info: Any
) -> tuple[torch.device, int]:
    sim_device = torch.device(str(_cfg_get(cfg, "sim_device", "cpu")))

    # RLinf will set `CUDA_VISIBLE_DEVICES` to each sub process according to the `component_placement` config,
    # So for EmbodiChain, we should always use `gpu_id=0` and cuda device `cuda:0` to access the GPU (which is actually the GPU assigned to the current process by RLinf).

    return sim_device, 0


def _clone_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_nested(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_nested(v) for v in value)
    return copy.deepcopy(value)


def _masked_update(dst: Any, src: Any, mask: torch.Tensor) -> Any:
    if src is None:
        return dst
    if isinstance(src, torch.Tensor):
        src = src.clone()
        if dst is None:
            return src
        dst = dst.clone()
        if src.ndim > 0 and dst.ndim > 0 and src.shape[0] == mask.shape[0]:
            dst[mask] = src[mask]
            return dst
        return src
    if isinstance(src, dict):
        dst_dict = (
            {}
            if not isinstance(dst, dict)
            else {k: _clone_nested(v) for k, v in dst.items()}
        )
        for key, value in src.items():
            dst_dict[key] = _masked_update(dst_dict.get(key), value, mask)
        return dst_dict
    if isinstance(src, (list, tuple)):
        src_seq = [_clone_nested(v) for v in src]
        if dst is None:
            return type(src)(src_seq)
        dst_seq = list(dst)
        for idx, value in enumerate(src_seq):
            if idx >= len(dst_seq):
                dst_seq.append(value)
            else:
                dst_seq[idx] = _masked_update(dst_seq[idx], value, mask)
        return type(src)(dst_seq)
    return _clone_nested(src)


class EmbodiChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ):
        super().__init__()
        self.cfg = cfg
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info

        self.seed = int(_cfg_get(cfg, "seed", 0)) + self.seed_offset
        self.num_envs = int(num_envs)
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = self.num_envs // self.group_size
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.max_episode_steps = int(_cfg_get(cfg, "max_episode_steps", 500))
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self.state_keys = list(_cfg_get(cfg, "state_keys", ["qpos", "qvel", "qf"]))
        self.main_camera_uid = _cfg_get(cfg, "main_camera_uid", "cam_high")
        self.wrist_camera_uids = list(_cfg_get(cfg, "wrist_camera_uids", []))
        self.clip_actions = bool(_cfg_get(cfg, "clip_actions", False))
        self.task_prompt = _text_from_value(
            _cfg_get(cfg, "task_prompt", _cfg_get(cfg, "task_description", None))
        )
        self._sim_device, self._gpu_id = _resolve_sim_device_and_gpu_id(
            cfg, worker_info
        )
        self._device = self._sim_device
        self._is_start = True
        self._elapsed_steps = torch.zeros(0, dtype=torch.int32)

        self.env = self._build_env()
        if not self.auto_reset:
            # EmbodiChain performs its own reset after terminal transitions.
            # Keep the terminal simulator state available to evaluators that
            # explicitly request no auto-reset, matching replay semantics.
            setattr(self.env, "_replay_no_auto_reset", True)
        if self.task_prompt is None:
            self.task_prompt = _env_task_prompt(self.env)
        self.task_descriptions = [self.task_prompt or ""] * self.num_envs
        action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        if action_low.ndim > 1:
            action_low = action_low[0]
            action_high = action_high[0]
        self.action_low = torch.as_tensor(
            action_low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            action_high, dtype=torch.float32, device=self.device
        )
        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            shape=tuple(action_low.shape),
            dtype=np.float32,
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._init_metrics()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps.to(self.device)

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.fail_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is None:
            self.prev_step_reward.zero_()
            self.success_once.zero_()
            self.fail_once.zero_()
            self.returns.zero_()
            self._elapsed_steps.zero_()
            return

        env_idx = torch.as_tensor(env_idx, dtype=torch.long, device=self.device)
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.fail_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def _record_metrics(
        self, step_reward: torch.Tensor, infos: dict[str, Any]
    ) -> dict[str, Any]:
        episode_info: dict[str, Any] = {}
        self.returns += step_reward

        if "success" in infos:
            self.success_once = self.success_once | infos["success"].bool()
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"].bool()
            episode_info["fail_once"] = self.fail_once.clone()

        episode_info["return"] = self.returns.clone()
        episode_len = self.elapsed_steps.to(self.device)
        episode_info["episode_len"] = episode_len.clone()
        denom = torch.clamp(episode_len.float(), min=1.0)
        episode_info["reward"] = episode_info["return"] / denom
        infos["episode"] = episode_info
        return infos

    def _build_env(self):
        from embodichain.lab.gym.utils.gym_utils import (
            config_to_cfg,
            get_manager_modules,
        )

        # EmbodiChain >=0.2.4 moved build_env into the core registry and
        # relocated official task envs into the bundled embodichain_tasks
        # package discovered via entry points.
        from embodichain.lab.gym.utils.registration import (
            build_env,
            discover_task_packages,
            execute_init_hooks,
        )
        from embodichain.utils.utility import load_config

        gym_config_path_cfg = _cfg_get(self.cfg, "gym_config_path")
        if not gym_config_path_cfg:
            raise ValueError(
                "EmbodiChain requires `gym_config_path` in the env config."
            )

        discover_task_packages()
        execute_init_hooks()

        gym_config_path_str = str(gym_config_path_cfg)
        gym_config_path = _resolve_gym_config_path(gym_config_path_str)
        gym_config = load_config(gym_config_path)
        if self.task_prompt is None:
            self.task_prompt = _config_task_prompt(
                gym_config, gym_config_path, load_config
            )

        env_cfg = config_to_cfg(
            copy.deepcopy(gym_config),
            manager_modules=get_manager_modules(),
            source_path=str(gym_config_path),
        )
        env_cfg.num_envs = self.num_envs
        env_cfg.max_episode_steps = self.max_episode_steps
        env_cfg.sim_cfg.headless = bool(_cfg_get(self.cfg, "headless", True))
        env_cfg.sim_cfg.sim_device = self._sim_device
        env_cfg.sim_cfg.gpu_id = self._gpu_id
        if bool(_cfg_get(self.cfg, "is_eval", False)) or bool(
            _cfg_get(self.cfg, "disable_dataset_recording", False)
        ):
            # Keep online rollouts out of the task's expert dataset destination.
            env_cfg.filter_dataset_saving = True
        return build_env(gym_config["id"], base_env_cfg=env_cfg)

    def _get_sensor_image(
        self, raw_obs: dict[str, Any], sensor_uid: str
    ) -> Optional[torch.Tensor]:
        """Return one camera's RGB image in ``[B, H, W, 3]`` format."""
        sensors = raw_obs.get("sensor")
        if sensors is None:
            return None
        if sensor_uid not in sensors:
            raise KeyError(
                f"Configured EmbodiChain camera {sensor_uid!r} is missing; "
                f"available sensors: {list(sensors.keys())}."
            )

        sensor_data = sensors[sensor_uid]
        if isinstance(sensor_data, Mapping):
            image = sensor_data.get("color", sensor_data.get("rgb"))
        else:
            image = sensor_data
        if image is None:
            return None
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, device=self.device)
        image = image.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(
                f"EmbodiChain camera {sensor_uid!r} must have shape [B,H,W,C], "
                f"got {tuple(image.shape)}."
            )
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[-1] != 3:
            raise ValueError(
                f"EmbodiChain camera {sensor_uid!r} must have RGB/RGBA channels, "
                f"got {tuple(image.shape)}."
            )
        if image.is_floating_point():
            if image.numel() > 0:
                finite = bool(torch.isfinite(image).all().item())
                min_value = float(image.amin().item())
                max_value = float(image.amax().item())
            else:
                finite = True
                min_value = 0.0
                max_value = 0.0
            if finite and min_value >= 0.0 and max_value <= 1.0:
                image = image * 255.0
            image = image.clamp(0.0, 255.0)
        return image.to(dtype=torch.uint8)

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        robot_obs = raw_obs["robot"]
        state_parts: list[torch.Tensor] = []
        for key in self.state_keys:
            if key not in robot_obs:
                continue
            value = robot_obs[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            value = value.to(self.device, dtype=torch.float32).reshape(
                self.num_envs, -1
            )
            state_parts.append(value)
        if not state_parts:
            raise ValueError(
                f"Failed to construct EmbodiChain state from keys {self.state_keys}."
            )
        wrapped: dict[str, Any] = {"states": torch.cat(state_parts, dim=-1)}
        # EmbodiChain policies may opt into a normalized episode phase input.
        # Keep it separate from qpos so the default state contract remains
        # backward compatible for existing recipes.
        wrapped["episode_steps"] = self._elapsed_steps.clone()
        task_descriptions = raw_obs.get("task_descriptions")
        if task_descriptions is None:
            task_descriptions = raw_obs.get("task_description")
        if task_descriptions is None:
            task_descriptions = getattr(
                self,
                "task_descriptions",
                getattr(self, "task_prompt", ""),
            )
        wrapped["task_descriptions"] = _normalise_task_descriptions(
            task_descriptions, self.num_envs
        )
        if self.main_camera_uid:
            main_image = self._get_sensor_image(raw_obs, self.main_camera_uid)
            if main_image is not None:
                wrapped["main_images"] = main_image
                if self.wrist_camera_uids:
                    wrist_images = [
                        self._get_sensor_image(raw_obs, sensor_uid)
                        for sensor_uid in self.wrist_camera_uids
                    ]
                    if any(image is None for image in wrist_images):
                        missing = [
                            uid
                            for uid, image in zip(
                                self.wrist_camera_uids, wrist_images, strict=True
                            )
                            if image is None
                        ]
                        raise ValueError(
                            "EmbodiChain wrist camera(s) missing from observations: "
                            f"{missing}."
                        )
                    wrapped["extra_view_images"] = torch.stack(
                        [image for image in wrist_images if image is not None], dim=1
                    )
                else:
                    wrapped["extra_view_images"] = None
        return wrapped

    def _wrap_info(self, infos: Any) -> dict[str, Any]:
        if infos is None:
            return {}
        if isinstance(infos, dict):
            return infos
        return dict(infos)

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        options = {} if options is None else dict(options)
        env_idx = options.pop("env_idx", None)
        if env_idx is None:
            raw_obs, infos = self.env.reset(seed=seed)
            self._reset_metrics()
        else:
            reset_ids = torch.as_tensor(env_idx, dtype=torch.int32, device=self.device)
            raw_obs, infos = self.env.reset(options={"reset_ids": reset_ids})
            self._reset_metrics(reset_ids)
        self._is_start = True
        return self._wrap_obs(raw_obs), self._wrap_info(infos)

    def step(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action_tensor = (
            actions.to(self.device, dtype=torch.float32)
            if isinstance(actions, torch.Tensor)
            else torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        )
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        action_tensor = action_tensor.reshape(self.num_envs, -1)
        if not torch.isfinite(action_tensor).all():
            raise ValueError("EmbodiChain actions must contain only finite values.")

        action_clipped = torch.zeros_like(action_tensor, dtype=torch.bool)
        if self.clip_actions:
            if action_tensor.shape[-1] != self.action_low.shape[-1]:
                raise ValueError(
                    "EmbodiChain clipping requires actions to match the Box bounds: "
                    f"got {action_tensor.shape[-1]}, expected {self.action_low.shape[-1]}."
                )
            bounded_action = action_tensor.clamp(self.action_low, self.action_high)
            action_clipped = bounded_action != action_tensor
            action_tensor = bounded_action
        applied_action = action_tensor.detach().clone()

        raw_obs, rewards, terminations, truncations, infos = self.env.step(
            action_tensor
        )
        infos = self._wrap_info(infos)
        infos["applied_action"] = applied_action
        infos["action_clipped"] = action_clipped
        self._elapsed_steps += 1
        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations = torch.zeros_like(
                terminations, dtype=torch.bool, device=self.device
            )
            if "episode" in infos:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()
        dones = torch.logical_or(terminations, truncations)
        if dones.any() and self.auto_reset:
            done_ids = torch.nonzero(dones, as_tuple=False).flatten().to(torch.int32)
            self._reset_metrics(done_ids)
        self._is_start = False
        return (
            self._wrap_obs(raw_obs),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def chunk_step(self, chunk_actions: Union[np.ndarray, torch.Tensor]):
        return self._chunk_step(
            chunk_actions, collect_final_info=self.auto_reset, preserve_step_dones=False
        )

    def _chunk_step(
        self,
        chunk_actions: Union[np.ndarray, torch.Tensor],
        *,
        collect_final_info: bool,
        preserve_step_dones: bool,
    ):
        chunk_actions = (
            chunk_actions.to(self.device, dtype=torch.float32)
            if isinstance(chunk_actions, torch.Tensor)
            else torch.as_tensor(chunk_actions, dtype=torch.float32, device=self.device)
        )
        if chunk_actions.ndim != 3:
            raise ValueError(
                "chunk_actions must have shape [num_envs, chunk_steps, action_dim], "
                f"got {tuple(chunk_actions.shape)}."
            )

        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_terms = []
        raw_truncs = []
        aggregated_final_info = None
        aggregated_final_obs = None

        for step_idx in range(int(chunk_actions.shape[1])):
            obs, rewards, terminations, truncations, infos = self.step(
                chunk_actions[:, step_idx]
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(rewards)
            raw_terms.append(terminations)
            raw_truncs.append(truncations)
            step_dones = torch.logical_or(terminations, truncations)
            if step_dones.any() and collect_final_info:
                aggregated_final_info = _masked_update(
                    aggregated_final_info,
                    infos.get("final_info", infos),
                    step_dones,
                )
                aggregated_final_obs = _masked_update(
                    aggregated_final_obs,
                    infos.get("final_observation", obs),
                    step_dones,
                )

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)
        raw_terms_t = torch.stack(raw_terms, dim=1)
        raw_truncs_t = torch.stack(raw_truncs, dim=1)

        past_terminations = raw_terms_t.any(dim=1)
        past_truncations = raw_truncs_t.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and collect_final_info:
            infos_list[-1] = dict(infos_list[-1])
            infos_list[-1]["final_info"] = (
                aggregated_final_info
                if aggregated_final_info is not None
                else _clone_nested(infos_list[-1].get("final_info", infos_list[-1]))
            )
            infos_list[-1]["final_observation"] = (
                aggregated_final_obs
                if aggregated_final_obs is not None
                else _clone_nested(obs_list[-1])
            )
            infos_list[-1]["_final_info"] = past_dones
            infos_list[-1]["_final_observation"] = past_dones
            infos_list[-1]["_elapsed_steps"] = past_dones

        if preserve_step_dones:
            chunk_terminations = raw_terms_t
            chunk_truncations = raw_truncs_t
        else:
            chunk_terminations = torch.zeros_like(raw_terms_t)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_truncs_t)
            chunk_truncations[:, -1] = past_truncations

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def update_reset_state_ids(self):
        return None

    def sample_action_space(self) -> torch.Tensor:
        return torch.as_tensor(
            self.action_space.sample(), dtype=torch.float32, device=self.device
        )

    def close(self):
        try:
            # RLinf owns the worker process after simulator teardown.
            self.env.close(exit_process=False)
        except Exception:
            pass
