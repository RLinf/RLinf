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
from typing import Any, Optional, Union

import gym
import numpy as np
import torch

__all__ = ["FrankaSimEnv"]


# ==========================================================
# Obs utils: object scalar unwrap + robust flatten
# ==========================================================
def _unwrap_object_scalar(x: Any, max_unwrap: int = 10) -> Any:
    """Unwrap numpy object scalar (shape=(), dtype=object) safely."""
    cur = x
    for _ in range(max_unwrap):
        arr = np.asarray(cur)
        if arr.dtype == object and arr.shape == ():
            try:
                cur = arr.item()
            except Exception:
                break
        else:
            break
    return cur


def _flatten_any_safe(
    x: Any,
    visited: Optional[set[int]] = None,
    depth: int = 0,
    max_depth: int = 20,
) -> np.ndarray:
    """Flatten nested dict/list/tuple/object-arr into 1D float32 array."""
    if visited is None:
        visited = set()
    if depth > max_depth:
        raise RuntimeError("Observation flatten exceeded max_depth; obs may be cyclic.")

    obj_id = id(x)
    if obj_id in visited:
        return np.zeros((0,), dtype=np.float32)
    visited.add(obj_id)

    if isinstance(x, dict):
        parts = [
            _flatten_any_safe(x[k], visited, depth + 1, max_depth)
            for k in sorted(x.keys())
        ]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)

    if isinstance(x, (list, tuple)):
        parts = [_flatten_any_safe(v, visited, depth + 1, max_depth) for v in x]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)

    arr = np.asarray(x)

    if arr.dtype == object and arr.shape == ():
        y = _unwrap_object_scalar(x)
        if id(y) == id(x):
            return np.zeros((0,), np.float32)
        return _flatten_any_safe(y, visited, depth + 1, max_depth)

    if arr.dtype == object:
        parts = [
            _flatten_any_safe(v, visited, depth + 1, max_depth)
            for v in arr.ravel().tolist()
        ]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)

    return arr.astype(np.float32, copy=False).reshape(-1)


def extract_serl_state(raw_obs: Any, state_key: str = "states") -> np.ndarray:
    """Extract and flatten SERL state."""
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]
    if isinstance(raw_obs, dict):
        raw_obs = raw_obs[state_key]
    raw_obs = _unwrap_object_scalar(raw_obs)
    return _flatten_any_safe(raw_obs)


def extract_serl_images_dict(
    raw_obs: Any,
    image_key: str = "images",
) -> dict[str, np.ndarray]:
    """Extract SERL images dict from raw obs."""
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]
    if not isinstance(raw_obs, dict):
        raise TypeError(f"Expected dict obs, got {type(raw_obs)}.")

    images_obj = raw_obs[image_key]
    images_dict = _unwrap_object_scalar(images_obj)
    if not isinstance(images_dict, dict):
        raise TypeError(f"Expected images_dict to be dict, got {type(images_dict)}.")
    return {k: np.asarray(v, dtype=np.uint8) for k, v in images_dict.items()}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Fetch cfg value for dict/attr config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _torch_clone_dict(x: Any) -> Any:
    """Clone nested tensor structures, deepcopy non-tensors."""
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: _torch_clone_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_torch_clone_dict(v) for v in x]
    return copy.deepcopy(x)


# ==========================================================
# SERLFrankaEnv (Maniskill-style)
# ==========================================================
class FrankaSimEnv(gym.Env):
    """SERL FrankaSim wrapper aligned with ManiSkill env wrapper style."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ):
        """Initialize SERL wrapper.

        Args:
          cfg: Config object/dict.
          num_envs: Number of parallel envs (manual list of gym envs).
          seed_offset: Seed offset added to cfg.seed.
          total_num_processes: Total num processes for interface parity.
          worker_info: Worker metadata.
          record_metrics: Whether to record episode metrics.
        """
        super().__init__()
        env_seed = int(_cfg_get(cfg, "seed", 0))
        self.seed = env_seed + int(seed_offset)

        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.cfg = cfg

        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))

        self.num_group = int(num_envs) // int(_cfg_get(cfg, "group_size", 1))
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.use_fixed_reset_state_ids = bool(
            _cfg_get(cfg, "use_fixed_reset_state_ids", False)
        )

        self.video_cfg = _cfg_get(cfg, "video_cfg", None)

        self.wrap_obs_mode = str(
            _cfg_get(cfg, "wrap_obs_mode", _cfg_get(cfg, "obs_format", "openvla"))
        ).lower()
        self.obs_mode = str(_cfg_get(cfg, "obs_mode", "state")).lower()
        self.obs_mode = (
            "rgb" if self.obs_mode in ("rgb", "image", "vision") else "state"
        )

        self.state_key = str(_cfg_get(cfg, "state_key", "states"))
        self.image_key = str(_cfg_get(cfg, "image_key", "images"))

        self.main_camera = str(
            _cfg_get(cfg, "main_camera", _cfg_get(cfg, "camera", "front"))
        )
        self.extra_camera = str(_cfg_get(cfg, "extra_camera", "wrist"))
        self.use_wrist_as_extra_view = bool(
            _cfg_get(cfg, "use_wrist_as_extra_view", True)
        )

        self.task_prompt = str(_cfg_get(cfg, "task_prompt", "Pick up the cube."))

        self._device = torch.device("cpu")

        self._configure_mujoco()

        self.env_id = str(_cfg_get(cfg, "gym_id", "PandaPickCube-v0"))
        self.envs = [self._make_env(i) for i in range(int(num_envs))]

        self.single_action_space = self.envs[0].action_space
        self.action_space = self.single_action_space

        raw0, _ = self.envs[0].reset(seed=self.seed)
        self._state_dim = int(extract_serl_state(raw0, self.state_key).shape[0])
        self._init_observation_space(raw0)

        # Maniskill-style metrics fields.
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )
        self.record_metrics = bool(record_metrics)
        self._is_start = True
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._needs_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._init_reset_state_ids()
        self.info_logging_keys = ["success", "fail"]
        if self.record_metrics:
            self._init_metrics()

        self._last_obs: Optional[dict[str, Any]] = None
        self._last_info: dict[str, Any] = {}

    # -------------------- properties (match ManiSkill wrapper) --------------------
    @property
    def total_num_group_envs(self) -> int:
        return np.iinfo(np.uint8).max // 2

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def instruction(self) -> list[str]:
        return [self.task_prompt] * self.num_envs

    # -------------------- init helpers --------------------
    def _configure_mujoco(self) -> None:
        if self.obs_mode != "rgb":
            self.video_cfg = None

    def _make_env(self, env_idx: int) -> gym.Env:
        kwargs = {}
        if self.obs_mode == "rgb":
            kwargs["image_obs"] = True
            kwargs["render_mode"] = "rgb_array"
        env = gym.make(self.env_id, disable_env_checker=True, **kwargs)
        try:
            env.reset(seed=self.seed + env_idx)
        except Exception:
            pass
        return env

    def _init_observation_space(self, raw0: Any) -> None:
        if self.obs_mode != "rgb":
            self.observation_space = gym.spaces.Dict(
                {
                    "states": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_envs, self._state_dim),
                        dtype=np.float32,
                    ),
                }
            )
            return

        imgs = extract_serl_images_dict(raw0, self.image_key)
        main = imgs.get(self.main_camera, imgs[sorted(imgs.keys())[0]])
        h, w, c = main.shape
        spaces = {
            "main_images": gym.spaces.Box(
                0, 255, shape=(self.num_envs, h, w, c), dtype=np.uint8
            ),
            "states": gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.num_envs, self._state_dim),
                dtype=np.float32,
            ),
        }
        if self.use_wrist_as_extra_view:
            spaces["extra_view_images"] = gym.spaces.Box(
                0, 255, shape=(self.num_envs, 1, h, w, c), dtype=np.uint8
            )
        self.observation_space = gym.spaces.Dict(spaces)

    # -------------------- reset-state ids (compat only) --------------------
    def _init_reset_state_ids(self) -> None:
        self.reset_state_ids = None

    def update_reset_state_ids(self) -> None:
        return

    # -------------------- obs wrap (match ManiSkill style) --------------------
    def _wrap_obs(self, raw_obs: Any) -> dict[str, Any]:
        if self.obs_mode == "state":
            state_np = extract_serl_state(raw_obs, self.state_key)
            state = torch.from_numpy(state_np).to(self.device)
            if self.wrap_obs_mode == "simple":
                return {"states": state}
            return {"states": state, "task_descriptions": self.task_prompt}

        state_np = extract_serl_state(raw_obs, self.state_key)
        state = torch.from_numpy(state_np).to(self.device)

        main_np, extra_np = self._pick_images(raw_obs)
        main = torch.from_numpy(np.ascontiguousarray(main_np, np.uint8)).to(self.device)
        extra = torch.from_numpy(np.ascontiguousarray(extra_np, np.uint8)).to(
            self.device
        )
        extra_view = extra.unsqueeze(0) if self.use_wrist_as_extra_view else None

        if self.wrap_obs_mode == "simple":
            return {
                "main_images": main,
                "extra_view_images": extra_view,
                "states": state,
            }
        if self.wrap_obs_mode == "openpi":
            return {
                "main_images": main,
                "wrist_images": extra,
                "extra_view_images": extra_view,
                "states": state,
                "task_descriptions": self.task_prompt,
            }
        # default openvla
        return {
            "main_images": main,
            "extra_view_images": extra_view,
            "states": state,
            "task_descriptions": self.task_prompt,
        }

    def _pick_images(self, raw_obs: dict) -> tuple[np.ndarray, np.ndarray]:
        images = extract_serl_images_dict(raw_obs, self.image_key)
        keys = sorted(images.keys())
        main = images.get(self.main_camera, images[keys[0]])
        extra = images.get(self.extra_camera, main)
        return main, extra

    def _collate_obs(self, obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = set().union(*[o.keys() for o in obs_list])

        for k in sorted(keys):
            vals = [o.get(k, None) for o in obs_list]
            if all(v is None for v in vals):
                out[k] = None
            elif isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals

        # pad states to max dim
        states = [o["states"].view(-1) for o in obs_list]
        max_d = max((s.numel() for s in states), default=0)
        padded = torch.zeros(
            (len(states), max_d), dtype=torch.float32, device=self.device
        )
        for i, s in enumerate(states):
            padded[i, : s.numel()] = s
        out["states"] = padded
        return out

    def _collate_infos(self, info_list: list[dict]) -> dict[str, Any]:
        keys = set().union(*[inf.keys() for inf in info_list if isinstance(inf, dict)])
        out: dict[str, Any] = {}
        for k in sorted(keys):
            vals = [inf.get(k, None) for inf in info_list]
            is_bool = all(isinstance(v, (bool, np.bool_)) or v is None for v in vals)
            is_num = all(
                isinstance(v, (int, float, np.number)) or v is None for v in vals
            )
            if is_bool:
                out[k] = torch.tensor(
                    [bool(v) if v is not None else False for v in vals],
                    device=self.device,
                    dtype=torch.bool,
                )
            elif is_num:
                out[k] = torch.tensor(
                    [float(v) if v is not None else 0.0 for v in vals],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                out[k] = vals
        return out

    # -------------------- reward / metrics (match ManiSkill style) --------------------
    def _calc_step_reward(self, reward: torch.Tensor) -> torch.Tensor:
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward
        return reward_diff if self.use_rel_reward else reward

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self._elapsed_steps[mask] = 0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0.0
        else:
            self.prev_step_reward[:] = 0.0
            self._elapsed_steps[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(
        self, step_reward: torch.Tensor, infos: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.record_metrics:
            return infos
        episode_info: dict[str, Any] = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"].bool()
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"].bool()
            episode_info["fail_once"] = self.fail_once.clone()

        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        denom = torch.clamp(episode_info["episode_len"].float(), min=1.0)
        episode_info["reward"] = episode_info["return"] / denom
        infos["episode"] = episode_info
        return infos

    # -------------------- RLinf API: reset/step (match ManiSkill style) --------------------
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options is None:
            seed = self.seed if seed is None else seed
            options = {}

        env_idx = options.get("env_idx", None) if isinstance(options, dict) else None
        reset_options = dict(options)
        reset_options.pop("env_idx", None)

        if env_idx is None:
            idxs = range(self.num_envs)
            self._reset_metrics()
            self._needs_reset[:] = False
        else:
            env_idx = torch.as_tensor(env_idx, dtype=torch.int64, device=self.device)
            idxs = env_idx.detach().cpu().tolist()
            self._reset_metrics(env_idx)
            self._needs_reset[env_idx] = False

        obs_list, info_list = [], []
        for i in range(self.num_envs):
            if i in idxs:
                seed_i = None
                if seed is not None:
                    seed_i = (
                        int(seed[i])
                        if isinstance(seed, (list, tuple, np.ndarray))
                        else int(seed) + i
                    )
                raw_obs, info = self.envs[i].reset(seed=seed_i, options=reset_options)
                obs_list.append(self._wrap_obs(raw_obs))
                info_list.append(info if isinstance(info, dict) else {})
            else:
                obs_list.append(self._index_cached_obs(i))
                info_list.append({})

        obs = self._collate_obs(obs_list)
        infos = self._collate_infos(info_list)

        self._is_start = True
        self._last_obs, self._last_info = obs, infos
        return obs, infos

    def step(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        act_np = self._normalize_actions(actions)

        obs_list, info_list = [], []
        rew_list, term_list, trunc_list = [], [], []
        stepped_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i, env in enumerate(self.envs):
            obs_i, info_i, rew_i, term_i, trunc_i, stepped = self._step_one_env(
                env_idx=i,
                env=env,
                action=act_np[i],
                auto_reset=auto_reset,
            )
            obs_list.append(obs_i)
            info_list.append(info_i)
            rew_list.append(rew_i)
            term_list.append(term_i)
            trunc_list.append(trunc_i)
            stepped_mask[i] = stepped

        self._elapsed_steps[stepped_mask] += 1

        obs = self._collate_obs(obs_list)
        infos = self._collate_infos(info_list)

        raw_reward = torch.tensor(rew_list, device=self.device, dtype=torch.float32)
        step_reward = self._calc_step_reward(raw_reward)

        reward_scale = float(_cfg_get(self.cfg, "reward_scale", 1))
        step_reward = step_reward * reward_scale

        terminations = torch.tensor(term_list, device=self.device, dtype=torch.bool)
        truncations = torch.tensor(trunc_list, device=self.device, dtype=torch.bool)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics and "episode" in infos:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = bool(auto_reset) and bool(self.auto_reset)
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        self._last_obs, self._last_info = obs, infos
        return obs, step_reward, terminations, truncations, infos

    def _normalize_actions(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        act_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        if act_np.ndim == 1:
            act_np = np.repeat(act_np[None, :], self.num_envs, axis=0)
        if act_np.shape[0] != self.num_envs:
            raise ValueError(
                "Invalid action batch dimension. Expected shape [num_envs, act_dim] "
                f"with num_envs={self.num_envs}, got {act_np.shape}."
            )
        return act_np.astype(np.float32, copy=False)

    def _step_one_env(
        self,
        env_idx: int,
        env: gym.Env,
        action: np.ndarray,
        auto_reset: bool,
    ) -> tuple[dict[str, Any], dict, float, bool, bool, bool]:
        if self._needs_reset[env_idx]:
            if auto_reset and self.auto_reset:
                env.reset()
                self._needs_reset[env_idx] = False
                self._reset_metrics(torch.tensor([env_idx], device=self.device))
            else:
                return self._index_cached_obs(env_idx), {}, 0.0, True, False, False

        out = env.step(np.clip(action.reshape(-1), -1.0, 1.0))
        if len(out) == 5:
            raw_obs, rew, terminated, truncated, info = out
        else:
            raw_obs, rew, done, info = out
            terminated, truncated = bool(done), False

        obs = self._wrap_obs(raw_obs)
        info = info if isinstance(info, dict) else {}
        return obs, info, float(rew), bool(terminated), bool(truncated), True

    def _index_cached_obs(self, env_idx: int) -> dict[str, Any]:
        if self._last_obs is None:
            raw_obs, _ = self.envs[env_idx].reset()
            return self._wrap_obs(raw_obs)
        out: dict[str, Any] = {}
        for k, v in self._last_obs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == self.num_envs:
                out[k] = v[env_idx]
            elif isinstance(v, list) and len(v) == self.num_envs:
                out[k] = v[env_idx]
            else:
                out[k] = v
        return out

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = _torch_clone_dict(obs)
        final_info = _torch_clone_dict(infos)

        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        obs, infos = self.reset(options={"env_idx": env_idx})

        infos = dict(infos)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    # -------------------- chunk_step / sample_action_space (match ManiSkill style) --------------------
    def chunk_step(self, chunk_actions: Union[np.ndarray, torch.Tensor]):
        chunk_actions = (
            chunk_actions
            if isinstance(chunk_actions, torch.Tensor)
            else torch.from_numpy(np.asarray(chunk_actions))
        )
        if chunk_actions.ndim != 3:
            raise ValueError(
                "chunk_actions must have shape [num_envs, chunk_steps, act_dim], "
                f"got {tuple(chunk_actions.shape)}."
            )

        chunk_size = int(chunk_actions.shape[1])
        obs_list = []
        infos_list = []
        chunk_rewards, raw_terms, raw_truncs = [], [], []

        for i in range(chunk_size):
            actions = chunk_actions[:, i].to(self.device)
            obs, rew, term, trunc, infos = self.step(actions, auto_reset=False)
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(rew)
            raw_terms.append(term)
            raw_truncs.append(trunc)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_terms = torch.stack(raw_terms, dim=1)
        raw_truncs = torch.stack(raw_truncs, dim=1)

        past_terminations = raw_terms.any(dim=1)
        past_truncations = raw_truncs.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_terms)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_truncs)
        chunk_truncations[:, -1] = past_truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def sample_action_space(self) -> torch.Tensor:
        a = self.action_space.sample()
        return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)
