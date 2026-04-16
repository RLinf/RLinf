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

"""D4RL offline RL dataset: load transitions and expose PyTorch ``Dataset`` items."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset

try:
    import d4rl
except ImportError:  # pragma: no cover
    d4rl = None  # type: ignore[assignment]

try:
    import gym  # type: ignore
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import gym as _gym

    GymEnv = _gym.Env
else:
    GymEnv = Any


class D4RLDataset(Dataset):
    """D4RL offline transitions as a map-style ``Dataset`` (one row per index).

    DataLoader construction, distributed sampling, and iteration state belong on
    the training worker (see ``EmbodiedIQLFSDPPolicy.build_offline_dataloader``).
    """

    def __init__(
        self,
        env: GymEnv,
        task_name: str,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        if d4rl is None or gym is None:  # pragma: no cover
            missing = []
            if gym is None:
                missing.append("gym")
            if d4rl is None:
                missing.append("d4rl")
            raise ImportError(
                "D4RLDataset requires optional dependencies: "
                + ", ".join(missing)
                + ". Please install them to use D4RL offline datasets."
            )

        raw = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            raw["actions"] = np.clip(raw["actions"], -lim, lim)

        dones_float = self._compute_dones_float(
            raw["observations"], raw["next_observations"], raw["terminals"]
        )
        self.observations = raw["observations"].astype(np.float32)
        self.actions = raw["actions"].astype(np.float32)
        self.rewards = raw["rewards"].astype(np.float32)
        self.masks = 1.0 - raw["terminals"].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = raw["next_observations"].astype(np.float32)
        self.size = len(self.observations)

        self._apply_reward_postprocess(task_name)
        self._init_tensor_cache()

    def _init_tensor_cache(self) -> None:
        self._torch_observations: torch.Tensor | None = None
        self._torch_actions: torch.Tensor | None = None
        self._torch_rewards: torch.Tensor | None = None
        self._torch_masks: torch.Tensor | None = None
        self._torch_next_observations: torch.Tensor | None = None

    def _ensure_torch_cache(self) -> None:
        if self._torch_observations is None:
            self._torch_observations = torch.from_numpy(self.observations).float()
            self._torch_actions = torch.from_numpy(self.actions).float()
            self._torch_rewards = torch.from_numpy(self.rewards).float()
            self._torch_masks = torch.from_numpy(self.masks).float()
            self._torch_next_observations = torch.from_numpy(
                self.next_observations
            ).float()

    def __len__(self) -> int:
        return int(self.size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_torch_cache()
        assert self._torch_observations is not None
        assert self._torch_actions is not None
        assert self._torch_rewards is not None
        assert self._torch_masks is not None
        assert self._torch_next_observations is not None
        return {
            "observations": self._torch_observations[idx],
            "actions": self._torch_actions[idx],
            "rewards": self._torch_rewards[idx],
            "masks": self._torch_masks[idx],
            "next_observations": self._torch_next_observations[idx],
        }

    @staticmethod
    def _compute_dones_float(
        observations: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        dones_float = np.zeros_like(terminals)
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(observations[i + 1] - next_observations[i]) > 1e-6
                or terminals[i] == 1.0
            ):
                dones_float[i] = 1.0
            else:
                dones_float[i] = 0.0
        if len(dones_float) > 0:
            dones_float[-1] = 1.0
        return dones_float

    def _apply_reward_postprocess(self, task_name: str) -> None:
        if "antmaze" in task_name:
            self.rewards -= 1.0
        elif any(name in task_name for name in ("halfcheetah", "walker2d", "hopper")):
            self._normalize_rewards_for_mujoco()

    @staticmethod
    def _default_dataset_path(task_name: str) -> str:
        if gym is None:  # pragma: no cover
            raise ImportError(
                "D4RLDataset.from_path requires 'gym' (or install the embodied env deps)."
            )
        try:
            canonical_env_id = gym.spec(task_name).id
        except Exception:
            canonical_env_id = task_name
        return str(Path.home() / ".d4rl" / "datasets" / f"{canonical_env_id}.hdf5")

    @staticmethod
    def infer_obs_action_dims_from_env(task_name: str) -> tuple[int, int]:
        """Infer flat obs/action dims from a gym env's spaces."""
        if gym is None:  # pragma: no cover
            raise RuntimeError(
                "Failed to infer D4RL obs/action dims: missing 'gym' dependency."
            )
        env = gym.make(task_name)
        try:
            obs_shape = getattr(env.observation_space, "shape", None)
            act_shape = getattr(env.action_space, "shape", None)
            if obs_shape is None or act_shape is None:
                raise RuntimeError(
                    f"Env {task_name!r} does not expose Box-like observation/action shape."
                )
            return int(np.prod(obs_shape)), int(np.prod(act_shape))
        finally:
            env.close()

    @classmethod
    def from_path(
        cls,
        dataset_path: str | os.PathLike[str] | None,
        task_name: str,
        *,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ) -> D4RLDataset:
        path = (
            str(dataset_path)
            if dataset_path is not None
            else cls._default_dataset_path(task_name)
        )

        if d4rl is None or gym is None:  # pragma: no cover
            missing = []
            if gym is None:
                missing.append("gym")
            if d4rl is None:
                missing.append("d4rl")
            raise ImportError(
                "D4RLDataset.from_path requires optional dependencies: "
                + ", ".join(missing)
                + ". Please install them to load D4RL datasets."
            )

        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading D4RL dataset from path requires 'h5py'. "
                "Install it or use env-based construction."
            ) from exc

        if not os.path.exists(path):
            env = gym.make(task_name)
            try:
                return cls(
                    env=env, task_name=task_name, clip_to_eps=clip_to_eps, eps=eps
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        with h5py.File(path, "r") as f:
            observations = np.asarray(f["observations"], dtype=np.float32)
            actions = np.asarray(f["actions"], dtype=np.float32)
            rewards = np.asarray(f["rewards"], dtype=np.float32)
            terminals = np.asarray(f["terminals"], dtype=np.float32)
            next_observations = np.asarray(f["next_observations"], dtype=np.float32)

        if clip_to_eps:
            lim = 1 - eps
            actions = np.clip(actions, -lim, lim)

        ds = cls.from_arrays(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=1.0 - terminals,
            dones_float=cls._compute_dones_float(
                observations, next_observations, terminals
            ),
            next_observations=next_observations,
        )
        ds._apply_reward_postprocess(task_name)
        return ds

    @classmethod
    def from_arrays(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
    ) -> D4RLDataset:
        self = cls.__new__(cls)
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = len(observations)
        self._init_tensor_cache()
        return self

    def get_obs_action_dims(self) -> tuple[int, int]:
        return int(self.observations.shape[-1]), int(self.actions.shape[-1])

    def get_dataset_size(self) -> int:
        return int(self.size)

    def _normalize_rewards_for_mujoco(self) -> None:
        trajs = self._split_into_trajectories(
            self.observations,
            self.actions,
            self.rewards,
            self.masks,
            self.dones_float,
            self.next_observations,
        )

        def compute_returns(traj: list[tuple[np.ndarray, ...]]) -> float:
            return float(sum(rew for _, _, rew, _, _, _ in traj))

        trajs.sort(key=compute_returns)
        denom = compute_returns(trajs[-1]) - compute_returns(trajs[0])
        if abs(denom) < 1e-12:
            return
        self.rewards /= denom
        self.rewards *= 1000.0

    @staticmethod
    def _split_into_trajectories(
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
    ) -> list[list[tuple[np.ndarray, ...]]]:
        trajs: list[list[tuple[np.ndarray, ...]]] = [[]]
        for i in tqdm.tqdm(range(len(observations))):
            trajs[-1].append(
                (
                    observations[i],
                    actions[i],
                    rewards[i],
                    masks[i],
                    dones_float[i],
                    next_observations[i],
                )
            )
            if dones_float[i] == 1.0 and i + 1 < len(observations):
                trajs.append([])
        return trajs


def build_d4rl_dataset_from_cfg(cfg: Any) -> D4RLDataset:
    """Resolve ``cfg.data`` and return a ``D4RLDataset`` instance."""
    dataset_type = str(cfg.data.get("dataset_type", "d4rl")).lower()
    if dataset_type != "d4rl":
        raise NotImplementedError(
            f"Offline IQL currently only supports dataset_type='d4rl', got {dataset_type!r}."
        )
    dataset_task_name = cfg.data.get("task_name")
    if not dataset_task_name:
        raise ValueError("Offline dataset requires data.task_name.")
    dataset_path = cfg.data.get("dataset_path", None)

    dataset_init_kwargs_cfg = cfg.data.get("dataset_init_kwargs", {})
    if OmegaConf.is_config(dataset_init_kwargs_cfg):
        dataset_init_kwargs = OmegaConf.to_container(
            dataset_init_kwargs_cfg, resolve=True
        )
    else:
        dataset_init_kwargs = dict(dataset_init_kwargs_cfg)
    dataset_init_kwargs.setdefault("dataset_path", dataset_path)
    dataset_init_kwargs.setdefault("task_name", str(dataset_task_name))

    return D4RLDataset.from_path(**dataset_init_kwargs)
