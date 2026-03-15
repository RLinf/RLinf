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

"""D4RL offline RL datasets: single class D4RLDataset (build from env + partition).

This module provides numpy-based datasets for D4RL benchmarks (e.g. MuJoCo,
AntMaze, Adroit). Use D4RLDataset(env, env_name) then .partition(rank, world_size)
for data-parallel training. Not a torch Dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import gym
import numpy as np
import tqdm

try:
    import d4rl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "d4rl is required for D4RL offline datasets. Please install d4rl."
    ) from exc


def split_into_trajectories(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    dones_float: np.ndarray,
    next_observations: np.ndarray,
) -> list[list[tuple[np.ndarray, ...]]]:
    """Split flat transition arrays into a list of trajectories.

    Each trajectory is a list of (obs, action, reward, mask, done_float, next_obs).

    Args:
        observations: Observation array [N, obs_dim].
        actions: Action array [N, action_dim].
        rewards: Reward array [N].
        masks: Mask array [N].
        dones_float: Done float array [N].
        next_observations: Next observation array [N, obs_dim].

    Returns:
        List of trajectories; each trajectory is a list of 6-tuples.
    """
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


class D4RLDataset:
    """Single class for D4RL offline data: build from env+env_name, partition by rank.

    Usage: dataset = D4RLDataset(env, env_name); shard = dataset.partition(rank, world_size).
    """

    def __init__(
        self,
        env: gym.Env,
        env_name: str,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        raw = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            raw["actions"] = np.clip(raw["actions"], -lim, lim)
        dones_float = np.zeros_like(raw["rewards"])
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(raw["observations"][i + 1] - raw["next_observations"][i])
                > 1e-6
                or raw["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1
        self.observations = raw["observations"].astype(np.float32)
        self.actions = raw["actions"].astype(np.float32)
        self.rewards = raw["rewards"].astype(np.float32)
        self.masks = 1.0 - raw["terminals"].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = raw["next_observations"].astype(np.float32)
        self.size = len(self.observations)
        if "antmaze" in env_name:
            self.rewards -= 1.0
        elif any(name in env_name for name in ("halfcheetah", "walker2d", "hopper")):
            normalize(self)

    @staticmethod
    def _default_dataset_path(env_name: str) -> str:
        """Default D4RL hdf5 path for an env_name."""
        # Gym can canonicalize an env id (e.g. fill default version). Note: this
        # is only about env id parsing; gym does not define any dataset path.
        try:
            canonical_env_id = gym.spec(env_name).id
        except Exception:
            canonical_env_id = env_name

        # D4RL commonly stores datasets under ~/.d4rl/datasets/{env_id}.hdf5
        return str(Path.home() / ".d4rl" / "datasets" / f"{canonical_env_id}.hdf5")

    @classmethod
    def from_path(
        cls,
        dataset_path: str | os.PathLike[str] | None,
        env_name: str,
        *,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ) -> D4RLDataset:
        """Load D4RL dataset from a local hdf5 file, without creating an env.

        Args:
            dataset_path: Path to the D4RL dataset hdf5. If None, uses the default
                under ~/.d4rl/datasets/{env_name}.hdf5.
            env_name: Env name string; used to apply reward normalization logic.
            clip_to_eps: Whether to clip actions to [-1+eps, 1-eps].
            eps: Epsilon for action clipping.

        Returns:
            D4RLDataset loaded from file.
        """
        path = (
            str(dataset_path)
            if dataset_path is not None
            else cls._default_dataset_path(env_name)
        )
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading D4RL dataset from path requires 'h5py'. "
                "Install it or use env-based construction."
            ) from exc

        if not os.path.exists(path):
            # Fallback: use d4rl.qlearning_dataset(env). This creates an env but
            # avoids requiring users to pre-download hdf5 files.
            env = gym.make(env_name)
            try:
                return cls(env=env, env_name=env_name, clip_to_eps=clip_to_eps, eps=eps)
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

        dones_float = np.zeros_like(rewards)
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

        ds = cls.from_arrays(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=1.0 - terminals,
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
        )
        if "antmaze" in env_name:
            ds.rewards -= 1.0
        elif any(name in env_name for name in ("halfcheetah", "walker2d", "hopper")):
            normalize(ds)
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
        """Build from arrays (e.g. a partition shard). Not a torch Dataset."""
        self = cls.__new__(cls)
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = len(observations)
        return self

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "next_observations": self.next_observations[indices],
        }

    def partition(self, rank: int, world_size: int) -> D4RLDataset:
        """Return a copied shard for this rank (data-parallel training)."""
        n = self.size
        per_rank = n // world_size
        start = rank * per_rank
        end = start + per_rank if rank < world_size - 1 else n
        indices = np.arange(start, end, dtype=np.int64)
        return D4RLDataset.from_arrays(
            self.observations[indices].copy(),
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            self.masks[indices].copy(),
            self.dones_float[indices].copy(),
            self.next_observations[indices].copy(),
        )


def normalize(dataset: D4RLDataset) -> None:
    """Normalize rewards by trajectory return range (for MuJoCo-style tasks)."""
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        return sum(rew for _, _, rew, _, _, _ in traj)

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0
