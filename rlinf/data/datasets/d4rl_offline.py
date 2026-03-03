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
import time

import gym
import numpy as np
import tqdm
from gym.spaces import Box, Dict as SpaceDict

try:
    import d4rl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "d4rl is required for D4RL offline datasets. Please install d4rl."
    ) from exc


def _compat_reset(env: gym.Env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _compat_step(env: gym.Env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        if truncated:
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info
    return out


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, done, info = _compat_step(self.env, action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}
        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self, "get_normalized_score"):
                info["episode"]["return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )
        return observation, reward, done, info

    def reset(self):
        self._reset_stats()
        return _compat_reset(self.env)


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high, obs_space.shape)
        elif isinstance(self.observation_space, SpaceDict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = SpaceDict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation):
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        if isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
        return observation


def split_into_trajectories(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    dones_float: np.ndarray,
    next_observations: np.ndarray,
):
    trajs = [[]]
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


class Dataset:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "next_observations": self.next_observations[indices],
        }


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
        dones_float = np.zeros_like(dataset["rewards"])
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1
        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


def normalize(dataset: D4RLDataset):
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


def make_d4rl_env_and_dataset(env_name: str, seed: int) -> tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    env = SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)
    if "antmaze" in env_name:
        dataset.rewards -= 1.0
    elif any(name in env_name for name in ("halfcheetah", "walker2d", "hopper")):
        normalize(dataset)
    return env, dataset


def evaluate_policy(agent, env: gym.Env, num_episodes: int) -> dict[str, float]:
    stats = {"return": [], "length": []}
    for _ in range(num_episodes):
        observation, done = _compat_reset(env), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = _compat_step(env, action)
        for key in stats:
            stats[key].append(info["episode"][key])
    return {k: float(np.mean(v)) for k, v in stats.items()}
