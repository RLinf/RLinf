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


import numpy as np
import torch


class OGBenchEnv:
    """Expose the OGBench Gymnasium environment to the evaluation adapter."""

    def __init__(self, cfg, **kwargs):
        import ogbench

        kwargs = {"dataset_dir": cfg.dataset_dir} if cfg.get("dataset_dir") else {}
        self.env = ogbench.make_env_and_datasets(cfg.task_name, env_only=True, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(np.clip(action, -1, 1))

    def close(self):
        self.env.close()


class OGBenchEvalEnv:
    """One OGBench episode per RLinf EnvWorker evaluation epoch."""

    def __init__(self, cfg, num_envs=1, seed_offset=0, **kwargs):
        if num_envs != 1:
            raise ValueError("OGBench evaluation currently requires one environment")
        self.env = OGBenchEnv(cfg)
        self.num_envs = num_envs
        self.seed = (
            int(cfg.seed) + int(cfg.get("evaluation_seed_offset", 0)) + seed_offset
        )
        self.episode_index = 0
        self.is_start = True
        self.elapsed_steps = np.zeros(1, dtype=np.int32)
        self.info_logging_keys = []
        self.episode_return = 0.0
        self.success = 0.0

    def reset(self, **kwargs):
        obs, _ = self.env.reset(seed=self.seed + self.episode_index)
        self.episode_index += 1
        self.elapsed_steps[:] = 0
        self.episode_return = 0.0
        self.success = 0.0
        self.is_start = False
        return {"states": torch.as_tensor(obs[None], dtype=torch.float32)}, {}

    def chunk_step(self, chunk_actions):
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if actions.shape != (1, 1, self.env.action_space.shape[0]):
            raise ValueError(f"Expected one OGBench action chunk, got {actions.shape}")
        obs, reward, terminated, truncated, info = self.env.step(actions[0, 0])
        self.elapsed_steps += 1
        self.episode_return += float(reward)
        self.success = max(self.success, float(info.get("success", 0)))
        done = terminated or truncated
        infos = (
            {
                "episode": {
                    "success": torch.tensor([self.success], dtype=torch.float32),
                    "return": torch.tensor([self.episode_return], dtype=torch.float32),
                    "episode_len": torch.tensor(
                        self.elapsed_steps.copy(), dtype=torch.float32
                    ),
                }
            }
            if done
            else {}
        )
        return (
            [{"states": torch.as_tensor(obs[None], dtype=torch.float32)}],
            torch.tensor([[reward]], dtype=torch.float32),
            torch.tensor([[terminated]], dtype=torch.bool),
            torch.tensor([[truncated]], dtype=torch.bool),
            [infos],
        )

    def close(self):
        self.env.close()

    def update_reset_state_ids(self):
        """The task sampler advances on each reset; no extra state is needed."""
