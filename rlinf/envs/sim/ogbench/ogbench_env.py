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


import copy

import numpy as np


class OGBenchEnv:
    """Keep terminated and truncated separate for bootstrap masks.

    This adapter exposes Gymnasium's single-env API. QAMRunner deliberately uses
    it in a colocated evaluation/collection component, not the vector EnvWorker.
    """

    def __init__(self, cfg, **kwargs):
        import ogbench

        self.env = ogbench.make_env_and_datasets(cfg.task_name, env_only=True)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(np.clip(action, -1, 1))

    def close(self):
        self.env.close()

    def state_dict(self):
        """Save AntMaze physics, goal, RNG and wrapper progress explicitly."""
        import mujoco

        raw = self.env.unwrapped
        if not hasattr(raw, "cur_goal_xy"):
            raise NotImplementedError(
                "Exact online resume is currently validated for AntMaze only"
            )
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(raw.model, spec))
        mujoco.mj_getState(raw.model, raw.data, state, spec)
        wrappers, wrapper = [], self.env
        while wrapper is not raw:
            keys = (
                "_elapsed_steps",
                "_has_reset",
                "checked_reset",
                "checked_step",
                "checked_render",
                "close_called",
            )
            wrappers.append(
                {k: getattr(wrapper, k) for k in keys if k in vars(wrapper)}
            )
            wrapper = wrapper.env
        return {
            "physics": state,
            "rng": copy.deepcopy(raw.np_random.bit_generator.state),
            "task": {
                k: copy.deepcopy(getattr(raw, k))
                for k in ("cur_task_id", "cur_task_info", "cur_goal_xy")
            },
            "wrappers": wrappers,
        }

    def load_state_dict(self, state):
        """Restore the same AntMaze continuation without pickling MuJoCo."""
        import mujoco

        self.reset(seed=0)
        raw = self.env.unwrapped
        for k, v in state["task"].items():
            setattr(raw, k, copy.deepcopy(v))
        raw.np_random.bit_generator.state = copy.deepcopy(state["rng"])
        mujoco.mj_setState(
            raw.model, raw.data, state["physics"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(raw.model, raw.data)
        wrapper = self.env
        for fields in state["wrappers"]:
            for k, v in fields.items():
                setattr(wrapper, k, v)
            wrapper = wrapper.env

    def evaluate(self, policy, episodes, seed):
        """Evaluate stochastic flow actions; report success in [0, 1]."""
        returns, lengths, successes = [], [], []
        for episode in range(episodes):
            obs, _ = self.reset(seed=seed + episode)
            done, total, length, success = False, 0.0, 0, 0.0
            while not done:
                chunks, _ = policy.predict_action_batch(observations=obs[None])
                for action in chunks[0]:
                    obs, reward, terminated, truncated, info = self.step(action)
                    total += float(reward)
                    length += 1
                    success = max(success, float(info.get("success", 0)))
                    done = terminated or truncated
                    if done:
                        break
            returns.append(total)
            lengths.append(length)
            successes.append(success)
        return {
            "success": float(np.mean(successes)),
            "episode_return": float(np.mean(returns)),
            "episode_length": float(np.mean(lengths)),
            "eval_episodes": episodes,
        }
