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


import os

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.total_cnt = 0
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        # Initialize TrajectoryReplayBuffer
        # Change directory name to 'demos' as requested
        buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")

        self.buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

    def _process_obs(self, obs):
        """
        Process observations to match the format expected by EmbodiedRolloutResult.
        """
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)

            val = val.cpu()

            # Map keys: 'images' -> 'main_images', others remain
            if "images" == key:
                ret_obs["main_images"] = val.clone()  # Keep uint8
            else:
                ret_obs[key] = val.clone()

        return ret_obs

    def run(self):
        obs, _ = self.env.reset()
        success_cnt = 0
        progress_bar = tqdm(
            range(self.num_data_episodes), desc="Collecting Data Episodes:"
        )

        current_rollout = EmbodiedRolloutResult(
            max_episode_length=self.cfg.env.eval.max_episode_steps,
            model_weights_id="demo_expert",
        )

        current_obs_processed = self._process_obs(obs)

        while success_cnt < self.num_data_episodes:
            action = np.zeros((1, 6))
            next_obs, reward, done, _, info = self.env.step(action)

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = self._process_obs(next_obs)

            # --- Construct ChunkStepResult ---
            # Prepare action tensor [1, 6]
            if isinstance(action, torch.Tensor):
                action_tensor = action.float().cpu()
            else:
                action_tensor = torch.from_numpy(action).float()

            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)

            # Reward and Done [1, 1]
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward.float().cpu()
            else:
                reward_tensor = torch.tensor(reward).float()
            if reward_tensor.ndim == 1:
                reward_tensor = reward_tensor.unsqueeze(1)

            if isinstance(done, torch.Tensor):
                done_tensor = done.bool().cpu()
            else:
                done_tensor = torch.tensor(done).bool()
            if done_tensor.ndim == 1:
                done_tensor = done_tensor.unsqueeze(1)

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor,
                terminations=done_tensor,
                truncations=torch.zeros_like(done_tensor),
                forward_inputs={"action": action_tensor},
            )

            current_rollout.append_step_result(step_result)
            current_rollout.append_transitions(
                curr_obs=current_obs_processed, next_obs=next_obs_processed
            )

            obs = next_obs
            current_obs_processed = next_obs_processed

            if done:
                r_val = (
                    reward[0]
                    if hasattr(reward, "__getitem__") and len(reward) > 0
                    else reward
                )
                if isinstance(r_val, torch.Tensor):
                    r_val = r_val.item()

                success_cnt += int(r_val)
                self.total_cnt += 1
                self.log_info(
                    f"Success: {r_val}. Total: {success_cnt}/{self.num_data_episodes}"
                )

                # Save Trajectory to the 'demos' directory
                trajectory = current_rollout.to_trajectory()
                trajectory.intervene_flags = torch.ones_like(trajectory.intervene_flags)
                self.buffer.add_trajectories([trajectory])

                # Reset for next episode
                obs, _ = self.env.reset()
                current_obs_processed = self._process_obs(obs)
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                    model_weights_id="demo_expert",
                )
                progress_bar.update(1)

        self.buffer.close()
        self.log_info(
            f"Finished. Demos saved in: {os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
        )
        self.env.close()


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
