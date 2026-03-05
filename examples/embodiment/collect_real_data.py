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
import pickle as pkl

import hydra
import numpy as np
import torch
from tqdm import tqdm

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

        self.data_list = []

    def _extract_obs(self, obs):
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)
        ret_obs = {}
        for key in obs:
            ret_obs[key] = obs[key][0]
        return ret_obs

    def run(self):
        obs, _ = self.env.reset()
        success_cnt = 0
        progress_bar = tqdm(
            range(self.num_data_episodes), desc="Collecting Data Episodes:"
        )
        while success_cnt < self.num_data_episodes:
            action = np.zeros((1, 6))
            next_obs, reward, done, _, info = self.env.step(action)
            if "intervene_action" in info:
                action = info["intervene_action"]

            # Handle vector env
            single_obs = self._extract_obs(obs)
            single_next_obs = self._extract_obs(next_obs)
            single_action = action[0]
            single_reward = reward[0]
            single_done = done[0]

            # Handle chunk
            chunk_done = single_done[None, ...]
            chunk_reward = single_reward[None, ...]

            transition = copy.deepcopy(
                {
                    "obs": single_obs,
                    "next_obs": single_next_obs,
                }
            )
            data = copy.deepcopy(
                {
                    "transitions": transition,
                    "action": single_action,
                    "rewards": chunk_reward,
                    "dones": chunk_done,
                    "terminations": chunk_done,
                    "truncations": torch.zeros_like(chunk_done),
                }
            )
            self.data_list.append(data)

            obs = next_obs

            if done:
                success_cnt += reward
                self.total_cnt += 1
                self.log_info(
                    f"{reward}\tGot {success_cnt} successes of {self.total_cnt} trials. {self.num_data_episodes} successes needed."
                )
                obs, _ = self.env.reset()
                progress_bar.update(1)
            else:
                self.log_info("Done is False, continue current episode.")

        save_file_path = os.path.join(self.cfg.runner.logger.log_path, "data.pkl")
        with open(save_file_path, "wb") as f:
            pkl.dump(self.data_list, f)
            self.log_info(
                f"Saved {self.num_data_episodes} demos with {len(self.data_list)} samples to {save_file_path}"
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
