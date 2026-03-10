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

    def _check_classifier_success(self, info):
        """Check classifier-based success from info dict.

        Returns (is_success, classifier_reward) if classifier wrapper is
        active, otherwise falls back to reward-based check.
        """
        # Vectorized info: "succeed" is a numpy array of shape (num_envs,)
        succeed = info.get("succeed", None)
        clf_reward = info.get("classifier_reward", None)
        if succeed is not None:
            if isinstance(succeed, np.ndarray):
                return bool(succeed[0]), clf_reward[0] if clf_reward is not None else None
            return bool(succeed), clf_reward
        return None, None

    def run(self):
        max_steps = self.cfg.env.eval.max_episode_steps
        use_classifier = self.cfg.env.eval.get("classifier_reward_wrapper", None) is not None

        obs, _ = self.env.reset()
        success_cnt = 0
        episode_cnt = 0

        self.log_info(
            f"\n{'=' * 60}\n"
            f"  Data collection started\n"
            f"  Target successful demos: {self.num_data_episodes}\n"
            f"  Max steps per episode: {max_steps}\n"
            f"  Success criterion: {'visual classifier' if use_classifier else 'target pose (built-in)'}\n"
            f"{'=' * 60}"
        )

        current_rollout = EmbodiedRolloutResult(
            max_episode_length=max_steps,
            model_weights_id="demo_expert",
        )

        current_obs_processed = self._process_obs(obs)
        step_in_ep = 0

        # Print first episode header
        episode_cnt += 1
        self.log_info(
            f"\n{'#' * 50}\n"
            f"  Episode {episode_cnt}  "
            f"success: {success_cnt}/{self.num_data_episodes}\n"
            f"  >>> Start teleoperation <<<\n"
            f"{'#' * 50}"
        )

        progress_bar = tqdm(
            total=self.num_data_episodes, desc="Collecting Data Episodes:"
        )

        while success_cnt < self.num_data_episodes:
            action_dim = self.env.env.single_action_space.shape[0]
            action = np.zeros((1, action_dim))
            next_obs, reward, done, truncated, info = self.env.step(action)
            step_in_ep += 1

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

            if isinstance(truncated, torch.Tensor):
                trunc_tensor = truncated.bool().cpu()
            else:
                trunc_tensor = torch.tensor(truncated).bool()
            if trunc_tensor.ndim == 1:
                trunc_tensor = trunc_tensor.unsqueeze(1)

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor | trunc_tensor,
                terminations=done_tensor,
                truncations=trunc_tensor,
                forward_inputs={"action": action_tensor},
            )

            current_rollout.append_step_result(step_result)
            current_rollout.append_transitions(
                curr_obs=current_obs_processed, next_obs=next_obs_processed
            )

            obs = next_obs
            current_obs_processed = next_obs_processed

            episode_done = bool(done) or bool(truncated)
            if episode_done:
                # Determine success: prefer classifier info, fallback to reward
                clf_success, clf_reward_val = self._check_classifier_success(info)
                if clf_success is not None:
                    is_success = clf_success
                else:
                    r_val = (
                        reward[0]
                        if hasattr(reward, "__getitem__") and len(reward) > 0
                        else reward
                    )
                    if isinstance(r_val, torch.Tensor):
                        r_val = r_val.item()
                    is_success = int(r_val) > 0

                if is_success:
                    success_cnt += 1
                self.total_cnt += 1

                # Build descriptive status string
                if is_success:
                    status = "✅ SUCCESS"
                elif bool(truncated):
                    status = "⏱️  TRUNCATED (max steps)"
                else:
                    status = "❌ FAIL"

                clf_info_str = ""
                if clf_reward_val is not None:
                    clf_info_str = f"  classifier_reward={clf_reward_val:.3f}"

                self.log_info(
                    f"Episode {episode_cnt} ended [{step_in_ep}/{max_steps} steps]  "
                    f"{status}{clf_info_str}\n"
                    f"    success: {success_cnt}/{self.num_data_episodes}  "
                    f"total episodes: {self.total_cnt}"
                )

                # Save Trajectory to the 'demos' directory
                trajectory = current_rollout.to_trajectory()
                trajectory.intervene_flags = torch.ones_like(trajectory.intervene_flags)
                self.buffer.add_trajectories([trajectory])

                progress_bar.update(1)

                # Reset for next episode
                if success_cnt < self.num_data_episodes:
                    obs, _ = self.env.reset()
                    current_obs_processed = self._process_obs(obs)
                    current_rollout = EmbodiedRolloutResult(
                        max_episode_length=max_steps,
                        model_weights_id="demo_expert",
                    )
                    step_in_ep = 0
                    episode_cnt += 1
                    self.log_info(
                        f"\n{'#' * 50}\n"
                        f"  Episode {episode_cnt}  "
                        f"success: {success_cnt}/{self.num_data_episodes}\n"
                        f"  >>> Start teleoperation <<<\n"
                        f"{'#' * 50}"
                    )

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

    # If classifier_reward_wrapper.remote is set, start the classifier
    # server on a GPU node before launching the env worker.
    clf_cfg = cfg.env.eval.get("classifier_reward_wrapper", None)
    server_handle = None
    if clf_cfg is not None and clf_cfg.get("remote", False):
        import ray
        from rlinf.workers.reward.classifier_reward_server import (
            ClassifierRewardServer,
        )

        server_name = clf_cfg.get("server_name", "ClassifierRewardServer")
        server_handle = ClassifierRewardServer.options(
            name=server_name,
            num_gpus=0.05,
        ).remote(
            checkpoint_path=clf_cfg.checkpoint_path,
            image_keys=clf_cfg.get("image_keys", None),
            device=clf_cfg.get("device", "cuda"),
        )
        ray.get(server_handle.ready.remote())
        print(f"[collect_real_data] ClassifierRewardServer '{server_name}' ready on GPU node.")

    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()

    if server_handle is not None:
        import ray
        ray.kill(server_handle)


if __name__ == "__main__":
    main()
