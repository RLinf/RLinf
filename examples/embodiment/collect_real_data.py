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
import time

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
        override_cfg = cfg.env.eval.get("override_cfg", {})
        self.manual_episode_control_only = bool(
            override_cfg.get("manual_episode_control_only", False)
        )
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        dc_cfg = cfg.env.eval.get("data_collection")
        if dc_cfg and getattr(dc_cfg, "enabled", False):
            from rlinf.envs.wrappers import CollectEpisode

            self.env = CollectEpisode(
                self.env,
                save_dir=dc_cfg.save_dir,
                export_format=dc_cfg.get("export_format", "pickle"),
                robot_type=dc_cfg.get("robot_type", "panda"),
                fps=dc_cfg.get("fps", 10),
                only_success=dc_cfg.get("only_success", False),
                finalize_interval=dc_cfg.get("finalize_interval", 100),
                resume=bool(dc_cfg.get("resume", False)),
            )
            self._preexisting_success = int(
                getattr(self.env, "preexisting_episode_count", 0)
            )
            if self._preexisting_success:
                self.log_info(
                    f"[resume] {self._preexisting_success} pre-existing episodes; "
                    f"continuing toward {self.num_data_episodes}"
                )
        else:
            self._preexisting_success = 0

        # Read from the wrapped action space so GripperCloseEnv / dual-arm all just work.
        self.action_dim = int(self.env.action_space.shape[-1])

        buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")

        self.buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

        # Outer rate limiter for envs that don't self-pace (e.g. direct-stream).
        fps = dc_cfg.get("fps") if dc_cfg else None
        self._target_step_period = 1.0 / float(fps) if fps else None

    def _process_obs(self, obs):
        """Reshape env obs into the dict EmbodiedRolloutResult expects."""
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            # Below is for not crashed by 
            # [extra_view_image_names/record_task_description=True].
            if not isinstance(val, torch.Tensor):
                continue
            val = val.cpu()
            if key == "images":
                ret_obs["main_images"] = val.clone()
            else:
                ret_obs[key] = val.clone()
        return ret_obs

    def run(self):
        obs, _ = self.env.reset()
        # Seed from preexisting episodes so resume bar + stop target line up.
        success_cnt = self._preexisting_success
        if success_cnt >= self.num_data_episodes:
            self.log_info(f"[resume] target {self.num_data_episodes} already met.")
            self.env.close()
            return
        progress_bar = tqdm(
            total=self.num_data_episodes,
            initial=success_cnt,
            desc="Collecting Data Episodes:",
        )

        current_rollout = EmbodiedRolloutResult(
            max_episode_length=self.cfg.env.eval.max_episode_steps,
        )

        current_obs_processed = self._process_obs(obs)

        while success_cnt < self.num_data_episodes:
            iter_start = time.perf_counter()
            # Teleop wrapper overrides this via info["intervene_action"].
            action = np.zeros((1, self.action_dim))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # ``kb_phase is None`` ⇒ no keyboard wrapper attached → upstream "record every step".
            kb_event = info["keyboard_event"][0] if "keyboard_event" in info else None
            kb_phase = info["keyboard_phase"][0] if "keyboard_phase" in info else None
            if kb_event:
                self.log_info(f"[keyboard] {kb_event}")

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = self._process_obs(next_obs)

            terminated_tensor = terminated.unsqueeze(1)
            truncated_tensor = truncated.unsqueeze(1)
            done_tensor = terminated_tensor | truncated_tensor
            done = bool(done_tensor.any().item())

            action_tensor = torch.as_tensor(action, dtype=torch.float32)
            reward_tensor = reward.float().unsqueeze(1)

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor,
                terminations=terminated_tensor,
                truncations=truncated_tensor,
                forward_inputs={"action": action_tensor},
            )

            # Rebuild rollout on rec-start or abort; ``restart`` kept for older wrappers.
            if kb_event in ("start", "restart", "abort"):
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                )
            if kb_phase in (None, "rec"):
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

                manual_done = False
                if "manual_done" in info:
                    md = info["manual_done"]
                    if hasattr(md, "__getitem__") and len(md) > 0:
                        manual_done = bool(md[0])
                    else:
                        manual_done = bool(md)

                self.total_cnt += 1
                if self.manual_episode_control_only:
                    save_episode = bool(manual_done)
                else:
                    save_episode = bool(r_val >= 0.5 or manual_done)

                if save_episode:
                    success_cnt += 1

                    self.log_info(
                        f"Success (reward={r_val}, manual_done={manual_done}). "
                        f"Total: {success_cnt}/{self.num_data_episodes}"
                    )

                    trajectory = current_rollout.to_trajectory()
                    trajectory.intervene_flags = torch.ones_like(
                        trajectory.intervene_flags
                    )
                    self.buffer.add_trajectories([trajectory])

                    progress_bar.update(1)
                else:
                    self.log_info(
                        f"Episode ended (reward={r_val:.2f}). "
                        f"Discarded. Total success: {success_cnt}/{self.num_data_episodes}"
                    )

                reset_options = None
                if success_cnt >= self.num_data_episodes:
                    reset_options = {"skip_wait_for_start": True}
                obs, _ = self.env.reset(options=reset_options)
                current_obs_processed = self._process_obs(obs)
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                )

            # Pin loop period; on ``done`` env.reset usually exceeds it → sleep_for≤0 no-ops.
            if self._target_step_period is not None:
                elapsed = time.perf_counter() - iter_start
                sleep_for = self._target_step_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

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
