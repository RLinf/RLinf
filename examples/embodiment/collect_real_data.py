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

from rlinf.data.schema.embodied_trajectory import TrajectoryAccumulator
from rlinf.data.schema.embodied_types import (
    TrajectoryStep,
)
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.envs.real import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class DataCollector(Worker):
    # Preserve replay-buffer behavior for existing collector subclasses.
    save_demos = True

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
                streaming=bool(dc_cfg.get("streaming", False)),
            )
            preexisting_episodes = int(
                getattr(self.env, "preexisting_episode_count", 0)
            )
            # A resumed LeRobot shard may include failed episodes. Count old
            # episodes toward the success target only when the earlier run
            # saved successes exclusively; streaming saves every outcome.
            self._preexisting_success = (
                preexisting_episodes
                if dc_cfg.get("only_success", False)
                and not dc_cfg.get("streaming", False)
                else 0
            )
            if preexisting_episodes:
                self.log_info(
                    f"[resume] {preexisting_episodes} pre-existing episodes; "
                    f"{self._preexisting_success} count toward the "
                    f"{self.num_data_episodes}-success target"
                )
        else:
            self._preexisting_success = 0

        # Read from the wrapped action space so GripperCloseEnv / dual-arm all just work.
        self.action_dim = int(self.env.action_space.shape[-1])

        # ``save_demos: false`` skips the RLinf replay buffer entirely. The
        # rollout builder accumulates every frame's raw images in memory until
        # episode end, so LeRobot-only collectors (e.g. streaming YAM
        # collection) should turn this off to keep RAM flat.
        self.save_demos = bool(getattr(cfg.runner, "save_demos", True))
        if self.save_demos:
            buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
            self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")

            self.buffer = TrajectoryReplayBuffer(
                seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
                enable_cache=False,
                auto_save=True,
                auto_save_path=buffer_path,
                trajectory_format="pt",
            )
        else:
            self.buffer = None

        # Outer rate limiter for envs that don't self-pace (e.g. direct-stream).
        fps = dc_cfg.get("fps") if dc_cfg else None
        self._target_step_period = 1.0 / float(fps) if fps else None

    def _process_obs(self, obs):
        """Reshape env observations for the internal trajectory accumulator."""
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            if isinstance(val, torch.Tensor):
                processed = val.detach().cpu().clone()
            else:
                processed = val
            if key == "images":
                ret_obs["main_images"] = processed
            else:
                ret_obs[key] = processed
        return ret_obs

    @staticmethod
    def _drop_task_descriptions(obs: dict) -> dict:
        """Remove task metadata before stacking trajectory observations."""
        return {key: value for key, value in obs.items() if key != "task_descriptions"}

    def run(self):
        try:
            return self._run_collection()
        finally:
            # A recorder or camera error must not leave robot outputs open.
            try:
                self.env.close()
            finally:
                if self.save_demos and self.buffer is not None:
                    self.buffer.close()

    def _run_collection(self):
        # Seed from preexisting episodes so resume bar + stop target line up.
        success_cnt = self._preexisting_success
        if success_cnt >= self.num_data_episodes:
            self.log_info(f"[resume] target {self.num_data_episodes} already met.")
            return
        obs, _ = self.env.reset()
        progress_bar = tqdm(
            total=self.num_data_episodes,
            initial=success_cnt,
            desc="Collecting Data Episodes:",
        )

        current_rollout = (
            TrajectoryAccumulator(
                max_episode_length=self.cfg.env.eval.max_episode_steps,
            )
            if self.save_demos
            else None
        )
        current_obs_processed = self._process_obs(obs) if self.save_demos else None

        while success_cnt < self.num_data_episodes:
            iter_start = time.perf_counter()
            # Teleop wrapper overrides this via info["intervene_action"].
            action = np.zeros((1, self.action_dim))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # ``kb_phase is None`` ⇒ no keyboard wrapper attached → upstream "record every step".
            kb_event = info["keyboard_event"][0] if "keyboard_event" in info else None
            kb_phase = info["keyboard_phase"][0] if "keyboard_phase" in info else None
            record_reset = bool(np.asarray(info.get("record_reset", False)).any())
            pre_record = bool(np.asarray(info.get("pre_record", False)).any())
            if kb_event:
                self.log_info(f"[keyboard] {kb_event}")

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = (
                self._process_obs(next_obs) if self.save_demos else None
            )

            terminated_tensor = terminated.unsqueeze(1)
            truncated_tensor = truncated.unsqueeze(1)
            done_tensor = terminated_tensor | truncated_tensor
            done = bool(done_tensor.any().item())

            if self.save_demos:
                action_tensor = torch.as_tensor(action, dtype=torch.float32)
                reward_tensor = reward.float().unsqueeze(1)

                step_result = TrajectoryStep(
                    actions=action_tensor,
                    rewards=reward_tensor,
                    dones=done_tensor,
                    terminations=terminated_tensor,
                    truncations=truncated_tensor,
                    forward_inputs={"action": action_tensor},
                    curr_obs=self._drop_task_descriptions(current_obs_processed),
                    next_obs=self._drop_task_descriptions(next_obs_processed),
                )

                # Rebuild rollout on rec-start or abort; ``restart`` kept for older wrappers.
                if record_reset or kb_event in ("start", "restart", "abort"):
                    current_rollout = TrajectoryAccumulator(
                        max_episode_length=self.cfg.env.eval.max_episode_steps,
                    )
                # Match CollectEpisode: the start/abort transition establishes the
                # next observation as the new initial frame; it is not recorded.
                if not record_reset and not pre_record and kb_phase in (None, "rec"):
                    current_rollout.append(step_result)

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

                if bool(np.asarray(info.get("recording_invalid", False)).any()):
                    save_episode = False
                    self.log_info(
                        "Recording overflow: incomplete episode excluded from success count."
                    )

                discarded = bool(np.asarray(info.get("episode_discarded", False)).any())
                if discarded:
                    save_episode = False

                if save_episode:
                    success_cnt += 1

                    self.log_info(
                        f"Success (reward={r_val}, manual_done={manual_done}). "
                        f"Total: {success_cnt}/{self.num_data_episodes}"
                    )

                    if self.save_demos:
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
                if self.save_demos:
                    current_obs_processed = self._process_obs(obs)
                    current_rollout = TrajectoryAccumulator(
                        max_episode_length=self.cfg.env.eval.max_episode_steps,
                    )

            # Pin loop period; on ``done`` env.reset usually exceeds it → sleep_for≤0 no-ops.
            if self._target_step_period is not None:
                elapsed = time.perf_counter() - iter_start
                sleep_for = self._target_step_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        if self.save_demos:
            self.log_info(
                f"Finished. Demos saved in: {os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
            )


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
