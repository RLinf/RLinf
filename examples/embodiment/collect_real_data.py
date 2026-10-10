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

"""Collect real-robot demonstrations selected by ``data_collection.export_format``.

Teleoperation drives one ``RealWorldEnv``. ``export_format`` is one of
``replay_buffer``, ``pickle_episode``, and ``lerobot_dataset``, or a list.
``replay_buffer`` may be combined with one episode format. Omitting the key
selects ``replay_buffer`` and ``lerobot_dataset``.

* ``replay_buffer`` writes successful episodes as ``TrajectoryReplayBuffer``
  ``.pt`` files under ``{save_dir}/demo_buffer``. Real-robot RLPD
  loads that directory through ``algorithm.demo_buffer.load_path`` (Franka peg
  insertion, DoSW1). Raw images stay in memory until the episode ends.
* ``lerobot_dataset`` writes ``{save_dir}/lerobot_dataset``.
  OpenPI SFT, HG-DAgger, and other LeRobot imitation pipelines read these shards.
  ``streaming: true`` writes each frame immediately, so RAM does not grow with
  episode length. Dual-YAM PICO collection uses this format alone.
* ``pickle_episode`` writes one ``.pkl`` per episode under
  ``{save_dir}/pickle_episode``. Reward-model preprocessing reads them:
  ``examples/reward/preprocess_reward_dataset.py`` builds the ResNet binary
  dataset, and ``examples/reward/vlm_trend/`` builds VLM trend labels. RLPD and
  LeRobot training do not read these files.
"""

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

_EXPORT_FORMATS = frozenset({"replay_buffer", "pickle_episode", "lerobot_dataset"})
# CollectEpisode still takes these two argument values.
_COLLECT_EPISODE_FORMAT = {
    "pickle_episode": "pickle",
    "lerobot_dataset": "lerobot",
}


class DataCollector(Worker):
    """Step a real robot and write the enabled demonstration stores."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.total_cnt = 0
        self.replay_buffer = None
        self._export_formats = self._selected_export_formats(cfg)
        override_cfg = cfg.env.eval.get("override_cfg", {})
        self.manual_episode_control_only = bool(
            override_cfg.get("manual_episode_control_only", False)
        )
        self.env = self._build_env(cfg)
        # Read from the wrapped action space so GripperCloseEnv / dual-arm all just work.
        self.action_dim = int(self.env.action_space.shape[-1])
        self.replay_buffer = self._build_replay_buffer(cfg)

        dc_cfg = cfg.env.eval.get("data_collection")
        fps = dc_cfg.get("fps") if dc_cfg else None
        self._target_step_period = 1.0 / float(fps) if fps else None

    def _build_env(self, cfg):
        """Create the real env and attach episode export when configured."""
        env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )
        dc_cfg = cfg.env.eval.get("data_collection")
        episode_format = self._episode_export_format()
        if (
            not dc_cfg
            or not getattr(dc_cfg, "enabled", False)
            or episode_format is None
        ):
            self._preexisting_success = 0
            return env

        from rlinf.envs.wrappers import CollectEpisode

        env = CollectEpisode(
            env,
            save_dir=self._save_root(cfg),
            export_format=episode_format,
            robot_type=dc_cfg.get("robot_type", "panda"),
            fps=dc_cfg.get("fps", 10),
            only_success=dc_cfg.get("only_success", False),
            finalize_interval=dc_cfg.get("finalize_interval", 100),
            resume=bool(dc_cfg.get("resume", False)),
            streaming=bool(dc_cfg.get("streaming", False)),
        )
        preexisting_episodes = int(getattr(env, "preexisting_episode_count", 0))
        # A resumed LeRobot shard may include failed episodes. Count old
        # episodes toward the success target only when the earlier run
        # saved successes exclusively; streaming saves every outcome.
        only_success = bool(dc_cfg.get("only_success", False))
        streaming = bool(dc_cfg.get("streaming", False))
        self._preexisting_success = (
            preexisting_episodes if only_success and not streaming else 0
        )
        if preexisting_episodes:
            self.log_info(
                f"[resume] {preexisting_episodes} pre-existing episodes; "
                f"{self._preexisting_success} count toward the "
                f"{self.num_data_episodes}-success target"
            )
        return env

    def _data_collection_cfg(self, cfg):
        eval_cfg = getattr(cfg.env, "eval", None)
        if eval_cfg is None or not hasattr(eval_cfg, "get"):
            return None
        return eval_cfg.get("data_collection")

    def _selected_export_formats(self, cfg) -> frozenset[str]:
        """Resolve ``export_format`` to the stores this run writes."""
        dc_cfg = self._data_collection_cfg(cfg)
        if dc_cfg is None:
            return frozenset({"replay_buffer"})
        raw = dc_cfg.get("export_format", ["replay_buffer", "lerobot_dataset"])
        values = [raw] if isinstance(raw, str) else list(raw)
        selected: list[str] = []
        for value in values:
            name = str(value)
            if name not in _EXPORT_FORMATS:
                known = ", ".join(sorted(_EXPORT_FORMATS))
                raise ValueError(
                    f"Unsupported export_format={value!r}. Expected one or more of: {known}."
                )
            if name not in selected:
                selected.append(name)
        episode_formats = [name for name in selected if name in _COLLECT_EPISODE_FORMAT]
        if len(episode_formats) > 1:
            raise ValueError(
                "export_format can include replay_buffer plus only one of "
                "pickle_episode or lerobot_dataset."
            )
        return frozenset(selected)

    def _episode_export_format(self) -> str | None:
        """Return the CollectEpisode format, or ``None`` when it is not selected."""
        for name, collect_format in _COLLECT_EPISODE_FORMAT.items():
            if name in self._export_formats:
                return collect_format
        return None

    def _configured_save_dir(self, cfg) -> str | None:
        dc_cfg = self._data_collection_cfg(cfg)
        if dc_cfg is None or not hasattr(dc_cfg, "get"):
            return None
        save_dir = dc_cfg.get("save_dir", None)
        return str(save_dir) if save_dir else None

    def _save_root(self, cfg) -> str:
        return self._configured_save_dir(cfg) or cfg.runner.logger.log_path

    def _build_replay_buffer(self, cfg):
        """Open the on-disk replay buffer, or skip it for LeRobot-only runs."""
        if "replay_buffer" not in self._export_formats:
            self.log_info(
                "Replay buffer disabled. Episodes are not stored as Trajectory files."
            )
            return None

        buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=os.path.join(self._save_root(cfg), "demo_buffer"),
            trajectory_format="pt",
        )
        self.log_info(f"Initializing replay buffer at: {buffer.auto_save_path}")
        return buffer

    def _process_obs(self, obs):
        """Copy env observations into CPU tensors for the replay accumulator."""
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

    def _new_replay_rollout(self):
        return TrajectoryAccumulator(
            max_episode_length=self.cfg.env.eval.max_episode_steps,
        )

    def _start_replay_rollout(self, obs):
        """Return a new in-memory episode and its first observation."""
        if self.replay_buffer is None:
            return None, None
        return self._new_replay_rollout(), self._process_obs(obs)

    @staticmethod
    def _recording_signals(info):
        """Read keyboard and recorder flags that gate replay-buffer steps."""
        kb_event = info["keyboard_event"][0] if "keyboard_event" in info else None
        kb_phase = info["keyboard_phase"][0] if "keyboard_phase" in info else None
        record_reset = bool(np.asarray(info.get("record_reset", False)).any())
        pre_record = bool(np.asarray(info.get("pre_record", False)).any())
        return kb_event, kb_phase, record_reset, pre_record

    def _append_replay_step(
        self,
        rollout,
        replay_obs,
        *,
        action,
        reward,
        terminated_tensor,
        truncated_tensor,
        done_tensor,
        next_obs,
        record_reset,
        pre_record,
        kb_event,
        kb_phase,
    ):
        """Append one transition and return the rollout plus the next observation."""
        if self.replay_buffer is None:
            return None, None

        next_obs_processed = self._process_obs(next_obs)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        step_result = TrajectoryStep(
            actions=action_tensor,
            rewards=reward.float().unsqueeze(1),
            dones=done_tensor,
            terminations=terminated_tensor,
            truncations=truncated_tensor,
            forward_inputs={"action": action_tensor},
            curr_obs=self._drop_task_descriptions(replay_obs),
            next_obs=self._drop_task_descriptions(next_obs_processed),
        )

        # Rebuild rollout on rec-start or abort; ``restart`` kept for older wrappers.
        if record_reset or kb_event in ("start", "restart", "abort"):
            rollout = self._new_replay_rollout()
        # Match CollectEpisode: the start/abort transition establishes the
        # next observation as the new initial frame; it is not recorded.
        if not record_reset and not pre_record and kb_phase in (None, "rec"):
            rollout.append(step_result)

        return rollout, next_obs_processed

    def _commit_replay_episode(self, rollout):
        """Mark one successful episode as expert data and append it."""
        if self.replay_buffer is None:
            return
        trajectory = rollout.to_trajectory()
        trajectory.intervene_flags = torch.ones_like(trajectory.intervene_flags)
        self.replay_buffer.add_trajectories([trajectory])

    def _reward_value(self, reward):
        r_val = (
            reward[0] if hasattr(reward, "__getitem__") and len(reward) > 0 else reward
        )
        if isinstance(r_val, torch.Tensor):
            r_val = r_val.item()
        return r_val

    @staticmethod
    def _manual_done(info) -> bool:
        if "manual_done" not in info:
            return False
        manual_done = info["manual_done"]
        if hasattr(manual_done, "__getitem__") and len(manual_done) > 0:
            return bool(manual_done[0])
        return bool(manual_done)

    def _should_save_episode(self, reward, info):
        """Decide whether a finished episode counts toward the success target."""
        r_val = self._reward_value(reward)
        manual_done = self._manual_done(info)
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
        if bool(np.asarray(info.get("episode_discarded", False)).any()):
            save_episode = False
        return save_episode, r_val, manual_done

    def _sleep_to_period(self, iter_start: float) -> None:
        """Pin the loop period. A slow ``reset`` makes the sleep a no-op."""
        if self._target_step_period is None:
            return
        sleep_for = self._target_step_period - (time.perf_counter() - iter_start)
        if sleep_for > 0:
            time.sleep(sleep_for)

    def _log_finished(self):
        if self.replay_buffer is None:
            return
        self.log_info(
            f"Finished. Replay buffer saved in: {self.replay_buffer.auto_save_path}"
        )

    def run(self):
        try:
            return self._run_collection()
        finally:
            # A recorder or camera error must not leave robot outputs open.
            try:
                self.env.close()
            finally:
                if self.replay_buffer is not None:
                    self.replay_buffer.close()

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
        replay_rollout, replay_obs = self._start_replay_rollout(obs)

        while success_cnt < self.num_data_episodes:
            iter_start = time.perf_counter()
            # Teleop wrapper overrides this via info["intervene_action"].
            action = np.zeros((1, self.action_dim))
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            kb_event, kb_phase, record_reset, pre_record = self._recording_signals(info)
            if kb_event:
                self.log_info(f"[keyboard] {kb_event}")
            if "intervene_action" in info:
                action = info["intervene_action"]

            terminated_tensor = terminated.unsqueeze(1)
            truncated_tensor = truncated.unsqueeze(1)
            done_tensor = terminated_tensor | truncated_tensor
            done = bool(done_tensor.any().item())
            replay_rollout, replay_obs = self._append_replay_step(
                replay_rollout,
                replay_obs,
                action=action,
                reward=reward,
                terminated_tensor=terminated_tensor,
                truncated_tensor=truncated_tensor,
                done_tensor=done_tensor,
                next_obs=next_obs,
                record_reset=record_reset,
                pre_record=pre_record,
                kb_event=kb_event,
                kb_phase=kb_phase,
            )

            if done:
                save_episode, r_val, manual_done = self._should_save_episode(
                    reward, info
                )
                if save_episode:
                    success_cnt += 1
                    self.log_info(
                        f"Success (reward={r_val}, manual_done={manual_done}). "
                        f"Total: {success_cnt}/{self.num_data_episodes}"
                    )
                    self._commit_replay_episode(replay_rollout)
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
                replay_rollout, replay_obs = self._start_replay_rollout(obs)

            self._sleep_to_period(iter_start)

        self._log_finished()


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
