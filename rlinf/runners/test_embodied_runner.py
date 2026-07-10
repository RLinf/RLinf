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

import time

from rlinf.scheduler import WorkerGroupFuncResult as Handle

from .embodied_runner import EmbodiedRunner


class TestEmbodiedRunner(EmbodiedRunner):
    """Embodied PPO runner that receives rollout batches via TrajectoryWorker."""

    def __init__(self, *args, trajectory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = trajectory
        self.trajectory_loop_handle: Handle | None = None
        self.reward_loop_handle: Handle | None = None

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        trajectory_handle = self.trajectory.init_worker()

        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        trajectory_handle.wait()
        self.actor.init_worker().wait()
        self.trajectory_loop_handle = self.trajectory.run_loop()
        if self.reward is not None:
            self.reward_loop_handle = self.reward.compute_trajectory_rewards()

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=None,
                        actor_channel=None,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    reward_handle = None
                    self.actor.recv_trajectory_worker_trajectories().wait()
                    rollout_handle.wait()

                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                actor_training_handle: Handle = self.actor.run_training()
                actor_training_metrics = actor_training_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        if self.trajectory is not None:
            self.trajectory.stop().wait()
            if self.trajectory_loop_handle is not None:
                self.trajectory_loop_handle.wait()
        if self.reward is not None:
            self.reward.stop().wait()
        self._finish_run()
