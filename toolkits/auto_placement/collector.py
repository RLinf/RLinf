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

from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle


class EmbodiedCostCollector(EmbodiedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_step: int = OmegaConf.select(self.cfg, "runner.max_steps")
        self.env_cost_list = []
        self.rollout_cost_list = []
        self.actor_cost_list = []
        self.env_cost = 0.0
        self.rollout_cost = 0.0
        self.actor_cost = 0.0

    def collect(self):
        for step in range(self.profile_step):
            self.actor.set_global_step(step)
            self.rollout.set_global_step(step)

            # sync weight if needed
            if step % self.weight_sync_interval == 0:
                self.update_rollout_weights()

            # generate rollouts
            env_handle: Handle = self.env.interact(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
            rollout_handle: Handle = self.rollout.generate(
                input_channel=self.env_channel,
                output_channel=self.rollout_channel,
                actor_channel=self.actor_channel,
            )
            self.actor.recv_rollout_trajectories(
                input_channel=self.actor_channel
            ).wait()
            rollout_handle.wait()

            # compute advantages and returns
            self.actor.compute_advantages_and_returns().wait()

            # actor training
            actor_training_handle: Handle = self.actor.run_training()
            actor_training_handle.wait()

            # collect original time profile data (from the consume_durations of each handle)
            env_durations = env_handle.consume_durations()
            rollout_durations = rollout_handle.consume_durations()
            actor_durations = actor_training_handle.consume_durations()

            self.env_cost_list.append(env_durations["interact"])
            self.rollout_cost_list.append(rollout_durations["generate_one_epoch"])
            self.actor_cost_list.append(actor_durations["run_training"])

        self.env_cost = sum(self.env_cost_list) / len(self.env_cost_list)
        self.rollout_cost = sum(self.rollout_cost_list) / len(self.rollout_cost_list)
        self.actor_cost = sum(self.actor_cost_list) / len(self.actor_cost_list)
