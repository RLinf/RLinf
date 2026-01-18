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

import logging
import os
from typing import TYPE_CHECKING, Optional, Union

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import RewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    """Runner for embodied RL training with optional reward model support."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor", "EmbodiedSACFSDPPolicy", "AsyncEmbodiedSACFSDPPolicy"
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        demo_buffer: Optional[SACReplayBuffer] = None,
        critic=None,
        reward: Optional["RewardWorker"] = None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.demo_buffer = demo_buffer
        self.critic = critic
        self.reward = reward

        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")

        if self.demo_buffer is not None:
            self.demo_data_channel = Channel.create("DemoBufferChannel")

        if self.reward is not None:
            self.reward_input_channel = Channel.create("RewardInput")
            self.reward_output_channel = Channel.create("RewardOutput")

        self.run_timer = run_timer
        self.consumed_samples = 0
        self.global_step = 0

        self.set_max_steps()
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        if self.reward is not None:
            self.reward.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()

        if self.reward is not None:
            reward_checkpoint_path = os.path.join(resume_dir, "reward")
            if os.path.exists(reward_checkpoint_path):
                self.reward.load_checkpoint(reward_checkpoint_path).wait()

        self.global_step = int(resume_dir.split("global_step_")[-1])

    def send_demo_buffer(self):
        if self.demo_buffer is not None:
            sub_demo_buffer_ls = self.demo_buffer.split_to_dict(self.actor._world_size)
            for sub_demo_buffer in sub_demo_buffer_ls:
                self.demo_data_channel.put(sub_demo_buffer, async_op=True)
            self.actor.recv_demo_data(self.demo_data_channel).wait()

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def compute_rewards_with_model(self):
        """Compute rewards using the reward model via channel communication.

        Data flow:
        1. Get main_images from env worker (collected during rollout)
        2. Flatten and send to reward_input_channel
        3. Reward worker computes rewards
        4. Receive from reward_output_channel
        5. Reshape and update actor's rollout_batch rewards

        Supports two modes (configured via reward.reward_mode):
        - "per_step": Compute reward for every frame (default)
        - "terminal": Only compute reward at episode end (last frame)
        """
        if self.reward is None:
            return None

        # Get main_images from env worker (collected during rollout)
        images_result = self.env.get_rollout_images().wait()
        images = images_result[0] if images_result else None
        if images is None:
            return None

        # images shape: (n_steps, n_envs, H, W, C)
        n_steps, n_envs = images.shape[0], images.shape[1]

        # Check reward mode
        reward_mode = self.cfg.reward.get("reward_mode", "per_step")

        if reward_mode == "terminal":
            # Terminal mode: only use the last frame of each episode
            last_frame_images = images[-1]  # Shape: (n_envs, H, W, C)

            # Send only last frame images to reward worker
            self.reward_input_channel.put(
                {"main_images": last_frame_images}, async_op=False
            )

            # Start reward computation
            reward_handle: Handle = self.reward.compute_rewards(
                input_channel=self.reward_input_channel,
                output_channel=self.reward_output_channel,
            )

            # Wait for reward computation and get results
            reward_handle.wait()
            reward_data = self.reward_output_channel.get()

            if reward_data is not None and "rewards" in reward_data:
                terminal_rewards = reward_data[
                    "rewards"
                ]  # Shape: (n_envs,), prob in [0,1]

                # Asymmetric reward mapping: success gets big reward, fail gets small penalty
                success_reward = self.cfg.reward.get("success_reward", 1.0)
                fail_reward = self.cfg.reward.get("fail_reward", 0.0)
                threshold = self.cfg.reward.get("reward_threshold", 0.5)

                # Binary classification: prob > threshold = success, else fail
                is_success = terminal_rewards > threshold
                mapped_rewards = torch.where(
                    is_success,
                    torch.tensor(success_reward, device=terminal_rewards.device),
                    torch.tensor(fail_reward, device=terminal_rewards.device),
                )

                # Create full reward tensor: zeros except for the last step
                rewards = torch.zeros(n_steps, n_envs, 1)
                rewards[-1, :, 0] = mapped_rewards  # Only last step gets reward

                # Update actor's rollout_batch with computed rewards (remote call)
                self.actor.update_rewards(rewards).wait()
        else:
            # Per-step mode: compute reward for every frame (original behavior)
            flat_images = images.view(n_steps * n_envs, *images.shape[2:])

            # Send images to reward worker via channel
            self.reward_input_channel.put({"main_images": flat_images}, async_op=False)

            # Start reward computation
            reward_handle: Handle = self.reward.compute_rewards(
                input_channel=self.reward_input_channel,
                output_channel=self.reward_output_channel,
            )

            # Wait for reward computation and get results
            reward_handle.wait()
            reward_data = self.reward_output_channel.get()

            if reward_data is not None and "rewards" in reward_data:
                rewards = reward_data["rewards"]

                # Reshape rewards to (n_steps, n_envs, 1) to match rollout_batch
                rewards = rewards.view(n_steps, n_envs, 1)

                # Update actor's rollout_batch with computed rewards (remote call)
                self.actor.update_rewards(rewards).wait()

        return None  # Already waited

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        self.send_demo_buffer()

        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            if self.reward is not None:
                self.reward.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    self.update_rollout_weights()

                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.env_channel,
                        output_channel=self.rollout_channel,
                        actor_channel=self.actor_channel,
                    )
                    self.actor.recv_rollout_batch(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()

                reward_metrics = {}
                if self.reward is not None and self.cfg.reward.get(
                    "use_reward_model", False
                ):
                    with self.timer("compute_rewards"):
                        reward_handle = self.compute_rewards_with_model()
                        if reward_handle is not None:
                            reward_handle.wait()

                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                with self.timer("actor_training"):
                    actor_training_metrics = self.actor.run_training().wait()

                if (
                    self.reward is not None
                    and self.cfg.reward.get("train_reward_model", False)
                    and self.cfg.reward.get("use_reward_model", False)
                ):
                    with self.timer("reward_training"):
                        reward_training_metrics = self._train_reward_model()
                        if reward_training_metrics:
                            reward_metrics.update(reward_training_metrics)

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

            env_results_list = [
                results for results in env_handle.wait() if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }

            if reward_metrics:
                reward_metrics = {f"reward/{k}": v for k, v in reward_metrics.items()}
                self.metric_logger.log(reward_metrics, _step)

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            if reward_metrics:
                logging_metrics.update(reward_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _train_reward_model(self) -> Optional[dict]:
        if self.reward is None:
            return None
        return None

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )

        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

        if self.reward is not None and self.cfg.reward.get("use_reward_model", False):
            reward_save_path = os.path.join(base_output_dir, "reward")
            os.makedirs(reward_save_path, exist_ok=True)
            self.reward.save_checkpoint(reward_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
