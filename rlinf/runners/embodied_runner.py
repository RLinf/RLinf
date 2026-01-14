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

logger = logging.getLogger(__name__)

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

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
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.demo_buffer = demo_buffer
        self.critic = critic
        self.reward = reward

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.demo_buffer is not None:
            self.demo_data_channel = Channel.create("DemoBufferChannel")
        
        # Reward channel for reward model training
        if self.reward is not None:
            self.reward_channel = Channel.create("RewardChannel")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        # Check if we're in reward-training-only mode
        reward_training_only = self.cfg.get("reward_training", {}).get("only", False)
        
        if reward_training_only:
            # Only initialize reward worker for reward model training
            if self.reward is not None:
                self.reward.init_worker().wait()
                logger.info("Initialized reward worker for training-only mode")
            return
        
        # Normal mode: create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()
        
        # Initialize reward worker if configured
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
        self.global_step = int(resume_dir.split("global_step_")[-1])
        
        # Load reward model checkpoint if exists
        if self.reward is not None:
            reward_checkpoint_path = os.path.join(resume_dir, "reward")
            if os.path.exists(reward_checkpoint_path):
                self.reward.load_checkpoint(reward_checkpoint_path).wait()

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

    def _apply_reward_model(self):
        """Apply reward model to replace/add rewards in actor's rollout batch.
        
        Computes model rewards for all timesteps in parallel for efficiency.
        """
        reward_mode = self.cfg.get("reward", {}).get("mode", "replace")
        
        # Get rollout batch from actor
        rollout_batch = self.actor.get_rollout_batch().wait()
        if rollout_batch is None or "obs" not in rollout_batch:
            return
        
        obs = rollout_batch["obs"]
        if not isinstance(obs, dict) or "main_images" not in obs:
            return
        
        # obs["main_images"] shape: [n_chunk_steps, batch, C, H, W]
        images = obs["main_images"]
        n_steps, batch_size = images.shape[0], images.shape[1]
        
        # Flatten all timesteps for parallel processing: [n_steps * batch, C, H, W]
        flat_images = images.reshape(n_steps * batch_size, *images.shape[2:])
        flat_obs = {"main_images": flat_images}
        
        # Compute rewards for all timesteps in one batch
        model_rewards = self.reward.compute_batch_rewards(
            observations=flat_obs,
        ).wait()
        
        if model_rewards is not None:
            # Reshape back: [n_steps * batch] -> [n_steps, batch, 1]
            model_rewards = model_rewards.reshape(n_steps, batch_size, 1)
            
            # Expand to match rewards shape [n_steps, batch, num_action_chunks]
            if "rewards" in rollout_batch:
                num_chunks = rollout_batch["rewards"].shape[-1]
                model_rewards = model_rewards.expand(n_steps, batch_size, num_chunks)
            
            self.actor.update_rewards(model_rewards, mode=reward_mode).wait()

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
        # Check if we're in reward-model-training-only mode
        reward_training_only = self.cfg.get("reward_training", {}).get("only", False)
        
        if reward_training_only:
            return self._run_reward_training_only()
        
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        self.send_demo_buffer()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

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
                
                # Compute model rewards and update actor's rollout batch
                if self.reward is not None and self.cfg.get("reward", {}).get("use_reward_model", False):
                    with self.timer("compute_model_rewards"):
                        self._apply_reward_model()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                with self.timer("actor_training"):
                    actor_training_metrics = self.actor.run_training().wait()

                # reward model training (if enabled)
                reward_training_metrics = [{}]
                if self.reward is not None and self.cfg.get("reward_training", {}).get("enabled", False):
                    with self.timer("reward_training"):
                        reward_training_metrics = self.reward.run_training().wait()

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
            env_results_list = [
                results for results in env_handle.wait() if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)

            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            reward_metrics = {
                f"reward/{k}": v for k, v in reward_training_metrics[0].items()
            } if reward_training_metrics and reward_training_metrics[0] else {}
            
            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            if reward_metrics:
                self.metric_logger.log(reward_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            logging_metrics.update(reward_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _run_reward_training_only(self):
        """Run reward model training only, without environment interaction.
        
        This mode is for training the reward model from pre-collected data.
        """
        if self.reward is None:
            raise RuntimeError("Reward worker not initialized for reward training")
        
        training_cfg = self.cfg.get("reward_training", {})
        epochs = training_cfg.get("epochs", 100)
        save_interval = training_cfg.get("save_interval", 10)
        save_dir = training_cfg.get("save_dir", "./reward_checkpoints")
        
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting reward model training for {epochs} epochs")
        
        best_acc = 0.0
        global_pbar = tqdm(
            total=epochs,
            desc="Training Reward Model",
            ncols=100,
        )
        
        for epoch in range(epochs):
            # Run one training epoch (internally iterates over full dataset)
            metrics = self.reward.run_training().wait()
            
            # Handle list return from worker group
            if isinstance(metrics, list):
                metrics = metrics[0] if metrics else {}
            
            loss = metrics.get("loss", 0.0)
            acc = metrics.get("accuracy", 0.0)
            lr = metrics.get("learning_rate", 0.0)
            
            # Log metrics
            self.metric_logger.log({
                "reward/loss": loss,
                "reward/accuracy": acc,
                "reward/learning_rate": lr,
            }, epoch)
            
            global_pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "acc": f"{acc:.4f}",
            })
            global_pbar.update(1)
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                self.reward.save_checkpoint(save_dir, "best").wait()
                logger.info(f"New best model saved: acc={acc:.4f}")
            
            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                ckpt_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.reward.save_checkpoint(ckpt_dir, epoch).wait()
        
        global_pbar.close()
        self.metric_logger.finish()
        logger.info(f"Reward model training completed. Best accuracy: {best_acc:.4f}")

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        
        # Save reward model checkpoint if configured
        if self.reward is not None and self.cfg.get("reward_training", {}).get("enabled", False):
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
