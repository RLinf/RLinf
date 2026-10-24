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

"""Synchronous PPO scheduling for multi-turn agents."""

from typing import Any

from tqdm import tqdm

from rlinf.data.schema.agent_trajectory import prepare_agent_ppo_targets
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger
from rlinf.utils.runner_utils import check_progress

logger = get_logger()


class AgentPPORunner(AgentRunner):
    """Complete both model updates and weight installation before each rollout."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        placement = self.component_placement
        actor_gpus = set(placement.get_hardware_ranks("actor"))
        critic_gpus = set(placement.get_hardware_ranks("critic"))
        rollout_gpus = set(placement.get_hardware_ranks("rollout"))
        self.shared_training = bool(actor_gpus & critic_gpus)
        self.release_rollout = bool(rollout_gpus & (actor_gpus | critic_gpus))

        self.rollout_output_channels = [self.rollout_channel]
        if not self.shared_training and self.recompute_logprobs:
            self.actor_rollout_channel = Channel.create("ActorRollout")
            self.rollout_output_channels.append(self.actor_rollout_channel)
        self.training_channel = Channel.create("PPOTraining")

    def init_rollout_workers(self) -> None:
        """Reuse agent initialization and convert the independent critic checkpoint."""
        super().init_rollout_workers()
        if self.cfg.runner.resume_dir is None and self.cfg.critic.megatron.use_hf_ckpt:
            from rlinf.utils.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                convert_hf_to_mg,
            )

            convert_hf_to_mg(
                self.cfg.critic.megatron.ckpt_convertor.hf_model_path,
                self.cfg.critic.megatron.ckpt_convertor,
            )

    def _sync_weights(
        self, do_offload: bool = True, weight_version: int | None = None
    ) -> None:
        if self.component_placement.is_collocated:
            super()._sync_weights(do_offload=do_offload, weight_version=weight_version)
            return

        super()._sync_weights(do_offload=False, weight_version=weight_version)
        if not do_offload:
            return
        if self.cfg.actor.offload_weight or self.cfg.actor.offload_grad:
            self.actor.offload_model_weights_and_grad(
                offload_weight=self.cfg.actor.offload_weight,
                offload_grad=self.cfg.actor.offload_grad,
            ).wait()
        if self.cfg.actor.offload_optimizer:
            self.actor.offload_megatron_optimizer().wait()

    def run(self) -> None:
        """Run complete rollout, prediction, update, and synchronization stages."""
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0 or self.global_steps >= self.max_steps:
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.run_timer.start_time()
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        for tool_worker in self.tool_workers:
            tool_worker.start_server()
        try:
            for _ in epoch_iter:
                for batch in self.train_dataloader:
                    with self.timer("step"):
                        with self.timer("prepare_data"):
                            self._put_batch(batch, self.batch_split_num)

                        with self.timer("sync_weights"):
                            self._sync_weights()

                        # Rollout: finish the whole batch before using training GPUs.
                        rollout_handle: Handle = self.agent_loop.run_agentloop_rollout(
                            input_channel=self.dataloader_channel,
                            output_channel=self.rollout_output_channels,
                        )
                        agent_metrics = rollout_handle.wait()[0]
                        if self.release_rollout:
                            self.rollout.offload_engine().wait()

                        # Old predictions: independent devices can infer in parallel.
                        critic_infer_handle: Handle = self.critic.run_inference(
                            input_channel=self.rollout_channel,
                            output_channel=self.value_channel,
                            do_offload=False,
                        )
                        actor_infer_handle = None
                        if not self.shared_training and self.recompute_logprobs:
                            actor_infer_handle: Handle = self.actor.run_inference(
                                input_channel=self.actor_rollout_channel,
                                output_channel=self.actor_inference_channel,
                                compute_ref_logprobs=False,
                            )
                            actor_infer_handle.wait()
                        critic_infer_handle.wait()

                        # GAE: fix globally normalized targets for every minibatch.
                        with self.timer("advantages"):
                            prepare_agent_ppo_targets(
                                input_channel=self.value_channel,
                                output_channel=[
                                    self.training_channel,
                                    self.critic_output_channel,
                                ],
                                batch_size=self.cfg.data.rollout_batch_size,
                                gamma=self.cfg.algorithm.gamma,
                                gae_lambda=self.cfg.algorithm.gae_lambda,
                                normalize_advantages=self.cfg.algorithm.normalize_advantages,
                                logprob_channel=(
                                    None
                                    if self.shared_training
                                    or not self.recompute_logprobs
                                    else self.actor_inference_channel
                                ),
                            )

                        # Updates: serialize shared devices; otherwise train in parallel.
                        critic_handle: Handle = self.critic.run_training(
                            input_channel=self.critic_output_channel,
                            compute_rollout_metrics=False,
                        )
                        if not self.shared_training:
                            actor_handle: Handle = self.actor.run_training(
                                input_channel=self.training_channel,
                            )
                        critic_metrics = critic_handle.wait()
                        critic_training_metrics = critic_metrics[0][1]

                        if self.shared_training:
                            actor_input_channel = self.training_channel
                            if self.recompute_logprobs:
                                actor_infer_handle = self.actor.run_inference(
                                    input_channel=self.training_channel,
                                    output_channel=self.actor_inference_channel,
                                    compute_ref_logprobs=False,
                                )
                                actor_infer_handle.wait()
                                actor_input_channel = self.actor_inference_channel
                            actor_handle: Handle = self.actor.run_training(
                                input_channel=actor_input_channel,
                            )
                        actor_metrics = actor_handle.wait()
                        actor_rollout_metrics = actor_metrics[0][0]
                        actor_training_metrics = actor_metrics[0][1]
                        self.global_steps += 1

                        # Checkpoint both models after their updates finish.
                        run_time_exceeded = self.run_timer.is_finished()
                        _, save_model, is_train_end = check_progress(
                            self.global_steps,
                            self.max_steps,
                            self.cfg.runner.val_check_interval,
                            self.cfg.runner.save_interval,
                            1.0,
                            run_time_exceeded=run_time_exceeded,
                        )
                        if save_model:
                            self._save_checkpoint()

                    time_metrics = self.timer.consume_durations()
                    time_metrics["training"] = actor_handle.consume_duration()
                    time_metrics["critic/training"] = critic_handle.consume_duration()
                    time_metrics["rollout"] = rollout_handle.consume_duration()
                    time_metrics["inference"] = (
                        actor_infer_handle.consume_duration(reduction_type="min")
                        if actor_infer_handle is not None
                        else 0.0
                    )
                    time_metrics["critic/inference"] = (
                        critic_infer_handle.consume_duration(reduction_type="min")
                    )

                    logging_steps = (
                        self.global_steps - 1
                    ) * self.cfg.algorithm.n_minibatches
                    log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

                    self.metric_logger.log(agent_metrics, logging_steps)
                    self.metric_logger.log(log_time_metrics, logging_steps)
                    if actor_rollout_metrics is not None:
                        rollout_metrics = {
                            f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
                        }
                        self.metric_logger.log(rollout_metrics, logging_steps)
                    for i in range(self.cfg.algorithm.n_minibatches):
                        training_metrics = {
                            f"train/{k}": v
                            for k, v in actor_training_metrics[i].items()
                        }
                        training_metrics.update(
                            {
                                f"train/{k}": v
                                for k, v in critic_training_metrics[i].items()
                            }
                        )
                        self.metric_logger.log(training_metrics, logging_steps + i)

                    logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}
                    logging_metrics.update(agent_metrics)
                    if actor_rollout_metrics is not None:
                        logging_metrics.update(actor_rollout_metrics)
                    logging_metrics.update(actor_training_metrics[-1])
                    logging_metrics.update(critic_training_metrics[-1])
                    global_pbar.set_postfix(logging_metrics, refresh=False)
                    global_pbar.update(1)

                    if is_train_end:
                        logger.info(
                            f"Step limit given by max_steps={self.max_steps} reached. Stopping run"
                        )
                        return
                    if run_time_exceeded:
                        logger.info(
                            f"Time limit given by run_timer={self.run_timer} reached. Stopping run"
                        )
                        return
        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()
            global_pbar.close()
            self.metric_logger.finish()
