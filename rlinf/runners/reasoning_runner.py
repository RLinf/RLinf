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
import typing
from typing import Optional, Union

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer

if typing.TYPE_CHECKING:
    from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
    from rlinf.utils.placement import ModelParallelComponentPlacement
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.actor.megatron_critic_worker import MegatronCritic
    from rlinf.workers.inference.fsdp_inference_worker import FSDPInference
    from rlinf.workers.inference.megatron_inference_worker import MegatronActorInference, MegatronCriticInference
    from rlinf.workers.reward.reward_worker import RewardWorker
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class ReasoningRunner:
    """Runner for reasoning task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: "ModelParallelComponentPlacement",
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        actor_inference: Optional[Union["MegatronActorInference", "FSDPInference"]],
        critic_inference: Optional[Union["MegatronCriticInference", "FSDPInference"]],
        actor: Union["FSDPActor", "MegatronActor"],
        reward: Optional["RewardWorker"],
        critic: Optional["MegatronCritic"] = None,
        scheduler: "SchedulerWorker" = None,
    ):
        self.cfg = cfg
        self.component_placement = placement
        self.is_pipeline = self.component_placement.is_pipeline
        self.has_dedicated_actor_inference = actor_inference is not None
        self.has_dedicated_critic_inference = critic_inference is not None

        # Workers
        self.rollout = rollout
        self.actor = actor
        self.reward = reward
        self.critic = critic

        # Collocated mode reuses actor worker and critic worker as inference workers
        self.actor_inference = actor_inference if self.has_dedicated_actor_inference else self.actor
        self.critic_inference = critic_inference if self.has_dedicated_actor_inference else self.critic

        # Scheduler task
        self.scheduler = scheduler
        self.use_pre_process_policy = (scheduler is not None) and getattr(
            self.cfg.cluster, "use_pre_process_policy", False
        )

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")
        # Create a local channel (i.e., a channel that is different in every process)
        # if inference is not a dedicated worker
        self.actor_inference_channel = Channel.create("ActorInference")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = self.rollout_channel

        if self.critic:
            if self.is_pipeline:
                self.critic_inference_channel_to_actor_train = Channel.create("CriticInferenceToActor")
                self.critic_inference_channel_to_critic_train = Channel.create("CriticInferenceToCritic")
                self.value_channel = [self.critic_inference_channel_to_actor_train,
                                      self.critic_inference_channel_to_critic_train]
            else:
                self.value_channel = Channel.create("Value")
            self.critic_output_channel = Channel.create("CriticOutput")
        else:
            self.value_channel = None
            self.critic_output_channel = None

        # Configurations
        self.compute_ref_logprobs = (
            self.cfg.algorithm.kl_beta > 0
            or self.cfg.algorithm.get("reinpp_kl_beta", 0) > 0
        )
        self.recompute_logprobs = self.cfg.algorithm.recompute_logprobs
        self.consumed_samples = 0
        self.global_steps = 0

        # Build dataloader and compute `max_steps`
        self._build_dataloader(train_dataset, val_dataset)
        self._set_max_steps()

        # Wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        # Timers
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.metric_logger = MetricLogger(cfg)

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        """
        Creates the train and validation dataloaders.
        """
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if collate_fn is None:
            from rlinf.data.datasets import collate_fn

        # Use a sampler to facilitate checkpoint resumption.
        # If shuffling is enabled in the data configuration, create a random sampler.
        if self.cfg.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.cfg.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
            sampler = SequentialSampler(data_source=self.train_dataset)

        num_workers = self.cfg.data.num_workers

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("max_num_gen_batches", 1),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        val_batch_size = (
            self.cfg.data.val_rollout_batch_size
        )  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.cfg.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        logging.info(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

    def init_rollout_workers(self):
        """init rollout worker."""
        rollout_handle = self.rollout.init_worker()

        # Must be done before actor init
        if self.cfg.runner.resume_dir is None:
            logging.info("Training from scratch")
            if (
                self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from toolkits.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                    convert_hf_to_mg,
                )

                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )

            if (
                self.critic is not None
                and self.cfg.critic.training_backend == "megatron"
                and self.cfg.critic.megatron.use_hf_ckpt
            ):
                from toolkits.ckpt_convertor.megatron_convertor.convert_hf_to_mg import convert_hf_to_mg

                convert_hf_to_mg(
                    self.cfg.critic.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.critic.megatron.ckpt_convertor,
                )

        rollout_handle.wait()
        if self.use_pre_process_policy:
            self.rollout.offload_engine().wait()

    def init_actor_critic_workers(self):
        """init actor worker, critic worker and reward worker."""
        if self.reward is not None:
            self.reward.init_worker().wait()

        actor_handle = self.actor.init_worker().wait()
        if self.has_dedicated_actor_inference:
            actor_inference_handle = self.actor_inference.init_worker()

        if self.critic is not None:
            critic_handle = self.critic.init_worker().wait()
            if self.has_dedicated_critic_inference:
                critic_inference_handle = self.critic_inference.init_worker()

        if self.has_dedicated_actor_inference:
            actor_inference_handle.wait()
        if self.critic is not None:
            if self.has_dedicated_critic_inference:
                critic_inference_handle.wait()

        if self.cfg.runner.resume_dir is None:
            return

        # Resume from checkpoint
        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"Setting global step to {self.global_steps}")

        actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")
        self.actor.load_checkpoint(actor_checkpoint_path).wait()

        if self.critic is not None:
            critic_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "critic")
            self.critic.load_checkpoint(critic_checkpoint_path).wait()

        # load data
        dataloader_local_path = os.path.join(self.cfg.runner.resume_dir, "data/data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logging.warning(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def init_workers(self):
        if self.cfg.runner.resume_dir == "auto":
            checkpoints_dir = os.path.join(
                self.cfg.runner.logger.log_path, "checkpoints"
            )

            checkpoint_steps = []
            if not os.path.exists(checkpoints_dir):
                logging.info("No checkpoints found, starting from scratch")
                self.cfg.runner.resume_dir = None
            else:
                checkpoint_steps = [
                    int(d.split("global_step_")[-1])
                    for d in os.listdir(checkpoints_dir)
                    if d.startswith("global_step_")
                    and os.path.isdir(os.path.join(checkpoints_dir, d))
                ]

            if checkpoint_steps:
                max_step = max(checkpoint_steps)
                self.cfg.runner.resume_dir = os.path.join(
                    checkpoints_dir, f"global_step_{max_step}"
                )
                logging.info(
                    f"Auto resume from checkpoint: {self.cfg.runner.resume_dir}"
                )
            else:
                self.cfg.runner.resume_dir = None
                logging.info("No checkpoints found, starting from scratch")
        self.init_rollout_workers()
        self.init_actor_critic_workers()

    def _compute_flops_metrics(self, time_metrics, act_rollout_metrics) -> dict:
        rollout_time = time_metrics.get("rollout")
        inference_time = time_metrics.get("inference", -1)
        training_time = time_metrics.get("training")

        num_gpus_actor = self.component_placement.actor_world_size
        num_gpus_rollout = self.component_placement.rollout_world_size

        rollout_tflops = act_rollout_metrics["rollout_tflops"]
        inference_tflops = act_rollout_metrics["inference_tflops"]
        training_tflops = act_rollout_metrics["training_tflops"]

        flops_metrics = {
            "rollout_tflops_per_gpu": 0.0,
            "inference_tflops_per_gpu": 0.0,
            "training_tflops_per_gpu": 0.0,
        }
        if rollout_time > 0 and rollout_tflops > 0:
            flops_metrics["rollout_tflops_per_gpu"] = (
                rollout_tflops / rollout_time / num_gpus_rollout
            )

        if inference_time > 0 and inference_tflops > 0:
            num_gpus_inference = self.component_placement.inference_world_size
            if num_gpus_inference == 0:
                num_gpus_inference = self.component_placement.actor_world_size
            flops_metrics["inference_tflops_per_gpu"] = (
                inference_tflops / inference_time / num_gpus_inference
            )

        if training_time > 0 and training_tflops > 0:
            flops_metrics["training_tflops_per_gpu"] = (
                training_tflops / training_time / num_gpus_actor
            )

        return flops_metrics

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name,
            f"checkpoints/global_step_{self.global_steps}",
        )

        # actor
        actor_save_path = os.path.join(base_output_dir, "actor")
        self.actor.save_checkpoint(actor_save_path, self.global_steps).wait()

        # critic
        if self.critic:
            critic_save_path = os.path.join(base_output_dir, "critic")
            self.critic.save_checkpoint(critic_save_path, self.global_steps).wait()

        # data
        data_save_path = os.path.join(base_output_dir, "data")
        local_mkdir_safe(data_save_path)
        dataloader_local_path = os.path.join(data_save_path, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

    def _set_max_steps(self):
        self.num_steps_per_epoch = len(self.train_dataloader)
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_steps // self.num_steps_per_epoch

    def _put_batch(self, batch: dict[str, torch.Tensor]):
        prompt_ids = batch["prompt"].tolist()
        lengths = batch["length"].tolist()
        answers = batch["answer"]
        image_data = batch["image_data"]
        multi_modal_inputs = batch["multi_modal_inputs"]
        prompt_ids = [ids[-pmp_len:] for ids, pmp_len in zip(prompt_ids, lengths)]
        rollout_dp_size = self.component_placement.rollout_dp_size

        for input_ids, answers, image_data, multi_modal_inputs in zip(
            split_list(prompt_ids, rollout_dp_size, enforce_divisible_batch=False),
            split_list(answers, rollout_dp_size, enforce_divisible_batch=False),
            split_list(image_data, rollout_dp_size, enforce_divisible_batch=False),
            split_list(
                multi_modal_inputs, rollout_dp_size, enforce_divisible_batch=False
            ),
        ):
            request = RolloutRequest(
                n=self.cfg.algorithm.group_size,
                input_ids=input_ids,
                answers=answers,
                image_data=image_data,
                multi_modal_inputs=multi_modal_inputs,
            )
            self.dataloader_channel.put(request, async_op=True)

    def _sync_weights(self):
        if self.has_dedicated_actor_inference:
            self.actor.sync_model_to_inference()
            self.actor_inference.sync_model_from_actor().wait()

        if self.has_dedicated_critic_inference:
            self.critic.sync_model_to_inference()
            self.critic_inference.sync_model_from_actor().wait() # TODO change this name

        self.actor.sync_model_to_rollout()
        self.rollout.sync_model_from_actor().wait()
        self.actor.del_reshard_state_dict().wait()

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=1240,
        )

        self.run_timer.start_time()
        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    with self.timer("sync_weights"):
                        self._sync_weights()

                    if self.scheduler is not None:
                        scheduler_handle = self.scheduler.schedule()

                    # Rollout
                    rollout_handle: Handle = self.rollout.rollout(
                        input_channel=self.dataloader_channel,
                        output_channel=self.rollout_channel,
                    )

                    # Rewards
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.rollout_channel,
                        output_channel=self.reward_channel,
                    )

                    if self.recompute_logprobs:
                        # Inference prev/ref logprobs
                        actor_infer_handle: Handle = self.actor_inference.run_inference(
                            input_channel=self.reward_channel,
                            output_channel=self.actor_inference_channel,
                            compute_ref_logprobs=self.compute_ref_logprobs,
                        )
                        actor_inference_channel = self.actor_inference_channel
                    else:
                        actor_infer_handle = None
                        actor_inference_channel = self.reward_channel

                    if self.critic:
                        critic_infer_handle: Handle = self.critic_inference.run_inference(
                            input_channel=actor_inference_channel, # TODO the channel contains generated trajetories
                            output_channel=self.value_channel,
                            do_offload=False,
                        )
                        training_input_channel = self.value_channel
                    else:
                        critic_infer_handle = None
                        training_input_channel = actor_inference_channel

                    critic_warmup_steps = self.cfg.algorithm.get('critic_warmup_steps', 0)
                    critic_warmup = (critic_warmup_steps > 0
                                      and self.global_steps < critic_warmup_steps)
                    if critic_warmup:
                        assert False, "critic warm up is not implemented yet"
                        assert self.critic is not None, \
                            "critic_warmup_steps is configured but no critic model is provided"
                        logging.info(f'critic model is warming up at step {self.global_steps}')

                    if self.is_pipeline:
                        if self.critic:
                            actor_training_input_channel = training_input_channel[0]
                            critic_training_input_channel = training_input_channel[1]
                        else:
                            actor_training_input_channel = training_input_channel
                    else:
                        if self.critic:
                            critic_training_input_channel = training_input_channel
                            critic_training_output_channel = self.critic_output_channel
                            actor_training_input_channel = critic_training_output_channel
                        else:
                            actor_training_input_channel = training_input_channel

                    # Critic training
                    if self.critic:
                        critic_train_handle: Handle = self.critic.run_training(
                            input_channel=critic_training_input_channel,
                            output_channel=critic_training_output_channel,
                            compute_rollout_metrics=False,
                        )
                    else:
                        critic_train_handle = None

                    # actor training
                    if not critic_warmup:
                        actor_handle: Handle = self.actor.run_training(
                            input_channel=actor_training_input_channel,
                            do_offload=False
                        )
                    else:
                        actor_handle = None

                    actor_metrics = None
                    if actor_handle:
                        actor_metrics = actor_handle.wait()
                    critic_training_metrics = None
                    if critic_train_handle is not None:
                        critic_metrics = critic_train_handle.wait()
                        _, critic_training_metrics = critic_metrics[0]

                    if self.scheduler is not None:
                        scheduler_handle.wait()
                    actor_rollout_metrics = actor_metrics[0][0]
                    actor_training_metrics = actor_metrics[0][1]
                    self.global_steps += 1

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

                    if is_train_end:
                        logging.info(
                            f"Step limit given by max_steps={self.max_steps} reached. Stopping run"
                        )
                        return

                    if run_time_exceeded:
                        logging.info(
                            f"Time limit given by run_timer={self.run_timer} reached. Stopping run"
                        )
                        return

                time_metrics = self.timer.consume_durations()

                time_metrics["actor/training"] = actor_handle.consume_duration()
                time_metrics["rollout"] = rollout_handle.consume_duration()
                time_metrics["reward"] = reward_handle.consume_duration()
                if actor_infer_handle is not None:
                    actor_infer_handle.wait()
                    # Inference time should be the min time across ranks, because different DP receive the rollout results differently
                    # But at the begin of the pp schedule, there is a timer barrier
                    # This makes all DP end at the same time, while they start at differnt times, and thus only the min time is correct
                    time_metrics["actor/inference"] = actor_infer_handle.consume_duration(
                        reduction_type="min"
                    )
                if critic_infer_handle is not None:
                    critic_infer_handle.wait()
                    time_metrics["critic/inference"] = critic_infer_handle.consume_duration()
                if critic_train_handle is not None:
                    time_metrics["critic/training"] = critic_train_handle.consume_duration()

                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                # add prefix to the metrics
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                rollout_metrics = {
                    f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
                }

                self.metric_logger.log(log_time_metrics, logging_steps)
                self.metric_logger.log(rollout_metrics, logging_steps)
                for i in range(self.cfg.algorithm.n_minibatches):
                    training_metrics = {}
                    if actor_training_metrics is not None:
                        for k, v in actor_training_metrics[i].items():
                            training_metrics[f"actor/training/{k}"] = v
                    if critic_training_metrics is not None:
                        for k, v in critic_training_metrics[i].items():
                            training_metrics[f"critic/training/{k}"] = v
                        
                    self.metric_logger.log(training_metrics, logging_steps + i)

                logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}

                if self.cfg.actor.get("calculate_flops", False):
                    flops_metrics = self._compute_flops_metrics(
                        time_metrics, actor_rollout_metrics
                    )
                    flops_metrics = {f"flops/{k}": v for k, v in flops_metrics.items()}
                    self.metric_logger.log(flops_metrics, logging_steps)
                    logging_metrics.update(flops_metrics)

                logging_metrics.update(actor_rollout_metrics)
                if actor_training_metrics is not None:
                    logging_metrics.update(actor_training_metrics[-1])
                if critic_training_metrics is not None:
                    logging_metrics.update(critic_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics, refresh=False)
                global_pbar.update(1)

        self.metric_logger.finish()
