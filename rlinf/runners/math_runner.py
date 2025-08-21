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
from typing import Dict

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.rollout.sglang.sglang_worker import (
    AsyncSGLangWorker,
    SGLangWorker,
)

logging.getLogger().setLevel(logging.INFO)


class MathRunner:
    """Runner for math model training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: SGLangWorker,
        inference: MegatronInference,
        actor: MegatronActor,
    ):
        """"""
        self.cfg = cfg
        self.component_placement = placement

        # Workers
        self.rollout = rollout
        self.actor = actor
        # Collocated mode uses actor as inference
        self.inference = inference if inference is not None else self.actor

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")
        if self.inference is not None:
            self.inference_channel = Channel.create("Inference")
        self.actor_channel = Channel.create("Actor")

        # Configurations
        self.compute_ref_logprobs = self.cfg.algorithm.kl_beta > 0
        self.recompute_logprobs = self.cfg.rollout.recompute_logprobs
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
            batch_size=self.cfg.data.rollout_batch_size,
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

    def init_workers(self):
        # init rollout engine
        self.rollout.init_worker().wait()

        if self.cfg.runner.resume_dir is None:
            logging.info("Training from scratch")
            if (
                self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from tools.ckpt_convertor.convert_hf_to_mg import convert_hf_to_mg

                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )
            self.actor.init_worker().wait()
            return

        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])

        logging.info(f"Setting global step to {self.global_steps}")

        actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")

        # load actor
        self.actor.init_worker().wait()
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
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

    def _compute_flops_metrics(self, time_metrics, act_rollout_metrics) -> dict:
        rollout_time = time_metrics.get("rollout")
        inference_time = time_metrics.get("prev_logprobs")
        training_time = time_metrics.get("actor_training")

        num_gpus_actor = self.component_placement.actor_world_size
        num_gpus_inference = self.component_placement.inference_world_size
        num_gpus_rollout = self.component_placement.rollout_world_size

        prefill_decode_flops = act_rollout_metrics.get("prefill_decode_flops")
        prefill_total_flops = act_rollout_metrics.get("prefill_total_flops")

        flops_metrics = {
            "rollout_tflops_per_gpu": 0.0,
            "inference_tflops_per_gpu": 0.0,
            "training_tflops_per_gpu": 0.0,
        }

        if rollout_time > 0 and num_gpus_rollout > 0 and prefill_decode_flops > 0:
            flops_metrics["rollout_tflops_per_gpu"] = (
                prefill_decode_flops / rollout_time / 1e12 / num_gpus_rollout
            )

        if inference_time > 0 and num_gpus_inference > 0 and prefill_total_flops > 0:
            flops_metrics["inference_tflops_per_gpu"] = (
                prefill_total_flops / inference_time / 1e12 / num_gpus_inference
            )

        if training_time > 0 and num_gpus_actor > 0 and prefill_total_flops > 0:
            # here we use factor 3 to approximate the training tflops
            flops_metrics["training_tflops_per_gpu"] = (
                3 * prefill_total_flops / training_time / 1e12 / num_gpus_actor
            )

        return flops_metrics

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name,
            f"checkpoints/global_step_{self.global_steps}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        data_save_path = os.path.join(base_output_dir, "data")

        # actor
        self.actor.save_checkpoint(actor_save_path, self.global_steps).wait()

        # data
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

    def _put_batch(self, batch: Dict[str, torch.Tensor]):
        prompt_ids = batch["prompt"].tolist()
        lengths = batch["length"].tolist()
        answers = batch["answer"].tolist()
        prompts = [ids[-pmp_len:] for ids, pmp_len in zip(prompt_ids, lengths)]
        rollout_dp_size = self.component_placement.rollout_dp_size

        for input_ids, answers in zip(
            split_list(prompts, rollout_dp_size),
            split_list(answers, rollout_dp_size),
        ):
            request = RolloutRequest(
                n=self.cfg.algorithm.group_size,
                input_ids=input_ids,
                answers=answers,
            )
            self.dataloader_channel.put(request, async_op=True)

        # End mark
        for _ in range(rollout_dp_size):
            self.dataloader_channel.put(None, async_op=True)

    def _sync_weights(self):
        self.actor.sync_model_to_rollout()
        self.rollout.sync_model_from_actor().wait()

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.run_timer.start_time()
        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    # generate response and compute rule-based rewards.
                    with self.timer("rollout"):
                        self._sync_weights()
                        self.rollout.rollout(
                            input_channel=self.dataloader_channel,
                            output_channel=self.rollout_channel,
                        )
                        self.inference.process_rollout_result(
                            input_channel=self.rollout_channel
                        ).wait()

                    # recompute rollout policy logprobs, otherwise will use sglang logprobs.
                    if self.recompute_logprobs:
                        with self.timer("prev_logprobs"):
                            self.inference.compute_logprobs().wait()

                    # compute ref policy logprobs.
                    if self.compute_ref_logprobs:
                        with self.timer("ref_logprobs"):
                            self.inference.compute_ref_logprobs().wait()

                    # compute advantages and returns.
                    with self.timer("cal_adv_and_returns"):
                        actor_rollout_metrics = (
                            self.inference.compute_advantages_and_returns().wait()
                        )

                    # actor training.
                    with self.timer("actor_training"):
                        actor_training_metrics = self.actor.run_training().wait()

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
                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                # add prefix to the metrics
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                rollout_metrics = {
                    f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
                }

                self.metric_logger.log(log_time_metrics, logging_steps)
                self.metric_logger.log(rollout_metrics, logging_steps)
                for i in range(self.cfg.algorithm.n_minibatches):
                    training_metrics = {
                        f"train/{k}": v for k, v in actor_training_metrics[0][i].items()
                    }
                    self.metric_logger.log(training_metrics, logging_steps + i)

                logging_metrics = time_metrics

                if self.cfg.actor.get("calculate_flops", False):
                    flops_metrics = self._compute_flops_metrics(
                        time_metrics, actor_rollout_metrics[0]
                    )
                    flops_metrics = {f"flops/{k}": v for k, v in flops_metrics.items()}
                    self.metric_logger.log(flops_metrics, logging_steps)
                    logging_metrics.update(flops_metrics)

                logging_metrics.update(actor_rollout_metrics[0])
                logging_metrics.update(actor_training_metrics[0][-1])

                global_pbar.set_postfix(logging_metrics)
                global_pbar.update(1)

        self.metric_logger.finish()


class MathPipelineRunner:
    def __init__(
        self,
        cfg: DictConfig,
        run_timer,
        dataserver,
        actor: MegatronActor,
        rollout: AsyncSGLangWorker,
        inference: MegatronInference,
        critic=None,
        reward=None,
    ):
        self.cfg = cfg

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.dataserver = dataserver
        self.actor = actor
        self.rollout = rollout
        self.inference = inference
        self.critic = critic
        self.reward = reward

        self.compute_ref_logprobs = self.cfg.algorithm.kl_beta > 0
        self.recompute_logprobs = self.cfg.rollout.recompute_logprobs

        self.consumed_samples = 0
        # the step here is global step
        self.global_steps = 0
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

        self._set_max_steps()

    def init_workers(self):
        rollout_init_future = self.rollout.init_worker()
        inference_init_future = self.inference.init_worker()

        if self.cfg.runner.resume_dir is None:
            logging.info("Training from scratch")
            if (
                self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from tools.ckpt_convertor.convert_hf_to_mg import convert_hf_to_mg

                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )
            self.actor.init_worker().wait()
            rollout_init_future.wait()
            inference_init_future.wait()

            self._sync_weights()
            return

        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])

        logging.info(f"Setting global step to {self.global_steps}")

        actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")

        # load actor
        self.actor.init_worker().wait()
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        rollout_init_future.wait()
        inference_init_future.wait()

        self._sync_weights()

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

    def _sync_weights(self):
        rollout_sync_handle = self.rollout.sync_model_from_actor()
        actor_sync_handle = self.actor.sync_model_to_rollout()

        actor_send_handle = self.actor.sync_model_to_inference()
        infer_recv_handle = self.inference.sync_model_from_actor()

        rollout_sync_handle.wait()
        actor_sync_handle.wait()
        actor_send_handle.wait()
        infer_recv_handle.wait()

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name,
            f"checkpoints/global_step_{self.global_steps}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        data_save_path = os.path.join(base_output_dir, "data")

        # actor
        self.actor.save_checkpoint(actor_save_path, self.global_steps).wait()

        # data
        local_mkdir_safe(data_save_path)
        dataloader_local_path = os.path.join(data_save_path, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

    def _set_max_steps(self):
        self.max_steps = self.dataserver.get_train_dataset_size().wait()[0]
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def run(self):
        self.run_timer.start_time()
        self.dataserver.get_batch()

        epoch_iter = range(self.max_steps)

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            leave=True,
            desc="Global Step",
            ncols=620,
        )

        for _ in epoch_iter:
            with self.timer("step"):
                rollout_handle = self.rollout.rollout()
                infer_handle = self.inference.run_inference()
                train_handle = self.actor.run_training_pipeline()

                with self.timer("training"):
                    with self.timer("inference"):
                        with self.timer("rollout"):
                            rollout_handle.wait()
                        infer_handle.wait()
                    rollout_actor_metrics = train_handle.wait()

                with self.timer("sync_weights"):
                    self._sync_weights()

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
            logging_steps = (self.global_steps - 1) * self.cfg.algorithm.n_minibatches
            log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            self.metric_logger.log(log_time_metrics, logging_steps)
            rollout_metrics = {
                f"rollout/{k}": v for k, v in rollout_actor_metrics[0][0].items()
            }
            self.metric_logger.log(rollout_metrics, logging_steps)
            for i in range(self.cfg.algorithm.n_minibatches):
                training_metrics = {
                    f"train/{k}": v for k, v in rollout_actor_metrics[0][1][i].items()
                }
                self.metric_logger.log(training_metrics, logging_steps + i)

            self.timer._timer.reset()
            logging_metrics = time_metrics
            rollout_metrics = {
                f"rollout/{k}": v for k, v in rollout_actor_metrics[0][0].items()
            }
            training_metrics = {
                f"train/{k}": v for k, v in rollout_actor_metrics[0][1][-1].items()
            }
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            global_pbar.set_postfix(logging_metrics)
            global_pbar.update(1)
