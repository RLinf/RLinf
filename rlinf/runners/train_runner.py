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

import itertools
import logging
import os
import typing

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
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.agent.agent_loop import MultiTurnAgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolChannelInfo, ToolWorker, ToolWorkerInfo

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

logging.getLogger().setLevel(logging.INFO)


class AgentRunner:
    """Runner for agent task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: "SGLangWorker",
        actor: MegatronActor,
        agent_loop: MultiTurnAgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, "SGLangWorker"] = {},
    ):
        # Initialize base attributes
        self.cfg = cfg
        self.component_placement = placement
        self.is_pipeline = self.component_placement.is_pipeline

        # Workers
        self.rollout = rollout
        self.actor = actor

        # Scheduler task
        self.scheduler = None
        self.use_pre_process_policy = False

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")

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

        # Agent-specific attributes
        all_tool_calls = list(
            itertools.chain(
                *(worker_info.tool_names for worker_info in tool_workers.values())
            )
        )
        all_tool_worker_group_names = [
            worker.worker_group_name for worker in tool_workers
        ]
        assert len(set(all_tool_worker_group_names)) == len(
            all_tool_worker_group_names
        ), (
            f"AgentRunner: tool workers must be unique. all tool_worker_group_names are {all_tool_worker_group_names}"
        )
        assert len(set(all_tool_calls)) == len(all_tool_calls), (
            f"AgentRunner: tool_calls must be unique. all tool_calls are {all_tool_calls}"
        )
        self.agent_loop = agent_loop
        self.tool_workers = tool_workers
        self.solid_rollouts = solid_rollouts
        self.generate_input_channel = Channel.create("GenerateInput")
        self.generate_output_channel = Channel.create("GenerateOutput")
        self.solid_generate_input_channels = {}
        if self.solid_rollouts is not None:
            for solid_rollout_name in self.solid_rollouts:
                self.solid_generate_input_channels[solid_rollout_name] = Channel.create(
                    f"SolidRolloutInput-{solid_rollout_name}"
                )
        # tool worker name to tool channel info.
        self.tool_channel_info_map = {}
        # tool name to tool worker. a tool worker may have multiple tools.
        self.tool_name_map = {}
        for worker, worker_info in self.tool_workers.items():
            self.tool_channel_info_map[worker.worker_group_name] = ToolChannelInfo(
                tool_names=worker_info.tool_names,
                has_session=worker_info.has_session,
                input_channel=Channel.create(f"Tool-{worker.worker_group_name}"),
            )
            for tool_name in worker_info.tool_names:
                self.tool_name_map[tool_name] = worker.worker_group_name

        self.tool_output_channel = Channel.create("ToolOutput")

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

    def init_actor_workers(self):
        actor_handle = self.actor.init_worker()
        actor_handle.wait()

        if self.cfg.runner.resume_dir is None:
            return

        # Resume from checkpoint
        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"Setting global step to {self.global_steps}")

        actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")
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

    def init_workers(self):
        self.init_rollout_workers()
        self.init_actor_workers()

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

    def init_rollout_workers(self):
        # Init workers
        rollout_handles = [self.rollout.init_worker()]
        if self.solid_rollouts is not None:
            for solid_rollout in self.solid_rollouts.values():
                rollout_handle = solid_rollout.init_worker()
                rollout_handles.append(rollout_handle)

        for worker in self.tool_workers:
            input_channel = self.tool_channel_info_map[
                worker.worker_group_name
            ].input_channel
            tool_handle = worker.init_worker(input_channel, self.tool_output_channel)
            rollout_handles.append(tool_handle)

        # Must be done before actor init
        if self.cfg.runner.resume_dir is None:
            logging.info("Training from scratch")
            if (
                self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from toolkits.ckpt_convertor.convert_hf_to_mg import convert_hf_to_mg

                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )

        for rollout_handle in rollout_handles:
            rollout_handle.wait()
        if self.use_pre_process_policy:
            self.rollout.offload_engine().wait()

        self.agent_loop.init_worker(
            self.generate_input_channel,
            self.generate_output_channel,
            self.tool_channel_info_map,
            self.tool_name_map,
            self.tool_output_channel,
            self.solid_generate_input_channels,
        ).wait()

    def _sync_weights(self):
        self.actor.sync_model_to_rollout()
        self.rollout.sync_model_from_actor().wait()
        self.actor.del_reshard_state_dict().wait()
        if not self.is_pipeline and self.solid_rollouts is not None:
            for solid_rollout in self.solid_rollouts.values():
                solid_rollout.onload_engine().wait()

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
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        if self.solid_rollouts is not None:
            for solid_rollout_name, solid_rollout in self.solid_rollouts.items():
                solid_rollout.rollout_serverless(
                    self.solid_generate_input_channels[solid_rollout_name],
                    self.generate_output_channel,
                )
        for tool_worker in self.tool_workers:
            tool_worker.start_server()
        try:
            for _ in epoch_iter:
                for batch in self.train_dataloader:
                    with self.timer("step"):
                        with self.timer("prepare_data"):
                            self._put_batch(batch)

                        with self.timer("sync_weights"):
                            self._sync_weights()

                        # Rollout
                        rollout_handle: Handle = self.agent_loop.run_agentloop_rollout(
                            input_channel=self.dataloader_channel,
                            output_channel=self.rollout_channel,
                        )

                        if not self.is_pipeline:
                            rollout_handle.wait()
                            self.rollout.offload_engine().wait()
                            if self.solid_rollouts is not None:
                                for solid_rollout in self.solid_rollouts.values():
                                    solid_rollout.offload_engine().wait()

                        # Actor training, Advantages and returns
                        actor_handle: Handle = self.actor.run_training(
                            input_channel=self.rollout_channel,
                        )

                        metrics = actor_handle.wait()
                        actor_rollout_metrics = metrics[0][0]
                        actor_training_metrics = metrics[0][1]
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
                    time_metrics["training"] = actor_handle.consume_duration()
                    time_metrics["rollout"] = rollout_handle.consume_duration()

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
                        if actor_training_metrics != []:
                            training_metrics = {
                                f"train/{k}": v
                                for k, v in actor_training_metrics[i].items()
                            }
                            self.metric_logger.log(training_metrics, logging_steps + i)

                    logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}

                    logging_metrics.update(actor_rollout_metrics)
                    if actor_training_metrics != []:
                        logging_metrics.update(actor_training_metrics[-1])

                    global_pbar.set_postfix(logging_metrics, refresh=False)
                    global_pbar.update(1)
        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()
            self.metric_logger.finish()
