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
from typing import Union

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelEvalComponentPlacement
from rlinf.utils.timers import Timer
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class ReasoningEvalRunner:
    """Runner for reasoning task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelEvalComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        reward: RewardWorker,
    ):
        """"""
        self.cfg = cfg
        self.component_placement = placement

        # Workers
        self.rollout = rollout
        self.reward = reward

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")
        # Create a local channel (i.e., a channel that is different in every process)
        # if inference is not a dedicated worker
        self.reward_channel = Channel.create("Reward")

        # Configurations
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

    def init_workers(self):
        # Init workers
        self.rollout.init_worker().wait()
        self.reward.init_worker().wait()

        if self.cfg.runner.resume_dir is None:
            return

        # Resume from checkpoint
        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"Setting global step to {self.global_steps}")

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

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        self.run_timer.start_time()
        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    with self.timer("sync_weights"):
                        self._sync_weights()
