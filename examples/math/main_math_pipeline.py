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

import json
from typing import List

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.io_struct import RolloutRequest
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.runners.math_runner import MathPipelineRunner
from rlinf.scheduler import ChannelWorker, Cluster, PackedPlacementStrategy
from rlinf.scheduler.channel.channel_worker import WeightedItem
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.timers import Timer
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.rollout.sglang.sglang_worker import AsyncSGLangWorker
from rlinf.workers.rollout.utils import split_sequence

"""Script to start GRPO training"""
mp.set_start_method("spawn", force=True)


class DataServer(ChannelWorker):
    def __init__(self, config, maxsize: int = 0):
        super().__init__(maxsize)
        self.cfg = config
        self.name: str = self.cfg.actor.channel.name
        self.component_placement = ModelParallelComponentPlacement(self.cfg)

        # >>> rollout configuration
        self._rollout_dp_size: int = self.component_placement.rollout_dp_size
        self._rollout_queue_name = self.cfg.rollout.channel.queue_name
        self._rollout_output_queue_name = self.cfg.rollout.channel.output_queue_name
        # <<< #########################

        # >>> inference configuration
        self._infer_queue_name_prefix: str = self.cfg.inference.channel.queue_name
        self._infer_queue_size: int = self.cfg.inference.channel.queue_size
        self._inference_dp_size: int = self.component_placement.inference_dp_size
        self._inference_input_iter_len: int = (
            self.cfg.data.rollout_batch_size // self._inference_dp_size
        )
        # <<< #########################

        # >>> actor configuration
        self._actor_queue_name_prefix: str = self.cfg.actor.channel.queue_name
        self._actor_queue_size: int = self.cfg.actor.channel.queue_size
        self._actor_dp_size: int = self.component_placement.actor_dp_size
        self._actor_iter_len: int = (
            self.cfg.data.rollout_batch_size // self._actor_dp_size
        )
        # <<< #########################

        self._validate_attributes()

        # load dataset
        tokenizer = hf_tokenizer(self.cfg.actor.tokenizer.tokenizer_model)
        train_ds, val_ds = create_rl_dataset(self.cfg.data, tokenizer)
        self._build_dataloader(train_ds, val_ds)

        self._create_queue()

        self.log_info(
            f"DataServer {self.name} initialized with rollout_dp_size: {self._rollout_dp_size}, "
            f"inference_dp_size: {self._inference_dp_size}, actor_dp_size: {self._actor_dp_size}. "
            f"inference iter len = {self._inference_input_iter_len}"
        )

    def _validate_attributes(self):
        assert self.cfg.data.rollout_batch_size % self._inference_dp_size == 0, (
            f"rollout_batch_size {self.cfg.data.rollout_batch_size} must be divisible by "
            f"inference_dp_size {self._inference_dp_size}."
        )

        assert self.cfg.data.rollout_batch_size % self._actor_dp_size == 0, (
            f"rollout_batch_size {self.cfg.data.rollout_batch_size} must be divisible by "
            f"actor_dp_size {self._actor_dp_size}."
        )

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
            self.cfg.algorithm.val_rollout_batch_size_per_gpu
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

        self.log_debug(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

    def _create_queue(self):
        # input queue for all rollout dp
        self.create_queue(
            self._rollout_queue_name, maxsize=self.cfg.rollout.channel.queue_size
        )
        # output queue for all rollout dp
        self.create_queue(self._rollout_output_queue_name, maxsize=0)
        # input queue for all actor dp
        self.create_queue(self._infer_queue_name_prefix, maxsize=0)

    def _get_output_queue_name(self, dst_group_name: str):
        # define to aviod assertion in ChannelIORouter
        self.log_info(f"Getting output queue name: {self._rollout_queue_name}")
        return self._rollout_queue_name

    async def create_queue_for_inference(self):
        queue_names: List[str] = []
        for i in range(self._inference_dp_size):
            queue_name = f"{self._infer_queue_name_prefix}_{i}"
            self.create_queue(queue_name, maxsize=0)
            queue_names.append(queue_name)
        await self.round_robin_template(
            queue_name_input=self._rollout_output_queue_name,
            queue_names_output=queue_names,
            iter_len=self._inference_input_iter_len,
        )

    async def create_queue_for_actor_dp(self):
        queue_names: List[str] = []
        for i in range(self._actor_dp_size):
            queue_name = f"{self._actor_queue_name_prefix}_{i}"
            self.create_queue(queue_name, maxsize=self._actor_queue_size)
            queue_names.append(queue_name)
        await self.round_robin_template(
            queue_name_input=self._infer_queue_name_prefix,
            queue_names_output=queue_names,
            iter_len=self._actor_iter_len,
        )

    async def round_robin_template(
        self,
        queue_name_input: str,
        queue_names_output: List[str],
        iter_len: int,
    ):
        """
        Round-robin method to process requests from multiple queues.
        This is a template method that can be used for different types of queues.
        """
        current_queue_id = 0
        iter_counter = [0] * len(queue_names_output)
        while True:
            current_queue_name = queue_names_output[current_queue_id]
            item: WeightedItem = await self._queue_map[queue_name_input].get()
            await self._queue_map[current_queue_name].put(item)
            iter_counter[current_queue_id] += 1
            if iter_counter[current_queue_id] % iter_len == 0:
                self.log_info(
                    f"Dataserver round-robin policy: putting end magic to dst queue {current_queue_id}: {current_queue_name}."
                )
            current_queue_id = (current_queue_id + 1) % len(queue_names_output)

    async def get_batch(self):
        self.log_info(
            f"DataServer {self.name} is getting a batch. len of stateful dataloader: {len(self.train_dataloader)}"
        )
        for data in self.train_dataloader:
            prompt_ids = data["prompt"].tolist()
            lengths = data["length"].tolist()
            answers = data["answer"].tolist()
            prompts = [ids[-pmp_len:] for ids, pmp_len in zip(prompt_ids, lengths)]

            for input_ids, answers in zip(
                split_sequence(prompts, self._rollout_dp_size),
                split_sequence(answers, self._rollout_dp_size),
            ):
                request = RolloutRequest(
                    n=self.cfg.algorithm.group_size,
                    input_ids=input_ids,
                    answers=answers,
                )
                await self._queue_map[self._rollout_queue_name].put(
                    WeightedItem(weight=0, item=request)
                )

    async def get_train_dataset_size(self):
        """
        Returns the size of the training dataset.
        This is used to determine how many batches to process.
        """
        return len(self.train_dataloader)


@hydra.main(
    version_base="1.1", config_path="qwen2.5-math", config_name="grpo-1.5b-megatron"
)
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        num_nodes=cfg.cluster.num_nodes, num_gpus_per_node=cfg.cluster.num_gpus_per_node
    )

    component_placement = ModelParallelComponentPlacement(cfg)

    ds_placement_strategy = PackedPlacementStrategy(start_gpu_id=0, end_gpu_id=0)
    dataserver = DataServer.create_group(cfg).launch(
        cluster=cluster,
        name=cfg.actor.channel.name,
        placement_strategy=ds_placement_strategy,
    )
    dataserver.create_queue_for_inference()
    dataserver.create_queue_for_actor_dp()

    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout = AsyncSGLangWorker.create_group(cfg, component_placement).launch(
        cluster=cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    inference_placement_strategy = component_placement.get_strategy("inference")
    inference = MegatronInference.create_group(cfg, component_placement).launch(
        cluster=cluster,
        name=cfg.inference.group_name,
        placement_strategy=inference_placement_strategy,
    )

    actor_placement_strategy = component_placement.get_strategy("actor")
    actor = MegatronActor.create_group(cfg, component_placement).launch(
        cluster=cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement_strategy,
    )

    timer = Timer(None)

    runner = MathPipelineRunner(
        cfg=cfg,
        run_timer=timer,
        dataserver=dataserver,
        actor=actor,
        rollout=rollout,
        inference=inference,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
