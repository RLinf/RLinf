import logging
import time
from typing import Optional, Any

import torch
from omegaconf.dictconfig import DictConfig

import ray.util

from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer


from agentlightning.adapter.triplet import TraceToTripletBase
from agentlightning.store.base import LightningStore


from rlinf.workers.rollout.server.sglang_http_server_worker import SGLangHTTPServerWorker
from rlinf.workers.agent.agentlightning_rollout_worker import AgentLightningRolloutWorker

import typing

if typing.TYPE_CHECKING:
    from rlinf.scheduler import Channel
    from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.inference.megatron_inference_worker import MegatronInference
    from rlinf.workers.reward.reward_worker import RewardWorker
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class AgentLightningRLinfRunner(ReasoningRunner):

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: "SGLangWorker",
        inference: Optional["MegatronInference"],
        actor: "MegatronActor",
        reward: Optional["RewardWorker"],
        store: LightningStore,
        adapter: TraceToTripletBase,
        sglang_http_server: SGLangHTTPServerWorker,
        agentlightning_rollout_worker: AgentLightningRolloutWorker,
        scheduler: Optional["SchedulerWorker"] = None,
    ):
        super().__init__(
            cfg,
            placement,
            train_dataset,
            val_dataset,
            rollout,
            inference,
            actor,
            reward,
        )
        
        self.store = store
        self.adapter = adapter
        self.sglang_http_server = sglang_http_server
        self.agentlightning_rollout_worker = agentlightning_rollout_worker

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        
        
        if collate_fn is None:
            def agl_collate_fn(data_list: list[dict]) -> dict[str, Any]:
                batch = {}
                keys = list(data_list[0].keys())
                for key in keys:
                    batch[key] = [item[key] for item in data_list]
                return batch
            collate_fn = agl_collate_fn

        if self.cfg.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.cfg.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
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

        val_batch_size = self.cfg.data.val_rollout_batch_size or len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.cfg.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )


    def init_rollout_workers(self):
        rollout_handle = self.rollout.init_worker()

        if self.cfg.runner.resume_dir is None:
            logging.info("[AgentLightningRLinfRunner] Training from scratch")
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

        rollout_handle.wait()
        if self.use_pre_process_policy:
            self.rollout.offload_engine().wait()
        
        self.sglang_http_server.init_worker(self.rollout).wait()

        self.agentlightning_rollout_worker.init_worker(
            store=self.store,
            adapter=self.adapter,
            server_addresses=[],
            group_size=self.cfg.algorithm.group_size,
            model=self.cfg.rollout.model.model_path,
            reward_fillna_value=self.cfg.algorithm.get("reward_fillna_value", 0.0),
        ).wait()

    def _put_batch(self, batch: dict):
        self.dataloader_channel.put(batch, async_op=True)

    def run(self):
        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.sglang_http_server.server_start()
        
        if ray is not None:
            node_ip = ray.util.get_node_ip_address()
            server_port = self.cfg.server.sglang_http.get('port', 8020)
            server_addresses = [f"{node_ip}:{server_port}"]
            self.agentlightning_rollout_worker.update_server_addresses(server_addresses)
            logging.info(f"[AgentLightningRLinfRunner] Updated server addresses to {server_addresses} after server start")
        
        self.run_timer.start_time()

        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            return

        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):

                    with self.timer("sync_weights"):
                        self._sync_weights()

                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    rollout_handle: Handle = self.agentlightning_rollout_worker.process_rollout_batch(
                        input_channel=self.dataloader_channel,
                        output_channel=self.rollout_channel,
                    )

                    reward_handle = None
                    inference_input_channel = self.rollout_channel

                    if self.recompute_logprobs:
                        infer_handle: Handle = self.inference.run_inference(
                            input_channel=inference_input_channel,
                            output_channel=self.inference_channel,
                            compute_ref_logprobs=self.compute_ref_logprobs,
                        )
                        inference_channel = self.inference_channel
                    else:
                        infer_handle = None
                        inference_channel = inference_input_channel

                    actor_handle: Handle = self.actor.run_training(
                        input_channel=inference_channel,
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
                if reward_handle is not None:
                    time_metrics["reward"] = reward_handle.consume_duration()
                if infer_handle is not None:
                    time_metrics["inference"] = infer_handle.consume_duration()

                logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}
                logging_metrics.update(actor_rollout_metrics)
                logging_metrics.update(actor_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics, refresh=False)
                global_pbar.update(1)
            


        self.sglang_http_server.server_stop()
        self.metric_logger.finish()
