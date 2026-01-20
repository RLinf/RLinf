import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig

try:
    import ray.util
except ImportError:
    ray = None
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest, RolloutResult
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

try:
    from agentlightning import LLM, NamedResources, RolloutLegacy
    from agentlightning.adapter.triplet import TraceToTripletBase
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.store.base import LightningStore
    from agentlightning.types.core import EnqueueRolloutRequest, Rollout, RolloutConfig, Task
except ImportError as e:
    raise ImportError(
        "AgentLightning is required for AgentLightningRLinfRunner. "
        "Please install agentlightning: pip install agentlightning"
    ) from e

from rlinf.workers.rollout.server.sglang_http_server_worker import SGLangHTTPServerWorker

import typing

if typing.TYPE_CHECKING:
    from rlinf.scheduler import Channel
    from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.inference.megatron_inference_worker import MegatronInference
    from rlinf.workers.reward.reward_worker import RewardWorker
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class RLinfAgentModeDaemon:

    def __init__(
        self,
        store: LightningStore,
        adapter: Optional[TraceToTripletBase] = None,
        llm_proxy: Optional[LLMProxy] = None,
        server_addresses: Optional[List[str]] = None,
        llm_timeout_seconds: float = 1200.0,
        group_size: int = 1,
        model: str = "default-model",
        reward_fillna_value: float = 0.0,
    ):
        self.store = store
        if llm_proxy is None:
            import socket
            def _find_available_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    return s.getsockname()[1]
            
            from agentlightning.utils.server_launcher import PythonServerLauncherArgs
            
            os.environ.setdefault("LITELLM_LOG", "WARNING")
            
            launcher_args = PythonServerLauncherArgs(
                port=_find_available_port(),
                log_level=logging.WARNING,
            )
            
            self.llm_proxy = LLMProxy(
                launcher_args=launcher_args,
                model_list=[],
                store=store,
            )
        else:
            self.llm_proxy = llm_proxy
        if adapter is None:
            from agentlightning.adapter.triplet import TracerTraceToTriplet
            self.adapter = TracerTraceToTriplet()
        else:
            self.adapter = adapter
        self.server_addresses = server_addresses or []
        self.llm_timeout_seconds = llm_timeout_seconds
        self.model = model
        self.group_size = group_size
        self.reward_fillna_value = reward_fillna_value
        self._resources_id: Optional[str] = None
        self._last_is_train: Optional[bool] = None
        self._rollout_id_to_original_sample: Dict[str, Dict[str, Any]] = {}
        self._total_tasks_queued = 0
        self._completed_rollout_ids: Dict[str, RolloutLegacy] = {}
        self._data_id_to_rollout_ids: Dict[str, List[str]] = {}

    async def async_setup_data(
        self,
        data: Dict[str, Any],
        server_addresses: Optional[List[str]] = None
    ):
        server_addresses_changed = False
        if server_addresses is not None and server_addresses != self.server_addresses:
            self.server_addresses = server_addresses
            await self._update_proxy_server()
            server_addresses_changed = True

        if self._resources_id is None and self.server_addresses and len(self.server_addresses) > 0:
            await self._update_proxy_server()

        if server_addresses_changed or self._resources_id is None:
            llm_resource = self.llm_proxy.as_resource(
                sampling_parameters={
                    "temperature": 0.7
                },
            )

            resources: NamedResources = {"main_llm": llm_resource}
            resources_update = await self.store.add_resources(resources)
            self._resources_id = resources_update.resources_id


        resources_id = self._resources_id

        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        group_size = self.group_size

        enqueue_rollout_requests: List[EnqueueRolloutRequest] = []
        data_id_to_original_sample: Dict[str, Dict[str, Any]] = {}

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id
            data_id_to_original_sample[data_id] = original_sample

            for rollout_idx in range(group_size):
                task_metadata = {"data_id": data_id}
                enqueue_rollout_requests.append(
                    EnqueueRolloutRequest(
                        input=original_sample,
                        mode="train",
                        resources_id=resources_id,
                        config=RolloutConfig(
                            unresponsive_seconds=self.llm_timeout_seconds,
                            timeout_seconds=self.llm_timeout_seconds,
                        ),
                        metadata=task_metadata,
                    )
                )

        rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)
        for rollout in rollouts:
            data_id = cast(Dict[str, Any], rollout.metadata)["data_id"]
            self._rollout_id_to_original_sample[rollout.rollout_id] = data_id_to_original_sample[data_id]
            if data_id not in self._data_id_to_rollout_ids:
                self._data_id_to_rollout_ids[data_id] = []
            self._data_id_to_rollout_ids[data_id].append(rollout.rollout_id)
        self._total_tasks_queued += len(rollouts)

    async def _update_proxy_server(self):
        import os
        from agentlightning.llm_proxy import ModelConfig
        
        model_name = os.path.basename(str(self.model)) if os.path.sep in str(self.model) else str(self.model)
        
        self.llm_proxy.update_model_list(
            [
                ModelConfig(
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            "model": "openai/" + model_name,
                            "api_base": f"http://{address}/v1/",
                            "api_key": "sk-placeholder",
                        },
                    }
                )
                for address in self.server_addresses
            ],
        )
        await self.llm_proxy.restart()


    async def _change_to_triplets(self, rollout: Rollout) -> RolloutLegacy:
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")

        if not spans:
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)

        final_reward: Optional[float] = None
        if triplets:
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        task = Task(
            rollout_id=rollout.rollout_id,
            input=rollout.input,
            mode=rollout.mode,
            resources_id=rollout.resources_id,
            metadata=rollout.metadata or {},
        )

        result_rollout = RolloutLegacy(
            rollout_id=rollout.rollout_id,
            task=task,
            final_reward=final_reward,
            triplets=triplets,
            metadata=rollout.metadata or {},
        )


        return result_rollout

    def clear_data(self):
        self._completed_rollout_ids.clear()
        self._rollout_id_to_original_sample.clear()
        self._data_id_to_rollout_ids.clear()
        self._total_tasks_queued = 0
    
    async def _async_get_completed_data_ids(self) -> List[str]:
        completed_data_ids = []
        for data_id, rollout_ids in self._data_id_to_rollout_ids.items():
            if all(rollout_id in self._completed_rollout_ids for rollout_id in rollout_ids):
                if data_id not in completed_data_ids:
                    completed_data_ids.append(data_id)
        return completed_data_ids

    
    async def _async_get_rollout_result_for_data_id(self, data_id: str) -> Optional[RolloutResult]:
        if data_id not in self._data_id_to_rollout_ids:
            return None
        
        rollout_ids = self._data_id_to_rollout_ids[data_id]
        if not all(rollout_id in self._completed_rollout_ids for rollout_id in rollout_ids):
            return None
        
        rollouts = [self._completed_rollout_ids[rollout_id] for rollout_id in rollout_ids]
        
        prompt_ids_list: List[List[int]] = []
        response_ids_list: List[List[int]] = []
        prompt_lengths_list: List[int] = []
        response_lengths_list: List[int] = []
        is_end_list: List[bool] = []
        rewards_list: List[float] = []
        
        for rollout_legacy in rollouts:
            if rollout_legacy.triplets is None or len(rollout_legacy.triplets) == 0:
                continue
            
            for triplet_idx, triplet in enumerate(rollout_legacy.triplets):
                prompt_token_ids = triplet.prompt.get("token_ids", [])
                if not isinstance(prompt_token_ids, list):
                    prompt_token_ids = []
                
                response_token_ids = triplet.response.get("token_ids", [])
                if not isinstance(response_token_ids, list):
                    response_token_ids = []
                
                if len(prompt_token_ids) == 0 or len(response_token_ids) == 0:
                    continue
                
                prompt_ids_list.append(prompt_token_ids)
                response_ids_list.append(response_token_ids)
                prompt_lengths_list.append(len(prompt_token_ids))
                response_lengths_list.append(len(response_token_ids))
                
                is_end = (triplet_idx == len(rollout_legacy.triplets) - 1)
                is_end_list.append(is_end)
                
                reward = rollout_legacy.final_reward
                if reward is None:
                    reward = self.reward_fillna_value
                rewards_list.append(reward)
        
        if len(prompt_ids_list) == 0:
            return None
        
        actual_group_size = len(rollouts)
        num_sequences = len(prompt_ids_list)
        
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        
        return RolloutResult(
            num_sequence=num_sequences,
            group_size=actual_group_size,
            prompt_lengths=prompt_lengths_list,
            prompt_ids=prompt_ids_list,
            response_lengths=response_lengths_list,
            response_ids=response_ids_list,
            is_end=is_end_list,
            rewards=rewards_tensor,
        )


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
        llm_proxy: LLMProxy,
        adapter: TraceToTripletBase,
        sglang_http_server: SGLangHTTPServerWorker,
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
        self.llm_proxy = llm_proxy
        self.adapter = adapter
        self.sglang_http_server = sglang_http_server
        
        self.daemon = RLinfAgentModeDaemon(
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            server_addresses=[],
            group_size=cfg.algorithm.group_size,
            model=cfg.rollout.model.model_path,
            reward_fillna_value=cfg.algorithm.get("reward_fillna_value", 0.0),
        )

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

        logging.info(
            f"[AgentLightningRLinfRunner] Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
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


    async def _async_process_dataloader_channel(self, expected_chunks: int):
        processed_chunks = 0
        while processed_chunks < expected_chunks:
            get_handle = self.dataloader_channel.get(async_op=True)
            chunk_data = await asyncio.to_thread(get_handle.wait)
            if chunk_data is None:
                break
            await self.daemon.async_setup_data(
                data=chunk_data,
                server_addresses=self.daemon.server_addresses
            )
            processed_chunks += 1

    async def _async_collect_rollout_results(self, rollout_channel: "Channel"):
        initial_data_ids_count = len(self.daemon._data_id_to_rollout_ids)
        processed_data_ids = set()
        
        while len(processed_data_ids) < initial_data_ids_count:
            rollout_ids_to_query = list(self.daemon._rollout_id_to_original_sample.keys())
            if not rollout_ids_to_query:
                break
                
            completed_batch = await self.daemon.store.wait_for_rollouts(
                rollout_ids=rollout_ids_to_query,
                timeout=0.0
            )
            
            for rollout in completed_batch:
                if rollout.rollout_id in self.daemon._completed_rollout_ids:
                    continue
                if rollout.rollout_id not in self.daemon._rollout_id_to_original_sample:
                    continue
                rollout = await self.daemon._change_to_triplets(rollout) if isinstance(rollout, Rollout) else rollout
                self.daemon._completed_rollout_ids[rollout.rollout_id] = rollout
            
            completed_data_ids = await self.daemon._async_get_completed_data_ids()
            for data_id in completed_data_ids:
                if data_id in processed_data_ids:
                    continue
                
                rollout_result = await self.daemon._async_get_rollout_result_for_data_id(data_id)
                if rollout_result is not None:
                    rollout_channel.put(rollout_result, async_op=True)
                    processed_data_ids.add(data_id)
            
            if len(processed_data_ids) < initial_data_ids_count:
                await asyncio.sleep(0.1)
        
        self.daemon.clear_data()

    def _put_batch(self, batch: dict):
        from rlinf.utils.data_iter_utils import split_list
        
        rollout_dp_size = self.component_placement.rollout_dp_size
        
        split_data_chunks = []
        for chunk_idx in range(rollout_dp_size):
            chunk_data = {}
            for field_name, field_data in batch.items():
                split_field_data = split_list(
                    field_data, rollout_dp_size, enforce_divisible_batch=False
                )
                if chunk_idx < len(split_field_data):
                    chunk_data[field_name] = split_field_data[chunk_idx]
                else:
                    chunk_data[field_name] = []
            
            if len(chunk_data) > 0 and len(chunk_data.get(list(chunk_data.keys())[0], [])) > 0:
                split_data_chunks.append(chunk_data)
        
        for chunk_data in split_data_chunks:
            self.dataloader_channel.put(chunk_data, async_op=True)
        
        return len(split_data_chunks)

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
            self.daemon.server_addresses = [f"{node_ip}:{server_port}"]
            logging.info(f"[AgentLightningRLinfRunner] Updated server addresses to {self.daemon.server_addresses} after server start")
        
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
                        expected_chunks = self._put_batch(batch)

                    async def process_batch():
                        await self._async_process_dataloader_channel(expected_chunks)
                        await self._async_collect_rollout_results(self.rollout_channel)
                    
                    asyncio.run(process_batch())
                    
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.rollout_channel,
                            output_channel=self.reward_channel,
                        )
                        inference_input_channel = self.reward_channel
                    else:
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

                time_metrics = self.timer.consume_durations()
                time_metrics["training"] = actor_handle.consume_duration()
                if reward_handle is not None:
                    time_metrics["reward"] = reward_handle.consume_duration()
                if infer_handle is not None:
                    time_metrics["inference"] = infer_handle.consume_duration(reduction_type="min")

                logging_steps = self.global_steps
                
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                
                rollout_metrics = {}
                for k, v in actor_rollout_metrics.items():
                    if k == "rewards":
                        rollout_metrics["training/reward"] = v
                        rollout_metrics["critic/rewards/mean"] = v
                    elif k == "reward_scores":
                        rollout_metrics["rollout/reward_scores"] = v
                    elif k.startswith("advantages"):
                        if k == "advantages_mean":
                            rollout_metrics["critic/advantages/mean"] = v
                        elif k == "advantages_max":
                            rollout_metrics["critic/advantages/max"] = v
                        elif k == "advantages_min":
                            rollout_metrics["critic/advantages/min"] = v
                    else:
                        rollout_metrics[f"rollout/{k}"] = v

                self.metric_logger.log(log_time_metrics, logging_steps)
                self.metric_logger.log(rollout_metrics, logging_steps)
                
                training_metrics = {
                    f"train/{k}": v for k, v in actor_training_metrics[-1].items()
                }
                self.metric_logger.log(training_metrics, logging_steps)

                logging_metrics = time_metrics
                logging_metrics.update(actor_rollout_metrics)
                logging_metrics.update(actor_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics, refresh=False)
                global_pbar.update(1)

        self.sglang_http_server.server_stop()
        self.metric_logger.finish()
