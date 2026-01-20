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

"""Runner for AgentLightning + RLinf integration.

This runner integrates AgentLightning's tracing and rollout management with RLinf's training loop.
It uses SGLangHTTPServerWorker to provide HTTP API for AgentLightning agents, and manages
the data flow from AgentLightning rollouts to RLinf training batches.
"""

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

# AgentLightning imports
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

# RLinf worker imports
from rlinf.workers.rollout.server.sglang_http_server_worker import SGLangHTTPServerWorker

# Type imports for type hints
import typing

if typing.TYPE_CHECKING:
    from rlinf.scheduler import Channel
    from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.inference.megatron_inference_worker import MegatronInference
    from rlinf.workers.reward.reward_worker import RewardWorker
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

logging.getLogger().setLevel(logging.INFO)

# Suppress AgentLightning verbose trace/triplet logging
# These JSON outputs are from AgentLightning's tracer/adapter when converting spans to triplets
# This matches VERL's approach of suppressing verbose logging
logging.getLogger("agentlightning").setLevel(logging.WARNING)
logging.getLogger("agentlightning.tracer").setLevel(logging.WARNING)
logging.getLogger("agentlightning.adapter").setLevel(logging.WARNING)
logging.getLogger("agentlightning.store").setLevel(logging.WARNING)

# Suppress LiteLLM Router verbose logging (LLMProxy uses LiteLLM)
# LiteLLM Router logs full request/response JSON which can be very long
# VERL doesn't have this issue because it uses vLLM directly, but RLinf uses LLMProxy which uses LiteLLM
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


class RLinfAgentModeDaemon:
    """Daemon for managing AgentLightning rollouts and converting them to RLinf training batches.

    This daemon handles:
    1. Enqueuing rollouts to AgentLightning store
    2. Waiting for rollouts to complete
    3. Converting completed rollouts (with spans) to RolloutResult format for RLinf training
    """

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
            # _find_available_port is not exported from agentlightning.llm_proxy
            # Define it locally (same implementation as in agentlightning.verl.daemon)
            import socket
            def _find_available_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    return s.getsockname()[1]
            
            # Suppress LiteLLM verbose logging in LLMProxy subprocess
            # LLMProxy runs in a separate process (launch_mode="mp" by default),
            # so we need to configure logging before creating LLMProxy
            # 1. Set uvicorn log level via launcher_args
            # 2. Set LiteLLM logging via environment variable (inherited by subprocess)
            from agentlightning.utils.server_launcher import PythonServerLauncherArgs
            
            # Set environment variable for LiteLLM (will be inherited by subprocess)
            # This suppresses LiteLLM Router's verbose request/response logging
            os.environ.setdefault("LITELLM_LOG", "WARNING")
            
            launcher_args = PythonServerLauncherArgs(
                port=_find_available_port(),
                log_level=logging.WARNING,  # Set uvicorn log level to WARNING
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
        self.is_train = True
        self.group_size = group_size
        self.reward_fillna_value = reward_fillna_value
        self._resources_id: Optional[str] = None
        self._last_is_train: Optional[bool] = None
        self._rollout_id_to_original_sample: Dict[str, Dict[str, Any]] = {}
        self._total_tasks_queued = 0
        self._completed_rollout_ids: Dict[str, RolloutLegacy] = {}
        self._data_id_to_rollout_ids: Dict[str, List[str]] = {}  # Track rollouts by data_id
        self._internal_loop: Optional[asyncio.AbstractEventLoop] = None
        self._internal_loop_thread = threading.Thread(target=self._internal_loop_runner, daemon=True)
        self._internal_loop_thread.start()

    def _internal_loop_runner(self):
        """Run the internal async event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._internal_loop = loop
        loop.run_forever()
        loop.close()

    async def async_setup_data(
        self,
        data: Dict[str, Any],
        server_addresses: Optional[List[str]] = None,
        is_train: Optional[bool] = None,
    ):
        """Setup data and enqueue rollouts to AgentLightning store."""
        if is_train is not None:
            self.is_train = is_train

        server_addresses_changed = False
        if server_addresses is not None and server_addresses != self.server_addresses:
            self.server_addresses = server_addresses
            # v1 模式：如果 server_addresses 变化，需要更新 LLMProxy 配置
            # 参考 VERL daemon v1 模式：更新 model_list 并重启 proxy
            await self._update_proxy_server()
            server_addresses_changed = True

        # 第一次调用时，如果 server_addresses 已设置但 proxy 未更新，需要先更新
        if self._resources_id is None and self.server_addresses and len(self.server_addresses) > 0:
            await self._update_proxy_server()

        is_train_changed = self._last_is_train != self.is_train
        if server_addresses_changed or is_train_changed or self._resources_id is None:
            # v1 模式：使用 llm_proxy.as_resource() 创建 resource
            # 参考 VERL daemon v1 模式的实现
            # LLMProxy 会转发请求到 backend server worker
            # LLMProxy 的配置应该已经在上面更新好了
            
            # 使用 llm_proxy.as_resource() 创建 resource
            # 这会返回一个 endpoint 指向 LLMProxy 的 LLM resource
            # LLMProxy 会根据配置的 model_list 转发请求到 backend server
            llm_resource = self.llm_proxy.as_resource(
                sampling_parameters={
                    "temperature": 0.7 if self.is_train else 0.0
                },
            )

            resources: NamedResources = {"main_llm": llm_resource}
            resources_update = await self.store.add_resources(resources)
            self._resources_id = resources_update.resources_id
            self._last_is_train = self.is_train

        resources_id = self._resources_id

        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        group_size = self.group_size if is_train else 1

        enqueue_rollout_requests: List[EnqueueRolloutRequest] = []
        data_id_to_original_sample: Dict[str, Dict[str, Any]] = {}

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id
            data_id_to_original_sample[data_id] = original_sample

            for rollout_idx in range(group_size):
                task_metadata = {"data_id": data_id, "is_train": self.is_train}
                enqueue_rollout_requests.append(
                    EnqueueRolloutRequest(
                        input=original_sample,
                        mode="train" if self.is_train else "val",
                        resources_id=resources_id,
                        config=RolloutConfig(
                            unresponsive_seconds=self.llm_timeout_seconds,
                            timeout_seconds=self.llm_timeout_seconds,
                        ),
                        metadata=task_metadata,
                    )
                )

        rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)
        # Track rollouts by data_id for async processing
        for rollout in rollouts:
            data_id = cast(Dict[str, Any], rollout.metadata)["data_id"]
            self._rollout_id_to_original_sample[rollout.rollout_id] = data_id_to_original_sample[data_id]
            if data_id not in self._data_id_to_rollout_ids:
                self._data_id_to_rollout_ids[data_id] = []
            self._data_id_to_rollout_ids[data_id].append(rollout.rollout_id)
        self._total_tasks_queued += len(rollouts)

    async def _update_proxy_server(self):
        """Update LLM proxy server addresses.
        
        参考 VERL daemon v1 模式的实现：
        - 更新 model_list 配置 backend server 地址
        - 调用 restart() 应用配置（如果 proxy 未运行，restart() 会启动它）
        
        ⭐ 注意：SGLangHTTPServerWorker 提供 OpenAI 兼容 API，不是 vLLM API
        所以使用 "openai" 而不是 "hosted_vllm"
        """
        import os
        from agentlightning.llm_proxy import ModelConfig
        
        # 从模型路径中提取模型名称（basename）
        # 如果 self.model 是路径，提取 basename；否则直接使用
        if os.path.sep in str(self.model):
            model_name = os.path.basename(str(self.model))
        else:
            model_name = str(self.model)
        
        self.llm_proxy.update_model_list(
            [
                ModelConfig(
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            # ⭐ SGLangHTTPServerWorker 提供 OpenAI 兼容 API
                            # 对于 OpenAI 兼容 API，需要使用 "openai/" 前缀来标识后端类型
                            # VERL 使用 "hosted_vllm/" + model_name 是因为它使用 vLLM 后端
                            # 使用 "openai/" 前缀时，LiteLLM 会使用 OpenAI 客户端模块，需要设置 api_key
                            # 即使后端不需要真正的 key，也需要设置一个占位符来满足 LiteLLM 的要求
                            "model": "openai/" + model_name,
                            "api_base": f"http://{address}/v1/",
                            "api_key": "sk-placeholder",  # 占位符，SGLang 的 OpenAI 兼容 API 不需要真正的 key
                        },
                    }
                )
                for address in self.server_addresses
            ],
        )
        # 调用 restart() 应用配置（如果 proxy 未运行，restart() 会启动它）
        # 参考 VERL daemon v1 模式：总是调用 restart()
        await self.llm_proxy.restart()

    def setup_data_sync(
        self,
        data: Dict[str, Any],
        server_addresses: Optional[List[str]] = None,
        is_train: Optional[bool] = None,
    ):
        """Synchronously setup data and enqueue rollouts."""
        coro = self.async_setup_data(
            data=data,
            server_addresses=server_addresses,
            is_train=is_train,
        )

        if self._internal_loop is None:
            raise RuntimeError("Internal loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._internal_loop)
        try:
            future.result(timeout=300)
        except Exception as e:
            logging.error(f"[RLinfAgentModeDaemon] Failed to set up data on server: {e}")
            raise

    def _validate_data(self, rollout: RolloutLegacy):
        """Validate rollout data."""
        if rollout.final_reward is None:
            logging.warning(
                f"[RLinfAgentModeDaemon] Warning: Reward is None for rollout {rollout.rollout_id}, "
                f"will be auto-set to {self.reward_fillna_value}."
            )
        if rollout.triplets is None:
            logging.warning(f"[RLinfAgentModeDaemon] Warning: Triplet is None for rollout {rollout.rollout_id}.")
        elif len(rollout.triplets) == 0:
            logging.warning(f"[RLinfAgentModeDaemon] Warning: Length of triplets is 0 for rollout {rollout.rollout_id}.")
        elif any(not r.response.get("token_ids", []) for r in rollout.triplets):
            logging.warning(f"[RLinfAgentModeDaemon] Warning: Rollout {rollout.rollout_id} contains empty response")
        elif any(not r.prompt.get("token_ids", []) for r in rollout.triplets):
            logging.warning(f"[RLinfAgentModeDaemon] Warning: Rollout {rollout.rollout_id} contains empty prompt")

    async def _change_to_triplets(self, rollout: Rollout) -> RolloutLegacy:
        """Convert Rollout to RolloutLegacy by querying spans and converting to triplets."""
        # Query spans for this rollout (latest attempt)
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")

        # Convert spans to triplets using the adapter
        if not spans:
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)

        # Extract final reward from triplets
        final_reward: Optional[float] = None
        if triplets:
            # Search backwards through triplets for the first non-None reward
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        # Construct the Task object from Rollout
        task = Task(
            rollout_id=rollout.rollout_id,
            input=rollout.input,
            mode=rollout.mode,
            resources_id=rollout.resources_id,
            metadata=rollout.metadata or {},
        )

        # Create the RolloutLegacy object
        result_rollout = RolloutLegacy(
            rollout_id=rollout.rollout_id,
            task=task,
            final_reward=final_reward,
            triplets=triplets,
            metadata=rollout.metadata or {},
        )

        # Validate the rollout
        self._validate_data(result_rollout)

        return result_rollout

    def clear_data(self):
        """Reset the internal state for the next run."""
        self._completed_rollout_ids.clear()
        self._rollout_id_to_original_sample.clear()
        self._data_id_to_rollout_ids.clear()
        self._total_tasks_queued = 0
    
    async def _async_get_completed_data_ids(self) -> List[str]:
        """Get list of data_ids that have all their rollouts completed."""
        completed_data_ids = []
        for data_id, rollout_ids in self._data_id_to_rollout_ids.items():
            if all(rollout_id in self._completed_rollout_ids for rollout_id in rollout_ids):
                if data_id not in completed_data_ids:
                    completed_data_ids.append(data_id)
        return completed_data_ids
    
    async def _remove_processed_data_id(self, data_id: str):
        """Remove processed data_id and its associated rollouts from state dictionaries (FIFO behavior).
        
        This method implements FIFO behavior: once a RolloutResult is put to channel,
        the corresponding data is removed from state dictionaries, similar to how channel
        automatically removes data after consumption.
        
        This ensures that data doesn't accumulate across batches in pipeline mode.
        """
        if data_id not in self._data_id_to_rollout_ids:
            return
        
        # Get all rollout IDs for this data_id
        rollout_ids = self._data_id_to_rollout_ids[data_id]
        
        # Remove from all state dictionaries
        for rollout_id in rollout_ids:
            # Remove from _rollout_id_to_original_sample
            if rollout_id in self._rollout_id_to_original_sample:
                del self._rollout_id_to_original_sample[rollout_id]
            
            # Remove from _completed_rollout_ids
            if rollout_id in self._completed_rollout_ids:
                del self._completed_rollout_ids[rollout_id]
        
        # Remove data_id from _data_id_to_rollout_ids
        del self._data_id_to_rollout_ids[data_id]
        
        # Update total tasks count
        self._total_tasks_queued -= len(rollout_ids)
    
    async def _async_get_rollout_result_for_data_id(self, data_id: str) -> Optional[RolloutResult]:
        """Convert completed rollouts for a specific data_id to RolloutResult."""
        if data_id not in self._data_id_to_rollout_ids:
            return None
        
        rollout_ids = self._data_id_to_rollout_ids[data_id]
        # Check if all rollouts for this data_id are completed
        if not all(rollout_id in self._completed_rollout_ids for rollout_id in rollout_ids):
            return None
        
        # Get all rollouts for this data_id
        rollouts = [self._completed_rollout_ids[rollout_id] for rollout_id in rollout_ids]
        
        # Convert to RolloutResult (similar to gettrainbatch logic)
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
                
                # ⭐ Align with VERL+AGL: Use final_reward for all triplets (episodic reward)
                # VERL uses sample_info["reward"] (which is final_reward) for all turns
                # This is more appropriate for tasks where only the final result has reward
                reward = rollout_legacy.final_reward
                if reward is None:
                    reward = self.reward_fillna_value
                rewards_list.append(reward)
        
        if len(prompt_ids_list) == 0:
            return None
        
        actual_group_size = len(rollouts)
        num_sequences = len(prompt_ids_list)
        
        # Convert rewards_list to tensor if not empty
        # This ensures to_actor_batch can call .cuda() on rewards
        rewards_tensor = None
        if rewards_list:
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
    
    async def async_stream_rollout_results(
        self,
        rollout_channel: "Channel",
        verbose: bool = True,
    ) -> int:
        """Async stream rollout results to channel as they complete.
        
        Implements FIFO behavior: data is removed from state dictionaries immediately
        after being put to channel, similar to how channel automatically removes data.
        This ensures that data doesn't accumulate across batches in pipeline mode.
        
        Returns the number of RolloutResults put into the channel.
        """
        processed_data_ids = set()
        total_put = 0
        
        # ⭐ Track initial data_ids count for this batch
        # This allows us to handle the case where new data_ids are added while streaming
        initial_data_ids_count = len(self._data_id_to_rollout_ids)
        
        while len(processed_data_ids) < initial_data_ids_count:
            # Check for newly completed rollouts
            # ⭐ Only query rollouts that are still in our tracking dictionaries
            rollout_ids_to_query = list(self._rollout_id_to_original_sample.keys())
            if not rollout_ids_to_query:
                # All rollouts have been processed and removed (FIFO)
                break
                
            completed_batch = await self.store.wait_for_rollouts(
                rollout_ids=rollout_ids_to_query,
                timeout=0.0
            )
            
            # Process newly completed rollouts
            for rollout in completed_batch:
                if rollout.rollout_id in self._completed_rollout_ids:
                    continue
                if rollout.rollout_id not in self._rollout_id_to_original_sample:
                    # This rollout was already processed and removed (FIFO)
                    continue
                if isinstance(rollout, Rollout):
                    rollout = await self._change_to_triplets(rollout)
                else:
                    self._validate_data(rollout)
                self._completed_rollout_ids[rollout.rollout_id] = rollout
            
            # Check for completed data_ids and put them to channel
            completed_data_ids = await self._async_get_completed_data_ids()
            for data_id in completed_data_ids:
                if data_id in processed_data_ids:
                    continue
                
                rollout_result = await self._async_get_rollout_result_for_data_id(data_id)
                if rollout_result is not None:
                    # Put to channel asynchronously
                    rollout_channel.put(rollout_result, async_op=True)
                    
                    # ⭐ FIFO: Remove data immediately after putting to channel
                    await self._remove_processed_data_id(data_id)
                    
                    processed_data_ids.add(data_id)
                    total_put += 1
                    if verbose:
                        logging.info(f"[RLinfAgentModeDaemon] Put RolloutResult for data_id {data_id} to channel ({total_put}/{initial_data_ids_count})")
            
            if verbose and len(processed_data_ids) < initial_data_ids_count:
                logging.info(f"[RLinfAgentModeDaemon] Completed {len(processed_data_ids)}/{initial_data_ids_count} data_ids...")
            
            await asyncio.sleep(0.1)  # Small sleep to avoid busy waiting
        
        logging.info(f"[RLinfAgentModeDaemon] All {total_put} RolloutResults have been put to channel.")
        return total_put
    
    def stream_rollout_results_sync(
        self,
        rollout_channel: "Channel",
        verbose: bool = True,
    ) -> int:
        """Synchronously stream rollout results to channel."""
        loop = self._internal_loop
        if loop is None:
            raise RuntimeError("Internal loop is not running.")
        
        coro = self.async_stream_rollout_results(rollout_channel, verbose)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result()
        except Exception as e:
            logging.error(f"[RLinfAgentModeDaemon] Error while streaming rollout results: {e}")
            raise



class AgentLightningRLinfRunner(ReasoningRunner):
    """Runner for AgentLightning + RLinf integration.
    
    This runner integrates AgentLightning's tracing system with RLinf's training loop.
    It uses SGLangHTTPServerWorker to provide HTTP API for AgentLightning agents,
    and manages the conversion from AgentLightning rollouts to RLinf training batches.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: "SGLangWorker",  # Base rollout worker (SGLangWorker)
        inference: Optional["MegatronInference"],
        actor: "MegatronActor",
        reward: Optional["RewardWorker"],
        # AgentLightning components
        store: LightningStore,
        llm_proxy: LLMProxy,
        adapter: TraceToTripletBase,
        sglang_http_server: SGLangHTTPServerWorker,  # HTTP server worker
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
            # AgentRunner doesn't pass scheduler, so we align with AgentRunner
            # scheduler=None,
        )
        
        # AgentLightning components
        self.store = store
        self.llm_proxy = llm_proxy
        self.adapter = adapter
        self.sglang_http_server = sglang_http_server
        
        # Create daemon for managing AgentLightning rollouts
        # Server addresses will be set after server starts (like main_searchr1_testhttpserver.py)
        # Initialize with empty addresses, will be updated in run() after server_start()
        self.daemon = RLinfAgentModeDaemon(
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            server_addresses=[],  # Will be set after server starts
            group_size=cfg.algorithm.group_size,
            model=cfg.rollout.model.model_path,
            reward_fillna_value=cfg.algorithm.get("reward_fillna_value", 0.0),
        )
        
        # Note: We override _build_dataloader to handle AgentLightning dataset format (dict)
        # AgentLightning datasets are dict-based (e.g., MathProblem), not DatasetItem-based
        # We need a custom collate_fn that preserves the original dict format

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        """
        Creates the train and validation dataloaders.
        Override ReasoningRunner._build_dataloader to handle AgentLightning dict format.
        """
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        
        # Check if dataset is AgentLightning format (list of dicts)
        is_agl_format = (
            isinstance(train_dataset, list) 
            and len(train_dataset) > 0 
            and isinstance(train_dataset[0], dict)
        )
        
        if collate_fn is None:
            if is_agl_format:
                # Custom collate_fn for AgentLightning dict format
                def agl_collate_fn(data_list: list[dict]) -> dict[str, Any]:
                    """Collate function for AgentLightning dict format datasets."""
                    # Group all fields by key
                    batch = {}
                    if len(data_list) > 0:
                        # Get all keys from first item
                        keys = list(data_list[0].keys())
                        for key in keys:
                            batch[key] = [item[key] for item in data_list]
                    return batch
                collate_fn = agl_collate_fn
            else:
                # Use standard RLinf collate_fn for DatasetItem format
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
            f"[AgentLightningRLinfRunner] Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation dataset.
        
        This method runs validation rollouts via AgentLightning and computes metrics.
        Similar to VERL+AgentLightning's validation flow.
        
        Reference: agentlightning/verl/trainer.py _validate() method
        
        Returns:
            Dictionary of validation metrics.
        """
        assert len(self.val_dataloader) > 0, "Validation dataloader is empty!"
        
        # ⭐ Reference VERL: assert len(self.val_dataloader) == 1 for better throughput
        # VERL expects the entire validation dataset as a single batch
        # We follow the same pattern: use the entire validation dataset
        val_batch = next(iter(self.val_dataloader))
        
        # Set daemon to validation mode
        self.daemon.is_train = False
        
        # ⭐ Setup data and server for validation (similar to VERL's set_up_data_and_server)
        # Use daemon.server_addresses (aligned with _put_batch)
        self.daemon.setup_data_sync(
            data=val_batch,
            server_addresses=self.daemon.server_addresses,
            is_train=False,
        )
        
        # ⭐ Wait for all rollouts to complete (similar to VERL's run_until_all_finished)
        # Reference: agentlightning/verl/daemon.py _async_run_until_finished()
        import asyncio
        loop = self.daemon._internal_loop
        if loop is None:
            raise RuntimeError("Daemon internal loop is not running.")
        
        async def wait_for_all_rollouts():
            """Wait for all rollouts to complete, similar to VERL's _async_run_until_finished."""
            verbose = True
            while len(self.daemon._completed_rollout_ids) < self.daemon._total_tasks_queued:
                completed_batch = await self.daemon.store.wait_for_rollouts(
                    rollout_ids=list(self.daemon._rollout_id_to_original_sample.keys()),
                    timeout=0.0  # ⭐ Use timeout=0.0 like VERL (non-blocking poll)
                )
                for rollout in completed_batch:
                    if rollout.rollout_id in self.daemon._completed_rollout_ids:
                        continue
                    if isinstance(rollout, Rollout):
                        rollout = await self.daemon._change_to_triplets(rollout)
                    else:
                        self.daemon._validate_data(rollout)
                    if rollout.rollout_id in self.daemon._rollout_id_to_original_sample:
                        self.daemon._completed_rollout_ids[rollout.rollout_id] = rollout
                
                if verbose:
                    logging.info(
                        f"[AgentLightningRLinfRunner] Completed {len(self.daemon._completed_rollout_ids)}/{self.daemon._total_tasks_queued} validation rollouts..."
                    )
                await asyncio.sleep(5)  # ⭐ Use 5s sleep like VERL (less frequent polling)
            
            logging.info("[AgentLightningRLinfRunner] All validation rollouts finished.")
        
        coro = wait_for_all_rollouts()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            future.result(timeout=3600)  # 1 hour timeout
        except Exception as e:
            logging.error(f"[AgentLightningRLinfRunner] Error while waiting for validation rollouts: {e}")
            raise
        
        # ⭐ Verify all rollouts are completed (similar to VERL's get_test_metrics assertion)
        # VERL uses assert, but we use warning to avoid crashing the training loop
        if len(self.daemon._completed_rollout_ids) != self.daemon._total_tasks_queued:
            logging.warning(
                f"[AgentLightningRLinfRunner] Validation incomplete: "
                f"completed {len(self.daemon._completed_rollout_ids)}/{self.daemon._total_tasks_queued} rollouts. "
                f"Some rollouts may have failed or timed out."
            )
        
        # ⭐ Verify we're in validation mode (similar to VERL's assertion)
        assert not self.daemon.is_train, "This method should only be called during validation."
        
        # ⭐ Get validation metrics from completed rollouts (similar to VERL's get_test_metrics)
        eval_metrics = self._compute_val_metrics()
        
        # ⭐ Clean up (similar to VERL's clear_data_and_server)
        # Note: VERL uses clear_data_and_server(), but our daemon only has clear_data()
        # We don't have async_rollout_manager.sleep() because we use HTTP server directly
        self.daemon.clear_data()
        
        # Reset daemon to training mode
        self.daemon.is_train = True
        
        return eval_metrics
    
    def _compute_val_metrics(self) -> Dict[str, float]:
        """Compute validation metrics from completed rollouts.
        
        This method extracts metrics similar to VERL's get_test_metrics.
        
        Returns:
            Dictionary of validation metrics.
        """
        import numpy as np
        
        # Get all completed rollouts
        completed_rollouts = list(self.daemon._completed_rollout_ids.values())
        
        if not completed_rollouts:
            logging.warning("[AgentLightningRLinfRunner] No completed rollouts for validation")
            return {
                "n_rollouts": 0,
                "n_rollouts_w_trace": 0,
                "n_rollouts_w_reward": 0,
                "reward": 0.0,
                "mean_response_length": 0.0,
                "sum_response_length": 0.0,
                "turn_count": 0.0,
            }
        
        sample_stat_list = []
        for rollout_legacy in completed_rollouts:
            final_reward_raw = rollout_legacy.final_reward
            final_reward = final_reward_raw if final_reward_raw is not None else self.daemon.reward_fillna_value
            
            if not rollout_legacy.triplets:
                logging.warning(f"No triplets found for validation rollout {rollout_legacy.rollout_id}")
                sample_stat_list.append({
                    "reward": final_reward,
                    "has_reward": final_reward_raw is not None,
                })
                continue
            
            response_length_list = [
                len(triplet.response.get("token_ids", [])) 
                for triplet in rollout_legacy.triplets
            ]
            
            sample_stat_list.append({
                "sum_response_length": np.sum(response_length_list),
                "mean_response_length": np.mean(response_length_list) if response_length_list else 0.0,
                "turn_count": len(rollout_legacy.triplets),
                "reward": final_reward,
                "has_reward": final_reward_raw is not None,
            })
        
        stats_w_trace = [stat for stat in sample_stat_list if "sum_response_length" in stat]
        
        metric_dict = {
            "n_rollouts": len(sample_stat_list),
            "n_rollouts_w_trace": len(stats_w_trace),
            "n_rollouts_w_reward": len([stat for stat in sample_stat_list if stat["has_reward"]]),
            "reward": np.mean([stat["reward"] for stat in sample_stat_list]),
            "mean_response_length": np.mean([stat["mean_response_length"] for stat in stats_w_trace]) if stats_w_trace else 0.0,
            "sum_response_length": np.mean([stat["sum_response_length"] for stat in stats_w_trace]) if stats_w_trace else 0.0,
            "turn_count": np.mean([stat["turn_count"] for stat in stats_w_trace]) if stats_w_trace else 0.0,
        }
        
        return metric_dict

    def init_rollout_workers(self):
        """init rollout worker."""
        rollout_handle = self.rollout.init_worker()

        # Must be done before actor init
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
        
        # Initialize HTTP server worker (needs rollout worker)
        self.sglang_http_server.init_worker(self.rollout).wait()

    def init_actor_workers(self):
        """init actor worker and reward worker."""
        if self.reward is not None:
            self.reward.init_worker().wait()

        actor_handle = self.actor.init_worker()
        if self.has_dedicated_inference:
            inference_handle = self.inference.init_worker()

        actor_handle.wait()
        if self.has_dedicated_inference:
            inference_handle.wait()

        if self.cfg.runner.resume_dir is None:
            return

        # Resume from checkpoint
        logging.info(f"[AgentLightningRLinfRunner] Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"[AgentLightningRLinfRunner] Setting global step to {self.global_steps}")

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
                f"[AgentLightningRLinfRunner] Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def init_workers(self):
        self.init_rollout_workers()
        self.init_actor_workers()



    def _put_batch(self, batch: dict):
        """Put batch data to AgentLightning store via daemon.
        
        This method is different from ReasoningRunner._put_batch because:
        - We don't use RolloutRequest and dataloader_channel
        - Instead, we enqueue rollouts to AgentLightning store via daemon
        - AgentLightning agents will execute rollouts via HTTP API
        - We need to split by rollout_dp_size but keep original data format (dict)
        
        The batch from collate_fn is already in dict format (AgentLightning format):
        - For MathProblem: {"id": List[str], "question": List[str], "chain": List[str], "result": List[str], "source": List[str]}
        - Or other fields from the dataset
        
        We split these fields by rollout_dp_size and pass to daemon.setup_data_sync.
        """
        from rlinf.utils.data_iter_utils import split_list
        
        # Batch is already in dict format with all fields
        # Each key maps to a List of values
        # Split by rollout_dp_size (similar to ReasoningRunner._put_batch)
        rollout_dp_size = self.component_placement.rollout_dp_size
        
        # Split each field by rollout_dp_size
        split_data_chunks = []
        for chunk_idx in range(rollout_dp_size):
            chunk_data = {}
            for field_name, field_data in batch.items():
                # Split the list by rollout_dp_size
                split_field_data = split_list(
                    field_data, rollout_dp_size, enforce_divisible_batch=False
                )
                # Get the chunk for this rollout_dp_idx
                if chunk_idx < len(split_field_data):
                    chunk_data[field_name] = split_field_data[chunk_idx]
                else:
                    # Empty chunk if not enough data
                    chunk_data[field_name] = []
            
            # Only add non-empty chunks
            if len(chunk_data) > 0 and len(chunk_data.get(list(chunk_data.keys())[0], [])) > 0:
                split_data_chunks.append(chunk_data)
        
        # For each chunk, enqueue rollouts to AgentLightning store
        # Note: We could parallelize this, but for now we do it sequentially
        # The daemon.setup_data_sync will handle the enqueuing
        for chunk_data in split_data_chunks:
            self.daemon.setup_data_sync(
                data=chunk_data,
                server_addresses=self.daemon.server_addresses,
                is_train=True,
            )

    def run(self):
        """Run the training loop.
        
        The training flow is:
        1. Start HTTP server (for AgentLightning agents)
        2. For each batch:
           a. Enqueue rollouts to AgentLightning store
           b. Wait for rollouts to complete (agents execute via HTTP)
           c. Convert completed rollouts to RolloutResult
           d. Put RolloutResult to dataloader_channel
           e. Run inference (if recompute_logprobs)
           f. Run actor training
        3. Stop HTTP server
        """
        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        # ⭐ Start HTTP server (similar to main_searchr1_testhttpserver.py)
        self.sglang_http_server.server_start()
        
        # ⭐ Get actual node IP after server starts and update daemon server_addresses
        # This ensures we use the correct node IP where the server is actually running
        try:
            if ray is not None:
                node_ip = ray.util.get_node_ip_address()
                server_port = self.cfg.server.sglang_http.get('port', 8020)
                self.daemon.server_addresses = [f"{node_ip}:{server_port}"]
                

                logging.info(f"[AgentLightningRLinfRunner] Updated server addresses to {self.daemon.server_addresses} after server start")
            else:
                logging.warning("[AgentLightningRLinfRunner] Ray is not available, cannot get node IP")
        except Exception as e:
            logging.error(f"[AgentLightningRLinfRunner] Failed to update server addresses after server start: {e}")
            raise
        
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

                    # ⭐ Rollout via AgentLightning HTTP API (async streaming with pipeline mode)
                    # 启动后台任务持续 streaming RolloutResult 到 channel
                    # 每个 data_id 完成就立即 put，不等待所有完成，实现 pipeline 并发
                    streaming_done = threading.Event()
                    streaming_exception = [None]  # 用于传递异常
                    
                    def streaming_worker():
                        """后台线程持续 streaming RolloutResult 到 channel"""
                        try:
                            num_rollout_results = self.daemon.stream_rollout_results_sync(
                                rollout_channel=self.rollout_channel,
                                verbose=True,
                            )
                            logging.info(f"[AgentLightningRLinfRunner] Streaming completed: {num_rollout_results} RolloutResults")
                        except Exception as e:
                            streaming_exception[0] = e
                            logging.error(f"[AgentLightningRLinfRunner] Streaming error: {e}")
                        finally:
                            streaming_done.set()
                    
                    # 启动后台 streaming 任务
                    streaming_thread = threading.Thread(target=streaming_worker, daemon=False)
                    streaming_thread.start()
                    
                    # ⭐ 立即启动下游处理（pipeline 模式）
                    # Reward/inference/actor 会从 channel pull 数据，不需要等待所有 rollout 完成
                    # 这样可以看到：每个 RolloutResult 完成就立即被下游处理，同时 rollout 继续处理其他样本
                    
                    # Rewards
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.rollout_channel,
                            output_channel=self.reward_channel,
                        )
                        inference_input_channel = self.reward_channel
                    else:
                        reward_handle = None
                        # For AgentLightning, rewards are already in RolloutResult
                        # But if no reward worker, use rollout_channel directly
                        inference_input_channel = self.rollout_channel

                    if self.recompute_logprobs:
                        # Inference prev/ref logprobs
                        infer_handle: Handle = self.inference.run_inference(
                            input_channel=inference_input_channel,
                            output_channel=self.inference_channel,
                            compute_ref_logprobs=self.compute_ref_logprobs,
                        )
                        inference_channel = self.inference_channel
                    else:
                        infer_handle = None
                        inference_channel = inference_input_channel

                    # Actor training
                    actor_handle: Handle = self.actor.run_training(
                        input_channel=inference_channel,
                    )

                    # ⭐ Pipeline 模式：不等待 streaming（对齐 agent_runner 的 rollout 逻辑）
                    # ⭐ 非 Pipeline 模式：等待 streaming 和 offload
                    if not self.is_pipeline:
                        # 等待 streaming 完成
                        if not streaming_done.wait(timeout=3600):  # 1小时超时
                            logging.error("[AgentLightningRLinfRunner] Streaming timeout!")
                            raise RuntimeError("Streaming timeout")
                        
                        if streaming_exception[0] is not None:
                            raise streaming_exception[0]
                        
                        # Offload engine
                        offload_handles = [self.rollout.offload_engine()]
                        for handle in offload_handles:
                            handle.wait()
                        
                        # ⭐ Clear daemon data (safety measure, though FIFO mechanism should have already cleaned up)
                        # FIFO mechanism automatically removes data after putting to channel,
                        # but we keep this as a safety measure for non-pipeline mode
                        self.daemon.clear_data()

                    metrics = actor_handle.wait()

                    actor_rollout_metrics = metrics[0][0]
                    actor_training_metrics = metrics[0][1]
                    self.global_steps += 1

                    run_time_exceeded = self.run_timer.is_finished()
                    run_val, save_model, is_train_end = check_progress(
                        self.global_steps,
                        self.max_steps,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=run_time_exceeded,
                    )

                    # ⭐ Run validation if needed（只在需要 eval 时才等待 streaming）
                    # ⚠️ EVAL 部分已注释，内容保留以便后续恢复
                    eval_metrics = {}
                    # if run_val:
                    #     # ⭐ Pipeline 模式：现在才等待 streaming（因为之前没等待过）
                    #     if self.is_pipeline:
                    #         # 等待 streaming 完成
                    #         if not streaming_done.wait(timeout=3600):
                    #             logging.error("[AgentLightningRLinfRunner] Streaming timeout!")
                    #             raise RuntimeError("Streaming timeout")
                    #         
                    #         if streaming_exception[0] is not None:
                    #             raise streaming_exception[0]
                    #         
                    #         # ⭐ Clear daemon data (safety measure, though FIFO mechanism should have already cleaned up)
                    #         # FIFO mechanism automatically removes data after putting to channel,
                    #         # but we keep this as a safety measure before eval
                    #         self.daemon.clear_data()
                    #     
                    #     # ⭐ Sync weights before evaluation to ensure using latest model
                    #     # This is important because eval happens after training, and we need to
                    #     # sync the updated model weights to rollout worker before evaluation
                    #     with self.timer("sync_weights_for_eval"):
                    #         self._sync_weights()
                    #     
                    #     with self.timer("eval"):
                    #         eval_metrics = self.evaluate()
                    #         eval_metrics = {f"val/{k}": v for k, v in eval_metrics.items()}
                    #         self.metric_logger.log(data=eval_metrics, step=self.global_steps)

                    if save_model:
                        self._save_checkpoint()

                    if is_train_end:
                        logging.info(
                            f"[AgentLightningRLinfRunner] Step limit given by max_steps={self.max_steps} reached. Stopping run"
                        )
                        self.sglang_http_server.server_stop()
                        self.metric_logger.finish()
                        return

                    if run_time_exceeded:
                        logging.info(
                            f"[AgentLightningRLinfRunner] Time limit given by run_timer={self.run_timer} reached. Stopping run"
                        )
                        self.sglang_http_server.server_stop()
                        self.metric_logger.finish()
                        return

                time_metrics = self.timer.consume_durations()
                time_metrics["training"] = actor_handle.consume_duration()
                if reward_handle is not None:
                    time_metrics["reward"] = reward_handle.consume_duration()
                if infer_handle is not None:
                    # Inference time should be the min time across ranks, because different DP receive the rollout results differently
                    # But at the begin of the pp schedule, there is a timer barrier
                    # This makes all DP end at the same time, while they start at differnt times, and thus only the min time is correct
                    time_metrics["inference"] = infer_handle.consume_duration(reduction_type="min")

                # ⭐ Use global_steps directly for TensorBoard logging (aligned with VERL+AgentLightning)
                logging_steps = self.global_steps
                
                # ⭐ Align metrics naming with VERL+AgentLightning
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                
                # Convert rollout metrics to VERL-style naming
                rollout_metrics = {}
                for k, v in actor_rollout_metrics.items():
                    if k == "rewards":
                        # Map rollout/rewards to training/reward (VERL style)
                        rollout_metrics["training/reward"] = v
                        # Also add critic/rewards/mean for consistency
                        rollout_metrics["critic/rewards/mean"] = v
                    elif k == "reward_scores":
                        # Keep reward_scores as rollout/reward_scores for reference
                        rollout_metrics["rollout/reward_scores"] = v
                    elif k.startswith("advantages"):
                        # Map advantages_* to critic/advantages/*
                        if k == "advantages_mean":
                            rollout_metrics["critic/advantages/mean"] = v
                        elif k == "advantages_max":
                            rollout_metrics["critic/advantages/max"] = v
                        elif k == "advantages_min":
                            rollout_metrics["critic/advantages/min"] = v
                    else:
                        # Keep other rollout metrics as-is
                        rollout_metrics[f"rollout/{k}"] = v

                # Log time and rollout metrics at global_steps
                self.metric_logger.log(log_time_metrics, logging_steps)
                self.metric_logger.log(rollout_metrics, logging_steps)
                
                # ⭐ Only log the last minibatch's training metrics (aligned with VERL+AgentLightning)
                # VERL logs aggregated metrics, not per-minibatch
                training_metrics = {
                    f"train/{k}": v for k, v in actor_training_metrics[-1].items()
                }
                self.metric_logger.log(training_metrics, logging_steps)

                logging_metrics = time_metrics
                logging_metrics.update(actor_rollout_metrics)
                logging_metrics.update(actor_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics, refresh=False)
                global_pbar.update(1)

        # ⭐ Stop HTTP server
        self.sglang_http_server.server_stop()
        self.metric_logger.finish()

