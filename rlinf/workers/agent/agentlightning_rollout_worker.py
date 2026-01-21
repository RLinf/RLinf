import asyncio
import logging
import os
import socket
import uuid
from typing import Any, Dict, List, Optional, Set, cast

import torch
from omegaconf import DictConfig, OmegaConf
try:
    from agentlightning import NamedResources, RolloutLegacy
    from agentlightning.adapter.triplet import TraceToTripletBase
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.store.base import LightningStore
    from agentlightning.types.core import EnqueueRolloutRequest, Rollout, RolloutConfig, Task
except ImportError as e:
    raise ImportError(
        "AgentLightning is required for AgentLightningRolloutWorker. "
        "Please install agentlightning: pip install agentlightning"
    ) from e

from rlinf.data.io_struct import RolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from agentlightning.llm_proxy import ModelConfig

def _find_available_port() -> int:
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class AgentLightningRolloutWorker(Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__()
        self.cfg = cfg
        self.store: Optional[LightningStore] = None
        self.llm_proxy: Optional[LLMProxy] = None
        self.adapter: Optional[TraceToTripletBase] = None
        self.server_addresses: List[str] = []
        self.llm_timeout_seconds: float = 1200.0
        self.model: str = "default-model"
        self.group_size: int = 1
        self.reward_fillna_value: float = 0.0
        self._resources_id: Optional[str] = None
        self._rollout_ids: Set[str] = set()
        self._total_tasks_queued = 0
        self._completed_rollout_ids: Dict[str, RolloutLegacy] = {}
        self._data_id_to_rollout_ids: Dict[str, List[str]] = {}

    def init_worker(
        self,
        store: LightningStore,
        adapter: TraceToTripletBase,
        server_addresses: Optional[List[str]] = None,
        group_size: int = 1,
        model: str = "default-model",
        reward_fillna_value: float = 0.0,
    ):
        self.store = store
        
        self.llm_proxy = LLMProxy(
            port=_find_available_port(),
            model_list=[],
            store=store,
        )
        self.llm_proxy.start()
        self.adapter = adapter
        self.server_addresses = server_addresses or []
        self.group_size = group_size
        self.model = model
        self.reward_fillna_value = reward_fillna_value

    async def _async_setup_data(
        self,
        data: Dict[str, Any],
    ):
        if self._resources_id is None and self.server_addresses and len(self.server_addresses) > 0:
            await self._update_proxy_server()
            sampling_params = self.cfg.algorithm.get("sampling_params", {})
        
            if isinstance(sampling_params, DictConfig):
                
                sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            llm_resource = self.llm_proxy.as_resource(
                sampling_parameters=sampling_params
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
            # Initialize the list for this data_id
            self._data_id_to_rollout_ids[data_id] = []

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
            self.log_info(f"[Rollout] {rollout}")
            data_id = cast(Dict[str, Any], rollout.metadata)["data_id"]
            self._rollout_ids.add(rollout.rollout_id)
            self._data_id_to_rollout_ids[data_id].append(rollout.rollout_id)
        self._total_tasks_queued += len(rollouts)

    async def _update_proxy_server(self):

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
        for span in spans:
            self.log_info(f"[Span] {span}")

        triplets = self.adapter.adapt(spans)
        for triplet in triplets:
            self.log_info(f"[Triplet] {triplet}")

        final_reward: Optional[float] = None
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
        self.log_info(f"[RolloutLegacy] {result_rollout}")

        return result_rollout

    def _clear_data(self):
        self._completed_rollout_ids.clear()
        self._rollout_ids.clear()
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

        rollout_ids = self._data_id_to_rollout_ids[data_id]
        
        rollouts = [self._completed_rollout_ids[rollout_id] for rollout_id in rollout_ids]
        
        prompt_ids_list: List[List[int]] = []
        response_ids_list: List[List[int]] = []
        prompt_lengths_list: List[int] = []
        response_lengths_list: List[int] = []
        is_end_list: List[bool] = []
        rewards_list: List[float] = []
        
        for rollout_legacy in rollouts:
            
            for triplet_idx, triplet in enumerate(rollout_legacy.triplets):
            
                prompt_token_ids = triplet.prompt.get("token_ids", [])
                response_token_ids = triplet.response.get("token_ids", [])

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
        
        num_sequences = len(prompt_ids_list)
        
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        
        rollout_result = RolloutResult(
            num_sequence=num_sequences,
            group_size=len(rollouts),
            prompt_lengths=prompt_lengths_list,
            prompt_ids=prompt_ids_list,
            response_lengths=response_lengths_list,
            response_ids=response_ids_list,
            is_end=is_end_list,
            rewards=rewards_tensor,
        )
        self.log_info(f"[RolloutResult] {rollout_result}")
        return rollout_result

    async def process_rollout_batch(
        self, input_channel: Channel, output_channel: Channel
    ):

        with self.worker_timer():
            batch_data = input_channel.get()
            
            await self._async_setup_data(
                data=batch_data,
            )
            
            initial_data_ids_count = len(self._data_id_to_rollout_ids)
            processed_data_ids = set()
            
            while len(processed_data_ids) < initial_data_ids_count:
                rollout_ids_to_query = [
                    rid for rid in self._rollout_ids 
                    if rid not in self._completed_rollout_ids
                ]

                completed_batch = await self.store.wait_for_rollouts(
                    rollout_ids=rollout_ids_to_query,
                    timeout=0.0
                )
                
                for rollout in completed_batch:
                    rollout = await self._change_to_triplets(rollout) if isinstance(rollout, Rollout) else rollout
                    self._completed_rollout_ids[rollout.rollout_id] = rollout
                
                completed_data_ids = await self._async_get_completed_data_ids()
                for data_id in completed_data_ids:
                    if data_id in processed_data_ids:
                        continue
                    
                    rollout_result = await self._async_get_rollout_result_for_data_id(data_id)
                    if rollout_result is not None:
                        output_channel.put(rollout_result, async_op=True)
                        processed_data_ids.add(data_id)
                
                if len(processed_data_ids) < initial_data_ids_count:
                    await asyncio.sleep(0.1)
            
            self._clear_data()

    def update_server_addresses(self, server_addresses: List[str]):
        self.server_addresses = server_addresses

