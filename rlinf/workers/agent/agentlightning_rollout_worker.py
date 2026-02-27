import asyncio
import logging
import os
import socket
import uuid
from typing import Any, Dict, List, Optional, Set, cast
from transformers import AutoTokenizer
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from agentlightning import NamedResources, RolloutLegacy
from agentlightning.adapter.triplet import TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore
from agentlightning.types.core import EnqueueRolloutRequest, Rollout, RolloutConfig, Task, Triplet


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.rollout.model.model_path)
        self.is_eval_mode: bool = False
        self.advantage_mode: str = self.cfg.algorithm.get("advantage_mode", "turn")

    def init_worker(
        self,
        store: LightningStore,
        adapter: TraceToTripletBase,
        server_addresses: Optional[List[str]] = None,
        group_size: int = 1,
        model: str = "default-model",
        reward_fillna_value: float = 0.0,
        is_eval_mode: bool = False,
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
        self.group_size = 1 if is_eval_mode else group_size
        self.model = model
        self.reward_fillna_value = reward_fillna_value
        self.is_eval_mode = is_eval_mode

    async def _async_setup_data(
        self,
        data: Dict[str, Any],
    ):
        if self._resources_id is None and self.server_addresses and len(self.server_addresses) > 0:
            await self._update_proxy_server()
            sampling_params = self.cfg.algorithm.get("sampling_params", {})
            if isinstance(sampling_params, DictConfig):
                sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            if self.is_eval_mode:
                sampling_params["temperature"] = 0.0
            
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
            self._data_id_to_rollout_ids[data_id] = []

            for rollout_idx in range(group_size):
                task_metadata = {"data_id": data_id}
                rollout_mode = "val" if self.is_eval_mode else "train"
                enqueue_rollout_requests.append(
                    EnqueueRolloutRequest(
                        input=original_sample,
                        mode=rollout_mode,
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

        triplets = self.adapter.adapt(spans)

        final_reward: Optional[float] = None
        for triplet in reversed(triplets):
            if triplet.reward is not None:
                final_reward = triplet.reward
                break
        if final_reward is None:
            final_reward = self.reward_fillna_value
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

    def _count_tool_calls_in_triplet(self, triplet: Triplet) -> int:
        if triplet.metadata and "tool_calls" in triplet.metadata:
            tool_calls = triplet.metadata["tool_calls"]
            if isinstance(tool_calls, list):
                return len(tool_calls)
            elif isinstance(tool_calls, int):
                return tool_calls
        
        if isinstance(triplet.response, dict):
            if "tool_calls" in triplet.response:
                tool_calls = triplet.response["tool_calls"]
                if isinstance(tool_calls, list):
                    return len(tool_calls)
            if "metadata" in triplet.response and isinstance(triplet.response["metadata"], dict):
                if "tool_calls" in triplet.response["metadata"]:
                    tool_calls = triplet.response["metadata"]["tool_calls"]
                    if isinstance(tool_calls, list):
                        return len(tool_calls)
        
        return 0

    def _compute_rollout_metrics(
        self, 
        rollout_results: List[RolloutResult],
        rollouts: List[RolloutLegacy]
    ) -> Dict[str, float]:
        if not rollout_results:
            return {
                "agent/reward": 0.0,
                "agent/n_rollouts": 0,
                "agent/n_rollouts_w_trace": 0,
                "agent/n_rollouts_w_reward": 0,
            }
        
        all_rewards: List[float] = []
        total_response_lengths: List[int] = []
        n_rollouts = 0
        n_rollouts_w_trace = 0
        
        total_tool_calls = 0
        total_turns = 0
        n_triplets = 0
        n_rollouts_w_reward = 0
        
        for rollout_result in rollout_results:
            n_rollouts += rollout_result.batch_size
            n_triplets += rollout_result.num_sequence
            
            if rollout_result.rewards is not None:
                if isinstance(rollout_result.rewards, torch.Tensor):
                    rewards_list = rollout_result.rewards.tolist()
                else:
                    rewards_list = rollout_result.rewards
                all_rewards.extend(rewards_list)
            else:
                all_rewards.extend([self.reward_fillna_value] * rollout_result.batch_size)
            
            if rollout_result.response_lengths:
                n_rollouts_w_trace += rollout_result.batch_size
                total_response_lengths.extend(rollout_result.response_lengths)
        
        for rollout_legacy in rollouts:
            if rollout_legacy.final_reward is not None:
                n_rollouts_w_reward += 1
            
            if rollout_legacy.triplets:
                num_turns = len(rollout_legacy.triplets)
                total_turns += num_turns
                for triplet in rollout_legacy.triplets:
                    total_tool_calls += self._count_tool_calls_in_triplet(triplet)
        
        training_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        metrics = {
            "agent/reward": float(training_reward),
            "agent/n_rollouts": n_rollouts,
            "agent/n_rollouts_w_trace": n_rollouts_w_trace,
            "agent/n_rollouts_w_reward": n_rollouts_w_reward,
            "agent/turn_count": n_triplets,
            "agent/mean_turn_count_per_rollout": float(total_turns / n_rollouts) if n_rollouts > 0 else 0.0,
            "agent/mean_response_length": float(np.mean(total_response_lengths)) if total_response_lengths else 0.0,
            "agent/total_tool_calls": total_tool_calls,
            "agent/mean_tool_calls_per_rollout": float(total_tool_calls / n_rollouts) if n_rollouts > 0 else 0.0,
            "agent/mean_tool_calls_per_turn": float(total_tool_calls / total_turns) if total_turns > 0 else 0.0,
        }
        
        return metrics

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

    async def _async_get_rollout_result_for_data_id(self, data_id: str) -> Optional[Union[RolloutResult, DynamicRolloutResult]]:
        rollout_ids = self._data_id_to_rollout_ids[data_id]
        rollouts = [self._completed_rollout_ids[rollout_id] for rollout_id in rollout_ids]
        
        max_prompt_len = int(self.cfg.data.get("max_prompt_length", 4096))
        max_response_length = int(self.cfg.data.get("max_response_length", 2048))
        
        if self.advantage_mode == "turn":
            idx_to_traj: List[int] = []
            input_ids_list: List[List[int]] = []
            prompt_lengths_list: List[int] = []
            response_lengths_list: List[int] = []
            is_end_list: List[bool] = []
            rewards_list: List[float] = []
            rollout_logprobs_list: List[List[float]] = []
            
            for traj_idx, rollout_legacy in enumerate(rollouts):
                for triplet in rollout_legacy.triplets:
                    prompt_ids = triplet.prompt.get("token_ids", [])
                    response_ids = triplet.response.get("token_ids", [])
                    
                    if len(prompt_ids) > max_prompt_len:
                        prompt_ids = prompt_ids[:max_prompt_len]
                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                    
                    input_ids = prompt_ids + response_ids
                    
                    turn_logprobs: List[float] = []
                    if self.cfg.rollout.return_logprobs:
                        logprobs = triplet.response.get("logprobs", [])
                        if logprobs:
                            turn_logprobs = [lp.get("logprob", 0.0) for lp in logprobs]
                            if len(turn_logprobs) > max_response_length:
                                turn_logprobs = turn_logprobs[:max_response_length]
                    
                    idx_to_traj.append(traj_idx)
                    input_ids_list.append(input_ids)
                    prompt_lengths_list.append(len(prompt_ids))
                    response_lengths_list.append(len(response_ids))
                    is_end_list.append(True)
                    rewards_list.append(rollout_legacy.final_reward)
                    rollout_logprobs_list.append(turn_logprobs)

            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)

            dynamic_rollout_result = DynamicRolloutResult(
                num_sequence=len(input_ids_list),
                group_size=len(rollouts),
                idx_to_traj=idx_to_traj,
                input_ids=input_ids_list,
                prompt_lengths=prompt_lengths_list,
                response_lengths=response_lengths_list,
                is_end=is_end_list,
                rewards=rewards_tensor,
                rollout_logprobs=rollout_logprobs_list if self.cfg.rollout.return_logprobs else None,
            )
            return dynamic_rollout_result
        else:
            prompt_ids_list: List[List[int]] = []
            response_ids_list: List[List[int]] = []
            prompt_lengths_list: List[int] = []
            response_lengths_list: List[int] = []
            is_end_list: List[bool] = []
            rewards_list: List[float] = []
            response_mask_list: List[List[int]] = []
            prompt_texts_list: List[str] = []
            response_texts_list: List[str] = []
            rollout_logprobs_list: List[List[float]] = []

            for rollout_legacy in rollouts:

                first_triplet = rollout_legacy.triplets[0]
                orig_prompt_ids = first_triplet.prompt.get("token_ids", [])
                
                if len(orig_prompt_ids) > max_prompt_len:
                    orig_prompt_ids = orig_prompt_ids[:max_prompt_len]
                
                accumulated_response_mask: List[int] = []
                accumulated_full_sequence: List[int] = []
                accumulated_logprobs: List[float] = []
                
                for triplet_idx, triplet in enumerate(rollout_legacy.triplets):
                    prompt_token_ids = triplet.prompt.get("token_ids", [])
                    response_token_ids = triplet.response.get("token_ids", [])

                    if triplet_idx == 0:
                        accumulated_full_sequence = orig_prompt_ids + response_token_ids
                        accumulated_response_mask += [1] * len(response_token_ids)
                        if self.cfg.rollout.return_logprobs:
                            logprobs = triplet.response.get("logprobs", [])
                            if logprobs:
                                accumulated_logprobs += [lp.get("logprob", 0.0) for lp in logprobs]
                    else:
                        tool_response_ids = prompt_token_ids[len(accumulated_full_sequence):]
                        accumulated_full_sequence.extend(tool_response_ids)
                        accumulated_full_sequence.extend(response_token_ids)
                        accumulated_response_mask += [0] * len(tool_response_ids)
                        accumulated_response_mask += [1] * len(response_token_ids)
                        if self.cfg.rollout.return_logprobs:
                            logprobs = triplet.response.get("logprobs", [])
                            if logprobs:
                                accumulated_logprobs += [0.0] * len(tool_response_ids)
                                accumulated_logprobs += [lp.get("logprob", 0.0) for lp in logprobs]
                
                response_ids = accumulated_full_sequence[len(orig_prompt_ids):]
                
                if len(response_ids) > max_response_length:
                    response_ids = response_ids[:max_response_length]
                    accumulated_response_mask = accumulated_response_mask[:max_response_length]
                    if self.cfg.rollout.return_logprobs:
                        accumulated_logprobs = accumulated_logprobs[:max_response_length]


                prompt_ids_list.append(orig_prompt_ids)
                response_ids_list.append(response_ids)
                prompt_lengths_list.append(len(orig_prompt_ids))
                response_lengths_list.append(len(response_ids))
                response_mask_list.append(accumulated_response_mask)
                is_end_list.append(True)
                
                reward = rollout_legacy.final_reward
                rewards_list.append(reward)
                
                if self.cfg.rollout.return_logprobs:
                    if len(accumulated_logprobs) > max_response_length:
                        accumulated_logprobs = accumulated_logprobs[:max_response_length]
                    rollout_logprobs_list.append(accumulated_logprobs)

            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
            
            prompt_texts_list = [self.tokenizer.decode(prompt_ids) for prompt_ids in prompt_ids_list]
            response_texts_list = [self.tokenizer.decode(response_ids) for response_ids in response_ids_list]
            rollout_result = RolloutResult(
                num_sequence=len(prompt_ids_list),
                group_size=len(prompt_ids_list),
                prompt_lengths=prompt_lengths_list,
                prompt_ids=prompt_ids_list,
                response_lengths=response_lengths_list,
                response_ids=response_ids_list,
                is_end=is_end_list,
                rewards=rewards_tensor,
                response_mask=response_mask_list,
                prompt_texts= prompt_texts_list,
                response_texts= response_texts_list,
                rollout_logprobs=rollout_logprobs_list if self.cfg.rollout.return_logprobs else None,
            )

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
            rollout_results: List[RolloutResult] = []
            
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
                        rollout_results.append(rollout_result)
                        output_channel.put(rollout_result, async_op=True)
                    
                    processed_data_ids.add(data_id)
                
                if len(processed_data_ids) < initial_data_ids_count:
                    await asyncio.sleep(0.1)
            
            rollouts_list = list(self._completed_rollout_ids.values())
            metrics = self._compute_rollout_metrics(rollout_results, rollouts_list)
            self._clear_data()
            return metrics

    async def process_eval_batch(
        self, input_channel: Channel
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
                    processed_data_ids.add(data_id)
                
                if len(processed_data_ids) < initial_data_ids_count:
                    await asyncio.sleep(0.1)
            
            all_rewards: List[float] = []
            for rollout_id, rollout_legacy in self._completed_rollout_ids.items():
                all_rewards.append(rollout_legacy.final_reward)
            
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            logging.info(f"Eval rewards: {all_rewards}, count: {len(all_rewards)}, avg: {avg_reward}")
            self._clear_data()
            return avg_reward

    def update_server_addresses(self, server_addresses: List[str]):
        self.server_addresses = server_addresses

