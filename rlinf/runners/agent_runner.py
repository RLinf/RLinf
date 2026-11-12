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
import threading
import typing
from typing import Optional, Union

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolChannelInfo, ToolWorker, ToolWorkerInfo

if typing.TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic
    from rlinf.workers.inference.fsdp_inference_worker import FSDPInference
    from rlinf.workers.inference.megatron_inference_worker import MegatronInference
    from rlinf.workers.reward.reward_worker import RewardWorker
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class AgentRunner(ReasoningRunner):
    """Runner for agent task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        rollout: Union["SGLangWorker", "VLLMWorker"],
        inference: Optional[Union["FSDPInference", "MegatronInference"]],
        actor: Union["FSDPActor", "MegatronActor"],
        reward: Optional["RewardWorker"],
        agent_loop: AgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, Union["SGLangWorker", "VLLMWorker"]] = {},
        critic: Optional["MAMegatronCritic"] = None,
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
            critic=critic,
        )
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
        self.batch_split_num = len(agent_loop._workers)
        self.tool_workers = tool_workers
        self.solid_rollouts = solid_rollouts
        self.generate_input_channel = Channel.create("GenerateInput")
        self.generate_output_channel = Channel.create("GenerateOutput")
        self.solid_generate_input_channels = {}
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
        if self.recompute_logprobs:
            self.inference_channel = Channel.create(
                "Inference", local=not self.has_dedicated_actor_inference
            )

    def _stop_rollouts(self):
        """Stop the main and solid rollout engines after training finishes."""
        super()._stop_rollouts()
        for solid_rollout in self.solid_rollouts.values():
            if hasattr(solid_rollout, "stop"):
                solid_rollout.stop().wait()

    def init_rollout_workers(self):
        """init rollout workers, tool workers and agent loop worker."""
        rollout_handles = [self.rollout.init_worker()]
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
                and not getattr(self.cfg.actor.megatron, "mbridge", False)
            ):
                from rlinf.utils.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                    convert_hf_to_mg,
                )

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
        super()._sync_weights()
        if not self.is_pipeline:
            onload_handles = []
            for solid_rollout in self.solid_rollouts.values():
                onload_handles.append(solid_rollout.onload_engine())
            for handle in onload_handles:
                handle.wait()

    def _run_training(
        self, rollout_handle: Handle
    ) -> tuple[dict | None, list[dict], dict]:
        """Update the actor and return rollout metrics, update metrics, and durations."""
        handles = {}
        input_channel = self.rollout_channel
        if self.reward is not None:
            handles["reward"] = self.reward.compute_rewards(
                input_channel=input_channel, output_channel=self.reward_channel
            )
            input_channel = self.reward_channel
        if self.recompute_logprobs:
            handles["inference"] = self.inference.run_inference(
                input_channel=input_channel,
                output_channel=self.inference_channel,
                compute_ref_logprobs=self.compute_ref_logprobs,
            )
            input_channel = self.inference_channel
        if self.is_pipeline:
            rollout_handle.wait()
        if self.recompute_logprobs and not self.has_dedicated_actor_inference:
            handles["inference"].wait()
        handles["training"] = self.actor.run_training(input_channel=input_channel)
        rollout_metrics, training_metrics = handles["training"].wait()[0]
        durations = {
            key: handle.consume_duration(reduction_type="min")
            if key == "inference"
            else handle.consume_duration()
            for key, handle in handles.items()
        }
        return rollout_metrics, training_metrics, durations

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0 or self.global_steps >= self.max_steps:
            # epoch done
            self._stop_rollouts()
            return

        tqdm.set_lock(threading.RLock())
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
                            self._put_batch(batch, self.batch_split_num)

                        with self.timer("sync_weights"):
                            self._sync_weights()

                        # Rollout
                        rollout_handle: Handle = self.agent_loop.run_agentloop_rollout(
                            input_channel=self.dataloader_channel,
                            output_channel=self.rollout_channel,
                        )

                        if not self.is_pipeline:
                            agent_metrics = rollout_handle.wait()[0]
                            offload_handles = [self.rollout.offload_engine()]
                            for solid_rollout in self.solid_rollouts.values():
                                offload_handles.append(solid_rollout.offload_engine())
                            for handle in offload_handles:
                                handle.wait()

                        (
                            actor_rollout_metrics,
                            actor_training_metrics,
                            training_times,
                        ) = self._run_training(rollout_handle)
                        agent_metrics = rollout_handle.wait()[0]
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
                    time_metrics.update(training_times)
                    time_metrics["rollout"] = rollout_handle.consume_duration()

                    logging_steps = (
                        self.global_steps - 1
                    ) * self.cfg.algorithm.n_minibatches
                    # add prefix to the metrics
                    log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

                    self.metric_logger.log(agent_metrics, logging_steps)
                    self.metric_logger.log(log_time_metrics, logging_steps)
                    if actor_rollout_metrics is not None:
                        rollout_metrics = {
                            f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
                        }
                        self.metric_logger.log(rollout_metrics, logging_steps)
                    for i, metrics in enumerate(actor_training_metrics):
                        training_metrics = {f"train/{k}": v for k, v in metrics.items()}
                        self.metric_logger.log(training_metrics, logging_steps + i)

                    logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}

                    logging_metrics.update(agent_metrics)
                    if actor_rollout_metrics is not None:
                        logging_metrics.update(actor_rollout_metrics)
                    if actor_training_metrics:
                        logging_metrics.update(actor_training_metrics[-1])

                    global_pbar.set_postfix(logging_metrics, refresh=False)
                    global_pbar.update(1)

                    if is_train_end or run_time_exceeded:
                        self._stop_rollouts()
                        return

            self._stop_rollouts()
        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()
            global_pbar.close()
            self.metric_logger.finish()
