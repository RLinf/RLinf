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
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolChannelInfo, ToolWorker, ToolWorkerInfo
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
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
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        reward: RewardWorker,
        agent_loop: AgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
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
        self.generate_input_channel = Channel.create("GenerateInput")
        self.generate_output_channel = Channel.create("GenerateOutput")
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

    def init_workers(self):
        """init tool workers and agent loop worker."""
        for worker in self.tool_workers:
            input_channel = self.tool_channel_info_map[
                worker.worker_group_name
            ].input_channel
            worker.init_worker(input_channel, self.tool_output_channel).wait()

        self.agent_loop.init_worker(
            self.generate_input_channel,
            self.generate_output_channel,
            self.tool_channel_info_map,
            self.tool_name_map,
            self.tool_output_channel,
        ).wait()

        super().init_workers()

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

                        # Rewards
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.rollout_channel,
                            output_channel=self.reward_channel,
                        )

                        if self.recompute_logprobs:
                            # Inference prev/ref logprobs
                            infer_handle: Handle = self.inference.run_inference(
                                input_channel=self.reward_channel,
                                output_channel=self.inference_channel,
                                compute_ref_logprobs=self.compute_ref_logprobs,
                            )
                            inference_channel = self.inference_channel
                        else:
                            infer_handle = None
                            inference_channel = self.reward_channel

                        # Actor training, Advantages and returns
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
                    time_metrics["reward"] = reward_handle.consume_duration()
                    if infer_handle is not None:
                        # Inference time should be the min time across ranks, because different DP receive the rollout results differently
                        # But at the begin of the pp schedule, there is a timer barrier
                        # This makes all DP end at the same time, while they start at differnt times, and thus only the min time is correct
                        time_metrics["inference"] = infer_handle.consume_duration(
                            reduction_type="min"
                        )

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
                        training_metrics = {
                            f"train/{k}": v
                            for k, v in actor_training_metrics[i].items()
                        }
                        self.metric_logger.log(training_metrics, logging_steps + i)

                    logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}

                    if self.cfg.actor.get("calculate_flops", False):
                        flops_metrics = self._compute_flops_metrics(
                            time_metrics, actor_rollout_metrics
                        )
                        flops_metrics = {
                            f"flops/{k}": v for k, v in flops_metrics.items()
                        }
                        self.metric_logger.log(flops_metrics, logging_steps)
                        logging_metrics.update(flops_metrics)

                    logging_metrics.update(actor_rollout_metrics)
                    logging_metrics.update(actor_training_metrics[-1])

                    global_pbar.set_postfix(logging_metrics, refresh=False)
                    global_pbar.update(1)
        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()
            self.metric_logger.finish()
