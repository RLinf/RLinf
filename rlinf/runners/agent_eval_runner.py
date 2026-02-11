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
import json
import logging
import os
import typing
from typing import Optional, Union

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from rlinf.data.io_struct import RolloutResult
from rlinf.runners.reasoning_eval_runner import ReasoningEvalRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import local_mkdir_safe
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolChannelInfo, ToolWorker, ToolWorkerInfo
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class AgentEvalRunner(ReasoningEvalRunner):
    """Runner for agent task RL evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        reward: Optional[RewardWorker],
        agent_loop: AgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, Union["SGLangWorker", "VLLMWorker"]] = {},
    ):
        super().__init__(
            cfg,
            placement,
            val_dataset,
            rollout,
            reward,
        )
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

        for rollout_handle in rollout_handles:
            rollout_handle.wait()
        self.agent_loop.init_worker(
            self.generate_input_channel,
            self.generate_output_channel,
            self.tool_channel_info_map,
            self.tool_name_map,
            self.tool_output_channel,
            self.solid_generate_input_channels,
        ).wait()

    def run(self):
        """Run evaluation on validation dataset.

        This function:
        1. Runs rollout on validation data (accumulates raw results)
        2. Aggregates all results after all batches complete
        3. Saves results to files
        """
        context = self.pre_process()

        # Initialize progress bar
        eval_pbar = tqdm(
            total=len(self.val_dataloader),
            desc="Evaluation",
            ncols=100,
        )

        # Start rollout server and tool workers
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
            # Process validation batches
            for batch_idx, batch in enumerate(self.val_dataloader):
                logging.info(
                    f"\nProcessing batch {batch_idx + 1}/{len(self.val_dataloader)}"
                )

                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    # Rollout
                    rollout_handle: Handle = self.agent_loop.run_agentloop_rollout(
                        input_channel=self.dataloader_channel,
                        output_channel=self.rollout_channel,
                    )

                    # Rewards
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.rollout_channel,
                            output_channel=self.reward_channel,
                        )
                        eval_input_channel = self.reward_channel
                    else:
                        eval_input_channel = self.rollout_channel

                    # Accumulate and log results
                    self.log_eval(
                        context,
                        input_channel=eval_input_channel,
                        batch_idx=batch_idx,
                        batch=batch,
                    )

                    # Wait for all handles to complete
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                time_metrics = self.timer.consume_durations()
                time_metrics["rollout"] = rollout_handle.consume_duration()
                if self.reward is not None:
                    time_metrics["reward"] = reward_handle.consume_duration()

                self.update_pbar(context, eval_pbar, time_metrics)

                self.global_steps += 1

        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()

        eval_pbar.close()
        self.post_process(context)

        self.metric_logger.finish()

    def pre_process(self) -> dict:
        raise NotImplementedError()

    def log_eval(
        self,
        context: dict,
        input_channel,
        batch_idx,
        batch,
    ):
        raise NotImplementedError()

    def post_process(
        self,
        context: dict,
    ) -> dict:
        raise NotImplementedError()

    def update_pbar(
        self,
        context: dict,
        eval_pbar,
        time_metrics,
    ):
        # Update progress bar with current metrics
        eval_pbar.set_postfix(
            {
                "rollout_time": f"{time_metrics.get('rollout', 0):.2f}s",
            }
        )
        eval_pbar.update(1)
