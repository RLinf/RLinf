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

from rlinf.runners.reasoning_eval_runner import ReasoningRunnerEval
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolChannelInfo, ToolWorker, ToolWorkerInfo

from rlinf.workers.reward.reward_worker import RewardWorker

from rlinf.data.io_struct import RolloutResult

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class MASToolAgentRunnerEval(ReasoningRunnerEval):
    """Runner for agent task RL training."""

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
        if self.solid_rollouts is not None:
            for solid_rollout_name in self.solid_rollouts:
                self.solid_generate_input_channels[solid_rollout_name] = Channel.create(f"SolidRolloutInput-{solid_rollout_name}")        
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

        # Initialize storage for accumulating evaluation results across all batches
        self.accumulated_results = []

    def _save_eval_results(self, all_results, aggregated_metrics, total_count):
        """Save evaluation results to JSON file.

        Args:
            all_results: List of result dictionaries for each query
            aggregated_metrics: Dictionary with 9 aggregated metrics
            total_count: Total number of queries evaluated
        """
        # Create output directory in the experiment folder
        output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)

        # Fixed filename (no timestamp)
        output_file_key = os.path.join(output_dir, "metrics.json")
        output_file_all = os.path.join(output_dir, "allresults.json")

        # Prepare timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert OmegaConf objects to plain Python objects
        eval_metrics = self.cfg.reward.get("eval_metric", [])
        if eval_metrics:
            eval_metrics = OmegaConf.to_container(eval_metrics, resolve=True)

        data_paths = self.cfg.data.val_data_paths
        if data_paths:
            data_paths = OmegaConf.to_container(data_paths, resolve=True)

        # Prepare complete results structure with 9 key metrics
        results_data_key = {
            "dataset_size": total_count,
            "experiment_name": self.cfg.runner.experiment_name,
            "timestamp": timestamp,
            "config": {
                "group_size": self.cfg.algorithm.get("group_size", 1),
                "max_turns": self.cfg.agentloop.get("max_turns", 5),
                "eval_metrics": eval_metrics,
                "reward_type": self.cfg.reward.get("reward_type", "EM"),
                "data_paths": data_paths,
            },
            # 9 key metrics: 3 methods (pass@1, pass@k, avg@k) × 3 metrics (EM, F1, LLM)
            "metrics": aggregated_metrics
        }  

        # Write results to JSON with readable formatting
        with open(output_file_key, 'w', encoding='utf-8') as f:
            json.dump(results_data_key, f, ensure_ascii=False, indent=2)

        with open(output_file_all, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)            

        logging.info(f"Evaluation results saved to: {output_file_key}")
        return output_file_key

    def init_workers(self):
        """init tool workers and agent loop worker."""
        for worker in self.tool_workers:
            input_channel = self.tool_channel_info_map[
                worker.worker_group_name
            ].input_channel
            worker.init_worker(input_channel, self.tool_output_channel).wait()

        if self.solid_rollouts is not None:
            for solid_rollout in self.solid_rollouts.values():
                solid_rollout.init_worker_no_sync().wait()

        self.agent_loop.init_worker(
            self.generate_input_channel,
            self.generate_output_channel,
            self.tool_channel_info_map,
            self.tool_name_map,
            self.tool_output_channel,
            self.solid_generate_input_channels,
        ).wait()

        super().init_workers()

    def log_eval(self, input_channel, expected_batch_size=None):
        """Collect evaluation results and compute metrics for a single batch.

        This function:
        1. Collects rollout results from the rollout channel for one batch
        2. Computes pass@1, pass@k, avg@k metrics for all eval metrics (EM, F1, LLM)
        3. Accumulates results (does NOT save to file yet)

        Note: For MAS eval, we receive evaluation dictionary directly from agent loop
        with human-readable text and multiple samples (group_size=k).

        Args:
            input_channel: The channel to receive rollout results from
            expected_batch_size: Expected number of samples in this batch.
                                If None, uses rollout_batch_size from config.
        """
        recv_batch_size = 0
        world_size = 1

        # For eval, group_size can be k (for pass@k)
        group_size = self.cfg.algorithm.get("group_size", 1)

        # Use provided expected_batch_size or fall back to config
        if expected_batch_size is not None:
            total_batch_size_per_dp = expected_batch_size
        else:
            total_batch_size_per_dp = (
                self.cfg.data.rollout_batch_size * group_size // world_size
            )

        # Storage for this batch's results
        batch_results = []

        # Counters for pass@1 and pass@k metrics
        # We track 3 evaluation methods (EM, F1, LLM), each with pass@1, pass@k, avg@k
        metric_types = ["EM", "F1", "LLM"]

        # Initialize counters for each metric
        pass1_correct = {m: 0 for m in metric_types}  # pass@1
        passk_correct = {m: 0 for m in metric_types}  # pass@k
        avgk_sum = {m: 0.0 for m in metric_types}     # avg@k

        total_queries = 0

        while recv_batch_size < total_batch_size_per_dp:
            # Receive evaluation dictionary with k samples
            eval_result: dict = input_channel.get()

            recv_batch_size += 1  # Each eval_result is one query with k samples
            total_queries += 1

            group_size = eval_result.get("group_size", 1)
            answer = eval_result.get("answer", None)
            samples = eval_result.get("samples", [])

            # Collect length metrics and tool call metrics from all samples
            prompt_lengths = []
            response_lengths = []
            total_lengths = []
            num_turns_list = []
            subtask_counts = []
            search_counts = []
            access_counts = []
            num_valid_planner_turns = 0
            num_valid_worker_turns = 0

            for sample in samples:
                turns = sample.get("turns", [])
                num_turns_list.append(len(turns))

                for turn in turns:
                    prompt_len = turn.get("prompt_ids_length", 0)
                    response_len = turn.get("response_ids_length", 0)
                    prompt_lengths.append(prompt_len)
                    response_lengths.append(response_len)
                    total_lengths.append(prompt_len + response_len)

                # Collect tool call counts from sample
                # Track valid turns: only count turns that actually made tool calls
                for turn in turns:
                    tool_call_info = turn.get("tool_call_info", None)
                    if tool_call_info is not None:
                        role = tool_call_info.get("role", "")
                        subtask_count = tool_call_info.get("subtask", 0)
                        search_count = tool_call_info.get("search", 0)
                        access_count = tool_call_info.get("access", 0)

                        subtask_counts.append(subtask_count)
                        search_counts.append(search_count)
                        access_counts.append(access_count)

                        # Track valid turns by role (same logic as in agent_loop.py)
                        if role == "planner":
                            assert subtask_count > 0
                            num_valid_planner_turns += 1
                        elif role == "worker" or role == "single":
                            assert search_count > 0 or access_count > 0
                            num_valid_worker_turns += 1

            # Total number of turns for this query
            total_num_turns = len(prompt_lengths)

            # Compute length metrics statistics
            # Store sum values for weighted averaging
            length_metrics = {}
            length_metrics["num_turns"] = total_num_turns
            if prompt_lengths:
                length_metrics["sum_prompt_length"] = sum(prompt_lengths)
                length_metrics["sum_response_length"] = sum(response_lengths)
                length_metrics["sum_total_length"] = sum(total_lengths)
                length_metrics["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
                length_metrics["max_prompt_length"] = max(prompt_lengths)
                length_metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)
                length_metrics["max_response_length"] = max(response_lengths)
                length_metrics["avg_total_length"] = sum(total_lengths) / len(total_lengths)
                length_metrics["max_total_length"] = max(total_lengths)
            else:
                length_metrics["sum_prompt_length"] = 0
                length_metrics["sum_response_length"] = 0
                length_metrics["sum_total_length"] = 0
                length_metrics["avg_prompt_length"] = 0.0
                length_metrics["max_prompt_length"] = 0
                length_metrics["avg_response_length"] = 0.0
                length_metrics["max_response_length"] = 0
                length_metrics["avg_total_length"] = 0.0
                length_metrics["max_total_length"] = 0

            # Compute trajectory metrics
            trajectory_metrics = {}
            trajectory_metrics["num_trajectories"] = group_size
            trajectory_metrics["total_num_turns"] = sum(num_turns_list)
            if num_turns_list:
                trajectory_metrics["avg_turns_per_traj"] = sum(num_turns_list) / len(num_turns_list)
                trajectory_metrics["turns_per_traj"] = num_turns_list
            else:
                trajectory_metrics["avg_turns_per_traj"] = 0.0
                trajectory_metrics["turns_per_traj"] = []

            # Compute tool call metrics (turn-level and trajectory-level)
            # IMPORTANT: Turn-level averages should be computed over VALID turns only
            tool_call_metrics = {}
            tool_call_metrics["num_valid_planner_turns"] = num_valid_planner_turns
            tool_call_metrics["num_valid_worker_turns"] = num_valid_worker_turns

            # Store sum values for weighted averaging in global aggregation
            turn_sum_subtask = sum(subtask_counts)
            turn_sum_search = sum(search_counts)
            turn_sum_access = sum(access_counts)

            tool_call_metrics["turn_sum_subtask"] = turn_sum_subtask
            tool_call_metrics["turn_sum_search"] = turn_sum_search
            tool_call_metrics["turn_sum_access"] = turn_sum_access

            if subtask_counts or search_counts or access_counts:
                # Turn-level metrics (only over valid turns that made tool calls)
                # For planner: divide by num_valid_planner_turns
                # For worker: divide by num_valid_worker_turns
                if num_valid_planner_turns > 0:
                    # Compute average subtask calls per valid planner turn
                    tool_call_metrics["turn_avg_subtask"] = turn_sum_subtask / num_valid_planner_turns
                    tool_call_metrics["turn_max_subtask"] = max(subtask_counts) if subtask_counts else 0
                else:
                    tool_call_metrics["turn_avg_subtask"] = 0.0
                    tool_call_metrics["turn_max_subtask"] = 0

                if num_valid_worker_turns > 0:
                    # Compute average search/access calls per valid worker turn
                    tool_call_metrics["turn_avg_search"] = turn_sum_search / num_valid_worker_turns
                    tool_call_metrics["turn_max_search"] = max(search_counts) if search_counts else 0
                    tool_call_metrics["turn_avg_access"] = turn_sum_access / num_valid_worker_turns
                    tool_call_metrics["turn_max_access"] = max(access_counts) if access_counts else 0

                    # Search + Access combined metrics
                    search_plus_access_counts = [s + a for s, a in zip(search_counts, access_counts)]
                    tool_call_metrics["turn_avg_search+access"] = sum(search_plus_access_counts) / num_valid_worker_turns
                    tool_call_metrics["turn_max_search+access"] = max(search_plus_access_counts) if search_plus_access_counts else 0
                    tool_call_metrics["turn_sum_search+access"] = sum(search_plus_access_counts)
                else:
                    tool_call_metrics["turn_avg_search"] = 0.0
                    tool_call_metrics["turn_max_search"] = 0
                    tool_call_metrics["turn_avg_access"] = 0.0
                    tool_call_metrics["turn_max_access"] = 0
                    tool_call_metrics["turn_avg_search+access"] = 0.0
                    tool_call_metrics["turn_max_search+access"] = 0
                    tool_call_metrics["turn_sum_search+access"] = 0

                # Trajectory-level metrics: aggregate counts per trajectory
                # For simplicity, we'll compute trajectory totals by dividing by num_trajectories
                traj_subtask_total = turn_sum_subtask
                traj_search_total = turn_sum_search
                traj_access_total = turn_sum_access

                tool_call_metrics["traj_avg_subtask"] = traj_subtask_total / group_size if group_size > 0 else 0.0
                tool_call_metrics["traj_avg_search"] = traj_search_total / group_size if group_size > 0 else 0.0
                tool_call_metrics["traj_avg_access"] = traj_access_total / group_size if group_size > 0 else 0.0
                tool_call_metrics["traj_avg_search+access"] = (traj_search_total + traj_access_total) / group_size if group_size > 0 else 0.0

                # Trajectory-level max (max across all trajectories)
                # Since we're aggregating from turns, we use max of turn counts
                tool_call_metrics["traj_max_subtask"] = max(subtask_counts) if subtask_counts else 0
                tool_call_metrics["traj_max_search"] = max(search_counts) if search_counts else 0
                tool_call_metrics["traj_max_access"] = max(access_counts) if access_counts else 0
                if search_counts and access_counts:
                    search_plus_access_counts = [s + a for s, a in zip(search_counts, access_counts)]
                    tool_call_metrics["traj_max_search+access"] = max(search_plus_access_counts)
                else:
                    tool_call_metrics["traj_max_search+access"] = 0
            else:
                tool_call_metrics["turn_avg_subtask"] = 0.0
                tool_call_metrics["turn_max_subtask"] = 0
                tool_call_metrics["turn_avg_search"] = 0.0
                tool_call_metrics["turn_max_search"] = 0
                tool_call_metrics["turn_avg_access"] = 0.0
                tool_call_metrics["turn_max_access"] = 0
                tool_call_metrics["turn_avg_search+access"] = 0.0
                tool_call_metrics["turn_max_search+access"] = 0
                tool_call_metrics["turn_sum_search+access"] = 0
                tool_call_metrics["traj_avg_subtask"] = 0.0
                tool_call_metrics["traj_avg_search"] = 0.0
                tool_call_metrics["traj_avg_access"] = 0.0
                tool_call_metrics["traj_avg_search+access"] = 0.0
                tool_call_metrics["traj_max_subtask"] = 0
                tool_call_metrics["traj_max_search"] = 0
                tool_call_metrics["traj_max_access"] = 0
                tool_call_metrics["traj_max_search+access"] = 0

            # Check if we're in markdown mode
            is_markdown = self.cfg.data.get("is_markdown", False)

            if is_markdown:
                # For markdown, eval_metric has nested structure:
                # {"EM": {"score": ..., "precision_by_row": ..., ...}, "F1": {...}, "LLM": {...}}

                # Markdown metrics: 10 metrics per evaluation method
                # 7 standard + 3 search-specific (non-primary-key columns only)
                markdown_metrics = [
                    "score",
                    "precision_by_row",
                    "recall_by_row",
                    "f1_by_row",
                    "precision_by_item",
                    "recall_by_item",
                    "f1_by_item",
                    "search_precision_by_item",
                    "search_recall_by_item",
                    "search_f1_by_item"
                ]

                # Initialize storage for all metrics
                query_metrics = {}
                for metric_type in metric_types:
                    query_metrics[metric_type] = {m: [] for m in markdown_metrics}

                # Collect from all samples
                for sample in samples:
                    eval_metric = sample.get("eval_metric", {})
                    for metric_type in metric_types:
                        method_metrics = eval_metric.get(metric_type, {})
                        for md_metric in markdown_metrics:
                            value = method_metrics.get(md_metric, 0.0)
                            query_metrics[metric_type][md_metric].append(value)

                # Compute pass@1, pass@k/avg@k/max@k for each (metric_type, md_metric) pair
                query_pass1 = {}
                query_passk = {}
                query_avgk = {}
                query_maxk = {}

                for metric_type in metric_types:
                    query_pass1[metric_type] = {}
                    query_passk[metric_type] = {}
                    query_avgk[metric_type] = {}
                    query_maxk[metric_type] = {}

                    for md_metric in markdown_metrics:
                        values = query_metrics[metric_type][md_metric]
                        if len(values) > 0:
                            # For "score": compute pass@1, pass@k, avg@k
                            if md_metric == "score":
                                query_pass1[metric_type][md_metric] = values[0] > 0
                                query_passk[metric_type][md_metric] = any(v > 0 for v in values)
                                query_avgk[metric_type][md_metric] = sum(values) / len(values)
                                query_maxk[metric_type][md_metric] = 0.0  # Not used for score
                            # For other 6 metrics: compute pass@1, avg@k, max@k
                            else:
                                query_pass1[metric_type][md_metric] = values[0]
                                query_passk[metric_type][md_metric] = 0.0  # Not used for non-score
                                query_avgk[metric_type][md_metric] = sum(values) / len(values)
                                query_maxk[metric_type][md_metric] = max(values)
                        else:
                            query_pass1[metric_type][md_metric] = 0.0
                            query_passk[metric_type][md_metric] = 0.0
                            query_avgk[metric_type][md_metric] = 0.0
                            query_maxk[metric_type][md_metric] = 0.0

            else:
                # Original non-markdown logic
                query_metrics = {
                    "EM": [],
                    "F1": [],
                    "LLM": []
                }

                # Collect metrics from all samples
                for sample in samples:
                    eval_metric = sample.get("eval_metric", {})
                    for metric_type in metric_types:
                        score = eval_metric.get(metric_type, None)
                        if score is not None:
                            query_metrics[metric_type].append(score)

                # Compute metrics for each type
                query_pass1 = {}
                query_passk = {}
                query_avgk = {}
                query_maxk = {}

                for metric_type in metric_types:
                    scores = query_metrics[metric_type]
                    if len(scores) > 0:
                        # pass@1: use first sample (all metrics)
                        query_pass1[metric_type] = scores[0] > 0
                        if query_pass1[metric_type]: # FIXME: F1 score do not need pass@1
                            pass1_correct[metric_type] += 1

                        # avg@k: average of all k samples (all metrics)
                        query_avgk[metric_type] = sum(scores) / len(scores)
                        avgk_sum[metric_type] += query_avgk[metric_type]

                        # For F1: use max@k instead of pass@k
                        if metric_type == "F1":
                            query_passk[metric_type] = 0.0  # Not used for F1
                            query_maxk[metric_type] = max(scores)
                        else:
                            # For EM and LLM: use pass@k
                            query_passk[metric_type] = any(s > 0 for s in scores)
                            if query_passk[metric_type]:
                                passk_correct[metric_type] += 1
                            query_maxk[metric_type] = 0.0  # Not used for EM/LLM
                    else:
                        query_pass1[metric_type] = False
                        query_passk[metric_type] = False
                        query_avgk[metric_type] = 0.0
                        query_maxk[metric_type] = 0.0

            # Create result entry
            result_entry = {
                "index": len(self.accumulated_results),  # Use global index
                "group_size": group_size,
                "answer": answer,
                "samples": samples,  # All k samples with their data
                # Per-query aggregated metrics
                "pass@1": query_pass1,
                "pass@k": query_passk,
                "avg@k": query_avgk,
                "max@k": query_maxk,  # For markdown metrics
                # Length metrics
                "length_metrics": length_metrics,
                # Trajectory metrics
                "trajectory_metrics": trajectory_metrics,
                # Tool call metrics
                "tool_call_metrics": tool_call_metrics,
            }

            batch_results.append(result_entry)
            self.accumulated_results.append(result_entry)

        assert recv_batch_size == total_batch_size_per_dp, (
            f"Expected {total_batch_size_per_dp} queries from channel, but got {recv_batch_size}"
        )

        if is_markdown:
            primary_accuracy = (
                sum(result["avg@k"].get("EM", {}).get("f1_by_item", 0.0) for result in batch_results) / total_queries
                if total_queries > 0 else 0.0
            )

        else:
            # Primary accuracy is EM pass@1
            primary_accuracy = (
                sum(1.0 if result["pass@1"].get("EM", False) else 0.0 for result in batch_results) / total_queries
                if total_queries > 0 else 0.0
            )

        return primary_accuracy

    def calc_markdown_summary(self, all_results):
        metric_types = ["EM", "F1", "LLM"]
        markdown_metrics = [
            "score",
            "precision_by_row",
            "recall_by_row",
            "f1_by_row",
            "precision_by_item",
            "recall_by_item",
            "f1_by_item",
            "search_precision_by_item",
            "search_recall_by_item",
            "search_f1_by_item"
        ]

        summary = {}
        for metric_type in metric_types:
            summary[metric_type] = {}
            for md_metric in markdown_metrics:
                # Collect values across all queries
                if md_metric == "score":
                    pass1_values = []
                    passk_values = []
                    avgk_values = []

                    for result in all_results:
                        metrics = result.get("pass@1", {}).get(metric_type, {})
                        pass1_values.append(1.0 if metrics.get(md_metric, False) else 0.0)

                        metrics = result.get("pass@k", {}).get(metric_type, {})
                        passk_values.append(1.0 if metrics.get(md_metric, False) else 0.0)

                        metrics = result.get("avg@k", {}).get(metric_type, {})
                        avgk_values.append(metrics.get(md_metric, 0.0))

                    summary[metric_type][md_metric] = {
                        "pass@1": sum(pass1_values) / len(pass1_values) if pass1_values else 0.0,
                        "pass@k": sum(passk_values) / len(passk_values) if passk_values else 0.0,
                        "avg@k": sum(avgk_values) / len(avgk_values) if avgk_values else 0.0
                    }
                else:
                    pass1_values = []
                    avgk_values = []
                    maxk_values = []

                    for result in all_results:
                        metrics = result.get("pass@1", {}).get(metric_type, {})
                        pass1_values.append(metrics.get(md_metric, 0.0))

                        metrics = result.get("avg@k", {}).get(metric_type, {})
                        avgk_values.append(metrics.get(md_metric, 0.0))

                        metrics = result.get("max@k", {}).get(metric_type, {})
                        maxk_values.append(metrics.get(md_metric, 0.0))

                    summary[metric_type][md_metric] = {
                        "pass@1": sum(pass1_values) / len(pass1_values) if pass1_values else 0.0,
                        "avg@k": sum(avgk_values) / len(avgk_values) if avgk_values else 0.0,
                        "max@k": sum(maxk_values) / len(maxk_values) if maxk_values else 0.0
                    }

        return summary

    def run(self):
        """Run evaluation on validation dataset.

        This function:
        1. Runs rollout on validation data
        2. Computes rewards
        3. Collects and logs evaluation results
        """
        logging.info("=" * 80)
        logging.info("Starting Multi-Agent System Evaluation")
        logging.info("=" * 80)
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Batch size: {self.cfg.data.val_rollout_batch_size}")
        logging.info(f"Group size: {self.cfg.algorithm.get('group_size', 1)}")
        logging.info(f"Max turns: {self.cfg.agentloop.get('max_turns', 5)}")
        logging.info("=" * 80)

        # Initialize progress bar
        eval_pbar = tqdm(
            total=len(self.val_dataloader),
            desc="Evaluation",
            ncols=100,
        )

        # Start rollout server
        self.run_timer.start_time()
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        if self.solid_rollouts is not None:
            for solid_rollout_name, solid_rollout in self.solid_rollouts.items():
                solid_rollout.rollout_serverless(
                    self.solid_generate_input_channels[solid_rollout_name],
                    self.generate_output_channel,
                )
        # Start tool workers
        for tool_worker in self.tool_workers:
            tool_worker.start_server()

        # Aggregate results across all batches
        all_batch_results = []
        metric_types = ["EM", "F1", "LLM"]

        # Initialize global counters
        # For EM and LLM: pass@1, pass@k, avg@k
        # For F1: pass@1, avg@k, max@k
        global_pass1_correct = {m: 0 for m in metric_types}
        global_passk_correct = {m: 0 for m in metric_types}
        global_avgk_sum = {m: 0.0 for m in metric_types}
        global_maxk_sum = {m: 0.0 for m in metric_types}  # For F1
        total_queries = 0

        try:
            # Process validation batches
            for batch_idx, batch in enumerate(self.val_dataloader):
                logging.info(f"\nProcessing batch {batch_idx + 1}/{len(self.val_dataloader)}")

                # Calculate actual batch size from the batch data
                # The batch dict should have a 'prompt' tensor or 'answer' list
                if 'prompt' in batch:
                    actual_batch_size = batch['prompt'].shape[0]
                elif 'answer' in batch:
                    actual_batch_size = len(batch['answer'])
                else:
                    # Fallback to config value
                    actual_batch_size = self.cfg.data.val_rollout_batch_size

                logging.info(f"Actual batch size for this batch: {actual_batch_size}")

                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    rollout_handle: Handle = self.agent_loop.run_agentloop_rollout(
                        input_channel=self.dataloader_channel,
                        output_channel=self.rollout_channel,
                        is_eval=True,
                    )

                    batch_accuracy = self.log_eval(
                        input_channel=self.rollout_channel,
                        expected_batch_size=actual_batch_size
                    )

                    rollout_handle.wait()

                    total_queries = len(self.accumulated_results)

                time_metrics = self.timer.consume_durations()
                time_metrics["rollout"] = rollout_handle.consume_duration()

                eval_pbar.set_postfix({
                    "batch_acc": f"{batch_accuracy:.4f}",
                    "queries": total_queries,
                    "rollout_time": f"{time_metrics.get('rollout', 0):.2f}s",
                })
                eval_pbar.update(1)

                self.global_steps += 1

        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()

        eval_pbar.close()

        is_markdown = self.cfg.data.get("is_markdown", False)

        if is_markdown:
            # Use markdown-specific summary calculation
            final_metrics = self.calc_markdown_summary(self.accumulated_results)

        else:
            for result in self.accumulated_results:
                for metric_type in metric_types:
                    if result["pass@1"].get(metric_type, False):
                        global_pass1_correct[metric_type] += 1

                    # For F1: use max@k instead of pass@k
                    if metric_type == "F1":
                        global_maxk_sum[metric_type] += result["max@k"].get(metric_type, 0.0)
                    else:
                        # For EM and LLM: use pass@k
                        if result["pass@k"].get(metric_type, False):
                            global_passk_correct[metric_type] += 1

                    global_avgk_sum[metric_type] += result["avg@k"].get(metric_type, 0.0)

            final_metrics = {}
            for metric_type in metric_types:
                final_metrics[f"pass@1_{metric_type}"] = global_pass1_correct[metric_type] / total_queries if total_queries > 0 else 0.0
                final_metrics[f"avg@k_{metric_type}"] = global_avgk_sum[metric_type] / total_queries if total_queries > 0 else 0.0

                if metric_type == "F1":
                    final_metrics[f"max@k_{metric_type}"] = global_maxk_sum[metric_type] / total_queries if total_queries > 0 else 0.0
                else:
                    final_metrics[f"pass@k_{metric_type}"] = global_passk_correct[metric_type] / total_queries if total_queries > 0 else 0.0

        # Aggregate length, trajectory, and tool call metrics across all queries
        additional_metrics = {}

        # Aggregate for weighted averaging
        sum_prompt_lengths = 0
        sum_response_lengths = 0
        sum_total_lengths = 0
        total_num_turns = 0
        max_prompt_lengths = []
        max_response_lengths = []
        max_total_lengths = []

        # Aggregate trajectory metrics
        num_trajectories_list = []
        total_turns_list = []

        # Aggregate tool call metrics - use sum for weighted averaging
        turn_sum_subtask = 0
        turn_sum_search = 0
        turn_sum_access = 0
        turn_sum_search_plus_access = 0
        num_valid_planner_turns = 0
        num_valid_worker_turns = 0

        turn_max_subtasks = []
        turn_max_searches = []
        turn_max_accesses = []
        turn_max_search_plus_access_list = []

        traj_avg_subtasks = []
        traj_avg_searches = []
        traj_avg_accesses = []
        traj_avg_search_plus_access_list = []
        traj_max_subtasks = []
        traj_max_searches = []
        traj_max_accesses = []
        traj_max_search_plus_access_list = []

        for result in self.accumulated_results:
            # Length metrics - use weighted averaging
            length_metrics = result.get("length_metrics", {})
            sum_prompt_lengths += length_metrics.get("sum_prompt_length", 0)
            sum_response_lengths += length_metrics.get("sum_response_length", 0)
            sum_total_lengths += length_metrics.get("sum_total_length", 0)
            total_num_turns += length_metrics.get("num_turns", 0)
            max_prompt_lengths.append(length_metrics.get("max_prompt_length", 0))
            max_response_lengths.append(length_metrics.get("max_response_length", 0))
            max_total_lengths.append(length_metrics.get("max_total_length", 0))

            # Trajectory metrics
            traj_metrics = result.get("trajectory_metrics", {})
            num_trajectories_list.append(traj_metrics.get("num_trajectories", 0))
            total_turns_list.append(traj_metrics.get("total_num_turns", 0))

            # Tool call metrics - use weighted averaging for turn-level
            tool_metrics = result.get("tool_call_metrics", {})
            num_valid_planner_turns += tool_metrics.get("num_valid_planner_turns", 0)
            num_valid_worker_turns += tool_metrics.get("num_valid_worker_turns", 0)

            turn_sum_subtask += tool_metrics.get("turn_sum_subtask", 0)
            turn_sum_search += tool_metrics.get("turn_sum_search", 0)
            turn_sum_access += tool_metrics.get("turn_sum_access", 0)
            turn_sum_search_plus_access += tool_metrics.get("turn_sum_search+access", 0)

            turn_max_subtasks.append(tool_metrics.get("turn_max_subtask", 0))
            turn_max_searches.append(tool_metrics.get("turn_max_search", 0))
            turn_max_accesses.append(tool_metrics.get("turn_max_access", 0))
            turn_max_search_plus_access_list.append(tool_metrics.get("turn_max_search+access", 0))

            # Trajectory-level metrics
            traj_avg_subtasks.append(tool_metrics.get("traj_avg_subtask", 0.0))
            traj_avg_searches.append(tool_metrics.get("traj_avg_search", 0.0))
            traj_avg_accesses.append(tool_metrics.get("traj_avg_access", 0.0))
            traj_avg_search_plus_access_list.append(tool_metrics.get("traj_avg_search+access", 0.0))
            traj_max_subtasks.append(tool_metrics.get("traj_max_subtask", 0))
            traj_max_searches.append(tool_metrics.get("traj_max_search", 0))
            traj_max_accesses.append(tool_metrics.get("traj_max_access", 0))
            traj_max_search_plus_access_list.append(tool_metrics.get("traj_max_search+access", 0))

        # Compute global aggregates
        if total_queries > 0:
            # Length metrics - weighted average by number of turns
            if total_num_turns > 0:
                additional_metrics["avg_prompt_length"] = sum_prompt_lengths / total_num_turns
                additional_metrics["avg_response_length"] = sum_response_lengths / total_num_turns
                additional_metrics["avg_total_length"] = sum_total_lengths / total_num_turns
            else:
                additional_metrics["avg_prompt_length"] = 0.0
                additional_metrics["avg_response_length"] = 0.0
                additional_metrics["avg_total_length"] = 0.0

            additional_metrics["max_prompt_length"] = max(max_prompt_lengths) if max_prompt_lengths else 0
            additional_metrics["max_response_length"] = max(max_response_lengths) if max_response_lengths else 0
            additional_metrics["max_total_length"] = max(max_total_lengths) if max_total_lengths else 0

            # Trajectory metrics
            additional_metrics["total_num_trajectories"] = sum(num_trajectories_list)
            total_turns_all = sum(total_turns_list)
            additional_metrics["avg_turns_per_traj"] = total_turns_all / sum(num_trajectories_list) if sum(num_trajectories_list) > 0 else 0.0

            # Tool call metrics - turn level (weighted average by valid turns)
            if num_valid_planner_turns > 0:
                additional_metrics["turn_avg_subtask"] = turn_sum_subtask / num_valid_planner_turns
            else:
                additional_metrics["turn_avg_subtask"] = 0.0
            additional_metrics["turn_max_subtask"] = max(turn_max_subtasks) if turn_max_subtasks else 0

            if num_valid_worker_turns > 0:
                additional_metrics["turn_avg_search"] = turn_sum_search / num_valid_worker_turns
                additional_metrics["turn_avg_access"] = turn_sum_access / num_valid_worker_turns
                additional_metrics["turn_avg_search+access"] = turn_sum_search_plus_access / num_valid_worker_turns
            else:
                additional_metrics["turn_avg_search"] = 0.0
                additional_metrics["turn_avg_access"] = 0.0
                additional_metrics["turn_avg_search+access"] = 0.0

            additional_metrics["turn_max_search"] = max(turn_max_searches) if turn_max_searches else 0
            additional_metrics["turn_max_access"] = max(turn_max_accesses) if turn_max_accesses else 0
            additional_metrics["turn_max_search+access"] = max(turn_max_search_plus_access_list) if turn_max_search_plus_access_list else 0

            # Tool call metrics - trajectory level
            additional_metrics["traj_avg_subtask"] = sum(traj_avg_subtasks) / total_queries
            additional_metrics["traj_avg_search"] = sum(traj_avg_searches) / total_queries
            additional_metrics["traj_avg_access"] = sum(traj_avg_accesses) / total_queries
            additional_metrics["traj_avg_search+access"] = sum(traj_avg_search_plus_access_list) / total_queries
            additional_metrics["traj_max_subtask"] = max(traj_max_subtasks) if traj_max_subtasks else 0
            additional_metrics["traj_max_search"] = max(traj_max_searches) if traj_max_searches else 0
            additional_metrics["traj_max_access"] = max(traj_max_accesses) if traj_max_accesses else 0
            additional_metrics["traj_max_search+access"] = max(traj_max_search_plus_access_list) if traj_max_search_plus_access_list else 0

        # Merge additional metrics into final_metrics
        final_metrics.update(additional_metrics)

        # Save all accumulated results to JSON file
        logging.info(f"Saving {len(self.accumulated_results)} results to JSON file...")
        self._save_eval_results(self.accumulated_results, final_metrics, total_queries)

        self.metric_logger.finish()
