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

import pandas as pd
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer
from rlinf.workers.agent.agent_loop import MultiTurnAgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolChannelInfo, ToolWorker, ToolWorkerInfo

if typing.TYPE_CHECKING:
    from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

logging.getLogger().setLevel(logging.INFO)


class AgentEvalRunner:
    """Runner for agent task evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        val_dataset: Dataset,
        rollout: "SGLangWorker",
        agent_loop: MultiTurnAgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, "SGLangWorker"] = {},
    ):
        # Initialize base attributes
        self.cfg = cfg
        self.component_placement = placement
        self.is_pipeline = self.component_placement.is_pipeline

        # Workers
        self.rollout = rollout

        # Scheduler task
        self.scheduler = None
        self.use_pre_process_policy = False

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")

        # Configurations
        self.compute_ref_logprobs = (
            self.cfg.algorithm.kl_beta > 0
            or self.cfg.algorithm.get("reinpp_kl_beta", 0) > 0
        )
        self.recompute_logprobs = self.cfg.algorithm.recompute_logprobs
        self.consumed_samples = 0
        self.global_steps = 0

        # Build dataloader and compute `max_steps`
        self._build_dataloader(val_dataset)
        self._set_max_steps()

        # Wandb table
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        # Timers
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.metric_logger = MetricLogger(cfg)

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

        # Initialize storage for accumulating raw evaluation results across all batches
        # Each item is the raw eval_result dict from agent_loop
        self.accumulated_raw_results = []

    def _build_dataloader(self, val_dataset, collate_fn=None):
        """
        Creates the train and validation dataloaders.
        """
        self.val_dataset = val_dataset
        if collate_fn is None:
            from rlinf.data.datasets import collate_fn

        num_workers = self.cfg.data.num_workers

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

    def _compute_flops_metrics(self, time_metrics, act_rollout_metrics) -> dict:
        rollout_time = time_metrics.get("rollout")
        training_time = time_metrics.get("training")

        num_gpus_actor = self.component_placement.actor_world_size
        num_gpus_rollout = self.component_placement.rollout_world_size

        rollout_tflops = act_rollout_metrics["rollout_tflops"]
        training_tflops = act_rollout_metrics["training_tflops"]

        flops_metrics = {
            "rollout_tflops_per_gpu": 0.0,
            "training_tflops_per_gpu": 0.0,
        }
        if rollout_time > 0 and rollout_tflops > 0:
            flops_metrics["rollout_tflops_per_gpu"] = (
                rollout_tflops / rollout_time / num_gpus_rollout
            )

        if training_time > 0 and training_tflops > 0:
            flops_metrics["training_tflops_per_gpu"] = (
                training_tflops / training_time / num_gpus_actor
            )

        return flops_metrics

    def _set_max_steps(self):
        self.num_steps_per_epoch = len(self.val_dataloader)
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_steps // self.num_steps_per_epoch

    def _put_batch(self, batch: dict[str, torch.Tensor]):
        prompt_ids = batch["prompt"].tolist()
        lengths = batch["length"].tolist()
        answers = batch["answer"]
        image_data = batch["image_data"]
        multi_modal_inputs = batch["multi_modal_inputs"]
        prompt_ids = [ids[-pmp_len:] for ids, pmp_len in zip(prompt_ids, lengths)]
        rollout_dp_size = self.component_placement.rollout_dp_size

        for input_ids, answers, image_data, multi_modal_inputs in zip(
            split_list(prompt_ids, rollout_dp_size, enforce_divisible_batch=False),
            split_list(answers, rollout_dp_size, enforce_divisible_batch=False),
            split_list(image_data, rollout_dp_size, enforce_divisible_batch=False),
            split_list(
                multi_modal_inputs, rollout_dp_size, enforce_divisible_batch=False
            ),
        ):
            request = RolloutRequest(
                n=self.cfg.algorithm.group_size,
                input_ids=input_ids,
                answers=answers,
                image_data=image_data,
                multi_modal_inputs=multi_modal_inputs,
            )
            self.dataloader_channel.put(request, async_op=True)

    def _save_eval_results(self, all_results, aggregated_metrics, total_count):
        """Save evaluation results to JSON files and per-response directory.

        Saves three types of outputs:
        1. metrics.json - Key aggregated metrics
        2. allresults.json - All detailed results
        3. responses/ directory - Per-instance response files

        Args:
            all_results: List of result dictionaries for each query
            aggregated_metrics: Dictionary with aggregated metrics
            total_count: Total number of queries evaluated
        """
        import datetime

        # Create output directory in the experiment folder
        output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)

        # Create responses subdirectory
        response_dir = os.path.join(output_dir, "responses")
        local_mkdir_safe(response_dir)

        # Fixed filenames
        output_file_key = os.path.join(output_dir, "metrics.json")
        output_file_all = os.path.join(output_dir, "allresults.json")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert OmegaConf objects to plain Python objects
        eval_metrics = self.cfg.reward.get("eval_metric", [])
        if eval_metrics:
            eval_metrics = OmegaConf.to_container(eval_metrics, resolve=True)

        data_paths = self.cfg.data.val_data_paths
        if data_paths:
            data_paths = OmegaConf.to_container(data_paths, resolve=True)

        model_config_name = self.cfg.runner.experiment_name

        # Prepare complete results structure
        results_data_key = {
            "dataset_size": total_count,
            "experiment_name": self.cfg.runner.experiment_name,
            "timestamp": timestamp,
            "config": {
                "group_size": self.cfg.algorithm.get("group_size", 1),
                "eval_metrics": eval_metrics,
                "reward_type": self.cfg.reward.get("reward_type", "EM"),
                "data_paths": data_paths,
            },
            "metrics": aggregated_metrics
        }

        # Write metrics.json
        with open(output_file_key, 'w', encoding='utf-8') as f:
            json.dump(results_data_key, f, ensure_ascii=False, indent=2)

        # Write allresults.json
        with open(output_file_all, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # Save per-response files
        for result in all_results:
            samples = result.get("samples", [])
            answer = result.get("answer", {})
            instance_id = answer.get("instance_id", "unknown") if isinstance(answer, dict) else "unknown"

            for trial_idx, sample in enumerate(samples):
                file_trial_idx = trial_idx
                while True:
                    response_file = os.path.join(
                        response_dir,
                        f"{model_config_name}_{instance_id}_{file_trial_idx}_response.jsonl"
                    )
                    if not os.path.exists(response_file):
                        break
                    file_trial_idx += 1

                # Extract final_answer - handle DataFrame conversion
                final_answer = sample.get("final_answer", None)
                if isinstance(final_answer, pd.DataFrame):
                    final_answer = final_answer.to_dict(orient='records')

                response_data = {
                    "instance_id": instance_id,
                    "trial_idx": file_trial_idx,
                    "final_answer": final_answer,
                    "final_answer_text": sample.get("final_answer_text", None),
                    "eval_metric": sample.get("eval_metric", {}),
                    "num_turns": sample.get("num_turns", 0),
                    "origin_question": sample.get("origin_question", None),
                }

                with open(response_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Evaluation results saved to: {output_file_key}")
        logging.info(f"Per-response files saved to: {response_dir}")
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
                solid_rollout.init_worker().wait()
                solid_rollout.onload_engine().wait()

        self.agent_loop.init_worker(
            self.generate_input_channel,
            self.generate_output_channel,
            self.tool_channel_info_map,
            self.tool_name_map,
            self.tool_output_channel,
            self.solid_generate_input_channels,
        ).wait()

        # Init workers
        self.rollout.init_worker().wait()
        if self.use_pre_process_policy:
            self.rollout.offload_engine().wait()

        if self.cfg.runner.resume_dir is None:
            return

        # Resume from checkpoint
        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"Setting global step to {self.global_steps}")

    def log_eval(self, input_channel, expected_batch_size=None):
        """Collect raw evaluation results from channel for a single batch.

        Simply accumulates raw eval_result dicts from agent_loop without processing.
        All metric computation is deferred to _aggregate_all_results().

        Args:
            input_channel: The channel to receive rollout results from
            expected_batch_size: Expected number of queries in this batch.

        Returns:
            Number of queries received in this batch
        """
        group_size = self.cfg.algorithm.get("group_size", 1)

        if expected_batch_size is not None:
            total_batch_size_per_dp = expected_batch_size
        else:
            total_batch_size_per_dp = self.cfg.data.rollout_batch_size

        recv_batch_size = 0
        while recv_batch_size < total_batch_size_per_dp:
            # Receive raw evaluation dictionary from agent_loop
            eval_result: dict = input_channel.get()
            self.accumulated_raw_results.append(eval_result)
            recv_batch_size += 1

        assert recv_batch_size == total_batch_size_per_dp, (
            f"Expected {total_batch_size_per_dp} queries from channel, but got {recv_batch_size}"
        )

        return recv_batch_size

    def _aggregate_all_results(self):
        """Aggregate all accumulated raw results into final metrics.

        This function processes all raw eval_results and computes:
        1. Eval metrics (pass@1, pass@k, avg@k, max@k for EM, F1, LLM)
        2. Length metrics (turn-level weighted averages)
        3. Tool call metrics (trajectory and turn level)
        4. Format score metrics
        5. MAS turn metrics

        Returns:
            Tuple of (processed_results list, aggregated_metrics dict)
        """
        is_markdown = self.cfg.data.get("is_markdown", False)
        metric_types = ["EM", "F1", "LLM"]
        markdown_metrics = [
            "score", "precision_by_row", "recall_by_row", "f1_by_row",
            "precision_by_item", "recall_by_item", "f1_by_item",
            "search_precision_by_item", "search_recall_by_item", "search_f1_by_item"
        ]

        processed_results = []
        total_queries = len(self.accumulated_raw_results)

        # Accumulators for global aggregation
        # Length metrics (weighted by num_turns)
        total_num_turns = 0
        sum_prompt_length = 0
        sum_response_length = 0
        sum_total_length = 0
        max_prompt_lengths = []
        max_response_lengths = []
        max_total_lengths = []

        # Trajectory metrics
        total_num_trajectories = 0
        total_turns_all = 0

        # Tool call metrics (weighted by valid turns)
        sum_turn_subtask = 0
        sum_turn_search = 0
        sum_turn_access = 0
        sum_turn_search_plus_access = 0
        total_valid_planner_turns = 0
        total_valid_worker_turns = 0
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

        # Format score metrics
        format_score_sums = {}

        # MAS turn metrics
        mas_sum_main_agent_turns = 0
        mas_sum_subagent_turns = 0
        mas_sum_num_subagents = 0
        mas_num_valid_trajs = 0

        # Eval metrics accumulators (for final computation)
        if is_markdown:
            # {metric_type: {md_metric: {"pass1": [], "passk": [], "avgk": [], "maxk": []}}}
            eval_accumulators = {
                mt: {mm: {"pass1": [], "passk": [], "avgk": [], "maxk": []} for mm in markdown_metrics}
                for mt in metric_types
            }
        else:
            # {metric_type: {"pass1": [], "passk": [], "avgk": [], "maxk": []}}
            eval_accumulators = {
                mt: {"pass1": [], "passk": [], "avgk": [], "maxk": []} for mt in metric_types
            }

        # Process each raw result
        for idx, raw_result in enumerate(self.accumulated_raw_results):
            group_size = raw_result.get("group_size", 1)
            answer = raw_result.get("answer", None)
            samples = raw_result.get("samples", [])

            # --- Extract length metrics from samples ---
            prompt_lengths = []
            response_lengths = []
            total_lengths = []
            num_turns_list = []
            subtask_counts = []
            search_counts = []
            access_counts = []
            num_valid_planner_turns = 0
            num_valid_worker_turns = 0
            mas_main_agent_turns_list = []
            mas_subagent_turns_list = []
            mas_num_subagents_list = []

            for sample in samples:
                turns = sample.get("turns", [])
                num_turns_list.append(len(turns))

                # MAS turn list
                total_turn_list = sample.get("total_turn_list", None)
                if total_turn_list is not None and len(total_turn_list) > 0:
                    mas_main_agent_turns_list.append(total_turn_list[-1])
                    subagent_turns_list = total_turn_list[:-1]
                    mas_subagent_turns_list.append(sum(subagent_turns_list))
                    mas_num_subagents_list.append(len(subagent_turns_list))

                for turn in turns:
                    prompt_lengths.append(turn.get("prompt_ids_length", 0))
                    response_lengths.append(turn.get("response_ids_length", 0))
                    total_lengths.append(prompt_lengths[-1] + response_lengths[-1])

                    tool_call_info = turn.get("tool_call_info", None)
                    if tool_call_info is not None:
                        role = tool_call_info.get("role", "")
                        subtask_counts.append(tool_call_info.get("subtask", 0))
                        search_counts.append(tool_call_info.get("search", 0))
                        access_counts.append(tool_call_info.get("access", 0))
                        if role == "planner":
                            num_valid_planner_turns += 1
                        elif role in ("worker", "single"):
                            num_valid_worker_turns += 1

            # Accumulate length metrics
            num_turns = len(prompt_lengths)
            total_num_turns += num_turns
            sum_prompt_length += sum(prompt_lengths) if prompt_lengths else 0
            sum_response_length += sum(response_lengths) if response_lengths else 0
            sum_total_length += sum(total_lengths) if total_lengths else 0
            if prompt_lengths:
                max_prompt_lengths.append(max(prompt_lengths))
                max_response_lengths.append(max(response_lengths))
                max_total_lengths.append(max(total_lengths))

            # Accumulate trajectory metrics
            total_num_trajectories += group_size
            total_turns_all += sum(num_turns_list)

            # Accumulate tool call metrics
            total_valid_planner_turns += num_valid_planner_turns
            total_valid_worker_turns += num_valid_worker_turns
            sum_turn_subtask += sum(subtask_counts)
            sum_turn_search += sum(search_counts)
            sum_turn_access += sum(access_counts)
            search_plus_access = [s + a for s, a in zip(search_counts, access_counts)] if search_counts else []
            sum_turn_search_plus_access += sum(search_plus_access)

            if subtask_counts:
                turn_max_subtasks.append(max(subtask_counts))
            if search_counts:
                turn_max_searches.append(max(search_counts))
            if access_counts:
                turn_max_accesses.append(max(access_counts))
            if search_plus_access:
                turn_max_search_plus_access_list.append(max(search_plus_access))

            # Trajectory-level tool call averages
            if group_size > 0:
                traj_avg_subtasks.append(sum(subtask_counts) / group_size if subtask_counts else 0.0)
                traj_avg_searches.append(sum(search_counts) / group_size if search_counts else 0.0)
                traj_avg_accesses.append(sum(access_counts) / group_size if access_counts else 0.0)
                traj_avg_search_plus_access_list.append(sum(search_plus_access) / group_size if search_plus_access else 0.0)
            if subtask_counts:
                traj_max_subtasks.append(max(subtask_counts))
            if search_counts:
                traj_max_searches.append(max(search_counts))
            if access_counts:
                traj_max_accesses.append(max(access_counts))
            if search_plus_access:
                traj_max_search_plus_access_list.append(max(search_plus_access))

            # Accumulate format_score metrics
            if samples:
                sample_format_score = samples[0].get("eval_metric", {}).get("format_score", {})
                for sub_metric in sample_format_score.keys():
                    values = [s.get("eval_metric", {}).get("format_score", {}).get(sub_metric, 0) for s in samples]
                    avg_val = sum(values) / len(values) if values else 0.0
                    format_score_sums[sub_metric] = format_score_sums.get(sub_metric, 0.0) + avg_val

            # Accumulate MAS turn metrics
            if mas_main_agent_turns_list:
                mas_sum_main_agent_turns += sum(mas_main_agent_turns_list)
                mas_sum_subagent_turns += sum(mas_subagent_turns_list)
                mas_sum_num_subagents += sum(mas_num_subagents_list)
                mas_num_valid_trajs += len(mas_main_agent_turns_list)

            # --- Compute eval metrics for this query ---
            if is_markdown:
                for metric_type in metric_types:
                    for md_metric in markdown_metrics:
                        values = []
                        for sample in samples:
                            val = sample.get("eval_metric", {}).get(metric_type, {}).get(md_metric, 0.0)
                            values.append(val)

                        if values:
                            if md_metric == "score":
                                eval_accumulators[metric_type][md_metric]["pass1"].append(1.0 if values[0] > 0 else 0.0)
                                eval_accumulators[metric_type][md_metric]["passk"].append(1.0 if any(v > 0 for v in values) else 0.0)
                                eval_accumulators[metric_type][md_metric]["avgk"].append(sum(values) / len(values))
                            else:
                                eval_accumulators[metric_type][md_metric]["pass1"].append(values[0])
                                eval_accumulators[metric_type][md_metric]["avgk"].append(sum(values) / len(values))
                                eval_accumulators[metric_type][md_metric]["maxk"].append(max(values))
            else:
                for metric_type in metric_types:
                    values = []
                    for sample in samples:
                        val = sample.get("eval_metric", {}).get(metric_type, None)
                        if val is not None:
                            values.append(val)

                    if values:
                        eval_accumulators[metric_type]["pass1"].append(1.0 if values[0] > 0 else 0.0)
                        eval_accumulators[metric_type]["avgk"].append(sum(values) / len(values))
                        if metric_type == "F1":
                            eval_accumulators[metric_type]["maxk"].append(max(values))
                        else:
                            eval_accumulators[metric_type]["passk"].append(1.0 if any(v > 0 for v in values) else 0.0)

            # Build processed result entry
            processed_results.append({
                "index": idx,
                "group_size": group_size,
                "answer": answer,
                "samples": samples,
            })

        # --- Compute final aggregated metrics ---
        aggregated_metrics = {}

        # Eval metrics
        if is_markdown:
            for metric_type in metric_types:
                aggregated_metrics[metric_type] = {}
                for md_metric in markdown_metrics:
                    acc = eval_accumulators[metric_type][md_metric]
                    if md_metric == "score":
                        aggregated_metrics[metric_type][md_metric] = {
                            "pass@1": sum(acc["pass1"]) / len(acc["pass1"]) if acc["pass1"] else 0.0,
                            "pass@k": sum(acc["passk"]) / len(acc["passk"]) if acc["passk"] else 0.0,
                            "avg@k": sum(acc["avgk"]) / len(acc["avgk"]) if acc["avgk"] else 0.0,
                        }
                    else:
                        aggregated_metrics[metric_type][md_metric] = {
                            "pass@1": sum(acc["pass1"]) / len(acc["pass1"]) if acc["pass1"] else 0.0,
                            "avg@k": sum(acc["avgk"]) / len(acc["avgk"]) if acc["avgk"] else 0.0,
                            "max@k": sum(acc["maxk"]) / len(acc["maxk"]) if acc["maxk"] else 0.0,
                        }
        else:
            for metric_type in metric_types:
                acc = eval_accumulators[metric_type]
                aggregated_metrics[f"pass@1_{metric_type}"] = sum(acc["pass1"]) / len(acc["pass1"]) if acc["pass1"] else 0.0
                aggregated_metrics[f"avg@k_{metric_type}"] = sum(acc["avgk"]) / len(acc["avgk"]) if acc["avgk"] else 0.0
                if metric_type == "F1":
                    aggregated_metrics[f"max@k_{metric_type}"] = sum(acc["maxk"]) / len(acc["maxk"]) if acc["maxk"] else 0.0
                else:
                    aggregated_metrics[f"pass@k_{metric_type}"] = sum(acc["passk"]) / len(acc["passk"]) if acc["passk"] else 0.0

        # Length metrics
        if total_num_turns > 0:
            aggregated_metrics["avg_prompt_length"] = sum_prompt_length / total_num_turns
            aggregated_metrics["avg_response_length"] = sum_response_length / total_num_turns
            aggregated_metrics["avg_total_length"] = sum_total_length / total_num_turns
        else:
            aggregated_metrics["avg_prompt_length"] = 0.0
            aggregated_metrics["avg_response_length"] = 0.0
            aggregated_metrics["avg_total_length"] = 0.0

        aggregated_metrics["max_prompt_length"] = max(max_prompt_lengths) if max_prompt_lengths else 0
        aggregated_metrics["max_response_length"] = max(max_response_lengths) if max_response_lengths else 0
        aggregated_metrics["max_total_length"] = max(max_total_lengths) if max_total_lengths else 0

        # Trajectory metrics
        aggregated_metrics["total_num_trajectories"] = total_num_trajectories
        aggregated_metrics["avg_turns_per_traj"] = total_turns_all / total_num_trajectories if total_num_trajectories > 0 else 0.0

        # Tool call metrics - turn level
        if total_valid_planner_turns > 0:
            aggregated_metrics["turn_avg_subtask"] = sum_turn_subtask / total_valid_planner_turns
        else:
            aggregated_metrics["turn_avg_subtask"] = 0.0
        aggregated_metrics["turn_max_subtask"] = max(turn_max_subtasks) if turn_max_subtasks else 0

        if total_valid_worker_turns > 0:
            aggregated_metrics["turn_avg_search"] = sum_turn_search / total_valid_worker_turns
            aggregated_metrics["turn_avg_access"] = sum_turn_access / total_valid_worker_turns
            aggregated_metrics["turn_avg_search+access"] = sum_turn_search_plus_access / total_valid_worker_turns
        else:
            aggregated_metrics["turn_avg_search"] = 0.0
            aggregated_metrics["turn_avg_access"] = 0.0
            aggregated_metrics["turn_avg_search+access"] = 0.0

        aggregated_metrics["turn_max_search"] = max(turn_max_searches) if turn_max_searches else 0
        aggregated_metrics["turn_max_access"] = max(turn_max_accesses) if turn_max_accesses else 0
        aggregated_metrics["turn_max_search+access"] = max(turn_max_search_plus_access_list) if turn_max_search_plus_access_list else 0

        # Tool call metrics - trajectory level
        if total_queries > 0:
            aggregated_metrics["traj_avg_subtask"] = sum(traj_avg_subtasks) / total_queries
            aggregated_metrics["traj_avg_search"] = sum(traj_avg_searches) / total_queries
            aggregated_metrics["traj_avg_access"] = sum(traj_avg_accesses) / total_queries
            aggregated_metrics["traj_avg_search+access"] = sum(traj_avg_search_plus_access_list) / total_queries
        else:
            aggregated_metrics["traj_avg_subtask"] = 0.0
            aggregated_metrics["traj_avg_search"] = 0.0
            aggregated_metrics["traj_avg_access"] = 0.0
            aggregated_metrics["traj_avg_search+access"] = 0.0

        aggregated_metrics["traj_max_subtask"] = max(traj_max_subtasks) if traj_max_subtasks else 0
        aggregated_metrics["traj_max_search"] = max(traj_max_searches) if traj_max_searches else 0
        aggregated_metrics["traj_max_access"] = max(traj_max_accesses) if traj_max_accesses else 0
        aggregated_metrics["traj_max_search+access"] = max(traj_max_search_plus_access_list) if traj_max_search_plus_access_list else 0

        # Format score metrics
        for sub_metric, total_sum in format_score_sums.items():
            aggregated_metrics[f"format_score/{sub_metric}"] = total_sum / total_queries if total_queries > 0 else 0.0

        # MAS turn metrics
        if mas_num_valid_trajs > 0:
            aggregated_metrics["mas/avg_main_agent_turns_per_traj"] = mas_sum_main_agent_turns / mas_num_valid_trajs
            aggregated_metrics["mas/avg_subagent_turns_per_traj"] = mas_sum_subagent_turns / mas_num_valid_trajs
            aggregated_metrics["mas/avg_num_subagents_per_traj"] = mas_sum_num_subagents / mas_num_valid_trajs

        return processed_results, aggregated_metrics

    def run(self):
        """Run evaluation on validation dataset.

        This function:
        1. Runs rollout on validation data (accumulates raw results)
        2. Aggregates all results after all batches complete
        3. Saves results to files
        """
        logging.info("=" * 80)
        logging.info("Starting Multi-Agent System Evaluation")
        logging.info("=" * 80)
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Batch size: {self.cfg.data.val_rollout_batch_size}")
        logging.info(f"Group size: {self.cfg.algorithm.get('group_size', 1)}")
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

        try:
            # Process validation batches - just accumulate raw results
            for batch_idx, batch in enumerate(self.val_dataloader):
                logging.info(f"\nProcessing batch {batch_idx + 1}/{len(self.val_dataloader)}")

                # Calculate actual batch size from the batch data
                if 'prompt' in batch:
                    actual_batch_size = batch['prompt'].shape[0]
                elif 'answer' in batch:
                    actual_batch_size = len(batch['answer'])
                else:
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

                    # Simply accumulate raw results
                    batch_count = self.log_eval(
                        input_channel=self.rollout_channel,
                        expected_batch_size=actual_batch_size
                    )

                    rollout_handle.wait()

                time_metrics = self.timer.consume_durations()
                time_metrics["rollout"] = rollout_handle.consume_duration()

                eval_pbar.set_postfix({
                    "queries": len(self.accumulated_raw_results),
                    "rollout_time": f"{time_metrics.get('rollout', 0):.2f}s",
                })
                eval_pbar.update(1)

                self.global_steps += 1

        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()

        eval_pbar.close()

        # Aggregate all results after all batches complete
        logging.info(f"Aggregating {len(self.accumulated_raw_results)} results...")
        processed_results, final_metrics = self._aggregate_all_results()

        total_queries = len(self.accumulated_raw_results)

        # Save all results to files
        logging.info(f"Saving {total_queries} results to JSON files...")
        self._save_eval_results(processed_results, final_metrics, total_queries)

        self.metric_logger.finish()
