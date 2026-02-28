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

import json
import logging
import os
import typing
from typing import Optional, Union

import pandas as pd
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import local_mkdir_safe
from rlinf.workers.agent.agent_loop import MultiTurnAgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolWorker, ToolWorkerInfo
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class WideSeekR1AgentEvalRunner(AgentEvalRunner):
    """Runner for wideseek r1 task RL evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        reward: Optional[RewardWorker],
        agent_loop: MultiTurnAgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, Union["SGLangWorker", "VLLMWorker"]] = {},
    ):
        """Initialize the evaluation runner and in-memory result accumulator.

        Args:
            cfg: Global runtime/config object.
            placement: Placement strategy for distributed workers.
            val_dataset: Validation dataset used for rollout evaluation.
            rollout: Main rollout worker.
            reward: Optional reward worker.
            agent_loop: Agent-loop worker used for multi-turn generation.
            tool_workers: Tool workers attached to this runner.
            solid_rollouts: Optional fixed rollout workers.
        """
        super().__init__(
            cfg,
            placement,
            val_dataset,
            rollout,
            reward,
            agent_loop,
            tool_workers,
            solid_rollouts,
        )
        # Initialize storage for accumulating raw evaluation results across all batches
        # Each item is the raw eval_result dict from agent_loop
        self.accumulated_raw_results = []

        # Specific Configurations
        self.compute_ref_logprobs = (
            self.cfg.algorithm.kl_beta > 0
            or self.cfg.algorithm.get("reinpp_kl_beta", 0) > 0
        )
        self.recompute_logprobs = self.cfg.algorithm.recompute_logprobs

    def _save_eval_results(self, all_results, aggregated_metrics, total_count):
        """Persist aggregated metrics and per-sample responses to disk.

        Args:
            all_results: Per-question processed evaluation payloads.
            aggregated_metrics: Dataset-level aggregated metric dictionary.
            total_count: Number of evaluated questions.

        Returns:
            Path to the saved `metrics.json` file.
        """
        import datetime

        output_dir = os.path.join(
            self.cfg.runner.output_dir, self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)

        response_dir = os.path.join(output_dir, "responses")
        local_mkdir_safe(response_dir)

        output_file_key = os.path.join(output_dir, "metrics.json")
        output_file_all = os.path.join(output_dir, "allresults.json")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
                "data_paths": data_paths,
            },
            "metrics": aggregated_metrics,
        }

        with open(output_file_key, "w", encoding="utf-8") as f:
            json.dump(results_data_key, f, ensure_ascii=False, indent=2)

        with open(output_file_all, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        for result in all_results:
            samples = result.get("samples", [])
            answer = result.get("answer", {})
            instance_id = (
                answer.get("instance_id", "unknown")
                if isinstance(answer, dict)
                else "unknown"
            )

            for trial_idx, sample in enumerate(samples):
                file_trial_idx = trial_idx
                while True:
                    response_file = os.path.join(
                        response_dir,
                        f"{model_config_name}_{instance_id}_{file_trial_idx}_response.jsonl",
                    )
                    if not os.path.exists(response_file):
                        break
                    file_trial_idx += 1

                final_answer = sample.get("final_answer", None)
                if isinstance(final_answer, pd.DataFrame):
                    final_answer = final_answer.to_dict(orient="records")

                response_data = {
                    "instance_id": instance_id,
                    "trial_idx": file_trial_idx,
                    "final_answer": final_answer,
                    "final_answer_text": sample.get("final_answer_text", None),
                    "llm_reward": sample.get("llm_reward", 0.0),
                    "final_answer_format": sample.get("final_answer_format", 0),
                    "num_turns": sample.get("num_turns", 0),
                    "origin_question": sample.get("origin_question", None),
                }

                with open(response_file, "w", encoding="utf-8") as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Evaluation results saved to: {output_file_key}")
        logging.info(f"Per-response files saved to: {response_dir}")
        return output_file_key

    def _aggregate_all_results(self):
        """Aggregate cached raw rollout results into final evaluation metrics.

        Returns:
            Tuple of `(processed_results, aggregated_metrics)`.
        """
        is_markdown = self.cfg.data.get("is_markdown", False)

        processed_results = []
        total_queries = len(self.accumulated_raw_results)

        total_num_turns = 0
        sum_prompt_length = 0
        sum_response_length = 0
        sum_total_length = 0

        total_num_trajectories = 0
        total_turns_all = 0

        sum_turn_subtask = 0
        sum_turn_search = 0
        sum_turn_access = 0
        sum_turn_search_plus_access = 0
        total_valid_planner_turns = 0
        total_valid_worker_turns = 0
        traj_avg_subtasks = []
        traj_avg_searches = []
        traj_avg_accesses = []
        traj_avg_search_plus_access_list = []

        final_answer_format_sum = 0.0

        mas_sum_main_agent_turns = 0
        mas_sum_subagent_turns = 0
        mas_sum_num_subagents = 0
        mas_num_valid_trajs = 0

        if is_markdown:
            acc = {}
        else:
            acc = {"pass1": [], "passk": [], "avgk": [], "maxk": []}

        for idx, raw_result in enumerate(self.accumulated_raw_results):
            group_size = raw_result.get("group_size", 1)
            answer = raw_result.get("answer", None)
            samples = raw_result.get("samples", [])

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

            num_turns = len(prompt_lengths)
            total_num_turns += num_turns
            sum_prompt_length += sum(prompt_lengths) if prompt_lengths else 0
            sum_response_length += sum(response_lengths) if response_lengths else 0
            sum_total_length += sum(total_lengths) if total_lengths else 0

            total_num_trajectories += group_size
            total_turns_all += sum(num_turns_list)

            total_valid_planner_turns += num_valid_planner_turns
            total_valid_worker_turns += num_valid_worker_turns
            sum_turn_subtask += sum(subtask_counts)
            sum_turn_search += sum(search_counts)
            sum_turn_access += sum(access_counts)
            search_plus_access = (
                [s + a for s, a in zip(search_counts, access_counts)]
                if search_counts
                else []
            )
            sum_turn_search_plus_access += sum(search_plus_access)

            if group_size > 0:
                traj_avg_subtasks.append(
                    sum(subtask_counts) / group_size if subtask_counts else 0.0
                )
                traj_avg_searches.append(
                    sum(search_counts) / group_size if search_counts else 0.0
                )
                traj_avg_accesses.append(
                    sum(access_counts) / group_size if access_counts else 0.0
                )
                traj_avg_search_plus_access_list.append(
                    sum(search_plus_access) / group_size if search_plus_access else 0.0
                )

            if samples:
                final_answer_format_values = [
                    float(sample.get("final_answer_format", 0) or 0)
                    for sample in samples
                ]
                if final_answer_format_values:
                    final_answer_format_sum += sum(final_answer_format_values) / len(
                        final_answer_format_values
                    )

            if mas_main_agent_turns_list:
                mas_sum_main_agent_turns += sum(mas_main_agent_turns_list)
                mas_sum_subagent_turns += sum(mas_subagent_turns_list)
                mas_sum_num_subagents += sum(mas_num_subagents_list)
                mas_num_valid_trajs += len(mas_main_agent_turns_list)

            if is_markdown:
                pass
            else:
                values = [float(sample.get("llm_reward", 0) or 0) for sample in samples]
                if values:
                    acc["pass1"].append(1.0 if values[0] > 0 else 0.0)
                    acc["avgk"].append(sum(values) / len(values))
                    acc["passk"].append(1.0 if any(v > 0 for v in values) else 0.0)

            processed_results.append(
                {
                    "index": idx,
                    "group_size": group_size,
                    "answer": answer,
                    "samples": samples,
                }
            )

        aggregated_metrics = {}
        if is_markdown:
            pass
        else:
            aggregated_metrics["pass@1"] = (
                sum(acc["pass1"]) / len(acc["pass1"]) if acc["pass1"] else 0.0
            )
            aggregated_metrics["avg@k"] = (
                sum(acc["avgk"]) / len(acc["avgk"]) if acc["avgk"] else 0.0
            )
            aggregated_metrics["pass@k"] = (
                sum(acc["passk"]) / len(acc["passk"]) if acc["passk"] else 0.0
            )

        if total_num_turns > 0:
            aggregated_metrics["avg_prompt_length"] = (
                sum_prompt_length / total_num_turns
            )
            aggregated_metrics["avg_response_length"] = (
                sum_response_length / total_num_turns
            )
            aggregated_metrics["avg_total_length"] = sum_total_length / total_num_turns
        else:
            aggregated_metrics["avg_prompt_length"] = 0.0
            aggregated_metrics["avg_response_length"] = 0.0
            aggregated_metrics["avg_total_length"] = 0.0

        aggregated_metrics["total_num_trajectories"] = total_num_trajectories
        aggregated_metrics["avg_turns_per_traj"] = (
            total_turns_all / total_num_trajectories
            if total_num_trajectories > 0
            else 0.0
        )

        if total_valid_planner_turns > 0:
            aggregated_metrics["turn_avg_subtask"] = (
                sum_turn_subtask / total_valid_planner_turns
            )
        else:
            aggregated_metrics["turn_avg_subtask"] = 0.0

        if total_valid_worker_turns > 0:
            aggregated_metrics["turn_avg_search"] = (
                sum_turn_search / total_valid_worker_turns
            )
            aggregated_metrics["turn_avg_access"] = (
                sum_turn_access / total_valid_worker_turns
            )
        else:
            aggregated_metrics["turn_avg_search"] = 0.0
            aggregated_metrics["turn_avg_access"] = 0.0

        if total_queries > 0:
            aggregated_metrics["traj_avg_subtask"] = (
                sum(traj_avg_subtasks) / total_queries
            )
            aggregated_metrics["traj_avg_search"] = (
                sum(traj_avg_searches) / total_queries
            )
            aggregated_metrics["traj_avg_access"] = (
                sum(traj_avg_accesses) / total_queries
            )
        else:
            aggregated_metrics["traj_avg_subtask"] = 0.0
            aggregated_metrics["traj_avg_search"] = 0.0
            aggregated_metrics["traj_avg_access"] = 0.0

        aggregated_metrics["final_answer_format"] = (
            final_answer_format_sum / total_queries if total_queries > 0 else 0.0
        )

        if mas_num_valid_trajs > 0:
            aggregated_metrics["avg_main_agent_turns_per_traj"] = (
                mas_sum_main_agent_turns / mas_num_valid_trajs
            )
            aggregated_metrics["avg_subagent_turns_per_traj"] = (
                mas_sum_subagent_turns / mas_num_valid_trajs
            )
            aggregated_metrics["avg_num_subagents_per_traj"] = (
                mas_sum_num_subagents / mas_num_valid_trajs
            )

        return processed_results, aggregated_metrics

    def log_eval(
        self,
        context: dict,
        batch_idx,
        batch,
        input_channel,
    ):
        """Collect raw evaluation results from channel for a single batch.

        Simply accumulates raw eval_result dicts from agent_loop without processing.
        All metric computation is deferred to _aggregate_all_results().

        Args:
            input_channel: The channel to receive rollout results from

        Returns:
            Number of queries received in this batch
        """
        # Calculate actual batch size from the batch data
        if "prompt" in batch:
            expected_batch_size = batch["prompt"].shape[0]
        elif "answer" in batch:
            expected_batch_size = len(batch["answer"])
        else:
            expected_batch_size = self.cfg.data.val_rollout_batch_size

        logging.info(f"Actual batch size for this batch: {expected_batch_size}")

        if expected_batch_size is not None:
            total_batch_size_per_dp = expected_batch_size
        else:
            total_batch_size_per_dp = self.cfg.data.rollout_batch_size

        recv_batch_size = 0
        while recv_batch_size < total_batch_size_per_dp:
            # Receive raw evaluation dictionary from agent_loop
            rollout_result = input_channel.get()
            eval_result: dict = self.extract_eval_result(rollout_result)
            self.accumulated_raw_results.append(eval_result)
            recv_batch_size += 1

        assert recv_batch_size == total_batch_size_per_dp, (
            f"Expected {total_batch_size_per_dp} queries from channel, but got {recv_batch_size}"
        )

        return recv_batch_size

    def extract_eval_result(
        self,
        rollout_result: DynamicRolloutResult,
        log_info=None,
    ) -> dict:
        """Convert one `DynamicRolloutResult` into serializable eval payload.

        Args:
            rollout_result: Rollout output from the agent-loop worker.
            log_info: Optional logging callback.

        Returns:
            A dict containing group-level answer metadata and sample-level turns.
        """
        group_size = rollout_result.group_size
        extra_fields_turn = rollout_result.extra_fields_turn or {}
        extra_fields_traj = rollout_result.extra_fields_traj or {}

        eval_metrics = extra_fields_traj.get("eval_metric") or [None] * group_size
        total_turn_list_metric = (
            extra_fields_traj.get("total_turn_list") or [None] * group_size
        )
        final_answer_format_metric = (
            extra_fields_traj.get("final_answer_format") or [0] * group_size
        )
        llm_reward_metric = extra_fields_traj.get("llm_reward") or [0.0] * group_size

        def _safe_idx(values, idx, default=None):
            """Safely index a list-like container with a default fallback."""
            if values is None or idx >= len(values):
                return default
            return values[idx]

        def _to_py_scalar(value, default=0.0):
            """Convert tensor-like scalars to Python scalars."""
            if value is None:
                return default
            if hasattr(value, "item"):
                return value.item()
            return value

        samples_data: list[dict] = []
        for traj_idx in range(group_size):
            eval_metric = _safe_idx(eval_metrics, traj_idx, None) or {}
            total_turn_list = _safe_idx(total_turn_list_metric, traj_idx, None)
            final_answer_format = (
                _safe_idx(final_answer_format_metric, traj_idx, 0) or 0
            )
            llm_reward = _safe_idx(llm_reward_metric, traj_idx, 0.0) or 0.0

            turn_idxes = [
                i for i, j in enumerate(rollout_result.idx_to_traj) if j == traj_idx
            ]
            turns = []
            for turn_idx in turn_idxes:
                reward_value = _to_py_scalar(
                    _safe_idx(rollout_result.rewards, turn_idx, 0.0), 0.0
                )
                turn_data = {
                    "prompt_text": _safe_idx(
                        extra_fields_turn.get("prompt_text"), turn_idx, None
                    ),
                    "response_text": _safe_idx(
                        extra_fields_turn.get("response_text"), turn_idx, None
                    ),
                    "prompt_ids_length": int(
                        _to_py_scalar(
                            _safe_idx(rollout_result.prompt_lengths, turn_idx, 0), 0
                        )
                    ),
                    "response_ids_length": int(
                        _to_py_scalar(
                            _safe_idx(rollout_result.response_lengths, turn_idx, 0), 0
                        )
                    ),
                    "is_end": bool(
                        _to_py_scalar(
                            _safe_idx(rollout_result.is_end, turn_idx, False), False
                        )
                    ),
                    "reward_score": float(reward_value),
                    "role": _safe_idx(extra_fields_turn.get("role"), turn_idx, None),
                    "tool_call_info": _safe_idx(
                        extra_fields_turn.get("tool_call_info"), turn_idx, None
                    ),
                }
                turns.append(turn_data)

            final_answer = _safe_idx(
                extra_fields_traj.get("final_answer"), traj_idx, None
            )
            if isinstance(final_answer, pd.DataFrame):
                final_answer = final_answer.to_dict(orient="records")
            samples_data.append(
                {
                    "sample_idx": traj_idx,
                    "num_turns": len(turn_idxes),
                    "turns": turns,
                    "origin_question": _safe_idx(
                        extra_fields_traj.get("origin_question"), traj_idx, None
                    ),
                    "final_answer": final_answer,
                    "final_answer_text": _safe_idx(
                        extra_fields_traj.get("final_answer_text"), traj_idx, None
                    ),
                    "planner_summary": _safe_idx(
                        extra_fields_traj.get("planner_summary"), traj_idx, None
                    ),
                    "eval_metric": eval_metric,
                    "total_turn_list": total_turn_list,
                    "final_answer_format": float(final_answer_format),
                    "llm_reward": float(llm_reward),
                }
            )

        answer = (rollout_result.extra_fields_group or {}).get("answer", None)
        eval_result = {
            "group_size": group_size,
            "answer": answer,
            "samples": samples_data,
        }
        if (
            isinstance(answer, dict)
            and "instance_id" in answer
            and log_info is not None
        ):
            log_info(f"finish question id {answer['instance_id']}")
        return eval_result

    def pre_process(self) -> dict:
        """Log evaluation context before the first validation batch."""
        logging.info("=" * 80)
        logging.info("Starting Multi-Agent System Evaluation")
        logging.info("=" * 80)
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Batch size: {self.cfg.data.val_rollout_batch_size}")
        logging.info(f"Group size: {self.cfg.algorithm.get('group_size', 1)}")
        logging.info("=" * 80)
        return {}

    def post_process(
        self,
        context: dict,
    ) -> dict:
        """Aggregate all accumulated batches and write final evaluation artifacts."""
        # Aggregate all results after all batches complete.
        logging.info(f"Aggregating {len(self.accumulated_raw_results)} results...")
        processed_results, final_metrics = self._aggregate_all_results()

        total_queries = len(self.accumulated_raw_results)

        # Save all results to files
        logging.info(f"Saving {total_queries} results to JSON files...")
        self._save_eval_results(processed_results, final_metrics, total_queries)

    def update_pbar(
        self,
        context: dict,
        eval_pbar,
        time_metrics,
    ):
        """Update the evaluation progress bar with live counters and timing."""
        # Update progress bar with current metrics.
        eval_pbar.set_postfix(
            {
                "queries": len(self.accumulated_raw_results),
                "rollout_time": f"{time_metrics.get('rollout', 0):.2f}s",
            }
        )
        eval_pbar.update(1)
