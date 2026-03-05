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
from typing import Union

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
    """Runner for agent task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
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

        # Initialize storage for accumulating evaluation results across all batches
        self.accumulated_results = []

    def _save_eval_results(self, all_results, accuracy, total_count):
        """Save evaluation results to JSON file.

        Args:
            all_results: List of result dictionaries for each sample
            accuracy: Overall accuracy score
            total_count: Total number of samples evaluated
        """
        # Create output directory in the experiment folder
        output_dir = os.path.join(
            self.cfg.runner.output_dir, self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)

        # Fixed filename (no timestamp)
        output_file = os.path.join(output_dir, "eval_results.json")

        # Prepare timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare complete results structure
        results_data = {
            "summary": {
                "dataset_size": total_count,
                "correct_count": sum(1 for r in all_results if r["is_correct"]),
                "accuracy": accuracy,
                "experiment_name": self.cfg.runner.experiment_name,
                "timestamp": timestamp,
                "config": {
                    "group_size": self.cfg.algorithm.get("group_size", 1),
                    "max_turns": self.cfg.agentloop.get("max_turns", 5),
                    "data_paths": OmegaConf.to_container(
                        self.cfg.data.val_data_paths, resolve=True
                    ),
                },
            },
            "results": all_results,
        }

        # Write results to JSON with readable formatting
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Evaluation results saved to: {output_file}")
        return output_file

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

    def log_eval(self, input_channel):
        """Collect evaluation results and compute metrics for a single batch.

        This function:
        1. Collects rollout results from the reward channel for one batch
        2. Computes batch accuracy metrics
        3. Accumulates results (does NOT save to file yet)
        """
        recv_batch_size = 0
        world_size = 1

        total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // world_size
        )

        # Storage for this batch's results
        batch_results = []
        correct_count = 0
        total_count = 0

        while recv_batch_size < total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence

            # Process each sequence in the batch
            # Since group_size=1 for evaluation, num_sequence equals batch_size
            for i in range(rollout_result.num_sequence):
                # Extract information for this sequence
                answer = rollout_result.answers[i] if rollout_result.answers else None
                reward = (
                    rollout_result.rewards[i].item()
                    if rollout_result.rewards is not None
                    else None
                )

                # Decode texts
                prompt_text = (
                    rollout_result.prompt_texts[i]
                    if rollout_result.prompt_texts
                    else None
                )
                response_text = (
                    rollout_result.response_texts[i]
                    if rollout_result.response_texts
                    else None
                )

                # Determine if the answer is correct based on reward
                # Assuming reward > 0 means correct
                is_correct = reward > 0 if reward is not None else False

                if is_correct:
                    correct_count += 1
                total_count += 1

                # Create result entry (index will be updated later)
                result_entry = {
                    "index": len(self.accumulated_results),  # Use global index
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "answer": answer,
                    "reward": reward,
                    "is_correct": is_correct,
                }

                batch_results.append(result_entry)
                # Add to accumulated results immediately
                self.accumulated_results.append(result_entry)

        assert recv_batch_size == total_batch_size_per_dp, (
            f"Expected {total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

        # Compute batch accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Log batch statistics
        logging.info("Batch Evaluation Summary:")
        logging.info(f"  Batch samples: {total_count}")
        logging.info(f"  Batch correct: {correct_count}")
        logging.info(f"  Batch accuracy: {accuracy:.4f}")
        logging.info(f"  Total accumulated samples: {len(self.accumulated_results)}")

        return accuracy, total_count

    def run(self):
        logging.info("=" * 80)
        logging.info("Starting Multi-Agent System Evaluation")
        logging.info("=" * 80)
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Batch size: {self.cfg.data.rollout_batch_size}")
        logging.info(f"Group size: {self.cfg.algorithm.get('group_size', 1)}")
        logging.info(f"Max turns: {self.cfg.agentloop.get('max_turns', 5)}")
        logging.info("=" * 80)

        eval_pbar = tqdm(
            total=len(self.val_dataloader),
            desc="Evaluation",
            ncols=100,
        )

        self.run_timer.start_time()
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        for tool_worker in self.tool_workers:
            tool_worker.start_server()

        total_correct = 0
        total_samples = 0

        try:
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
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.rollout_channel,
                        output_channel=self.reward_channel,
                    )

                    # Log evaluation
                    batch_accuracy, batch_count = self.log_eval(
                        input_channel=self.reward_channel
                    )

                    # Wait for all handles to complete
                    rollout_handle.wait()
                    reward_handle.wait()

                    # Update aggregated statistics
                    batch_correct = int(batch_accuracy * batch_count)
                    total_correct += batch_correct
                    total_samples += batch_count

                time_metrics = self.timer.consume_durations()
                time_metrics["rollout"] = rollout_handle.consume_duration()
                time_metrics["reward"] = reward_handle.consume_duration()

                # Update progress bar with current metrics
                current_accuracy = (
                    total_correct / total_samples if total_samples > 0 else 0.0
                )
                eval_pbar.set_postfix(
                    {
                        "batch_acc": f"{batch_accuracy:.4f}",
                        "overall_acc": f"{current_accuracy:.4f}",
                        "samples": total_samples,
                        "rollout_time": f"{time_metrics.get('rollout', 0):.2f}s",
                    }
                )
                eval_pbar.update(1)

                self.global_steps += 1

        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()

        eval_pbar.close()

        # Final summary
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        logging.info("\n" + "=" * 80)
        logging.info("EVALUATION COMPLETED")
        logging.info("=" * 80)
        logging.info(f"Total samples evaluated: {total_samples}")
        logging.info(f"Total correct: {total_correct}")
        logging.info(
            f"Final accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)"
        )
        logging.info("=" * 80)

        # Save all accumulated results to JSON file
        logging.info(f"Saving {len(self.accumulated_results)} results to JSON file...")
        self._save_eval_results(self.accumulated_results, final_accuracy, total_samples)

        self.metric_logger.finish()
