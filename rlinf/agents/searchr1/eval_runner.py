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

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from rlinf.data.io_struct import RolloutResult
from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import local_mkdir_safe
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolWorker, ToolWorkerInfo
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class Searchr1AgentEvalRunner(AgentEvalRunner):
    """Runner for search r1 task RL evaluation."""

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
            agent_loop,
            tool_workers,
            solid_rollouts
        )
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

    def log_eval(
        self,
        context: dict,
        input_channel,
        batch_idx,
        batch,
    ):
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

        # Update aggregated statistics
        batch_accuracy, batch_count = accuracy, total_count
        batch_correct = int(batch_accuracy * batch_count)
        context["total_correct"] += batch_correct
        context["total_samples"] += batch_count
        context["batch_accuracy"] = batch_accuracy

    def pre_process(self) -> dict:
        # raise NotImplementedError()
        logging.info("=" * 80)
        logging.info("Starting Agent System Evaluation")
        logging.info("=" * 80)
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Batch size: {self.cfg.data.rollout_batch_size}")
        logging.info(f"Group size: {self.cfg.algorithm.get('group_size', 1)}")
        logging.info(f"Max turns: {self.cfg.agentloop.get('max_turns', 5)}")
        logging.info("=" * 80)

        context = dict(
            total_correct=0,
            total_samples=0,
        )
        return context

    def post_process(
        self,
        context: dict,
    ) -> dict:
        total_correct = context["total_correct"]
        total_samples = context["total_samples"]
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

    def update_pbar(
        self,
        context: dict,
        eval_pbar,
        time_metrics,
    ):
        # Update progress bar with current metrics
        total_correct = context["total_correct"]
        total_samples = context["total_samples"]
        batch_accuracy = context["batch_accuracy"]
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
