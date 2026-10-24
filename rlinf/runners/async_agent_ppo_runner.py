# Copyright 2026 The RLinf Authors.
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

"""Sequential agent PPO training from a bounded completion Channel."""

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rlinf.data.schema.agent_trajectory import (
    construct_rollout_batch,
    prepare_agent_ppo_targets,
)
from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.runners.agent_ppo_runner import AgentPPORunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker

logger = get_logger()


class AsyncAgentPPORunner(AgentPPORunner):
    """Coordinate sequential model updates with background trajectory production."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        placement = self.component_placement
        actor_gpus = set(placement.get_hardware_ranks("actor"))
        critic_gpus = set(placement.get_hardware_ranks("critic"))
        rollout_gpus = set(placement.get_hardware_ranks("rollout"))
        if actor_gpus != critic_gpus or (
            rollout_gpus != actor_gpus and not rollout_gpus.isdisjoint(actor_gpus)
        ):
            raise ValueError(
                "Async PPO requires shared actor/critic GPUs and equal or disjoint rollout GPUs."
            )
        if (
            placement.has_dedicated_actor_inference
            or placement.has_dedicated_critic_inference
        ):
            raise ValueError("Async PPO uses the training workers for inference.")
        self.critic_input_channel = self.critic_output_channel
        self.async_rollout_channel = Channel.create(
            "AsyncRollout", maxsize=self.cfg.rollout.rollout_buffer_capacity
        )
        self.prompt_capacity = (
            self.cfg.rollout.rollout_buffer_capacity
            + self.cfg.rollout.max_in_flight_rollouts
        )
        self.published_version = 0
        self.actor_version = 0
        self.actor_updates = 0
        self.critic_updates = 0
        self._next_sample_id = 0
        self._producer_epoch = 0
        self._prompt_queue = deque()
        self._pending = {}
        self._data_iterator = None
        self._restored_pending = []  # prompts that were not finished at checkpoint time, but already submitted
        self._restored_buffer = []  # finished trajectories that were not consumed at checkpoint time
        self._stale_count = 0

    def _next_prompt(self) -> tuple[int, list[int], Any] | None:
        if self._next_sample_id >= self.max_steps * self.cfg.data.rollout_batch_size:
            return None
        if not self._prompt_queue:
            if self._data_iterator is None:
                self._data_iterator = iter(self.train_dataloader)
            try:
                batch = next(self._data_iterator)
            except StopIteration:
                self._producer_epoch += 1
                if self._producer_epoch >= self.cfg.runner.max_epochs:
                    raise RuntimeError(
                        "The prompt dataset ended before the learner's sample budget."
                    ) from None
                self._data_iterator = iter(self.train_dataloader)
                batch = next(self._data_iterator)
            requests = build_rollout_requests_from_batch(
                batch, group_size=1, split_size=1, enforce_divisible_batch=False
            )
            for request in requests:
                self._prompt_queue.extend(
                    zip(request.input_ids, request.answers, strict=True)
                )
        prompt, answer = self._prompt_queue.popleft()
        sample_id = self._next_sample_id
        self._next_sample_id += 1
        return sample_id, prompt, answer

    def _put_prompts(self) -> None:
        # Only consumed trajectories free capacity; stale retries keep their slot.
        while len(self._pending) < self.prompt_capacity:
            prompt = self._next_prompt()
            if prompt is None:
                break
            self._pending[prompt[0]] = prompt
            self.dataloader_channel.put(prompt)

    def _save_data_state(self) -> dict[str, Any]:
        buffer = self.async_rollout_channel.peek_all()
        completed = {entry.sample_id for entry in buffer}
        return {
            "dataloader": self.train_dataloader.state_dict(),
            "async_rollout": {
                "buffer": buffer,
                "pending": [
                    prompt
                    for sample_id, prompt in self._pending.items()
                    if sample_id not in completed
                ],
                "prompt_queue": list(self._prompt_queue),
                "next_sample_id": self._next_sample_id,
                "producer_epoch": self._producer_epoch,
                "published_version": self.published_version,
                "actor_version": self.actor_version,
                "actor_updates": self.actor_updates,
                "critic_updates": self.critic_updates,
                "stale_count": self._stale_count,
            },
        }

    def _load_data_state(self, state: dict[str, Any]) -> None:
        if "async_rollout" not in state:
            raise ValueError(
                "Async resume requires a checkpoint containing buffered rollout state."
            )
        rollout = state["async_rollout"]
        if len(rollout["buffer"]) > self.cfg.rollout.rollout_buffer_capacity:
            raise ValueError(
                "Restored trajectories exceed rollout.rollout_buffer_capacity."
            )
        self.train_dataloader.load_state_dict(state["dataloader"])
        self._restored_buffer = rollout["buffer"]
        self._restored_pending = rollout["pending"]
        self._stale_count = rollout.get("stale_count", 0)
        self._pending = {
            entry.sample_id: (entry.sample_id, entry.prompt_ids, entry.answer)
            for entry in self._restored_buffer
        }
        self._pending.update((prompt[0], prompt) for prompt in self._restored_pending)
        self._prompt_queue.extend(rollout["prompt_queue"])
        self._next_sample_id = rollout["next_sample_id"]
        self._producer_epoch = rollout["producer_epoch"]
        for key in (
            "published_version",
            "actor_version",
            "actor_updates",
            "critic_updates",
        ):
            setattr(self, key, rollout[key])

    def _publish(self, initial: bool = False) -> None:
        with self.timer("initial_sync_weights" if initial else "sync_weights"):
            self._sync_weights(weight_version=self.actor_version)
            self.published_version = self.actor_version
            self.rollout.resume_generation().wait()

    def run(self) -> None:
        """Train from completed trajectories while rollout workers run in the background."""
        if (
            self.epoch >= self.cfg.runner.max_epochs
            or self.global_steps >= self.max_steps
        ):
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )
        self.run_timer.start_time()
        self._publish(initial=True)
        for tool_worker in self.tool_workers:
            tool_worker.start_server()

        for entry in self._restored_buffer:
            self.async_rollout_channel.put(entry)
        self._restored_buffer = []
        server_handle: Handle = self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        rollout_handle: Handle = self.agent_loop.run_async_rollout(
            input_channel=self.dataloader_channel,
            output_channel=self.async_rollout_channel,
        )
        for prompt in self._restored_pending:
            self.dataloader_channel.put(prompt)
        self._restored_pending = []
        self._put_prompts()

        trace_path = Path(self.cfg.runner.logger.log_path) / "async_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        while self.global_steps < self.max_steps:
            with self.timer("step"):
                with self.timer("construct_rollout_batch"):
                    batch_state = construct_rollout_batch(
                        input_channel=self.async_rollout_channel,
                        output_channel=self.critic_input_channel,
                        retry_channel=self.dataloader_channel,
                        batch_size=self.cfg.data.rollout_batch_size,
                        seq_length=self.cfg.runner.seq_length,
                        gamma=self.cfg.algorithm.gamma,
                        published_version=self.published_version,
                        max_weight_staleness=self.cfg.rollout.max_weight_staleness,
                    )
                    self._stale_count += batch_state["stale_dropped"]
                    for sample_id in batch_state["sample_ids"]:
                        self._pending.pop(sample_id)
                    self._put_prompts()
                consumed_version = self.published_version
                agent_metrics = MultiAgentLoopWorker.aggregate_metrics(
                    batch_state["agent_metrics"]
                )
                paused_states = []
                if self.release_rollout:
                    with self.timer("pause_rollout"):
                        paused_states = self.rollout.pause_generation().wait()
                    with self.timer("offload_rollout"):
                        self.rollout.offload_engine().wait()
                rollout_state_before = self.rollout.get_async_rollout_state().wait()

                with self.timer("critic/training"):
                    critic_handle: Handle = self.critic.run_training(
                        input_channel=self.critic_input_channel,
                        output_channel=self.rollout_channel,
                        do_offload=False,
                        compute_rollout_metrics=False,
                    )
                    critic_metrics = critic_handle.wait()
                    critic_training_metrics = critic_metrics[0][1]

                # Policy GAE uses values from the updated critic.
                with self.timer("critic/inference"):
                    critic_infer_handle: Handle = self.critic.run_inference(
                        input_channel=self.rollout_channel,
                        output_channel=self.value_channel,
                    )
                    critic_infer_handle.wait()
                with self.timer("advantages"):
                    prepare_agent_ppo_targets(
                        input_channel=self.value_channel,
                        output_channel=self.training_channel,
                        batch_size=self.cfg.data.rollout_batch_size,
                        gamma=self.cfg.algorithm.gamma,
                        gae_lambda=self.cfg.algorithm.gae_lambda,
                        normalize_advantages=self.cfg.algorithm.normalize_advantages,
                        adv_type=self.cfg.algorithm.adv_type,
                        gae_lambda_alpha=self.cfg.algorithm.gae_lambda_alpha,
                    )
                with self.timer("training"):
                    actor_handle: Handle = self.actor.run_training(
                        input_channel=self.training_channel,
                    )
                    actor_metrics = actor_handle.wait()
                    actor_rollout_metrics = actor_metrics[0][0]
                    actor_training_metrics = actor_metrics[0][1]
                rollout_state_after = self.rollout.get_async_rollout_state().wait()
                actor_success = int(actor_training_metrics[0]["actor/update_success"])
                self.actor_updates += actor_success
                self.actor_version += actor_success
                self.critic_updates += sum(
                    int(metrics["critic/update_success"])
                    for metrics in critic_training_metrics
                )
                self.global_steps += 1

                # Checkpoint both models after their updates finish.
                run_time_exceeded = self.run_timer.is_finished()
                _, save_model, is_train_end = check_progress(
                    self.global_steps,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    0.0,
                    run_time_exceeded=run_time_exceeded,
                )
                if save_model:
                    # Shared GPUs still have rollout memory offloaded at this boundary.
                    with self.timer("checkpoint"):
                        self._save_checkpoint()
                if not self.release_rollout:
                    with self.timer("pause_rollout"):
                        paused_states = self.rollout.pause_generation().wait()
                self._publish()

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            time_metrics["critic/training"] = critic_handle.consume_duration()
            time_metrics["critic/inference"] = critic_infer_handle.consume_duration(
                reduction_type="min"
            )
            agent_states = self.agent_loop.get_async_rollout_state().wait()
            async_metrics = {
                "async/buffer_size": self.async_rollout_channel.qsize(),
                "async/buffer_capacity": self.cfg.rollout.rollout_buffer_capacity,
                "async/stale_dropped": self._stale_count,
                "async/in_flight": sum(state["in_flight"] for state in agent_states),
                "async/waiting_for_buffer": sum(
                    state["waiting_for_buffer"] for state in agent_states
                ),
                "async/published_version": self.published_version,
                "async/actor_updates": self.actor_updates,
                "async/critic_updates": self.critic_updates,
                "async/oldest_lag": consumed_version - batch_state["oldest_version"],
                "async/generated_during_training": sum(
                    state["generated_tokens"] for state in rollout_state_after
                )
                - sum(state["generated_tokens"] for state in rollout_state_before),
                "async/retracted_requests": sum(
                    state["parked_requests"] for state in paused_states
                ),
            }
            logging_steps = (self.global_steps - 1) * self.cfg.algorithm.n_minibatches
            log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

            self.metric_logger.log(agent_metrics, logging_steps)
            self.metric_logger.log(log_time_metrics, logging_steps)
            self.metric_logger.log(async_metrics, logging_steps)
            if actor_rollout_metrics is not None:
                rollout_metrics = {
                    f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
                }
                self.metric_logger.log(rollout_metrics, logging_steps)
            for i in range(self.cfg.algorithm.n_minibatches):
                training_metrics = {
                    f"train/{k}": v for k, v in actor_training_metrics[i].items()
                }
                training_metrics.update(
                    {f"train/{k}": v for k, v in critic_training_metrics[-1].items()}
                )
                self.metric_logger.log(training_metrics, logging_steps + i)
            for i, critic_update_metrics in enumerate(critic_training_metrics):
                self.metric_logger.log(
                    {f"critic_update/{k}": v for k, v in critic_update_metrics.items()},
                    (self.global_steps - 1) * len(critic_training_metrics) + i,
                )

            logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}
            logging_metrics.update(agent_metrics)
            if actor_rollout_metrics is not None:
                logging_metrics.update(actor_rollout_metrics)
            logging_metrics.update(actor_training_metrics[-1])
            logging_metrics.update(critic_training_metrics[-1])
            logging_metrics.update(async_metrics)
            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

            trace = {
                "step": self.global_steps,
                "time": time.time(),
                "sample_ids": batch_state["sample_ids"],
                "published_version": self.published_version,
                "actor_updates": self.actor_updates,
                "critic_updates": self.critic_updates,
                "generated_during_training": async_metrics[
                    "async/generated_during_training"
                ],
                "retracted_requests": async_metrics["async/retracted_requests"],
                "oldest_consumed_version": batch_state["oldest_version"],
                "consumed_version": consumed_version,
            }
            with trace_path.open("a") as stream:
                stream.write(json.dumps(trace) + "\n")
            if is_train_end:
                logger.info(
                    f"Step limit given by max_steps={self.max_steps} reached. Stopping run"
                )
                break
            if run_time_exceeded:
                logger.info(
                    f"Time limit given by run_timer={self.run_timer} reached. Stopping run"
                )
                break

        global_pbar.close()
        self.metric_logger.finish()

        self.agent_loop.stop_rollouts().wait()
        self.rollout.stop_serverless().wait()
        rollout_handle.wait()
        server_handle.wait()
        for tool_worker in self.tool_workers:
            tool_worker.stop_server()
