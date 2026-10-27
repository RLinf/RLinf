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

import asyncio
import logging
import os
import time
from typing import Optional, Union

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolWorker, ToolWorkerInfo
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.reward.reward_worker import RewardWorker
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker
from rlinf.workers.rollout_buffer.rollout_buffer_worker import RolloutBufferWorker

logging.getLogger().setLevel(logging.INFO)


class AsyncAgentRunner(AgentRunner):
    """Agent runner where rollout and training overlap across steps.

    The agent loop runs continuously (see `run_agentloop_rollout_continuously`)
    and the rollout engine is paused only around each weight sync, so data for
    the next step is generated while the actor trains on the current step. A
    `RolloutBufferWorker` owns admission, buffering and release: it issues
    prompts from its own dataloader as the staleness budget allows and
    releases the oldest finished groups when the runner asks for a batch.
    With `algorithm.staleness_threshold=0` the semantics match the
    synchronous `AgentRunner`.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        rollout: Union["SGLangWorker", "VLLMWorker"],
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        reward: Optional[RewardWorker],
        agent_loop: AgentLoopWorker,
        rollout_buffer: Optional[RolloutBufferWorker] = None,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, Union["SGLangWorker", "VLLMWorker"]] = {},
    ):
        assert placement.is_disaggregated, (
            "AsyncAgentRunner: rollout and actor must be deployed "
            "separately (disaggregated placement); with a shared placement "
            "the runner does not manage weight offload / onload."
        )
        assert reward is None, (
            "AsyncAgentRunner: reward workers are invoked once per training "
            "step by the synchronous runner and are not supported here; "
            "group filtering runs in the RolloutBufferWorker instead."
        )
        assert not cfg.algorithm.recompute_logprobs, (
            "AsyncAgentRunner: recompute_logprobs requires the actor to "
            "double as the inference engine, which is not supported here."
        )
        assert rollout_buffer is not None, (
            "AsyncAgentRunner: a RolloutBufferWorker is required; it owns "
            "the dataloader, the admission budget and the release of "
            "training batches."
        )
        super().__init__(
            cfg,
            placement,
            train_dataset,
            val_dataset,
            rollout,
            inference,
            actor,
            reward,
            agent_loop,
            tool_workers,
            solid_rollouts,
        )
        self.rollout_buffer = rollout_buffer
        self.staleness_threshold = cfg.algorithm.staleness_threshold
        self.pause_mode = cfg.rollout.pause_mode
        # Per-group agent metrics arrive here from the continuous agent loop.
        self.agent_metric_channel = Channel.create("AgentMetric")
        # Finished groups from the agent loop arrive here and are buffered
        # by the rollout buffer worker before reaching the actor.
        self.agent_output_channel = Channel.create("AgentOutput")
        # Weight version each rollout instance has loaded; versions are
        # reported to the buffer worker, which admits prompts against the
        # minimum.
        self._rollout_loaded_versions: dict[int, int] = dict.fromkeys(
            range(self.component_placement.rollout_num_model_instances),
            self.global_steps,
        )

    def _rollout_target(self, instance_ids: Optional[set[int]]):
        """The rollout group restricted to `instance_ids` (None = all).

        `execute_on` applies to exactly one remote call and takes worker
        ranks, so callers must take a fresh target for each call and this
        helper converts instance ids to their entry-process ranks first.
        """
        if instance_ids is None:
            return self.rollout
        ranks = [
            instance_id * self.component_placement.rollout_nnodes_per_model_instance
            for instance_id in instance_ids
        ]
        return self.rollout.execute_on(*ranks)

    def _sync_weights_for_instances(
        self,
        version: int,
        instance_ids: Optional[set[int]] = None,
        pause_generation: bool = True,
    ):
        """Pause generation, sync actor weights to rollout, then resume.

        Order: pause -> transfer (actor pushes, rollout pulls with `version`,
        actor drops the reshard buffers) -> continue. `instance_ids` selects
        the rollout instances to sync; pause and continue target the same
        ones. `pause_generation=False` skips the pause/continue pair: before
        the rollout serverless loop starts there is no generation to pause.
        """
        assert instance_ids is None, (
            "Syncing a subset of rollout instances is not supported: the "
            "actor pushes the weights to every rollout instance in one "
            "sync_model_to_rollout call, so a subset sync needs the actor "
            "side to change first."
        )
        if pause_generation:
            with self.timer("sync/pause"):
                self._rollout_target(instance_ids).pause_generation(
                    self.pause_mode
                ).wait()
        with self.timer("sync/transfer"):
            self.actor.sync_model_to_rollout()
            self._rollout_target(instance_ids).sync_model_from_actor(
                version=version
            ).wait()
            self.actor.del_reshard_state_dict().wait()
        if pause_generation:
            with self.timer("sync/continue"):
                self._rollout_target(instance_ids).continue_generation().wait()
        for instance_id in self._synced_instances(instance_ids):
            self._rollout_loaded_versions[instance_id] = version

    def _synced_instances(self, instance_ids: Optional[set[int]]) -> set[int]:
        """The instance ids a sync with `instance_ids` covers (None = all)."""
        if instance_ids is None:
            return set(self._rollout_loaded_versions)
        return instance_ids

    def _wait_actor(
        self,
        actor_handle: Handle,
        watched_handles: list[Handle],
    ):
        """Wait for the actor while watching the resident loops for death.

        If the agent loop or a rollout serverless loop exits (normally or
        with an error) while the actor is still waiting for data, the actor
        would hang forever - poll them and surface the failure immediately.
        """
        while not actor_handle.done():
            for handle in watched_handles:
                if handle.done():
                    handle.wait()
            time.sleep(0.1)
        return actor_handle.wait()

    def _aggregate_agent_metrics(self, metrics_list: list[dict]) -> dict:
        """Aggregate per-group agent metric dicts in one pass.

        Follows the `post_process_metric` key convention: `__sum__/`,
        `__max__/` and `__min__/` keys merge directly, `__mean__/` values are
        (numerator, denominator) pairs summed separately then divided; keys
        without one of these prefixes are dropped.
        """
        if len(metrics_list) == 0:
            return {}

        all_keys: set[str] = set()
        for metric_dict in metrics_list:
            all_keys.update(metric_dict.keys())

        aggregated = {}
        stat_methods = {
            "__sum__/": sum,
            "__max__/": max,
            "__min__/": min,
            "__mean__/": lambda x: sum(i[0] for i in x) / sum(i[1] for i in x),
        }
        for key in all_keys:
            for stat_key in stat_methods:
                if key.startswith(stat_key):
                    real_key = key[len(stat_key) :]
                    break
            else:
                stat_key, real_key = None, None
            if stat_key is not None:
                values = [metric_dict[key] for metric_dict in metrics_list]
                aggregated[real_key] = stat_methods[stat_key](values)

        return aggregated

    def get_agent_metrics(self) -> dict:
        """Drain and aggregate the agent metrics accumulated since the last call.

        Drains `agent_metric_channel` with `get_nowait()` until
        `asyncio.QueueEmpty`, keeps the records with the bucket key None (all
        records carry it currently) and aggregates them in one pass.
        """
        metrics_list: list[dict] = []
        while True:
            try:
                record = self.agent_metric_channel.get_nowait()
            except asyncio.QueueEmpty:
                break
            if record["bucket"] is None:
                metrics_list.append(record["metrics"])
        return self._aggregate_agent_metrics(metrics_list)

    def _save_checkpoint(self):
        """Save the actor first and the buffer worker's dataloader last.

        `data/data.pt` written by the buffer worker is the completeness
        marker auto resume checks for, so it must exist only after the
        actor checkpoint has landed.
        """
        base_output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name,
            f"checkpoints/global_step_{self.global_steps}",
        )

        # actor
        actor_save_path = os.path.join(base_output_dir, "actor")
        self.actor.save_checkpoint(actor_save_path, self.global_steps).wait()

        # data (dataloader state of the rollout buffer worker)
        data_save_path = os.path.join(base_output_dir, "data")
        self.rollout_buffer.save_checkpoint(data_save_path).wait()

    def _load_buffer_checkpoint(self):
        """Hand the resume directory to the buffer worker.

        Overrides the base runner's dataloader restore: this runner's own
        train_dataloader only sizes `max_steps` and is never iterated or
        checkpointed; the buffer worker's dataloader is the live one.
        """
        if self.cfg.runner.resume_dir is None:
            return
        self.rollout_buffer.load_checkpoint(
            os.path.join(self.cfg.runner.resume_dir, "data"),
            start_step=self.global_steps,
        ).wait()

    def init_actor_critic_workers(self):
        super().init_actor_critic_workers()
        self._load_buffer_checkpoint()

    def run(self):
        if self.global_steps >= self.max_steps:
            # epoch done
            self._stop_rollouts()
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.run_timer.start_time()

        # First weight sync, with the version at the resume point. No
        # pause/continue: the rollout serverless loop has not started yet.
        # Its timing is discarded so it does not add to the first step's.
        self._sync_weights_for_instances(
            version=self.global_steps, pause_generation=False
        )
        self.rollout_buffer.set_loaded_versions(self._rollout_loaded_versions).wait()
        self.timer.consume_durations()

        rollout_handle: Handle = self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        for solid_rollout_name, solid_rollout in self.solid_rollouts.items():
            solid_rollout.rollout_serverless(
                self.solid_generate_input_channels[solid_rollout_name],
                self.generate_output_channel,
            )
        for tool_worker in self.tool_workers:
            tool_worker.start_server()

        agent_handle: Optional[Handle] = None
        buffer_handle: Optional[Handle] = None
        try:
            agent_handle = self.agent_loop.run_agentloop_rollout_continuously(
                input_channel=self.dataloader_channel,
                output_channel=self.agent_output_channel,
                agent_metric_channel=self.agent_metric_channel,
            )
            buffer_handle = self.rollout_buffer.run_continuously(
                input_channel=self.agent_output_channel,
                dataloader_channel=self.dataloader_channel,
                rollout_channel=self.rollout_channel,
            )
            while self.global_steps < self.max_steps:
                with self.timer("step"):
                    # The buffer worker releases the oldest finished
                    # groups as soon as they form a batch; no need to
                    # wait for the release to land.
                    self.rollout_buffer.release_batch(version=self.global_steps)

                    actor_handle: Handle = self.actor.run_training(
                        input_channel=self.rollout_channel,
                        version=self.global_steps,
                    )
                    step_start = time.perf_counter()
                    metrics = self._wait_actor(
                        actor_handle, [agent_handle, buffer_handle, rollout_handle]
                    )
                    actor_wait_time = time.perf_counter() - step_start

                    actor_rollout_metrics = metrics[0][0]
                    actor_training_metrics = metrics[0][1]
                    self.global_steps += 1

                    # Agent loop idle time since the last step, sampled while
                    # the sync has not yet replaced any rollout weights.
                    idle_stats_list = self.agent_loop.pop_idle_stats().wait()
                    agent_loop_idle_ratios = [
                        idle_seconds / window_seconds
                        for idle_seconds, window_seconds in idle_stats_list
                        if window_seconds > 0
                    ]
                    with self.timer("sync_weights"):
                        # With algorithm.staleness_threshold=0 no group should
                        # be in flight and no batch unclaimed when the sync
                        # starts; warn if that does not hold.
                        num_inflight = self.agent_loop.num_inflight_groups().wait()
                        # The buffer is a single-worker group; wait() returns
                        # one result per worker.
                        buffer_metrics = self.rollout_buffer.get_metrics().wait()[0]
                        if self.staleness_threshold == 0 and (
                            any(count != 0 for count in num_inflight)
                            or buffer_metrics["inflight_groups"] != 0
                        ):
                            logging.warning(
                                f"Expected no in-flight groups with "
                                f"algorithm.staleness_threshold=0, found "
                                f"{num_inflight} in the agent loop and "
                                f"{buffer_metrics['inflight_groups']} in "
                                f"the rollout buffer"
                            )
                        self._sync_weights_for_instances(
                            version=self.global_steps,
                        )
                        self.rollout_buffer.set_loaded_versions(
                            self._rollout_loaded_versions
                        ).wait()

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

                    if is_train_end or run_time_exceeded:
                        reason = (
                            f"Step limit given by max_steps={self.max_steps}"
                            if is_train_end
                            else f"Time limit given by run_timer={self.run_timer}"
                        )
                        logging.info(f"{reason} reached. Stopping run")
                        break

                time_metrics = self.timer.consume_durations()
                time_metrics["training"] = actor_handle.consume_duration()
                # Wall-clock wait minus the actor's training time: how long
                # the actor spent waiting for rollout data. The synchronous
                # runner's time/rollout has no source here.
                time_metrics["wait_data"] = actor_wait_time - time_metrics["training"]
                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

                # Agent metrics of the groups completed since the last drain.
                agent_metrics = self.get_agent_metrics()

                self.metric_logger.log(agent_metrics, logging_steps)
                self.metric_logger.log(log_time_metrics, logging_steps)
                rollout_metrics = {
                    f"rollout/{k}": v for k, v in (actor_rollout_metrics or {}).items()
                }
                if agent_loop_idle_ratios:
                    rollout_metrics["rollout/agent_loop_idle_ratio"] = sum(
                        agent_loop_idle_ratios
                    ) / len(agent_loop_idle_ratios)
                rollout_metrics["rollout/inflight_groups"] = sum(num_inflight)
                rollout_metrics["rollout/buffer/size"] = buffer_metrics["size"]
                rollout_metrics["rollout/buffer/inflight_groups"] = buffer_metrics[
                    "inflight_groups"
                ]
                rollout_metrics["rollout/buffer/issued_groups"] = buffer_metrics[
                    "issued_groups"
                ]
                rollout_metrics["rollout/buffer/batch_issue_versions"] = buffer_metrics[
                    "batch_issue_versions"
                ]
                rollout_metrics["rollout/buffer/batch_oldest_issue_version_ratio"] = (
                    buffer_metrics["batch_oldest_issue_version_ratio"]
                )
                rollout_metrics["rollout/buffer/admission_wait_ratio"] = buffer_metrics[
                    "admission_wait_ratio"
                ]
                loaded_versions = list(self._rollout_loaded_versions.values())
                rollout_metrics["rollout/loaded_version_min"] = min(loaded_versions)
                rollout_metrics["rollout/loaded_version_max"] = max(loaded_versions)
                self.metric_logger.log(rollout_metrics, logging_steps)
                for i in range(self.cfg.algorithm.n_minibatches):
                    training_metrics = {
                        f"train/{k}": v for k, v in actor_training_metrics[i].items()
                    }
                    self.metric_logger.log(training_metrics, logging_steps + i)

                logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}

                logging_metrics.update(agent_metrics)
                if actor_rollout_metrics is not None:
                    logging_metrics.update(actor_rollout_metrics)
                logging_metrics.update(actor_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics, refresh=False)
                global_pbar.update(1)

            # Normal exit: stop the buffer worker and the agent loop,
            # confirm both have exited, then stop the rollout engines.
            self._stop_resident_workers(agent_handle, buffer_handle)
            self._stop_rollouts()
        finally:
            # Failure path: stop the resident workers again (a repeated
            # stop is harmless) and tear the servers down. On failures
            # these calls may themselves raise; wrap them so the original
            # error stays the one that surfaces.
            try:
                self._stop_resident_workers(agent_handle, buffer_handle)
            except Exception as e:
                logging.error(f"Stopping the resident workers failed: {e}")
            for tool_worker in self.tool_workers:
                try:
                    tool_worker.stop_server()
                except Exception as e:
                    logging.error(f"Stopping a tool server failed: {e}")
            self.metric_logger.finish()

    def _stop_resident_workers(
        self,
        agent_handle: Optional[Handle],
        buffer_handle: Optional[Handle],
    ):
        """Stop the agent loop and the buffer worker and await both exits."""
        self.agent_loop.stop().wait()
        self.rollout_buffer.stop().wait()
        if agent_handle is not None:
            agent_handle.wait()
            logging.info("Agent loop stopped")
        if buffer_handle is not None:
            buffer_handle.wait()
            logging.info("Rollout buffer worker stopped")
