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

"""Synchronous PPO scheduling for multi-turn agents."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from rlinf.data.schema.agent_trajectory import prepare_agent_ppo_targets
from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger

logger = get_logger()


class AgentPPORunner(AgentRunner):
    """Install each synchronous PPO or GRPO update before the next rollout."""

    def __init__(self, *, environment_cohort=None, **kwargs: Any) -> None:
        self.environment_cohort = environment_cohort
        super().__init__(**kwargs)
        placement = self.component_placement
        actor_gpus = set(placement.get_hardware_ranks("actor"))
        critic_gpus = (
            set(placement.get_hardware_ranks("critic"))
            if self.critic is not None
            else set()
        )
        rollout_gpus = set(placement.get_hardware_ranks("rollout"))
        self.shared_training = bool(actor_gpus & critic_gpus)
        self.release_rollout = bool(rollout_gpus & (actor_gpus | critic_gpus))

        self.rollout_output_channels = [self.rollout_channel]
        if (
            self.critic is not None
            and not self.shared_training
            and self.recompute_logprobs
        ):
            self.actor_rollout_channel = Channel.create("ActorRollout")
            self.rollout_output_channels.append(self.actor_rollout_channel)
        self.training_channel = Channel.create("PPOTraining")
        if self._uses_sampling_cohort():
            self.candidate_channel = Channel.create("SynchronousCandidates")
            if self.cfg.algorithm.group_size > 1:
                self.candidate_result_channel = Channel.create("SelectedEpisodes")

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        sampler = (
            self.environment_cohort.sampler(train_dataset)
            if self.environment_cohort is not None
            else None
        )
        super()._build_dataloader(
            train_dataset, val_dataset, collate_fn, batch_sampler=sampler
        )

    def _uses_sampling_cohort(self) -> bool:
        return (
            self.environment_cohort is not None
            or self._sampling_batch_size() > self.cfg.data.rollout_batch_size
        )

    def _sampling_batch_size(self) -> int:
        from rlinf.config import sampling_batch_size

        return sampling_batch_size(self.cfg)

    def _put_batch(self, batch: dict, split_size: Optional[int] = None) -> None:
        if not self._uses_sampling_cohort():
            return super()._put_batch(batch, split_size)
        if self.solid_rollouts:
            raise ValueError(
                "Synchronous oversampling requires one policy rollout group."
            )
        grouped = (
            self.environment_cohort is not None and self.cfg.algorithm.group_size > 1
        )
        requests = build_rollout_requests_from_batch(
            batch,
            group_size=self.cfg.algorithm.group_size,
            split_size=split_size or self.batch_split_num,
            enforce_divisible_batch=False,
            split_trajectories=not grouped,
        )
        if self.environment_cohort is not None:
            self.environment_cohort.prepare(batch, requests)
        cohort_id = f"step-{self.global_steps}-{uuid4().hex}"
        candidate_id = 0
        for request in requests:
            request.cohort_id = cohort_id
            request.weight_version = self.global_steps
            end = candidate_id + len(request.input_ids) * request.n
            request.group_ids = list(range(candidate_id, end, request.n))
            candidate_id = end
            self.dataloader_channel.put(request, async_op=True)
        expected = self._sampling_batch_size() * self.cfg.algorithm.group_size
        if candidate_id != expected:
            raise ValueError(
                f"Expected {expected} episode candidates, got {candidate_id}."
            )
        self._sampling_cohort_id = cohort_id

    def _start_rollout(self) -> Handle:
        if not self._uses_sampling_cohort():
            return super()._start_rollout()
        cohort_id = self._sampling_cohort_id
        candidate_channel = self.candidate_channel
        handle = self.agent_loop.run_agentloop_rollout(
            input_channel=self.dataloader_channel,
            output_channel=candidate_channel,
        )
        selected = []
        selected_groups = []
        complete = {}
        group_size = self.cfg.algorithm.group_size
        expected_candidates = self._sampling_batch_size() * group_size
        expected_selected = self.cfg.data.rollout_batch_size * group_size
        completions = []
        selection_error = None
        cleanup = []
        generation_cleanup = []
        started_at = time.time()
        environment_ready = None
        environment_release = None
        try:
            if self.environment_cohort is not None:
                environment_ready = self.environment_cohort.wait(
                    cohort_id, self.agent_loop
                )
                if environment_ready is not None:
                    logger.info(
                        "Native environments ready for %s: %s",
                        cohort_id,
                        {name: row["ready"] for name, row in environment_ready.items()},
                    )
            while len(selected) < expected_selected:
                event = candidate_channel.get()
                if event["cohort_id"] != cohort_id:
                    raise RuntimeError(
                        f"Expected candidate cohort {cohort_id}, got {event['cohort_id']}."
                    )
                completions.append(dict(event))
                if event["status"] == "error":
                    raise RuntimeError(
                        f"Candidate {event['candidate_id']} infrastructure failure: "
                        f"{event['error']}"
                    )
                candidate_id = event["candidate_id"]
                if not 0 <= candidate_id < expected_candidates:
                    raise RuntimeError(f"Unexpected episode candidate {candidate_id}.")
                if candidate_id in complete:
                    raise RuntimeError(
                        f"Candidate {event['candidate_id']} completed more than once."
                    )
                complete[candidate_id] = event
                group_id = candidate_id // group_size
                member_ids = range(group_id * group_size, (group_id + 1) * group_size)
                if not all(member_id in complete for member_id in member_ids):
                    continue
                if (
                    self.environment_cohort is not None
                    and not self.environment_cohort.select(group_id)
                ):
                    continue
                selected_groups.append(group_id)
                selected.extend(complete[member_id] for member_id in member_ids)
        except BaseException as error:
            selection_error = error
        finally:
            selection_finished_at = time.time()
            selected_ids = [event["candidate_id"] for event in selected]
            cancel_handle = self.agent_loop.cancel_rollout_candidates(
                cohort_id, selected_ids
            )
            try:
                # Abort only this fixed-policy cohort. Closing the cohort also
                # rejects late queued requests before they enter the engine.
                try:
                    generation_cleanup.extend(
                        self.rollout.finish_generation_cohort(cohort_id).wait()
                    )
                except BaseException as error:
                    selection_error = selection_error or error
                try:
                    cleanup = cancel_handle.wait()
                finally:
                    handle.wait()
                # All agent/session owners have stopped: fence their final puts.
                generation_cleanup.extend(
                    self.rollout.finish_generation_cohort(cohort_id).wait()
                )
                if self.environment_cohort is not None:
                    environment_release = self.environment_cohort.verify_release(
                        cohort_id, cleanup
                    )
                while True:
                    try:
                        event = candidate_channel.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if event["cohort_id"] != cohort_id:
                        raise RuntimeError(
                            "Candidate cleanup received a foreign cohort."
                        )
                    completions.append(dict(event))
            except BaseException as error:
                selection_error = selection_error or error
            candidates = [
                record for worker in cleanup for record in worker["candidates"]
            ]
            if selection_error is None:
                errors = [
                    record for record in candidates if record["status"] == "error"
                ]
                if errors:
                    selection_error = RuntimeError(
                        f"Candidate infrastructure failures cannot be replaced by "
                        f"oversampling: {errors}"
                    )
                elif (
                    len(candidates) != expected_candidates
                    or {record["candidate_id"] for record in candidates}
                    != set(range(expected_candidates))
                    or not all(record["cleanup_complete"] for record in candidates)
                ):
                    selection_error = RuntimeError(
                        f"Expected cleanup of all {expected_candidates} unique "
                        "candidates before learner admission."
                    )
            audit = {
                "cohort_id": cohort_id,
                "weight_version": self.global_steps,
                "sampling_batch_size": self._sampling_batch_size(),
                "effective_batch_size": self.cfg.data.rollout_batch_size,
                "group_size": group_size,
                "sampling_episode_count": expected_candidates,
                "effective_episode_count": expected_selected,
                "selected_group_ids": selected_groups,
                "started_at": started_at,
                "finished_at": time.time(),
                "status": "cleanup_passed" if selection_error is None else "failed",
                "selection_finished_at": selection_finished_at,
                "error": None if selection_error is None else repr(selection_error),
                "selected_ids": selected_ids,
                "completion_order": completions,
                "workers": cleanup,
                "generation_cleanup": generation_cleanup,
                "environment_ready": environment_ready,
                "environment_release": environment_release,
            }
            if self.environment_cohort is not None:
                self.environment_cohort.record_selection(audit)
            self._write_sampling_audit(audit)
        if selection_error is not None:
            self.agent_loop.admit_rollout_candidates(cohort_id, None).wait()
            raise selection_error
        try:
            output_channels = (
                [self.candidate_result_channel]
                if group_size > 1
                else self.rollout_output_channels
            )
            admissions = self.agent_loop.admit_rollout_candidates(
                cohort_id, output_channels
            ).wait()
            audit["admissions"] = admissions
            admitted = sum(worker["admitted"] for worker in admissions)
            admitted_ids = [
                candidate_id
                for worker in admissions
                for candidate_id in worker["admitted_ids"]
            ]
            if (
                admitted != expected_selected
                or len(admitted_ids) != admitted
                or set(admitted_ids) != set(selected_ids)
            ):
                raise RuntimeError(
                    f"Expected admission of {expected_selected} episodes, "
                    f"got {admitted}."
                )
            if group_size > 1:
                self._admit_selected_groups(audit)
        except BaseException as error:
            audit.update(status="failed", error=repr(error))
            self._write_sampling_audit(audit)
            raise
        audit.update(status="passed", admission_finished_at=time.time())
        self._write_sampling_audit(audit)
        return handle

    def _admit_selected_groups(self, audit: dict) -> None:
        """Reassemble members only after all candidate owners and queues are clean."""
        episodes = {}
        for _ in audit["selected_ids"]:
            result = self.candidate_result_channel.get()
            if (
                result.group_id in episodes
                or result.group_id not in audit["selected_ids"]
            ):
                raise ValueError(
                    "Selected episode payloads must have unique candidate IDs."
                )
            episodes[result.group_id] = result
        group_size = self.cfg.algorithm.group_size
        groups = []
        payloads = []
        for group_id in audit["selected_group_ids"]:
            result = DynamicRolloutResult.from_trajectory_group(
                [
                    episodes[group_id * group_size + member]
                    for member in range(group_size)
                ],
                group_id,
            )
            rewards = result.group_metadata().rewards
            groups.append({"group_id": group_id, "rewards": rewards})
            payloads.append(result)
        for result in payloads:
            for channel in self.rollout_output_channels:
                channel.put(result)
        audit["selected_groups"] = groups
        audit["zero_variance_groups"] = sum(
            len(set(group["rewards"])) == 1 for group in groups
        )

    def _write_sampling_audit(self, audit: dict) -> None:
        audit_dir = Path(self.cfg.runner.logger.log_path) / "sampling"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{audit['cohort_id']}.json"
        temporary = audit_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(audit, indent=2) + "\n")
        temporary.replace(audit_path)

    def init_rollout_workers(self) -> None:
        """Reuse agent initialization and convert the independent critic checkpoint."""
        if self.environment_cohort is not None:
            self.environment_cohort.validate()
        super().init_rollout_workers()
        if (
            self.critic is not None
            and self.cfg.runner.resume_dir is None
            and self.cfg.critic.megatron.use_hf_ckpt
            and not self.cfg.critic.megatron.get("mbridge", False)
        ):
            from rlinf.utils.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                convert_hf_to_mg,
            )

            convert_hf_to_mg(
                self.cfg.critic.megatron.ckpt_convertor.hf_model_path,
                self.cfg.critic.megatron.ckpt_convertor,
            )

    def _sync_weights(
        self, do_offload: bool = True, weight_version: Optional[int] = None
    ) -> None:
        if self.cfg.rollout.rollout_backend != "sglang":
            weight_version = None
        elif weight_version is None:
            weight_version = self.global_steps
        if (
            self.component_placement.is_collocated
            or self.component_placement.is_auto
            or self.solid_rollouts
        ):
            super()._sync_weights(do_offload=do_offload, weight_version=weight_version)
            return

        super()._sync_weights(do_offload=False, weight_version=weight_version)
        if not do_offload:
            return
        if self.cfg.actor.offload_weight or self.cfg.actor.offload_grad:
            self.actor.offload_model_weights_and_grad(
                offload_weight=self.cfg.actor.offload_weight,
                offload_grad=self.cfg.actor.offload_grad,
            ).wait()
        if self.cfg.actor.offload_optimizer:
            self.actor.offload_megatron_optimizer().wait()

    def _start_actor_update(
        self, input_channel: Channel, version: Optional[int] = None
    ) -> tuple[Handle, Optional[Handle]]:
        """Freeze actor logprobs before updating, preserving behavior metadata."""
        inference_handle = None
        if self.recompute_logprobs:
            inference_handle = self.actor.run_inference(
                input_channel=input_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=False,
            )
            inference_handle.wait()
            input_channel = self.actor_inference_channel
        return (
            self.actor.run_training(input_channel=input_channel, version=version),
            inference_handle,
        )

    def _run_training(
        self, rollout_handle: Handle
    ) -> tuple[Optional[dict], list[dict], dict]:
        """Prepare fixed targets and update models on shared or separate devices."""
        if self.critic is None:
            return super()._run_training(rollout_handle)
        if self.cfg.algorithm.adv_type == "gae_dynamic":
            return self._run_critic_owned_training(rollout_handle)
        # Old predictions: independent devices can infer in parallel.
        critic_infer_handle: Handle = self.critic.run_inference(
            input_channel=self.rollout_channel,
            output_channel=self.value_channel,
            do_offload=False,
        )
        actor_infer_handle = None
        if not self.shared_training and self.recompute_logprobs:
            actor_infer_handle: Handle = self.actor.run_inference(
                input_channel=self.actor_rollout_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=False,
            )
            actor_infer_handle.wait()
        critic_infer_handle.wait()

        # GAE: fix globally normalized targets for every minibatch.
        with self.timer("advantages"):
            prepare_agent_ppo_targets(
                input_channel=self.value_channel,
                output_channel=[
                    self.training_channel,
                    self.critic_output_channel,
                ],
                batch_size=self.cfg.data.rollout_batch_size,
                gamma=self.cfg.algorithm.gamma,
                gae_lambda=self.cfg.algorithm.gae_lambda,
                normalize_advantages=self.cfg.algorithm.normalize_advantages,
                logprob_channel=(
                    None
                    if self.shared_training or not self.recompute_logprobs
                    else self.actor_inference_channel
                ),
                adv_type=self.cfg.algorithm.adv_type,
                expected_weight_version=self.global_steps,
            )

        # Updates: serialize shared devices; otherwise train in parallel.
        critic_handle: Handle = self.critic.run_training(
            input_channel=self.critic_output_channel,
            compute_rollout_metrics=False,
        )
        if not self.shared_training:
            actor_handle: Handle = self.actor.run_training(
                input_channel=self.training_channel,
            )
        critic_metrics = critic_handle.wait()
        critic_training_metrics = critic_metrics[0][1]

        if self.shared_training:
            actor_handle, actor_infer_handle = self._start_actor_update(
                self.training_channel,
            )
        actor_metrics = actor_handle.wait()
        actor_rollout_metrics = actor_metrics[0][0]
        actor_training_metrics = actor_metrics[0][1]

        training_metrics = self._training_metrics(
            actor_training_metrics, critic_training_metrics
        )
        durations = {
            "training": actor_handle.consume_duration(),
            "critic/training": critic_handle.consume_duration(),
            "critic/inference": critic_infer_handle.consume_duration(
                reduction_type="min"
            ),
        }
        if actor_infer_handle is not None:
            durations["inference"] = actor_infer_handle.consume_duration(
                reduction_type="min"
            )
        return actor_rollout_metrics, training_metrics, durations

    def _run_critic_owned_training(
        self, rollout_handle: Handle
    ) -> tuple[Optional[dict], list[dict], dict]:
        """Compute fixed critic targets, update the critic, then update the actor."""
        critic_infer_handle = self.critic.run_inference(
            input_channel=self.rollout_channel,
            output_channel=self.value_channel,
            do_offload=False,
        )
        critic_infer_handle.wait()
        critic_handle = self.critic.run_training(
            input_channel=self.value_channel,
            output_channel=self.critic_output_channel,
            compute_rollout_metrics=False,
        )
        critic_training_metrics = critic_handle.wait()[0][1]

        actor_handle, actor_infer_handle = self._start_actor_update(
            self.critic_output_channel
        )
        rollout_metrics, training_metrics = actor_handle.wait()[0]
        training_metrics = self._training_metrics(
            training_metrics, critic_training_metrics
        )
        durations = {
            "training": actor_handle.consume_duration(),
            "critic/training": critic_handle.consume_duration(),
            "critic/inference": critic_infer_handle.consume_duration(
                reduction_type="min"
            ),
        }
        if actor_infer_handle is not None:
            durations["inference"] = actor_infer_handle.consume_duration(
                reduction_type="min"
            )
        return rollout_metrics, training_metrics, durations

    def _training_metrics(
        self, actor_metrics: list[dict], critic_metrics: list[dict]
    ) -> list[dict]:
        """Report each minibatch and its critic metrics from the last update epoch.

        Critic epochs can outnumber actor epochs. Keep the same logging unit as
        the agent runner instead of pairing optimizer updates one for one.
        """
        minibatches = self.cfg.algorithm.n_minibatches
        return [
            {**actor, **critic}
            for actor, critic in zip(
                actor_metrics, critic_metrics[-minibatches:], strict=True
            )
        ]
