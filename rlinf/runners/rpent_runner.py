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

"""Dedicated synchronous GRPO runner with oversampling for RPent."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.agents.rpent.simulators import SimulatorBatchSampler, SimulatorSpec
from rlinf.config import sampling_batch_size
from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.runners.agent_runner import AgentRunner
from rlinf.runners.async_agent_runner import AsyncAgentRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.runner_utils import check_progress

logger = logging.getLogger(__name__)


def _aggregate_metrics(raw_metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate RLinf ``__mean__/`` and ``__sum__/`` metric tuples across episodes."""
    sums: dict[str, float] = defaultdict(float)
    mean_num: dict[str, float] = defaultdict(float)
    mean_den: dict[str, float] = defaultdict(float)
    for item in raw_metrics_list:
        for key, value in item.items():
            if key.startswith("__sum__/"):
                sums[key.removeprefix("__sum__/")] += float(value)
            elif key.startswith("__mean__/"):
                num, den = value
                name = key.removeprefix("__mean__/")
                mean_num[name] += float(num)
                mean_den[name] += float(den)
    result: dict[str, float] = dict(sums)
    for name, num in mean_num.items():
        den = mean_den[name]
        result[name] = num / den if den > 0 else 0.0
    return result


def validate_sim_pool_capacity(
    *,
    simulators: dict[str, SimulatorSpec],
    default_endpoint: str | None,
    sampling_groups: int,
    group_size: int,
    num_agent_workers: int,
    episodes_per_worker: int | None,
    fetch_health=None,
) -> dict[str, dict[str, Any]]:
    """Fail fast if resident simulator workers or client concurrency throttle oversampling."""
    import httpx

    if fetch_health is None:

        def fetch_health(endpoint: str) -> dict[str, Any]:
            resp = httpx.get(f"{endpoint}/health", timeout=15.0, trust_env=False)
            resp.raise_for_status()
            return resp.json()

    demands: list[tuple[str, str, int]] = []
    if simulators:
        for name, spec in simulators.items():
            demands.append((name, spec.endpoint, spec.sampling_groups * group_size))
    elif default_endpoint:
        demands.append(
            ("default", default_endpoint.rstrip("/"), sampling_groups * group_size)
        )

    total_required = sampling_groups * group_size
    if episodes_per_worker is not None and episodes_per_worker > 0:
        max_groups_per_worker = math.ceil(sampling_groups / max(num_agent_workers, 1))
        per_worker_required = max_groups_per_worker * group_size
        if episodes_per_worker < per_worker_required:
            raise RuntimeError(
                f"tools.rpent.episodes_per_worker={episodes_per_worker} throttles single-step "
                f"oversampling ({sampling_groups} groups x {group_size} across "
                f"{num_agent_workers} agentloop workers needs >= {per_worker_required} "
                f"concurrent episodes per worker, or set episodes_per_worker: null)."
            )

    statuses: dict[str, dict[str, Any]] = {}
    for name, endpoint, required in demands:
        try:
            health = fetch_health(endpoint)
        except Exception as exc:
            raise RuntimeError(
                f"RPent simulator pool '{name}' at {endpoint}/health is unreachable: {exc}"
            ) from exc
        workers_total = int(health.get("workers_total", 0))
        workers_idle = int(health.get("workers_idle", 0))
        if workers_total < required:
            raise RuntimeError(
                f"RPent simulator pool '{name}' at {endpoint} has workers_total={workers_total}, "
                f"fewer than single-step oversampling demand {required} "
                f"(group_size={group_size}). Increase workers in sim_services.yaml or reduce sampling_batch_size."
            )
        if workers_idle < required:
            logger.warning(
                "RPent simulator pool '%s' at %s has workers_idle=%d < demand=%d "
                "(workers_total=%d, active_sessions=%s)",
                name,
                endpoint,
                workers_idle,
                required,
                workers_total,
                health.get("active_sessions"),
            )
        statuses[name] = {
            "endpoint": endpoint,
            "required": required,
            "workers_total": workers_total,
            "workers_idle": workers_idle,
        }

    if statuses:
        summary = ", ".join(
            f"{name}={info['workers_idle']}/{info['workers_total']} (need {info['required']})"
            for name, info in statuses.items()
        )
        logger.info(
            "Validated RPent simulator pool capacity: %s | total_demand=%d | episodes_per_worker=%s",
            summary,
            total_required,
            "unlimited"
            if episodes_per_worker is None or episodes_per_worker <= 0
            else episodes_per_worker,
        )
    return statuses


class RPentGRPORunner(AgentRunner):
    """Synchronous GRPO runner with oversampling and multi-simulator quotas.

    Keeps upstream ``AgentRunner`` and ``ReasoningRunner`` untouched while
    supporting oversampled candidate selection, instant straggler cancellation
    and SGLang cohort fencing, and clean collocated offload/reload.
    """

    def __init__(
        self,
        *,
        simulators: dict[str, SimulatorSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        self.simulators = simulators or {}
        super().__init__(**kwargs)
        actor_gpus = set(self.component_placement.get_hardware_ranks("actor"))
        rollout_gpus = set(self.component_placement.get_hardware_ranks("rollout"))
        self.release_rollout = bool(rollout_gpus & actor_gpus)
        self.candidate_channel = Channel.create("RPentCandidates")
        self.selected_channel = Channel.create("RPentSelected")
        self._cohort_id: str | None = None
        self._group_envs: dict[int, str] = {}

    def _sampling_batch_size(self) -> int:
        return sampling_batch_size(self.cfg)

    def _uses_oversampling(self) -> bool:
        return (
            bool(self.simulators)
            or self._sampling_batch_size() > self.cfg.data.rollout_batch_size
        )

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        if collate_fn is None:
            from rlinf.data.datasets.reasoning import collate_fn

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if self.simulators:
            sampler = SimulatorBatchSampler(
                pool_sizes=[len(ds) for ds in train_dataset.datasets],
                quotas=[spec.sampling_groups for spec in self.simulators.values()],
                shuffle=bool(self.cfg.data.shuffle),
                seed=int(self.cfg.data.get("seed", 1)),
            )
            self.train_dataloader = StatefulDataLoader(
                dataset=train_dataset,
                num_workers=self.cfg.data.num_workers,
                collate_fn=collate_fn,
                batch_sampler=sampler,
            )
        else:
            if self.cfg.data.shuffle:
                gen = torch.Generator().manual_seed(int(self.cfg.data.get("seed", 1)))
                sampler = RandomSampler(train_dataset, generator=gen)
            else:
                sampler = SequentialSampler(train_dataset)
            self.train_dataloader = StatefulDataLoader(
                dataset=train_dataset,
                batch_size=self._sampling_batch_size(),
                num_workers=self.cfg.data.num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        logger.info(
            "RPentGRPORunner train dataloader batches: %d", len(self.train_dataloader)
        )

    def _sync_weights(self) -> None:
        self.actor.sync_model_to_rollout()
        self.rollout.sync_model_from_actor(version=self.global_steps).wait()
        self.actor.del_reshard_state_dict().wait()
        if self.component_placement.is_collocated or self.component_placement.is_auto:
            self.rollout.onload_kv_cudagraph().wait()

    def _put_batch(self, batch: dict, split_size: Optional[int] = None) -> None:
        group_size = int(self.cfg.algorithm.group_size)
        requests = build_rollout_requests_from_batch(
            batch,
            group_size=group_size,
            split_size=split_size or self.batch_split_num,
            enforce_divisible_batch=False,
            split_trajectories=False,
        )
        cohort_id = f"step-{self.global_steps}-{uuid4().hex[:8]}"
        self._cohort_id = cohort_id
        self._group_envs = {}

        candidate_id = 0
        for request in requests:
            request.cohort_id = cohort_id
            request.weight_version = self.global_steps
            group_ids = []
            for answer in request.answers:
                gid = candidate_id // group_size
                task = json.loads(answer[0] if isinstance(answer, list) else answer)
                self._group_envs[gid] = task.get("environment", "default")
                group_ids.append(candidate_id)
                candidate_id += request.n
            request.group_ids = group_ids
            self.dataloader_channel.put(request, async_op=True)

        expected = self._sampling_batch_size() * group_size
        if candidate_id != expected:
            raise ValueError(f"Expected {expected} candidates, got {candidate_id}")

    def _run_oversampled_rollout(self) -> dict[str, float]:
        """Run oversampled candidate cohort, select complete groups, cancel stragglers."""
        cohort_id = self._cohort_id
        assert cohort_id is not None
        group_size = int(self.cfg.algorithm.group_size)
        expected_candidates = self._sampling_batch_size() * group_size
        target_groups = int(self.cfg.data.rollout_batch_size)
        remaining_quota = (
            {name: spec.rollout_groups for name, spec in self.simulators.items()}
            if self.simulators
            else None
        )

        handle = self.agent_loop.run_agentloop_rollout(
            input_channel=self.dataloader_channel,
            output_channel=self.candidate_channel,
        )

        complete: dict[int, dict[str, Any]] = {}
        selected_groups: list[int] = []
        selected_ids: list[int] = []
        closed_envs: set[str] = set()
        early_cancel_handles: list[Handle] = []
        selection_error: BaseException | None = None
        started = time.time()

        try:
            while len(selected_groups) < target_groups:
                event = self.candidate_channel.get()
                if event["cohort_id"] != cohort_id:
                    raise RuntimeError(
                        f"Expected cohort {cohort_id}, got {event['cohort_id']}"
                    )
                if event["status"] == "error":
                    raise RuntimeError(
                        f"Candidate {event['candidate_id']} failed: {event['error']}"
                    )
                cid = int(event["candidate_id"])
                complete[cid] = event
                gid = cid // group_size
                members = range(gid * group_size, (gid + 1) * group_size)
                if not all(m in complete for m in members):
                    continue
                if gid in selected_groups:
                    continue
                env: str | None = None
                if remaining_quota is not None:
                    env = self._group_envs[gid]
                    if remaining_quota.get(env, 0) <= 0:
                        continue
                    remaining_quota[env] -= 1
                selected_groups.append(gid)
                selected_ids.extend(members)
                if (
                    remaining_quota is not None
                    and env is not None
                    and remaining_quota[env] == 0
                    and env not in closed_envs
                    and len(selected_groups) < target_groups
                ):
                    closed_envs.add(env)
                    selected_set = set(selected_groups)
                    drop_ids = [
                        g * group_size + m
                        for g, e in self._group_envs.items()
                        if e == env and g not in selected_set
                        for m in range(group_size)
                    ]
                    if drop_ids:
                        logger.info(
                            "RPent simulator '%s' met quota (%d groups); "
                            "cancelling %d unselected straggler candidates early",
                            env,
                            self.simulators[env].rollout_groups,
                            len(drop_ids),
                        )
                        early_cancel_handles.append(
                            self.agent_loop.cancel_candidate_ids(cohort_id, drop_ids)
                        )
        except BaseException as exc:
            selection_error = exc
        finally:
            for early in early_cancel_handles:
                try:
                    early.wait()
                except BaseException as exc:
                    selection_error = selection_error or exc
            cancel_handle = self.agent_loop.cancel_rollout_candidates(
                cohort_id, selected_ids
            )
            try:
                self.rollout.finish_generation_cohort(cohort_id).wait()
            except BaseException as exc:
                selection_error = selection_error or exc
            try:
                cleanup = cancel_handle.wait()
            finally:
                handle.wait()
            try:
                self.rollout.finish_generation_cohort(cohort_id).wait()
            except BaseException as exc:
                selection_error = selection_error or exc
            # Fence simulator pools so no disconnected/late open leaves leaked slots.
            import httpx

            endpoints = (
                [spec.endpoint for spec in self.simulators.values()]
                if self.simulators
                else (
                    [str(self.cfg.tools.rpent.remote_toolkit_endpoint).rstrip("/")]
                    if self.cfg.tools.rpent.get("remote_toolkit_endpoint")
                    else []
                )
            )
            for ep in endpoints:
                try:
                    httpx.post(
                        f"{ep}/cohorts/{cohort_id}/close",
                        json={},
                        timeout=30.0,
                        trust_env=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed fencing simulator cohort %s on %s: %r",
                        cohort_id,
                        ep,
                        exc,
                    )
            # Drain any late completion notices sent before closure.
            while True:
                try:
                    self.candidate_channel.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if selection_error is not None:
            self.agent_loop.admit_rollout_candidates(cohort_id, None).wait()
            raise selection_error

        # Verify no infrastructure errors occurred in any candidate.
        all_records = [rec for worker in cleanup for rec in worker["candidates"]]
        errors = [rec for rec in all_records if rec["status"] == "error"]
        if errors:
            self.agent_loop.admit_rollout_candidates(cohort_id, None).wait()
            raise RuntimeError(
                f"Infrastructure failures cannot be masked by oversampling: {errors}"
            )

        admissions = self.agent_loop.admit_rollout_candidates(
            cohort_id, self.selected_channel
        ).wait()
        admitted_ids = [cid for worker in admissions for cid in worker["admitted_ids"]]
        if len(admitted_ids) != len(selected_ids) or set(admitted_ids) != set(
            selected_ids
        ):
            raise RuntimeError(
                f"Expected {len(selected_ids)} admitted episodes, got {len(admitted_ids)}"
            )

        episodes: dict[int, DynamicRolloutResult] = {}
        for _ in selected_ids:
            res = self.selected_channel.get()
            episodes[res.group_id] = res

        group_summaries = []
        for gid in selected_groups:
            trajs = [episodes[gid * group_size + m] for m in range(group_size)]
            group_result = DynamicRolloutResult.from_trajectory_group(trajs, gid)
            rewards = group_result.group_metadata().rewards
            group_summaries.append(
                {"group_id": gid, "env": self._group_envs[gid], "rewards": rewards}
            )
            self.rollout_channel.put(group_result)

        raw_metrics = [m for worker in admissions for m in worker["metrics"]]
        agent_metrics = _aggregate_metrics(raw_metrics)
        agent_metrics["rpent/oversample_candidates"] = float(expected_candidates)
        agent_metrics["rpent/zero_variance_groups"] = float(
            sum(len(set(g["rewards"])) == 1 for g in group_summaries)
        )

        # Persist compact per-step audit record.
        audit_dir = Path(self.cfg.runner.logger.log_path) / "sampling"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "cohort_id": cohort_id,
            "step": self.global_steps,
            "elapsed_s": round(time.time() - started, 2),
            "selected_groups": group_summaries,
            "metrics": agent_metrics,
        }
        (audit_dir / f"{cohort_id}.json").write_text(json.dumps(audit, indent=2) + "\n")

        if self.release_rollout:
            self.rollout.offload_engine().wait()

        return agent_metrics

    def _run_training_step(self) -> tuple[dict | None, list[dict], dict[str, float]]:
        handles: dict[str, Handle] = {}
        input_channel = self.rollout_channel
        if self.recompute_logprobs:
            handles["inference"] = self.actor_inference.run_inference(
                input_channel=input_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=self.compute_ref_logprobs,
            )
            input_channel = self.actor_inference_channel
        handles["training"] = self.actor.run_training(input_channel=input_channel)
        rollout_metrics, training_metrics = handles["training"].wait()[0]
        durations = {
            key: (
                handle.consume_duration(reduction_type="min")
                if key == "inference"
                else handle.consume_duration()
            )
            for key, handle in handles.items()
        }
        return rollout_metrics, training_metrics, durations

    def _sim_endpoints(self) -> list[str]:
        if self.simulators:
            return [spec.endpoint for spec in self.simulators.values()]
        rpent = (self.cfg.get("tools") or {}).get("rpent") or {}
        if rpent.get("remote_toolkit_endpoint"):
            return [str(rpent["remote_toolkit_endpoint"]).rstrip("/")]
        return []

    def _sleep_pools_enabled(self) -> bool:
        rpent = (self.cfg.get("tools") or {}).get("rpent") or {}
        return bool(rpent.get("sleep_sim_pools_during_training", False))

    def _pool_lifecycle(self, action: str) -> None:
        """Sleep or wake collocated simulator pools around Megatron training."""
        import concurrent.futures

        import httpx

        endpoints = self._sim_endpoints()
        if not endpoints:
            return
        timeout_s = 600.0 if action == "wake" else 120.0

        def _call(ep: str) -> tuple[str, dict[str, Any]]:
            resp = httpx.post(
                f"{ep}/pool/{action}", json={}, timeout=timeout_s, trust_env=False
            )
            resp.raise_for_status()
            return ep, resp.json()

        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            results = list(pool.map(_call, endpoints))
        summary = ", ".join(
            f"{info.get('environment', ep)}={info.get('workers_total')}/{info.get('workers_target')} "
            f"(sleeping={info.get('sleeping')})"
            for ep, info in results
        )
        logger.info(
            "RPent simulator pool %s completed in %.1fs: %s",
            action,
            time.time() - started,
            summary,
        )
        if action == "wake":
            for ep, info in results:
                target = int(info.get("workers_target", 0))
                total = int(info.get("workers_total", 0))
                if target > 0 and total < target:
                    raise RuntimeError(
                        f"Simulator pool at {ep} woke with {total}/{target} workers"
                    )

    def _validate_sim_pool_capacity(self) -> None:
        rpent = (self.cfg.get("tools") or {}).get("rpent") or {}
        ep_limit = rpent.get("episodes_per_worker")
        validate_sim_pool_capacity(
            simulators=self.simulators,
            default_endpoint=rpent.get("remote_toolkit_endpoint"),
            sampling_groups=self._sampling_batch_size(),
            group_size=int(self.cfg.algorithm.group_size),
            num_agent_workers=len(
                self.component_placement.get_hardware_ranks("agentloop")
            ),
            episodes_per_worker=int(ep_limit) if ep_limit is not None else None,
        )

    def run(self) -> None:
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0 or self.global_steps >= self.max_steps:
            self._stop_rollouts()
            return
        self._validate_sim_pool_capacity()

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="RPent GRPO Step",
            ncols=120,
        )
        self.run_timer.start_time()
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )

        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)
                    with self.timer("sync_weights"):
                        self._sync_weights()
                    with self.timer("rollout"):
                        agent_metrics = self._run_oversampled_rollout()
                    if self._sleep_pools_enabled():
                        with self.timer("sim_sleep"):
                            self._pool_lifecycle("sleep")
                    actor_rollout_metrics, actor_training_metrics, durations = (
                        self._run_training_step()
                    )

                self.global_steps += 1
                time_metrics = self.timer.consume_durations()
                time_metrics.update(durations)
                if (
                    actor_rollout_metrics is not None
                    and "rollout_tflops" in actor_rollout_metrics
                ):
                    time_metrics.update(
                        self._compute_flops_metrics(time_metrics, actor_rollout_metrics)
                    )

                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                self.metric_logger.log(
                    {f"time/{k}": v for k, v in time_metrics.items()}, logging_steps
                )
                if agent_metrics:
                    self.metric_logger.log(
                        {f"agent/{k}": v for k, v in agent_metrics.items()},
                        logging_steps,
                    )
                if actor_rollout_metrics:
                    self.metric_logger.log(
                        {f"rollout/{k}": v for k, v in actor_rollout_metrics.items()},
                        logging_steps,
                    )
                for i in range(self.cfg.algorithm.n_minibatches):
                    self.metric_logger.log(
                        {f"train/{k}": v for k, v in actor_training_metrics[i].items()},
                        logging_steps + i,
                    )

                logger.info(
                    "Step %d/%d | success_rate=%.2f%% | turns=%.1f | rows=%.1f | "
                    "rollout=%.1fs | train=%.1fs | sync=%.1fs | loss=%.4f",
                    self.global_steps,
                    self.max_steps,
                    100.0 * agent_metrics.get("rpent/success_rate", 0.0),
                    agent_metrics.get("rpent/turns", 0.0),
                    agent_metrics.get("rpent/rows", 0.0),
                    time_metrics.get("rollout", 0.0),
                    time_metrics.get("training", 0.0),
                    time_metrics.get("sync_weights", 0.0),
                    actor_training_metrics[-1].get(
                        "actor/final_loss",
                        actor_training_metrics[-1].get(
                            "actor/total_loss", float("nan")
                        ),
                    ),
                )

                global_pbar.update(1)
                _, save_model, is_train_end = check_progress(
                    self.global_steps,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=self.run_timer.is_finished(),
                )
                if save_model:
                    self._save_checkpoint()
                if is_train_end:
                    self._stop_rollouts()
                    self.metric_logger.finish()
                    return
                if self._sleep_pools_enabled():
                    with self.timer("sim_wake"):
                        self._pool_lifecycle("wake")

        self._stop_rollouts()
        self.metric_logger.finish()


class RPentAsyncGRPORunner(AsyncAgentRunner):
    """Collocated or disaggregated asynchronous GRPO runner for RPent."""

    def __init__(
        self,
        *,
        simulators: dict[str, SimulatorSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        self.simulators = simulators or {}
        super().__init__(**kwargs)

    def _validate_sim_pool_capacity(self) -> None:
        rpent = (self.cfg.get("tools") or {}).get("rpent") or {}
        ep_limit = rpent.get("episodes_per_worker")
        validate_sim_pool_capacity(
            simulators=self.simulators,
            default_endpoint=rpent.get("remote_toolkit_endpoint"),
            sampling_groups=int(self.cfg.rollout_buffer.max_inflight_groups),
            group_size=int(self.cfg.algorithm.group_size),
            num_agent_workers=len(
                self.component_placement.get_hardware_ranks("agentloop")
            ),
            episodes_per_worker=int(ep_limit) if ep_limit is not None else None,
        )

    def _on_step_logged(
        self,
        *,
        time_metrics: dict,
        agent_metrics: dict,
        rollout_metrics: dict,
        actor_training_metrics: list[dict],
    ) -> None:
        last_train = actor_training_metrics[-1] if actor_training_metrics else {}
        loss = last_train.get(
            "actor/final_loss", last_train.get("actor/total_loss", float("nan"))
        )
        stale_mean = rollout_metrics.get(
            "rollout/data_staleness/mean", float("nan")
        )
        stale_max = rollout_metrics.get("rollout/data_staleness/max", float("nan"))
        logger.info(
            "Async Step %d/%d | success_rate=%.2f%% | turns=%.1f | rows=%.1f | "
            "wait_data=%.1fs | pause=%.1fs | offload=%.1fs | train=%.1fs | "
            "sync=%.1fs | step=%.1fs | staleness=%.2f/%.0f | loss=%.4f",
            self.global_steps,
            self.max_steps,
            100.0 * agent_metrics.get("rpent/success_rate", 0.0),
            agent_metrics.get("rpent/turns", 0.0),
            agent_metrics.get("rpent/rows", 0.0),
            time_metrics.get("wait_data", 0.0),
            time_metrics.get("rollout/pause", 0.0),
            time_metrics.get("rollout/offload", 0.0),
            time_metrics.get("training", 0.0),
            time_metrics.get("sync_weights", 0.0),
            time_metrics.get("step", 0.0),
            stale_mean,
            stale_max,
            loss,
        )
        audit_dir = Path(self.cfg.runner.logger.log_path) / "sampling"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "mode": "async",
            "step": self.global_steps,
            "time_metrics": time_metrics,
            "agent_metrics": agent_metrics,
            "rollout_metrics": rollout_metrics,
            "train_metrics": last_train,
        }
        (audit_dir / f"async-step-{self.global_steps}.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n"
        )

    def run(self) -> None:
        self._validate_sim_pool_capacity()
        super().run()
