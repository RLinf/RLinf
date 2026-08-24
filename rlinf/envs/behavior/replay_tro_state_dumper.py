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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.behavior.env_access import (
    get_task_reward,
    stage_idx_from_info,
    stage_idx_from_reward,
    unwrap_behavior_env,
)
from rlinf.envs.behavior.instance_generator import build_output_path, dump_tro_state
from rlinf.envs.behavior.replay_initializer import BehaviorReplayInitializer
from rlinf.envs.utils import to_tensor


def make_replay_initializer(process, replay_cfg: dict) -> BehaviorReplayInitializer:
    replay_cfg = dict(replay_cfg)
    replay_cfg["enabled"] = True
    cfg = OmegaConf.create(OmegaConf.to_container(process.cfg, resolve=False))
    OmegaConf.update(cfg, "replay_init", replay_cfg, merge=False)
    return BehaviorReplayInitializer(cfg, seed_offset=process.replay_seed_offset)


def build_replay_tro_metadata(
    process,
    replay_initializer: BehaviorReplayInitializer,
    job: dict,
    plan,
    child_env,
    info: dict | None,
    output_instance_id: int,
) -> dict[str, Any]:
    stage_idx = stage_idx_from_info(info)
    if stage_idx is None:
        stage_idx = stage_idx_from_reward(child_env)
    if stage_idx is None:
        stage_idx = replay_initializer.stage_index

    metadata = {
        "source_instance_id": int(job["source_instance_id"]),
        "output_instance_id": int(output_instance_id),
        "replay_episode_index": int(plan.episode_index),
        "replay_instance_id": int(plan.instance_id),
        "replay_steps": int(plan.replay_steps),
        "replay_target_step": int(plan.target_step),
        "requested_stage_index": replay_initializer.stage_index,
        "stage_boundary": replay_initializer.stage_boundary,
        "stage_prompts": list(plan.stage_prompts),
    }
    if getattr(replay_initializer, "action_noise_std", 0.0) > 0:
        metadata["action_noise_std"] = float(replay_initializer.action_noise_std)
        metadata["action_noise_indices"] = (
            list(replay_initializer.action_noise_indices)
            if replay_initializer.action_noise_indices is not None
            else None
        )
        metadata["action_noise_stage_indices"] = (
            list(replay_initializer.action_noise_stage_indices)
            if replay_initializer.action_noise_stage_indices is not None
            else None
        )
    if stage_idx is not None:
        metadata["stage_index"] = int(stage_idx)
    return metadata


def step_replay_envs(
    process,
    actions: np.ndarray,
    need_obs: bool,
    env_indices: list[int],
):
    import omnigibson as og

    child_envs = getattr(process.env, "envs", [])
    observations, rewards, terminations, truncations, infos = [], [], [], [], []
    for local_idx, env_idx in enumerate(env_indices):
        child_envs[env_idx]._pre_step(actions[local_idx])
    with og.sim.render_on_step(need_obs):
        og.sim.step()
    for local_idx, env_idx in enumerate(env_indices):
        obs, reward, terminated, truncated, info = child_envs[env_idx]._post_step(
            actions[local_idx], get_obs=need_obs
        )
        observations.append(obs)
        rewards.append(reward)
        terminations.append(terminated)
        truncations.append(truncated)
        infos.append(info)
    if not need_obs:
        observations = None
    return (
        observations,
        to_tensor(rewards),
        to_tensor(terminations),
        to_tensor(truncations),
        infos,
    )


def get_grasp_rejection_targets(child_env) -> list[tuple[str, object]]:
    task_reward = get_task_reward(child_env)
    if task_reward is None:
        return []

    targets = []
    for attr_name in ("_radio_obj", "_target_obj"):
        target_obj = getattr(task_reward, attr_name, None)
        if target_obj is not None:
            targets.append((attr_name.lstrip("_"), target_obj))

    unique_targets = []
    seen = set()
    for name, target_obj in targets:
        target_key = id(target_obj)
        if target_key in seen:
            continue
        seen.add(target_key)
        unique_targets.append((name, target_obj))
    return unique_targets


def grasp_rejection_reason(child_env) -> str | None:
    try:
        from omnigibson.reward_functions.support_utils import is_target_in_hand
    except ImportError:
        return None

    try:
        robot = child_env.task.get_agent(child_env)
    except Exception as exc:
        return f"failed_to_read_robot_grasp_state:{type(exc).__name__}"

    for target_name, target_obj in get_grasp_rejection_targets(child_env):
        try:
            if is_target_in_hand(robot, target_obj):
                obj_name = getattr(target_obj, "name", target_name)
                return f"target_in_hand:{obj_name}"
        except Exception as exc:
            return f"failed_to_check_grasp:{target_name}:{type(exc).__name__}"

    obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
    if isinstance(obj_in_hand, dict):
        for arm_name, held_obj in obj_in_hand.items():
            if held_obj is None:
                continue
            obj_name = getattr(held_obj, "name", type(held_obj).__name__)
            return f"object_in_hand:{arm_name}:{obj_name}"
    return None


def replay_plans(
    process,
    env_indices: list[int],
    plans,
    reject_grasped_tro_states: bool = False,
):
    if not plans:
        return [], []
    max_steps = max(plan.replay_steps for plan in plans)
    rejection_reasons = [None for _ in plans]
    if max_steps <= 0:
        return [{} for _ in plans], rejection_reasons

    child_envs = getattr(process.env, "envs", [])
    action_dim = plans[0].actions.shape[-1]
    action_batch = np.zeros((len(plans), action_dim), dtype=np.float32)
    final_infos = [{} for _ in plans]
    for step_idx in range(max_steps):
        for local_idx, plan in enumerate(plans):
            if plan.replay_steps <= 0:
                action_batch[local_idx] = 0.0
            elif step_idx < plan.replay_steps:
                action_batch[local_idx] = plan.actions[step_idx]
            else:
                action_batch[local_idx] = plan.actions[-1]
        _, _, _, _, final_infos = step_replay_envs(
            process,
            action_batch,
            need_obs=step_idx == max_steps - 1,
            env_indices=env_indices,
        )
        if reject_grasped_tro_states:
            for local_idx, env_idx in enumerate(env_indices):
                if rejection_reasons[local_idx] is not None:
                    continue
                child_env = unwrap_behavior_env(child_envs[env_idx])
                reason = grasp_rejection_reason(child_env)
                if reason is not None:
                    rejection_reasons[local_idx] = f"step={step_idx}:{reason}"
    return final_infos, rejection_reasons


def job_source_instance_id(job: dict, attempt_idx: int) -> int:
    candidate_ids = job.get("candidate_source_instance_ids")
    if candidate_ids:
        candidate_ids = [int(instance_id) for instance_id in candidate_ids]
        offset = int(job.get("candidate_start_offset", 0))
        return candidate_ids[(offset + attempt_idx) % len(candidate_ids)]
    return int(job["source_instance_id"])


def dump_replay_tro_states(process, payload: dict) -> list[dict]:
    jobs = payload.get("jobs", [])
    if not jobs:
        return []

    replay_cfg = dict(payload["replay"])
    replay_initializer = make_replay_initializer(process, replay_cfg)
    output_dir = Path(payload["output_dir"])
    overwrite = bool(payload.get("overwrite", False))
    reject_grasped = bool(replay_cfg.get("reject_grasped_tro_states", False))
    max_attempts = int(replay_cfg.get("max_dump_attempts_per_state", 16))
    if max_attempts <= 0:
        raise ValueError(
            "replay.max_dump_attempts_per_state must be positive when dumping "
            "tro_state files."
        )
    child_envs = getattr(process.env, "envs", [])
    results = []
    attempts = {int(job["output_instance_id"]): 0 for job in jobs}
    pending_jobs = list(jobs)
    skipped_grasped = 0

    while pending_jobs:
        if all("env_slot" in job for job in pending_jobs):
            batch_jobs = []
            env_indices = []
            remaining_jobs = []
            used_env_indices = set()
            for job in pending_jobs:
                env_idx = int(job["env_slot"])
                if env_idx in used_env_indices:
                    remaining_jobs.append(job)
                    continue
                if env_idx < 0 or env_idx >= len(child_envs):
                    raise ValueError(
                        f"Replay dump env_slot={env_idx} is outside process env range."
                    )
                batch_jobs.append(job)
                env_indices.append(env_idx)
                used_env_indices.add(env_idx)
            pending_jobs = remaining_jobs
        else:
            batch_jobs = pending_jobs[: len(child_envs)]
            pending_jobs = pending_jobs[len(child_envs) :]
            env_indices = list(range(len(batch_jobs)))
        source_instance_ids = []
        attempt_jobs = []
        for job in batch_jobs:
            output_instance_id = int(job["output_instance_id"])
            attempt_idx = attempts[output_instance_id]
            source_instance_id = job_source_instance_id(job, attempt_idx)
            attempt_job = dict(job)
            attempt_job["source_instance_id"] = source_instance_id
            attempt_job["dump_attempt"] = attempt_idx + 1
            source_instance_ids.append(source_instance_id)
            attempt_jobs.append(attempt_job)

        process._reset_env_indices(env_indices, instance_ids=source_instance_ids)
        plans = [
            replay_initializer.sample_plan_for_instance(source_instance_id)
            for source_instance_id in source_instance_ids
        ]
        final_infos, rejection_reasons = replay_plans(
            process,
            env_indices,
            plans,
            reject_grasped_tro_states=reject_grasped,
        )

        for batch_idx, (env_idx, job, plan, info, rejection_reason) in enumerate(
            zip(
                env_indices,
                attempt_jobs,
                plans,
                final_infos,
                rejection_reasons,
                strict=True,
            )
        ):
            output_instance_id = int(job["output_instance_id"])
            if rejection_reason is not None:
                skipped_grasped += 1
                attempts[output_instance_id] += 1
                if attempts[output_instance_id] >= max_attempts:
                    raise RuntimeError(
                        "Failed to dump a clean replay tro_state after "
                        f"{max_attempts} attempts for output_instance_id="
                        f"{output_instance_id}; last_source_instance_id="
                        f"{job['source_instance_id']}; last_rejection="
                        f"{rejection_reason}."
                    )
                pending_jobs.append(batch_jobs[batch_idx])
                continue

            child_env = unwrap_behavior_env(child_envs[env_idx])
            metadata = build_replay_tro_metadata(
                process,
                replay_initializer,
                job,
                plan,
                child_env,
                info,
                output_instance_id,
            )
            if reject_grasped:
                metadata["grasp_rejection_checked"] = True
                metadata["dump_attempt"] = int(job["dump_attempt"])
            child_env.task.activity_instance_id = output_instance_id
            output_path = build_output_path(child_env.task, output_dir, "tro_state")
            dump_tro_state(
                child_env,
                output_path=output_path,
                overwrite=overwrite,
                capture_current_robot_pose=True,
                metadata=metadata,
            )
            results.append(
                {
                    "output_path": str(output_path),
                    "output_instance_id": output_instance_id,
                    "source_instance_id": int(job["source_instance_id"]),
                    "replay_episode_index": plan.episode_index,
                    "replay_steps": plan.replay_steps,
                    "replay_target_step": plan.target_step,
                    "replay_stage_idx": metadata.get("stage_index"),
                    "dump_attempt": int(job["dump_attempt"]),
                }
            )

    if skipped_grasped:
        print(
            f"[RLinf replay tro_state INFO] skipped {skipped_grasped} "
            "grasped replay candidates.",
            file=sys.stderr,
            flush=True,
        )
    return results
