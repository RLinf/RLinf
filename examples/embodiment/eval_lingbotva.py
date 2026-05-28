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

"""Hydra-driven evaluation driver for LingBot-VA on Libero-Object.

A separate driver exists because LingBot-VA's action model maintains
inter-chunk state (KV-cache replay on per-step raw observations). The
generic eval path in ``eval_embodied_agent.py`` routes obs through
``MultiStepRolloutWorker``, which keeps only the last step's wrapped obs
from each chunk and drops the per-step raw obs the model needs to call
``record_chunk_observations``. Running LingBot-VA through that path
silently collapses success rate to ~0%.

``eval_embodiment.sh`` dispatches here whenever
``actor.model.model_type == "lingbotva"``; every other policy continues
to go through ``eval_embodied_agent.py``.

Model hooks consumed (defined on ``LingbotVALiberoActionModel``):
  - ``model.enable_kv_cache_replay`` — gates the recorder loop
  - ``model.record_chunk_observations(env_idx, chunk_obs_list, prev_model_action)``
  - ``model.reset_episode(env_idx)`` on termination
  - ``model._episode_states[env_idx].prev_model_action`` carried across chunks
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models import get_model


def _build_libero_env(cfg: DictConfig) -> LiberoEnv:
    eval_cfg = cfg.env.eval
    return LiberoEnv(
        cfg=eval_cfg,
        num_envs=int(eval_cfg.total_num_envs),
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _chunk_step_with_obs(env: LiberoEnv, chunk_actions: np.ndarray):
    """Step the env chunk_size actions and capture per-step raw obs per env.

    LiberoEnv.chunk_step does the same internally but only exposes the last
    step's obs to callers; the model needs the full per-step history to
    rebuild its KV cache. Duplicating the inner loop here is the cost of
    keeping LiberoEnv.chunk_step generic.
    """
    num_envs, chunk_size, _ = chunk_actions.shape
    raw_obs_history: list[list[dict]] = [[] for _ in range(num_envs)]
    obs_list = []
    rewards = []
    terms = []
    truncs = []
    for i in range(chunk_size):
        action = chunk_actions[:, i]
        wrapped, r, t, tr, _ = env.step(action, auto_reset=False)
        for env_idx in range(num_envs):
            raw_obs_history[env_idx].append(env.current_raw_obs[env_idx])
        obs_list.append(wrapped)
        rewards.append(r)
        terms.append(t)
        truncs.append(tr)
    rewards = torch.stack(rewards, dim=1)
    terms = torch.stack(terms, dim=1)
    truncs = torch.stack(truncs, dim=1)
    past_dones = (terms | truncs).any(dim=1)
    if past_dones.any() and env.auto_reset:
        obs_list[-1], _ = env._handle_auto_reset(
            past_dones.cpu().numpy(), obs_list[-1], {}
        )
    return obs_list[-1], rewards, terms, truncs, raw_obs_history, past_dones


def _evaluate(
    model: torch.nn.Module,
    env: LiberoEnv,
    num_episodes: int,
) -> dict:
    print("[eval_lingbotva] resetting env...", flush=True)
    obs, _ = env.reset()
    print("[eval_lingbotva] env reset complete", flush=True)
    total_steps = 0
    total_episodes_done = 0
    successes = 0
    failures = 0
    task_stats: dict[int, dict[str, int]] = {}
    cur_episode_task_ids = list(env.task_ids.copy())

    target_total_episodes = num_episodes
    max_steps_safety = num_episodes * 260

    chunk_idx = 0
    enable_kv_replay = getattr(model, "enable_kv_cache_replay", False)
    while (
        total_episodes_done < target_total_episodes
        and total_steps < max_steps_safety
    ):
        t0 = time.time()
        action_tensor, _ = model.predict_action_batch(obs, mode="eval")
        infer_sec = time.time() - t0

        t1 = time.time()
        chunk_actions = action_tensor.detach().cpu().numpy()
        (
            final_obs,
            rewards,
            chunk_terminations,
            chunk_truncations,
            raw_obs_history,
            past_dones,
        ) = _chunk_step_with_obs(env, chunk_actions)
        step_sec = time.time() - t1

        successes_this_chunk = chunk_terminations.any(dim=1)
        for env_idx in range(past_dones.shape[0]):
            if past_dones[env_idx].item():
                task_id = int(cur_episode_task_ids[env_idx])
                stats = task_stats.setdefault(
                    task_id, {"success": 0, "total": 0}
                )
                stats["total"] += 1
                if successes_this_chunk[env_idx].item():
                    stats["success"] += 1
                    successes += 1
                else:
                    failures += 1
                total_episodes_done += 1
                cur_episode_task_ids[env_idx] = int(env.task_ids[env_idx])
                if hasattr(model, "reset_episode"):
                    model.reset_episode(env_idx)
            elif enable_kv_replay:
                state = model._episode_states.get(env_idx)
                if state is not None and state.prev_model_action is not None:
                    model.record_chunk_observations(
                        env_idx=env_idx,
                        chunk_obs_list=raw_obs_history[env_idx],
                        prev_model_action=state.prev_model_action,
                    )
        obs = final_obs
        total_steps += chunk_actions.shape[1]
        chunk_idx += 1
        print(
            f"  chunk={chunk_idx:3d} step={total_steps:4d} "
            f"infer={infer_sec:6.2f}s env={step_sec:5.2f}s "
            f"episodes={total_episodes_done} succ={successes} fail={failures}",
            flush=True,
        )

    total = successes + failures
    success_rate = successes / total if total > 0 else 0.0
    return {
        "success_rate": success_rate,
        "successes": successes,
        "failures": failures,
        "task_stats": {
            tid: {
                **stats,
                "rate": stats["success"] / stats["total"]
                if stats["total"]
                else 0.0,
            }
            for tid, stats in task_stats.items()
        },
    }


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_object_eval_lingbotva",
)
def main(cfg: DictConfig) -> None:
    if str(cfg.actor.model.model_type) != "lingbotva":
        raise ValueError(
            f"eval_lingbotva.py expects actor.model.model_type=lingbotva, "
            f"got {cfg.actor.model.model_type!r}. Use eval_embodied_agent.py "
            "for other policies."
        )

    cfg.runner.only_eval = True
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    num_envs = int(cfg.env.eval.total_num_envs)
    rollout_epoch = int(cfg.algorithm.eval_rollout_epoch)
    num_episodes = num_envs * rollout_epoch

    env = _build_libero_env(cfg)
    model = get_model(cfg.actor.model)
    if model is None:
        raise RuntimeError(
            f"Failed to build model of type {cfg.actor.model.model_type}"
        )

    t0 = time.time()
    metrics = _evaluate(model, env, num_episodes=num_episodes)
    metrics["elapsed_sec"] = time.time() - t0
    metrics["num_envs"] = num_envs
    metrics["num_episodes"] = num_episodes

    out_dir = Path(cfg.runner.logger.log_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
