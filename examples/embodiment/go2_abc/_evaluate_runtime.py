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

"""Run Go2 checkpoint inference in a single Isaac Sim process."""

import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

from rlinf.envs.sim.isaaclab.tasks.go2 import (
    ROUGH,
    STAIRS,
    install_rough_action_bound,
    register_go2_tasks,
)
from rlinf.utils.logging import get_logger

from ._passage_metrics import PassageRuntime, capture_geometry, summarize_passage
from ._tracking_metrics import (
    TASK_SUCCESS_CONFIGS,
    EpisodeStatsTracker,
    compute_step_success_from_errors,
    compute_tracking_errors,
)
from .evaluate import TASKS

logger = get_logger()


def _align_agent_cfg_to_checkpoint(agent_cfg: object, checkpoint_path: str) -> None:
    """Make the eval runner schema match the saved checkpoint before loading weights."""
    try:
        from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
    except ImportError:
        RslRlMLPModelCfg = None
    loaded = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    actor_state = loaded.get("actor_state_dict", {})
    critic_state = loaded.get("critic_state_dict", {})
    model_state = loaded.get("model_state_dict", {})

    actor_has_obs_norm = any("obs_normalizer." in key for key in actor_state) or any(
        key.startswith("actor_obs_normalizer.") for key in model_state
    )
    critic_has_obs_norm = any("obs_normalizer." in key for key in critic_state) or any(
        key.startswith("critic_obs_normalizer.") for key in model_state
    )
    actor_uses_log_std = (
        any("distribution.log_std_param" in key for key in actor_state)
        or "log_std" in model_state
    )

    if actor_has_obs_norm or critic_has_obs_norm:
        if hasattr(agent_cfg, "policy") and agent_cfg.policy is not None:
            if hasattr(agent_cfg.policy, "actor_obs_normalization"):
                agent_cfg.policy.actor_obs_normalization = actor_has_obs_norm
            if hasattr(agent_cfg.policy, "critic_obs_normalization"):
                agent_cfg.policy.critic_obs_normalization = critic_has_obs_norm
        if (
            hasattr(agent_cfg, "actor")
            and agent_cfg.actor is not None
            and hasattr(agent_cfg.actor, "obs_normalization")
        ):
            agent_cfg.actor.obs_normalization = actor_has_obs_norm
        if (
            hasattr(agent_cfg, "critic")
            and agent_cfg.critic is not None
            and hasattr(agent_cfg.critic, "obs_normalization")
        ):
            agent_cfg.critic.obs_normalization = critic_has_obs_norm
        logger.info(
            "[INFO] Eval cfg aligned to checkpoint observation normalization: "
            f"actor={actor_has_obs_norm}, critic={critic_has_obs_norm}"
        )

    if actor_uses_log_std:
        if (
            hasattr(agent_cfg, "policy")
            and agent_cfg.policy is not None
            and hasattr(agent_cfg.policy, "noise_std_type")
        ):
            agent_cfg.policy.noise_std_type = "log"
        if (
            hasattr(agent_cfg, "actor")
            and agent_cfg.actor is not None
            and RslRlMLPModelCfg is not None
            and hasattr(agent_cfg.actor, "distribution_cfg")
        ):
            dist_cfg = getattr(agent_cfg.actor, "distribution_cfg", None)
            init_std = getattr(dist_cfg, "init_std", 1.0)
            agent_cfg.actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=float(init_std),
                std_type="log",
            )
        logger.info(
            "[INFO] Eval cfg aligned to checkpoint actor distribution std_type=log."
        )


def _load_runner_checkpoint_compat(runner, checkpoint_path: str) -> None:
    """Load checkpoint with compatibility fallback for std/log-std key rename."""
    try:
        runner.load(checkpoint_path, map_location=runner.device)
        return
    except KeyError as exc:
        if str(exc).strip("'\"") != "actor_state_dict":
            raise

        loaded = torch.load(
            checkpoint_path, weights_only=True, map_location=runner.device
        )
        model_state = loaded.get("model_state_dict")
        if model_state is None:
            raise

        actor_state = {}
        critic_state = {}
        for key, value in model_state.items():
            if key == "std":
                actor_state["distribution.std_param"] = value
            elif key == "log_std":
                actor_state["distribution.log_std_param"] = value
            elif key.startswith("actor."):
                actor_state[f"mlp.{key.removeprefix('actor.')}"] = value
            elif key.startswith("actor_obs_normalizer."):
                actor_state[
                    f"obs_normalizer.{key.removeprefix('actor_obs_normalizer.')}"
                ] = value
            elif key.startswith("critic."):
                critic_state[f"mlp.{key.removeprefix('critic.')}"] = value
            elif key.startswith("critic_obs_normalizer."):
                critic_state[
                    f"obs_normalizer.{key.removeprefix('critic_obs_normalizer.')}"
                ] = value

        missing_actor, unexpected_actor = runner.alg.actor.load_state_dict(
            actor_state, strict=False
        )
        missing_critic, unexpected_critic = runner.alg.critic.load_state_dict(
            critic_state, strict=False
        )
        if loaded.get("iter") is not None:
            runner.current_learning_iteration = loaded["iter"]

        logger.info(
            "[WARN] Loaded newer monolithic ActorCritic checkpoint into local "
            "split actor/critic RSL-RL model for inference."
        )
        if missing_actor or unexpected_actor or missing_critic or unexpected_critic:
            logger.info(
                f"[WARN] actor missing={missing_actor}, unexpected={unexpected_actor}"
            )
            logger.info(
                f"[WARN] critic missing={missing_critic}, unexpected={unexpected_critic}"
            )
        return
    except RuntimeError as exc:
        message = str(exc)
        std_schema_mismatch = (
            "distribution.std_param" in message
            and "distribution.log_std_param" in message
        )
        if not std_schema_mismatch:
            raise

        logger.info(
            "[WARN] Checkpoint uses different distribution std schema "
            "(std_param vs log_std_param). Retrying non-strict load for inference."
        )
        runner.load(checkpoint_path, strict=False, map_location=runner.device)


def _summarize(episodes: list[dict]) -> dict:
    """aggregate episode-level stats"""
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0, "success_rate": 0.0}

    successes = sum(1 for e in episodes if e["episode_success"])
    return {
        "num_episodes": n,
        "success_rate": successes / n,
        "mean_success_step_ratio": sum(e["success_step_ratio"] for e in episodes) / n,
        "mean_lin_vel_error": sum(e["mean_lin_vel_error_xy"] for e in episodes) / n,
        "mean_ang_vel_error": sum(e["mean_ang_vel_error_z"] for e in episodes) / n,
        "mean_alive_time_s": sum(e["alive_time_s"] for e in episodes) / n,
        "failure_rate": sum(1 for e in episodes if e["termination_reason"] == "failure")
        / n,
    }


def _run_eval_on_env(policy, env, task, num_episodes, device, passage):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    raw_env = env.unwrapped
    success_cfg = TASK_SUCCESS_CONFIGS[task.gym_id]
    tracker = EpisodeStatsTracker(
        num_envs=raw_env.num_envs,
        step_dt=raw_env.step_dt,
        cfg=success_cfg,
        device=device,
    )
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    episodes = []
    with torch.inference_mode():
        for _ in range(50_001):
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
            lin_err, ang_err = compute_tracking_errors(raw_env)
            terminated = raw_env.termination_manager.terminated
            truncated = raw_env.termination_manager.time_outs
            rows = tracker.update(
                step_success=compute_step_success_from_errors(
                    lin_err, ang_err, terminated, success_cfg
                ),
                lin_err_xy=lin_err,
                ang_err_z=ang_err,
                terminated=terminated,
                truncated=truncated,
                task_id=task.task_id,
                family_id=task.family_id,
            )
            if passage is not None:
                done_ids = (terminated | truncated).nonzero().flatten()
                rows = passage.tracker.finish(done_ids, rows)
            episodes.extend(rows)
            if len(episodes) >= num_episodes:
                break
    return episodes


def _evaluate(args, task, env_cfg, agent_cfg):
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    except ImportError:
        try:
            from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
        except ImportError:
            handle_deprecated_rsl_rl_cfg = None

    device = args.device or "cuda:0"
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args.eval_seed
    torch.manual_seed(args.eval_seed)

    captured = {}
    stairs = task.gym_id == STAIRS
    if stairs:
        capture_geometry(env_cfg, "B2", captured)
    env = gym.make(task.gym_id, cfg=env_cfg)
    passage = None
    try:
        if task.gym_id == ROUGH:
            install_rough_action_bound(env.unwrapped)
        wrapped = RslRlVecEnvWrapper(env)
        installed_version = metadata.version("rsl-rl-lib")
        _align_agent_cfg_to_checkpoint(agent_cfg, args.checkpoint)
        if handle_deprecated_rsl_rl_cfg is not None:
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        _align_agent_cfg_to_checkpoint(agent_cfg, args.checkpoint)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        _load_runner_checkpoint_compat(runner, args.checkpoint)
        policy = runner.get_inference_policy(device=device)
        if stairs:
            passage = PassageRuntime(env.unwrapped, captured)
        episodes = _run_eval_on_env(
            policy, env, task, args.num_episodes, device, passage
        )
    finally:
        if passage is not None:
            passage.close()
        if not args.headless:
            env.close()
    summary = _summarize(episodes)
    if stairs and episodes:
        summary.update(summarize_passage(episodes))
    summary.update(train_task=args.report_train_task, eval_task=task.gym_id)
    directory = Path(args.output_dir) / args.report_train_task
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{task.name}.json").write_text(
        json.dumps({"summary": summary, "episodes": episodes}, indent=2)
    )
    (directory / "summary.json").write_text(json.dumps({task.name: summary}, indent=2))


def main() -> None:
    """Evaluate one saved policy with one simulator instance."""
    from isaaclab.app import AppLauncher

    tasks = {task.gym_id: task for task in TASKS.values()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tasks)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_train_task", required=True)
    parser.add_argument("--eval_task", required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    task = tasks[args.task]
    if args.eval_task != task.name:
        parser.error(f"--eval_task must be {task.name} for {args.task}")
    sys.argv = [sys.argv[0], *hydra_args]
    simulation_app = AppLauncher(args).app
    exit_code = 0
    try:
        from isaaclab_tasks.utils.hydra import hydra_task_config

        register_go2_tasks()

        @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
        def evaluate(env_cfg, agent_cfg):
            _evaluate(args, task, env_cfg, agent_cfg)

        evaluate()
    except KeyboardInterrupt:
        exit_code = 130
        if not args.headless:
            raise
    except Exception:
        exit_code = 1
        if not args.headless:
            raise
        logger.exception("Go2 evaluation failed")
    finally:
        if not args.headless:
            simulation_app.close()
    if args.headless:
        # The headless Kit instance can hang during interpreter finalization.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
