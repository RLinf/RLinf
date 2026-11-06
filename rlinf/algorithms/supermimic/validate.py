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

"""SuperMimic checks on top of :func:`rlinf.config.validate_embodied_cfg`."""

from __future__ import annotations

import math

from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM

# Policy settings the G1 envs build observations, goals and critic inputs from,
# and the compile switch they share with the policy.
ENV_MODEL_KEYS = (
    "goal_encoding",
    "goal_anchor",
    "obs_dim",
    "velocity_tracking",
    "critic_features",
    "input_norm",
    "torch_compile",
)
# Algorithm settings the G1 envs score and terminate with.
ENV_ALGORITHM_KEYS = ("reward_anchor", "termination_body", "gamma")

_CHOICES = {
    "goal_encoding": ("paper", "goal730"),
    "critic_features": ("root4", "body_error"),
    "input_norm": ("running", "none"),
    "wm_state_loss": ("paper", "scaled", "running"),
    "reward_anchor": ("pelvis", "torso"),
    "termination_body": ("pelvis", "torso"),
    "residual_frame": ("body", "heading"),
    "termination_head": ("learned", "analytic"),
    "motion_format": ("lafan1", "bones_seed"),
    "com_sampling": ("per_env", "per_reset"),
}


def _check_choice(node: DictConfig, name: str, prefix: str) -> None:
    value = node.get(name, None)
    if value not in _CHOICES[name]:
        raise ValueError(
            f"{prefix}.{name} must be one of {list(_CHOICES[name])}, got {value!r}"
        )


def _check_number(
    node: DictConfig,
    name: str,
    prefix: str,
    kind: type = float,
    low: float | None = None,
    high: float | None = None,
    open_low: bool = False,
    open_high: bool = False,
) -> None:
    value = node.get(name, None)
    allowed = (int,) if kind is int else (int, float)
    if isinstance(value, bool) or not isinstance(value, allowed):
        kind_name = "an integer" if kind is int else "a number"
        raise ValueError(f"{prefix}.{name} must be {kind_name}, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{prefix}.{name} must be finite, got {value!r}")
    below = low is not None and (value <= low if open_low else value < low)
    above = high is not None and (value >= high if open_high else value > high)
    if below or above:
        left = "(" if open_low else "["
        right = ")" if open_high else "]"
        lo = "-inf" if low is None else low
        hi = "inf" if high is None else high
        raise ValueError(
            f"{prefix}.{name} must be in {left}{lo}, {hi}{right}, got {value!r}"
        )


def _check_model(model: DictConfig, prefix: str) -> None:
    for name in ("goal_encoding", "critic_features", "input_norm", "wm_state_loss"):
        _check_choice(model, name, prefix)
    for name in ("goal_anchor", "velocity_tracking", "torch_compile"):
        value = model.get(name, None)
        if not isinstance(value, bool):
            raise ValueError(f"{prefix}.{name} must be a bool, got {value!r}")
    if int(model.action_dim) != ACTION_DIM:
        raise ValueError(f"{prefix}.action_dim must be {ACTION_DIM}")
    if int(model.num_action_chunks) != 1:
        raise ValueError(f"{prefix}.num_action_chunks must be 1")
    if model.obs_dim not in (93, 99):
        raise ValueError(f"{prefix}.obs_dim must be 93 or 99, got {model.obs_dim!r}")
    if model.goal_anchor and (model.goal_encoding != "paper" or model.obs_dim != 93):
        raise ValueError(
            f"{prefix}.goal_anchor needs goal_encoding=paper and obs_dim=93"
        )
    if model.velocity_tracking and (
        model.goal_encoding != "paper" or model.obs_dim != 93 or model.goal_anchor
    ):
        raise ValueError(
            f"{prefix}.velocity_tracking needs goal_encoding=paper, obs_dim=93 "
            "and no goal_anchor"
        )
    if model.get("termination_bias_init", None) is not None:
        _check_number(model, "termination_bias_init", prefix)


def _sync_env(env: DictConfig, model: DictConfig, algorithm: DictConfig, prefix: str):
    """Copy the policy and scoring settings into one env section."""
    sources = [(name, model.get(name)) for name in ENV_MODEL_KEYS]
    sources += [(name, algorithm.get(name)) for name in ENV_ALGORITHM_KEYS]
    with open_dict(env):
        for name, value in sources:
            if env.get(name, None) is not None and env.get(name) != value:
                raise ValueError(
                    f"{prefix}.{name}={env.get(name)!r} differs from the policy "
                    f"or algorithm value {value!r}; set it in one place"
                )
            env[name] = value


def _check_env(env: DictConfig, prefix: str, needs_motion: bool) -> None:
    for name in ("motion_format", "com_sampling"):
        _check_choice(env, name, prefix)
    if needs_motion and not str(env.get("motion_path", "") or "").strip():
        raise ValueError(
            f"{prefix}.motion_path is empty; set it or SUPERMIMIC_MOTION_PATH"
        )
    num_clips = env.get("num_clips", None)
    if num_clips is not None:
        _check_number(env, "num_clips", prefix, int, low=1)
    _check_number(env, "action_noise_std", prefix, low=0.0)
    _check_number(env, "max_episode_seconds", prefix, low=0.0, open_low=True)
    _check_number(env, "G_max", prefix, low=0.0, open_low=True)
    if env.get("auto_reset", False):
        raise ValueError(f"{prefix}.auto_reset must be false; the env resets rows")


def _check_training(cfg: DictConfig) -> None:
    alg = cfg.algorithm
    if int(alg.k) != 8:
        raise ValueError(f"algorithm.k must be 8, got {alg.k}")
    if int(alg.H) != 32:
        raise ValueError(f"algorithm.H must be 32, got {alg.H}")
    for name in ("residual_frame", "termination_head"):
        _check_choice(alg, name, "algorithm")
    _check_number(alg, "updates_per_env_step", "algorithm", low=0.0, open_low=True)
    _check_number(
        alg, "gamma", "algorithm", low=0.0, high=1.0, open_low=True, open_high=True
    )
    _check_number(alg, "lam", "algorithm", low=0.0, high=1.0)
    _check_number(alg, "bce_weight", "algorithm", low=0.0)
    for name in ("wm_batches", "wm_window_batch", "imagine_batch", "critic_epochs"):
        _check_number(alg, name, "algorithm", int, low=1)
    _check_number(alg, "critic_minibatches", "algorithm", int, low=1)
    for name in ("warmup_sim_steps", "wm_only_updates", "max_nonfinite_skips"):
        _check_number(alg, name, "algorithm", int, low=0)
    _check_number(alg, "replay_length", "algorithm", int, low=int(alg.k))

    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    placement = HybridComponentPlacement(cfg, Cluster())
    sizes = {
        name: placement.get_world_size(name) for name in ("actor", "env", "rollout")
    }
    if len(set(sizes.values())) != 1:
        raise ValueError(
            "SuperMimic pairs the env, rollout and actor worker of each rank; "
            f"their world sizes must match, got {sizes}"
        )
    world = sizes["actor"]
    for name in ("wm_window_batch", "imagine_batch"):
        if int(alg[name]) % world:
            raise ValueError(
                f"algorithm.{name}={alg[name]} is global and must be divisible by "
                f"the {world} actor ranks"
            )
    if cfg.actor.model.velocity_tracking:
        per_rank_envs = int(cfg.env.train.total_num_envs) // world
        if per_rank_envs % 2 or (int(alg.imagine_batch) // world) % 2:
            raise ValueError(
                "velocity tracking splits each rank's envs and imagination batch "
                "in half; env.train.total_num_envs and algorithm.imagine_batch "
                f"must be divisible by 2 * {world}"
            )


def validate_supermimic_cfg(cfg: DictConfig, only_eval: bool = False) -> DictConfig:
    """Check SuperMimic settings and share the policy settings with the envs.

    Args:
        cfg: The run config, after the generic embodied checks.
        only_eval: Whether the run only evaluates ``rollout.model``.

    Returns:
        ``cfg`` with the policy and scoring settings copied into ``env.train``
        and ``env.eval``.
    """
    model = cfg.rollout.model if only_eval else cfg.actor.model
    prefix = "rollout.model" if only_eval else "actor.model"
    _check_model(model, prefix)
    for name in ("reward_anchor", "termination_body"):
        _check_choice(cfg.algorithm, name, "algorithm")
    if cfg.rollout.pipeline_stage_num != 1:
        raise ValueError("SuperMimic needs rollout.pipeline_stage_num=1")
    if not only_eval:
        _check_training(cfg)
        _sync_env(cfg.env.train, model, cfg.algorithm, "env.train")
        _check_env(cfg.env.train, "env.train", needs_motion=True)
    if only_eval or cfg.runner.get("val_check_interval", -1) > 0:
        tasks = list(cfg.env.eval.get("eval_tasks", None) or [])
        if not tasks:
            raise ValueError("env.eval.eval_tasks must name the evaluation tasks")
        _sync_env(cfg.env.eval, model, cfg.algorithm, "env.eval")
        _check_env(cfg.env.eval, "env.eval", needs_motion=tasks != ["push_recovery"])
    return cfg
