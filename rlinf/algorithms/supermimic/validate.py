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
from typing import TYPE_CHECKING

import torch
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.algorithms.supermimic.representations import REPRESENTATIONS
from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, DT
from rlinf.utils.precision import torch_dtype_from_precision

if TYPE_CHECKING:
    from rlinf.utils.placement import HybridComponentPlacement

# Policy settings the G1 envs build observations from, and the compile switch
# they share with the policy.
ENV_MODEL_KEYS = ("representation", "torch_compile")

_CHOICES = {
    "representation": tuple(REPRESENTATIONS),
    "motion_format": ("lafan1", "bones_seed"),
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
    _check_choice(model, "representation", prefix)
    if not isinstance(model.get("torch_compile", None), bool):
        raise ValueError(
            f"{prefix}.torch_compile must be a bool, got {model.get('torch_compile')!r}"
        )
    if int(model.action_dim) != ACTION_DIM:
        raise ValueError(f"{prefix}.action_dim must be {ACTION_DIM}")
    if int(model.num_action_chunks) != 1:
        raise ValueError(f"{prefix}.num_action_chunks must be 1")


def _is_float32(precision) -> bool:
    return torch_dtype_from_precision(precision) in (None, torch.float32)


def _check_weight_precision(cfg: DictConfig, only_eval: bool) -> None:
    """The policy builder makes float32 weights whatever ``precision`` asks for."""
    nodes = [] if only_eval else [("actor.model", cfg.actor.model)]
    nodes.append(("rollout.model", cfg.rollout.model))
    for prefix, node in nodes:
        precision = node.get("precision", None)
        if not _is_float32(precision):
            raise ValueError(
                f"{prefix}.precision must be 32, got {precision!r}; SuperMimic "
                "keeps float32 weights and casts inside the networks with "
                "amp_autocast"
            )


def _check_actor_backend(actor: DictConfig) -> None:
    """The actor calls FSDP2 module methods and runs its own optimizer steps."""
    backend = actor.get("training_backend", None)
    fsdp = actor.fsdp_config
    strategy = fsdp.get("strategy", None)
    if backend != "fsdp" or strategy != "fsdp2":
        raise ValueError(
            "SuperMimic training needs actor.training_backend=fsdp and "
            f"actor.fsdp_config.strategy=fsdp2, got {backend!r} and {strategy!r}"
        )
    mixed = fsdp.get("mixed_precision", None) or {}
    for name in ("param_dtype", "reduce_dtype", "buffer_dtype"):
        if not _is_float32(mixed.get(name, None)):
            raise ValueError(
                f"actor.fsdp_config.mixed_precision.{name} must be null or fp32, "
                f"got {mixed.get(name)!r}; SuperMimic keeps float32 weights and "
                "casts inside the networks with fsdp_config.amp_autocast"
            )
    if fsdp.get("cpu_offload", False):
        raise ValueError(
            "SuperMimic training needs actor.fsdp_config.cpu_offload=False, got True"
        )
    if (fsdp.get("grad_scaler", None) or {}).get("enabled", False):
        raise ValueError(
            "actor.fsdp_config.grad_scaler.enabled must be False, got True; the "
            "SuperMimic optimizer steps do not use a grad scaler"
        )


def _sync_env(env: DictConfig, model: DictConfig, prefix: str):
    """Copy the policy settings into one env section."""
    with open_dict(env):
        for name in ENV_MODEL_KEYS:
            value = model.get(name)
            if env.get(name, None) is not None and env.get(name) != value:
                raise ValueError(
                    f"{prefix}.{name}={env.get(name)!r} differs from the policy "
                    f"value {value!r}; set it in one place"
                )
            env[name] = value


def _check_env(env: DictConfig, prefix: str, needs_motion: bool) -> None:
    _check_choice(env, "motion_format", prefix)
    if needs_motion and not str(env.get("motion_path", "") or "").strip():
        raise ValueError(
            f"{prefix}.motion_path is empty; set it or SUPERMIMIC_MOTION_PATH"
        )
    num_clips = env.get("num_clips", None)
    if num_clips is not None:
        _check_number(env, "num_clips", prefix, int, low=1)
    if env.get("auto_reset", False):
        raise ValueError(f"{prefix}.auto_reset must be false; the env resets rows")


def _check_training(cfg: DictConfig) -> None:
    alg = cfg.algorithm
    _check_number(alg, "k", "algorithm", int, low=2)
    _check_number(alg, "H", "algorithm", int, low=1)
    _check_number(
        alg, "gamma", "algorithm", low=0.0, high=1.0, open_low=True, open_high=True
    )
    _check_number(alg, "lam", "algorithm", low=0.0, high=1.0)
    _check_number(alg, "replay_ratio", "algorithm", low=0.0, open_low=True)
    for name in (
        "wm_batches",
        "wm_window_batch",
        "imagine_batch",
        "critic_epochs",
        "critic_minibatches",
    ):
        _check_number(alg, name, "algorithm", int, low=1)
    _check_number(alg, "max_nonfinite_skips", "algorithm", int, low=0)
    _check_number(alg, "replay_length", "algorithm", int, low=int(alg.k))
    env = cfg.env.train
    _check_number(env, "action_noise_std", "env.train", low=0.0)
    _check_number(env, "max_episode_seconds", "env.train", low=0.0, open_low=True)
    episode_steps = round(float(env.max_episode_seconds) / DT)
    if episode_steps < int(alg.k):
        raise ValueError(
            f"env.train.max_episode_seconds={env.max_episode_seconds} allows "
            f"{episode_steps} control steps per episode; world-model windows "
            f"need at least algorithm.k={alg.k}"
        )


def check_supermimic_placement(
    cfg: DictConfig, placement: HybridComponentPlacement
) -> None:
    """Check the worker counts SuperMimic training pairs and divides batches by.

    Args:
        cfg: The run config, after :func:`validate_supermimic_cfg`.
        placement: The component placement the run starts its workers with.
    """
    alg = cfg.algorithm
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


def validate_supermimic_cfg(cfg: DictConfig, only_eval: bool = False) -> DictConfig:
    """Check SuperMimic settings and share the policy settings with the envs.

    The worker counts are checked separately, by
    :func:`check_supermimic_placement`.

    Args:
        cfg: The run config, after the generic embodied checks.
        only_eval: Whether the run only evaluates ``rollout.model``.

    Returns:
        ``cfg`` with the policy settings copied into ``env.train`` and
        ``env.eval``.
    """
    model = cfg.rollout.model if only_eval else cfg.actor.model
    prefix = "rollout.model" if only_eval else "actor.model"
    _check_model(model, prefix)
    _check_weight_precision(cfg, only_eval)
    if cfg.rollout.pipeline_stage_num != 1:
        raise ValueError("SuperMimic needs rollout.pipeline_stage_num=1")
    if not only_eval:
        # serve, ingest and run_updates assume the networks stay on the device.
        for group in ("actor", "rollout"):
            if cfg[group].get("enable_offload", False):
                raise ValueError(
                    f"SuperMimic training needs {group}.enable_offload=False, got True"
                )
        _check_actor_backend(cfg.actor)
        _sync_env(cfg.env.train, model, "env.train")
        _check_env(cfg.env.train, "env.train", needs_motion=True)
        _check_training(cfg)
    if only_eval or cfg.runner.get("val_check_interval", -1) > 0:
        tasks = list(cfg.env.eval.get("eval_tasks", None) or [])
        if not tasks:
            raise ValueError("env.eval.eval_tasks must name the evaluation tasks")
        _sync_env(cfg.env.eval, model, "env.eval")
        _check_env(cfg.env.eval, "env.eval", needs_motion=tasks != ["push_recovery"])
    return cfg
