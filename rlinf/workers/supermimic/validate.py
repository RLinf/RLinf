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

"""SuperMimic config checks. Importable without Ray."""

from __future__ import annotations

import math
import re

from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.envs.sim.g1_mimic.constants import critic_input_dim

OBS_FRAME_DIM = 93
GOAL_FRAME_DIM = 68
ANCHOR_DIM = 9


def _check_number(
    node: DictConfig,
    name: str,
    kind: type,
    low: float | None = None,
    high: float | None = None,
    low_open: bool = False,
    high_open: bool = False,
    prefix: str = "algorithm",
) -> None:
    """Reject a non-numeric or out-of-range ``<prefix>.<name>`` before Ray starts."""
    value = node.get(name, None)
    allowed = (int,) if kind is int else (int, float)
    if isinstance(value, bool) or not isinstance(value, allowed):
        raise AssertionError(
            f"{prefix}.{name} must be {'an integer' if kind is int else 'a number'}, "
            f"got {value!r}"
        )
    if not math.isfinite(value):
        raise AssertionError(f"{prefix}.{name} must be finite, got {value!r}")
    too_low = low is not None and (value <= low if low_open else value < low)
    too_high = high is not None and (value >= high if high_open else value > high)
    if too_low or too_high:
        left = "(" if low_open else "["
        right = ")" if high_open else "]"
        bounds = f"{left}{'-inf' if low is None else low}, {'inf' if high is None else high}{right}"
        raise AssertionError(f"{prefix}.{name} must be in {bounds}, got {value!r}")


def validate_supermimic_eval_cfg(eval_cfg: DictConfig) -> None:
    """Check the ``eval`` section of a SuperMimic eval config before any work.

    Args:
        eval_cfg: The ``eval`` node of an ``evaluations/supermimic`` config.
    """
    velocity = str(eval_cfg.get("velocity", "none"))
    if velocity not in {"none", "smoke", "g2"}:
        raise ValueError(f"eval.velocity must be none, smoke or g2, got {velocity!r}")
    _check_number(eval_cfg, "seed", int, prefix="eval")
    _check_number(
        eval_cfg, "velocity_smoke_seconds", float, low=0.0, low_open=True, prefix="eval"
    )
    _check_number(eval_cfg, "velocity_smoke_start_clip", int, low=0, prefix="eval")
    _check_number(eval_cfg, "g2_trials", int, low=1, prefix="eval")
    _check_number(eval_cfg, "g2_batch", int, low=1, prefix="eval")
    _check_number(eval_cfg, "g2_seconds", float, low=0.0, low_open=True, prefix="eval")
    _check_number(
        eval_cfg,
        "g2_steady_seconds",
        float,
        low=0.0,
        high=float(eval_cfg.g2_seconds),
        low_open=True,
        prefix="eval",
    )


def placement_world_size(spec: int | str) -> int:
    """Rank count of a plain GPU list: ``0``, ``"3"``, ``"0-3"``, ``"0-1,4-5"``.

    ``all`` and node-group forms depend on the cluster; the training entry
    passes the count from ``HybridComponentPlacement`` instead.
    """
    if isinstance(spec, int) and not isinstance(spec, bool):
        return 1
    text = str(spec).strip()
    count = 0
    for part in text.split(","):
        part = part.strip()
        if part.isdigit():
            count += 1
            continue
        low, sep, high = part.partition("-")
        if sep and low.strip().isdigit() and high.strip().isdigit():
            if int(high) < int(low):
                raise AssertionError(f"bad GPU range {part!r} in supermimic placement")
            count += int(high) - int(low) + 1
            continue
        raise AssertionError(
            f"cannot count ranks in cluster.component_placement.supermimic={spec!r} "
            "without the cluster; pass world_size"
        )
    return count


def validate_supermimic_cfg(
    cfg: DictConfig, world_size: int | None = None
) -> DictConfig:
    """Validate SuperMimic YAML and fill derived tensor sizes.

    ``world_size`` is the number of SuperMimic ranks. Batch sizes in the
    YAML are global and are split evenly over ranks.
    """
    placement = cfg.cluster.get("component_placement", None)
    if placement is None:
        raise AssertionError(
            "supermimic requires cluster.component_placement: {supermimic: 0}"
        )
    keys = list(placement.keys())
    if keys != ["supermimic"]:
        raise AssertionError(
            f"supermimic placement must contain only 'supermimic', got {keys}"
        )
    if world_size is None:
        world_size = placement_world_size(placement.supermimic)
    world_size = int(world_size)
    if world_size < 1:
        raise AssertionError(f"supermimic world size must be >= 1, got {world_size}")

    alg = cfg.algorithm
    env = cfg.env
    if int(alg.k) != 8:
        raise AssertionError(f"algorithm.k must be 8, got {alg.k}")
    if int(alg.H) != 32:
        raise AssertionError(f"algorithm.H must be 32, got {alg.H}")
    # Goal builders emit 10 frames. 680 = 68 * 10 and 730 = 73 * 10.
    # History is 10 observation frames. Neither width is configurable yet.
    if int(alg.goal_frames) != 10:
        raise AssertionError(
            "goal builders are fixed at 10 frames; "
            f"algorithm.goal_frames={alg.goal_frames}"
        )
    if int(alg.history_len) != 10:
        raise AssertionError(
            f"observation history is 10 frames; algorithm.history_len={alg.history_len}"
        )
    if float(alg.updates_per_env_step) <= 0:
        raise AssertionError(
            "algorithm.updates_per_env_step must be > 0, "
            f"got {alg.updates_per_env_step}"
        )
    if int(env.num_envs) <= 0:
        raise AssertionError(f"env.num_envs must be positive, got {env.num_envs}")
    if int(alg.wm_window_batch) <= 0 or int(alg.imagine_batch) <= 0:
        raise AssertionError(
            "algorithm.wm_window_batch and algorithm.imagine_batch must be positive, "
            f"got {alg.wm_window_batch} and {alg.imagine_batch}"
        )
    for name, value in (
        ("env.num_envs", env.num_envs),
        ("algorithm.wm_window_batch", alg.wm_window_batch),
        ("algorithm.imagine_batch", alg.imagine_batch),
    ):
        if int(value) % world_size:
            raise AssertionError(
                f"{name}={value} is a global size and must be divisible by "
                f"the SuperMimic world size {world_size}"
            )
    if int(alg.total_updates) <= 0:
        raise AssertionError(
            f"algorithm.total_updates must be positive, got {alg.total_updates}"
        )
    if int(env.replay_length) <= 0:
        raise AssertionError(
            f"env.replay_length must be positive, got {env.replay_length}"
        )
    if int(alg.action_dim) != 29:
        raise AssertionError(f"algorithm.action_dim must be 29, got {alg.action_dim}")
    if not isinstance(alg.goal_anchor, bool):
        raise AssertionError("algorithm.goal_anchor must be a bool")
    encoding = str(alg.get("goal_encoding", "paper"))
    obs_dim = int(alg.obs_dim)
    if encoding == "paper":
        if obs_dim not in {OBS_FRAME_DIM, 99}:
            raise AssertionError(f"algorithm.obs_dim must be 93 or 99, got {obs_dim}")
        if obs_dim == 99 and bool(alg.goal_anchor):
            raise AssertionError(
                "99-d observation does not combine with the anchor goal"
            )
        if int(alg.goal_frame_dim) != GOAL_FRAME_DIM:
            raise AssertionError(
                f"algorithm.goal_frame_dim must be 68, got {alg.goal_frame_dim}"
            )
    elif encoding == "goal730":
        if bool(alg.goal_anchor):
            raise AssertionError("730-d goal does not combine with the anchor goal")
        if obs_dim not in {OBS_FRAME_DIM, 99}:
            raise AssertionError(
                f"730-d goal uses a 93-d or 99-d observation, got {obs_dim}"
            )
        if int(alg.goal_frame_dim) != 73:
            raise AssertionError(f"goal730 frame must be 73, got {alg.goal_frame_dim}")
    else:
        raise AssertionError(f"unknown algorithm.goal_encoding {encoding!r}")
    if bool(alg.get("velocity_tracking", False)):
        if encoding != "paper" or obs_dim != OBS_FRAME_DIM or bool(alg.goal_anchor):
            raise AssertionError(
                "velocity tracking uses the 93-d observation and the 680-d goal"
            )
        if (int(env.num_envs) // world_size) % 2 or (
            int(alg.imagine_batch) // world_size
        ) % 2:
            raise AssertionError(
                "velocity tracking splits each rank's envs and imagination in half; "
                "env.num_envs and algorithm.imagine_batch must be divisible by "
                f"2 * world_size = {2 * world_size}"
            )
    motion_format = str(env.get("motion_format", "lafan1"))
    if motion_format not in {"lafan1", "bones_seed"}:
        raise AssertionError(f"unknown env.motion_format {motion_format!r}")
    termination_head = str(alg.get("termination_head", "learned"))
    if termination_head not in {"learned", "analytic"}:
        raise AssertionError(
            f"algorithm.termination_head must be learned or analytic, got {termination_head!r}"
        )
    # Omitted keys follow zita-ch. The paper yaml writes the other values.
    wm_state_loss = str(alg.get("wm_state_loss", "running"))
    if wm_state_loss not in {"paper", "scaled", "running"}:
        raise AssertionError(
            "algorithm.wm_state_loss must be paper, scaled or running, "
            f"got {wm_state_loss!r}"
        )
    reward_anchor = str(alg.get("reward_anchor", "pelvis"))
    if reward_anchor not in {"pelvis", "torso"}:
        raise AssertionError(
            f"algorithm.reward_anchor must be pelvis or torso, got {reward_anchor!r}"
        )
    input_norm = str(alg.get("input_norm", "running"))
    if input_norm not in {"running", "none"}:
        raise AssertionError(
            f"algorithm.input_norm must be running or none, got {input_norm!r}"
        )
    residual_frame = str(alg.get("residual_frame", "body"))
    if residual_frame not in {"body", "heading"}:
        raise AssertionError(
            f"algorithm.residual_frame must be body or heading, got {residual_frame!r}"
        )
    critic_features = str(alg.get("critic_features", "root4"))
    if critic_features not in {"root4", "zita"}:
        raise AssertionError(
            f"algorithm.critic_features must be root4 or zita, got {critic_features!r}"
        )
    matmul = str(alg.get("matmul", "tf32"))
    if matmul not in {"tf32", "bf16"}:
        raise AssertionError(f"algorithm.matmul must be tf32 or bf16, got {matmul!r}")
    termination_body = str(alg.get("termination_body", "pelvis"))
    if termination_body not in {"pelvis", "torso"}:
        raise AssertionError(
            "algorithm.termination_body must be pelvis or torso, "
            f"got {termination_body!r}"
        )
    _check_number(alg, "lr", float, low=0.0, low_open=True)
    _check_number(alg, "gamma", float, low=0.0, high=1.0, low_open=True, high_open=True)
    _check_number(alg, "lam", float, low=0.0, high=1.0)
    _check_number(alg, "action_noise_std", float, low=0.0)
    _check_number(alg, "max_episode_seconds", float, low=0.0, low_open=True)
    _check_number(alg, "G_max", float, low=0.0, low_open=True)
    for name in ("wm_batches", "critic_epochs", "critic_minibatches"):
        _check_number(alg, name, int, low=1)
    for name in (
        "param_check_interval",
        "max_nonfinite_skips",
        "warmup_sim_steps",
        "wm_only_updates",
    ):
        _check_number(alg, name, int, low=0)
    for name in ("seed", "termination_bias_init"):
        if alg.get(name, None) is not None:
            _check_number(alg, name, int if name == "seed" else float)
    device = str(env.get("device", "cuda"))
    if device != "cpu" and not re.fullmatch(r"cuda(:\d+)?", device):
        raise AssertionError(
            f"env.device must be cpu, cuda or cuda:<index>, got {device!r}"
        )
    num_clips = env.get("num_clips", None)
    if num_clips is not None and (isinstance(num_clips, bool) or int(num_clips) <= 0):
        raise AssertionError(
            f"env.num_clips must be a positive count or null (every clip), got {num_clips!r}"
        )
    if int(env.replay_length) < int(alg.k):
        raise AssertionError("env.replay_length must be at least algorithm.k")
    val_interval = int(cfg.runner.get("val_check_interval", -1))
    save_interval = int(cfg.runner.get("save_interval", -1))
    # check_progress asserts this after the first update; fail before warmup.
    if val_interval > 0 and save_interval > 0 and save_interval % val_interval:
        raise AssertionError(
            f"runner.save_interval={save_interval} must be a multiple of "
            f"runner.val_check_interval={val_interval}"
        )
    from rlinf.envs.sim.g1_mimic.env import env_backends

    backend = str(env.get("backend", "mjlab"))
    if backend not in env_backends():
        raise AssertionError(
            f"env.backend={backend!r} is not available; SuperMimic trains on mjlab "
            f"(MuJoCo Warp). Registered: {sorted(env_backends())}"
        )
    if not str(env.get("motion_path", "") or "").strip():
        raise AssertionError(f"env.backend={backend} requires env.motion_path")

    history_dim = int(alg.history_len) * int(alg.obs_dim)
    if encoding == "goal730":
        goal_ref_dim = 730
        goal_dim = 730
    else:
        goal_ref_dim = int(alg.goal_frames) * int(alg.goal_frame_dim)
        goal_dim = goal_ref_dim + (ANCHOR_DIM if bool(alg.goal_anchor) else 0)
    critic_in_dim = critic_input_dim(goal_dim, critic_features)

    with open_dict(cfg):
        cfg.algorithm.world_size = world_size
        cfg.algorithm.obs_history_dim = history_dim
        cfg.algorithm.goal_ref_dim = goal_ref_dim
        cfg.algorithm.goal_dim = goal_dim
        cfg.algorithm.encoder_in_dim = goal_dim
        cfg.algorithm.critic_in_dim = critic_in_dim
        cfg.algorithm.wm_state_loss = wm_state_loss
        cfg.algorithm.reward_anchor = reward_anchor
        cfg.algorithm.input_norm = input_norm
        cfg.algorithm.residual_frame = residual_frame
    return cfg
