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

from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

OBS_FRAME_DIM = 93
GOAL_FRAME_DIM = 68
ANCHOR_DIM = 9
CRITIC_EXTRA_DIM = 71 + 33 + 4


def validate_supermimic_cfg(cfg: DictConfig) -> DictConfig:
    """Validate SuperMimic YAML and fill derived tensor sizes."""
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
    gpu = placement.supermimic
    if str(gpu).strip() in {"all", "0-1", "0-7"} or (
        isinstance(gpu, str) and "-" in gpu and gpu != "0-0"
    ):
        raise AssertionError(
            "supermimic world size must be 1; "
            f"cluster.component_placement.supermimic={gpu!r} would start more ranks"
        )
    if not (
        gpu == 0 or gpu == "0" or gpu == "0-0" or (isinstance(gpu, int) and gpu == 0)
    ):
        raise AssertionError(
            "first-version SuperMimic uses GPU 0 only, "
            f"got cluster.component_placement.supermimic={gpu!r}"
        )
    if int(cfg.cluster.get("num_nodes", 1)) != 1:
        raise AssertionError("supermimic supports a single node only")

    alg = cfg.algorithm
    env = cfg.env
    if int(alg.k) != 8:
        raise AssertionError(f"algorithm.k must be 8, got {alg.k}")
    if int(alg.H) != 32:
        raise AssertionError(f"algorithm.H must be 32, got {alg.H}")
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
            raise AssertionError("99-d observation does not combine with the anchor goal")
        if int(alg.goal_frame_dim) != GOAL_FRAME_DIM:
            raise AssertionError(
                f"algorithm.goal_frame_dim must be 68, got {alg.goal_frame_dim}"
            )
    elif encoding == "goal730":
        if obs_dim != OBS_FRAME_DIM or bool(alg.goal_anchor):
            raise AssertionError("730-d goal keeps the 93-d observation and no anchor")
        if int(alg.goal_frame_dim) != 73:
            raise AssertionError(
                f"goal730 frame must be 73, got {alg.goal_frame_dim}"
            )
    else:
        raise AssertionError(f"unknown algorithm.goal_encoding {encoding!r}")
    if bool(alg.get("velocity_tracking", False)):
        if encoding != "paper" or obs_dim != OBS_FRAME_DIM or bool(alg.goal_anchor):
            raise AssertionError(
                "velocity tracking uses the 93-d observation and the 680-d goal"
            )
        if int(env.num_envs) % 2 or int(alg.imagine_batch) % 2:
            raise AssertionError("velocity tracking splits envs and imagination in half")
    motion_format = str(env.get("motion_format", "lafan1"))
    if motion_format not in {"lafan1", "bones_seed"}:
        raise AssertionError(f"unknown env.motion_format {motion_format!r}")
    if int(env.replay_length) < int(alg.k):
        raise AssertionError("env.replay_length must be at least algorithm.k")
    backend = str(env.get("backend", "mjlab"))
    if backend == "mjlab" and not str(env.get("motion_path", "")).strip():
        raise AssertionError("env.backend=mjlab requires env.motion_path")
    if backend not in {"mjlab", "analytic"}:
        raise AssertionError(f"unknown env.backend {backend!r}")

    history_dim = int(alg.history_len) * int(alg.obs_dim)
    if encoding == "goal730":
        goal_ref_dim = 730
        goal_dim = 730
    else:
        goal_ref_dim = int(alg.goal_frames) * int(alg.goal_frame_dim)
        goal_dim = goal_ref_dim + (ANCHOR_DIM if bool(alg.goal_anchor) else 0)
    critic_in_dim = goal_dim + CRITIC_EXTRA_DIM

    with open_dict(cfg):
        cfg.algorithm.obs_history_dim = history_dim
        cfg.algorithm.goal_ref_dim = goal_ref_dim
        cfg.algorithm.goal_dim = goal_dim
        cfg.algorithm.encoder_in_dim = goal_dim
        cfg.algorithm.critic_in_dim = critic_in_dim
    return cfg
