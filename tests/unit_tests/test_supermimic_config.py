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

"""Hydra composition and validation of the SuperMimic configs."""

from __future__ import annotations

import copy
import math
import os
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf, open_dict

from rlinf.workers.supermimic.validate import validate_supermimic_cfg

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


RUNNABLE = (
    "supermimic_lafan1",
    "supermimic_lafan1_anchor",
    "supermimic_smoke",
    "supermimic_lafan1_goal730",
    "supermimic_lafan1_obs99",
    "supermimic_lafan1_zita",
    "supermimic_velocity",
    "supermimic_bones_seed",
    "supermimic_bones_seed_2gpu",
)


def compose_cfg(name: str, overrides: list[str] | None = None):
    # Skip at run time, not at collection, so a per-file run never exits 5.
    hydra = pytest.importorskip("hydra")
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return hydra.compose(config_name=name, overrides=overrides or [])


@pytest.fixture(scope="module")
def composed():
    return {name: compose_cfg(name) for name in ("supermimic_base", *RUNNABLE)}


def _load(name: str):
    return OmegaConf.load(CONFIG_DIR / f"{name}.yaml")


def _compose(name: str):
    """Replay Hydra inheritance without importing Hydra (JobConf is broken on 3.11)."""
    if name == "supermimic_base":
        cfg = _load(name)
    elif name == "supermimic_lafan1":
        cfg = OmegaConf.merge(_load("supermimic_base"), _load(name))
    elif name in {"supermimic_lafan1_anchor", "supermimic_smoke"}:
        cfg = OmegaConf.merge(
            _load("supermimic_base"), _load("supermimic_lafan1"), _load(name)
        )
    else:
        raise KeyError(name)
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    return cfg


def test_each_config_has_its_own_log_path(composed):
    paths = {name: str(composed[name].runner.logger.log_path) for name in RUNNABLE}
    assert len(set(paths.values())) == len(RUNNABLE)
    for name, path in paths.items():
        experiment = str(composed[name].runner.logger.experiment_name)
        assert os.path.basename(path) == experiment
    explicit = compose_cfg("supermimic_smoke", ["runner.logger.log_path=/tmp/old_run"])
    assert explicit.runner.logger.log_path == "/tmp/old_run"


def test_hydra_resolves_paper_and_smoke_sizes(composed):
    lafan1 = copy.deepcopy(composed["supermimic_lafan1"])
    lafan1.env.motion_path = "/tmp/lafan1_g1"
    lafan1 = validate_supermimic_cfg(lafan1)
    assert lafan1.algorithm.k == 8 and lafan1.algorithm.H == 32
    assert lafan1.algorithm.goal_dim == 680
    assert lafan1.algorithm.obs_history_dim == 930
    assert lafan1.algorithm.critic_in_dim == 680 + 71 + 33 + 4
    assert lafan1.env.num_envs == 4096
    assert lafan1.algorithm.wm_window_batch == 2048
    assert lafan1.algorithm.imagine_batch == 4096
    assert lafan1.algorithm.total_updates == 320000
    assert lafan1.algorithm.warmup_sim_steps == 40
    assert lafan1.algorithm.wm_only_updates == 0
    assert lafan1.algorithm.wm_state_loss == "paper"
    assert lafan1.algorithm.reward_anchor == "torso"
    assert lafan1.algorithm.input_norm == "none"
    assert lafan1.algorithm.residual_frame == "heading"
    assert lafan1.algorithm.critic_features == "root4"
    assert lafan1.algorithm.matmul == "tf32"

    anchor = copy.deepcopy(composed["supermimic_lafan1_anchor"])
    anchor.env.motion_path = "/tmp/lafan1_g1"
    anchor = validate_supermimic_cfg(anchor)
    assert anchor.algorithm.goal_anchor is True
    assert anchor.algorithm.goal_dim == 689
    assert anchor.algorithm.obs_dim == 93

    smoke = copy.deepcopy(composed["supermimic_smoke"])
    smoke.env.motion_path = "/tmp/lafan1_g1"
    smoke = validate_supermimic_cfg(smoke)
    assert smoke.env.backend == "mjlab"
    assert smoke.env.num_envs == 8
    assert smoke.algorithm.imagine_batch == 8
    assert smoke.algorithm.wm_window_batch == 64
    assert smoke.algorithm.total_updates == 3
    assert smoke.algorithm.k == 8 and smoke.algorithm.H == 32
    assert smoke.algorithm.warmup_sim_steps == 40
    assert smoke.algorithm.goal_dim == 680

    goal730 = copy.deepcopy(composed["supermimic_lafan1_goal730"])
    goal730.env.motion_path = "/tmp/lafan1_g1"
    goal730 = validate_supermimic_cfg(goal730)
    assert goal730.algorithm.goal_dim == 730
    assert goal730.algorithm.obs_dim == 93
    assert goal730.algorithm.goal_anchor is False

    obs99 = copy.deepcopy(composed["supermimic_lafan1_obs99"])
    obs99.env.motion_path = "/tmp/lafan1_g1"
    obs99 = validate_supermimic_cfg(obs99)
    assert obs99.algorithm.obs_dim == 99
    assert obs99.algorithm.obs_history_dim == 990
    assert obs99.algorithm.goal_dim == 680

    zita = copy.deepcopy(composed["supermimic_lafan1_zita"])
    zita.env.motion_path = "/tmp/lafan1_g1"
    zita = validate_supermimic_cfg(zita)
    assert zita.algorithm.goal_encoding == "goal730"
    assert zita.algorithm.goal_dim == 730
    assert zita.algorithm.obs_dim == 99
    assert zita.algorithm.obs_history_dim == 990
    assert zita.algorithm.termination_head == "analytic"
    assert zita.algorithm.wm_only_updates == 500
    assert zita.algorithm.wm_state_loss == "running"
    assert zita.algorithm.reward_anchor == "pelvis"
    assert zita.algorithm.input_norm == "running"
    assert zita.algorithm.residual_frame == "body"
    assert zita.algorithm.critic_features == "zita"
    assert zita.algorithm.matmul == "bf16"
    assert zita.algorithm.critic_in_dim == 730 + 71 + 29 + 14 * 3 + 3 + 6 + 33
    assert zita.algorithm.warmup_sim_steps == 64
    assert zita.algorithm.total_updates == 16000
    assert zita.runner.save_interval == 2000
    anchored = copy.deepcopy(zita)
    anchored.algorithm.goal_anchor = True
    with pytest.raises(AssertionError, match="does not combine with the anchor"):
        validate_supermimic_cfg(anchored)

    velocity = copy.deepcopy(composed["supermimic_velocity"])
    velocity.env.motion_path = "/tmp/lafan1_g1"
    velocity = validate_supermimic_cfg(velocity)
    assert velocity.algorithm.velocity_tracking is True
    assert velocity.algorithm.goal_dim == 680
    assert velocity.env.num_envs % 2 == 0

    bones = copy.deepcopy(composed["supermimic_bones_seed"])
    bones.env.motion_path = "/tmp/bones_seed"
    bones = validate_supermimic_cfg(bones)
    assert bones.env.motion_format == "bones_seed"
    assert bones.algorithm.goal_dim == 680
    assert bones.algorithm.goal_frames == 10
    assert bones.algorithm.history_len == 10
    assert bones.algorithm.updates_per_env_step > 0

    omitted = copy.deepcopy(composed["supermimic_lafan1"])
    omitted.env.motion_path = "/tmp/lafan1_g1"
    with open_dict(omitted):
        for key in ("reward_anchor", "input_norm", "residual_frame", "wm_state_loss"):
            del omitted.algorithm[key]
    omitted = validate_supermimic_cfg(omitted)
    assert omitted.algorithm.reward_anchor == "pelvis"
    assert omitted.algorithm.input_norm == "running"
    assert omitted.algorithm.residual_frame == "body"
    assert omitted.algorithm.wm_state_loss == "running"


def test_paper_two_gpu_recipe_keeps_one_gpu_sizes_per_rank(composed):
    one = copy.deepcopy(composed["supermimic_bones_seed"])
    one.env.motion_path = "/tmp/bones_g1"
    one = validate_supermimic_cfg(one)
    two = copy.deepcopy(composed["supermimic_bones_seed_2gpu"])
    two.env.motion_path = "/tmp/bones_g1"
    two = validate_supermimic_cfg(two)
    assert two.algorithm.world_size == 2
    for key in ("wm_window_batch", "imagine_batch"):
        assert two.algorithm[key] // 2 == one.algorithm[key]
    assert two.env.num_envs // 2 == one.env.num_envs == 4096
    assert two.algorithm.lr == pytest.approx(2 * math.sqrt(2) * 1e-4, rel=1e-7)
    assert two.algorithm.updates_per_env_step == one.algorithm.updates_per_env_step
    assert two.algorithm.total_updates == one.algorithm.total_updates
    # Plain data parallelism keeps the one-GPU global batch and learning rate.
    plain = copy.deepcopy(composed["supermimic_bones_seed"])
    plain.env.motion_path = "/tmp/bones_g1"
    plain.cluster.component_placement.supermimic = "0-1"
    plain = validate_supermimic_cfg(plain)
    assert plain.algorithm.world_size == 2
    assert plain.env.num_envs == 4096 and plain.algorithm.imagine_batch == 4096
    assert plain.algorithm.lr == one.algorithm.lr


def test_bones_loads_every_clip_while_lafan1_keeps_its_forty(composed):
    assert composed["supermimic_bones_seed"].env.num_clips is None
    assert composed["supermimic_lafan1"].env.num_clips == 40
    assert composed["supermimic_smoke"].env.num_clips == 1
    bones = copy.deepcopy(composed["supermimic_bones_seed"])
    bones.env.motion_path = "/tmp/bones_g1"
    assert validate_supermimic_cfg(bones).env.num_clips is None


def test_illegal_overrides_fail_before_a_model_is_built(composed):
    cfg = copy.deepcopy(composed["supermimic_smoke"])
    cfg.env.motion_path = "/tmp/lafan1_g1"
    validate_supermimic_cfg(copy.deepcopy(cfg))

    def rejected(field, value, match):
        bad = copy.deepcopy(cfg)
        parent, name = field.split(".")
        setattr(getattr(bad, parent), name, value)
        with pytest.raises(AssertionError, match=match):
            validate_supermimic_cfg(bad)

    rejected("algorithm.goal_frames", 9, "10 frames")
    rejected("algorithm.history_len", 9, "10 frames")
    rejected("algorithm.updates_per_env_step", 0, "must be > 0")
    rejected("algorithm.total_updates", 0, "total_updates")
    rejected("algorithm.wm_window_batch", 0, "positive")
    rejected("algorithm.imagine_batch", 0, "positive")
    rejected("env.num_envs", 0, "num_envs")
    rejected("env.replay_length", 0, "replay_length")
    rejected("env.num_clips", 0, "num_clips")
    rejected("env.num_clips", True, "num_clips")
    rejected("algorithm.lr", "abc", "algorithm.lr must be a number")
    rejected("algorithm.lr", 0.0, r"algorithm.lr must be in \(0.0, inf\]")
    rejected("algorithm.lr", -1e-4, "algorithm.lr must be in")
    rejected("algorithm.gamma", 1.5, r"algorithm.gamma must be in \(0.0, 1.0\)")
    rejected("algorithm.lam", 1.5, "algorithm.lam must be in")
    rejected("algorithm.action_noise_std", -0.1, "action_noise_std must be in")
    rejected("algorithm.wm_batches", 0, "wm_batches must be in")
    rejected("algorithm.critic_epochs", 0, "critic_epochs must be in")
    rejected("algorithm.critic_minibatches", 0, "critic_minibatches must be in")
    rejected("algorithm.seed", "abc", "algorithm.seed must be an integer")
    rejected("algorithm.wm_only_updates", -1, "wm_only_updates must be in")
    rejected(
        "algorithm.wm_state_loss",
        "std",
        "wm_state_loss must be paper, scaled or running",
    )
    rejected("algorithm.reward_anchor", "head", "reward_anchor must be pelvis or torso")
    rejected("algorithm.input_norm", "batch", "input_norm must be running or none")
    rejected(
        "algorithm.residual_frame", "world", "residual_frame must be body or heading"
    )
    rejected(
        "algorithm.critic_features", "paper", "critic_features must be root4 or zita"
    )
    rejected("algorithm.matmul", "fp16", "matmul must be tf32 or bf16")
    rejected("algorithm.termination_bias_init", "abc", "termination_bias_init must be")
    rejected("algorithm.param_check_interval", "abc", "param_check_interval must be")
    rejected("algorithm.termination_body", "head", "pelvis or torso, got 'head'")
    rejected("env.device", "tpu", "env.device must be cpu, cuda")
    rejected("env.backend", "nope", "env.backend='nope' is not available")
    for field, value in (
        ("algorithm.termination_body", "torso"),
        ("algorithm.seed", None),
        ("algorithm.termination_bias_init", -4.6),
        ("algorithm.action_noise_std", 0.0),
        ("env.device", "cuda:1"),
    ):
        legal = copy.deepcopy(cfg)
        parent, name = field.split(".")
        setattr(getattr(legal, parent), name, value)
        validate_supermimic_cfg(legal)
    intervals = copy.deepcopy(cfg)
    intervals.runner.save_interval = 1
    intervals.runner.val_check_interval = 3
    with pytest.raises(AssertionError, match="multiple of runner.val_check_interval"):
        validate_supermimic_cfg(copy.deepcopy(intervals))
    intervals.runner.save_interval = 6
    validate_supermimic_cfg(intervals)
    velocity = copy.deepcopy(composed["supermimic_velocity"])
    velocity.env.motion_path = "/tmp/lafan1_g1"
    velocity.env.num_envs = 5
    with pytest.raises(AssertionError, match="in half"):
        validate_supermimic_cfg(velocity)


def test_eval_options_are_checked_before_a_checkpoint_loads():
    from rlinf.workers.supermimic.validate import validate_supermimic_eval_cfg

    common = OmegaConf.load(
        CONFIG_DIR.parents[2]
        / "evaluations"
        / "supermimic"
        / "supermimic_eval_common.yaml"
    )
    validate_supermimic_eval_cfg(common.eval)
    for name, value, match in (
        ("velocity", "bogus", "none, smoke or g2, got 'bogus'"),
        ("g2_trials", 0, r"eval.g2_trials must be in \[1, inf\]"),
        ("g2_batch", 0, "eval.g2_batch must be in"),
        ("g2_steady_seconds", 6.0, r"eval.g2_steady_seconds must be in \(0.0, 5.0\]"),
        ("velocity_smoke_seconds", 0.0, "eval.velocity_smoke_seconds must be in"),
        ("seed", "abc", "eval.seed must be an integer"),
    ):
        bad = copy.deepcopy(common.eval)
        bad[name] = value
        with pytest.raises((AssertionError, ValueError), match=match):
            validate_supermimic_eval_cfg(bad)


def test_hydra_four_yaml_inheritance():
    base = _compose("supermimic_base")
    lafan1 = _compose("supermimic_lafan1")
    anchor = _compose("supermimic_lafan1_anchor")
    smoke = _compose("supermimic_smoke")

    assert base.algorithm.k == 8
    assert base.algorithm.H == 32
    assert base.algorithm.obs_dim == 93
    assert base.algorithm.goal_frame_dim == 68
    assert base.env.num_envs == 4096

    lafan1.env.motion_path = "/tmp/lafan1_g1"
    lafan1 = validate_supermimic_cfg(lafan1)
    assert lafan1.algorithm.goal_anchor is False
    assert lafan1.algorithm.goal_dim == 680
    assert lafan1.algorithm.obs_history_dim == 930
    assert lafan1.algorithm.critic_in_dim == 680 + 71 + 33 + 4
    assert lafan1.env.num_envs == 4096
    assert lafan1.algorithm.imagine_batch == 4096
    assert lafan1.algorithm.wm_window_batch == 2048
    assert lafan1.algorithm.total_updates == 320000

    anchor.env.motion_path = "/tmp/lafan1_g1"
    anchor = validate_supermimic_cfg(anchor)
    assert anchor.algorithm.goal_anchor is True
    assert anchor.algorithm.goal_dim == 689
    assert anchor.algorithm.obs_dim == 93
    assert anchor.algorithm.obs_history_dim == 930
    assert anchor.env.num_envs == 4096
    assert anchor.algorithm.k == 8
    assert anchor.algorithm.H == 32

    smoke.env.motion_path = "/tmp/lafan1_g1"
    smoke = validate_supermimic_cfg(smoke)
    assert smoke.algorithm.goal_dim == 680
    assert smoke.env.num_envs == 8
    assert smoke.algorithm.imagine_batch == 8
    assert smoke.algorithm.wm_window_batch == 64
    assert smoke.algorithm.total_updates == 3
    assert smoke.algorithm.k == 8
    assert smoke.algorithm.H == 32
    assert smoke.env.backend == "mjlab"


def test_validate_rejects_bad_k_and_missing_motion():
    cfg = _compose("supermimic_lafan1")
    cfg.algorithm.k = 7
    with pytest.raises(AssertionError, match="algorithm.k"):
        validate_supermimic_cfg(cfg)

    cfg = _compose("supermimic_lafan1")
    cfg.env.motion_path = ""
    cfg.env.backend = "mjlab"
    with pytest.raises(AssertionError, match="motion_path"):
        validate_supermimic_cfg(cfg)


def test_validate_multi_gpu_placement_splits_global_batches():
    cfg = _compose("supermimic_smoke")
    cfg.env.motion_path = "/tmp/lafan1_g1"
    cfg.cluster.component_placement.supermimic = "all"
    with pytest.raises(AssertionError, match="world_size"):
        validate_supermimic_cfg(cfg)
    cfg = _compose("supermimic_smoke")
    cfg.env.motion_path = "/tmp/lafan1_g1"
    cfg.cluster.component_placement.supermimic = "0-1"
    cfg.env.num_envs = 4
    cfg.algorithm.wm_window_batch = 8
    cfg.algorithm.imagine_batch = 8
    assert validate_supermimic_cfg(cfg).algorithm.world_size == 2
    cfg = _compose("supermimic_smoke")
    cfg.env.motion_path = "/tmp/lafan1_g1"
    cfg.env.num_envs = 3
    with pytest.raises(AssertionError, match="divisible"):
        validate_supermimic_cfg(cfg, world_size=2)


def test_validate_does_not_mutate_user_k():
    cfg = _compose("supermimic_smoke")
    cfg.env.motion_path = "/tmp/lafan1_g1"
    before = deepcopy(OmegaConf.to_container(cfg.algorithm, resolve=True))
    out = validate_supermimic_cfg(cfg)
    assert out.algorithm.k == before["k"]
    assert out.algorithm.H == before["H"]
    assert "goal_dim" in out.algorithm
