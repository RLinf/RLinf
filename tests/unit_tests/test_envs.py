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

"""Heterogeneous env modes, and the observations env workers send the policy."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import rlinf.workers.env.env_worker as env_worker_module
from rlinf.config import (
    resolve_env_mode_cfg,
    validate_cfg,
    validate_embodied_cfg,
    validate_env_policy_fields,
    validate_env_training_cfg,
    validate_eval_env_cfg,
)
from rlinf.data.schema.embodied_types import PolicyInput
from rlinf.envs import get_env_cls
from rlinf.envs.layout import EnvAssignment, EnvLayout
from rlinf.envs.policy_obs import PolicyObsSpec
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.workers.env.env_worker import EnvWorker


def _env(env_type: str, total_num_envs: int, **fields) -> dict:
    return {
        "env_type": env_type,
        "total_num_envs": total_num_envs,
        "rollout_epoch": 1,
        "max_steps_per_rollout_epoch": 80,
        "max_episode_steps": 80,
        "auto_reset": False,
        "ignore_terminations": False,
        "group_size": 1,
        "video_cfg": {"save_video": False},
        **fields,
    }


def _hetero_mode(**envs: dict):
    return OmegaConf.create({"envs": envs})


def test_homogeneous_mode_is_hosted_unchanged_by_every_worker():
    mode_cfg = OmegaConf.create(_env("libero", 8))
    layout = EnvLayout(mode_cfg, world_size=2, path="env.train")

    assignment = layout.assign(1)

    assert assignment.cfg is mode_cfg
    assert (assignment.name, assignment.rank, assignment.world_size) == (None, 1, 2)
    assert layout.mode_fields() == {}
    metrics = {"success_once": torch.ones(2)}
    assert assignment.scoped(metrics) == metrics


def test_listed_envs_get_contiguous_worker_blocks_sized_by_their_share():
    mode_cfg = _hetero_mode(libero=_env("libero", 48), maniskill=_env("maniskill", 16))
    layout = EnvLayout(mode_cfg, world_size=4, path="env.train")

    hosted = [(a.name, a.rank, a.world_size) for a in map(layout.assign, range(4))]

    assert hosted == [
        ("libero", 0, 3),
        ("libero", 1, 3),
        ("libero", 2, 3),
        ("maniskill", 0, 1),
    ]
    assert layout.assign(3).cfg is mode_cfg.envs.maniskill


@pytest.mark.parametrize(
    ("libero_fields", "agreed"),
    [
        ({}, {"max_steps_per_rollout_epoch": 80, "max_episode_steps": 80}),
        ({"max_episode_steps": 300, "max_steps_per_rollout_epoch": 320}, {}),
    ],
)
def test_heterogeneous_mode_carries_only_the_loop_fields_its_envs_agree_on(
    libero_fields, agreed
):
    mode_cfg = _hetero_mode(
        libero=_env("libero", 8, **libero_fields),
        maniskill=_env("maniskill", 8),
    )

    fields = EnvLayout(mode_cfg, world_size=2, path="env.train").mode_fields()

    assert fields == {
        "total_num_envs": 16,
        "rollout_epoch": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        **agreed,
    }


@pytest.mark.parametrize(
    ("mode", "world_size", "match"),
    [
        # 64 envs on 2 workers is 32 per worker, which 48 LIBERO envs cannot fill.
        (
            {
                "envs": {
                    "libero": _env("libero", 48),
                    "maniskill": _env("maniskill", 16),
                }
            },
            2,
            r"libero.total_num_envs must be a multiple of 32",
        ),
        (
            {
                "max_steps_per_rollout_epoch": 80,
                "envs": {
                    "libero": _env("libero", 8),
                    "maniskill": _env("maniskill", 8, max_steps_per_rollout_epoch=40),
                },
            },
            2,
            r"env.train.max_steps_per_rollout_epoch is 80, but the envs under "
            r"env.train.envs set max_steps_per_rollout_epoch to",
        ),
        (
            {"video_cfg": {"save_video": False}, "envs": {"libero": _env("libero", 8)}},
            1,
            r"env.train.video_cfg would apply to none of them",
        ),
        (
            {
                "total_num_envs": 32,
                "envs": {
                    "libero": _env("libero", 8),
                    "maniskill": _env("maniskill", 8),
                },
            },
            2,
            r"env.train.total_num_envs is 32, but the envs under env.train.envs give 16",
        ),
        (
            {"envs": {"libero": {"total_num_envs": 8}}},
            1,
            r"env.train.envs.libero must be an env config with env_type",
        ),
        (
            {"envs": {"libero": _env("libero", 0)}},
            1,
            r"total_num_envs must be a positive int",
        ),
    ],
)
def test_layout_rejects_modes_its_workers_cannot_run(mode, world_size, match):
    with pytest.raises(ValueError, match=match):
        EnvLayout(OmegaConf.create(mode), world_size=world_size, path="env.train")


def test_each_env_reports_its_own_metrics_beside_the_overall_ones():
    layout = EnvLayout(
        _hetero_mode(libero=_env("libero", 4), maniskill=_env("maniskill", 2)),
        world_size=3,
        path="env.eval",
    )
    success_per_rank = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ]

    metrics = compute_evaluate_metrics(
        [
            {
                "success_once": success,
                **layout.assign(rank).scoped({"success_once": success}),
            }
            for rank, success in enumerate(success_per_rank)
        ]
    )

    assert float(metrics["success_once"]) == pytest.approx(0.5)
    assert float(metrics["libero/success_once"]) == pytest.approx(0.75)
    assert float(metrics["maniskill/success_once"]) == pytest.approx(0.0)
    assert metrics["num_trajectories"] == 6


def test_env_worker_runs_its_env_on_the_workers_hosting_that_env(monkeypatch):
    created = []

    class Simulator:
        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, **_):
            created.append((cfg.env_type, num_envs, seed_offset, total_num_processes))

    requested = []

    def get_env_cls(env_type, env_cfg):
        requested.append(env_type)
        return Simulator

    monkeypatch.setattr(env_worker_module, "get_env_cls", get_env_cls)
    mode_cfg = _hetero_mode(
        libero=_env("libero", 8),
        maniskill=_env(
            "maniskill", 8, override_cfgs=[{"scene": "first"}, {"scene": "second"}]
        ),
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"env": {}})
    worker.stage_num = 2
    worker._worker_info = None
    worker.use_reward_model = False
    worker.use_realworld_reward = False
    # The last of 4 workers is the second of the 2 hosting ManiSkill.
    worker.train_assignment = EnvLayout(mode_cfg, 4, "env.train").assign(3)
    worker.eval_assignment = None

    worker.update_env_cfg()
    worker._setup_env_and_wrappers(
        worker.train_assignment, num_envs_per_stage=2, insert_delay=False
    )

    assert requested == ["maniskill"]
    assert created == [("maniskill", 2, 2, 4), ("maniskill", 2, 3, 4)]
    assert mode_cfg.envs.maniskill.override_cfg.scene == "second"


def test_config_gives_a_heterogeneous_mode_the_loop_fields_rollout_reads():
    mode_cfg = _hetero_mode(libero=_env("libero", 8), maniskill=_env("maniskill", 8))
    OmegaConf.set_struct(mode_cfg, True)

    resolve_env_mode_cfg(mode_cfg, "env.train", env_world_size=2)

    assert mode_cfg.total_num_envs == 16
    assert mode_cfg.max_steps_per_rollout_epoch == 80
    assert mode_cfg.max_episode_steps == 80
    assert mode_cfg.auto_reset is False


def test_spec_normalizes_masked_and_unmasked_trajectories_to_their_own_length():
    layout = EnvLayout(
        _hetero_mode(
            libero=_env(
                "libero", 4, max_episode_steps=300, max_steps_per_rollout_epoch=320
            ),
            maniskill=_env("maniskill", 4, ignore_terminations=True),
        ),
        world_size=2,
        path="env.train",
    )
    libero, maniskill = layout.spec_of_rank(0), layout.spec_of_rank(1)

    assert (libero.name, libero.ranks) == ("libero", range(0, 1))
    assert (libero.has_loss_mask, libero.reference_length) == (True, 300)
    assert (maniskill.has_loss_mask, maniskill.reference_length) == (False, 80)
    assert libero.chunk_steps(num_action_chunks=8) == 40
    assert layout.spec("maniskill") is maniskill


@pytest.mark.parametrize(
    ("env_weighting", "expected"),
    [
        # 4 LIBERO trajectories of 320 steps and 8 ManiSkill ones of 80 steps.
        ("trajectory", {"libero": 160.0, "maniskill": 160.0}),
        ("sample", {"libero": 320.0, "maniskill": 80.0}),
    ],
)
def test_env_weighting_picks_one_shared_or_one_own_length_per_env(
    env_weighting, expected
):
    layout = EnvLayout(
        _hetero_mode(
            libero=_env(
                "libero", 4, max_episode_steps=320, max_steps_per_rollout_epoch=320
            ),
            maniskill=_env("maniskill", 4, rollout_epoch=2),
        ),
        world_size=2,
        path="env.train",
    )

    assert layout.reference_lengths(env_weighting) == expected


def test_one_env_keeps_its_own_length_under_either_weighting():
    layout = EnvLayout(OmegaConf.create(_env("libero", 8)), world_size=2, path="p")

    assert layout.reference_lengths("trajectory") == {None: 80.0}
    assert layout.reference_lengths("sample") == {None: 80.0}
    with pytest.raises(ValueError, match="'trajectory' or 'sample', got 'episode'"):
        layout.reference_lengths("episode")


def test_each_env_sees_the_algorithm_with_its_own_group_size_and_overrides():
    layout = EnvLayout(
        _hetero_mode(
            libero=_env("libero", 8, group_size=8),
            maniskill=_env("maniskill", 8, group_size=4),
        ),
        world_size=2,
        path="env.train",
    )
    algorithm_cfg = OmegaConf.create(
        {
            "group_size": 2,
            "gamma": 1.0,
            "envs": {"maniskill": {"gamma": 0.99, "loss_weight": 2.0}},
        }
    )

    libero = layout.spec("libero").algorithm_cfg(algorithm_cfg)
    maniskill = layout.spec("maniskill").algorithm_cfg(algorithm_cfg)

    assert (libero.group_size, libero.gamma, libero.get("loss_weight")) == (
        8,
        1.0,
        None,
    )
    assert (maniskill.group_size, maniskill.gamma, maniskill.loss_weight) == (
        4,
        0.99,
        2.0,
    )
    assert "envs" not in maniskill


def _training_cfg(algorithm=None, runner=None, **envs):
    return OmegaConf.create(
        {
            "env": {"train": {"envs": envs}},
            "algorithm": {"loss_type": "actor_critic", **(algorithm or {})},
            "runner": runner or {},
        }
    )


@pytest.mark.parametrize(
    ("algorithm", "runner", "match"),
    [
        ({"envs": {"metaworld": {}}}, None, r"must name an env under env.train.envs"),
        (
            {"envs": {"libero": {"group_size": 4}}},
            None,
            r"may set \['loss_weight'.*got \['group_size'\]",
        ),
        ({"adv_norm_scope": "rank"}, None, r"must be 'env' or 'all', got 'rank'"),
        (
            {"adv_type": "grpo", "adv_norm_scope": "all"},
            None,
            r"adv_norm_scope='all' sets the scope .* got adv_type='grpo'",
        ),
        (
            {"adv_type": "gae", "adv_norm_scope": "all", "normalize_advantages": False},
            None,
            r"normalize_advantages=False",
        ),
        (
            {"adv_type": "raw", "adv_norm_scope": "all"},
            None,
            r"normalize_advantages=False",
        ),
        (
            {"adv_type": "gae_dynamic", "adv_norm_scope": "all"},
            None,
            r"got adv_type='gae_dynamic'",
        ),
        (
            {"adv_type": "grpo", "group_size": 8},
            None,
            r"so env.train.envs.libero needs group_size > 1, got 1",
        ),
        (
            {"loss_type": "embodied_nft"},
            None,
            r"heterogeneous NFT needs algorithm.env_weighting='sample'",
        ),
        (
            {"envs": {"libero": {"loss_weight": 0.0}}},
            None,
            r"Expected a positive loss_weight for env.train.envs.libero, got 0.0",
        ),
        (
            {"envs": {"maniskill": {"loss_weight": -0.5}}},
            None,
            r"Expected a positive loss_weight for env.train.envs.maniskill, got -0.5",
        ),
        ({}, {"use_training_pipeline": True}, r"runner.use_training_pipeline"),
    ],
)
def test_config_rejects_per_env_settings_the_run_cannot_honor(algorithm, runner, match):
    cfg = _training_cfg(
        algorithm, runner, libero=_env("libero", 8), maniskill=_env("maniskill", 8)
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=match):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)


@pytest.mark.parametrize(("maniskill_steps", "accepted"), [(80, True), (84, False)])
def test_config_checks_the_horizon_of_each_eval_env(maniskill_steps, accepted):
    # The envs' horizons differ, so env.eval itself carries none.
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "envs": {
                        "libero": _env("libero", 8, max_steps_per_rollout_epoch=320),
                        "maniskill": _env(
                            "maniskill", 8, max_steps_per_rollout_epoch=maniskill_steps
                        ),
                    }
                }
            }
        }
    )

    if accepted:
        assert validate_eval_env_cfg(cfg, 2, 1, num_action_chunks=8) == 8
        return
    with pytest.raises(
        AssertionError, match=r"env.eval.envs.maniskill.max_steps_per_rollout_epoch"
    ):
        validate_eval_env_cfg(cfg, 2, 1, num_action_chunks=8)


def test_config_keeps_reward_models_to_one_env_configuration():
    cfg = _training_cfg(libero=_env("libero", 8), maniskill=_env("maniskill", 8))
    cfg.reward = {"use_reward_model": True}
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=r"reward.use_reward_model"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)


def test_config_runs_listed_envs_only_on_the_hf_rollout_backend():
    # The SGLang eval worker reads one loop per mode, as on a homogeneous run.
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False, "task_type": "embodied"},
            "actor": {"model": {"model_type": "openvla_oft"}},
            "rollout": {"rollout_backend": "sglang"},
            "algorithm": {},
            "env": {
                "eval": {
                    "envs": {
                        "libero": _env("libero", 8),
                        "maniskill": _env("maniskill", 8),
                    }
                }
            },
        }
    )

    with pytest.raises(
        AssertionError, match=r"only rollout_backend='huggingface' runs listed envs"
    ):
        validate_embodied_cfg(cfg)


def test_config_needs_each_env_split_evenly_across_actors_in_whole_groups():
    # One worker of 8 ManiSkill envs feeds 4 actors 2 envs each, which splits
    # its groups of 4.
    cfg = _training_cfg(
        libero=_env("libero", 8), maniskill=_env("maniskill", 8, group_size=4)
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)
    with pytest.raises(AssertionError, match=r"4 shards of whole groups of 4"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=4)


@pytest.mark.parametrize(
    ("model_type", "accepted"), [("openvla_oft", True), ("openpi", False)]
)
def test_config_gives_per_env_action_statistics_only_to_a_policy_that_reads_them(
    model_type, accepted
):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "envs": {
                        "libero": _env("libero", 8),
                        "maniskill": _env("maniskill", 8, unnorm_key="bridge_orig"),
                    }
                }
            }
        }
    )
    model_cfg = OmegaConf.create({"model_type": model_type})

    if accepted:
        validate_env_policy_fields(cfg, model_cfg)
        return
    with pytest.raises(
        AssertionError,
        match=r"maniskill.unnorm_key is read by model_type \['openvla_oft'\], "
        r"got 'openpi'",
    ):
        validate_env_policy_fields(cfg, model_cfg)


def test_config_keeps_smooth_intervention_to_a_homogeneous_mode():
    mode_cfg = _hetero_mode(
        real=_env("real", 1, smooth_intervene=True), maniskill=_env("maniskill", 1)
    )

    with pytest.raises(
        AssertionError, match="real.smooth_intervene requires a homogeneous"
    ):
        resolve_env_mode_cfg(mode_cfg, "env.train", env_world_size=2)


def test_policy_obs_spec_leaves_observations_alone_when_unset():
    obs = {"main_images": torch.zeros(1, 4, 4, 3, dtype=torch.uint8)}

    spec = PolicyObsSpec.from_cfg(OmegaConf.create(_env("libero", 1)))

    assert spec.apply(obs) is obs


def test_observations_of_different_simulators_conform_to_one_policy_batch():
    # LIBERO sends a 256x256 main and wrist view; this ManiSkill scene sends
    # one 480x640 view and a differently sized state.
    libero_obs = {
        "main_images": torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.zeros(2, 8),
        "task_descriptions": ["open the drawer", "open the drawer"],
    }
    maniskill_obs = {
        "main_images": torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8),
        "wrist_images": None,
        "extra_view_images": None,
        "states": torch.zeros(1, 7),
        "task_descriptions": ["put the carrot on the plate"],
    }
    spec = PolicyObsSpec.from_cfg(
        OmegaConf.create(
            {
                "policy_obs": {
                    "keys": ["main_images", "task_descriptions"],
                    "image_size": [224, 224],
                }
            }
        )
    )

    batch = PolicyInput.merge(
        [
            PolicyInput(obs=spec.apply(libero_obs)),
            PolicyInput(obs=spec.apply(maniskill_obs)),
        ]
    )

    assert batch.obs["main_images"].shape == (3, 224, 224, 3)
    assert batch.obs["main_images"].dtype == torch.uint8
    assert batch.obs["wrist_images"] is None
    assert batch.obs["states"] is None
    assert batch.obs["task_descriptions"] == [
        "open the drawer",
        "open the drawer",
        "put the carrot on the plate",
    ]
    assert libero_obs["wrist_images"] is not None


def test_policy_obs_spec_resizes_stacked_views_and_keeps_pixel_values():
    spec = PolicyObsSpec(image_size=[2, 2])
    obs = {
        "wrist_images": np.full((1, 3, 4, 4, 3), 200, dtype=np.uint8),
        "states": torch.ones(1, 7),
    }

    conformed = spec.apply(obs)

    assert conformed["wrist_images"].shape == (1, 3, 2, 2, 3)
    assert torch.equal(
        conformed["wrist_images"], torch.full((1, 3, 2, 2, 3), 200, dtype=torch.uint8)
    )
    assert conformed["states"] is obs["states"]


def test_policy_obs_spec_passes_control_entries_through_and_conforms_history():
    active_mask = torch.tensor([True, False])
    frame_mask = torch.ones(2, 3, dtype=torch.bool)
    history = {
        "main_images": torch.zeros(2, 3, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 3, 4, 4, 3, dtype=torch.uint8),
        "frame_mask": frame_mask,
    }
    obs = {
        "main_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 8),
        "active_mask": active_mask,
        "history": history,
    }

    conformed = PolicyObsSpec(keys=["main_images"], image_size=[2, 2]).apply(obs)

    assert conformed["states"] is None
    assert conformed["active_mask"] is active_mask
    assert conformed["history"]["main_images"].shape == (2, 3, 2, 2, 3)
    assert conformed["history"]["wrist_images"] is None
    assert conformed["history"]["frame_mask"] is frame_mask
    assert history["main_images"].shape == (2, 3, 4, 4, 3)


@pytest.mark.parametrize("only_eval", [True, False])
def test_psi0_validation_checks_each_listed_env(only_eval):
    from rlinf.config import _validate_psi0_eval_cfg, _validate_psi0_train_cfg

    mode = "eval" if only_eval else "train"
    cfg = OmegaConf.create(
        {
            "runner": {
                "only_eval": only_eval,
                "val_check_interval": -1,
                "rtc": {"enabled": False},
                "enable_decoupled_mode": False,
            },
            "rollout": {
                "rollout_backend": "huggingface",
                "collect_prev_infos": not only_eval,
                "recompute_logprobs": False,
            },
            "actor": {
                "model": {
                    "num_action_chunks": 24,
                    "action_dim": 36,
                    "add_value_head": True,
                    "psi0": {
                        "plan_horizon": 30,
                        "overlap_horizon": 6,
                        "sampler": (
                            "deterministic_euler"
                            if only_eval
                            else "stochastic_transition"
                        ),
                        "rtc": {"enabled": True},
                    },
                },
                "fsdp_config": {"use_orig_params": True},
            },
            "algorithm": {"adv_type": "gae", "loss_type": "actor_critic"},
            "critic": {"use_critic_model": False},
            "env": {
                mode: {
                    "envs": {
                        "a": {"env_type": "simple", "total_num_envs": 1},
                        "b": {"env_type": "simple", "total_num_envs": 1},
                    }
                }
            },
        }
    )
    OmegaConf.set_struct(cfg, True)
    validator = _validate_psi0_eval_cfg if only_eval else _validate_psi0_train_cfg

    validator(cfg, cfg.actor.model, only_eval)

    cfg.env[mode].envs.b.env_type = "libero"
    with pytest.raises(AssertionError, match=r"env_type='simple'"):
        validator(cfg, cfg.actor.model, only_eval)


def test_policy_obs_spec_rejects_keys_the_policy_batch_does_not_carry():
    with pytest.raises(ValueError, match=r"unknown keys \['main_image'\]"):
        PolicyObsSpec(keys=["main_image"])


def test_policy_obs_spec_aligns_missing_camera_keys_across_eval_obs():
    from rlinf.data.schema.embodied_types import batch_format

    spec = PolicyObsSpec(keys=["main_images", "task_descriptions"], image_size=[8, 8])
    libero_obs = {
        "main_images": torch.zeros(2, 16, 16, 3),
        "wrist_images": torch.zeros(2, 16, 16, 3),
        "states": torch.zeros(2, 7),
        "task_descriptions": ["pick", "place"],
    }
    maniskill_obs = {
        "main_images": torch.zeros(2, 12, 20, 3),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["put", "stack"],
    }
    assert batch_format(spec.apply(libero_obs)) == batch_format(
        spec.apply(maniskill_obs)
    )


def test_validate_eval_env_cfg_defaults_missing_group_size():
    eval_cfg = _env("d4rl", 4)
    eval_cfg.pop("group_size")
    cfg = OmegaConf.create({"env": {"eval": eval_cfg}})
    OmegaConf.set_struct(cfg, True)
    assert (
        validate_eval_env_cfg(cfg, env_world_size=2, stage_num=1, num_action_chunks=8)
        == 2
    )


def test_config_needs_grpo_envs_to_stop_trajectories_at_their_first_done():
    cfg = _training_cfg(
        {"adv_type": "grpo"},
        libero=_env("libero", 8, group_size=2),
        maniskill=_env("maniskill", 8, group_size=2, ignore_terminations=True),
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=r"env.train.envs.maniskill needs"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)


@pytest.fixture
def arena_config(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "evaluations/isaaclab_arena"), version_base="1.1"
    ):
        return compose(config_name="isaaclab_arena_pi05_eval")


@pytest.fixture
def arena_python(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("ARENA_TEST_GPU", "5")
    monkeypatch.setenv("LOCAL_RANK", "0")
    runtime = Path(__file__).with_name("_isaaclab_arena.py")
    python = tmp_path / "arena-python"
    python.write_text(
        f"#!{sys.executable}\nimport os, runpy\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['ARENA_TEST_GPU'], 'Arena must inherit the worker GPU mask'\n"
        f"runpy.run_path({str(runtime)!r}, run_name='__main__')\n"
    )
    python.chmod(0o755)
    monkeypatch.setenv("ISAACLAB_ARENA_PYTHON", str(python))


@pytest.mark.parametrize("registered", [False, True], ids=["graph", "registered"])
def test_isaaclab_arena_evaluation(arena_config, arena_python, registered, monkeypatch):
    cfg = arena_config.env.eval
    if registered:
        cfg.init_params.environment = "pick_and_place_maple_table"
        monkeypatch.setenv("ARENA_TEST_RENDER_FPS", "24")
        monkeypatch.setenv("ARENA_TEST_RERENDERS", "3")
    OmegaConf.set_readonly(arena_config, True)
    cls = get_env_cls(cfg.env_type, cfg)
    env = cls(
        cfg,
        num_envs=3,
        seed_offset=2,
        total_num_processes=1,
        worker_info=SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=5),
    )
    with env:
        obs, _ = env.reset()
        assert obs["task_descriptions"] == ["Place the banana on the plate."] * 3, (
            "Expected the Arena task prompt in every lane"
        )
        from rlinf.envs.wrappers.record_video import RecordVideo

        with pytest.raises(ValueError, match="DROID actions"):
            env.chunk_step(torch.zeros(3, 1, 7))
        recorder = RecordVideo(env, cfg.video_cfg)
        assert recorder._fps == (24 if registered else 15), (
            "The video must use the native control frequency"
        )
        torch.testing.assert_close(obs["states"][:, 7], torch.full((3,), 0.02))
        torch.testing.assert_close(obs["wrist_images"], obs["main_images"] + 50)

        worker = object.__new__(EnvWorker)
        worker.eval_assignment = EnvAssignment(None, cfg, rank=0, world_size=1)
        worker.model_cfg = arena_config.rollout.model
        worker.eval_env_list = [env]
        worker.eval_prev_done = [torch.zeros(3, dtype=torch.bool)]
        worker.use_external_reward_model = False
        actions = torch.zeros(
            3,
            arena_config.rollout.model.num_action_chunks,
            arena_config.rollout.model.action_dim,
        )
        actions[..., -1] = 0.75
        output, episode = worker.env_evaluate_step(actions, stage_id=0)
        torch.testing.assert_close(episode["episode_len"], torch.tensor([1, 2, 3]))
        torch.testing.assert_close(episode["return"], torch.tensor([1.0, 4.0, 9.0]))
        torch.testing.assert_close(
            episode["success_once"], torch.tensor([True, False, False])
        )
        torch.testing.assert_close(
            output.obs["states"][:, 0], torch.tensor([1.0, 2.0, 3.0])
        )
        torch.testing.assert_close(output.obs["states"][:, 7], torch.full((3,), 0.75))
        assert not output.obs["active_mask"].any(), "Completed lanes must be inactive"
        torch.testing.assert_close(
            output.env_infos["executed_mask"].sum(dim=1), torch.tensor([1, 2, 3])
        )
        metrics = compute_evaluate_metrics([episode])
        assert metrics["success_once"] == pytest.approx(1 / 3), (
            "Only the first lane reached the goal"
        )
        assert metrics["num_trajectories"] == 3, (
            "Evaluation should count one trajectory per lane"
        )
        _, later = worker.env_evaluate_step(actions, stage_id=0)
        assert not later, "Inactive lanes must not contribute another episode"
        obs, _ = env.reset(seed=9)
        torch.testing.assert_close(obs["states"][:, 7], torch.full((3,), 0.09))
    env.close()


def test_isaaclab_arena_same_step_autoreset_and_sparse_reward():
    from unit_tests._isaaclab_arena import Droid

    from rlinf.envs.sim.isaaclab_arena.simulation import ArenaRollout

    batch = ArenaRollout(
        gym.wrappers.OrderEnforcing(Droid(with_rewards=False)), auto_reset=True
    )
    batch.reset(0)
    actions = np.zeros((3, 1, 8), dtype=np.float32)
    for step in range(1, 4):
        obs, rewards, terms, truncs, infos = batch.chunk_step(actions)
        np.testing.assert_array_equal(rewards[:, 0], [1, 0, 0])
        np.testing.assert_array_equal(terms[:, 0], [True, step == 2, False])
        np.testing.assert_array_equal(truncs[:, 0], [False, False, step == 3])
        np.testing.assert_array_equal(
            infos[0]["_final_observation"], terms[:, 0] | truncs[:, 0]
        )
    np.testing.assert_array_equal(obs[0]["states"][:, 0], [0, 1, 0])
    np.testing.assert_array_equal(
        infos[0]["final_observation"]["states"][:, 0], [1, 1, 3]
    )
    np.testing.assert_array_equal(
        infos[0]["final_info"]["episode"]["episode_len"], [1, 1, 3]
    )
    with pytest.raises(ValueError, match="num_action_chunks=1"):
        batch.chunk_step(np.zeros((3, 2, 8)))


@pytest.mark.parametrize("shape", [(3, 8), (3, 0, 8), (1, 2, 8), (3, 2, 7)])
def test_isaaclab_arena_invalid_action_batch(shape):
    from unit_tests._isaaclab_arena import Droid

    from rlinf.envs.sim.isaaclab_arena.simulation import ArenaRollout

    batch = ArenaRollout(Droid(), auto_reset=False)
    batch.reset(None)
    with pytest.raises(ValueError, match="DROID actions"):
        batch.chunk_step(np.zeros(shape))


@pytest.mark.parametrize(
    ("environment", "args", "error"),
    [
        ("missing.yaml", {}, "No Arena graph at .*missing.yaml"),
        (
            "pick_and_place_maple_table",
            {"embodiment": "franka"},
            "requires droid_abs_joint_pos, got 'franka'",
        ),
    ],
)
def test_isaaclab_arena_startup_failure(
    arena_config, arena_python, environment, args, error
):
    cfg = arena_config.env.eval
    cfg.init_params.environment = environment
    cfg.init_params.environment_args = args
    with pytest.raises(RuntimeError, match=error):
        get_env_cls(cfg.env_type)(
            cfg, 1, 0, 1, SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=0)
        )


def test_isaaclab_arena_training_step_bootstraps_at_the_time_limit(
    arena_config, arena_python
):
    from rlinf.scheduler import AcceleratorType

    cfg = arena_config.env.eval
    cfg.is_eval = False
    cfg.auto_reset = True
    cfg.max_episode_steps = 1
    cfg.init_params.language_instruction = "Pick up the apple."
    model = arena_config.rollout.model
    model.num_action_chunks = 1
    OmegaConf.set_readonly(arena_config, True)
    with get_env_cls(cfg.env_type)(
        cfg, 2, 0, 1, SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=0)
    ) as env:
        with pytest.raises(
            ValueError, match=r"env.train.seed.+reset\(\) without a seed"
        ):
            env.reset(seed=42)
        env.reset()
        actions = torch.full((2, 1, 8), 0.75)
        worker = object.__new__(EnvWorker)
        worker._accelerator_type = AcceleratorType.NO_ACCEL
        worker._timer_metrics = {}
        worker.train_assignment = EnvAssignment(None, cfg, rank=0, world_size=1)
        worker.model_cfg = model
        worker.env_list = [env]
        worker.enable_online_lerobot = False
        worker.use_external_reward_model = False
        output, metrics, _ = worker.env_interact_step(actions, stage_id=0)
        torch.testing.assert_close(
            output.terminations[:, 0], torch.tensor([True, False])
        )
        torch.testing.assert_close(
            output.truncations[:, 0], torch.tensor([False, True])
        )
        torch.testing.assert_close(output.obs["states"][:, 0], torch.zeros(2))
        torch.testing.assert_close(output.final_obs["states"][:, 0], torch.ones(2))
        for camera in ("main_images", "wrist_images"):
            torch.testing.assert_close(
                output.final_obs[camera][:, :, 1:], output.obs[camera][:, :, 1:] + 1
            )
        with pytest.raises(ValueError, match="num_action_chunks=1"):
            env.chunk_step(torch.zeros(2, 3, 8))
        rewards = output.transition.compute_rewards(
            bootstrap_values=torch.tensor([4.0, 4.0]),
            env_reward_weight=1.0,
            reward_weight=0.0,
            reward_scale=1.0,
            auto_reset=True,
            bootstrap_type="standard",
            gamma=0.5,
        )
        torch.testing.assert_close(rewards[:, 0], torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(metrics["success_once"], torch.tensor([True, False]))
        obs, reward, _, _, info = env.step(actions[:, 0])
        with pytest.raises(ValueError, match="DROID actions"):
            env.step(actions)
        assert obs["task_descriptions"] == ["Pick up the apple."] * 2, (
            "Expected the custom instruction"
        )
        torch.testing.assert_close(reward, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(
            info["episode"]["episode_len"], torch.ones(2, dtype=torch.long)
        )


def test_isaaclab_arena_requires_nvidia_placement(arena_config):
    cfg = arena_config.env.eval
    with pytest.raises(ValueError, match="NVIDIA RTX"):
        get_env_cls(cfg.env_type)(
            cfg,
            1,
            0,
            1,
            SimpleNamespace(accelerator_type="AMD_GPU", accelerator_rank=0),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("max_episode_steps", None, "positive integer max_episode_steps"),
        ("max_episode_steps", 0, "positive integer max_episode_steps"),
        ("max_steps_per_rollout_epoch", 1, "every assigned episode"),
        ("auto_reset", True, "auto_reset=false for evaluation"),
        ("group_size", 2, "independent placements"),
        ("use_fixed_reset_state_ids", True, "independent placements"),
        ("use_ordered_reset_state_ids", True, "independent placements"),
    ],
)
def test_isaaclab_arena_invalid_protocol(arena_config, field, value, error):
    from rlinf.envs.sim.isaaclab_arena.config import validate_arena_config

    cfg = arena_config.env.eval
    assert cfg.max_episode_steps == cfg.max_steps_per_rollout_epoch == 600, (
        "Banana on plate allows 40 seconds at 15 Hz"
    )
    OmegaConf.update(cfg, field, value, force_add=True)
    with pytest.raises(ValueError, match=error):
        validate_arena_config(cfg, evaluation=True)


@pytest.fixture
def arena_ppo_config(arena_config, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    base = (
        root
        / "examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi_pi05.yaml"
    )
    data = []
    for language in ("en", "zh"):
        guide = (
            root
            / f"docs/source-{language}/rst_source/evaluations/guides/isaaclab_arena.rst"
        )
        block = (
            guide.read_text()
            .split(".. code-block:: yaml\n\n", 1)[1]
            .split("\n\n", 1)[0]
        )
        data.append(OmegaConf.create(textwrap.dedent(block)))
    assert data[0] == data[1], "The two PPO recipes must compose the same settings"
    template = OmegaConf.create(
        base.read_text().replace(
            "env/polaris_droid_tapeintocontainer@", "env/isaaclab_arena@"
        )
    )
    OmegaConf.save(
        OmegaConf.merge(template, data[0]), tmp_path / "arena_droid_ppo.yaml"
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "checkpoint"))
    with initialize_config_dir(config_dir=str(tmp_path), version_base="1.1"):
        return compose(config_name="arena_droid_ppo")


@pytest.fixture
def arena_cluster(monkeypatch):
    from unit_tests.test_placement import create_fake_cluster

    cluster = create_fake_cluster(1, 1)
    monkeypatch.setattr("rlinf.config.Cluster", lambda *args, **kwargs: cluster)


def test_isaaclab_arena_eval_with_unused_training(
    arena_config, arena_ppo_config, arena_cluster
):
    # Evaluation samples 15 actions; the retained training recipe samples one.
    OmegaConf.update(
        arena_config, "env.train", arena_ppo_config.env.train, force_add=True
    )
    assert arena_config.rollout.model.num_action_chunks == 15, (
        "Exercise chunked evaluation"
    )
    validate_cfg(arena_config)


def test_isaaclab_arena_train_with_unused_evaluation(arena_ppo_config, arena_cluster):
    cfg = arena_ppo_config
    cfg.env.eval.max_steps_per_rollout_epoch = 15
    assert cfg.runner.val_check_interval == -1, (
        "Exercise training without periodic evaluation"
    )
    validate_cfg(cfg)
    cfg.runner.val_check_interval = 1
    with pytest.raises(ValueError, match="every assigned episode"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("mode", "export_format", "only_success"),
    [
        ("eval", "lerobot_dataset", False),
        ("train", "pickle_episode", True),
    ],
)
def test_isaaclab_arena_rejects_data_collection(
    arena_ppo_config, arena_cluster, mode, export_format, only_success
):
    arena_ppo_config.runner.val_check_interval = 5
    cfg = arena_ppo_config.env[mode]
    OmegaConf.update(
        cfg,
        "data_collection",
        {
            "enabled": True,
            "export_format": export_format,
            "streaming": mode == "eval",
            "only_success": only_success,
        },
        force_add=True,
    )
    with pytest.raises(ValueError, match=r"data_collection\.enabled=false"):
        validate_cfg(arena_ppo_config)
    with pytest.raises(ValueError, match=r"data_collection\.enabled=false"):
        get_env_cls(cfg.env_type)(
            cfg, 1, 0, 1, SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=0)
        )


def test_isaaclab_arena_documented_ppo_config(
    arena_ppo_config, arena_python, arena_cluster
):
    from rlinf.envs.sim.isaaclab_arena.config import validate_arena_config

    cfg = arena_ppo_config
    cfg.runner.val_check_interval = 5
    validate_cfg(cfg)
    for settings in (cfg.env.train, cfg.env.eval):
        with get_env_cls(settings.env_type)(
            settings,
            1,
            0,
            1,
            SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=5),
        ) as env:
            env.reset()
            env.step(torch.zeros(1, 8))
    cfg.env.train.max_episode_steps = None
    with pytest.raises(ValueError, match="positive integer max_episode_steps"):
        validate_arena_config(
            cfg.env.train,
            evaluation=False,
            action_chunk=cfg.actor.model.num_action_chunks,
        )
    cfg.env.train.max_episode_steps = 600
    cfg.env.train.group_size = 2
    cfg.algorithm.adv_type = "grpo"
    with pytest.raises(ValueError, match="independent placements"):
        validate_arena_config(
            cfg.env.train,
            evaluation=False,
            action_chunk=cfg.actor.model.num_action_chunks,
        )
    cfg.env.train.group_size = 1
    cfg.algorithm.adv_type = "gae"
    cfg.actor.model.num_action_chunks = 3
    with pytest.raises(ValueError, match="num_action_chunks=1"):
        validate_arena_config(
            cfg.env.train,
            evaluation=False,
            action_chunk=cfg.actor.model.num_action_chunks,
        )
    cfg.env.train.auto_reset = False
    with pytest.raises(ValueError, match="auto_reset=true for training"):
        validate_arena_config(
            cfg.env.train,
            evaluation=False,
            action_chunk=cfg.actor.model.num_action_chunks,
        )


def test_isaaclab_arena_round_placements_are_independent_of_earlier_steps(
    arena_config, arena_python, monkeypatch
):
    cfg = arena_config.env.eval
    snapshots = []
    # Both isolated workers use local rank zero. One resets mid-episode and
    # the other finishes all lanes before the next placement epoch.
    for gpu, horizon in [(5, 1), (6, 15)]:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(gpu))
        monkeypatch.setenv("ARENA_TEST_GPU", str(gpu))
        with get_env_cls(cfg.env_type)(
            cfg,
            3,
            0,
            2,
            SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=gpu),
        ) as env:
            first, _ = env.reset()
            watchdog = env._watchdog.pid
            env.chunk_step(torch.zeros(3, horizon, 8))
            second, _ = env.reset()
            assert not Path(f"/proc/{watchdog}").exists(), (
                "Rebuilding the pool must reap the previous watchdog"
            )
            watchdog = env._watchdog.pid
            assert not torch.equal(first["main_images"], second["main_images"]), (
                "Each epoch must advance the placement seed"
            )
            snapshots.append(second)
        assert not Path(f"/proc/{watchdog}").exists(), (
            "Closing the environment must reap its watchdog"
        )
    for camera in ("main_images", "wrist_images"):
        torch.testing.assert_close(snapshots[0][camera], snapshots[1][camera])


def test_isaaclab_arena_reset_during_step(arena_config, arena_python, monkeypatch):
    from rlinf.envs.sim.isaaclab_arena import isaaclab_arena_env as adapter

    cfg = arena_config.env.eval
    with (
        get_env_cls(cfg.env_type)(
            cfg, 1, 0, 1, SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=5)
        ) as env,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        env.reset()
        kit = env._process
        started, resume = Event(), Event()
        receive = adapter.receive_message

        def wait_for_reset(*args):
            started.set()
            assert resume.wait(10), "Timed out waiting for the overlapping reset"
            return receive(*args)

        with monkeypatch.context() as patch:
            patch.setattr(adapter, "receive_message", wait_for_reset)
            future = pool.submit(env.chunk_step, torch.zeros(1, 1, 8))
            try:
                assert started.wait(10), "Kit must have an outstanding step request"
                with pytest.raises(RuntimeError, match="already in progress"):
                    env.reset()
                assert env._process is kit and kit.poll() is None, (
                    "The current Kit process must survive"
                )
                assert env._round == 1, (
                    "A rejected reset must preserve the placement round"
                )
            finally:
                resume.set()
            _, _, term, _, _ = future.result(timeout=10)
            assert term.item(), "The original step must complete normally"
        env.reset()
        assert env._round == 2, "The next evaluation round must still be available"


def test_isaaclab_arena_smoke_checks_graphics_device(monkeypatch):
    from e2e_tests.evaluations.isaaclab_arena_smoke import check_kit_devices

    def gpu(uuid, kind):
        return f"<gpu><uuid>{uuid}</uuid><processes><process_info><pid>42</pid><type>{kind}</type></process_info></processes></gpu>"

    inventory = gpu("GPU-A", "C+G")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda args: f"<nvidia_smi_log>{inventory}</nvidia_smi_log>",
    )
    check_kit_devices(42, "GPU-A")
    inventory = gpu("GPU-A", "C") + gpu("GPU-B", "G")
    with pytest.raises(AssertionError, match="graphics"):
        check_kit_devices(42, "GPU-A")


def test_isaaclab_arena_compact_chunk_and_native_budget():
    from unit_tests._isaaclab_arena import Droid

    from rlinf.envs.sim.isaaclab_arena.simulation import ArenaRollout

    outputs = []
    for all_frames in (False, True):
        native = Droid()
        batch = ArenaRollout(native, auto_reset=False, all_frames=all_frames)
        batch.reset(42)
        outputs.append(batch.chunk_step(np.zeros((3, 15, 8), dtype=np.float32)))
        assert native.starts == 3, "Finished lanes must not consume another placement"
    compact, recorded = outputs
    assert len(compact[0]) == 1 and len(recorded[0]) == 15, (
        "Only frame consumers need every step"
    )
    for key in ("states", "main_images", "wrist_images", "active_mask"):
        np.testing.assert_array_equal(compact[0][-1][key], recorded[0][-1][key])
    for index in (1, 2, 3):
        np.testing.assert_array_equal(compact[index], recorded[index])
    np.testing.assert_array_equal(
        compact[-1][-1]["executed_mask"], recorded[-1][-1]["executed_mask"]
    )


def test_isaaclab_arena_same_weight_ppo_ratio(arena_ppo_config):
    from unit_tests._isaaclab_arena import Droid

    from rlinf.algorithms.registry import policy_loss
    from rlinf.data.schema.embodied_types import (
        EnvPart,
        EnvTransition,
        PolicyOutput,
        PolicyPart,
        TrajectoryStep,
    )
    from rlinf.envs.sim.isaaclab_arena.simulation import ArenaRollout
    from rlinf.models.embodiment.openpi.modules.siglip import Encoder1DBlock

    batch = ArenaRollout(Droid(), auto_reset=True)
    batch.reset(0)
    numerics = arena_ppo_config.actor.model.openpi.pytorch_numerics
    assert numerics, (
        "The documented PPO recipe must use the same SigLIP numerics in both modes"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        vision = Encoder1DBlock(32, 4, 64, pytorch_numerics=numerics).to(torch.bfloat16)
        vision.requires_grad_(False)
        actor = torch.nn.Sequential(
            vision, torch.nn.Linear(32, 8, dtype=torch.bfloat16)
        )
        patches = torch.randn(3, 16, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        mean = actor.eval()(patches)[:, :1].float()
        actions = mean + 1.0
        old_logprobs = torch.distributions.Normal(mean, 1.0).log_prob(actions)

    with torch.enable_grad():
        mean = actor.train()(patches)[:, :1].float()
        logprobs = torch.distributions.Normal(mean, 1.0).log_prob(actions)
    assert logprobs.requires_grad, (
        "The actor must replay the rollout action with autograd"
    )
    _, rewards, terms, _, infos = batch.chunk_step(actions.numpy())
    assert terms[0, 0], "The first sampled action should end its episode"
    transition = EnvTransition(
        rewards=torch.from_numpy(rewards),
        executed_mask=torch.from_numpy(infos[-1]["executed_mask"]),
    )
    sampled = PolicyPart(
        sources=[],
        obs={},
        output=PolicyOutput(
            forward_inputs={"action": actions}, prev_logprobs=old_logprobs
        ),
    )
    env = EnvPart(sources=[], transition=transition)
    step = TrajectoryStep.from_parts(
        sampled,
        env,
        rewards=transition.rewards,
        collect_prev_infos=True,
        collect_transitions=False,
        enable_prefix=False,
        include_final_value=False,
    )
    _, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=8,
        logprobs=logprobs,
        old_logprobs=step.prev_logprobs,
        advantages=torch.ones(3, 1),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert torch.equal(logprobs, step.prev_logprobs), (
        "Every replayed log probability must match exactly"
    )
    assert metrics["actor/ratio"] == 1.0, (
        "Unchanged policy weights must give a PPO ratio of exactly one"
    )


@pytest.mark.parametrize(
    ("checkout", "valid"), [("new", True), ("pinned", True), ("other-revision", False)]
)
def test_isaaclab_arena_install_contract(tmp_path, checkout, valid):
    root = Path(__file__).resolve().parents[2]
    pin = "a72262a0f1896f56bb3a1fdf329a1548ba55aad5"
    source = tmp_path / "arena-src"
    if checkout != "new":
        source.mkdir()
        (source / "revision").write_text(pin if valid else "main")
    uv = tmp_path / "uv"
    uv.write_text(
        textwrap.dedent("""\
        #!/bin/bash
        set -eu
        case "$1 $2" in
          'python find') test -f "$TEST_ROOT/managed-3.12" ;;
          'python install')
            if [[ "$3" == --help ]]; then echo --no-bin; exit; fi
            test "${!#}" = 3.12 && touch "$TEST_ROOT/managed-3.12" ;;
          'sync --project')
            test "$UV_PROJECT_ENVIRONMENT" = "$TEST_ROOT/controller/arena-runtime"
            test -z "${UV_TORCH_BACKEND:-}" && test "${!#}" = 3.12
            touch "$TEST_ROOT/synced" ;;
          *) exit 2 ;;
        esac
    """)
    )
    uv.chmod(0o755)
    script = textwrap.dedent("""\
        source requirements/install.sh
        ENV_NAME=isaaclab_arena TORCH_VERSION=2.7.1 USE_MIRRORS=0
        unset UV_TORCH_BACKEND ISAACLAB_ARENA_PYTHON
        configure_nvidia
        test "$PLATFORM_TORCH_STR" = +cu128
        test "$PLATFORM_TORCH_INDEX" = https://download.pytorch.org/whl/cu128
        test "$UV_TORCH_BACKEND" = cu128
        test "${PLATFORM_TORCH_PACKAGES[*]}" = 'torch torchvision torchaudio'
        export UV_PYTHON_DOWNLOADS=manual VIRTUAL_ENV="$TEST_ROOT/controller"
        VENV_DIR="$VIRTUAL_ENV"
        git_clone_with_retry() { mkdir -p "${!#}"; echo main > "${!#}/revision"; }
        git() {
          local dir="$2"; shift 2
          case "$1" in
            checkout) test "$2" = --detach && echo "$3" > "$dir/revision" ;;
            rev-parse) cat "$dir/revision" ;;
            -c) test "$3" = submodule ;;
            *) return 2 ;;
          esac
        }
        install_isaaclab_arena_env
    """)
    result = subprocess.run(
        ["bash", "-ec", script],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "TEST_ROOT": str(tmp_path),
            "ISAACLAB_ARENA_PATH": str(source),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == valid, result.stderr
    assert (tmp_path / "synced").exists() == valid, (
        "Sync only the pinned checkout with managed Python 3.12"
    )
    assert (source / "revision").read_text().strip() == (pin if valid else "main"), (
        "An existing user checkout must retain its revision"
    )
    if not valid:
        assert pin in result.stderr, (
            "Report the expected revision of an existing checkout"
        )


@pytest.mark.parametrize("candidate", ["directory", "non-python", "runtime"])
def test_isaaclab_arena_external_interpreter(tmp_path, candidate):
    runtime = tmp_path if candidate == "directory" else Path("/bin/true")
    if candidate == "runtime":
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(tmp_path / "empty")],
            check=True,
        )
        runtime = tmp_path / "empty/bin/python"
    result = subprocess.run(
        [
            "bash",
            "-ec",
            "source requirements/install.sh; PLATFORM=nvidia; install_isaaclab_arena_env",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "ISAACLAB_ARENA_PYTHON": str(runtime)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Python 3.12" in result.stderr, (
        "Reject an invalid external runtime during installation"
    )


def test_isaaclab_arena_wire_arrays_load_with_openpi_numpy():
    import io
    import pickle
    import socket

    from rlinf.envs.sim.isaaclab_arena.runtime import send_message

    class OpenPIUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy._core.numeric":
                raise ModuleNotFoundError("NumPy 1.26 has no numpy._core.numeric")
            return super().find_class(module, name)

    action = np.arange(8, dtype=np.float32)
    sender, receiver = socket.socketpair()
    with sender, receiver, receiver.makefile("rb") as stream:
        send_message(sender, action)
        size = int.from_bytes(stream.read(8), "big")
        decoded = OpenPIUnpickler(io.BytesIO(stream.read(size))).load()
    np.testing.assert_array_equal(decoded, action)


def test_isaaclab_arena_kit_paths(tmp_path, monkeypatch):
    from rlinf.envs.sim.isaaclab_arena import runtime

    root = tmp_path / "isaacsim"
    apps = root / "apps"
    apps.mkdir(parents=True)
    kit = tmp_path / "separate-site/omni/kit_app"
    kit.mkdir(parents=True)
    (kit / "__init__.py").touch()
    monkeypatch.setattr(
        runtime,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(root / "__init__.py"))
        if name == "isaacsim"
        else SimpleNamespace(submodule_search_locations=[str(kit.parent)]),
    )
    icd = tmp_path / "nvidia.json"
    icd.touch()
    manifest_dir = tmp_path / "drivers"
    manifest_dir.mkdir()
    (manifest_dir / "vendor.json").touch()
    monkeypatch.setenv("VK_DRIVER_FILES", f"{icd}:{manifest_dir}:/missing/icd.json")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/missing/icd.json")
    for name in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
        monkeypatch.setenv(name, str(tmp_path))
    runtime._prepare_kit_environment()
    assert not {
        "ISAAC_PATH",
        "EXP_PATH",
        "CARB_APP_PATH",
        "VK_ICD_FILENAMES",
    }.intersection(os.environ), (
        "Foreign Kit paths and missing ICD overrides must be cleared"
    )
    assert os.environ["VK_DRIVER_FILES"] == f"{icd}:{manifest_dir}", (
        "Keep valid driver overrides"
    )
    paths = {"ISAAC_PATH": str(root), "EXP_PATH": str(apps), "CARB_APP_PATH": str(kit)}
    for name, path in paths.items():
        monkeypatch.setenv(name, path)
    runtime._prepare_kit_environment()
    assert all(os.environ[name] == path for name, path in paths.items()), (
        "Keep paths belonging to this Kit installation"
    )


def _wait_for_arena_processes(pidfile):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if pidfile.exists() and len(pids := pidfile.read_text().split()) == 2:
            return [int(pid) for pid in pids]
        time.sleep(0.02)
    pytest.fail("The Arena process group did not reach the injected fault")


def _assert_arena_exited(pids):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        running = []
        for pid in pids:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
            except FileNotFoundError:
                continue
            if state != "Z":
                running.append(pid)
        if not running:
            return
        time.sleep(0.02)
    pytest.fail(f"Arena processes still running after shutdown: {running}")


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("partial", TimeoutError),
        ("partial-eof", RuntimeError),
        ("no-response", TimeoutError),
        ("blocked-send", TimeoutError),
        ("interrupt", KeyboardInterrupt),
        ("leader-exit", None),
    ],
)
def test_isaaclab_arena_interrupted_exchange(
    arena_python, arena_config, monkeypatch, tmp_path, mode, error
):
    cfg = arena_config.env.eval
    pidfile = tmp_path / "fault-pids"
    monkeypatch.setenv("ARENA_FAULT", mode)
    monkeypatch.setenv("ARENA_FAULT_PIDS", str(pidfile))
    env = get_env_cls(cfg.env_type)(
        cfg, 1, 0, 1, SimpleNamespace(accelerator_type="NV_GPU", accelerator_rank=5)
    )
    try:
        pids = _wait_for_arena_processes(pidfile)
        if mode == "leader-exit":
            env._process.wait(timeout=10)
        else:
            env._request_timeout = 0.5
            started = time.monotonic()
            with pytest.raises(error):
                if mode == "blocked-send":
                    env.chunk_step(np.zeros((1, 262144, 8), dtype=np.float32))
                else:
                    env.reset()
            assert time.monotonic() - started < 5, (
                "The deadline must cover the whole exchange"
            )
            with pytest.raises(RuntimeError, match="closed"):
                env.reset()
    finally:
        env.close()
    _assert_arena_exited(pids)


@pytest.mark.parametrize("phase", ["startup", "no-response"])
def test_isaaclab_arena_owner_death(arena_python, tmp_path, monkeypatch, phase):
    from rlinf.envs.sim.isaaclab_arena import runtime

    pidfile = tmp_path / "fault-pids"
    monkeypatch.setenv("ARENA_FAULT", phase)
    monkeypatch.setenv("ARENA_FAULT_PIDS", str(pidfile))
    owner = textwrap.dedent("""
        import os, runpy, signal, socket, subprocess, sys
        protocol = runpy.run_path(sys.argv[1])
        parent, child = socket.socketpair()
        kit = subprocess.Popen([os.environ['ISAACLAB_ARENA_PYTHON'], '-I', sys.argv[1], str(child.fileno())], pass_fds=(child.fileno(),), start_new_session=True)
        watchdog = protocol['start_watchdog'](kit.pid)
        child.close()
        protocol['send_message'](parent, {})
        if os.environ['ARENA_FAULT'] != 'startup':
            protocol['receive_message'](parent)
            protocol['send_message'](parent, ('chunk_step', None))
        signal.pause()
    """)
    with subprocess.Popen([sys.executable, "-c", owner, runtime.__file__]) as process:
        try:
            pids = _wait_for_arena_processes(pidfile)
        finally:
            process.send_signal(signal.SIGKILL)
    _assert_arena_exited(pids)
