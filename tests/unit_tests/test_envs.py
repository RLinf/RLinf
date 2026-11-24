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

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.env.env_worker as env_worker_module
from rlinf.config import (
    resolve_env_mode_cfg,
    validate_embodied_cfg,
    validate_env_policy_fields,
    validate_env_training_cfg,
    validate_eval_env_cfg,
)
from rlinf.data.schema.embodied_types import PolicyInput
from rlinf.envs.layout import EnvLayout
from rlinf.envs.policy_obs import PolicyObsSpec
from rlinf.utils.eef_rot6d import EEFNormalizer, decode_eef_actions
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.torch_rot6d import canonicalize_rot6d, rot6d_to_matrix
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
def eef_stats(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        "q01": torch.linspace(-1.5, -0.5, 10).tolist(),
                        "q99": torch.linspace(0.8, 2.0, 10).tolist(),
                    }
                    for key in ("state", "actions")
                }
            }
        )
    )
    return str(path)


def _raw_actions(batch=3, steps=2):
    actions = torch.randn(batch, steps, 10) * 0.01
    actions[..., 3] += 1
    actions[..., 7] += 1
    actions[..., 9] = -1
    return actions


def test_eef_decoder_matches_dataset_output_and_normalization(eef_stats):
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
    )

    raw = _raw_actions()
    normalizer = EEFNormalizer(eef_stats)
    normalized = normalizer.normalize(raw, "actions")
    decoded = decode_eef_actions(normalizer.unnormalize(normalized, "actions"))
    expected = ManiSkillEEFRot6dOutputs()({"actions": raw.numpy()})["actions"]
    torch.testing.assert_close(decoded, torch.tensor(expected), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(raw, normalizer.unnormalize(normalized, "actions"))


@pytest.mark.parametrize("pitch", [-math.pi / 2, -0.1, 0.0, 0.1, math.pi / 2])
def test_eef_decoder_intrinsic_xyz_round_trip(pitch):
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    matrix = rotation.from_euler("XYZ", [0.08, pitch, -0.04]).as_matrix()
    raw = torch.zeros(1, 10, dtype=torch.float64)
    raw[:, 3:9] = torch.tensor(np.concatenate((matrix[:, 0], matrix[:, 1])))
    command = decode_eef_actions(raw)
    actual = rotation.from_euler("XYZ", command[0, 3:6].numpy() * -0.1).as_matrix()
    np.testing.assert_allclose(actual, matrix, atol=1e-6)


def test_eef_heads_ignore_redundant_rotation_coordinates(eef_stats):
    from rlinf.models.embodiment.mlp_policy.prefix_mlp_policy import PrefixPolicy

    cfg = {"hidden_dims": [16, 16], "norm_stats_path": eef_stats}
    policy = PrefixPolicy(
        prefix_dim=4,
        proprio_dim=10,
        action_dim=10,
        num_action_chunks=2,
        ref_num_action_chunks=2,
        actor_head_cfg=OmegaConf.create(
            {**cfg, "name": "eef_rot6d_residual_mlp", "edit_scale": 0.02}
        ),
        critic_head_cfg=OmegaConf.create(
            {**cfg, "name": "eef_rot6d_multi_q_mlp", "num_q_heads": 4}
        ),
    )
    # Exercise the learned conditioning path, beyond the zero-initialized editor.
    torch.nn.init.normal_(policy.actor_head.editor[-1].weight, std=0.1)
    normalizer = EEFNormalizer(eef_stats)
    raw = _raw_actions()
    equivalent = raw.clone()
    equivalent[..., 3:6] *= 2
    equivalent[..., 6:9] = 3 * raw[..., 6:9] + raw[..., 3:6]
    a = normalizer.normalize(raw, "actions")
    b = normalizer.normalize(equivalent, "actions")
    visual = torch.randn(3, 4)
    obs_a = {
        "z_rl": visual,
        "proprio": normalizer.normalize(raw[:, 0], "state"),
        "ref_chunk": a,
    }
    obs_b = {
        "z_rl": visual,
        "proprio": normalizer.normalize(equivalent[:, 0], "state"),
        "ref_chunk": b,
    }
    out_a, lp_a, _ = policy.sac_forward(obs_a, deterministic=True)
    out_b, lp_b, _ = policy.sac_forward(obs_b, deterministic=True)
    torch.testing.assert_close(
        out_a - a.flatten(1), out_b - b.flatten(1), atol=1e-6, rtol=1e-4
    )
    torch.testing.assert_close(lp_a, lp_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        policy.sac_q_forward(obs_a, a),
        policy.sac_q_forward(obs_b, b),
        atol=1e-6,
        rtol=1e-5,
    )
    assert (out_a - a.flatten(1)).abs().max() <= 0.020001
    loss = -policy.sac_q_forward(obs_a, out_a).mean()
    loss.backward()
    assert policy.actor_head.editor[-1].weight.grad.isfinite().all()
    assert policy.actor_head.editor[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "settings",
    [
        {"gap_m": 0},
        {"gap_m": -0.1},
        {"lateral_jitter_m": float("nan")},
        {"rotation_jitter_deg": 11},
    ],
)
def test_preinsert_validates_initial_distribution(settings):
    pytest.importorskip("mani_skill")
    from rlinf.envs.sim.maniskill.peg_preinsert import PegPreinsertConfig

    with pytest.raises(ValueError):
        PegPreinsertConfig(**settings)


def test_rot6d_degenerate_frames_have_finite_right_handed_rotations():
    vectors = torch.tensor(
        [[0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 2, 0, 0]], requires_grad=True
    )
    matrix = rot6d_to_matrix(vectors)
    torch.testing.assert_close(
        matrix.transpose(-1, -2) @ matrix, torch.eye(3).expand(2, -1, -1)
    )
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(2))
    canonicalize_rot6d(vectors).sum().backward()
    assert vectors.grad.isfinite().all()


def test_preinsert_prefix_config_composes_and_builds_expo(eef_stats, monkeypatch):
    from hydra import compose, initialize_config_dir

    from rlinf.algorithms.prefix_off_policy import build_prefix_off_policy_algorithm
    from rlinf.models.embodiment.mlp_policy.prefix_mlp_policy import get_model

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_prefix_expo_openpi_pi05_eef_preinsert_async"
        )
    for head in ("actor_head", "critic_head"):
        cfg.actor.model[head].norm_stats_path = eef_stats
    from rlinf.config import validate_embodied_cfg

    validate_embodied_cfg(cfg)
    policy = get_model(cfg.actor.model)
    algorithm = build_prefix_off_policy_algorithm(cfg)
    assert cfg.env.train.init_params.preinsert.gap_m == 0.05
    assert cfg.env.train.init_params.hole_clearance == 0.001
    assert cfg.env.train.max_episode_steps == 100
    assert cfg.env.train.reward_mode == "only_success"
    assert cfg.env.eval.video_cfg.save_video
    assert cfg.rollout.prefix_feature_model.prefix.action_space == "normalized_model"
    assert policy.flat_action_dim == 100
    assert algorithm.target_entropy == pytest.approx(100 * (math.log(0.02) - 0.5))
    raw = _raw_actions(batch=2, steps=10)
    normalizer = EEFNormalizer(eef_stats)
    obs = {
        "z_rl": torch.zeros(2, 2048),
        "proprio": normalizer.normalize(raw[:, 0], "state"),
        "ref_chunk": normalizer.normalize(raw, "actions"),
    }
    obs["candidates"] = obs["ref_chunk"][:, None].expand(-1, 8, -1, -1)
    action, _ = algorithm.select_actions(policy, policy, obs)
    assert action.shape == (2, 100)
    assert torch.isfinite(action).all()


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics, renderer, and GPU IK assets.",
)
@pytest.mark.parametrize("seed", [420000, 420002])
def test_preinsert_gpu_reset_normalized_execution_and_replay(eef_stats, seed):
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from toolkits.lerobot.collect_maniskill_peg_preinsert import (
        encode_action,
        expert_action,
    )

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/maniskill_peg_preinsert.yaml"
    )
    if os.environ.get("RLINF_TEST_RENDER_BACKEND"):
        cfg.init_params.render_backend = os.environ["RLINF_TEST_RENDER_BACKEND"]
    cfg.normalized_actions_stats_path = eef_stats
    cfg.video_cfg.save_video = False
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    normalizer = EEFNormalizer(eef_stats).to(env.device)
    try:
        task = env.env.unwrapped
        obs, info = env.reset(seed=seed, options={"reconfigure": True})
        initial = obs["states"].clone()
        head = (task.box_hole_pose.inv() * task.peg_head_pose).p
        torch.testing.assert_close(
            -task.peg_half_sizes[:, 0] - head[:, 0],
            torch.full((1,), 0.05, device=env.device),
            atol=2e-6,
            rtol=0,
        )
        assert head[:, 1:].abs().max() <= 0.030001
        assert task.hole_clearance == 0.001
        assert not info["success"].any()
        assert obs["states"].shape == (1, 10)
        assert "privileged_state" not in obs
        commands = []
        labels = []
        for _ in range(100):
            label = torch.tensor(
                encode_action(task, expert_action(task)), device=env.device
            )[None]
            labels.append(label.cpu().numpy())
            normalized = normalizer.normalize(label, "actions")
            commands.append(normalized.clone())
            result = env.chunk_step(normalized[:, None])
            if result[2].any() or result[3].any():
                break
        success = bool(result[-1][-1]["success"].all())
        if seed == 420000:
            assert success
        final = result[0][-1]["states"].clone()
        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
        torch.testing.assert_close(obs["states"], initial, atol=2e-5, rtol=0)
        for command in commands:
            result = env.chunk_step(command[:, None])
        assert bool(result[-1][-1]["success"].all()) == success
        torch.testing.assert_close(result[0][-1]["states"], final, atol=0.002, rtol=0)
    finally:
        env.env.close()

    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
    )

    cfg.normalized_actions_stats_path = None
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
        torch.testing.assert_close(obs["states"], initial, atol=2e-5, rtol=0)
        decoder = ManiSkillEEFRot6dOutputs()
        for label in labels:
            command = torch.tensor(
                decoder({"actions": label})["actions"], device=env.device
            )
            result = env.chunk_step(command[:, None])
        assert bool(result[-1][-1]["success"].all()) == success
        torch.testing.assert_close(result[0][-1]["states"], final, atol=0.002, rtol=0)
    finally:
        env.env.close()


def test_preinsert_sft_config_matches_eef_data_contract(monkeypatch):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/sft/config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_peg_preinsert_eef_sft_openpi_pi05")
    assert cfg.runner.max_steps == 2000
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.model.action_dim == cfg.actor.model.openpi.action_env_dim == 10
    assert cfg.actor.model.openpi.config_name == "pi05_maniskill_eef_rot6d"
    assert not cfg.actor.model.openpi.use_rlt
    assert cfg.actor.model.prefix.pool == "masked_mean"
    assert cfg.actor.optim.total_training_steps == 3000
    assert cfg.actor.optim.lr_warmup_steps == 1000
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    data = get_openpi_config(cfg.actor.model.openpi.config_name).data
    assert data.eef_rot6d and data.output_action_dim == 10
    assert not data.extra_delta_transform
