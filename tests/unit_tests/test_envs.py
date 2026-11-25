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


def _write_robouniverse_task_zip(
    archive_path,
    *,
    case: str,
    robot: str,
    action_dim: int,
    instruction: str,
    verdict_mode: str = "threshold",
    action_lower: list[float] | None = None,
    action_upper: list[float] | None = None,
    strict_bounds: bool = False,
    init_sleep_s: float = 0.0,
) -> str:
    """Create a real operable-env/0.1 CPU fixture archive and return its SHA-256."""
    import hashlib
    import json
    import zipfile

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    joint_names = [f"joint_{i}" for i in range(action_dim)]
    lower = action_lower if action_lower is not None else [-1.0] * action_dim
    upper = action_upper if action_upper is not None else [1.0] * action_dim
    spec = {
        "delivery_api": "operable-env/0.1",
        "status": "ready",
        "task_id": case,
        "revision": "1",
        "instruction_path": "task_instruction.md",
        "backend_entrypoint": "env_backend:create_backend",
        "evaluator_entrypoint": "evaluator_adapter:evaluate",
        "timing": {"physics_dt_s": 1.0 / 120.0, "control_dt_s": 0.05},
        "action": {
            "kind": "joint_position_target",
            "names": joint_names,
            "units": ["rad"] * action_dim,
            "lower": lower,
            "upper": upper,
            "gripper_mapping": "fixture",
        },
        "observation": {
            "joint_names": joint_names,
            "joint_units": ["rad"] * action_dim,
            "cameras": [
                {"id": "external", "role": "external", "width": 8, "height": 6},
                {"id": "wrist", "role": "wrist", "width": 8, "height": 6},
            ],
        },
        "reset": {"mode": "fixed", "supported_seeds": [0]},
    }
    env_backend_src = f"""
import json
import time
from pathlib import Path
import numpy as np

class FixtureBackend:
    def __init__(self, workspace, runtime_root, device):
        if {init_sleep_s} > 0:
            time.sleep({init_sleep_s})
        self.workspace = Path(workspace)
        self.dim = {action_dim}
        self.lower = np.asarray({lower!r}, dtype=np.float64)
        self.upper = np.asarray({upper!r}, dtype=np.float64)
        self.strict_bounds = {strict_bounds!r}
        self.step_index = 0
        self.last_action = np.zeros(self.dim, dtype=np.float64)

    def reset(self, *, seed=0):
        if seed != 0:
            raise ValueError(f"Expected seed 0, found {{seed}}")
        self.step_index = 0
        self.last_action = np.zeros(self.dim, dtype=np.float64)
        return self._obs()

    def _obs(self):
        ext = np.full((6, 8, 3), self.step_index % 255, dtype=np.uint8)
        wrist = np.full((6, 8, 3), (self.step_index + 10) % 255, dtype=np.uint8)
        return {{
            "episode_id": "{case}",
            "step_index": self.step_index,
            "sim_time_s": self.step_index * 0.05,
            "rgb": {{"external": ext, "wrist": wrist}},
            "proprio": {{
                "joint_position": self.last_action.copy(),
                "joint_velocity": np.ones(self.dim, dtype=np.float64) * 0.5,
            }},
        }}

    def step(self, action):
        arr = np.asarray(action, dtype=np.float64)
        if self.strict_bounds and (np.any(arr < self.lower) or np.any(arr > self.upper)):
            raise ValueError("action outside float64 joint limits")
        self.last_action = arr.copy()
        self.step_index += 1
        return self._obs()

    def export_evidence(self, destination):
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "trace.json").write_text(
            json.dumps({{"case": "{case}", "last_sum": float(self.last_action.sum())}})
        )
        return {{"files": ["trace.json"]}}

    def close(self):
        pass

def create_backend(*, workspace, runtime_root, device):
    return FixtureBackend(workspace, runtime_root, device)
"""
    if verdict_mode == "error":
        eval_src = """
def evaluate(*, run_dir):
    raise RuntimeError("simulated verifier crash")
"""
    elif verdict_mode == "always_fail":
        eval_src = """
def evaluate(*, run_dir):
    return {"verdict": "failure", "details": {"mode": "always_fail"}}
"""
    else:
        eval_src = """
import json
from pathlib import Path

def evaluate(*, run_dir):
    data = json.loads((Path(run_dir) / "trace.json").read_text())
    verdict = "success" if data["last_sum"] > 0.5 else "failure"
    return {"verdict": verdict, "details": data}
"""
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("release.json", json.dumps({"case": case, "robot": robot}))
        zf.writestr("upstream.json", json.dumps({}))
        zf.writestr("environment/sim/workspace/env_spec.json", json.dumps(spec))
        zf.writestr("environment/sim/workspace/task_instruction.md", instruction)
        zf.writestr("environment/sim/workspace/env_backend.py", env_backend_src)
        zf.writestr("environment/sim/workspace/evaluator_adapter.py", eval_src)
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def _build_robouniverse_fixture_dataset(root):
    import csv

    entries = [
        ("franka_01", "franka", 9, "Rotate the mug", "threshold", True),
        ("franka_02", "franka", 9, "Press the button", "always_fail", True),
        ("franka_err", "franka", 9, "Broken grader", "error", False),
        ("agilex_01", "agilex", 16, "Place peach in basket", "threshold", True),
    ]
    rows = []
    heldout = []
    for case, robot, dim, instr, mode, in_heldout in entries:
        rel = f"tasks/{robot}/{case}.zip"
        sha = _write_robouniverse_task_zip(
            root / rel,
            case=case,
            robot=robot,
            action_dim=dim,
            instruction=instr,
            verdict_mode=mode,
        )
        rows.append(
            {
                "case": case,
                "robot": robot,
                "archive": rel,
                "sha256": sha,
                "bytes": str((root / rel).stat().st_size),
            }
        )
        if in_heldout:
            heldout.append({"case": case})

    with (root / "catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case", "robot", "archive", "sha256", "bytes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    (root / "splits").mkdir(parents=True, exist_ok=True)
    with (root / "splits" / "heldout500.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["case"])
        writer.writeheader()
        writer.writerows(heldout)
    return root


def test_robouniverse_catalog_filtering_and_sharding(tmp_path):
    from rlinf.envs.sim.robouniverse.catalog import RoboUniverseCatalog

    data_root = _build_robouniverse_fixture_dataset(tmp_path / "dataset")
    catalog = RoboUniverseCatalog(data_root)

    heldout_franka = catalog.select(split="heldout500", robot="franka")
    assert [r.case for r in heldout_franka] == ["franka_01", "franka_02"]

    train_tasks = catalog.select(split="train", robot="franka")
    assert [r.case for r in train_tasks] == ["franka_err"]

    explicit = catalog.select(cases=["agilex_01"])
    assert [r.case for r in explicit] == ["agilex_01"]

    assert [r.case for r in catalog.shard(heldout_franka, rank=0, world_size=2)] == [
        "franka_01"
    ]
    assert [r.case for r in catalog.shard(heldout_franka, rank=1, world_size=2)] == [
        "franka_02"
    ]
    assert [r.case for r in catalog.shard(explicit, rank=2, world_size=3)] == [
        "agilex_01"
    ]

    with pytest.raises(ValueError, match="Unknown RoboUniverse cases"):
        catalog.select(cases=["missing_case"])


def test_robouniverse_materializer_checksum_traversal_and_lru(tmp_path):
    import zipfile

    from rlinf.envs.sim.robouniverse.catalog import RoboUniverseCatalog, TaskRecord
    from rlinf.envs.sim.robouniverse.materializer import TaskMaterializer

    data_root = _build_robouniverse_fixture_dataset(tmp_path / "dataset")
    catalog = RoboUniverseCatalog(data_root)
    franka_tasks = catalog.select(split="heldout500", robot="franka")

    mat = TaskMaterializer(data_root, tmp_path / "work", max_cached_tasks=1)
    first = mat.materialize(franka_tasks[0])
    assert first.instruction == "Rotate the mug"
    assert first.workspace_dir.is_dir()

    second = mat.materialize(franka_tasks[1])
    assert second.instruction == "Press the button"
    assert not first.task_dir.exists()
    assert second.task_dir.exists()

    bad_record = TaskRecord(
        case=franka_tasks[0].case,
        robot=franka_tasks[0].robot,
        archive=franka_tasks[0].archive,
        sha256="0" * 64,
        size_bytes=franka_tasks[0].size_bytes,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mat.materialize(bad_record)

    evil_zip = data_root / "tasks" / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escape.txt", "boom")
    evil_record = TaskRecord(
        case="evil",
        robot="franka",
        archive="tasks/evil.zip",
        sha256="0" * 64,
        size_bytes=evil_zip.stat().st_size,
    )
    unsafe_mat = TaskMaterializer(
        data_root, tmp_path / "work_evil", verify_sha256=False
    )
    with pytest.raises(ValueError, match="safe relative path"):
        unsafe_mat.materialize(evil_record)


def test_robouniverse_env_end_to_end_cpu_fixture(tmp_path):
    import sys

    from rlinf.envs import SupportedEnvType, get_env_cls
    from rlinf.envs.action_utils import prepare_actions

    data_root = _build_robouniverse_fixture_dataset(tmp_path / "dataset")
    assert SupportedEnvType("robouniverse") is SupportedEnvType.ROBOUNIVERSE
    env_cls = get_env_cls("robouniverse")

    prepared = prepare_actions(
        raw_chunk_actions=np.ones((1, 2, 16), dtype=np.float32),
        env_type="robouniverse",
        model_type="openpi",
        num_action_chunks=2,
        action_dim=9,
    )
    assert prepared.shape == (1, 2, 9)
    with pytest.raises(ValueError, match="finite values"):
        prepare_actions(
            raw_chunk_actions=np.full((1, 2, 9), np.nan, dtype=np.float32),
            env_type="robouniverse",
            model_type="openpi",
            num_action_chunks=2,
            action_dim=9,
        )

    base_cfg = OmegaConf.create(
        {
            "env_type": "robouniverse",
            "seed": 7,
            "group_size": 1,
            "max_episode_steps": 3,
            "auto_reset": False,
            "ignore_terminations": False,
            "init_params": {
                "data_root": str(data_root),
                "split": "heldout500",
                "robot": "franka",
                "work_dir": str(tmp_path / "work_eval"),
                "sim_python": sys.executable,
                "device": "cpu",
                "clip_actions": True,
                "allow_cpu_fixture": True,
                "accept_isaac_eula": False,
            },
        }
    )

    # Without allow_cpu_fixture or accept_isaac_eula, construction fails fast.
    eula_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    eula_cfg.init_params.allow_cpu_fixture = False
    with pytest.raises(ValueError, match="accept_isaac_eula"):
        env_cls(cfg=eula_cfg, num_envs=1, seed_offset=0, total_num_processes=1)

    # Mixed robot families in one env group must fail loudly.
    mixed_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    mixed_cfg.init_params.robot = None
    with pytest.raises(ValueError, match="single robot family"):
        env_cls(cfg=mixed_cfg, num_envs=1, seed_offset=0, total_num_processes=1)

    env = env_cls(cfg=base_cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    try:
        obs, info = env.reset()
        assert info["cases"] == ["franka_01"]
        assert obs["main_images"].shape == (1, 6, 8, 3)
        assert obs["wrist_images"].shape == (1, 6, 8, 3)
        assert obs["states"].shape == (1, 9)
        assert obs["task_descriptions"] == ["Rotate the mug"]

        # Compatible with PolicyObsSpec resizing.
        spec = PolicyObsSpec(
            keys=["main_images", "wrist_images", "states", "task_descriptions"],
            image_size=[4, 4],
        )
        conformed = spec.apply(obs)
        assert conformed["main_images"].shape == (1, 4, 4, 3)

        # Exceeding bounds clips cleanly when clip_actions=True and grades success on step 3.
        _, r1, term1, trunc1, _ = env.step(np.full((1, 9), 2.0, dtype=np.float32))
        assert (r1.item(), term1.item(), trunc1.item()) == (0.0, False, False)

        _, rewards, terms, truncs, infos = env.chunk_step(
            np.full((1, 2, 9), 0.5, dtype=np.float32)
        )
        assert rewards.shape == (1, 2)
        assert rewards[0, 1].item() == pytest.approx(1.0)
        assert terms[0, 1].item() is True
        assert truncs[0, 1].item() is False
        assert infos[-1]["episode"]["success_once"].item() == pytest.approx(1.0)
        assert infos[-1]["verdicts"] == ["success"]

        # Second reset advances to franka_02 and evicts franka_01's workspace modules.
        obs2, info2 = env.reset()
        assert info2["cases"] == ["franka_02"]
        assert obs2["task_descriptions"] == ["Press the button"]
        _, rewards2, terms2, truncs2, infos2 = env.chunk_step(
            np.full((1, 3, 9), 0.5, dtype=np.float32)
        )
        assert rewards2[0, -1].item() == pytest.approx(0.0)
        assert terms2[0, -1].item() is False
        assert truncs2[0, -1].item() is True
        assert infos2[-1]["verdicts"] == ["failure"]
    finally:
        env.close()
        env.close()  # idempotent


def test_robouniverse_env_auto_reset_and_infra_error_isolation(tmp_path):
    import sys

    from rlinf.envs import get_env_cls

    data_root = _build_robouniverse_fixture_dataset(tmp_path / "dataset")
    env_cls = get_env_cls("robouniverse")
    cfg = OmegaConf.create(
        {
            "env_type": "robouniverse",
            "seed": 0,
            "group_size": 1,
            "max_episode_steps": 2,
            "auto_reset": True,
            "ignore_terminations": False,
            "init_params": {
                "data_root": str(data_root),
                "cases": ["franka_err", "franka_01"],
                "work_dir": str(tmp_path / "work_auto"),
                "sim_python": sys.executable,
                "device": "cpu",
                "allow_cpu_fixture": True,
            },
        }
    )

    env = env_cls(cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    try:
        obs, _ = env.reset()
        assert obs["task_descriptions"] == ["Broken grader"]

        obs_list, rewards, terms, truncs, infos = env.chunk_step(
            np.zeros((1, 2, 9), dtype=np.float32)
        )
        assert rewards[0, -1].item() == pytest.approx(0.0)
        assert terms[0, -1].item() is False
        assert truncs[0, -1].item() is True
        assert infos[-1]["final_info"]["episode"][
            "infra_error"
        ].item() == pytest.approx(1.0)
        assert infos[-1]["final_info"]["verdicts"] == ["error"]
        assert infos[-1]["_final_observation"].tolist() == [True]
        # Auto-reset immediately moved the slot onto the next task.
        assert obs_list[-1]["task_descriptions"] == ["Rotate the mug"]
    finally:
        env.close()


def test_robouniverse_float64_clipping_and_envworker_invariants(tmp_path):
    import csv
    import sys

    from rlinf.envs import get_env_cls

    # Non-power-of-2 float64 limits (e.g. +-2.89, +-0.05) overflow when clipped
    # in float32 and widened to float64.
    data_root = tmp_path / "dataset_bounds"
    rel = "tasks/franka/franka_bounds.zip"
    lower = [-2.89, -1.76, -2.89, -3.07, -2.89, -0.0175, -2.89, -0.05, -0.05]
    upper = [2.89, 1.76, 2.89, -0.0698, 2.89, 3.75, 2.89, 0.05, 0.05]
    sha = _write_robouniverse_task_zip(
        data_root / rel,
        case="franka_bounds",
        robot="franka",
        action_dim=9,
        instruction="Clamp strictly in float64",
        verdict_mode="threshold",
        action_lower=lower,
        action_upper=upper,
        strict_bounds=True,
    )
    with (data_root / "catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case", "robot", "archive", "sha256", "bytes"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": "franka_bounds",
                "robot": "franka",
                "archive": rel,
                "sha256": sha,
                "bytes": str((data_root / rel).stat().st_size),
            }
        )

    env_cls = get_env_cls("robouniverse")
    clip_cfg = OmegaConf.create(
        {
            "env_type": "robouniverse",
            "seed": 0,
            "group_size": 1,
            "max_episode_steps": 2,
            "auto_reset": True,
            "ignore_terminations": False,
            "init_params": {
                "data_root": str(data_root),
                "cases": ["franka_bounds"],
                "work_dir": str(tmp_path / "work_bounds"),
                "sim_python": sys.executable,
                "device": "cpu",
                "clip_actions": True,
                "allow_cpu_fixture": True,
            },
        }
    )
    env = env_cls(cfg=clip_cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    try:
        env.reset()
        # Horizon 4 > max_episode_steps 2 exercises mid-chunk auto_reset alignment
        # and out-of-range actions exercise float64 boundary clamping.
        _, _, terms, truncs, infos = env.chunk_step(
            np.full((1, 4, 9), 10.0, dtype=np.float32)
        )
        assert infos[-1]["final_info"]["episode"]["infra_error"].item() == 0.0
        assert infos[-1]["final_info"]["verdicts"] == ["success"]
        dones_last = (terms | truncs)[:, -1]
        assert dones_last.tolist() == [True]
        assert terms[:, :-1].any().item() is False
        assert truncs[:, :-1].any().item() is False
        assert infos[-1]["final_info"]["episode"]["success_once"][
            dones_last
        ].tolist() == [1.0]
    finally:
        env.close()

    # ignore_terminations=True converts success terminations into truncations on
    # the final chunk step so EnvWorker's chunk_truncations[:, -1].all() holds.
    mixed_root = _build_robouniverse_fixture_dataset(tmp_path / "dataset_mixed")
    ignore_cfg = OmegaConf.create(
        {
            "env_type": "robouniverse",
            "seed": 0,
            "group_size": 1,
            "max_episode_steps": 2,
            "auto_reset": False,
            "ignore_terminations": True,
            "init_params": {
                "data_root": str(mixed_root),
                "split": "heldout500",
                "robot": "franka",
                "work_dir": str(tmp_path / "work_ignore"),
                "sim_python": sys.executable,
                "device": "cpu",
                "allow_cpu_fixture": True,
            },
        }
    )
    env_ignore = env_cls(
        cfg=ignore_cfg, num_envs=2, seed_offset=0, total_num_processes=1
    )
    try:
        env_ignore.reset()
        _, _, terms, truncs, infos = env_ignore.chunk_step(
            np.full((2, 3, 9), 0.5, dtype=np.float32)
        )
        assert terms.any().item() is False
        assert truncs[:, -1].all().item() is True
        assert truncs[:, :-1].any().item() is False
        assert infos[-1]["verdicts"] == ["success", "failure"]
    finally:
        env_ignore.close()


def test_robouniverse_timeout_isolation_ipc_and_eval_configs(tmp_path):
    import csv
    import socket
    import sys
    from pathlib import Path

    from rlinf.envs import get_env_cls
    from rlinf.envs.sim.ipc import (
        receive_message,
        resolve_physical_cuda_device,
        send_message,
    )
    from rlinf.envs.sim.robouniverse.materializer import _quote_url

    # Shared IPC framing and CUDA_VISIBLE_DEVICES unmasking.
    s1, s2 = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        send_message(s1, {"ok": [1, 2, 3]}, protocol=4)
        assert receive_message(s2, peer_name="Test") == {"ok": [1, 2, 3]}
    finally:
        s1.close()
        s2.close()

    physical, env_map = resolve_physical_cuda_device(
        "cuda:1", {"CUDA_VISIBLE_DEVICES": "3, 5", "KEEP": "1"}
    )
    assert physical == 5
    assert "CUDA_VISIBLE_DEVICES" not in env_map
    assert env_map["KEEP"] == "1"

    # URL quoting preserves query parameters and fragments.
    quoted = _quote_url("https://example.com/dir name/a.usd?download=1&sig=a=b#top")
    assert quoted == "https://example.com/dir%20name/a.usd?download=1&sig=a=b#top"

    # startup_timeout_s guards slow cold backend initialization inside load_task.
    slow_root = tmp_path / "dataset_slow"
    rel = "tasks/franka/franka_slow.zip"
    sha = _write_robouniverse_task_zip(
        slow_root / rel,
        case="franka_slow",
        robot="franka",
        action_dim=9,
        instruction="Slow startup",
        init_sleep_s=1.0,
    )
    with (slow_root / "catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case", "robot", "archive", "sha256", "bytes"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": "franka_slow",
                "robot": "franka",
                "archive": rel,
                "sha256": sha,
                "bytes": str((slow_root / rel).stat().st_size),
            }
        )
    env_cls = get_env_cls("robouniverse")
    slow_cfg = OmegaConf.create(
        {
            "env_type": "robouniverse",
            "seed": 0,
            "group_size": 1,
            "max_episode_steps": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "init_params": {
                "data_root": str(slow_root),
                "cases": ["franka_slow"],
                "work_dir": str(tmp_path / "work_slow"),
                "sim_python": sys.executable,
                "device": "cpu",
                "startup_timeout_s": 0.2,
                "allow_cpu_fixture": True,
            },
        }
    )
    slow_env = env_cls(cfg=slow_cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    try:
        with pytest.raises(TimeoutError):
            slow_env.reset()
    finally:
        slow_env.close()

    # Two ranks sharing work_dir with max_cached_tasks=1 stay isolated on runs/,
    # tasks/, and kit_cache/ without evicting each other's active task.
    shared_root = _build_robouniverse_fixture_dataset(tmp_path / "dataset_shared")
    shared_work = tmp_path / "shared_work"
    shared_cfg = OmegaConf.create(
        {
            "env_type": "robouniverse",
            "seed": 0,
            "group_size": 1,
            "max_episode_steps": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "init_params": {
                "data_root": str(shared_root),
                "split": "heldout500",
                "robot": "franka",
                "work_dir": str(shared_work),
                "max_cached_tasks": 1,
                "sim_python": sys.executable,
                "device": "cpu",
                "allow_cpu_fixture": True,
            },
        }
    )
    env_r0 = env_cls(cfg=shared_cfg, num_envs=1, seed_offset=0, total_num_processes=2)
    env_r1 = env_cls(cfg=shared_cfg, num_envs=1, seed_offset=1, total_num_processes=2)
    try:
        _, info0 = env_r0.reset()
        _, info1 = env_r1.reset()
        assert info0["cases"] == ["franka_01"]
        assert info1["cases"] == ["franka_02"]
        assert (shared_work / "tasks" / "rank_0" / "franka_01").is_dir()
        assert (shared_work / "tasks" / "rank_1" / "franka_02").is_dir()
        assert (shared_work / "runs" / "rank_0" / "slot_0").is_dir()
        assert (shared_work / "runs" / "rank_1" / "slot_0").is_dir()
        assert (shared_work / "kit_cache" / "rank_0" / "slot_0").is_dir()
        assert (shared_work / "kit_cache" / "rank_1" / "slot_0").is_dir()
    finally:
        env_r0.close()
        env_r1.close()

    # Evaluation YAMLs configure OpenPI eval mode and keep model/pi0_5.yaml
    # interpolations intact.
    repo_root = Path(__file__).resolve().parents[2]
    pi05_base = OmegaConf.load(
        repo_root / "examples/embodiment/config/model/pi0_5.yaml"
    )
    for robot, expected_dim, expected_cfg in (
        ("franka", 9, "pi05_robouniverse_franka"),
        ("agilex", 16, "pi05_robouniverse_agilex"),
    ):
        eval_yaml = OmegaConf.load(
            repo_root
            / f"evaluations/robouniverse/robouniverse_{robot}_eval_openpi.yaml"
        )
        merged_model = OmegaConf.merge(pi05_base, eval_yaml.rollout.model)
        assert merged_model.openpi.task == "eval"
        assert merged_model.openpi.config_name == expected_cfg
        assert merged_model.openpi.torch_compile is False
        assert merged_model.openpi_data.norm_stats_path is None
        assert merged_model.openpi.action_env_dim == expected_dim
        assert merged_model.openpi.action_chunk == 15
        assert merged_model.openpi.num_steps == 10
        # Overriding action_dim propagates via ${..action_dim} interpolation.
        merged_model.action_dim = 24
        assert merged_model.openpi.action_env_dim == 24
