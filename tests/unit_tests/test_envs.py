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

from typing import Any

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
from rlinf.envs.layout import EnvLayout, resolve_env_worker_interpreter
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


def test_num_workers_assigns_a_block_and_its_interpreter():
    mode_cfg = _hetero_mode(
        libero=_env(
            "libero",
            4,
            num_workers=2,
            python_interpreter_path="/opt/venv/libero/bin/python",
        ),
        robodojo=_env(
            "robodojo",
            2,
            num_workers=1,
            python_interpreter_path="/opt/venv/robodojo/bin/python",
        ),
    )
    layout = EnvLayout(mode_cfg, world_size=3, path="env.train")

    hosted = [(a.name, a.rank, a.world_size) for a in map(layout.assign, range(3))]

    assert hosted == [
        ("libero", 0, 2),
        ("libero", 1, 2),
        ("robodojo", 0, 1),
    ]
    assert layout.python_interpreter_path(0) == "/opt/venv/libero/bin/python"
    assert layout.python_interpreter_path(2) == "/opt/venv/robodojo/bin/python"
    assert layout.python_interpreter_path(1) == layout.python_interpreter_path(0)


def test_num_workers_must_cover_the_placement_and_share_the_env_count():
    with pytest.raises(ValueError, match="request 3 workers"):
        EnvLayout(
            _hetero_mode(
                libero=_env("libero", 2, num_workers=2),
                robodojo=_env("robodojo", 2, num_workers=1),
            ),
            world_size=2,
            path="env.train",
        )
    with pytest.raises(ValueError, match="same number of envs"):
        EnvLayout(
            _hetero_mode(
                libero=_env("libero", 4, num_workers=2),
                robodojo=_env("robodojo", 2, num_workers=2),
            ),
            world_size=4,
            path="env.train",
        )


def test_one_rank_keeps_a_single_interpreter_across_train_and_eval():
    env = {
        "envs": {
            "libero": _env(
                "libero",
                1,
                num_workers=1,
                python_interpreter_path="/opt/venv/libero/bin/python",
            ),
            "robodojo": _env(
                "robodojo",
                1,
                num_workers=1,
                python_interpreter_path="/opt/venv/robodojo/bin/python",
            ),
        }
    }
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False, "val_check_interval": 1},
            "env": {"train": env, "eval": env},
        }
    )

    assert resolve_env_worker_interpreter(cfg, 1, 2) == "/opt/venv/robodojo/bin/python"
    assert EnvWorker.python_interpreter_for_rank(0, 2, cfg) == (
        "/opt/venv/libero/bin/python"
    )

    cfg.env.eval.envs.robodojo.python_interpreter_path = "/opt/venv/other/bin/python"
    with pytest.raises(ValueError, match="two interpreters"):
        resolve_env_worker_interpreter(cfg, 1, 2)


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


@pytest.mark.parametrize(
    ("recipe", "env_type", "chunk_size"),
    [
        ("robodojo_put_bottles_into_dustbin_ppo_openpi_pi05", "robodojo", 50),
        ("robolab_banana_in_bowl_ppo_openpi_pi05", "robolab", 15),
    ],
)
def test_robodojo_and_robolab_training_recipes_compose(
    recipe, env_type, chunk_size, monkeypatch
):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    config_dir = root / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=recipe)
    OmegaConf.resolve(cfg)

    assert cfg.env.train.env_type == env_type
    assert cfg.env.train.auto_reset
    assert not cfg.env.train.enable_offload
    assert cfg.actor.model.num_action_chunks == chunk_size
    if env_type == "robolab":
        assert cfg.actor.model.openpi_data.norm_stats_path.endswith(
            "assets/droid/norm_stats.json"
        )
    samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (cfg.env.train.max_steps_per_rollout_epoch // chunk_size)
    )
    assert samples % cfg.actor.global_batch_size == 0


def test_robolab_chunk_exposes_terminal_data_and_execution_mask(monkeypatch):
    from types import SimpleNamespace

    from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv

    def _robolab_obs(num_envs: int) -> dict:
        return {
            "main_images": np.zeros((num_envs, 8, 8, 3), dtype=np.uint8),
            "wrist_images": np.zeros((num_envs, 8, 8, 3), dtype=np.uint8),
            "states": np.zeros((num_envs, 8), dtype=np.float32),
            "task_descriptions": ["put the banana in the bowl"] * num_envs,
            "active_mask": np.ones(num_envs, dtype=bool),
        }

    env = object.__new__(RoboLabEnv)
    env.num_envs = 2
    env._closed = True
    env.cfg = OmegaConf.create({"video_cfg": {"save_video": False}, "is_eval": False})
    terminal_obs = _robolab_obs(2)
    reset_obs = _robolab_obs(2)
    reset_obs["states"][0, 0] = 1

    result = {
        "observation": reset_obs,
        "final_observation": terminal_obs,
        "final_mask": np.array([True, False]),
        "executed_steps": np.array([2, 3]),
        "rewards": np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
        "terminations": np.array(
            [[False, False, True], [False, False, False]], dtype=bool
        ),
        "truncations": np.zeros((2, 3), dtype=bool),
        "episode": {
            "success_once": np.array([1, 0], dtype=np.float32),
            "return": np.array([1, 0], dtype=np.float32),
            "episode_len": np.array([2, 3], dtype=np.float32),
            "reward": np.array([0.5, 0], dtype=np.float32),
        },
    }
    # Simulate Ray Plasma read-only NumPy deserialization.
    for value in (
        result["final_mask"],
        result["executed_steps"],
        result["rewards"],
        result["terminations"],
        result["truncations"],
        *result["episode"].values(),
    ):
        value.setflags(write=False)

    class RemoteChunkStep:
        def remote(self, actions, collect_frames=False):
            del actions, collect_frames
            return result

    env._process = SimpleNamespace(chunk_step=RemoteChunkStep())
    monkeypatch.setattr(
        "rlinf.envs.sim.robolab.robolab_env.ray.get", lambda value: value
    )

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observations, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 3, 8), dtype=np.float32)
        )

    assert not [w for w in caught if "not writable" in str(w.message)]

    assert len(observations) == len(infos) == 1
    assert rewards.shape == terminations.shape == truncations.shape == (2, 3)
    assert "executed_mask" not in infos[0]
    assert infos[0]["valid_action_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert infos[0]["_final_observation"].tolist() == [True, False]
    assert infos[0]["final_info"]["episode"]["return"].tolist() == [1, 0]
    assert observations[0]["states"][0, 0].item() == 1
    assert infos[0]["final_observation"]["states"][0, 0].item() == 0


def test_robolab_process_done_flags_and_partial_reset_contract():
    from types import SimpleNamespace

    from rlinf.envs.sim.robolab.robolab_process import RoboLabProcess

    process_cls = RoboLabProcess.__ray_metadata__.modified_class

    def _raw_obs(value: float) -> dict:
        return {
            "image_obs": {
                "over_shoulder_left_camera": torch.full(
                    (1, 4, 4, 3), int(value), dtype=torch.uint8
                ),
                "wrist_cam": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
            },
            "proprio_obs": {
                "arm_joint_pos": torch.full((1, 7), value, dtype=torch.float32),
                "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
            },
        }

    def _make_process(
        *, auto_reset: bool, max_steps: int | None, finish_on_step: int | None
    ):
        events: list[tuple[str, Any]] = []
        step_counter = {"n": 0}

        class FakeRecorder:
            def export_episodes(self, env_ids=None):
                events.append(("export_episodes", list(env_ids)))

            def set_episode_index(self, episode_index, env_ids=None):
                events.append(("set_episode_index", episode_index, list(env_ids)))

        class FakeEnv:
            device = torch.device("cpu")

            def __init__(self):
                self._frozen_envs = torch.zeros(1, dtype=torch.bool)
                self._pre_step_frozen = torch.zeros(1, dtype=torch.bool)
                self._env_results = {}
                self._env_term_step = {}
                self._has_stepped = True
                self.recorder_manager = FakeRecorder()

            @property
            def active_env_ids(self):
                return (~self._frozen_envs).nonzero(as_tuple=False).squeeze(-1).tolist()

            def step(self, action):
                del action
                step_counter["n"] += 1
                if finish_on_step is not None and step_counter["n"] == finish_on_step:
                    self._frozen_envs[0] = True
                    self._env_results[0] = True
                return _raw_obs(float(step_counter["n"])), None, None, None, {}

            def get_env_results(self):
                return [
                    {
                        "env_id": 0,
                        "success": self._env_results.get(0),
                        "step": step_counter["n"],
                    }
                ]

            def reset(self, seed=None, env_ids=None):
                del seed
                assert not self._has_stepped
                events.append(("env.reset", env_ids.tolist()))
                return _raw_obs(99.0), {}

        proc = object.__new__(process_cls)
        proc.env = FakeEnv()
        proc._image_tools = SimpleNamespace(resize_with_pad=lambda img, h, w: img)
        proc._end_episode = lambda env: events.append(("end_episode", None))
        proc._get_current_subtask_info = lambda env, env_id=0: {"score": 0.5}
        proc.enable_subtask = False
        proc.prompt = "pick"
        proc.timeline = SimpleNamespace(is_playing=lambda: True)
        proc.kit_app = SimpleNamespace(update=lambda: None)
        proc.num_envs = 1
        proc.max_steps = max_steps
        proc.auto_reset = auto_reset
        proc.steps = np.zeros(1, dtype=np.int32)
        proc.finished = np.zeros(1, dtype=bool)
        proc.success = np.zeros(1, dtype=np.float32)
        proc.score = np.zeros(1, dtype=np.float32)
        proc.episode_indices = np.arange(1, dtype=np.int64)
        proc.last_obs = _raw_obs(0.0)
        proc.episode_open = True
        return proc, events

    # Eval (auto_reset=False): done appears once at finishing step, not duplicated at -1,
    # and no final_observation/final_mask is emitted (avoiding double-pack and duplicate video frames).
    eval_proc, _ = _make_process(auto_reset=False, max_steps=10, finish_on_step=2)
    pack_calls = {"n": 0}
    orig_pack = eval_proc._pack_obs

    def _counting_pack(*args, **kwargs):
        pack_calls["n"] += 1
        return orig_pack(*args, **kwargs)

    eval_proc._pack_obs = _counting_pack
    eval_res = eval_proc.chunk_step(np.zeros((1, 4, 8), dtype=np.float32))
    assert eval_res["terminations"].tolist() == [[False, True, False, False]]
    assert not eval_res["truncations"].any()
    assert eval_res["final_observation"] is None
    assert eval_res["final_mask"] is None
    assert pack_calls["n"] == 1

    # Training (auto_reset=True) timeout: done folds cleanly to -1, recorder exports before reset,
    # public env.reset(env_ids=...) supplies fresh post-reset obs, and episode index advances after.
    train_proc, events = _make_process(
        auto_reset=True, max_steps=2, finish_on_step=None
    )
    train_res = train_proc.chunk_step(np.zeros((1, 4, 8), dtype=np.float32))
    assert not train_res["terminations"].any()
    assert train_res["truncations"].tolist() == [[False, False, False, True]]
    assert train_res["observation"]["states"][0, 0].item() == 99.0
    assert train_res["final_observation"]["states"][0, 0].item() == 2.0
    assert events == [
        ("export_episodes", [0]),
        ("env.reset", [0]),
        ("set_episode_index", 1, [0]),
    ]

    # Multi-env mid-chunk timeout: slot 0 times out at step 2 while slot 1 stays active to step 4.
    # Slot 0 is frozen immediately and final_observation keeps its step-2 snapshot, not step-4 obs.
    def _multi_raw_obs(values: list[float]) -> dict:
        return {
            "image_obs": {
                "over_shoulder_left_camera": torch.stack(
                    [torch.full((4, 4, 3), int(v), dtype=torch.uint8) for v in values]
                ),
                "wrist_cam": torch.zeros((len(values), 4, 4, 3), dtype=torch.uint8),
            },
            "proprio_obs": {
                "arm_joint_pos": torch.stack(
                    [torch.full((7,), v, dtype=torch.float32) for v in values]
                ),
                "gripper_pos": torch.zeros((len(values), 1), dtype=torch.float32),
            },
        }

    frozen_during_steps: list[list[bool]] = []
    multi_step = {"n": 0}

    class MultiFakeEnv:
        device = torch.device("cpu")

        def __init__(self):
            self._frozen_envs = torch.zeros(2, dtype=torch.bool)
            self._pre_step_frozen = torch.zeros(2, dtype=torch.bool)
            self._env_results = {}
            self._env_term_step = {}
            self._has_stepped = True
            self.recorder_manager = None

        @property
        def active_env_ids(self):
            return (~self._frozen_envs).nonzero(as_tuple=False).squeeze(-1).tolist()

        def step(self, action):
            del action
            multi_step["n"] += 1
            frozen_during_steps.append(self._frozen_envs.tolist())
            # Overwrite both slots every step to confirm slot 0's snapshot came from step 2.
            val = float(multi_step["n"])
            return _multi_raw_obs([val * 10.0, val]), None, None, None, {}

        def get_env_results(self):
            return [
                {"env_id": i, "success": None, "step": multi_step["n"]}
                for i in range(2)
            ]

        def reset(self, seed=None, env_ids=None):
            del seed, env_ids
            return _multi_raw_obs([99.0, float(multi_step["n"])]), {}

    multi_proc = object.__new__(process_cls)
    multi_proc.env = MultiFakeEnv()
    multi_proc._image_tools = SimpleNamespace(resize_with_pad=lambda img, h, w: img)
    multi_proc._end_episode = lambda env: None
    multi_proc._get_current_subtask_info = lambda env, env_id=0: {"score": 0.5}
    multi_proc.enable_subtask = False
    multi_proc.prompt = "pick"
    multi_proc.timeline = SimpleNamespace(is_playing=lambda: True)
    multi_proc.kit_app = SimpleNamespace(update=lambda: None)
    multi_proc.num_envs = 2
    multi_proc.max_steps = 4
    multi_proc.auto_reset = True
    multi_proc.steps = np.array([2, 0], dtype=np.int32)
    multi_proc.finished = np.zeros(2, dtype=bool)
    multi_proc.success = np.zeros(2, dtype=np.float32)
    multi_proc.score = np.zeros(2, dtype=np.float32)
    multi_proc.episode_indices = np.arange(2, dtype=np.int64)
    multi_proc.last_obs = _multi_raw_obs([0.0, 0.0])
    multi_proc.episode_open = True

    multi_res = multi_proc.chunk_step(np.zeros((2, 3, 8), dtype=np.float32))
    # Slot 0 hit max_steps=4 at sub-step 2, so sub-step 3 saw slot 0 frozen.
    assert frozen_during_steps == [[False, False], [False, False], [True, False]]
    assert multi_res["final_mask"].tolist() == [True, False]
    assert multi_res["final_observation"]["states"][0, 0].item() == 20.0
    assert multi_res["observation"]["states"][0, 0].item() == 99.0
    assert multi_res["observation"]["states"][1, 0].item() == 3.0


def test_robodojo_chunk_exposes_execution_mask_and_auto_reset():
    from types import MethodType

    from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv

    env = object.__new__(RoboDojoEnv)
    env.num_envs = 2
    env.auto_reset = True
    env._done = torch.zeros(2, dtype=torch.bool)
    env.cfg = OmegaConf.create({"video_cfg": {"save_video": False}, "is_eval": False})
    obs = {"states": torch.zeros(2, 14)}
    info = {"executed_steps": np.array([2, 3])}

    env.step = MethodType(
        lambda self, actions, auto_reset=False: (
            obs,
            torch.tensor([1.0, 0.0]),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
            info,
        ),
        env,
    )

    def auto_reset(self, dones, final_obs, final_info):
        return {"states": torch.ones(2, 14)}, {
            "final_observation": final_obs,
            "final_info": final_info,
            "_final_observation": dones,
            "_final_info": dones,
        }

    env._auto_reset = MethodType(auto_reset, env)
    observations, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((2, 3, 14), dtype=np.float32)
    )

    assert rewards.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert terminations.tolist() == [
        [False, False, True],
        [False, False, False],
    ]
    assert not truncations.any()
    assert observations[0]["states"].eq(1).all()
    assert "executed_mask" not in infos[0]
    assert infos[0]["valid_action_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_robodojo_save_video_chunk_step_does_not_inflate_reward_or_overshoot(
    monkeypatch,
):
    from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv

    env = object.__new__(RoboDojoEnv)
    env.num_envs = 2
    env.auto_reset = False
    env.ignore_terminations = False
    env.use_custom_reward = True
    env.use_rel_reward = False
    env.record_metrics = True
    env._action_perm = None
    env._state_perm = None
    env.center_crop = False
    env.cfg = OmegaConf.create(
        {
            "reward_coef": 1.0,
            "max_episode_steps": 3,
            "video_cfg": {"save_video": True},
            "is_eval": False,
        }
    )
    env._elapsed_steps = torch.zeros(2, dtype=torch.long)
    env.prev_step_reward = torch.zeros(2)
    env._done = torch.zeros(2, dtype=torch.bool)
    env._init_metrics()

    calls = {"n": 0}
    raw_obs = [
        {
            "full_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "task",
        }
        for _ in range(2)
    ]

    class FakeProcess:
        class Step:
            def remote(self, actions):
                del actions
                calls["n"] += 1
                # Slot 0 latches terminated at step 2; slot 1 runs until RLinf max_episode_steps=3.
                slot0_done = calls["n"] >= 2
                return (
                    raw_obs,
                    [0.0, 0.0],
                    [slot0_done, False],
                    [False, False],
                    [
                        {
                            "success": slot0_done,
                            "executed_steps": 0 if calls["n"] > 2 else 1,
                        },
                        {"success": False, "executed_steps": 1},
                    ],
                )

        step = Step()

    env._process = FakeProcess()
    monkeypatch.setattr(
        "rlinf.envs.sim.robodojo.robodojo_env.ray.get", lambda value: value
    )

    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((2, 5, 14), dtype=np.float32)
    )

    # Stopped stepping once both slots finished (slot 0 at step 2, slot 1 at step 3).
    assert calls["n"] == 3
    assert len(obs_list) == 3
    assert rewards[:, -1].tolist() == [1.0, 0.0]
    assert terminations[:, -1].tolist() == [True, False]
    assert truncations[:, -1].tolist() == [False, True]
    assert infos[0]["episode"]["return"].tolist() == [1.0, 0.0]
    assert infos[0]["episode"]["episode_len"].tolist() == [2, 3]
    assert "executed_mask" not in infos[0]
    assert infos[0]["valid_action_mask"].tolist() == [
        [True, True, False, False, False],
        [True, True, True, False, False],
    ]

    # A second chunk without reset (auto_reset=False) must keep already-done slots latched
    # and not re-add reward or increment elapsed_steps.
    obs_list2, rewards2, terminations2, truncations2, infos2 = env.chunk_step(
        np.zeros((2, 5, 14), dtype=np.float32)
    )
    assert len(obs_list2) == 1
    assert rewards2[:, -1].tolist() == [0.0, 0.0]
    assert terminations2[:, -1].tolist() == [False, False]
    assert truncations2[:, -1].tolist() == [False, False]
    assert infos2[0]["episode"]["return"].tolist() == [1.0, 0.0]
    assert infos2[0]["episode"]["episode_len"].tolist() == [2, 3]


def test_robodojo_no_video_separates_natural_termination_from_timeout(monkeypatch):
    from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv

    raw_obs = [
        {
            "full_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "task",
        }
    ]

    def _make_env(*, ignore_terminations: bool, max_episode_steps: int):
        env = object.__new__(RoboDojoEnv)
        env.num_envs = 1
        env.auto_reset = False
        env.ignore_terminations = ignore_terminations
        env.use_custom_reward = True
        env.use_rel_reward = False
        env.record_metrics = True
        env._action_perm = None
        env._state_perm = None
        env.center_crop = False
        env.cfg = OmegaConf.create(
            {
                "reward_coef": 1.0,
                "max_episode_steps": max_episode_steps,
                "video_cfg": {"save_video": False},
                "is_eval": False,
            }
        )
        env._elapsed_steps = torch.zeros(1, dtype=torch.long)
        env.prev_step_reward = torch.zeros(1)
        env._done = torch.zeros(1, dtype=torch.bool)
        env._init_metrics()
        return env

    class NaturalTermProcess:
        class Step:
            def remote(self, actions):
                del actions
                return (
                    raw_obs,
                    [0.0],
                    [True],
                    [False],
                    [{"success": True, "executed_steps": 2}],
                )

        step = Step()

    monkeypatch.setattr(
        "rlinf.envs.sim.robodojo.robodojo_env.ray.get", lambda value: value
    )

    # save_video=False, ignore_terminations=False: natural termination at step 2 on a
    # 5-step chunk with max_episode_steps=5 must not mark truncated=True and counts 2 steps.
    env_train = _make_env(ignore_terminations=False, max_episode_steps=5)
    env_train._process = NaturalTermProcess()
    _, _, terminations, truncations, infos = env_train.chunk_step(
        np.zeros((1, 5, 14), dtype=np.float32)
    )
    assert terminations[:, -1].tolist() == [True]
    assert truncations[:, -1].tolist() == [False]
    assert infos[0]["episode"]["episode_len"].tolist() == [2]

    # save_video=False, ignore_terminations=True (eval): after success, elapsed_steps still
    # advances by full chunk horizon and emits truncated=True at max_episode_steps.
    env_eval = _make_env(ignore_terminations=True, max_episode_steps=5)
    env_eval._process = NaturalTermProcess()
    _, _, eval_term, eval_trunc, eval_infos = env_eval.chunk_step(
        np.zeros((1, 5, 14), dtype=np.float32)
    )
    assert eval_term[:, -1].tolist() == [False]
    assert eval_trunc[:, -1].tolist() == [True]
    assert eval_infos[0]["episode"]["episode_len"].tolist() == [5]
    assert eval_infos[0]["episode"]["success_once"].tolist() == [True]


def test_robodojo_init_env_defaults_eula(monkeypatch):
    from types import SimpleNamespace

    from rlinf.envs.sim.robodojo import robodojo_env

    captured = {}

    class FakeOptions:
        def remote(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(ready=SimpleNamespace(remote=lambda: None))

    def fake_options(**kwargs):
        captured.update(kwargs)
        return FakeOptions()

    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.setattr(robodojo_env.RoboDojoProcess, "options", fake_options)
    monkeypatch.setattr(
        robodojo_env.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "a" * 56),
    )
    monkeypatch.setattr(robodojo_env.ray, "get", lambda value: value)

    env = object.__new__(robodojo_env.RoboDojoEnv)
    env.cfg = OmegaConf.create({"task_config": {"task_name": "t"}})
    env.num_envs = 1
    env.reset_state_ids = torch.tensor([0])
    env._init_env()

    assert captured["runtime_env"]["env_vars"]["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_hetero_model_envs_config_validation(monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from rlinf.config import validate_model_envs

    root = Path(__file__).resolve().parents[2]
    config_dir = root / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    monkeypatch.setenv("REPO_PATH", str(root))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="libero_robodojo_robolab_ppo_openpi_pi05")

    validate_model_envs(cfg, cfg.actor.model)
    validate_env_policy_fields(cfg, cfg.actor.model)
    resolve_env_mode_cfg(cfg.env.train, "env.train", env_world_size=3)
    train_layout = EnvLayout(cfg.env.train, 3, "env.train")
    validate_env_training_cfg(
        cfg,
        train_layout,
        stage_num=cfg.rollout.pipeline_stage_num,
        actor_world_size=1,
    )
    validate_eval_env_cfg(
        cfg,
        env_world_size=3,
        stage_num=cfg.rollout.pipeline_stage_num,
        num_action_chunks=cfg.actor.model.num_action_chunks,
    )
    assert cfg.actor.model.model_path == cfg.rollout.model.model_path
    assert list(cfg.env.train.envs) == ["libero", "robodojo", "robolab"]
    assert list(cfg.actor.model.envs) == ["libero", "robodojo", "robolab"]

    # Unknown env key under model.envs is rejected.
    unknown_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(unknown_cfg):
        unknown_cfg.actor.model.envs.missing = {
            "config_name": "pi05_libero",
            "action_dim": 7,
        }
    with pytest.raises(AssertionError, match=r"model\.envs\.missing must name"):
        validate_model_envs(unknown_cfg, unknown_cfg.actor.model)

    # Multi-env model.envs under NFT is rejected at config validation.
    nft_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(nft_cfg):
        nft_cfg.algorithm.loss_type = "embodied_nft"
    with pytest.raises(AssertionError, match="NFT training requires"):
        validate_model_envs(nft_cfg, nft_cfg.actor.model)

    # Models that have not registered per-env validation fail fast.
    unsupported_cfg = OmegaConf.create(
        {
            "env": {"train": _hetero_mode(libero=_env("libero", 2))},
            "actor": {
                "model": {
                    "model_type": "openvla",
                    "envs": {"libero": {"action_dim": 7}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="not supported by model_type='openvla'"):
        validate_model_envs(unsupported_cfg, unsupported_cfg.actor.model)
