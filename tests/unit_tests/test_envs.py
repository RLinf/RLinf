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
from rlinf.config import resolve_env_mode_cfg, validate_env_training_cfg
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
    assert assignment.scope_metrics(metrics) is metrics


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
            layout.assign(rank).scope_metrics({"success_once": success})
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
        ({"loss_type": "embodied_sac"}, None, r"algorithm.loss_type=embodied_sac"),
        ({}, {"enable_decoupled_mode": True}, r"runner.enable_decoupled_mode"),
    ],
)
def test_config_rejects_per_env_settings_the_run_cannot_honor(algorithm, runner, match):
    cfg = _training_cfg(
        algorithm, runner, libero=_env("libero", 8), maniskill=_env("maniskill", 8)
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=match):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)


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


def test_policy_obs_spec_rejects_keys_the_policy_batch_does_not_carry():
    with pytest.raises(ValueError, match=r"unknown keys \['main_image'\]"):
        PolicyObsSpec(keys=["main_image"])


def test_config_needs_grpo_envs_to_stop_trajectories_at_their_first_done():
    cfg = _training_cfg(
        {"adv_type": "grpo"},
        libero=_env("libero", 8),
        maniskill=_env("maniskill", 8, ignore_terminations=True),
    )
    layout = EnvLayout(cfg.env.train, world_size=2, path="env.train")

    with pytest.raises(AssertionError, match=r"env.train.envs.maniskill needs"):
        validate_env_training_cfg(cfg, layout, stage_num=1, actor_world_size=2)
