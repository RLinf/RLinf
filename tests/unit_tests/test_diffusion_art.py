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

"""DiffusionART tensor-kernel tests."""

import inspect
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_cfg, validate_openpi_art_cfg
from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.envs.utils import grouped_state_max_error
from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.workers.actor.fsdp_art_policy_worker import (
    EmbodiedARTFSDPPolicy,
    _art_loss,
    _art_mask_count_metrics,
    _art_scalar_covariance,
    _art_trajectory_guidance,
    _art_velocity_drift_stats,
    _prompt_disjoint_windows,
    _validate_art_rollout,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import (
    MultiStepRolloutWorker,
    _art_branch_observation_stats,
    _repeat_art_group_leaders,
)


def test_art_velocity_target_validation_and_prompt_windows(monkeypatch):
    """The supported ART path keeps its sign and validates outside the hot path."""
    assert tuple(inspect.signature(_art_loss).parameters) == (
        "current_velocity",
        "old_velocity",
        "sigma",
        "sigma_next",
        "guidance_direction",
        "guidance_magnitude",
        "loss_space",
        "mask_metric",
        "mask_threshold",
        "reference_noise_level",
        "corrective_mask_enabled",
        "coordinate_mask",
        "output_gradient_stats",
    )
    states = torch.zeros(4, 2, 2)
    sigmas = torch.tensor([[1.0, 0.5]]).expand(4, -1)
    sigmas_next = torch.tensor([[0.5, 0.0]]).expand(4, -1)
    magnitudes = torch.full((4,), 0.25)
    _validate_art_rollout(states, sigmas, sigmas_next, magnitudes)

    old = torch.zeros(4, 2)
    current = old.clone().requires_grad_(True)
    direction = torch.tensor([[1.0, -1.0]]).expand_as(old)

    def fail_hot_path_scan(*args, **kwargs):
        raise AssertionError("art_loss must not scan tensor values in the hot path")

    with monkeypatch.context() as context:
        context.setattr(torch, "isfinite", fail_hot_path_scan)
        terms = _art_loss(
            current,
            old,
            sigmas[:, 1],
            sigmas_next[:, 1],
            direction,
            magnitudes,
            mask_metric="fixed_mse",
            mask_threshold=1.0,
        )

    expected_target = old - magnitudes[:, None] * direction
    torch.testing.assert_close(terms["target"], expected_target)
    torch.testing.assert_close(
        terms["per_item_loss"],
        sigmas[:, 1] * (expected_target - current).square().mean(1),
    )
    terms["per_item_loss"].sum().backward()
    assert torch.all(current.grad * expected_target < 0)

    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    order = [3, 4, 0, 7, 2, 5, 1, 6]
    windows = _prompt_disjoint_windows(groups, updates=2, order=order)
    assert windows == ((3, 4, 0, 7), (2, 5, 1, 6))
    assert set(windows[0]).isdisjoint(windows[1])


def test_art_rollout_rejects_non_decreasing_sigma():
    states = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="sigma_next < sigma"):
        _validate_art_rollout(
            states,
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[0.5, 0.5]]),
            torch.ones(1),
        )


def test_art_complete_group_covariance_and_constant_rewards():
    """Guidance uses complete prompt groups and freezes constant-reward groups."""
    endpoints = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [10.0, 5.0], [14.0, 9.0], [7.0, -3.0], [9.0, 5.0]],
        requires_grad=True,
    )
    rewards = torch.tensor([0.0, 2.0, 1.0, 3.0, 0.1, 0.1], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    stats = _art_scalar_covariance(
        endpoints,
        rewards,
        groups,
        group_count=3,
        zero_signal_mode="exact",
    )

    covariance = torch.tensor([[2.0, 4.0], [4.0, 4.0], [0.0, 0.0]])
    direction = covariance / covariance.square().mean(1, keepdim=True).sqrt().clamp_min(
        1e-5
    )
    torch.testing.assert_close(stats["group_covariance"], covariance)
    torch.testing.assert_close(stats["group_directions"], direction)
    torch.testing.assert_close(
        stats["group_guidance_magnitudes"], torch.tensor([1.0, 1.0, 0.0])
    )
    torch.testing.assert_close(stats["group_active"], torch.tensor([True, True, False]))
    torch.testing.assert_close(stats["group_counts"], torch.full((3,), 2.0))
    assert all(not value.requires_grad for value in stats.values())


def test_art_covariance_uses_only_executed_coordinates():
    endpoints = torch.tensor(
        [[[0.0], [100.0]], [[2.0], [4.0]], [[999.0], [8.0]], [[123.0], [888.0]]]
    )
    mask = torch.tensor([[True, False], [True, True], [False, True], [False, False]])
    stats = _art_scalar_covariance(
        endpoints,
        torch.tensor([0.0, 1.0, 2.0, 999.0]),
        torch.zeros(4, dtype=torch.long),
        1,
        coordinate_mask=mask[..., None],
        zero_signal_mode="exact",
        direction_mode="normalized_aggregate",
    )
    torch.testing.assert_close(
        stats["group_covariance"], torch.tensor([[[1.0], [2.0]]])
    )
    torch.testing.assert_close(
        stats["group_coordinate_counts"], torch.tensor([[[2.0], [2.0]]])
    )
    torch.testing.assert_close(stats["group_counts"], torch.tensor([3.0]))
    torch.testing.assert_close(
        stats["group_directions"].square().mean(), torch.tensor(1.0)
    )


@pytest.mark.parametrize("valid_count", [0, 1, 4])
def test_art_constant_and_singleton_action_groups_have_zero_guidance(valid_count):
    mask = torch.arange(4) < valid_count
    stats = _art_scalar_covariance(
        torch.arange(8.0).reshape(4, 2),
        torch.full((4,), 0.1),
        torch.zeros(4, dtype=torch.long),
        1,
        coordinate_mask=mask[:, None],
        zero_signal_mode="exact",
    )
    assert not stats["group_active"].any()
    assert stats["group_guidance_magnitudes"].eq(0).all()
    assert stats["group_directions"].eq(0).all()
    assert all(torch.isfinite(value).all() for value in stats.values())


def test_art_trajectory_groups_keep_decisions_and_reset_groups_separate():
    endpoints = torch.tensor([0.0, 2.0, 100.0, 104.0, 2.0, 0.0, 104.0, 100.0]).reshape(
        2, 4, 1, 1
    )
    guidance = _art_trajectory_guidance(
        endpoints,
        torch.tensor([0.0, 1.0, 1.0, 1.0]),
        torch.ones(2, 4, 1, dtype=torch.bool),
        2,
        zero_signal_mode="exact",
    )
    assert guidance["stats"]["group_active"].tolist() == [True, False, True, False]
    torch.testing.assert_close(guidance["directions"][0, :2], torch.ones(2, 1, 1))
    torch.testing.assert_close(guidance["directions"][1, :2], -torch.ones(2, 1, 1))
    assert guidance["magnitudes"][:, 2:].eq(0).all()


def test_art_loss_masks_padding_and_keeps_inward_restoration():
    old = torch.zeros(2, 2, 3)
    current = torch.tensor(
        [
            [[-0.5, -0.5, 999.0], [999.0, 999.0, 999.0]],
            [[-2.0, -2.0, 999.0], [999.0, 999.0, 999.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor([[[True, True, False], [False, False, False]]]).expand_as(old)
    terms = _art_loss(
        current,
        old,
        torch.ones(2),
        torch.full((2,), 0.5),
        torch.ones_like(old),
        torch.ones(2),
        mask_metric="fixed_mse",
        mask_threshold=0.1,
        coordinate_mask=mask,
    )
    assert terms["masked"].tolist() == [True, False]
    torch.testing.assert_close(terms["movement_energy"], torch.tensor([0.25, 4.0]))
    torch.testing.assert_close(terms["per_item_loss"], torch.tensor([0.0, 1.0]))
    terms["per_item_loss"].sum().backward()
    assert current.grad[~mask].eq(0).all()
    assert current.grad[1, 0, :2].lt(0).all()
    assert not old.requires_grad


@pytest.mark.parametrize("loss_space", ["velocity_linear_sigma", "velocity_uniform"])
def test_art_output_gradient_statistics_match_autograd(loss_space):
    old = torch.zeros(5, 2)
    current = torch.tensor(
        [[-0.5, 999.0], [-2.0, -2.0], [1.0, 999.0], [2.0, 999.0], [0.5, 0.25]],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, False], [True, True], [True, False], [True, False], [True, True]]
    )
    sigma = torch.tensor([1.0, 0.5, 0.25, 0.0, 0.75])
    magnitude = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    direction = torch.ones_like(old)
    terms = _art_loss(
        current,
        old,
        sigma,
        sigma / 2,
        direction,
        magnitude,
        loss_space=loss_space,
        mask_metric="fixed_mse",
        mask_threshold=0.1,
        coordinate_mask=mask,
        output_gradient_stats=True,
    )
    assert terms["keep_mask"].tolist() == [False, True, False, False, True]
    weight = sigma if loss_space == "velocity_linear_sigma" else torch.ones_like(sigma)
    scale = 2 * weight * terms["keep_mask"] / (len(current) * mask.sum(1))
    guidance_gradient = scale[:, None] * torch.where(
        mask, magnitude[:, None] * direction, 0.0
    )
    restoration_gradient = scale[:, None] * torch.where(
        mask, current.detach() - old, 0.0
    )
    total_gradient = torch.autograd.grad(terms["per_item_loss"].mean(), current)[0]
    torch.testing.assert_close(total_gradient, guidance_gradient + restoration_gradient)
    expected = {
        "output_guidance_grad_sq": guidance_gradient.square().sum(),
        "output_restoration_grad_sq": restoration_gradient.square().sum(),
        "output_grad_inner": (guidance_gradient * restoration_gradient).sum(),
    }
    for key, value in expected.items():
        torch.testing.assert_close(terms[key], value)
        assert not terms[key].requires_grad

    on_policy = _art_loss(
        old,
        old,
        sigma,
        sigma / 2,
        direction,
        magnitude,
        coordinate_mask=mask,
        output_gradient_stats=True,
    )
    assert on_policy["output_restoration_grad_sq"] == 0
    assert on_policy["output_grad_inner"] == 0
    default = _art_loss(
        old, old, sigma, sigma / 2, direction, magnitude, coordinate_mask=mask
    )
    assert not set(expected).intersection(default)


def test_art_velocity_drift_pools_valid_coordinates_across_microbatches():
    old = torch.zeros(3, 4, 4)
    velocity = torch.ones_like(old)
    velocity[:, 2:, :2] = 3
    velocity[:, :, 2:] = 4
    velocity[2] = 999
    mask = torch.ones(3, 2, 2, dtype=torch.bool)
    mask[1, 1] = False
    mask[2] = False
    stats, max_error = _art_velocity_drift_stats(velocity, old, 2, 2, mask)
    expected = torch.tensor(
        [[336, 32], [6, 6], [330, 26], [72, 8], [256, 16]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(stats, expected)
    assert max_error == 1
    assert not stats.requires_grad
    assert not max_error.requires_grad
    parts = [
        _art_velocity_drift_stats(velocity[rows], old[rows], 2, 2, mask[rows])
        for rows in (slice(0, 1), slice(1, 3))
    ]
    torch.testing.assert_close(parts[0][0] + parts[1][0], stats)
    assert max(parts[0][1], parts[1][1]) == max_error


@pytest.mark.parametrize("valid", [False, True])
def test_art_velocity_drift_handles_empty_regions(valid):
    old = torch.zeros(2, 2, 2)
    velocity = torch.full_like(old, 0.25, requires_grad=True)
    mask = torch.full_like(old, valid, dtype=torch.bool)
    stats, max_error = _art_velocity_drift_stats(velocity, old, 2, 2, mask)
    if valid:
        torch.testing.assert_close(stats[:2], torch.tensor([[0.5, 8], [0.5, 8]]))
        assert max_error == 0.25
    else:
        assert not stats.any()
        assert max_error == 0
    assert not stats[2:].any()
    assert not stats.requires_grad


def _art_trajectory_worker(first_chunk_only=False):
    worker = object.__new__(EmbodiedARTFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": 2,
                "art_updates_per_rollout": 2,
                "art_first_chunk_only": first_chunk_only,
                "art_timestep_fraction": 0.5,
            },
        }
    )
    decisions, episodes, actions, dims, timesteps = 3, 4, 2, 2, 4
    endpoints = torch.arange(float(decisions * episodes * actions * dims)).reshape(
        decisions, episodes, actions, dims
    )
    dones = torch.zeros(decisions + 1, episodes, actions, dtype=torch.bool)
    dones[1, 0, 0] = True
    rewards = torch.zeros(decisions, episodes, actions)
    rewards[0, 0, 0] = 1.0
    rewards[1:, 0] = 999.0
    rewards[-1, -1, -1] = 1.0
    states = (
        torch.arange(float(timesteps))[None, None, :, None, None]
        .expand(decisions, episodes, timesteps, 4, 3)
        .clone()
    )
    worker.rollout_batch = {
        "rewards": rewards,
        "dones": dones,
        "loss_mask": compute_loss_mask(dones)[0].any(-1, keepdim=True),
        "forward_inputs": {
            "art_endpoint": endpoints,
            "art_states": states,
            "art_old_velocity": (states + 10).clone(),
            "art_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25])
            .expand(decisions, episodes, -1)
            .clone(),
            "art_sigmas_next": torch.tensor([0.75, 0.5, 0.25, 0.0])
            .expand(decisions, episodes, -1)
            .clone(),
            "observation/state": torch.arange(episodes)[None, :, None].expand(
                decisions, -1, -1
            ),
        },
    }
    return worker


@pytest.mark.parametrize("first_chunk_only", [False, True])
def test_art_trajectory_preparation_preserves_episode_windows_and_cached_velocities(
    first_chunk_only,
):
    worker = _art_trajectory_worker(first_chunk_only)
    generator = torch.Generator().manual_seed(7)
    metrics, active, windows = worker._prepare_art_trajectory_training_inputs(generator)
    assert active
    assert metrics["actor/art_reward_mean"] == 0.5
    assert metrics["actor/art_reward_std"] == 0.5
    assert set(windows[0].tolist()).isdisjoint(windows[1].tolist())
    assert set((windows[0] % 4).tolist()).isdisjoint((windows[1] % 4).tolist())
    assert all(len(set((window % 4 // 2).tolist())) == 2 for window in windows)
    rows = sum(len(window) for window in windows)
    assert rows == (4 if first_chunk_only else 12)
    worker._select_art_training_timesteps(generator)
    inputs = worker.rollout_batch["forward_inputs"]
    selected = inputs["art_states"][..., 0, 0]
    torch.testing.assert_close(inputs["art_old_velocity"][..., 0, 0], selected + 10)
    torch.testing.assert_close(inputs["art_sigmas"], 1 - selected / 4)
    torch.testing.assert_close(inputs["art_sigmas_next"], 0.75 - selected / 4)
    assert selected.shape == (rows, 2)
    for window in windows:
        batch = worker._index_nested(worker.rollout_batch, window)
        replay = worker._model_forward_inputs(batch["forward_inputs"])
        # Flattened rows are decision-major, so the episode is the row modulo 4.
        assert replay["observation/state"][:, 0].tolist() == (window % 4).tolist()
        assert not any(key.startswith("art_") for key in replay)
    if not first_chunk_only:
        assert not inputs["art_coordinate_mask"][4, ...].any()


@pytest.fixture(params=[2, 4])
def art_config_cluster(monkeypatch, request):
    node = NodeInfo(
        node_labels=[],
        node_rank=0,
        ray_id="test-node",
        node_ip="127.0.0.1",
        num_cpus=32,
        python_interpreter_path="/usr/bin/python3",
        default_env_vars={},
        env_vars={},
        hardware_resources=[
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[
                    HardwareInfo(type=Accelerator.HW_TYPE, model="NV_GPU:Mock")
                    for _ in range(request.param)
                ],
            )
        ],
    )
    group = NodeGroupInfo(
        label=NodeGroupInfo.DEFAULT_GROUP_LABEL,
        nodes=[node],
        hardware_type=Accelerator.HW_TYPE,
    )
    cluster = SimpleNamespace(get_node_group=lambda: group)
    monkeypatch.setattr("rlinf.config.Cluster", lambda **kwargs: cluster)


@pytest.mark.parametrize(
    "config_name", ["libero_object_0_art_openpi", "libero_object_0_art_states_openpi"]
)
def test_openpi_art_recipe_composes_with_matching_batch_budget(
    monkeypatch, art_config_cluster, config_name
):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    cfg = validate_cfg(cfg)
    assert (
        cfg.env.train.task_suite_name == cfg.env.eval.task_suite_name == "libero_object"
    )
    assert cfg.env.train.task_id_filter == cfg.env.eval.task_id_filter == [0]
    assert cfg.env.train.total_num_envs == 128
    assert (
        cfg.actor.global_batch_size
        == (
            cfg.env.train.total_num_envs
            * cfg.env.train.rollout_epoch
            // cfg.algorithm.art_updates_per_rollout
        )
        == 64
    )
    assert cfg.actor.model.openpi.is_nft is False
    assert cfg.actor.model.num_steps == 4
    assert cfg.algorithm.art_timestep_fraction == 1.0
    assert cfg.algorithm.art_reference_source == "actor"
    assert cfg.actor.optim.use_fp32_master_params
    assert cfg.runner.eval_at_start
    assert (
        cfg.env.train.init_params.ignore_done and cfg.env.eval.init_params.ignore_done
    )
    assert cfg.rollout.seed == cfg.actor.seed
    if "states" in config_name:
        assert (
            cfg.algorithm.art_state_sampling
            and cfg.algorithm.art_branch_chunk_index is None
        )
    else:
        assert (
            not cfg.algorithm.art_state_sampling
            and cfg.algorithm.art_branch_chunk_index == 8
        )


def test_art_branch_conditioning_checks_simulator_images_and_tasks():
    obs = {
        "states": torch.zeros(4, 3),
        "main_images": torch.zeros(4, 2, 2, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(4, 2, 2, 3, dtype=torch.uint8),
        "task_descriptions": ["pick", "pick", "place", "place"],
        "art_sim_state_error": torch.zeros(4),
    }
    valid, _, _ = _art_branch_observation_stats(obs, 2, 1e-5, 1.0)
    assert valid.all()
    obs["art_sim_state_error"][:2] = 1e-3
    obs["wrist_images"][3] = 3
    valid, state_error, image_error = _art_branch_observation_stats(obs, 2, 1e-5, 1.0)
    assert not valid.any()
    torch.testing.assert_close(
        state_error[:2], torch.full((2,), 1e-3, dtype=torch.float64)
    )
    assert image_error[2:].eq(3).all()
    obs["art_sim_state_error"].zero_()
    obs["wrist_images"].zero_()
    obs["task_descriptions"][1] = "different"
    valid, _, _ = _art_branch_observation_stats(obs, 2, 1e-5, 1.0)
    assert valid.tolist() == [False, False, True, True]
    del obs["art_sim_state_error"]
    with pytest.raises(ValueError, match="simulator state diagnostics"):
        _art_branch_observation_stats(obs, 2, 1e-5, 1.0)
    errors = grouped_state_max_error(
        [np.zeros(3), np.array([0.0, 0.2, 0.0]), np.ones(3), np.ones(3)], 2
    )
    np.testing.assert_allclose(errors, [0.2, 0.2, 0.0, 0.0])


@pytest.mark.parametrize("branch_index", [0, 1, 2])
def test_art_branch_selects_matching_observations_states_and_returns(branch_index):
    worker = _art_trajectory_worker()
    worker.cfg.algorithm.art_branch_chunk_index = branch_index
    batch = worker.rollout_batch
    batch["dones"].zero_()
    batch["loss_mask"].fill_(True)
    batch["rewards"].zero_()
    batch["rewards"][-1, 1::2, -1] = 1.0
    batch["rewards"][:branch_index, :, 0] = 9.0
    inputs = batch["forward_inputs"]
    inputs["art_branch_valid"] = torch.ones(3, 4, dtype=torch.bool)
    inputs["art_branch_state_error"] = torch.zeros(3, 4)
    inputs["art_branch_image_error"] = torch.zeros(3, 4)
    for index in range(3):
        inputs["art_states"][index] += index * 100
        inputs["art_old_velocity"][index] += index * 100
    inputs["observation/state"] = torch.arange(12).reshape(3, 4, 1)
    expected_states = inputs["art_states"][branch_index].clone()
    expected_old = inputs["art_old_velocity"][branch_index].clone()
    metrics, active, windows = worker._prepare_art_trajectory_training_inputs(
        torch.Generator().manual_seed(7)
    )
    assert active and metrics["actor/branch_valid_group_frac"] == 1
    assert metrics["actor/art_reward_mean"] == 0.5
    replay = worker.rollout_batch["forward_inputs"]
    torch.testing.assert_close(replay["art_states"], expected_states)
    torch.testing.assert_close(replay["art_old_velocity"], expected_old)
    assert replay["art_guidance_direction"].shape[0] == 4
    assert replay["observation/state"][:, 0].tolist() == list(
        range(branch_index * 4, branch_index * 4 + 4)
    )
    assert set(windows[0].tolist()).isdisjoint(windows[1].tolist())
    assert sum(rows.numel() for rows in windows) == 4


@pytest.mark.parametrize("invalid_reason", ["terminated", "state_mismatch"])
def test_art_branch_masks_complete_invalid_groups(invalid_reason):
    worker = _art_trajectory_worker()
    worker.cfg.algorithm.art_branch_chunk_index = 1
    batch = worker.rollout_batch
    inputs = batch["forward_inputs"]
    inputs["art_branch_valid"] = torch.ones(3, 4, dtype=torch.bool)
    inputs["art_branch_state_error"] = torch.zeros(3, 4)
    inputs["art_branch_image_error"] = torch.zeros(3, 4)
    if invalid_reason == "state_mismatch":
        batch["dones"].zero_()
        inputs["art_branch_valid"][1, 0] = False
    batch["loss_mask"] = compute_loss_mask(batch["dones"])[0].any(-1, keepdim=True)
    metrics, _, _ = worker._prepare_art_trajectory_training_inputs(
        torch.Generator().manual_seed(7)
    )
    assert metrics["actor/branch_valid_group_frac"] == 0.5
    inputs = worker.rollout_batch["forward_inputs"]
    assert not inputs["art_coordinate_mask"][:2].any()
    assert inputs["art_guidance_magnitude"][:2].eq(0).all()
    assert inputs["art_coordinate_mask"][2:].any()


def test_art_state_sampling_preserves_conditions_and_candidate_cache():
    worker = _art_trajectory_worker()
    worker.cfg.algorithm.art_state_sampling = True
    batch = worker.rollout_batch
    batch["dones"].zero_()
    batch["loss_mask"].fill_(True)
    inputs = batch["forward_inputs"]
    inputs["art_state_id"] = torch.tensor([[31, 31, 12, 12]]).expand(3, -1)
    inputs["art_branch_valid"] = torch.ones(3, 4, dtype=torch.bool)
    inputs["art_branch_state_error"] = torch.zeros(3, 4)
    inputs["art_branch_image_error"] = torch.zeros(3, 4)
    cached_old = inputs["art_old_velocity"].clone()
    _, active, windows = worker._prepare_art_trajectory_training_inputs(
        torch.Generator().manual_seed(7)
    )
    assert active
    assert sum(len(window) for window in windows) == 4
    assert set(windows[0].tolist()).isdisjoint(windows[1].tolist())
    selected = worker.rollout_batch["forward_inputs"]
    for window in windows:
        assert set(selected["art_state_id"][window].tolist()) == {12, 31}
    torch.testing.assert_close(selected["art_old_velocity"], cached_old[0])


def test_art_covariance_groups_by_snapshot_identity():
    endpoints = torch.tensor([[[[0.0]], [[100.0]], [[2.0]], [[96.0]]]])
    guidance = _art_trajectory_guidance(
        endpoints,
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.ones(1, 4, 1, dtype=torch.bool),
        2,
        condition_ids=torch.tensor([9, 4, 9, 4]),
    )
    assert guidance["directions"].flatten().tolist() == [1.0, -1.0, 1.0, -1.0]


def test_art_state_sampling_recipe_composes(monkeypatch, art_config_cluster):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="libero_object_0_art_states_openpi")
    cfg = validate_cfg(cfg)
    assert cfg.env.train.art_state_sampling
    assert cfg.env.train.total_num_envs == 128 and cfg.env.eval.total_num_envs == 48
    assert cfg.actor.global_batch_size == 64 and cfg.actor.micro_batch_size == 2
    assert cfg.actor.optim.lr == 5e-7
    assert cfg.runner.val_check_interval == 10
    cfg.runner.overlap_env_bootstrap = True
    with pytest.raises(AssertionError, match="after the policy weight sync"):
        validate_openpi_art_cfg(cfg)


@pytest.mark.parametrize("branch_index", [0, 8])
def test_art_fixed_branch_recipe_composes(
    monkeypatch, art_config_cluster, branch_index
):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(
            config_name="libero_object_0_art_openpi",
            overrides=[
                f"algorithm.art_branch_chunk_index={branch_index}",
                "actor.global_batch_size=64",
            ],
        )
    cfg = validate_cfg(cfg)
    assert (
        cfg.env.train.task_suite_name == cfg.env.eval.task_suite_name == "libero_object"
    )
    assert cfg.env.train.art_branch_chunk_index == branch_index
    cfg.env.train.art_branch_chunk_index = branch_index + 1
    with pytest.raises(AssertionError, match="derived from algorithm"):
        validate_openpi_art_cfg(cfg)
    cfg.env.train.art_branch_chunk_index = branch_index
    cfg.algorithm.art_first_chunk_only = True
    with pytest.raises(AssertionError, match="mutually exclusive"):
        validate_openpi_art_cfg(cfg)
    cfg.algorithm.art_first_chunk_only = False
    cfg.algorithm.art_branch_chunk_index = (
        cfg.env.train.max_episode_steps // cfg.actor.model.num_action_chunks
    )
    with pytest.raises(AssertionError, match="before the episode limit"):
        validate_openpi_art_cfg(cfg)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("env.train.auto_reset", True, "auto_reset=False"),
        (
            "env.train.use_fixed_reset_state_ids",
            False,
            "use_fixed_reset_state_ids=True",
        ),
        ("actor.model.openpi.is_nft", True, "is_nft=False"),
        ("actor.model.openpi.noise_method", "flow_sde", "flow_ode"),
        ("actor.model.openpi.action_chunk", 10, "match num_action_chunks"),
        ("actor.model.openpi.action_env_dim", 6, "action_env_dim=action_dim"),
        ("actor.model.openpi.model_action_dim", 6, "action_env_dim=action_dim"),
        ("algorithm.art_reward_reduction", "mean", "episode returns"),
        ("algorithm.art_reference_source", "invalid", "rollout or actor"),
        ("env.train.max_steps_per_rollout_epoch", 100, "complete episodes"),
    ],
)
def test_openpi_art_rejects_incompatible_rollouts(key, value, message, monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="libero_object_0_art_openpi")
    OmegaConf.update(cfg, key, value)
    with pytest.raises(AssertionError, match=message):
        validate_openpi_art_cfg(cfg)


def test_openpi_art_rollout_replay_and_two_frozen_target_updates():
    from rlinf.models.embodiment.openpi.modules.siglip import SigLIPViT
    from rlinf.models.embodiment.openpi.modules.utils import (
        set_torch_compile,
        torch_compile_enabled,
    )
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig

    compiled = torch_compile_enabled()
    threads = torch.get_num_threads()
    set_torch_compile(False)
    torch.set_num_threads(1)
    try:
        config = Pi0Config(
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=4,
            action_horizon=4,
            max_token_len=4,
        )
        # Meta construction avoids allocating the production vision encoder.
        with torch.device("meta"):
            model = Pi0RL(
                config,
                num_steps=4,
                action_chunk=2,
                action_env_dim=2,
                rl_cfg=Pi0RLConfig(rl_mode="art", train_expert_only=True),
            )
        model.img = SigLIPViT(variant="mu/14", num_classes=64)
        model.to_empty(device="cpu")
        with torch.no_grad():
            generator = torch.Generator().manual_seed(31)
            for parameter in model.parameters():
                parameter.normal_(std=0.02, generator=generator)
        model._input_transform_fn = lambda data: data
        model._output_transform_fn = lambda data: {
            "actions": data["actions"][..., :2] * 10
        }
        observation = config.fake_obs(batch_size=2)
        noise = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(32))
        model.eval()
        with torch.no_grad():
            actions, result = model._predict_train(
                observation, noise=noise, rng=None, compute_values=False
            )
        inputs = result["forward_inputs"]
        shared = _repeat_art_group_leaders(result, 2)
        shared_actions = _repeat_art_group_leaders(actions, 2)
        torch.testing.assert_close(shared_actions[0], shared_actions[1])
        torch.testing.assert_close(
            shared["forward_inputs"]["art_endpoint"], shared_actions / 10
        )
        assert not torch.equal(inputs["art_endpoint"][0], inputs["art_endpoint"][1])
        assert inputs["art_states"].shape == (2, 4, 4, 4)
        assert inputs["art_old_velocity"].shape == inputs["art_states"].shape
        torch.testing.assert_close(inputs["art_endpoint"], actions / 10)
        torch.testing.assert_close(inputs["art_states"][:, 0], noise)
        for index in range(3):
            expected = (
                inputs["art_states"][:, index]
                - 0.25 * inputs["art_old_velocity"][:, index]
            )
            torch.testing.assert_close(inputs["art_states"][:, index + 1], expected)
        model.rl_cfg = replace(model.rl_cfg, rl_mode="default", is_nft=True)
        with torch.no_grad():
            nft_actions, nft_result = model._predict_train(
                observation, noise=noise, rng=None, compute_values=False
            )
        torch.testing.assert_close(nft_actions, actions)
        assert "nft_x0" in nft_result["forward_inputs"]
        assert "art_endpoint" not in nft_result["forward_inputs"]
        model.rl_cfg = replace(model.rl_cfg, rl_mode="art", is_nft=False)
        from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType

        worker = object.__new__(MultiStepRolloutWorker)
        worker.art_state_sampling = False
        worker.algorithm_cfg = OmegaConf.create({"group_size": 2})
        worker.model_cfg = OmegaConf.create({"model_type": "openpi"})
        worker.art_branch_chunk_index = 1
        worker.hf_model = model
        worker.expert_model = None
        worker.enable_dagger = False
        worker._train_sampling_params = {}
        worker._eval_sampling_params = {}
        worker._timer_metrics = {}
        worker._accelerator_type = AcceleratorType.NO_ACCEL
        env_obs = {
            "states": observation.state,
            "main_images": observation.images["base_0_rgb"],
            "wrist_images": observation.images["left_wrist_0_rgb"],
            "task_descriptions": ["pick", "pick"],
            "art_sim_state_error": torch.zeros(2),
        }
        env_worker = object.__new__(EnvWorker)
        policy_input = env_worker._build_policy_input(EnvOutput(obs=env_obs))
        env_obs = worker._merge_policy_inputs(policy_input.split([1, 1])).obs
        model._input_transform_fn = lambda sample: {
            "image": dict.fromkeys(observation.images, sample["observation/image"]),
            "image_mask": dict.fromkeys(observation.images, True),
            "state": sample["observation/state"],
            "tokenized_prompt": np.ones(4, dtype=np.int64),
            "tokenized_prompt_mask": np.ones(4, dtype=bool),
        }
        prefix_actions, prefix_result = worker._predict_art_branch_actions(env_obs, 0)
        torch.testing.assert_close(prefix_actions[0], prefix_actions[1])
        assert prefix_result["forward_inputs"]["art_branch_valid"].all()
        branch_actions, branch_result = worker._predict_art_branch_actions(env_obs, 1)
        assert not torch.equal(branch_actions[0], branch_actions[1])
        assert branch_result["forward_inputs"]["art_branch_valid"].all()
        torch.testing.assert_close(
            branch_result["forward_inputs"]["art_endpoint"], branch_actions / 10
        )
        env_obs["states"] = observation.state.clone()
        env_obs["states"][1] += 1
        _, tail_result = worker._predict_art_branch_actions(env_obs, 2)
        assert tail_result["forward_inputs"]["art_branch_valid"].all()
        worker._predict_art_branch_actions(env_obs, 0)
        env_obs["states"] = observation.state
        _, invalid_result = worker._predict_art_branch_actions(env_obs, 1)
        assert not invalid_result["forward_inputs"]["art_branch_valid"].any()
        worker.art_state_sampling = True
        worker.art_branch_chunk_index = 0
        env_obs["art_state_id"] = torch.tensor([19, 19])
        env_obs["art_state_steps"] = torch.tensor([40, 40])
        _, state_result = worker._predict_art_branch_actions(env_obs, 0)
        state_inputs = state_result["forward_inputs"]
        assert state_inputs["art_branch_valid"].all()
        assert state_inputs["art_state_id"].tolist() == [19, 19]
        model._input_transform_fn = lambda data: data
        for name, value in observation.images.items():
            inputs[f"obs_image__{name}"] = value
            inputs[f"obs_image_mask__{name}"] = observation.image_masks[name]
        inputs["obs_state"] = observation.state
        replay = EmbodiedARTFSDPPolicy._model_forward_inputs(inputs)
        with torch.no_grad():
            output = model.nft_forward(
                replay,
                nft_inputs={
                    "x_t": inputs["art_states"][:, 1],
                    "timesteps": inputs["art_sigmas"][:, 1],
                },
            )
        torch.testing.assert_close(output["v_theta"], inputs["art_old_velocity"][:, 1])
        actor = object.__new__(EmbodiedARTFSDPPolicy)
        actor.cfg = OmegaConf.create(
            {
                "actor": {
                    "micro_batch_size": 1,
                    "model": {"num_action_chunks": 2, "action_dim": 2},
                },
                "algorithm": {},
            }
        )
        actor.model = model
        actor.device = torch.device("cpu")
        actor.amp_context = nullcontext()
        actor.rollout_batch = {"forward_inputs": inputs}
        cached_old = inputs["art_old_velocity"].clone()
        model.rl_cfg = replace(model.rl_cfg, rl_mode="default", is_nft=True)
        with torch.no_grad():
            nft_output = model.nft_forward(
                replay,
                nft_inputs={
                    "x_t": inputs["art_states"][:, 1],
                    "timesteps": inputs["art_sigmas"][:, 1],
                },
            )
        assert nft_output["v_theta"].shape == (2, 2, 4)
        torch.testing.assert_close(nft_output["v_theta"], output["v_theta"][:, :2])
        model.rl_cfg = replace(model.rl_cfg, rl_mode="art", is_nft=False)
        model.train()
        rng_state = torch.get_rng_state().clone()
        replay_metrics = actor._art_replay_metrics()
        torch.testing.assert_close(torch.get_rng_state(), rng_state)
        for region in ("full", "exec", "unexec", "future", "padding"):
            assert replay_metrics[f"actor/replay_{region}_rms"] < 1e-5
        assert model.training
        assert replay_metrics["actor/replay_mse"] < 1e-10
        assert replay_metrics["actor/replay_max_abs"] < 1e-5
        torch.testing.assert_close(inputs["art_old_velocity"], cached_old)
        inputs["art_old_velocity"] += 0.25
        model.eval()
        replay_metrics = actor._art_replay_metrics()
        assert not model.training
        assert replay_metrics["actor/replay_mse"] == pytest.approx(0.25**2)
        assert replay_metrics["actor/replay_max_abs"] == pytest.approx(0.25)
        for region in ("full", "exec", "unexec", "future", "padding"):
            assert replay_metrics[f"actor/replay_{region}_rms"] == pytest.approx(0.25)
        inputs["art_old_velocity"].copy_(cached_old)
        inputs["art_old_velocity"][:, :, 2:, :2] += 0.5
        inputs["art_old_velocity"][:, :, :, 2:] += 1.0
        replay_metrics = actor._art_replay_metrics()
        assert replay_metrics["actor/replay_exec_rms"] < 1e-5
        assert replay_metrics["actor/replay_future_rms"] == pytest.approx(0.5)
        assert replay_metrics["actor/replay_padding_rms"] == pytest.approx(1.0)
        assert replay_metrics["actor/replay_unexec_rms"] == pytest.approx(0.75**0.5)
        inputs["art_old_velocity"].copy_(cached_old)
        guidance = _art_trajectory_guidance(
            inputs["art_endpoint"][None],
            torch.tensor([0.0, 1.0]),
            torch.ones(1, 2, 2, dtype=torch.bool),
            2,
            zero_signal_mode="exact",
        )
        inputs["art_coordinate_mask"] = guidance["coordinate_mask"][0]
        old = inputs["art_old_velocity"].clone()
        loss, _ = actor._art_forward_and_loss(
            inputs, replay, guidance["directions"][0], guidance["magnitudes"][0], 1
        )
        with torch.no_grad():
            reference = model.nft_forward(
                replay,
                nft_inputs={
                    "x_t": inputs["art_states"][:, 1],
                    "timesteps": inputs["art_sigmas"][:, 1],
                },
            )["v_theta"]
        expected_loss = _art_loss(
            reference[:, :2, :2],
            old[:, 1, :2, :2],
            inputs["art_sigmas"][:, 1],
            inputs["art_sigmas_next"][:, 1],
            guidance["directions"][0],
            guidance["magnitudes"][0],
            coordinate_mask=inputs["art_coordinate_mask"],
        )["per_item_loss"].mean()
        torch.testing.assert_close(loss, expected_loss)
        gradient = torch.autograd.grad(loss, model.action_out_proj.weight)[0]
        inputs["art_old_velocity"][:, :, 2:] += 100
        inputs["art_old_velocity"][:, :, :, 2:] -= 100
        unchanged_loss, _ = actor._art_forward_and_loss(
            inputs, replay, guidance["directions"][0], guidance["magnitudes"][0], 1
        )
        unchanged_gradient = torch.autograd.grad(
            unchanged_loss, model.action_out_proj.weight
        )[0]
        torch.testing.assert_close(unchanged_loss, loss)
        torch.testing.assert_close(unchanged_gradient, gradient)
        inputs["art_old_velocity"].copy_(old)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        for row in (0, 1):
            replay = EmbodiedARTFSDPPolicy._model_forward_inputs(
                inputs, torch.tensor([row])
            )
            output = model.nft_forward(
                replay,
                nft_inputs={
                    "x_t": inputs["art_states"][row : row + 1, 1],
                    "timesteps": inputs["art_sigmas"][row : row + 1, 1],
                },
            )
            terms = _art_loss(
                output["v_theta"][:, :2, :2],
                old[row : row + 1, 1, :2, :2],
                inputs["art_sigmas"][row : row + 1, 1],
                inputs["art_sigmas_next"][row : row + 1, 1],
                guidance["directions"][0, row : row + 1],
                guidance["magnitudes"][0, row : row + 1],
                corrective_mask_enabled=False,
            )
            optimizer.zero_grad()
            terms["per_item_loss"].mean().backward()
            assert model.action_out_proj.weight.grad is not None
            assert torch.isfinite(model.action_out_proj.weight.grad).all()
            assert model.action_out_proj.weight.grad.abs().sum() > 0
            previous_weight = model.action_out_proj.weight.detach().clone()
            optimizer.step()
            assert not torch.equal(model.action_out_proj.weight, previous_weight)
            torch.testing.assert_close(inputs["art_old_velocity"], old)
            gradient = model.action_out_proj.weight.grad.clone()
            rng_state = torch.get_rng_state().clone()
            drift = actor._art_replay_metrics(f"actor/drift_u{row + 1}")
            assert drift[f"actor/drift_u{row + 1}_full_rms"] > 0
            assert model.training
            torch.testing.assert_close(torch.get_rng_state(), rng_state)
            torch.testing.assert_close(model.action_out_proj.weight.grad, gradient)
            torch.testing.assert_close(inputs["art_old_velocity"], old)
    finally:
        set_torch_compile(compiled)
        torch.set_num_threads(threads)


class _VelocityPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(4, 4, 3))

    def forward(self, forward_type, forward_inputs, nft_inputs, compute_values):
        ids = forward_inputs["observation/state"][:, 0].long()
        return {"v_theta": nft_inputs["x_t"] + self.offset[ids]}


@pytest.fixture
def cpu_art_backend(monkeypatch):
    # Replace GPU/distributed boundaries only; execute the real worker kernels,
    # loss, optimizer, scheduler, and reference construction below.
    platform = SimpleNamespace(
        current_device=lambda: torch.device("cpu"),
        synchronize=lambda: None,
        ipc_collect=lambda: None,
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(Worker, "torch_platform", platform)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return platform


def _cpu_art_actor(platform, reference_source, state_sampling):
    actor = object.__new__(EmbodiedARTFSDPPolicy)
    actor._rank, actor._world_size, actor.version = 0, 1, 0
    actor.device, actor.torch_platform = torch.device("cpu"), platform
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": 2,
                "art_branch_chunk_index": None if state_sampling else 0,
                "art_state_sampling": state_sampling,
                "art_updates_per_rollout": 2,
                "art_timestep_fraction": 1.0,
                "art_reference_source": reference_source,
                "art_replay_diagnostics": True,
            },
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "global_batch_size": 2,
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 2,
                    "action_dim": 2,
                },
            },
            "env": {"train": {"max_episode_steps": 100}},
        }
    )
    states = torch.arange(4.0).view(1, 1, 4, 1, 1).expand(1, 4, 4, 4, 3).clone()
    inputs = {
        "art_endpoint": torch.arange(16.0).reshape(1, 4, 2, 2),
        "art_states": states,
        "art_old_velocity": states + 10,
        "art_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25]).expand(1, 4, 4),
        "art_sigmas_next": torch.tensor([0.75, 0.5, 0.25, 0.0]).expand(1, 4, 4),
        "observation/state": torch.arange(4).view(1, 4, 1),
        "art_branch_valid": torch.ones(1, 4, dtype=torch.bool),
        "art_branch_state_error": torch.zeros(1, 4),
        "art_branch_image_error": torch.zeros(1, 4),
        "art_state_id": torch.tensor([[0, 0, 1, 1]]),
    }
    actor.rollout_batch = {
        "forward_inputs": inputs,
        "rewards": torch.tensor([[[1.0, 0], [0, 0], [0, 0], [0, 1.0]]]),
        "dones": torch.zeros(2, 4, 2, dtype=torch.bool),
    }
    actor.model = _VelocityPolicy()
    actor.optimizer = torch.optim.AdamW(
        actor.model.parameters(), lr=0.01, weight_decay=0
    )
    actor.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor.optimizer, lambda step: 1
    )
    actor.amp_context = nullcontext()
    actor.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
    actor.is_weight_offloaded, actor.is_optimizer_offloaded = False, False
    actor.before_micro_batch = lambda *args, **kwargs: nullcontext()

    def optimizer_step():
        norm = torch.nn.utils.clip_grad_norm_(actor.model.parameters(), 1)
        actor.optimizer.step()
        return norm.item(), [0.01]

    actor.optimizer_step = optimizer_step
    return actor


@pytest.mark.parametrize("reference_source", ["rollout", "actor"])
@pytest.mark.parametrize("state_sampling", [False, True])
def test_art_reference_is_frozen_across_two_real_updates(
    cpu_art_backend, reference_source, state_sampling
):
    actor = _cpu_art_actor(cpu_art_backend, reference_source, state_sampling)
    metrics = inspect.unwrap(EmbodiedARTFSDPPolicy.run_training)(actor)
    inputs = actor.rollout_batch["forward_inputs"]
    offset = 0 if reference_source == "actor" else 10
    torch.testing.assert_close(
        inputs["art_old_velocity"], inputs["art_states"] + offset
    )
    assert metrics["actor/replay_mse"] == pytest.approx(100)
    assert metrics["actor/reference_pre_mse"] == pytest.approx(offset**2)
    assert actor.lr_scheduler.last_epoch == 2
    assert metrics["actor/active_count"] > 0
    if reference_source == "actor":
        assert metrics["actor/u1/mask_frac_global"] == 0
        assert metrics["actor/update_u1_exec_rms"] > 0
        assert metrics["actor/update_u2_unexec_rms"] == 0
    else:
        assert "actor/update_u1_exec_rms" not in metrics


def test_art_mask_ratios_pool_only_active_items():
    empty = torch.zeros(4)
    active = torch.tensor([2.0, 1.0, 1.0, 0.04])
    result = _art_mask_count_metrics(empty + active, "actor")
    assert result["actor/mask_frac_global"] == 0.5
    assert result["actor/movement_score_active"] == pytest.approx(0.02)
    assert _art_mask_count_metrics(empty, "actor")["actor/mask_frac_global"] == 0
